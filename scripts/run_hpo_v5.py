#!/usr/bin/env python3
"""Declare, then explicitly execute the fixed CPU-only GLiDE HPO cohort.

Import/build_plan never starts a worker. --plan-only is the default; executing an
existing immutable plan requires --execute. No worker is retried or killed by a
time limit, and no score can shorten its declared training horizon.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
CACHE = "artifacts/label-probe-inputs-v1/synthetic-label-cache.pt"
READOUT = "artifacts/readout-v2.pt"
ANCHOR = "artifacts/normalized-residual-dev-v3-continuation"
HORIZON = 512
SEED = 7
RATES = ((.0003, .001, .003), (.003, .018), (.003, .03))
POLICY = dict(device="cpu", max_workers=2, torch_threads_per_worker=4,
              cuda_visible_devices="", automatic_retries=0, time_limit_seconds=None,
              early_stopping=False, finite_candidates_finish_horizon=True)
CRITERION = dict(split="dev", selection="raw", strength=1., completed_steps=512,
                 metric="FP64 operand-first equal-example relative SSE", field="dev.mean",
                 preference_metrics="diagnostics only", tie_break="declared candidate index")


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, *, immutable=False):
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        try:
            with path.open("x") as stream:
                stream.write(encoded)
        except FileExistsError:
            if path.read_text() != encoded:
                raise ValueError(f"immutable plan already exists with different content: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)


def candidate_specs(project=PROJECT):
    """Pure, deterministic declaration: independent absolute BASE rates."""
    cache = str((Path(project) / CACHE).resolve())
    choices = [(g, d, bank, 8) for g, d, bank in product(*RATES)]
    choices.append((.001, .018, .03, 32))
    candidates = []
    for index, (g, d, bank, batch) in enumerate(choices):
        # These are the only official-PG rate translations; no optimizer math.
        options = dict(base_precision="float32", batch=batch, cache_file=cache,
                       critic="residual64", d_lr_mult=d / g, device="cpu",
                       ema_decay=None, lr=g, penalty_input_domain="continuous",
                       pg_source_profile="release-0.9.5", prior_lr_mult=bank / g,
                       seed=SEED, steps=HORIZON)
        config = dict(backend="glide-normalized-readout-particles-v3",
                      cohort="glide-04b-normalized-adapter-pg095-raw-v3", options=options)
        reuse = (g, d, bank, batch) == (.001, .018, .03, 8)
        candidates.append(dict(index=index, id=f"arm-{index:02d}",
            base_rates=dict(adapter=g, critic=d, bank=bank), config=config,
            config_sha256=digest(config), reuse=reuse,
            new_unique_updates=0 if reuse else HORIZON))
    return candidates


def validate_plan(plan):
    if plan.get("schema") != "glide-fixed-hpo-v5":
        raise ValueError("unsupported HPO plan schema")
    expected = candidate_specs(plan["project"])
    actual = [{k: c[k] for k in e} for c, e in zip(plan["candidates"], expected)]
    if actual != expected or len(plan["candidates"]) != len(expected):
        raise ValueError("candidate grid, ordering, exact reuse, or fixed budget differs")
    if plan["resource_policy"] != POLICY or plan["selection"] != CRITERION:
        raise ValueError("resource or sole selection policy differs")
    if plan["budget"] != dict(arms=13, reused_arms=1, new_arms=12,
            new_unique_updates=6144, reused_unique_updates=512, horizon=512):
        raise ValueError("fixed budget differs")
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(unsigned) != plan.get("plan_sha256"):
        raise ValueError("plan content hash differs")


def registry_derivative(project, historical_sha, current_sha):
    """Permit exactly the four audited v2/v3 registry metadata additions."""
    historical = Path(project) / "artifacts/source-v2-label-probe/pyproject.toml"
    current = Path(project) / "pyproject.toml"
    if file_sha(historical) != historical_sha or file_sha(current) != current_sha:
        raise ValueError("registry derivative source hashes differ")
    previous = tomllib.loads(historical.read_text())
    derived = tomllib.loads(current.read_text())
    added = {}
    for group, attribute in (("particle_sliders.response_backends", "BACKEND"),
                             ("particle_sliders.inference_backends", "INFERENCE")):
        entries = derived["project"]["entry-points"][group]
        for name, module in (("glide-residual-critic-particles-v2", "binding_residual_v2"),
                             ("glide-normalized-readout-particles-v3", "binding_normalized_v3")):
            value = f"glide_particle_sliders.{module}:{attribute}"
            if entries.pop(name, None) != value:
                raise ValueError("registry derivative is missing an exact audited entry")
            added[f"{group}/{name}"] = value
    if derived != previous:
        raise ValueError("pyproject differs beyond four audited registry entries")
    return dict(historical_path=str(historical), historical_sha256=historical_sha,
                current_sha256=current_sha, exact_toml_equality_after_removing=added)


def build_plan(project=PROJECT):
    """Hash declared local sources/assets and verify the existing exact anchor.

    This reads the existing anchor packet only. It neither loads cache tensors,
    opens a training session, reads new evaluation data, nor launches a process.
    """
    from particle_sliders.lifecycle import read_packet
    from glide_particle_sliders.cli import source_receipt

    project = Path(project).resolve()
    candidates = candidate_specs(project)
    anchor = next(c for c in candidates if c["reuse"])
    base = project / ANCHOR
    report_path = base / "results.json"
    report = json.loads(report_path.read_text())
    packet = read_packet(base / "step-512.pt")
    if report["config"] != anchor["config"] or packet["manifest"]["config"] != anchor["config"]:
        raise ValueError("existing anchor configuration is not exactly the declared batch8 arm")
    if packet["payload"]["game"]["completed_steps"] != HORIZON:
        raise ValueError("existing anchor did not complete512 updates")
    current = source_receipt()
    # Existing scientific source and dependency identities must agree. New
    # diagnostics/scripts may be added, but never rewrite a source-bound model.
    for name, identity in report["source"]["packages"].items():
        if current["packages"].get(name) != identity:
            raise ValueError(f"anchor dependency identity differs: {name}")
    derivatives = {}
    for name, identity in report["source"]["product_sources"].items():
        if current["product_sources"].get(name) != identity:
            if name == "pyproject.toml":
                derivatives[name] = registry_derivative(project, identity, current["product_sources"][name])
            else:
                raise ValueError(f"anchor source differs: {name}")
    final = next(s for s in report["snapshots"] if s["completed_steps"] == HORIZON)
    for split in ("train", "dev"):
        if not math.isfinite(final[split]["mean"]):
            raise ValueError("existing anchor metric is nonfinite")
    reuse = dict(report=str(report_path), checkpoint=str(base / "step-512.pt"),
        raw_export=str(base / "raw-inference.pt"), packet_sha256=packet["sha256"],
        report_sha256=file_sha(report_path), checkpoint_sha256=file_sha(base / "step-512.pt"),
        raw_export_sha256=file_sha(base / "raw-inference.pt"),
        train=final["train"], dev=final["dev"], original_source=report["source"],
        audited_metadata_derivatives=derivatives,
        provenance="unchanged existing result/checkpoint; no new training or relabeling")
    anchor["reuse_artifacts"] = reuse
    sources = dict(current, scripts={f"scripts/{name}": file_sha(project / "scripts" / name)
        for name in ("run_hpo_v5.py", "hpo_worker_v5.py")})
    plan = dict(schema="glide-fixed-hpo-v5", project=str(project),
        scientific_scope="old synthetic-label cache only; TRAIN/DEV; no TEST, fresh or unrelated evaluation",
        candidates=candidates, sources=sources,
        assets={name: dict(path=str(project / relative), sha256=file_sha(project / relative))
                for name, relative in (("cache", CACHE), ("readout", READOUT))},
        budget=dict(arms=13, reused_arms=1, new_arms=12, new_unique_updates=6144,
                    reused_unique_updates=512, horizon=512),
        resource_policy=POLICY.copy(), selection=CRITERION.copy(),
        snapshots=[128, 256, 512], warmup_updates=8,
        continuation_evidence=dict(checkpoint=256, natural_next_checkpoint=257,
            resume_proof="selected-only later; no duplicate update in search"))
    plan["plan_sha256"] = digest(plan)
    validate_plan(plan)
    return plan


def verify_closure(plan):
    """Fail before training if source/data artifacts drifted from the plan."""
    from glide_particle_sliders.cli import source_receipt
    validate_plan(plan)
    expected = {k: v for k, v in plan["sources"].items() if k != "scripts"}
    if source_receipt() != expected:
        raise ValueError("runtime source/dependency receipt changed after plan freeze")
    for name, expected_sha in plan["sources"]["scripts"].items():
        if file_sha(Path(plan["project"]) / name) != expected_sha:
            raise ValueError(f"planned script changed: {name}")
    for asset in plan["assets"].values():
        if file_sha(asset["path"]) != asset["sha256"]:
            raise ValueError(f"planned data changed: {asset['path']}")
    for candidate in plan["candidates"]:
        if candidate["reuse"]:
            reuse = candidate["reuse_artifacts"]
            for key in ("report", "checkpoint", "raw_export"):
                if file_sha(reuse[key]) != reuse[f"{key}_sha256"]:
                    raise ValueError(f"existing anchor changed: {key}")


def choose_winner(results):
    eligible = [r for r in results if r["status"] in ("completed", "reused")
        and r.get("completed_steps") == HORIZON
        and math.isfinite(r["dev"]["mean"])]
    if not eligible:
        raise ValueError("no finite completed512-step candidate")
    return min(eligible, key=lambda r: (r["dev"]["mean"], r["index"]))


def execute(plan_path):
    """Launch at most two owned CPU children; never retry or time-kill them."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify_closure(plan)
    root = plan_path.parent
    # Exclusive creation deliberately prevents duplicate controllers/relaunches.
    with (root / "controller-start.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=now(), plan_sha256=plan["plan_sha256"]), stream)
    state = dict(schema="glide-hpo-controller-v5", started=now(), pid=os.getpid(),
                 plan_sha256=plan["plan_sha256"], resource_policy=POLICY.copy(),
                 processes=[], results=[], automatic_retries=0)
    queue = []
    for candidate in plan["candidates"]:
        if candidate["reuse"]:
            reuse = candidate["reuse_artifacts"]
            state["results"].append(dict(id=candidate["id"], index=candidate["index"],
                status="reused", completed_steps=512, new_unique_updates=0,
                train=reuse["train"], dev=reuse["dev"], provenance=reuse))
        else:
            queue.append(candidate)
    active = {}
    write_json(root / "controller-progress.json", state)
    while queue or active:
        while queue and len(active) < POLICY["max_workers"]:
            candidate = queue.pop(0)
            output = root / candidate["id"]
            output.mkdir(exist_ok=False)
            stdout = (output / "stdout.log").open("x")
            stderr = (output / "stderr.log").open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
            command = [sys.executable, str(Path(plan["project"]) / "scripts/hpo_worker_v5.py"),
                       "--plan", str(plan_path), "--candidate", candidate["id"]]
            try:
                child = subprocess.Popen(command, cwd=plan["project"], env=env,
                                         stdout=stdout, stderr=stderr)
            except OSError as error:
                stdout.close(); stderr.close()
                state["results"].append(dict(id=candidate["id"], index=candidate["index"],
                    status="launch_failed", error=repr(error), new_unique_updates=0))
                continue
            process = dict(candidate=candidate["id"], pid=child.pid, command=command,
                           started=now(), status="running", stdout=str(output / "stdout.log"),
                           stderr=str(output / "stderr.log"), automatic_retries=0)
            state["processes"].append(process)
            active[child.pid] = (child, candidate, process, stdout, stderr)
            write_json(root / "controller-progress.json", state)
        for pid, owned in list(active.items()):
            child, candidate, process, stdout, stderr = owned
            code = child.poll()
            if code is None:
                continue
            stdout.close(); stderr.close()
            process.update(status="exited", returncode=code, finished=now())
            result_path = root / candidate["id"] / "results.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if code != 0 or result.get("completed_steps") != HORIZON:
                    result["status"] = "failed"
            else:
                result = dict(id=candidate["id"], index=candidate["index"], status="failed",
                              returncode=code, error="worker exited without results.json")
            state["results"].append(result)
            del active[pid]
            write_json(root / "controller-progress.json", state)
        if active:
            time.sleep(.25)
    winner = choose_winner(state["results"])
    selection = dict(schema="glide-hpo-selection-v5", selected_at=now(),
        plan_sha256=plan["plan_sha256"], criterion=plan["selection"], winner=winner,
        all_declared_arms_completed=all(r["status"] in ("completed", "reused") for r in state["results"]),
        test_evaluated=False, fresh_evaluated=False, unrelated_evaluated=False)
    state.update(finished=now(), winner=winner["id"])
    write_json(root / "controller-results.json", state)
    write_json(root / "selection.json", selection, immutable=True)
    print(json.dumps(dict(winner=winner["id"], dev=winner["dev"]["mean"],
                         all_arms_completed=selection["all_declared_arms_completed"])), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=PROJECT / "artifacts/hpo-v5/plan.json")
    args = parser.parse_args()
    if args.execute:
        execute(args.plan)
    else:
        plan = build_plan()
        write_json(args.plan, plan, immutable=True)
        print(json.dumps(dict(plan=str(args.plan.resolve()), plan_sha256=plan["plan_sha256"],
                              budget=plan["budget"], launched=0)), flush=True)


if __name__ == "__main__":
    main()
