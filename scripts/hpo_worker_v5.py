#!/usr/bin/env python3
"""One fixed-budget CPU arm, using the unmodified shared PG lifecycle only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

from run_hpo_v5 import HORIZON, PROJECT, file_sha, now, verify_closure, write_json


def optimizer_rates(game):
    """Observe actual groups; classify particles by parameter identity."""
    bank = game.backend.particles
    records = []
    for optimizer_index, (name, optimizer) in enumerate((("generator", game.opt_g), ("critic", game.opt_d))):
        for index, group in enumerate(optimizer.param_groups):
            role = "critic" if name == "critic" else (
                "bank" if any(p is bank for p in group["params"]) else "adapter")
            records.append(dict(optimizer=name, group=index, role=role,
                parameter_count=len(group["params"]), scheduled_lr=float(group["lr"]),
                base_lr=float(game.base_lrs[optimizer_index][index])))
    if sorted(r["role"] for r in records) != ["adapter", "bank", "critic"]:
        raise ValueError("unexpected shared optimizer group roles")
    return records


def snapshot(run, cache, step):
    import torch
    from glide_particle_sliders.cli import predictions
    from glide_particle_sliders.evaluation import diagnostics
    from glide_particle_sliders.preference_metrics import preference_metrics

    result = dict(completed_steps=step)
    for split in ("train", "dev"):
        ids = [i for i, s in enumerate(cache["splits"]) if s == split]
        rows = [cache["metadata"]["dataset"][i] for i in ids]
        values = predictions(run, cache, split, strength=1.)
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"nonfinite {split} prediction")
        result[split] = diagnostics(values, cache["teacher_logits"][ids], cache["neutral_logits"][ids], rows)
        result[f"{split}_preference"] = preference_metrics(values, cache["neutral_logits"][ids], rows)
        result[f"{split}_off"] = diagnostics(cache["neutral_logits"][ids],
            cache["teacher_logits"][ids], cache["neutral_logits"][ids], rows)
    return result


def run_candidate(plan_path, candidate_id):
    # Set policy before importing Torch/dependencies, even for direct execution.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("worker requires explicit CUDA_VISIBLE_DEVICES='' CPU policy")
    import torch
    from particle_sliders.inference import open_inference
    from particle_sliders.lifecycle import SessionConfig
    from glide_particle_sliders import binding_normalized_v3 as binding
    from glide_particle_sliders.binding import load_readout_file

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify_closure(plan)
    candidate = next(c for c in plan["candidates"] if c["id"] == candidate_id)
    if candidate["reuse"]:
        raise ValueError("exact existing anchor must be reused, never retrained")
    output = plan_path.parent / candidate_id
    output.mkdir(exist_ok=True)
    with (output / "worker-start.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=now(), plan_sha256=plan["plan_sha256"]), stream)
    report = dict(schema="glide-hpo-arm-v5", id=candidate_id, index=candidate["index"],
        status="running", started=now(), pid=os.getpid(), config=candidate["config"],
        config_sha256=candidate["config_sha256"], plan_sha256=plan["plan_sha256"],
        sources=plan["sources"], assets=plan["assets"], base_rates=candidate["base_rates"],
        resource=dict(device="cpu", torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(), cuda_visible_devices=""),
        snapshots=[], completed_steps=0, new_unique_updates=0, physical_updates=0,
        test_evaluated=False, fresh_evaluated=False, unrelated_evaluated=False,
        ema_decay=None, selection=plan["selection"])
    write_json(output / "progress.json", report)
    started = time.perf_counter()
    try:
        cfg = SessionConfig.from_dict(candidate["config"])
        cache = load_readout_file(plan["assets"]["cache"]["path"], cache=True)
        # No resume argument is passed to BACKEND.open. A later selected-only
        # proof must open this exact cfg, then call session.restore(read_packet).
        with binding.BACKEND.open(cfg) as run:
            slider = run.game.backend.slider
            center = run.game.network.input_center.detach().clone()
            rms = run.game.network.input_rms.detach().clone()
            initial_bank = run.game.backend.particles.detach().clone()
            rates = optimizer_rates(run.game)
            for record in rates:
                if not math.isclose(record["base_lr"], candidate["base_rates"][record["role"]], rel_tol=1e-14):
                    raise ValueError("shared actual optimizer BASE rate differs from plan")
            report["initial_optimizer_groups"] = rates
            durations = []
            with (output / "updates.jsonl").open("x") as log:
                while run.game.completed_steps < HORIZON:
                    before = optimizer_rates(run.game)
                    tick = time.perf_counter()
                    stats = run.step()
                    duration = time.perf_counter() - tick
                    step = run.game.completed_steps
                    durations.append(duration)
                    report.update(completed_steps=step, new_unique_updates=step, physical_updates=step)
                    row = dict(step=step, wall_step_seconds=duration,
                        optimizer_groups_before=before, optimizer_groups_after=optimizer_rates(run.game),
                        shared_step_metrics=stats, plan_sha256=plan["plan_sha256"],
                        config_sha256=candidate["config_sha256"])
                    # allow_nan=False ensures a divergent update is recorded as
                    # a failure, never silently promoted into a finite candidate.
                    log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                    if step in (256, 257, 512):
                        name = "next-uninterrupted.pt" if step == 257 else f"step-{step}.pt"
                        packet = run.checkpoint(output / name)
                        report.setdefault("checkpoints", {})[str(step)] = dict(path=str(output / name),
                            packet_sha256=packet["sha256"], file_sha256=file_sha(output / name))
                    if step in plan["snapshots"]:
                        report["snapshots"].append(snapshot(run, cache, step))
                        write_json(output / "progress.json", report)
                        print(json.dumps(dict(event="snapshot", candidate=candidate_id, step=step,
                            dev=report["snapshots"][-1]["dev"]["mean"])), flush=True)
            if not torch.equal(center, run.game.network.input_center) or not torch.equal(rms, run.game.network.input_rms):
                raise RuntimeError("frozen normalization buffers changed")
            if run.ema.decay is not None or run.ema.values is not None:
                raise RuntimeError("weight EMA unexpectedly enabled")
            frozen = torch.equal(slider.model.weight.detach().cpu(), cache["readout_weight"])
            inactive = float(slider.adapter.lora_up.weight[2:].detach().abs().max())
            if not frozen or inactive != 0.:
                raise RuntimeError("native base or inactive rows changed")
            report["native_mechanics"] = dict(frozen_readout_exact=frozen, inactive_up_weight_max=inactive,
                normalization_frozen_exact=True, ema_decay=None, ema_shadow_count=0,
                particle_bank_l2_change=float((run.game.backend.particles.detach() - initial_bank).double().norm()))
            export = output / "raw-inference.pt"
            run.export_inference(export, selection="raw")
            # Verify deployment artifact semantics using only old TRAIN hidden
            # states; no additional native encoder run or held-out data access.
            ids = [i for i, split in enumerate(cache["splits"]) if split == "train"]
            hidden = cache["neutral_hidden"][ids]
            expected = slider.response(hidden).detach().clone()
            with open_inference(export, backend=binding.INFERENCE, operation="response",
                options={"readout_file": plan["assets"]["readout"]["path"]}) as inference:
                if not torch.equal(expected, inference.response(hidden=hidden)):
                    raise RuntimeError("raw native-only export differs")
            report["raw_export"] = dict(path=str(export), sha256=file_sha(export), exact=True)
            report["final_checkpoint"] = report["checkpoints"]["512"]
            final = report["snapshots"][-1]
            report.update(train=final["train"], dev=final["dev"],
                train_preference=final["train_preference"], dev_preference=final["dev_preference"])
            measured = durations[plan["warmup_updates"]:]
            report["throughput"] = dict(warmup_updates_excluded=8, measured_updates=len(measured),
                measured_step_seconds=sum(measured), updates_per_second=len(measured) / sum(measured),
                all512_step_seconds=sum(durations), actual_completed_updates=len(durations),
                definition="shared step wall time, excluding first8 warm updates; evaluation/checkpoint I/O excluded")
        report.update(status="completed", finished=now(), wall_seconds=time.perf_counter() - started)
        write_json(output / "results.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", finished=now(), wall_seconds=time.perf_counter() - started,
                      error=repr(error), traceback=traceback.format_exc())
        write_json(output / "results.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    run_candidate(args.plan, args.candidate)


if __name__ == "__main__":
    main()
