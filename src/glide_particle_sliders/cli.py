"""Reproducible CPU preparation, bounded shared-PG experiment, and live check."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import resource
import shutil

import torch

from . import CORE_COMMIT, MODEL_REVISION
from .binding import config, load_readout_file, ReadoutSlider
from .evaluation import diagnostics, position_bias
from .native import prepare_cache, load_glide, encode_requests, sha256


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def source_receipt():
    root = Path(__file__).resolve().parents[2]
    paths = [*root.glob("src/**/*.py"), root / "pyproject.toml", root / "native-model.lock.json"]
    packages = {}
    for name in ("particle-sliders-core", "particlegan", "hypergan", "hndl", "torch", "transformers", "tokenizers", "safetensors"):
        dist = distribution(name)
        packages[name] = dict(version=dist.version, direct_url=json.loads(dist.read_text("direct_url.json") or "{}"))
    return dict(product_sources={str(p.relative_to(root)): sha256(p) for p in paths}, packages=packages,
                core_commit=CORE_COMMIT, model_revision=MODEL_REVISION)


def split_values(cache, split, values):
    indices = [i for i, s in enumerate(cache["splits"]) if s == split]
    rows = [cache["metadata"]["dataset"][i] for i in indices]
    return values[indices], cache["teacher_logits"][indices], cache["neutral_logits"][indices], rows


def assess(cache, split, values):
    return diagnostics(*split_values(cache, split, values))


def predictions(session, cache, split, strength=1.):
    indices = [i for i, s in enumerate(cache["splits"]) if s == split]
    return torch.cat([session.evaluate(i, split=split, strength=strength) for i in range(len(indices))])


def session_report(session, cache, split, strength=1.):
    indices = [i for i, s in enumerate(cache["splits"]) if s == split]
    return diagnostics(predictions(session, cache, split, strength), cache["teacher_logits"][indices],
                       cache["neutral_logits"][indices], [cache["metadata"]["dataset"][i] for i in indices])


def run_experiment(cache_file, readout_file, output_dir, *, steps=128, rates=(.0001, .001, .01), batch=8, threads=4):
    """Three equal-horizon candidates; select on DEV before opening TEST results."""
    from particle_sliders.response import open_response
    from particle_sliders.inference import open_inference
    if type(steps) is not int or steps < 2:
        raise ValueError("at least two declared updates required")
    if not rates or len(set(rates)) != len(rates) or any(isinstance(r, bool) or not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("distinct positive finite learning rates required")
    torch.set_num_threads(threads)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    cache = load_readout_file(cache_file, cache=True)
    if not {"train", "dev", "test"} <= set(cache["splits"]):
        raise ValueError("train/dev/test splits are required")
    readout = load_readout_file(readout_file)
    if readout["model_identity"] != cache["model_identity"] or not torch.equal(readout["readout_weight"], cache["readout_weight"]):
        raise ValueError("readout-only inference base differs from training cache")
    source = source_receipt()
    write_json(root / "source-receipt.json", source)
    report = dict(schema="glide-native-experiment-v1", started=now(), source=source,
                  cache_sha256=sha256(cache_file), readout_sha256=sha256(readout_file),
                  primary="RAW strength1 FP64 operand-first equal-example relative SSE on the two active native logits",
                  experiment_scope=cache["metadata"].get("teacher_protocol", "frozen encoder; native routed particle readout; neutral-to-concise instruction distillation; synthetic paired choices"),
                  selection=dict(split="dev", common_horizon=steps, completed_updates=steps, metric="mean relative SSE"),
                  resource=dict(device="cpu", cpu_threads=threads, gpu_launches=0, foreign_jobs_changed=False),
                  candidates=[])
    base_options = dict(cache_file=str(Path(cache_file).resolve()), steps=steps, batch=batch, seed=7,
                        device="cpu", d_lr_mult=18., prior_lr_mult=30., ema_decay=None)
    train_idx = [i for i, s in enumerate(cache["splits"]) if s == "train"]
    bias = position_bias(cache["teacher_logits"][train_idx], cache["neutral_logits"][train_idx])
    # No test measurement during fitting or candidate selection.
    report["dev_baselines"] = dict(off=assess(cache, "dev", cache["neutral_logits"]),
                                   explicit_teacher=assess(cache, "dev", cache["teacher_logits"]),
                                   constant_slot_bias=assess(cache, "dev", cache["neutral_logits"].double() + bias))
    write_json(root / "progress.json", report)
    for index, lr in enumerate(rates):
        candidate_dir = root / f"candidate-{index}-g{lr:g}"
        candidate_dir.mkdir()
        cfg = config(**base_options, lr=lr)
        candidate = dict(index=index, generator_lr=lr, d_lr_mult=18., prior_lr_mult=30.,
                         config=cfg.as_dict(), status="running", snapshots=[])
        report["candidates"].append(candidate)
        with open_response(cfg) as run:
            initial_bank = run.game.backend.particles.detach().clone()
            zero = predictions(run, cache, "dev", strength=0.)
            dev_indices = [i for i, s in enumerate(cache["splits"]) if s == "dev"]
            if not torch.equal(zero, cache["neutral_logits"][dev_indices]):
                # Native readout GEMM batching can change last bits; require exact
                # OFF against the same one-row direct native readout geometry.
                direct = torch.cat([torch.nn.functional.linear(cache["neutral_hidden"][i:i+1], cache["readout_weight"])[:, :2] for i in dev_indices])
                if not torch.equal(zero, direct):
                    raise RuntimeError("OFF does not reproduce the native readout exactly")
            candidate["off_exact_direct_native"] = True
            log = (candidate_dir / "updates.jsonl").open("w")
            try:
                def on_step(stats):
                    log.write(json.dumps(stats, allow_nan=False) + "\n"); log.flush()
                stops = sorted({max(1, steps // 4), max(1, steps // 2), steps})
                for stop in stops:
                    if run.game.completed_steps < stop:
                        run.train(stop_after=stop, on_step=on_step)
                    checkpoint = candidate_dir / f"step-{stop}.pt"
                    packet = run.checkpoint(checkpoint)
                    snapshot = dict(completed_steps=run.game.completed_steps, checkpoint=str(checkpoint),
                                    checkpoint_sha256=sha256(checkpoint), packet_sha256=packet["sha256"],
                                    dev=session_report(run, cache, "dev"))
                    candidate["snapshots"].append(snapshot)
                    print(json.dumps(dict(event="candidate_snapshot", index=index, lr=lr, step=stop,
                                          dev_relative_sse=snapshot["dev"]["mean"])), flush=True)
                    if index == 0 and stop == stops[0] and stop < steps:
                        # Exercise actual continuation after optimizer histories
                        # have accumulated. Keep the original candidate; only
                        # its next update is deliberately repeated in a fresh
                        # factory, never its committed training prefix.
                        on_step(run.step())
                        expected = run.checkpoint(root / "proof-next-uninterrupted.pt")
                        with open_response(cfg, resume=checkpoint) as resumed:
                            resumed.step()
                            actual = resumed.checkpoint(root / "proof-next-resumed.pt")
                        if actual["sha256"] != expected["sha256"]:
                            raise RuntimeError("native-data next update differs after strict full-state resume")
                        report["resume_proof"] = dict(exact_packet_match=True, from_step=stop, to_step=stop + 1,
                            packet_sha256=actual["sha256"], additional_physical_updates=1,
                            deliberate_duplicate_updates=1, replayed_prefix_updates=0, ema_decay=None)
                if run.ema.values is not None or run.ema.decay is not None:
                    raise RuntimeError("weight EMA unexpectedly enabled")
                candidate["export"] = str(candidate_dir / "raw-inference.pt")
                run.export_inference(candidate["export"], selection="raw")
                candidate["final_checkpoint"] = str(candidate_dir / f"step-{steps}.pt")
                candidate["status"] = "completed"
                candidate["completed_steps"] = run.game.completed_steps
                candidate["dev"] = candidate["snapshots"][-1]["dev"]
                slider = run.game.backend.slider
                candidate["native_mechanics"] = dict(
                    particle_bank_l2_change=float((slider.particles.detach() - initial_bank).double().norm()),
                    active_up_weight_l2=float(slider.adapter.lora_up.weight[:2].detach().double().norm()),
                    inactive_up_weight_max=float(slider.adapter.lora_up.weight[2:].detach().abs().max()),
                    frozen_readout_exact=torch.equal(slider.model.weight.detach().cpu(), cache["readout_weight"]),
                    ema_decay=run.ema.decay, ema_shadow_count=0 if run.ema.values is None else len(run.ema.values))
            finally:
                log.close()
        write_json(root / "progress.json", report)
    selected = min(report["candidates"], key=lambda c: (c["dev"]["mean"], c["index"]))
    report["winner"] = deepcopy(selected)
    report["selection_frozen_at"] = now()
    write_json(root / "selection.json", dict(winner=selected, criterion=report["selection"], selected_at=report["selection_frozen_at"]))
    report["test_baselines"] = dict(off=assess(cache, "test", cache["neutral_logits"]),
                                    explicit_teacher=assess(cache, "test", cache["teacher_logits"]),
                                    constant_slot_bias=assess(cache, "test", cache["neutral_logits"].double() + bias))
    cfg = config(**{k: v for k, v in selected["config"]["options"].items()})
    with open_response(cfg, resume=selected["final_checkpoint"]) as run:
        report["winner_test"] = session_report(run, cache, "test")
        report["strength_sweep_test"] = {str(a): session_report(run, cache, "test", a) for a in (0., .25, .5, .75, 1.)}
        report["actual_off_dev"] = session_report(run, cache, "dev", 0.)
        report["actual_off_test"] = report["strength_sweep_test"]["0.0"]
        # Input-independent and scrambled-context diagnostics help identify whether
        # an apparent logit fit is only a generic positional edit.
        with torch.no_grad():
            learned = run.game.backend.slider.response(cache["neutral_hidden"], strength=1.)
            direct = torch.nn.functional.linear(cache["neutral_hidden"], cache["readout_weight"])[:, :2]
            delta = learned - direct
            scrambled = cache["neutral_logits"] + delta.roll(1, 0)
        report["native_batch_roundoff"] = dict(max_logit_difference=float((direct - cache["neutral_logits"]).abs().max()),
            explanation="cached encoder/readout batch2; shared response readout batch1; actual OFF scores are measured separately")
        report["scrambled_delta_test"] = assess(cache, "test", scrambled)
        export_path = selected["export"]
        with open_inference(export_path, operation="response", options={"readout_file": str(Path(readout_file).resolve())}) as inference:
            restored = inference.response(hidden=cache["neutral_hidden"], strength=1.)
            if not torch.equal(learned, restored):
                raise RuntimeError("RAW native-only export differs from training network")
            report["raw_export_exact"] = True
    report["actual_updates"] = dict(sweep=sum(c["completed_steps"] for c in report["candidates"]),
                                    proof_additional_physical=1, deliberate_duplicate=1, replayed_prefix=0,
                                    total_physical=sum(c["completed_steps"] for c in report["candidates"]) + 1)
    report["resource"]["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["finished"] = now()
    write_json(root / "results.json", report)
    return report


def live_check(model_dir, lock_file, dataset_file, cache_file, experiment_dir, *, threads=4):
    """Re-load the 0.4B encoder in a fresh process and attach the exported head."""
    from particle_sliders.inference import read_inference, native_identity
    from .binding import NativeReadoutResponse, PROTOCOL
    root = Path(experiment_dir)
    result = json.loads((root / "results.json").read_text())
    cache = load_readout_file(cache_file, cache=True)
    model, identity = load_glide(model_dir, lock_file, threads=threads)
    if identity != cache["model_identity"]:
        raise ValueError("live model identity differs from the training model")
    packet = read_inference(result["winner"]["export"])
    selection = json.loads(Path(dataset_file).read_text())["rows"]
    # Same original/swapped TEST group plus a different TEST group.
    indices = [i for i, r in enumerate(selection) if r["split"] == "test"][:4]
    requests = [selection[i]["neutral"] for i in indices]
    hidden, direct_logits, _ = encode_requests(model, requests, batch_size=2)
    slider = ReadoutSlider(model.network.readout, model_identity=identity)
    try:
        native = NativeReadoutResponse(slider)
        actual_identity = native_identity(native, model_identity=slider.model_identity, protocol=PROTOCOL)
        if packet["selection"] != "raw" or actual_identity != packet["specification"]["identity"]:
            raise ValueError("export source/native model identity or RAW selection differs")
        slider.network.load_state_dict(packet["network"], strict=True)
        values = []
        zero = slider.response(hidden, strength=0.)
        if not torch.equal(zero, torch.nn.functional.linear(hidden, cache["readout_weight"])[:, :2]):
            raise RuntimeError("live OFF path differs from native readout")
        # Public Glide.decide exercises the full encoder and hooked readout, not
        # just cached vectors. Compare its probabilities against direct adapter.
        for scale in (0., .5, 1.):
            with slider.scaled(scale):
                public = model.decide_batch(requests)
            expected = torch.softmax(slider.response(hidden, strength=scale).detach().double(), -1)
            errors = []
            for row, response in zip(expected, public):
                p = response["answers"]["preference"]["probabilities"]
                errors.append(max(abs(float(row[j]) - p[k]) for j, k in enumerate(p)))
            if max(errors) > 2e-5:
                raise RuntimeError("full native public API differs from cached adapter")
            values.append(dict(strength=scale, max_probability_error=max(errors), responses=public))
        report = dict(model_identity=identity, public_live_checks=values, example_ids=[selection[i]["id"] for i in indices],
                      zero_exact_native=True, raw_export=True, encoder_frozen=not any(p.requires_grad for p in model.network.model.parameters()),
                      direct_logits_max_cache_difference=float((direct_logits[:, :2] - cache["neutral_logits"][indices]).abs().max()),
                      finished=now(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    finally:
        slider.close()
    write_json(root / "live-check.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model-dir", required=True); p.add_argument("--lock", default="native-model.lock.json")
    p.add_argument("--dataset", default="data/concise-preference-v1.json")
    p.add_argument("--cache", required=True); p.add_argument("--readout", required=True)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("experiment")
    p.add_argument("--cache", required=True); p.add_argument("--readout", required=True)
    p.add_argument("--output", required=True); p.add_argument("--steps", type=int, default=128)
    p.add_argument("--rates", type=float, nargs="+", default=[.0001, .001, .01])
    p.add_argument("--batch", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("live-check")
    p.add_argument("--model-dir", required=True); p.add_argument("--lock", default="native-model.lock.json")
    p.add_argument("--dataset", default="data/concise-preference-v1.json")
    p.add_argument("--cache", required=True); p.add_argument("--experiment", required=True)
    p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_cache(args.model_dir, args.lock, args.dataset, args.cache, args.readout, threads=args.threads)
        write_json(str(args.cache) + ".receipt.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    elif args.command == "experiment":
        result = run_experiment(args.cache, args.readout, args.output, steps=args.steps,
                                rates=tuple(args.rates), batch=args.batch, threads=args.threads)
        print(json.dumps(dict(winner=result["winner"]["generator_lr"], test=result["winner_test"], actual_updates=result["actual_updates"])), flush=True)
    else:
        result = live_check(args.model_dir, args.lock, args.dataset, args.cache, args.experiment, threads=args.threads)
        print(json.dumps(dict(live_public_api=True, example_ids=result["example_ids"])), flush=True)


if __name__ == "__main__":
    main()
