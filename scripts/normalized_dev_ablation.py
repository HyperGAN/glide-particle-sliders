#!/usr/bin/env python3
"""One fixed512-update normalized adapter arm; TRAIN/DEV only, never TEST."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from particle_sliders.inference import open_inference
from glide_particle_sliders import binding_normalized_v3 as module
from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.cli import source_receipt, write_json, session_report
from glide_particle_sliders.native import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic", required=True, choices=["conditional16", "residual64"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.output); root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cache_file = Path("artifacts/label-probe-inputs-v1/synthetic-label-cache.pt").resolve()
    readout_file = Path("artifacts/readout-v2.pt").resolve()
    cache = load_readout_file(cache_file, cache=True)
    cfg = module.config(cache_file=str(cache_file), steps=512, batch=8, seed=7, lr=.001,
                        d_lr_mult=18., prior_lr_mult=30., device="cpu", critic=args.critic)
    report = dict(schema="glide-normalized-dev-ablation-v3", started=datetime.now(timezone.utc).isoformat(),
        source=source_receipt(), script_sha256=sha256(__file__), config=cfg.as_dict(),
        cache_sha256=sha256(cache_file), test_evaluated=False, snapshots=[],
        protocol="Fixed 2x2 diagnostic; normalized adapter vs raw; conditional16 vs residual64; all PG math unchanged")
    with module.BACKEND.open(cfg) as run:
        report["normalization"] = dict(center_sha256=__import__('hashlib').sha256(run.game.network.input_center.numpy().tobytes()).hexdigest(),
            rms=float(run.game.network.input_rms), train_only=True, serialized_raw_network_buffers=True)
        with (root / "updates.jsonl").open("w") as log:
            def log_step(stats):
                log.write(json.dumps(stats, allow_nan=False) + "\n"); log.flush()
            for stop in (128, 256, 512):
                run.train(stop_after=stop, on_step=log_step)
                checkpoint = root / f"step-{stop}.pt"
                packet = run.checkpoint(checkpoint)
                stats = dict(completed_steps=stop, packet_sha256=packet["sha256"],
                    train=session_report(run, cache, "train"), dev=session_report(run, cache, "dev"))
                report["snapshots"].append(stats)
                print(json.dumps(dict(event="normalized_snapshot", critic=args.critic, step=stop,
                    train=stats["train"]["mean"], dev=stats["dev"]["mean"])), flush=True)
                if stop == 256:
                    log_step(run.step()); expected = run.checkpoint(root / "next-uninterrupted.pt")
                    with module.BACKEND.open(cfg, resume=checkpoint) as resumed:
                        resumed.step(); actual = resumed.checkpoint(root / "next-resumed.pt")
                    if expected["sha256"] != actual["sha256"]:
                        raise RuntimeError("normalized next update differs after full-state resume")
                    report["exact_resume"] = dict(from_step=256, to_step=257, packet_sha256=actual["sha256"])
                write_json(root / "progress.json", report)
        if run.ema.values is not None or run.ema.decay is not None:
            raise RuntimeError("unexpected EMA")
        slider = run.game.backend.slider
        report["native_mechanics"] = dict(frozen_readout_exact=torch.equal(slider.model.weight.detach(), cache["readout_weight"]),
            inactive_up_weight_max=float(slider.adapter.lora_up.weight[2:].detach().abs().max()), ema_decay=None)
        export = root / "raw-inference.pt"
        run.export_inference(export, selection="raw")
        # Only TRAIN hidden states are passed to this diagnostic reload.
        indices = [i for i,s in enumerate(cache["splits"]) if s == "train"]
        hidden = cache["neutral_hidden"][indices]
        expected = slider.response(hidden).detach().clone()
        with open_inference(export, backend=module.INFERENCE, operation="response",
                            options={"readout_file": str(readout_file)}) as inference:
            if not torch.equal(expected, inference.response(hidden=hidden)):
                raise RuntimeError("normalized raw export differs")
            report["raw_export_exact"] = True
    report.update(completed_steps=512, unique_updates=512, physical_updates=513,
                  deliberate_duplicate_updates=1, finished=datetime.now(timezone.utc).isoformat())
    write_json(root / "results.json", report)


if __name__ == "__main__":
    main()
