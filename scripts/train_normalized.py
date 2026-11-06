#!/usr/bin/env python3
"""Convenience entrypoint into the shared normalized GLiDE lifecycle."""
import argparse
import json
from pathlib import Path

import torch

from glide_particle_sliders import binding_normalized_v3 as backend
from glide_particle_sliders.cli import source_receipt, write_json
from glide_particle_sliders.native import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=512); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=.001); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--critic", choices=["conditional16", "residual64"], default="residual64")
    args = p.parse_args()
    cfg = backend.config(cache_file=str(Path(args.cache).resolve()), steps=args.steps, batch=args.batch,
        lr=args.lr, seed=args.seed, d_lr_mult=18., prior_lr_mult=30., critic=args.critic, device="cpu", ema_decay=None)
    root = Path(args.output); root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    with backend.BACKEND.open(cfg) as session:
        with (root / "updates.jsonl").open("w") as log:
            def record(stats):
                log.write(json.dumps(stats, allow_nan=False) + "\n"); log.flush()
            for stop in [*range(128, args.steps, 128), args.steps]:
                session.train(stop_after=stop, on_step=record)
                session.checkpoint(root / f"step-{stop}.pt")
        session.export_inference(root / "raw-inference.pt", selection="raw")
        write_json(root / "receipt.json", dict(config=cfg.as_dict(), source=source_receipt(),
            cache_sha256=sha256(args.cache), completed_steps=session.game.completed_steps,
            ema_decay=session.ema.decay, artifact_sha256=sha256(root / "raw-inference.pt")))


if __name__ == "__main__":
    main()
