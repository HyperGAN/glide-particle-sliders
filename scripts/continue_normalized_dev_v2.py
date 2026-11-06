#!/usr/bin/env python3
"""Continue saved257 state after a launcher API error; never replay the prefix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig, read_packet
from glide_particle_sliders import binding_normalized_v3 as module
from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.cli import source_receipt, write_json, session_report
from glide_particle_sliders.native import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    prior = Path(args.prior); root = Path(args.output); root.mkdir(exist_ok=False)
    prefix = [json.loads(line) for line in (prior / "updates.jsonl").read_text().splitlines()]
    if [row["step"] for row in prefix] != list(range(1, 258)):
        raise ValueError("expected a saved contiguous257-update prefix")
    metadata = json.loads((prior / "progress.json").read_text())
    cfg = SessionConfig.from_dict(metadata["config"])
    if cfg.backend != module.NAME or cfg.options["steps"] != 512:
        raise ValueError("only this exact normalized512 cohort can continue")
    original256 = read_packet(prior / "step-256.pt")
    original257 = read_packet(prior / "next-uninterrupted.pt")
    if original257["payload"]["game"]["completed_steps"] != 257:
        raise ValueError("strict continuation requires the last committed257 state")
    cache = load_readout_file(cfg.options["cache_file"], cache=True)
    torch.set_num_threads(4)
    report = dict(schema="glide-normalized-dev-ablation-v3-continuation", started=datetime.now(timezone.utc).isoformat(),
        source=source_receipt(), script_sha256=sha256(__file__), config=cfg.as_dict(),
        cache_sha256=sha256(cfg.options["cache_file"]), test_evaluated=False,
        recovered_from=dict(path=str(prior), status="launcher failed after257 committed updates; no result relabeling",
            error="TypeError: prepare() got an unexpected keyword argument 'resume'",
            correction="BACKEND.open then shared session.restore; unchanged native/scientific sources and options",
            last_committed_checkpoint_sha256=sha256(prior / "next-uninterrupted.pt"),
            saved_completed_steps=257, replayed_prefix_updates=0), snapshots=[])
    # Evaluate the saved256 endpoint and repeat just its next update, proving
    # exact full-state recovery against the original already-saved257 packet.
    with module.BACKEND.open(cfg) as witness:
        witness.restore(original256)
        stats256 = dict(completed_steps=256, packet_sha256=original256["sha256"],
            train=session_report(witness, cache, "train"), dev=session_report(witness, cache, "dev"))
        witness.step(); actual = witness.checkpoint(root / "next-resumed.pt")
        if actual["sha256"] != original257["sha256"]:
            raise RuntimeError("original256→257 full-state replay differs")
    report["exact_resume"] = dict(from_step=256, to_step=257, packet_sha256=actual["sha256"])
    report["snapshots"].extend([metadata["snapshots"][0], stats256])
    with module.BACKEND.open(cfg) as run:
        run.restore(original257)
        if run.game.completed_steps != 257:
            raise ValueError("continuation did not retain actual257 completed updates")
        center = run.game.network.input_center.clone(); rms = run.game.network.input_rms.clone()
        report["normalization"] = dict(rms=float(rms), train_only=True, serialized_raw_network_buffers=True)
        with (root / "updates-new.jsonl").open("w") as log:
            def log_step(stats):
                log.write(json.dumps(stats, allow_nan=False) + "\n"); log.flush()
            run.train(stop_after=512, on_step=log_step)
        final = run.checkpoint(root / "step-512.pt")
        stats512 = dict(completed_steps=512, packet_sha256=final["sha256"],
            train=session_report(run, cache, "train"), dev=session_report(run, cache, "dev"))
        report["snapshots"].append(stats512)
        if not torch.equal(center, run.game.network.input_center) or not torch.equal(rms, run.game.network.input_rms):
            raise RuntimeError("frozen normalization state changed")
        if run.ema.values is not None or run.ema.decay is not None:
            raise RuntimeError("unexpected EMA")
        slider = run.game.backend.slider
        report["native_mechanics"] = dict(frozen_readout_exact=torch.equal(slider.model.weight.detach(), cache["readout_weight"]),
            inactive_up_weight_max=float(slider.adapter.lora_up.weight[2:].detach().abs().max()), ema_decay=None,
            normalization_frozen_exact=True)
        export = root / "raw-inference.pt"; run.export_inference(export, selection="raw")
        indices = [i for i,s in enumerate(cache["splits"]) if s == "train"]
        hidden = cache["neutral_hidden"][indices]; expected = slider.response(hidden).detach().clone()
        with open_inference(export, backend=module.INFERENCE, operation="response",
                            options={"readout_file": str(Path("artifacts/readout-v2.pt").resolve())}) as inference:
            if not torch.equal(expected, inference.response(hidden=hidden)):
                raise RuntimeError("normalized raw export differs")
            report["raw_export_exact"] = True
    suffix = [json.loads(line) for line in (root / "updates-new.jsonl").read_text().splitlines()]
    if [row["step"] for row in suffix] != list(range(258, 513)):
        raise ValueError("continuation suffix is not contiguous258→512")
    (root / "updates-full.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prefix + suffix))
    report.update(completed_steps=512, unique_updates=512, physical_updates=513,
        original_attempt_updates=257, continuation_new_updates=255, deliberate_duplicate_updates=1,
        finished=datetime.now(timezone.utc).isoformat())
    write_json(root / "results.json", report)
    print(json.dumps(dict(critic=cfg.options["critic"], step=512,
                         train=stats512["train"]["mean"], dev=stats512["dev"]["mean"])), flush=True)


if __name__ == "__main__":
    main()
