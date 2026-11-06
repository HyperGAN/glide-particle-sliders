#!/usr/bin/env python3
"""One predeclared new-seed confirmation; never reselect on confirmation data."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig, read_packet
from glide_particle_sliders import binding_normalized_v3 as module
from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.cli import source_receipt, session_report, write_json
from glide_particle_sliders.native import sha256


def main():
    root = Path("artifacts/hpo-v5/seed11-confirmation"); root.mkdir(exist_ok=False)
    plan_path = Path("artifacts/hpo-v5/plan.json"); plan = json.loads(plan_path.read_text())
    selection_path = plan_path.parent / "selection.json"; selection = json.loads(selection_path.read_text())
    winner = selection["winner"]
    candidate = next(c for c in plan["candidates"] if c["id"] == winner["id"])
    cfg7 = SessionConfig.from_dict(candidate["config"])
    options = dict(cfg7.options, seed=11)
    cfg11 = SessionConfig(cfg7.backend, cfg7.cohort, **options)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    report = dict(schema="glide-hpo-new-seed-confirmation-v5", started=datetime.now(timezone.utc).isoformat(),
        hpo_plan_sha256=sha256(plan_path), selection_sha256=sha256(selection_path),
        source=source_receipt(), script_sha256=sha256(__file__), selected_arm=candidate["id"],
        config=cfg11.as_dict(), seed=11, horizon=512, reselected=False,
        test_evaluated=False, fresh_evaluated=False, unrelated_evaluated=False, snapshots=[])
    if winner["status"] == "reused":
        reused = json.loads(Path(winner["provenance"]["report"]).read_text())
        report["selected_seed7_resume"] = dict(reused_exact_proof=reused["exact_resume"], additional_physical_updates=0)
        report["seed7_artifact"] = winner["provenance"]["raw_export"]
    else:
        checkpoint256 = winner["checkpoints"]["256"]["path"]
        expected = read_packet(winner["checkpoints"]["257"]["path"])
        with module.BACKEND.open(cfg7) as witness:
            witness.restore(read_packet(checkpoint256)); witness.step()
            observed = witness.checkpoint(root / "selected-seed7-step257-resumed.pt")
        if observed["sha256"] != expected["sha256"]:
            raise RuntimeError("selected HPO checkpoint continuation differs")
        report["selected_seed7_resume"] = dict(exact_packet=True,from_step=256,to_step=257,
            packet_sha256=observed["sha256"],additional_physical_updates=1,replayed_prefix_updates=0)
        report["seed7_artifact"] = winner["raw_export"]["path"]
    cache = load_readout_file(options["cache_file"], cache=True)
    times=[]
    with module.BACKEND.open(cfg11) as run:
        with (root/"updates.jsonl").open("x") as log:
            while run.game.completed_steps < 512:
                t=time.perf_counter(); stats=run.step(); dt=time.perf_counter()-t;times.append(dt)
                log.write(json.dumps(dict(stats, wall_step_seconds=dt),allow_nan=False)+"\n");log.flush()
                step=run.game.completed_steps
                if step in (128,256,257,512):
                    p=root/f"step-{step}.pt";packet=run.checkpoint(p)
                    report.setdefault("checkpoints",{})[str(step)]=dict(path=str(p),sha256=sha256(p),packet_sha256=packet["sha256"])
                if step in (128,256,512):
                    report["snapshots"].append(dict(completed_steps=step,train=session_report(run,cache,"train"),dev=session_report(run,cache,"dev")))
                    print(json.dumps(dict(seed=11,step=step,dev=report["snapshots"][-1]["dev"]["mean"])),flush=True)
                    write_json(root/"progress.json",report)
        if run.ema.values is not None or run.ema.decay is not None: raise RuntimeError("EMA enabled")
        artifact=root/"raw-inference.pt";run.export_inference(artifact,selection="raw")
        ids=[i for i,s in enumerate(cache["splits"]) if s=="train"];hidden=cache["neutral_hidden"][ids]
        expected=run.game.backend.slider.response(hidden).detach().clone()
        with open_inference(artifact,backend=module.INFERENCE,operation="response",options={"readout_file":plan["assets"]["readout"]["path"]}) as inference:
            if not torch.equal(inference.response(hidden=hidden),expected): raise RuntimeError("raw seed11 export differs")
        report["raw_export"]=dict(path=str(artifact),sha256=sha256(artifact),exact=True)
    with module.BACKEND.open(cfg11) as witness:
        witness.restore(read_packet(root/"step-256.pt"));witness.step();observed=witness.checkpoint(root/"seed11-step257-resumed.pt")
        if observed["sha256"] != report["checkpoints"]["257"]["packet_sha256"]:
            raise RuntimeError("seed11 next-update replay differs")
    report.update(seed11_exact_resume=dict(from_step=256,to_step=257,packet_sha256=observed["sha256"],replayed_prefix_updates=0),
        completed_steps=512,new_unique_updates=512,seed11_physical_updates=513,
        selected_seed7_additional_physical_updates=report["selected_seed7_resume"]["additional_physical_updates"],
        throughput=dict(measured_updates=504,warmup_updates_excluded=8,updates_per_second=504/sum(times[8:])),
        ema_decay=None,finished=datetime.now(timezone.utc).isoformat())
    write_json(root/"results.json",report)


if __name__=="__main__":main()
