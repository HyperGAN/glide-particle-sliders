#!/usr/bin/env python3
"""Freeze selected, replicate, and pre-tuning artifacts before new data scores."""
from datetime import datetime, timezone
import json
from pathlib import Path

from glide_particle_sliders.cli import source_receipt, write_json
from glide_particle_sliders.native import sha256


def main():
    root=Path("artifacts/hpo-v5")
    hpo=json.loads((root/"plan.json").read_text())
    selection=json.loads((root/"selection.json").read_text())
    replicate=json.loads((root/"seed11-confirmation/results.json").read_text())
    if not selection["all_declared_arms_completed"] or replicate["reselected"]:
        raise ValueError("require completed fixed search and nonselective seed confirmation")
    winner=selection["winner"];candidate=next(c for c in hpo["candidates"] if c["id"]==winner["id"])
    if replicate["selected_arm"]!=winner["id"] or replicate["completed_steps"]!=512:
        raise ValueError("seed confirmation does not match selected arm/budget")
    compare=dict(candidate["config"]["options"],seed=11)
    if replicate["config"]["options"]!=compare: raise ValueError("replicate changes more than seed")
    anchor=next(c for c in hpo["candidates"] if c["reuse"])["reuse_artifacts"]["raw_export"]
    selected=winner["provenance"]["raw_export"] if winner["status"]=="reused" else winner["raw_export"]["path"]
    raw_models=[("pre-tuning-anchor-seed7",7,anchor),("tuned-seed7",7,selected),
                ("tuned-seed11",11,replicate["raw_export"]["path"])]
    models=[];seen={}
    for name,seed,file in raw_models:
        digest=sha256(file)
        if digest in seen:
            seen[digest]["aliases"].append(name)
        else:
            record=dict(name=name,seed=seed,artifact=str(Path(file).resolve()),artifact_sha256=digest,aliases=[name])
            seen[digest]=record;models.append(record)
    data=Path("data/concise-fresh-v3.json")
    if sha256(data)!="995871d69e3c1b7e0b8ee052d5e2fa68c9a472f8455408d39ba74cf2bf1b9059":
        raise ValueError("new authored holdout differs")
    plan=dict(schema="glide-hpo-blind-confirmation-plan-v5",frozen_at=datetime.now(timezone.utc).isoformat(),
        hpo_plan_sha256=sha256(root/"plan.json"),hpo_selection_sha256=sha256(root/"selection.json"),
        seed11_confirmation_sha256=sha256(root/"seed11-confirmation/results.json"),
        selected_arm=winner["id"],selected_config=candidate["config"],selected_base_rates=candidate["base_rates"],
        selection_metric=hpo["selection"],seed11_not_used_for_reselection=True,models=models,
        dataset_file=str(data.resolve()),dataset_sha256=sha256(data),
        preference_groups=40,preference_ordered_rows=80,unrelated_cases=12,unrelated_ordered_rows=24,
        target="Preference:T=N+delta with preferred+1.5/other-1.5; unrelated:T=N exactly",
        strengths=[0.,.25,.5,.75,1.],readout_file=hpo["assets"]["readout"]["path"],
        output_directory=str((root/"fresh-confirmation").resolve()),
        evaluation_script_sha256=sha256(Path("scripts/evaluate_hpo_fresh_v5.py")),
        product_sources=source_receipt()["product_sources"],
        further_training_or_selection_on_confirmation=False,weights="raw; no EMA",gpu_launches=0,
        metrics=dict(primary="strength1 RAW FP64 equal-example relative SSE on80 preference rows",
                     unrelated="all student values finite; teacher-only exact-zero24 exclusions, flips/accuracy/DeltaP/KL diagnostic",
                     preference="ordered and both-orders scenario choices, mean/worst-order Ppreferred, grouped probability gains"))
    out=root/"fresh-confirmation-plan.json"
    if out.exists(): raise FileExistsError(out)
    write_json(out,plan)
    print(json.dumps(dict(plan_sha256=sha256(out),selected_arm=winner["id"],models=[m["name"] for m in models])),flush=True)


if __name__=="__main__":main()
