#!/usr/bin/env python3
"""Blind, no-training confirmation of frozen HPO artifacts and unrelated tasks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from particle_sliders.inference import open_inference, read_inference, native_identity
from glide_particle_sliders import binding as original
from glide_particle_sliders import binding_normalized_v3 as module
from glide_particle_sliders.cli import source_receipt, write_json
from glide_particle_sliders.evaluation import relative_sse, diagnostics
from glide_particle_sliders.native import sha256, load_glide, encode_requests
from glide_particle_sliders.preference_metrics import preference_metrics


def unrelated_metrics(values, neutral, rows):
    s, n = values.detach().double(), neutral.detach().double()
    p, p0 = torch.softmax(s, -1), torch.softmax(n, -1)
    correct = torch.tensor([r["preferred_slot"] for r in rows])
    choice, baseline_choice = s.argmax(1), n.argmax(1)
    accurate, accurate0 = choice == correct, baseline_choice == correct
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(row["group"], []).append(i)
    return dict(cases=len(groups), ordered_judgments=len(rows),
        baseline_accuracy=float(accurate0.double().mean()),
        accuracy=float(accurate.double().mean()),
        both_orders_correct_rate=sum(bool(accurate[g].all()) for g in groups.values())/len(groups),
        baseline_both_orders_correct_rate=sum(bool(accurate0[g].all()) for g in groups.values())/len(groups),
        harmful_flip_rate=float((accurate0 & ~accurate).double().mean()),
        beneficial_flip_rate=float((~accurate0 & accurate).double().mean()),
        flip_rate=float((choice != baseline_choice).double().mean()),
        cases_with_any_order_flip=sum(bool((choice[g] != baseline_choice[g]).any()) for g in groups.values())/len(groups),
        mean_abs_probability_change=float((p-p0).abs().mean()), max_abs_probability_change=float((p-p0).abs().max()),
        mean_native_to_slider_kl=float((p0 * (torch.log_softmax(n, -1)-torch.log_softmax(s, -1))).sum(1).mean()),
        baseline_mean_p_correct=float(p0[torch.arange(len(rows)), correct].mean()),
        mean_p_correct=float(p[torch.arange(len(rows)), correct].mean()),
        ranking_gate=False, raw_relative_sse="undefined/excluded: teacher equals neutral exactly")


def main():
    plan_file = Path("artifacts/hpo-v5/fresh-confirmation-plan.json")
    plan_bytes = plan_file.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    if sha256(__file__) != plan["evaluation_script_sha256"]:
        raise ValueError("evaluation script differs from frozen HPO confirmation plan")
    for name, expected in plan["product_sources"].items():
        if sha256(name) != expected: raise ValueError("product source changed: " + name)
    data_file = Path(plan["dataset_file"]); data_bytes = data_file.read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("fresh dataset changed")
    for arm in plan["models"]:
        if sha256(arm["artifact"]) != arm["artifact_sha256"]: raise ValueError("frozen HPO artifact changed")
        packet = read_inference(arm["artifact"])
        if (packet["selection"] != "raw" or packet["specification"]["backend"] != module.NAME
                or packet["specification"]["configuration"]["seed"] != arm["seed"]):
            raise ValueError("artifact seed/backend/RAW provenance differs from frozen arm label")
    if (any(type(s) is not float for s in plan["strengths"])
            or not {0., .5, 1.} <= set(plan["strengths"])):
        raise ValueError("frozen strengths must explicitly include float0/.5/1")
    root = Path(plan["output_directory"]); root.mkdir(exist_ok=False)
    rows = json.loads(data_bytes)["rows"]
    fresh_ids = [i for i,r in enumerate(rows) if r["split"] == "fresh"]
    unrelated_ids = [i for i,r in enumerate(rows) if r["split"] == "unrelated"]
    if len(rows) != 104 or len(fresh_ids) != 80 or len(unrelated_ids) != 24:
        raise ValueError("expected80 preference plus24 unrelated ordered rows")
    torch.set_num_threads(4)
    started = datetime.now(timezone.utc).isoformat()
    model, identity = load_glide("artifacts/native-model", "native-model.lock.json", threads=4)
    start = time.perf_counter()
    hidden, original_logits, lengths = encode_requests(model, [r["neutral"] for r in rows], batch_size=2)
    native_encoding_seconds = time.perf_counter() - start
    with torch.no_grad(): neutral = model.network.readout(hidden)[:, :2].detach().clone()
    teacher = neutral.clone()
    for i in fresh_ids:
        teacher[i] -= 1.5; teacher[i, rows[i]["preferred_slot"]] += 3.
    if sha256(data_file) != plan["dataset_sha256"]: raise ValueError("dataset changed during encoding")
    if plan_file.read_bytes() != plan_bytes: raise ValueError("confirmation plan changed during encoding")
    torch.save(dict(neutral_hidden=hidden, neutral_logits=neutral, teacher_logits=teacher,
        readout_weight=model.network.readout.weight.detach().clone(), rows=rows,
        model_identity=identity, dataset_sha256=plan["dataset_sha256"], lengths=lengths), root/"fresh-cache.pt")
    report = dict(schema="glide-hpo-fresh-confirmation-v5", plan=plan, plan_sha256=plan_sha256,
        source=source_receipt(), model_identity=identity, training_updates_on_confirmation=0,
        native_encoding_seconds=native_encoding_seconds, encoded_ordered_requests=len(rows),
        native_encoding_requests_per_second=len(rows)/native_encoding_seconds,
        encoder_batch2_vs_bulk_readout_max_difference=float((original_logits[:, :2]-neutral).abs().max()),
        models=[], started=started, gpu_launches=0)
    fresh_rows=[rows[i] for i in fresh_ids]; unrelated_rows=[rows[i] for i in unrelated_ids]
    for arm in plan["models"]:
        measured=dict(name=arm["name"],seed=arm["seed"],artifact_sha256=arm["artifact_sha256"],strengths={})
        predictions={}
        with open_inference(arm["artifact"],backend=module.INFERENCE,operation="response",
                            options={"readout_file":plan["readout_file"]}) as inference:
            for strength in plan["strengths"]:
                values=inference.response(hidden=hidden,strength=strength)
                predictions[str(strength)]=values
                combined=relative_sse(values,teacher,neutral)  # finite even excluded unrelated fields
                if combined["valid"] != [r["split"] == "fresh" for r in rows]:
                    raise ValueError("exact teacher-only exclusion set differs")
                measured["strengths"][str(strength)]=dict(
                    primary=combined,
                    fresh_relative_sse=diagnostics(values[fresh_ids],teacher[fresh_ids],neutral[fresh_ids],fresh_rows),
                    preference=preference_metrics(values[fresh_ids],neutral[fresh_ids],fresh_rows),
                    unrelated=unrelated_metrics(values[unrelated_ids],neutral[unrelated_ids],unrelated_rows))
            if not torch.equal(predictions["0.0"],neutral): raise RuntimeError("OFF changed native readout")
        measured["off_exact_native"]=True
        # Full native public-API replay includes one main, one correctness and one
        # unrelated scenario, always both orders, chosen by file order alone.
        packet=read_inference(arm["artifact"])
        slider=module.NormalizedReadoutSlider(model.network.readout,model_identity=identity,
            center=torch.zeros(original.HIDDEN),rms=torch.ones(()),seed=arm["seed"])
        native=module.NormalizedNativeResponse(slider)
        try:
            if packet["selection"]!="raw" or native_identity(native,model_identity=slider.model_identity,protocol=module.PROTOCOL)!=packet["specification"]["identity"]:
                raise ValueError("live native source/model identity differs")
            slider.network.load_state_dict(packet["network"],strict=True)
            chosen=[]
            for category in ("main","control","unrelated"):
                group=next(r["group"] for r in rows if (r["split"]=="unrelated" if category=="unrelated" else r["split"]=="fresh" and r["control"]==(category=="control")))
                chosen.extend(i for i,r in enumerate(rows) if r["group"]==group)
            live=[]
            for strength in (0.,.5,1.):
                with slider.scaled(strength): actual=model.decide_batch([rows[i]["neutral"] for i in chosen])
                expected=torch.softmax(predictions[str(strength)][chosen].double(),-1)
                errors=[]
                for probabilities,response in zip(expected,actual):
                    p=response["answers"]["preference"]["probabilities"]
                    errors.append(max(abs(float(probabilities[j])-p[k]) for j,k in enumerate(p)))
                if max(errors)>2e-5: raise RuntimeError("public API differs from RAW exported head")
                live.append(dict(strength=strength,max_probability_error=max(errors)))
            measured["live_api"]=dict(row_ids=[rows[i]["id"] for i in chosen],scenario_groups=3,scales=live)
        finally: slider.close()
        torch.save(predictions,root/(arm["name"]+"-predictions.pt"))
        report["models"].append(measured)
        write_json(root/"progress.json",report)
    if plan_file.read_bytes() != plan_bytes: raise ValueError("confirmation plan changed during evaluation")
    report["finished"]=datetime.now(timezone.utc).isoformat();write_json(root/"results.json",report)
    for arm in report["models"]:
        v=arm["strengths"]["1.0"]
        print(json.dumps(dict(model=arm["name"],raw_sse=v["primary"]["mean"],main=v["preference"]["main"],controls=v["preference"]["control"],unrelated=v["unrelated"])),flush=True)


if __name__=="__main__": main()
