#!/usr/bin/env python3
"""Evaluate one preselected RAW artifact on fresh frozen scenarios; no training."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

import torch

from particle_sliders.inference import open_inference, read_inference, native_identity
from glide_particle_sliders import binding as original
from glide_particle_sliders import binding_normalized_v3 as normalized
from glide_particle_sliders.cli import write_json, source_receipt
from glide_particle_sliders.evaluation import diagnostics
from glide_particle_sliders.native import load_glide, encode_requests, sha256
from glide_particle_sliders.preference_metrics import preference_metrics


def main():
    plan_path = Path("artifacts/fresh-confirmation-plan-v4.json")
    plan = json.loads(plan_path.read_text())
    if sha256(__file__) != plan["evaluation_script_sha256"]:
        raise ValueError("fresh evaluation source differs from frozen plan")
    for path, expected in plan["product_sources"].items():
        if sha256(path) != expected:
            raise ValueError("product source differs from frozen fresh plan: " + path)
    dataset_path = Path(plan["dataset_file"])
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("fresh dataset differs from the predeclared dataset")
    for key in ("artifact", "ridge_coefficients"):
        if sha256(plan[key]["file"]) != plan[key]["sha256"]:
            raise ValueError("preselected artifact differs: " + key)
    module = importlib.import_module(plan["backend_module"])
    root = Path(plan["output_directory"]); root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    started = datetime.now(timezone.utc).isoformat()
    rows = json.loads(dataset_bytes)["rows"]
    if len(rows) != 80 or len({r["group"] for r in rows}) != 40:
        raise ValueError("expected the frozen40 scenario/80 order design")
    model, identity = load_glide("artifacts/native-model", "native-model.lock.json", threads=4)
    hidden, original_logits, lengths = encode_requests(model, [r["neutral"] for r in rows], batch_size=2)
    with torch.no_grad():
        # Same native head geometry for all evaluated arms and both operands.
        # The frozen encoder itself retains its native two-row batch geometry.
        neutral = model.network.readout(hidden)[:, :2].detach().clone()
    delta = torch.full_like(neutral, -1.5)
    for i, row in enumerate(rows):
        delta[i, row["preferred_slot"]] = 1.5
    teacher = neutral + delta
    cache = dict(schema=original.CACHE_SCHEMA, model_identity=identity, neutral_hidden=hidden,
        teacher_logits=teacher, neutral_logits=neutral, readout_weight=model.network.readout.weight.detach().clone(),
        readout_bias=None, row_ids=[r["id"] for r in rows], splits=["fresh"]*len(rows),
        metadata=dict(dataset=rows, dataset_sha256=plan["dataset_sha256"], lengths=lengths,
            teacher_protocol="synthetic preferred-slot delta+1.5/other-1.5; fresh frozen scenarios; no teacher prompting",
            native_encoder_readout_batch2_vs_bulk_readout_max_difference=float((original_logits[:, :2] - neutral).abs().max())))
    if sha256(dataset_path) != plan["dataset_sha256"]:
        raise ValueError("fresh dataset changed during encoding")
    torch.save(cache, root / "fresh-cache.pt")
    report = dict(schema="glide-fresh-preference-evaluation-v4", started=started, plan=plan,
        plan_sha256=sha256(plan_path), source=source_receipt(), dataset_sha256=plan["dataset_sha256"],
        fresh_cache_sha256=sha256(root / "fresh-cache.pt"), native_model_identity=identity,
        training_updates_on_fresh_data=0, strengths={}, gpu_launches=0, weight_ema=False)
    predicted = {}
    with open_inference(plan["artifact"]["file"], backend=module.INFERENCE, operation="response",
                        options={"readout_file": plan["readout_file"]}) as inference:
        for strength in plan["strengths"]:
            values = inference.response(hidden=hidden, strength=strength)
            predicted[str(strength)] = values
            report["strengths"][str(strength)] = dict(
                relative_sse=diagnostics(values, teacher, neutral, rows),
                preference=preference_metrics(values, neutral, rows))
        if not torch.equal(predicted["0.0"], neutral):
            raise RuntimeError("fresh strength0 differs from the exact native readout")
        report["off_exact_native"] = True
    selected = predicted["1.0"]
    ridge = torch.load(plan["ridge_coefficients"]["file"], map_location="cpu", weights_only=True)
    ridge_pred = neutral.double() + (hidden.double() - ridge["feature_train_mean"]) @ ridge["beta"] + ridge["residual_train_mean"]
    report["frozen_ridge_control"] = dict(
        relative_sse=diagnostics(ridge_pred, teacher, neutral, rows),
        preference=preference_metrics(ridge_pred, neutral, rows),
        regularization=plan["ridge_coefficients"]["regularization"])
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42))
    scrambled = neutral + (selected - neutral)[permutation]
    report["single_scrambled_delta_diagnostic"] = dict(seed=42,
        relative_sse=diagnostics(scrambled, teacher, neutral, rows),
        preference=preference_metrics(scrambled, neutral, rows),
        limitation="One fixed permutation, not a distribution over random controllers")
    report["synthetic_target"] = dict(relative_sse=diagnostics(teacher, teacher, neutral, rows),
        preference=preference_metrics(teacher, neutral, rows),
        limitation="Instructor-authored target; not a native instruction teacher or personal user preference labels")
    torch.save(dict(predictions=predicted, neutral=neutral, teacher=teacher, frozen_ridge=ridge_pred), root / "predictions.pt")
    # Exercise the full public API with this exact exported RAW network attached
    # to the genuine encoder. Select the first main and first control scenario,
    # retaining both option orders for each; no quality-dependent sample choice.
    packet = read_inference(plan["artifact"]["file"])
    if module.NAME == normalized.NAME:
        slider = normalized.NormalizedReadoutSlider(model.network.readout, model_identity=identity,
            center=torch.zeros(original.HIDDEN), rms=torch.ones(()), seed=packet["specification"]["configuration"]["seed"])
        native = normalized.NormalizedNativeResponse(slider); protocol = normalized.PROTOCOL
    else:
        slider = original.ReadoutSlider(model.network.readout, model_identity=identity,
            seed=packet["specification"]["configuration"]["seed"])
        native = module.NativeResponse(slider) if hasattr(module, "NativeResponse") else original.NativeReadoutResponse(slider)
        protocol = module.PROTOCOL
    try:
        if packet["selection"] != "raw" or native_identity(native, model_identity=slider.model_identity, protocol=protocol) != packet["specification"]["identity"]:
            raise ValueError("live native model/source identity differs from selected export")
        slider.network.load_state_dict(packet["network"], strict=True)
        wanted = []
        for control in (False, True):
            first_group = next(r["group"] for r in rows if r["control"] == control)
            wanted.extend(i for i, r in enumerate(rows) if r["group"] == first_group)
        live = []
        for strength in (0., .5, 1.):
            with slider.scaled(strength):
                responses = model.decide_batch([rows[i]["neutral"] for i in wanted])
            expected = torch.softmax(predicted[str(strength)][wanted].double(), -1)
            errors = []
            for probs, response in zip(expected, responses):
                actual = response["answers"]["preference"]["probabilities"]
                errors.append(max(abs(float(probs[j]) - actual[key]) for j, key in enumerate(actual)))
            if max(errors) > 2e-5:
                raise RuntimeError("full native API differs from selected raw head")
            live.append(dict(strength=strength, max_probability_error=max(errors), responses=responses))
        report["live_api"] = dict(rows=[rows[i]["id"] for i in wanted], scenario_groups=2,
            scales=live, frozen_encoder=not any(p.requires_grad for p in model.network.model.parameters()))
    finally:
        slider.close()
    report["finished"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "results.json", report)
    print(json.dumps(dict(fresh_relative_sse=report["strengths"]["1.0"]["relative_sse"]["mean"],
        main=report["strengths"]["1.0"]["preference"]["main"], control=report["strengths"]["1.0"]["preference"]["control"])), flush=True)


if __name__ == "__main__":
    main()
