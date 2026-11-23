"""Exact publisher, core and raw fit identity for the recurrent entry point."""

from hashlib import sha1, sha256
from importlib import metadata, resources
import io
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np
import torch
from particle_sliders import prompt_steering, strength_schedule

MODEL_ID = "fastino/GLiDE-9B"
MODEL_REVISION = "eff5e5ff7d69e005f5cc0b1e2bc4ae07561f811e"
CORE_COMMIT = "28745b7d47435f89bb55b86d1761adce6b8f1b06"
PSR_SOURCE_SHA256 = "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"
COORDINATE = "block23-final-token-two-euler-passes"
BLOCK17_COORDINATE = "block17-final-token-clocks18-24-fp32"
BLOCK17_SCHEMA = "glide-recurrent-block17-development-v1"
BLOCK17_CORE_COMMIT = "303d0a3fbe6d2258997f1c59dc23822a90a68d75"
BLOCK17_DOSE_SOURCE_SHA256 = "732e8536ec16cf38950edbf0c536f08019844247a87b552d8b7de02096362d6d"
BLOCK17_RAW_SHA256 = "9bae1827330dc3e3034b7cdb8525384f795b60d47ac77930861b65715edf8ac5"
BLOCK17_FULL_SHA256 = "d742bcff516ec08069dab7269db575892d365615c2e314f217461166f6d39d35"
BLOCK17_RECEIPT_SHA256 = "6bb833703629caa30ab57ff468c7443993a3d3f569fbdb15b0d1a25068a73ed1"
BLOCK17_FIT_SHA256 = "9635dbb504924ed8072708634ac9210f9cefc9a50b65db30ba350641bbe65775"
BLOCK17_DEVELOPMENT_SHA256 = "1496d8a85a95e1ad2ab1b90cf690ed79327f628a26a092d86e8978b555bc8bca"
BLOCK17_RANK_SHA256 = "7259c69f46f09683334b0b813dc68466e48963687cf5879b26db1c143037da4c"
AUTHOR_EPOCH_WIDTH = 16
AUTHOR_BUDGET = 80
AUTHOR_STOP_OBJECTIVE = 0.001
AUTHOR_LEARNING_RATE = 0.001
PARAMETER_SHAPES = {"direction.weight": (1, 4096), "probe.weight": (1, 4096), "probe.bias": (1,)}
DEPENDENCIES = {
    "torch": "2.14.1+cu130", "transformers": "5.17.0", "tokenizers": "0.23.3", "safetensors": "0.8.0",
    "huggingface_hub": "1.33.0", "jinja2": "3.1.6", "pillow": "13.0.0", "numpy": "1.26.4",
}


def _strict_object(entries):
    answer = {}
    for key, value in entries:
        if key in answer:
            raise ValueError(f"Repeated JSON field: {key}")
        answer[key] = value
    return answer


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def decode_json(data):
    return json.loads(data, object_pairs_hook=_strict_object, parse_constant=_invalid_constant)


def hash_json(data):
    return sha256(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def require_fields(value, fields, label):
    if not isinstance(value, dict) or value.keys() != set(fields):
        raise ValueError(f"Unexpected {label} fields")


def require_positive(value, label):
    if type(value) in (int, float):
        try:
            value = float(value)
        except OverflowError:
            value = float("inf")
        if isfinite(value) and value > 0:
            return value
    raise ValueError(f"{label} must be finite and positive")


def locked_manifest():
    data = resources.files("glide_recurrent_sliders").joinpath("native-model.lock.json").read_bytes()
    manifest = decode_json(data)
    if (manifest.get("model_id"), manifest.get("revision"), manifest.get("schema")) != (
        MODEL_ID, MODEL_REVISION, "glide-recurrent-assets-v1",
    ):
        raise ValueError("Unexpected bundled GLiDE-9B publisher manifest")
    return manifest, sha256(data).hexdigest()


def model_identity(*, block=23):
    if block == 23:
        coordinate, core = COORDINATE, CORE_COMMIT
    elif block == 17:
        coordinate, core = BLOCK17_COORDINATE, BLOCK17_CORE_COMMIT
    else:
        raise ValueError("Unsupported recurrent steering block")
    return dict(
        model_id=MODEL_ID, revision=MODEL_REVISION, asset_manifest_sha256=locked_manifest()[1],
        core_commit=core, coordinate=coordinate,
    )


def _verify_file(root, name, expected):
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid publisher asset path: {name}")
    path = root / relative
    if not path.is_file() or path.resolve() != path or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Publisher asset path or size differs: {name}")
    if "sha256" in expected:
        digest = sha256()
        reference = expected["sha256"]
    else:
        digest = sha1(usedforsecurity=False)
        digest.update(f"blob {expected['bytes']}\0".encode())
        reference = expected["git_blob_sha1"]
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference:
        raise ValueError(f"Publisher asset digest differs: {name}")


def verify_snapshot(path):
    root = Path(path).resolve(strict=True)
    manifest, _ = locked_manifest()
    for name, expected in manifest["files"].items():
        _verify_file(root, name, expected)
    return root


def verify_runtime():
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Expected the GLiDE-9B Python 3.11 runtime")
    actual = {name: metadata.version(name) for name in DEPENDENCIES}
    if actual != DEPENDENCIES or str(torch.__version__) != DEPENDENCIES["torch"]:
        raise ValueError("Python package versions differ from the GLiDE-9B runtime")
    if sha256(Path(prompt_steering.__file__).read_bytes()).hexdigest() != PSR_SOURCE_SHA256:
        raise ValueError("Shared PromptSteering source differs from the reviewed core pin")


def verify_block17_dose_source():
    if sha256(Path(strength_schedule.__file__).read_bytes()).hexdigest() != BLOCK17_DOSE_SOURCE_SHA256:
        raise ValueError("Shared binary32 dose source differs from the Block 17 evidence")


def _read_pin(path, expected, label):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError(f"Invalid {label} SHA-256")
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"{label} digest differs")
    return data


def _check_parameters(payload):
    require_fields(payload, PARAMETER_SHAPES, "probe parameter")
    for key, shape in PARAMETER_SHAPES.items():
        value = payload[key]
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid fitted probe tensor: {key}")


def _check_final_state(full, count):
    cursor, losses = full["cursor"], full["losses"]
    if (type(count) is not int or not AUTHOR_EPOCH_WIDTH <= count <= AUTHOR_BUDGET
            or count % AUTHOR_EPOCH_WIDTH or len(losses) != count
            or not all(type(value) in (float, int) and isfinite(value) and value >= 0 for value in losses)):
        raise ValueError("A completed author fit is required")
    epoch_means = [float(np.mean(losses[i:i + AUTHOR_EPOCH_WIDTH])) for i in range(0, count, AUTHOR_EPOCH_WIDTH)]
    converged = [i for i, mean in enumerate(epoch_means, 1) if mean < AUTHOR_STOP_OBJECTIVE]
    epochs = count // AUTHOR_EPOCH_WIDTH
    if converged and converged[0] == epochs:
        expected_stop = "epoch_threshold"
    elif not converged and count == AUTHOR_BUDGET:
        expected_stop = "declared_budget"
    else:
        raise ValueError("FULL history does not satisfy the authored stop rule")
    if cursor != dict(completed=count, epoch=epochs, next_order_index=0, epoch_means=epoch_means, stop_reason=expected_stop):
        raise ValueError("A completed author fit cursor is required")
    optimizer, schedule = full["optimizer"], full["scheduler"]
    [group] = optimizer["param_groups"]
    lr = AUTHOR_LEARNING_RATE * ((AUTHOR_BUDGET - count) / AUTHOR_BUDGET)
    if (group["params"] != [0, 1, 2] or group["lr"] != lr or schedule["last_epoch"] != count
            or schedule["_step_count"] != count + 1 or schedule["_last_lr"] != [lr]):
        raise ValueError("FULL AdamW and author scheduler state differ")
    require_fields(optimizer["state"], range(3), "AdamW parameter state")
    for index, shape in enumerate(PARAMETER_SHAPES.values()):
        moments = optimizer["state"][index]
        require_fields(moments, ("step", "exp_avg", "exp_avg_sq"), "AdamW moment")
        if moments["step"].shape != () or moments["step"].item() != count:
            raise ValueError("FULL AdamW update count differs")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moments[name]
            if tensor.shape != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise ValueError("Invalid FULL AdamW moment")
        if (moments["exp_avg_sq"] < 0).any().item():
            raise ValueError("Negative FULL AdamW second moment")
    for name in ("cpu_rng", "cuda_rng", "shuffle_state"):
        rng = full[name]
        if rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() == 0:
            raise ValueError(f"Invalid FULL RNG state: {name}")


def _load_fitted_state(plan, raw_path, full_path, full_receipt_path):
    raw = torch.load(io.BytesIO(_read_pin(raw_path, plan["raw_sha256"], "RAW adapter")), map_location="cpu", weights_only=True)
    full = torch.load(io.BytesIO(_read_pin(full_path, plan["full_sha256"], "FULL checkpoint")), map_location="cpu", weights_only=True)
    require_fields(raw, ("experiment", "full", "completed", "probe", "raw_weights", "ema_decay"), "RAW adapter")
    require_fields(full, ("experiment", "fitted_probe", "optimizer", "scheduler", "cpu_rng", "cuda_rng", "shuffle_state",
                          "orders_sha256", "losses", "cursor", "raw_weights", "ema_decay", "mutation_marker"), "FULL checkpoint")
    receipt_sha = raw["full"]["sha256"]
    if plan["schema"] == BLOCK17_SCHEMA and receipt_sha != BLOCK17_RECEIPT_SHA256:
        raise ValueError("Block 17 RAW does not identify the reviewed FULL receipt")
    receipt = decode_json(_read_pin(full_receipt_path, receipt_sha, "FULL receipt"))
    if (receipt["state"]["sha256"] != plan["full_sha256"]
            or receipt["experiment"] != raw["experiment"] or receipt["ema_decay"] is not None
            or receipt["mutation_marker"] != full["mutation_marker"]):
        raise ValueError("The RAW parent receipt does not identify the FULL checkpoint")
    cursor = full["cursor"]
    count = plan["completed_updates"]
    if count != cursor["completed"] or raw["completed"] != count or receipt["cursor"] != cursor or receipt["update"] != count:
        raise ValueError("A completed author fit is required")
    _check_final_state(full, count)
    experiment = full["experiment"]
    if (full["raw_weights"] is not True or raw["raw_weights"] is not True
            or full["ema_decay"] is not None or raw["ema_decay"] is not None
            or experiment != raw["experiment"] or experiment["sha256"] != plan["fit_recipe_sha256"]
            or full["orders_sha256"] != experiment["sha256"] or full["scheduler"]["last_epoch"] != count):
        raise ValueError("RAW and FULL optimizer or weight provenance differs")
    _check_parameters(raw["probe"])
    _check_parameters(full["fitted_probe"])
    if any(not torch.equal(tensor, full["fitted_probe"][key]) for key, tensor in raw["probe"].items()):
        raise ValueError("RAW adapter tensors differ from the completed FULL checkpoint")
    return raw["probe"]


def _block17_plan(trial_strength):
    return dict(
        schema=BLOCK17_SCHEMA, identity=model_identity(block=17), raw_sha256=BLOCK17_RAW_SHA256,
        full_sha256=BLOCK17_FULL_SHA256, fit_recipe_sha256=BLOCK17_FIT_SHA256,
        completed_updates=AUTHOR_BUDGET, development_evidence_sha256=BLOCK17_DEVELOPMENT_SHA256,
        trial_strength=trial_strength, selected_strength=None, ema_decay=None,
    )


def _block17_trial(plan, evidence_path):
    require_fields(plan, _block17_plan(1), "Block 17 development plan")
    strength = require_positive(plan["trial_strength"], "Development trial strength")
    if plan != _block17_plan(strength):
        raise ValueError("Block 17 development replay requires the reviewed fit, identity and no selected strength")
    evidence = decode_json(_read_pin(evidence_path, BLOCK17_DEVELOPMENT_SHA256, "development evidence"))
    fit, ranking = evidence["fit"], evidence["development_ranking"]
    if (evidence["schema"] != "glide-block17-development-evidence-v1" or evidence["held_out_cases"] != 0
            or fit["raw_checkpoint_sha256"] != BLOCK17_RAW_SHA256 or fit["full_checkpoint_sha256"] != BLOCK17_FULL_SHA256
            or fit["full_checkpoint_receipt_sha256"] != BLOCK17_RECEIPT_SHA256 or fit["ema_decay"] is not None
            or fit["completed_updates"] != AUTHOR_BUDGET or ranking["verification_sha256"] != BLOCK17_RANK_SHA256
            or ranking["selected_strength"] is not None or not ranking["identical_adaptive_choices_across_strengths"]
            or strength not in ranking["co_leading_strengths"]):
        raise ValueError("The requested strength is not a verified tied development trial")
    return strength


def write_block17_development_plan(
    plan_path, *, trial_strength, raw_path, full_path, full_receipt_path, development_evidence_path,
):
    """Write a new digest-bound plan for one measured trial, without selecting a preset."""
    strength = require_positive(trial_strength, "Development trial strength")
    plan = _block17_plan(strength)
    _block17_trial(plan, development_evidence_path)
    _load_fitted_state(plan, raw_path, full_path, full_receipt_path)
    data = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    with Path(plan_path).open("xb") as output:
        output.write(data)
    return sha256(data).hexdigest()


def load_replay(
    plan_path, plan_sha256, raw_path, full_path, full_receipt_path, calibration_path=None,
    *, development_evidence_path=None,
):
    plan = decode_json(_read_pin(plan_path, plan_sha256, "replay plan"))
    schema = plan.get("schema") if isinstance(plan, dict) else None
    if schema == BLOCK17_SCHEMA:
        if calibration_path is not None or development_evidence_path is None:
            raise ValueError("Block 17 replay requires development evidence without a selected calibration")
        strength = _block17_trial(plan, development_evidence_path)
    else:
        if development_evidence_path is not None or calibration_path is None:
            raise ValueError("Block 23 replay requires a selected calibration without development evidence")
        require_fields(plan, ("schema", "identity", "raw_sha256", "full_sha256", "fit_recipe_sha256",
                              "completed_updates", "calibration_sha256", "ema_decay"), "replay plan")
        if schema != "glide-recurrent-replay-v1" or plan["identity"] != model_identity() or plan["ema_decay"] is not None:
            raise ValueError("The replay plan belongs to another model, site or weight policy")
        if plan["raw_sha256"] == BLOCK17_RAW_SHA256 or plan["fit_recipe_sha256"] == BLOCK17_FIT_SHA256:
            raise ValueError("The reviewed Block 17 fit requires a development replay plan")
        calibration = decode_json(_read_pin(calibration_path, plan["calibration_sha256"], "calibration"))
        require_fields(calibration, ("schema", "raw_sha256", "selected_strength", "selection_metric"), "calibration")
        if (calibration["schema"] != "glide-recurrent-calibration-v1"
                or calibration["raw_sha256"] != plan["raw_sha256"]
                or calibration["selection_metric"] != "mean_matched_intended_effect_on_minus_off"):
            raise ValueError("Calibration is not bound to this RAW adapter and effect metric")
        strength = require_positive(calibration["selected_strength"], "Selected strength")
    return plan, strength, _load_fitted_state(plan, raw_path, full_path, full_receipt_path)
