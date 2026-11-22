"""Apply the shared prompt-steering probe at the two stock GLiDE-9B passes."""

from contextlib import contextmanager
import importlib
from math import isfinite
from pathlib import Path
import sys
from threading import Lock

import torch
from particle_sliders.prompt_steering import PromptSteering
from particle_sliders.strength_schedule import match_binary64_interval_dose

from .identity import hash_json, load_replay, require_fields, require_positive, verify_runtime, verify_snapshot

BLOCKS = ((12, 17), (18, 23))
PASSES = 2
HIDDEN = 4096
LAYERS = 32
_LOAD_LOCK = Lock()


def _check_backbone(backbone):
    recurrence = backbone.recurrence
    if (tuple(map(tuple, recurrence["blocks"])) != BLOCKS or recurrence["passes"] != PASSES
            or recurrence["strategy"] != "euler" or recurrence["conv"] != "A"
            or recurrence["gamma"] != 1 or recurrence["n_gate"] or recurrence["n_lora"]
            or backbone.config.hidden_size != HIDDEN or backbone.config.num_hidden_layers != LAYERS
            or len(backbone.layers) != LAYERS):
        raise ValueError("Unsupported GLiDE-9B recurrence")


def _matched_coefficient(coefficients):
    if (not isinstance(coefficients, list) or len(coefficients) != PASSES
            or not all(type(value) is float and isfinite(value) and value >= 0 for value in coefficients)):
        raise ValueError("Expected two finite nonnegative reference coefficients")
    match = match_binary64_interval_dose(coefficients, range(PASSES + 1))
    if match.adaptive_dose != match.constant_dose:
        raise ValueError("The two-pass dose has no exact binary64 constant")
    return match.constant_strength


@contextmanager
def _recurrent_edit(backbone, steering, strength, constant):
    _check_backbone(backbone)
    coefficients = []

    def at_block_end(_layer, _args, output):
        if (len(coefficients) >= PASSES or not isinstance(output, torch.Tensor)
                or output.ndim != 3 or output.shape[0] != 1
                or output.shape[1] == 0 or output.shape[-1] != HIDDEN):
            raise RuntimeError("Expected one text scoring row with two recurrent passes")
        if strength is None and constant is None:
            coefficients.append(0.0)
            return None
        residual = output[:, -1]
        if constant is None:
            coefficient = (strength * steering.coefficients(residual)).to(torch.float64)
        else:
            coefficient = residual.new_full((1,), constant, dtype=torch.float64)
        value = float(coefficient.item())
        if not isfinite(value) or value < 0:
            raise ValueError("PSR coefficient must be finite and nonnegative")
        coefficients.append(value)
        if value == 0:
            return None
        direction = steering.direction.weight[0].to(torch.float64)
        edited = (residual.to(torch.float64) + coefficient[:, None] * direction).to(output.dtype)
        if not torch.isfinite(edited).all().item():
            raise ValueError("PSR produced a nonfinite residual")
        return torch.cat((output[:, :-1], edited[:, None]), dim=1)

    handle = backbone.layers[BLOCKS[-1][1]].register_forward_hook(at_block_end)
    try:
        yield coefficients
        if len(coefficients) != PASSES:
            raise RuntimeError("The native scorer did not complete both recurrent passes")
    finally:
        handle.remove()


def _validate_request(model, request):
    if (not isinstance(request, dict) or set(request) - {"state", "questions", "images"}
            or request.get("images")):
        raise ValueError("Expected one native text-only decision request")
    questions = request.get("questions")
    if not isinstance(questions, dict) or len(questions) != 1:
        raise ValueError("Expected one choice question")
    question_key, question = next(iter(questions.items()))
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise ValueError("Expected one native choice question")
    normalized = dict(state=request.get("state"), questions=questions)
    entries, media = model._prepare(normalized)
    if media or len(entries) != 1:
        raise ValueError("Expected a single prepared text-only choice")
    key, question_type, options, _ = entries[0]
    if key != question_key or question_type != "choice" or len(options) != 2 or len(set(options)) != 2:
        raise ValueError("Expected two distinct native choice options")
    return normalized, question_key, options


def _validate_response(response, question_key, options):
    answer = response["answers"]
    if set(answer) != {question_key} or response["usage"]["output_tokens"] != 0:
        raise RuntimeError("Unexpected native decision response")
    choice = answer[question_key]
    probabilities = choice["probabilities"]
    if (choice["choice"] not in options or set(probabilities) != set(options)
            or not all(isfinite(value) and 0 <= value <= 1 for value in probabilities.values())):
        raise RuntimeError("Native decision fell outside the two-option contract")


def _load_publisher(root):
    with _LOAD_LOCK:
        for name, module in tuple(sys.modules.items()):
            if name != "glide" and not name.startswith("glide."):
                continue
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(root / "glide"):
                raise ValueError("Another GLiDE runtime is already imported")
        sys.path.insert(0, str(root))
        try:
            upstream = importlib.import_module("glide")
            recurrent = importlib.import_module("glide.recurrent")
            if Path(upstream.__file__).resolve() != root / "glide/__init__.py":
                raise ValueError("GLiDE import differs from the verified publisher runtime")
            if Path(recurrent.__file__).resolve() != root / "glide/recurrent.py":
                raise ValueError("Recurrent module differs from the verified publisher runtime")
            model = upstream.Glide.from_pretrained(str(root), device="cuda:0", dtype=torch.bfloat16, vision=False)
        finally:
            sys.path.remove(str(root))
    if (type(model.network.backbone) is not recurrent.RecurrentTextModel or model.network.visual is not None
            or model.temperature != 1 or len(model.codes) != 255 or model.quantization is not None
            or model.network.training or any(parameter.requires_grad or parameter.device != torch.device("cuda:0")
                                               or parameter.dtype != torch.bfloat16
                                               for parameter in model.network.parameters())):
        raise ValueError("The loaded publisher runtime differs from the fitted text-only BF16 model")
    _check_backbone(model.network.backbone)
    return model


class RecurrentChoice:
    """An installed opt-in scorer bound to an immutable replay plan and completed fit."""

    def __init__(self, model, steering, plan, plan_sha256, strength):
        self._model = model
        self._steering = steering
        self._raw_sha256 = plan["raw_sha256"]
        self._plan_sha256 = plan_sha256
        self._strength = require_positive(strength, "Selected strength")
        self._lock = Lock()

    @classmethod
    def open(cls, model_dir, *, plan_file, plan_sha256, raw_file, full_file, full_receipt_file, calibration_file):
        """Open local, verified BF16 publisher assets and the replay-bound RAW probe."""
        verify_runtime()
        plan, strength, parameters = load_replay(
            plan_file, plan_sha256, raw_file, full_file, full_receipt_file, calibration_file,
        )
        root = verify_snapshot(model_dir)
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("Expected one visible BF16-capable CUDA device for GLiDE-9B")
        model = _load_publisher(root)
        steering = PromptSteering(HIDDEN, device="cpu", dtype=torch.float32)
        steering.load_state_dict(parameters, strict=True)
        return cls(model, steering.eval().requires_grad_(False).to("cuda:0"), plan, plan_sha256, strength)

    def score(self, request, *, mode, reference=None):
        """Return the stock decision and both resolved recurrent coefficients."""
        with self._lock, torch.inference_mode():
            normalized, key, options = _validate_request(self._model, request)
            binding = dict(request_sha256=hash_json(normalized), replay_sha256=self._plan_sha256,
                           raw_sha256=self._raw_sha256)
            strength, constant = None, None
            if mode == "adaptive" and reference is None:
                strength = self._strength
            elif mode == "matched":
                require_fields(reference, (*binding, "mode", "resolved_coefficients"), "adaptive reference")
                if reference["mode"] != "adaptive" or any(reference[name] != value for name, value in binding.items()):
                    raise ValueError("The adaptive trace belongs to another request or replay")
                constant = _matched_coefficient(reference["resolved_coefficients"])
            elif mode != "off" or reference is not None:
                raise ValueError("The mode requires a valid adaptive, matched or off request")
            with _recurrent_edit(self._model.network.backbone, self._steering, strength, constant) as coefficients:
                response = self._model.decide(normalized["state"], normalized["questions"])
            _validate_response(response, key, options)
        return dict(response=response, trace=dict(**binding, mode=mode, resolved_coefficients=coefficients))
