"""Apply the shared prompt-steering probe at the two stock GLiDE-9B passes."""

from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib
from math import isfinite
from pathlib import Path
import sys
from threading import Lock

import torch
from particle_sliders.prompt_steering import PromptSteering
from particle_sliders.strength_schedule import match_binary32_interval_dose, match_binary64_interval_dose

from .identity import (
    BLOCK17_COORDINATE, BLOCK17_SCHEMA, hash_json, load_replay, require_fields, require_positive,
    verify_block17_dose_source, verify_runtime, verify_snapshot,
)

BLOCKS = ((12, 17), (18, 23))
PASSES = 2
HIDDEN = 4096
LAYERS = 32
BLOCK17_ROUTE = (*range(18), *range(12, 18), *range(18, 24), *range(18, 24), *range(24, LAYERS))
BLOCK17_VISITS = (18, 24)
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


def _block17_dose(coefficients, match_side):
    if (not isinstance(coefficients, list) or len(coefficients) != len(BLOCK17_VISITS)
            or not all(type(value) is float and isfinite(value) and value >= 0 for value in coefficients)):
        raise ValueError("Expected two finite nonnegative binary32 reference coefficients")
    if match_side not in ("nearest", "lower", "upper"):
        raise ValueError("Expected a nearest, lower or upper binary32 constant")
    match = match_binary32_interval_dose(coefficients, range(len(BLOCK17_VISITS) + 1))
    constant = match.constant_strength
    nearest_residual = match.constant_dose - match.adaptive_dose
    if (match_side == "lower" and nearest_residual > 0) or (match_side == "upper" and nearest_residual < 0):
        limit = -float("inf") if match_side == "lower" else float("inf")
        constant = torch.nextafter(torch.tensor(constant, dtype=torch.float32),
                                   torch.tensor(limit, dtype=torch.float32)).item()
    constant_dose = len(BLOCK17_VISITS) * Fraction(constant)
    residual = constant_dose - match.adaptive_dose
    return constant, dict(adaptive_dose=str(match.adaptive_dose), constant_dose=str(constant_dose),
                          constant_minus_adaptive=str(residual), exact=residual == 0,
                          side="lower" if residual < 0 else "upper" if residual > 0 else "exact",
                          requested_control=match_side)


@contextmanager
def _block17_edit(backbone, steering, strength, constant):
    _check_backbone(backbone)
    route, coefficients = [], []

    def at_block_end(block):
        def hook(_layer, _args, output):
            if len(route) >= len(BLOCK17_ROUTE) or block != BLOCK17_ROUTE[len(route)]:
                raise RuntimeError("The native scorer left the registered 44-call route")
            route.append(block)
            if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16 or output.ndim != 3
                    or output.shape[0] != 1 or not output.shape[1] or output.shape[-1] != HIDDEN):
                raise RuntimeError("Expected one native BF16 text residual")
            if len(route) not in BLOCK17_VISITS:
                return None
            if strength is None and constant is None:
                coefficients.append(0.0)
                return None
            if steering.direction.weight.dtype != torch.float32 or steering.probe.weight.dtype != torch.float32:
                raise ValueError("Block 17 requires the fitted FP32 probe and direction")
            mask = torch.arange(output.shape[1], device=output.device)[None] >= output.shape[1] - 1
            if constant is None:
                gain = steering.coefficients(output)
                if gain.dtype != torch.float32 or gain.shape != output.shape[:2] or not gain.isfinite().all().item():
                    raise ValueError("Expected finite FP32 prompt-steering gains")
                amount = gain * torch.tensor(strength, dtype=torch.float32, device=output.device)
            else:
                amount = torch.full(output.shape[:2], constant, dtype=torch.float32, device=output.device)
            amount = amount.masked_fill(~mask, 0)
            value = amount[0, -1].item()
            if not isfinite(value) or value < 0:
                raise ValueError("PSR coefficient must be finite and nonnegative")
            coefficients.append(value)
            edited = (output.float() + torch.einsum("bs,d->bsd", amount, steering.direction.weight[0])).bfloat16()
            if not edited[:, -1].isfinite().all().item():
                raise ValueError("PSR produced a nonfinite residual")
            return torch.where(mask[..., None], edited, output)
        return hook

    with ExitStack() as hooks:
        for block, layer in enumerate(backbone.layers):
            hooks.callback(layer.register_forward_hook(at_block_end(block)).remove)
        yield coefficients
        if tuple(route) != BLOCK17_ROUTE or len(coefficients) != len(BLOCK17_VISITS):
            raise RuntimeError("The native scorer did not complete the 44-call route and both Block 17 visits")


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
        self._block17 = plan.get("schema") == BLOCK17_SCHEMA
        self._strength = require_positive(strength, "Development trial strength" if self._block17 else "Selected strength")
        self._lock = Lock()

    @classmethod
    def open(cls, model_dir, *, plan_file, plan_sha256, raw_file, full_file, full_receipt_file,
             calibration_file=None, development_evidence_file=None):
        """Open local, verified BF16 publisher assets and the replay-bound RAW probe."""
        verify_runtime()
        plan, strength, parameters = load_replay(
            plan_file, plan_sha256, raw_file, full_file, full_receipt_file, calibration_file,
            development_evidence_path=development_evidence_file,
        )
        if plan["schema"] == BLOCK17_SCHEMA:
            verify_block17_dose_source()
        root = verify_snapshot(model_dir)
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("Expected one visible BF16-capable CUDA device for GLiDE-9B")
        model = _load_publisher(root)
        steering = PromptSteering(HIDDEN, device="cpu", dtype=torch.float32)
        steering.load_state_dict(parameters, strict=True)
        return cls(model, steering.eval().requires_grad_(False).to("cuda:0"), plan, plan_sha256, strength)

    def score(self, request, *, mode, reference=None, match_side=None):
        """Return the stock decision and both resolved recurrent coefficients."""
        with self._lock, torch.inference_mode():
            normalized, key, options = _validate_request(self._model, request)
            binding = dict(request_sha256=hash_json(normalized), replay_sha256=self._plan_sha256,
                           raw_sha256=self._raw_sha256)
            if match_side is not None and (not self._block17 or mode != "matched"):
                raise ValueError("Binary32 control sides are only accepted on a Block 17 matched replay")
            if self._block17:
                binding.update(coordinate=BLOCK17_COORDINATE, visit_clocks=list(BLOCK17_VISITS),
                               trial_strength=self._strength)
            strength, constant = None, None
            dose = None
            if mode == "adaptive" and reference is None:
                strength = self._strength
            elif mode == "matched":
                require_fields(reference, (*binding, "mode", "resolved_coefficients"), "adaptive reference")
                if reference["mode"] != "adaptive" or any(reference[name] != value for name, value in binding.items()):
                    raise ValueError("The adaptive trace belongs to another request or replay")
                if self._block17:
                    constant, dose = _block17_dose(reference["resolved_coefficients"],
                                                   "nearest" if match_side is None else match_side)
                else:
                    constant = _matched_coefficient(reference["resolved_coefficients"])
            elif mode != "off" or reference is not None:
                raise ValueError("The mode requires a valid adaptive, matched or off request")
            intervention = _block17_edit if self._block17 else _recurrent_edit
            with intervention(self._model.network.backbone, self._steering, strength, constant) as coefficients:
                response = self._model.decide(normalized["state"], normalized["questions"])
            _validate_response(response, key, options)
        trace = dict(**binding, mode=mode, resolved_coefficients=coefficients)
        if dose is not None:
            trace["binary32_dose"] = dose
        return dict(response=response, trace=trace)
