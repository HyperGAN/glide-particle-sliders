"""Opt-in authored GLiDE-9B Site23 full-suffix scoring on the publisher runtime."""

import json
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from struct import Struct
from threading import Lock
from types import ModuleType
from zipfile import BadZipFile, ZipFile

import torch
from particle_sliders import prompt_steering

from . import identity, runtime
from .identity import hash_json, require_fields

SITE = 23
VISITS = (30, 36)
ROUTE = runtime.BLOCK17_ROUTE  # Both Site23 and Block 17 use the same native route.
COORDINATE = "glide9b.site23.maximal-common-input-suffix.visits30-36"
SCHEMA = "glide.9b.site23.full-suffix-replay-v1"
REPLAY_FIELDS = (
    "schema",
    "identity",
    "raw_sha256",
    "dose_source_sha256",
    "alignment_sha256",
    "trial_strength",
    "selected_strength",
    "ema_decay",
)
RAW_SHA256 = "a882d1e6eee0b5be8d5f1d3b0ead6f71a6425dae01394ed66bebbde315d4df49"
RAW_BYTES = 35425
FIT_SHA256 = "b53212c10e082c5d0e4ddae9bb0d1f444925127101248d28319f0e7a9008559c"
FULL_RECEIPT_SHA256 = "f1344301449215e8246c2ba64a86ad462b6da9004ddcce01ea0077b46cb164fb"
DOSE_MEMBER = "particle_sliders/strength_schedule.py"
DOSE_SOURCE_SHA256 = "fed34a2d3f70fd793f8b13cb605e80442699912edb3cab3258c1e21f92e50282"
BINARY64 = Struct(">d")
SOURCE_SHA256 = (
    (runtime, "42e38a26ca86d354a47258cc166d43412e080c78c388e2021c63377bb9a63250"),
    (identity, "688cb202855ab218acba45fcc34bfb255acd408f88de3f505908a8d647c4643a"),
)
HEX = frozenset("0123456789abcdef")


def _hex(value, width, label):
    if (
        type(value) is not str
        or len(value) != width
        or any(char not in HEX for char in value)
    ):
        raise ValueError(f"Invalid {label}")
    return value


def _bits(value):
    return BINARY64.pack(value).hex()


def _float_from_bits(value):
    (result,) = BINARY64.unpack(bytes.fromhex(_hex(value, 16, "binary64 bits")))
    if not isfinite(result):
        raise ValueError("Nonfinite coefficient bits")
    return result


def receipt_digest(receipt):
    """Stable content digest for the canonical receipt, independent of JSON key order."""
    canonical = json.dumps(
        receipt,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return sha256(canonical.encode()).hexdigest()


def _readout_digest(readout):
    return sha256(
        readout.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _dose_source(wheel):
    # Leave the installed legacy dose module and its authentication unchanged.
    try:
        with ZipFile(wheel) as archive:
            member = archive.getinfo(DOSE_MEMBER)
            if member.file_size != 7125:
                raise ValueError(
                    "Site23 requires the reviewed shared dose module from PR #338 or #339"
                )
            source = archive.read(member)
    except (BadZipFile, KeyError) as error:
        raise ValueError(
            "Site23 requires a core wheel containing the shared dose module"
        ) from error
    if sha256(source).hexdigest() != DOSE_SOURCE_SHA256:
        raise ValueError(
            "Site23 requires the reviewed shared dose module from PR #338 or #339"
        )
    module = ModuleType(f"{__name__}._verified_dose")
    exec(compile(source, f"{wheel}/{DOSE_MEMBER}", "exec"), module.__dict__)  # noqa: S102
    return module


def _replay(alignment_sha256, trial_strength):
    if (
        type(trial_strength) is not float
        or not isfinite(trial_strength)
        or trial_strength == 0
    ):
        raise ValueError("Require an explicit finite, nonzero binary64 trial strength")
    return {
        "schema": SCHEMA,
        "identity": dict(identity.model_identity(), coordinate=COORDINATE),
        "raw_sha256": RAW_SHA256,
        "dose_source_sha256": DOSE_SOURCE_SHA256,
        "alignment_sha256": _hex(alignment_sha256, 64, "alignment digest"),
        "trial_strength": trial_strength,
        "selected_strength": None,
        "ema_decay": None,
    }


def write_site23_full_suffix_replay(path, *, alignment_sha256, trial_strength):
    """Record the caller's separately registered case alignment and intended trial."""
    plan = _replay(alignment_sha256, trial_strength)
    data = (json.dumps(plan, allow_nan=False, indent=2) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(data)
    return sha256(data).hexdigest()


def _raw_state(raw_file):
    if Path(raw_file).stat().st_size != RAW_BYTES:
        raise ValueError("Site23 RAW80 size differs from the reviewed artifact")
    raw = torch.load(
        BytesIO(identity._read_pin(raw_file, RAW_SHA256, "Site23 RAW80")),
        weights_only=True,
        map_location="cpu",
    )
    require_fields(
        raw,
        ("experiment", "full", "completed", "probe", "raw_weights", "ema_decay"),
        "Site23 RAW80",
    )
    for lineage in ("experiment", "full"):
        require_fields(
            raw[lineage], ("path", "bytes", "sha256"), f"Site23 RAW80 {lineage}"
        )
    if (
        raw["raw_weights"] is not True
        or raw["ema_decay"] is not None
        or type(raw["completed"]) is not int
        or raw["completed"] != 80
        or raw["experiment"]["sha256"] != FIT_SHA256
        or raw["full"]["sha256"] != FULL_RECEIPT_SHA256
    ):
        raise ValueError(
            "Require the reviewed completed Site23 RAW80 with its original FULL and fit lineage"
        )
    parameters = raw["probe"]
    identity._check_parameters(parameters)
    if any(
        tensor.device.type != "cpu" or tensor.layout != torch.strided
        for tensor in parameters.values()
    ):
        raise ValueError("Require portable fitted FP32 Site23 parameters")
    return parameters


class UnrepresentableConstant(ValueError):
    """At least one visit had no position-flat exact binary64 coefficient."""

    def __init__(self, doses):
        self.doses = doses
        failed = ", ".join(str(row["visit"]) for row in doses if row["residual"] != "0")
        super().__init__(f"No exact native FP64 constant at GLiDE visit(s) {failed}")


def _exact_constants(trace_rows, positions, schedule):
    if not isinstance(trace_rows, list) or len(trace_rows) != len(VISITS):
        raise ValueError("Require an adaptive trace for both GLiDE visits")
    constants, doses = [], []
    edges = range(positions + 1)
    for visit, row in zip(VISITS, trace_rows, strict=True):
        require_fields(
            row, ("visit", "post_strength_binary64_bits"), "per-visit adaptive trace"
        )
        bits = row["post_strength_binary64_bits"]
        if (
            row["visit"] != visit
            or not isinstance(bits, list)
            or len(bits) != positions
        ):
            raise ValueError("GLiDE trace visit or full suffix disagrees")
        values = tuple(_float_from_bits(value) for value in bits)
        # The adaptive post-strength product is FP32 before exact widening.
        real_adaptive = schedule.interval_dose(
            values, edges, coefficient_format="binary32"
        )
        match = schedule.match_binary64_interval_dose(values, edges)
        constant = match.constant_strength
        real_constant = schedule.interval_dose(
            (constant,) * positions, edges, coefficient_format="binary64"
        )
        if (real_adaptive, real_constant) != (match.adaptive_dose, match.constant_dose):
            raise RuntimeError("Shared matcher and exact interval dose disagree")
        constants.append(constant)
        doses.append(
            {
                "visit": visit,
                "constant_binary64_bits": _bits(constant),
                "adaptive_dose": str(real_adaptive),
                "constant_dose": str(real_constant),
                "residual": str(real_constant - real_adaptive),
            }
        )
    if any(row["residual"] != "0" for row in doses):
        raise UnrepresentableConstant(doses)
    return tuple(constants), doses


class _Site23Trace:
    def __init__(self, steering, full_length, positions, strength, constants):
        self._steering = steering
        self._shape = (1, full_length, runtime.HIDDEN)
        self._positions = positions
        self._strength = strength
        self._constants = constants
        self._calls = 0
        self.rows = []
        self.changed_positions = []

    def hook(self, block):
        def post_block(_module, _args, output):
            if self._calls >= len(ROUTE) or ROUTE[self._calls] != block:
                raise RuntimeError("GLiDE left the registered 44-call native route")
            self._calls += 1
            if (
                type(output) is not torch.Tensor
                or output.shape != self._shape
                or output.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "GLiDE left the registered unpadded BF16 residual shape"
                )
            if block != SITE:
                return None
            if self._calls not in VISITS:
                raise RuntimeError("Unexpected GLiDE Site23 visit")
            suffix = output[:, -self._positions :, :]
            if self._constants is None:
                coefficients = self._steering.coefficients(suffix)
                if (
                    coefficients.dtype != torch.float32
                    or coefficients.shape != suffix.shape[:2]
                    or not torch.isfinite(coefficients).all().item()
                ):
                    raise RuntimeError(
                        "Expected finite FP32 shared affine-ReLU coefficients"
                    )
                dose = (self._strength * coefficients).to(torch.float64)
            else:
                # The fixed ABI supplies the final scalar directly at the widened seam.
                dose, expected_bits = self._constants[len(self.rows)]
                if dose.device != suffix.device or dose.shape != suffix.shape[:2]:
                    raise RuntimeError(
                        "Preverified FP64 constant has the wrong native shape or device"
                    )
            if dose.dtype != torch.float64:
                raise RuntimeError(
                    "GLiDE position edit requires the widened FP64 coefficient"
                )
            observed = tuple(dose[0].tolist())
            if len(observed) != self._positions or not all(
                isfinite(x) for x in observed
            ):
                raise RuntimeError(
                    "GLiDE consumed a nonfinite or incomplete coefficient trace"
                )
            echo = [_bits(x) for x in observed]
            if self._constants is not None and echo != expected_bits:
                raise RuntimeError(
                    "Native widened constant failed the exact consumed-bits echo"
                )

            # Keep the original AlignedStudentTrace widened direction, add and BF16 return.
            direction = self._steering.direction.weight[0].to(torch.float64)
            changed = suffix.to(torch.float64) + dose[..., None] * direction
            edited = torch.where(dose[..., None] == 0, suffix, changed).to(output.dtype)
            if not torch.isfinite(edited).all().item():
                raise RuntimeError("GLiDE produced a nonfinite edited suffix")
            updated = torch.cat((output[:, : -self._positions, :], edited), dim=1)
            self.rows.append(
                {"visit": self._calls, "post_strength_binary64_bits": echo}
            )
            # Compare actual BF16 bit patterns after the return cast, including signed zero.
            self.changed_positions.append(
                (edited.view(torch.int16) != suffix.view(torch.int16))
                .any(-1)
                .sum()
                .item()
            )
            return updated

        return post_block

    def finish(self):
        if (
            self._calls != len(ROUTE)
            or tuple(row["visit"] for row in self.rows) != VISITS
        ):
            raise RuntimeError(
                "GLiDE did not close both Site23 visits on the 44-call route"
            )
        return self.rows


def _publisher_row(model, request):
    normalized, key, options = runtime._validate_request(model, request)
    entries, _ = model._prepare(normalized)
    prompt = entries[0][3]
    if type(prompt) is not str or not prompt:
        raise ValueError("Require one rendered publisher prompt")
    (ids,) = model.tokenizer([prompt], add_special_tokens=False)["input_ids"]
    if (
        type(ids) is not list
        or not ids
        or len(ids) + 8 > model.max_len
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError("Require one exact publisher token stream")
    return normalized, key, options, prompt, ids


def _align(model, neutral, positive):
    request, key, options, neutral_prompt, neutral_ids = _publisher_row(model, neutral)
    reference, positive_key, positive_options, positive_prompt, positive_ids = (
        _publisher_row(model, positive)
    )
    if (
        key != positive_key
        or options != positive_options
        or neutral_ids == positive_ids
    ):
        raise ValueError(
            "Require identical ordered options in two distinct aligned requests"
        )
    n = 0
    for lhs, rhs in zip(reversed(neutral_ids), reversed(positive_ids)):
        if lhs != rhs:
            break
        n += 1
    if not 1 <= n < min(len(neutral_ids), len(positive_ids)):
        raise ValueError(
            "Require a nonempty full common suffix and distinct unaligned prefixes"
        )
    alignment = {
        "neutral_request": hash_json(request),
        "positive_request": hash_json(reference),
        "neutral_prompt": hash_json(neutral_prompt),
        "positive_prompt": hash_json(positive_prompt),
        "neutral_ids": hash_json(neutral_ids),
        "positive_ids": hash_json(positive_ids),
        "neutral_length": len(neutral_ids),
        "positive_length": len(positive_ids),
        "suffix_length": n,
        "suffix_ids": hash_json(neutral_ids[-n:]),
    }
    return request, key, options, alignment


def site23_alignment_commitment(publisher, neutral, positive):
    """Hash the prepared native pair using the caller's already owned publisher."""
    return hash_json(_align(publisher, neutral, positive)[3])


class FullSuffixChoice:
    """Own a verified publisher model and the exact reviewed Site23 RAW80 probe."""

    def __init__(self):
        raise TypeError(
            "Use FullSuffixChoice.open with a registered replay and verified Site23 RAW80"
        )

    @classmethod
    def _from_verified_components(cls, model, steering, plan, replay_sha256, schedule):
        self = object.__new__(cls)
        self._model = model
        self._steering = steering
        self._alignment_sha256 = plan["alignment_sha256"]
        self._replay_sha256 = replay_sha256
        self._strength = plan["trial_strength"]
        self._schedule = schedule
        self._lock = Lock()
        return self

    @classmethod
    def open(cls, model_dir, *, replay_file, replay_sha256, raw_file, dose_core_wheel):
        """Open local publisher assets under the caller's exclusive CUDA ownership."""
        identity.verify_runtime()
        for module, expected in SOURCE_SHA256:
            if sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Site23 source differs for {module.__name__}")
        plan = identity.decode_json(
            identity._read_pin(replay_file, replay_sha256, "Site23 replay")
        )
        require_fields(plan, REPLAY_FIELDS, "Site23 replay")
        if plan != _replay(plan["alignment_sha256"], plan["trial_strength"]):
            raise ValueError(
                "Site23 replay requires the reviewed fit, registered scope and no selected strength"
            )
        schedule = _dose_source(dose_core_wheel)
        parameters = _raw_state(raw_file)
        root = identity.verify_snapshot(model_dir)
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError(
                "Expected one visible BF16-capable CUDA device for GLiDE-9B"
            )
        model = runtime._load_publisher(root)
        with torch.random.fork_rng(devices=[]):
            steering = prompt_steering.PromptSteering(
                runtime.HIDDEN, device="cpu", dtype=torch.float32
            )
        steering.load_state_dict(parameters, strict=True)
        return cls._from_verified_components(
            model,
            steering.eval().requires_grad_(False).to("cuda:0"),
            plan,
            replay_sha256,
            schedule,
        )

    def _bind(self, neutral, positive):
        request, key, options, alignment = _align(self._model, neutral, positive)
        if hash_json(alignment) != self._alignment_sha256:
            raise ValueError(
                "Publisher tokens or scope differ from the registered role alignment"
            )
        binding = {
            "coordinate": COORDINATE,
            "alignment_sha256": self._alignment_sha256,
            "replay_sha256": self._replay_sha256,
            "raw_sha256": RAW_SHA256,
            "inference_strength_binary64_bits": _bits(self._strength),
            "full_suffix_positions": alignment["suffix_length"],
        }
        return request, key, options, alignment, binding

    def _decide(self, request, key, options, trace=None, *, capture_readout=False):
        readouts = []
        network = self._model.network

        def capture(_module, _args, out):
            if (
                type(out) is not torch.Tensor
                or out.shape != (1, 255)
                or out.dtype != torch.bfloat16
                or not torch.isfinite(out).all().item()
            ):
                raise RuntimeError("Unexpected stock GLiDE BF16 readout")
            readouts.append(out.detach().clone())

        with ExitStack() as hooks:
            if trace is not None:
                for block, layer in enumerate(network.backbone.layers):
                    hooks.callback(
                        layer.register_forward_hook(trace.hook(block)).remove
                    )
            if capture_readout:
                hooks.callback(network.readout.register_forward_hook(capture).remove)
            response = self._model.decide(request["state"], request["questions"])
        runtime._validate_response(response, key, options)
        if capture_readout and len(readouts) != 1:
            raise RuntimeError("Expected one stock GLiDE readout")
        rows = trace.finish() if trace is not None else None
        changed = trace.changed_positions if trace is not None else None
        return response, rows, changed, readouts[0] if capture_readout else None

    def _trace(self, alignment, constants=None):
        return _Site23Trace(
            self._steering,
            alignment["neutral_length"],
            alignment["suffix_length"],
            self._strength,
            constants,
        )

    def _prepare_native_constants(self, positions, constants, expected_doses):
        """Verify the actual device FP64 tensors before any fixed publisher call."""
        device = self._steering.direction.weight.device
        materialized = []
        for visit, constant, expected in zip(
            VISITS, constants, expected_doses, strict=True
        ):
            dose = torch.full(
                (1, positions), constant, dtype=torch.float64, device=device
            )
            observed = tuple(dose[0].tolist())
            bits = [_bits(constant)] * positions
            if (
                dose.dtype != torch.float64
                or [_bits(x) for x in observed] != bits
                or str(
                    self._schedule.interval_dose(
                        observed, range(positions + 1), coefficient_format="binary64"
                    )
                )
                != expected
            ):
                raise RuntimeError(
                    f"Device FP64 constant preflight changed the exact dose at visit {visit}"
                )
            materialized.append((dose, bits))
        return tuple(materialized)

    def zero_parity(self, neutral, positive):
        """Verify stock response and raw BF16 readout with a genuine zero-seam replay."""
        with self._lock, torch.inference_mode():
            request, key, options, alignment, binding = self._bind(neutral, positive)
            stock, _, _, stock_readout = self._decide(
                request, key, options, capture_readout=True
            )
            zeros = self._prepare_native_constants(
                alignment["suffix_length"], (0.0, 0.0), ("0", "0")
            )
            hooked, rows, changed, hooked_readout = self._decide(
                request,
                key,
                options,
                self._trace(alignment, zeros),
                capture_readout=True,
            )
            if (
                stock != hooked
                or any(changed)
                or not torch.equal(
                    stock_readout.view(torch.int16), hooked_readout.view(torch.int16)
                )
            ):
                raise RuntimeError("Zero full-suffix edit differs from stock publisher")
        return {
            "mode": "zero_parity",
            "binding": binding,
            "response": stock,
            "visits": rows,
            "changed_bf16_positions": changed,
            "readout_bitwise_equal": True,
            "stock_readout_bf16_sha256": _readout_digest(stock_readout),
            "zero_readout_bf16_sha256": _readout_digest(hooked_readout),
        }

    def adaptive(self, neutral, positive):
        """Run the exact native adaptive edit and echo the consumed coefficients at both visits."""
        with self._lock, torch.inference_mode():
            request, key, options, alignment, binding = self._bind(neutral, positive)
            response, rows, changed, _ = self._decide(
                request, key, options, self._trace(alignment)
            )
        return {
            "mode": "adaptive",
            "binding": binding,
            "response": response,
            "visits": rows,
            "changed_bf16_positions": changed,
        }

    def exact_constant(
        self,
        neutral,
        positive,
        *,
        adaptive,
        adaptive_sha256,
        zero_parity,
        zero_parity_sha256,
    ):
        """Use per-visit exact constants only after both doses and parity bind to this case."""
        with self._lock, torch.inference_mode():
            request, key, options, alignment, binding = self._bind(neutral, positive)
            fields = ("mode", "binding", "response", "visits", "changed_bf16_positions")
            require_fields(adaptive, fields, "adaptive evidence")
            require_fields(
                zero_parity,
                (
                    *fields,
                    "readout_bitwise_equal",
                    "stock_readout_bf16_sha256",
                    "zero_readout_bf16_sha256",
                ),
                "zero parity evidence",
            )
            if (
                receipt_digest(adaptive) != _hex(adaptive_sha256, 64, "adaptive digest")
                or receipt_digest(zero_parity)
                != _hex(zero_parity_sha256, 64, "zero parity digest")
                or adaptive["mode"] != "adaptive"
                or zero_parity["mode"] != "zero_parity"
                or adaptive["binding"] != binding
                or zero_parity["binding"] != binding
                or zero_parity["readout_bitwise_equal"] is not True
                or _hex(
                    zero_parity["stock_readout_bf16_sha256"], 64, "stock readout digest"
                )
                != zero_parity["zero_readout_bf16_sha256"]
            ):
                raise ValueError(
                    "Require digest-bound adaptive and zero-parity receipts for this exact case"
                )
            positions = alignment["suffix_length"]
            zero_rows = [
                {
                    "visit": visit,
                    "post_strength_binary64_bits": [_bits(0.0)] * positions,
                }
                for visit in VISITS
            ]
            if zero_parity["visits"] != zero_rows or zero_parity[
                "changed_bf16_positions"
            ] != [0] * len(VISITS):
                raise ValueError(
                    "Zero-parity evidence lacks the exact same two zero-seam echoes"
                )
            changes = adaptive["changed_bf16_positions"]
            if (
                not isinstance(changes, list)
                or len(changes) != len(VISITS)
                or any(
                    type(count) is not int or not 0 <= count <= positions
                    for count in changes
                )
            ):
                raise ValueError(
                    "Adaptive receipt has invalid native BF16 change counts"
                )
            runtime._validate_response(adaptive["response"], key, options)
            runtime._validate_response(zero_parity["response"], key, options)
            constants, doses = _exact_constants(
                adaptive["visits"], positions, self._schedule
            )

            device_constants = self._prepare_native_constants(
                positions, constants, (row["adaptive_dose"] for row in doses)
            )
            # Both actual device FP64 tensors must match before the publisher forward.
            response, rows, changed, _ = self._decide(
                request, key, options, self._trace(alignment, device_constants)
            )
            for row, expected in zip(rows, doses, strict=True):
                values = tuple(
                    _float_from_bits(x) for x in row["post_strength_binary64_bits"]
                )
                actual = self._schedule.interval_dose(
                    values, range(positions + 1), coefficient_format="binary64"
                )
                if str(actual) != expected["adaptive_dose"]:
                    raise RuntimeError(
                        f"Consumed native FP64 dose differs at visit {row['visit']}"
                    )
        return {
            "mode": "exact_constant",
            "binding": binding,
            "response": response,
            "visits": rows,
            "changed_bf16_positions": changed,
            "doses": doses,
        }
