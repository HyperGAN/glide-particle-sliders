"""GLiDE native readout binding; GAN updates and lifecycle remain shared.

Only the model's existing ``Linear(1024, 255, bias=False)`` readout is adapted.
The first two outputs are the model's active native two-choice logits, not a
projection of encoder coordinates. Caching the frozen encoder is exact for this
readout-only arm. This module never loads an encoder or downloads model weights.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib.metadata import distribution, version
import math
from pathlib import Path
import re

import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, EvaluationRows, SessionConfig
from particle_sliders.pg09_backends import AdapterParameters
from particle_sliders.pg095 import ConditionalPG095, PG_VERSION, PG_SOURCE_PROFILE, engine_options
from particle_sliders.response import ResponseBackend, ResponseCapabilities

NAME = "glide-readout-particles"
COHORT = "glide-04b-readout-pg095-raw-v1"
PROTOCOL = "glide-native-active-two-choice-readout-v1"
CACHE_SCHEMA = "glide-readout-cache-v1"
MODEL_SCHEMA = "glide-readout-model-v1"
HIDDEN = 1024
OUTPUTS = 255
ACTIVE = 2
MODEL_ID = "fastino/GLiDE-0.4B"
CORE_COMMIT = "3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"
CRITIC_GRAPH = '''joined = concat(response, condition, axis=1)
features = linear(joined, 16, name="hidden")
features = silu(features)
linear(features, 1, name="score")
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _readout_identity(identity, weight):
    if (not isinstance(identity, dict) or identity.get("model_id") != MODEL_ID
            or re.fullmatch("[0-9a-f]{40}", str(identity.get("revision", ""))) is None):
        raise ValueError("explicit fastino/GLiDE-0.4B model_id and immutable40hex revision required")
    identity = deepcopy(identity)
    digest = hashlib.sha256(weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if "readout_weight_sha256" in identity and identity["readout_weight_sha256"] != digest:
        raise ValueError("native readout identity contradicts its actual weight bytes")
    identity["readout_weight_sha256"] = digest
    return identity


def _runtime():
    if version("particle-sliders-core") != "0.13.0":
        raise ValueError("GLiDE requires its pinned particle-sliders-core 0.13.0")
    if (PG_VERSION, PG_SOURCE_PROFILE) != ("0.9.5", "release-0.9.5"):
        raise ValueError("GLiDE requires official ParticleGAN 0.9.5")
    import json
    provenance = json.loads(distribution("particle-sliders-core").read_text("direct_url.json") or "{}")
    if (provenance.get("url") != "https://github.com/HyperGAN/particle-sliders.git"
            or provenance.get("subdirectory") != "packages/particle-sliders-core"
            or provenance.get("vcs_info", {}).get("vcs") != "git"
            or provenance.get("vcs_info", {}).get("commit_id") != CORE_COMMIT
            or provenance.get("dir_info", {}).get("editable", False)):
        raise ValueError("GLiDE requires the exact immutable particle-sliders Git source pin")
    return verified_builder()


def _tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != torch.float32 or value.requires_grad
            or not torch.isfinite(value).all()):
        raise ValueError(name + " must be a frozen finite float32 tensor with shape " + str(tuple(shape)))
    return value.detach().cpu().contiguous().clone()


def load_readout_file(path, *, cache=False):
    """Read the explicit tensor-only cache/model contract; no arbitrary pickle."""
    if not isinstance(path, (str, Path)) or not Path(path).is_file():
        raise ValueError("an existing explicit local readout/cache file is required")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    schema = CACHE_SCHEMA if cache else MODEL_SCHEMA
    if not isinstance(packet, dict) or packet.get("schema") != schema:
        raise ValueError("unsupported GLiDE readout/cache schema")
    identity = packet.get("model_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("explicit native model/source identity is required")
    # Identity remains JSON-like; the shared lifecycle hashes it independently.
    import json
    json.dumps(identity, allow_nan=False, sort_keys=True)
    weight = _tensor(packet.get("readout_weight"), (OUTPUTS, HIDDEN), "readout_weight")
    _readout_identity(identity, weight)
    if packet.get("readout_bias") is not None:
        raise ValueError("native GLiDE readout is bias-free")
    result = dict(packet, model_identity=deepcopy(identity), readout_weight=weight,
                  readout_bias=None)
    if not cache:
        return result
    ids = packet.get("row_ids")
    if (not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("cache requires at least two uniquely identified rows")
    count = len(ids)
    for key, width in (("neutral_hidden", HIDDEN), ("teacher_logits", ACTIVE), ("neutral_logits", ACTIVE)):
        result[key] = _tensor(packet.get(key), (count, width), key)
    splits = packet.get("splits", ["train"] * count)
    if (not isinstance(splits, list) or len(splits) != count
            or any(not isinstance(s, str) or not s for s in splits)):
        raise ValueError("cache splits must name every row")
    result["splits"] = list(splits)
    with torch.no_grad():
        actual = torch.nn.functional.linear(result["neutral_hidden"], weight)[:, :ACTIVE]
    if not torch.allclose(actual, result["neutral_logits"], atol=1e-5, rtol=1e-5):
        raise ValueError("cached neutral logits disagree with the native readout")
    return result


def readout_from_packet(packet, *, device="cpu"):
    if device != "cpu":
        raise ValueError("GLiDE readout cohort is CPU-only; GPU requires separate scheduler admission")
    model = nn.Linear(HIDDEN, OUTPUTS, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(packet["readout_weight"].to(device))
    return model.eval().requires_grad_(False)


class ReadoutSlider:
    """Own one shared routed bank and a reversible hook on the native readout."""
    def __init__(self, readout, *, model_identity, seed=7):
        if (not isinstance(readout, nn.Linear) or readout.in_features != HIDDEN
                or readout.out_features != OUTPUTS or readout.bias is not None
                or readout.weight.dtype != torch.float32 or readout.weight.device.type != "cpu"):
            raise ValueError("expected the native float32 bias-free GLiDE1024-to-255 readout")
        if (getattr(readout.forward, "__func__", None) is not nn.Linear.forward):
            raise ValueError("native readout is already wrapped")
        if not isinstance(model_identity, dict) or not model_identity:
            raise ValueError("explicit model identity required")
        self.model = readout
        self.model_identity = _readout_identity(model_identity, readout.weight)
        self._forward = readout.forward
        self._mode = readout.training
        self._requires_grad = readout.weight.requires_grad
        self.closed = False
        builder, identity = _runtime()
        devices = [readout.weight.device.index or 0] if readout.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            self.particles = nn.Parameter(torch.randn(128, 4, device=readout.weight.device))
            self.adapter = HNDLParticleProjection("readout", readout, self.particles, 8, 8,
                                                 builder=builder, identity=identity, attach=False)
            self.adapter.to(device=readout.weight.device, dtype=torch.float32)
        self.network = AdapterParameters(
            [("particles", self.particles)] + [("readout." + name, p) for name, p in self.adapter.named_parameters()],
            self.scaled)
        readout.eval().requires_grad_(False)
        readout.forward = self.adapter.forward

    @contextmanager
    def scaled(self, strength):
        if self.closed:
            raise RuntimeError("GLiDE readout slider is closed")
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("finite slider strength required")
        previous = self.adapter.multiplier
        try:
            self.adapter.multiplier = strength
            yield self
        finally:
            self.adapter.multiplier = previous

    def response(self, hidden, *, strength=1.):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[1] != HIDDEN
                or not torch.isfinite(hidden).all()):
            raise ValueError("finite native pooled hidden [B,1024] required")
        with self.network.scaled(strength):
            return self.model(hidden.to(self.model.weight))[:, :ACTIVE]

    def close(self):
        if self.closed:
            return
        self.adapter.multiplier = 0.
        if self.model.forward == self.adapter.forward:
            self.model.forward = self._forward
        elif self.model.forward != self._forward:
            raise RuntimeError("native readout forward ownership changed; foreign hook retained")
        self.model.training = self._mode
        self.model.weight.requires_grad_(self._requires_grad)
        self.closed = True


@dataclass(frozen=True)
class ReadoutRow:
    hidden: torch.Tensor
    row_id: str
    scale: float = 1.


@dataclass(frozen=True)
class PreparedReadout:
    row: ReadoutRow
    target: torch.Tensor
    neutral: torch.Tensor


class NativeReadoutResponse:
    def __init__(self, slider):
        self.slider = slider
        self.model = slider.model
        self.network = slider.network
        self.particles = slider.particles
        self.model_identity = slider.model_identity

    def generated(self, prepared):
        return self.slider.response(prepared.row.hidden, strength=prepared.row.scale)

    def metadata(self):
        return dict(protocol=PROTOCOL, model_identity=self.model_identity,
                    response_domain="continuous_vector", response_shape=[1, ACTIVE],
                    native_coordinates="first two active native choice logits",
                    adapter_target="native bias-free readout1024x255", frozen_encoder=True,
                    generator_arm="routed_particle_gan", trained_scales=[1.],
                    quality_qualified=False)

    def close(self):
        self.slider.close()


class ReadoutCritic(nn.Module):
    """The small conditional critic executes through the shared HNDL builder."""
    score_bound = 0

    def __init__(self):
        super().__init__()
        builder, self.identity = _runtime()
        self.graph = builder(CRITIC_GRAPH,
            input_shape={"response": ["B", ACTIVE], "condition": ["B", HIDDEN]},
            output_shape=["B", 1])

    def extra_repr(self):
        import json
        return json.dumps(dict(self.identity, graph_sha256=hashlib.sha256(CRITIC_GRAPH.encode()).hexdigest()), sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)


def capabilities(options=None):
    return ResponseCapabilities("routed_particle_gan", "text_classification", "native_active_choice_logits",
        "continuous_vector", (1, ACTIVE), PG_VERSION, PG_SOURCE_PROFILE,
        penalty_input_domain="continuous", strength_min=None, strength_max=None,
        base_precisions=("float32",),
        qualification="Readout-only native two-choice logit experiment; semantic preference quality requires held-out measurement")


def validate(options):
    allowed = {"cache_file", "split", "steps", "batch", "seed", "lr", "d_lr_mult", "prior_lr_mult",
               "ema_decay", "pg_version", "pg_source_profile", "penalty_input_domain", "base_precision", "device"}
    if set(options) - allowed:
        raise ValueError("unknown GLiDE session option: " + str(sorted(set(options) - allowed)))
    if options.get("ema_decay") is not None:
        raise ValueError("GLiDE particle sliders require no EMA; use ema_decay=None")
    if options.get("base_precision", "float32") != "float32":
        raise ValueError("native readout cohort uses float32")
    if options.get("device", "cpu") != "cpu":
        raise ValueError("GLiDE readout cohort is CPU-only; GPU requires separate scheduler admission")
    if engine_options(options) != dict(pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain="continuous"):
        raise ValueError("GLiDE requires its explicit continuous PG0.9.5 profile")
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        value = options.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(key + " must be positive and finite")
    split = options.get("split", "train")
    if split != "train":
        raise ValueError("this GLiDE product trains only on the train split")
    packet = load_readout_file(options.get("cache_file"), cache=True)
    if packet["splits"].count(split) < 2:
        raise ValueError("training split needs at least two rows")
    _runtime()


def _native(packet, options):
    model = readout_from_packet(packet, device=options.get("device", "cpu"))
    return NativeReadoutResponse(ReadoutSlider(model, model_identity=packet["model_identity"], seed=options.get("seed", 7)))


def _close_preserving(native, primary):
    try:
        native.close()
    except BaseException as cleanup_error:
        primary.add_note("GLiDE cleanup also failed; retry through exception.cleanup_owner")
        primary.cleanup_owner = native
        primary.cleanup_errors = (cleanup_error,)


def prepare(options):
    packet = load_readout_file(options["cache_file"], cache=True)
    native = _native(packet, options)
    try:
        device = native.model.weight.device
        banks = {}
        for index, split in enumerate(packet["splits"]):
            banks.setdefault(split, []).append(PreparedReadout(
                ReadoutRow(packet["neutral_hidden"][index:index+1].to(device), packet["row_ids"][index]),
                packet["teacher_logits"][index:index+1].to(device),
                packet["neutral_logits"][index:index+1].to(device)))
        split = options.get("split", "train")
        rows = banks[split]
        conditions = torch.cat([p.row.hidden for p in rows]).detach().clone()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(options.get("seed", 7) + 1)
            critic = ReadoutCritic().to(device)
        rates = {key: options[key] for key in ("lr", "d_lr_mult", "prior_lr_mult") if options.get(key) is not None}
        game = ConditionalPG095(native, rows, critic, conditions=conditions, particles=native.particles,
            response_domain="continuous_vector", batch_size=options.get("batch", 2),
            total_steps=options.get("steps", 4), seed=options.get("seed", 7), recipe_overrides=rates,
            **engine_options(options))
        evaluation = {name: EvaluationRows(values, dict(schema=CACHE_SCHEMA, cache_sha256=_sha(options["cache_file"]), split=name))
                      for name, values in banks.items() if name != split}
        specification = INFERENCE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=dict(seed=options.get("seed", 7)))
        return Binding(game, native.close, inference=specification, evaluation_rows=evaluation)
    except BaseException as primary:
        _close_preserving(native, primary)
        raise


def validate_inference(options, specification):
    if set(options) - {"readout_file", "device"} or "readout_file" not in options:
        raise ValueError("inference requires only the local readout_file and optional device")
    if set(specification.configuration) != {"seed"}:
        raise ValueError("unsupported GLiDE inference construction contract")
    if options.get("device", "cpu") != "cpu":
        raise ValueError("GLiDE readout cohort is CPU-only; GPU requires separate scheduler admission")
    packet = load_readout_file(options["readout_file"])
    if _readout_identity(packet["model_identity"], packet["readout_weight"]) != specification.identity["model"]:
        raise ValueError("native readout model/source identity differs")
    _runtime()


def prepare_inference(options, specification):
    packet = load_readout_file(options["readout_file"])
    native = _native(packet, dict(options, **specification.configuration))
    try:
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        def response(*, hidden, strength=1., seed=0):
            del seed
            return native.slider.response(hidden, strength=strength).detach().clone()
        return InferenceBinding(native.network, identity, native.close, response=response)
    except BaseException as primary:
        _close_preserving(native, primary)
        raise


BACKEND = ResponseBackend(NAME, capabilities(), prepare, validate, configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, capabilities(), prepare_inference, validate_inference,
                             configure_capabilities=capabilities)


def config(**options):
    options.setdefault("ema_decay", None)
    options.setdefault("base_precision", "float32")
    options.setdefault("pg_source_profile", PG_SOURCE_PROFILE)
    options.setdefault("penalty_input_domain", "continuous")
    return SessionConfig(NAME, COHORT, **options)
