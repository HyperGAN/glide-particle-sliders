"""Separate adapter-input normalization cohort; the pretrained readout is native.

Only the shared routed branch's input is centered/scaled using TRAIN statistics.
The fixed statistics are native-only artifact buffers, never EMA or trainables.
GAN mathematics, optimizer, HNDL projection and lifecycle remain shared.
"""
from copy import deepcopy

import torch

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, EvaluationRows, SessionConfig
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.response import ResponseBackend

from . import binding as original
from .binding_residual_v2 import ResidualCritic

NAME = "glide-normalized-readout-particles-v3"
COHORT = "glide-04b-normalized-adapter-pg095-raw-v3"
PROTOCOL = "glide-native-active-two-choice-normalized-adapter-v3"
NORMALIZATION = "train-only-mean-global-centered-coordinate-rms-v1"


def train_statistics(packet):
    ids = [i for i, split in enumerate(packet["splits"]) if split == "train"]
    if len(ids) < 2:
        raise ValueError("normalization requires at least two TRAIN rows")
    hidden = packet["neutral_hidden"][ids].double()
    center = hidden.mean(0)
    rms = (hidden - center).square().mean().sqrt()
    if not torch.isfinite(rms) or rms <= 0:
        raise ValueError("TRAIN representation must have finite nonzero centered variance")
    return center.float(), rms.float()


class NormalizedReadoutSlider(original.ReadoutSlider):
    def __init__(self, readout, *, model_identity, center, rms, seed=7):
        if (not isinstance(center, torch.Tensor) or center.shape != (original.HIDDEN,)
                or center.dtype != torch.float32 or center.requires_grad or not torch.isfinite(center).all()
                or not isinstance(rms, torch.Tensor) or rms.shape != () or rms.dtype != torch.float32
                or rms.requires_grad or not torch.isfinite(rms) or rms <= 0):
            raise ValueError("frozen finite TRAIN center[1024] and positive scalar RMS required")
        super().__init__(readout, model_identity=model_identity, seed=seed)
        try:
            self.network.register_buffer("input_center", center.detach().clone())
            self.network.register_buffer("input_rms", rms.detach().clone())
            self._center_handle = self.adapter.lora_down.register_forward_pre_hook(self._center_input)
        except BaseException:
            super().close()
            raise

    def _center_input(self, module, args):
        del module
        if len(args) != 1 or self.network.input_rms <= 0 or not torch.isfinite(self.network.input_rms):
            raise ValueError("invalid normalized native adapter input")
        return ((args[0] - self.network.input_center) / self.network.input_rms,)

    def close(self):
        super().close()
        if hasattr(self, "_center_handle"):
            self._center_handle.remove()


class NormalizedNativeResponse(original.NativeReadoutResponse):
    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL,
                    adapter_input_normalization=NORMALIZATION,
                    normalization_state="fixed raw network buffers; TRAIN only; pretrained base input unchanged")


def validate(options):
    if options.get("critic", "conditional16") not in ("conditional16", "residual64"):
        raise ValueError("select the declared conditional16 or residual64 critic arm")
    original.validate({k: v for k, v in options.items() if k != "critic"})
    train_statistics(original.load_readout_file(options["cache_file"], cache=True))


def prepare(options):
    packet = original.load_readout_file(options["cache_file"], cache=True)
    center, rms = train_statistics(packet)
    native = NormalizedNativeResponse(NormalizedReadoutSlider(original.readout_from_packet(packet),
        model_identity=packet["model_identity"], center=center, rms=rms, seed=options.get("seed", 7)))
    try:
        banks = {}
        for index, split in enumerate(packet["splits"]):
            banks.setdefault(split, []).append(original.PreparedReadout(
                original.ReadoutRow(packet["neutral_hidden"][index:index+1], packet["row_ids"][index]),
                packet["teacher_logits"][index:index+1], packet["neutral_logits"][index:index+1]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get("seed", 7) + 1)
            critic = ResidualCritic() if options.get("critic") == "residual64" else original.ReadoutCritic()
        rates = {key: options[key] for key in ("lr", "d_lr_mult", "prior_lr_mult") if options.get(key) is not None}
        game = ConditionalPG095(native, banks["train"], critic,
            conditions=torch.cat([p.row.hidden for p in banks["train"]]), particles=native.particles,
            response_domain="continuous_vector", batch_size=options.get("batch", 8),
            total_steps=options.get("steps", 512), seed=options.get("seed", 7),
            recipe_overrides=rates, **engine_options(options))
        evaluations = {name: EvaluationRows(rows, dict(schema=original.CACHE_SCHEMA,
            cache_sha256=original._sha(options["cache_file"]), split=name))
            for name, rows in banks.items() if name != "train"}
        specification = INFERENCE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=dict(seed=options.get("seed", 7), normalization=NORMALIZATION))
        return Binding(game, native.close, inference=specification, evaluation_rows=evaluations)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if set(options) - {"readout_file", "device"} or options.get("device", "cpu") != "cpu":
        raise ValueError("normalized GLiDE inference requires local CPU readout_file")
    if set(spec.configuration) != {"seed", "normalization"} or spec.configuration["normalization"] != NORMALIZATION:
        raise ValueError("normalization construction protocol differs")
    packet = original.load_readout_file(options.get("readout_file"))
    if original._readout_identity(packet["model_identity"], packet["readout_weight"]) != spec.identity["model"]:
        raise ValueError("native readout model/source identity differs")
    original._runtime()


def prepare_inference(options, spec):
    packet = original.load_readout_file(options["readout_file"])
    # Fixed transform values are subsequently loaded from the RAW artifact's
    # network buffers by the shared inference lifecycle. No training cache read.
    native = NormalizedNativeResponse(NormalizedReadoutSlider(original.readout_from_packet(packet),
        model_identity=packet["model_identity"], center=torch.zeros(original.HIDDEN), rms=torch.ones(()),
        seed=spec.configuration["seed"]))
    try:
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        def response(*, hidden, strength=1., seed=0):
            del seed
            return native.slider.response(hidden, strength=strength).detach().clone()
        return InferenceBinding(native.network, identity, native.close, response=response)
    except BaseException:
        native.close()
        raise


BACKEND = ResponseBackend(NAME, original.capabilities(), prepare, validate,
                          configure_capabilities=original.capabilities)
INFERENCE = InferenceBackend(NAME, original.capabilities(), prepare_inference, validate_inference,
                             configure_capabilities=original.capabilities)


def config(**options):
    baseline = original.config(**{k: v for k, v in options.items() if k != "critic"})
    return SessionConfig(NAME, COHORT, **baseline.options, critic=options.get("critic", "conditional16"))
