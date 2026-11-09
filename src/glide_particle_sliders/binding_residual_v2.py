"""Separate response-only residual critic cohort over the shared PG0.9.5 game.

The native routed readout and actual particle bank are unchanged. The shared
game presents Gaussian reference noise versus Gaussian noise plus the paired
student/teacher residual. This critic therefore sees the two native residual
coordinates only; it deliberately ignores encoder conditioning. All GAN losses,
penalties, optimizer rules, training, continuation and inference remain shared.
"""
from copy import deepcopy
import hashlib

import torch
from torch import nn

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, EvaluationRows, SessionConfig
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.response import ResponseBackend

from . import binding as base

NAME = "glide-residual-critic-particles-v2"
COHORT = "glide-04b-readout-pg095-residual-critic-raw-v2"
PROTOCOL = "glide-native-active-two-choice-readout-residual-critic-v2"
CRITIC_GRAPH = '''features = linear(response, 64, name="hidden0")
features = silu(features)
features = linear(features, 64, name="hidden1")
features = silu(features)
linear(features, 1, name="score")
'''


class ResidualCritic(nn.Module):
    """Shared HNDL response2/64/64/1 graph with no condition input."""
    score_bound = 0

    def __init__(self):
        super().__init__()
        builder, self.identity = base._runtime()
        self.graph = builder(CRITIC_GRAPH,
            input_shape={"response": ["B", base.ACTIVE]}, output_shape=["B", 1])

    def extra_repr(self):
        import json
        return json.dumps(dict(self.identity, condition_usage="ignored",
            graph_sha256=hashlib.sha256(CRITIC_GRAPH.encode()).hexdigest()), sort_keys=True)

    def forward(self, response, condition):
        del condition
        return self.graph(response=response).squeeze(-1)


class NativeResidualResponse(base.NativeReadoutResponse):
    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL,
                    critic_response="Gaussian reference plus native two-choice paired residual",
                    critic_conditioning="none; response-only64/64 HNDL critic")


def capabilities(options=None):
    cap = base.capabilities(options)
    from dataclasses import replace
    return replace(cap, qualification="Separate response-only residual critic readout experiment; native and semantic quality require measurement")


def validate(options):
    base.validate(options)


def _native(packet, options):
    model = base.readout_from_packet(packet, device=options.get("device", "cpu"))
    slider = base.ReadoutSlider(model, model_identity=packet["model_identity"], seed=options.get("seed", 7))
    return NativeResidualResponse(slider)


def prepare(options):
    packet = base.load_readout_file(options["cache_file"], cache=True)
    native = _native(packet, options)
    try:
        banks = {}
        for index, split in enumerate(packet["splits"]):
            banks.setdefault(split, []).append(base.PreparedReadout(
                base.ReadoutRow(packet["neutral_hidden"][index:index+1], packet["row_ids"][index]),
                packet["teacher_logits"][index:index+1], packet["neutral_logits"][index:index+1]))
        rows = banks["train"]
        conditions = torch.cat([row.row.hidden for row in rows]).detach().clone()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(options.get("seed", 7) + 1)
            critic = ResidualCritic()
        rates = {key: options[key] for key in ("lr", "d_lr_mult", "prior_lr_mult") if options.get(key) is not None}
        game = ConditionalPG095(native, rows, critic, conditions=conditions, particles=native.particles,
            response_domain="continuous_vector", batch_size=options.get("batch", 2),
            total_steps=options.get("steps", 4), seed=options.get("seed", 7), recipe_overrides=rates,
            **engine_options(options))
        evaluation = {name: EvaluationRows(values, dict(schema=base.CACHE_SCHEMA,
                      cache_sha256=base._sha(options["cache_file"]), split=name))
                      for name, values in banks.items() if name != "train"}
        specification = INFERENCE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=dict(seed=options.get("seed", 7)))
        return Binding(game, native.close, inference=specification, evaluation_rows=evaluation)
    except BaseException as primary:
        base._close_preserving(native, primary)
        raise


def validate_inference(options, specification):
    base.validate_inference(options, specification)


def prepare_inference(options, specification):
    packet = base.load_readout_file(options["readout_file"])
    native = _native(packet, dict(options, **specification.configuration))
    try:
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        def response(*, hidden, strength=1., seed=0):
            del seed
            return native.slider.response(hidden, strength=strength).detach().clone()
        return InferenceBinding(native.network, identity, native.close, response=response)
    except BaseException as primary:
        base._close_preserving(native, primary)
        raise


BACKEND = ResponseBackend(NAME, capabilities(), prepare, validate, configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, capabilities(), prepare_inference, validate_inference,
                             configure_capabilities=capabilities)


def config(**options):
    return SessionConfig(NAME, COHORT, **deepcopy(base.config(**options).options))
