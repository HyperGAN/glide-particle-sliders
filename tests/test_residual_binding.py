"""Actual shared lifecycle checks for the separate response-only critic cohort."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from glide_particle_sliders import binding as original
from glide_particle_sliders import binding_residual_v2 as product
from particle_sliders.inference import open_inference


@pytest.fixture
def inputs(tmp_path):
    rng = torch.Generator().manual_seed(31)
    weight = torch.randn(255, 1024, generator=rng) * .02
    hidden = torch.randn(6, 1024, generator=rng)
    neutral = torch.nn.functional.linear(hidden, weight)[:, :2]
    teacher = neutral + torch.tensor([[.1, -.1], [-.1, .1]] * 3)
    identity = dict(model_id=original.MODEL_ID, revision="1" * 40,
                    qualification="synthetic native-shaped fixture; no pretrained execution")
    cache = dict(schema=original.CACHE_SCHEMA, model_identity=identity, readout_weight=weight,
        readout_bias=None, neutral_hidden=hidden, neutral_logits=neutral, teacher_logits=teacher,
        row_ids=[str(i) for i in range(6)], splits=["train"] * 4 + ["dev"] * 2)
    cache_path = tmp_path / "cache.pt"
    model_path = tmp_path / "readout.pt"
    torch.save(cache, cache_path)
    torch.save(dict(schema=original.MODEL_SCHEMA, model_identity=identity,
                    readout_weight=weight, readout_bias=None), model_path)
    return cache_path, model_path, cache


def test_response_only_critic_uses_no_condition_coordinates():
    critic = product.ResidualCritic()
    response = torch.randn(4, 2, requires_grad=True)
    a = critic(response, torch.randn(4, 1024))
    b = critic(response, torch.randn(4, 1024) * 1000)
    assert torch.equal(a, b)
    assert critic.graph["hidden0"].weight.shape == (64, 2)
    assert critic.graph["hidden1"].weight.shape == (64, 64)
    gradient = torch.autograd.grad(a.sum(), response)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient).item() > 0


def test_new_cohort_exact_history_resume_and_raw_native_only_fresh_process(inputs):
    cache_path, model_path, cache = inputs
    cfg = product.config(cache_file=str(cache_path), steps=20, batch=2, seed=7,
                         lr=.001, d_lr_mult=18., prior_lr_mult=30., ema_decay=None)
    with product.BACKEND.open(cfg) as direct:
        assert direct.ema.decay is None and direct.ema.values is None
        assert direct.game.prior.z is direct.game.backend.particles
        assert direct.game._signature["generator_arm"] == "routed_particle_gan"
        direct.train(stop_after=16)
        history = direct.checkpoint()
        direct.step()
        expected = direct.checkpoint()
        expected_response = direct.evaluate(index=0, split="dev", strength=.5)
        artifact_path = cache_path.parent / "raw-inference.pt"
        artifact = direct.export_inference(artifact_path, selection="raw")
        assert artifact["origin"]["completed_steps"] == 17
        assert artifact["selection"] == "raw"
        assert torch.count_nonzero(direct.game.backend.slider.adapter.lora_up.weight[2:]) == 0
    with product.BACKEND.open(cfg) as resumed:
        resumed.restore(history)
        resumed.step()
        assert resumed.checkpoint()["sha256"] == expected["sha256"]
    cache_path.unlink()
    with open_inference(artifact, backend=product.INFERENCE, operation="response",
                        options=dict(readout_file=str(model_path))) as loaded:
        assert torch.equal(loaded.response(hidden=cache["neutral_hidden"][4:5], strength=.5), expected_response)
    with pytest.raises(Exception, match="native factory source|registration"):
        open_inference(artifact, backend=original.INFERENCE, operation="response",
                       options=dict(readout_file=str(model_path)))
    hidden_path, result_path = cache_path.parent / "hidden.pt", cache_path.parent / "fresh.pt"
    torch.save(cache["neutral_hidden"][4:5], hidden_path)
    code = '''import sys,torch
from glide_particle_sliders.binding_residual_v2 import INFERENCE
from particle_sliders.inference import open_inference
with open_inference(sys.argv[1],backend=INFERENCE,operation="response",options={"readout_file":sys.argv[2]}) as run:
    torch.save(run.response(hidden=torch.load(sys.argv[3],weights_only=True),strength=.5),sys.argv[4])
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(product.__file__).parents[1]),
                       CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code, str(artifact_path), str(model_path), str(hidden_path), str(result_path)],
                   check=True, env=environment, capture_output=True, text=True)
    assert torch.equal(torch.load(result_path, weights_only=True), expected_response)


def test_new_cohort_keeps_noema_cpu_and_split_guards(inputs):
    cache_path, _, _ = inputs
    for options, message in ((dict(ema_decay=.9), "no EMA"), (dict(device="cuda:0"), "CPU-only"),
                              (dict(split="dev"), "train split")):
        with pytest.raises(ValueError, match=message):
            product.BACKEND.open(product.config(cache_file=str(cache_path), **options))
