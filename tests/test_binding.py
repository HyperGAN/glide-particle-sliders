"""Real shared-core/HNDL lifecycle tests, using synthetic native-shaped inputs."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glide_particle_sliders import binding as product
from particle_sliders.inference import open_inference


@pytest.fixture
def cache(tmp_path):
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(255, 1024, generator=generator) * .02
    hidden = torch.randn(6, 1024, generator=generator)
    neutral = torch.nn.functional.linear(hidden, weight)[:, :2]
    teacher = neutral + torch.tensor([[.1, -.1], [-.1, .1], [.1, -.1], [-.1, .1], [.1, -.1], [-.1, .1]])
    packet = dict(schema=product.CACHE_SCHEMA, model_identity=dict(model_id=product.MODEL_ID,
        revision="1" * 40, qualification="synthetic1024x255 fixture; no pretrained execution"),
        readout_weight=weight, readout_bias=None, neutral_hidden=hidden,
        neutral_logits=neutral, teacher_logits=teacher,
        row_ids=[str(i) for i in range(6)], splits=["train"] * 4 + ["test"] * 2)
    path = tmp_path / "cache.pt"
    torch.save(packet, path)
    model = dict(schema=product.MODEL_SCHEMA, model_identity=packet["model_identity"],
                 readout_weight=weight, readout_bias=None)
    model_path = tmp_path / "readout.pt"
    torch.save(model, model_path)
    return path, packet, model_path


def session(path, **options):
    return product.BACKEND.open(product.config(cache_file=str(path), steps=4, **options))


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def test_native_head_hook_is_exact_off_and_restores_owner(cache):
    _, packet, _ = cache
    readout = product.readout_from_packet(packet)
    readout.train().requires_grad_(True)
    original = readout.forward
    hidden = packet["neutral_hidden"][:1]
    before = readout(hidden).detach()
    slider = product.ReadoutSlider(readout, model_identity=packet["model_identity"])
    try:
        assert torch.equal(slider.response(hidden, strength=0), before[:, :2])
        assert torch.equal(slider.response(hidden, strength=1), before[:, :2])
        assert slider.network.particles is slider.particles
        assert sum(p is slider.particles for p in slider.network.parameters()) == 1
        with torch.no_grad():
            slider.adapter.lora_up.weight.fill_(.01)
        assert not torch.equal(slider.response(hidden, strength=1), before[:, :2])
        assert torch.equal(slider.response(hidden, strength=0), before[:, :2])
        with slider.scaled(.5):
            with slider.scaled(-1):
                assert slider.adapter.multiplier == -1
            assert slider.adapter.multiplier == .5
        assert slider.adapter.multiplier == 0
    finally:
        slider.close()
    slider.close()
    assert readout.forward == original
    assert readout.training and readout.weight.requires_grad
    assert torch.equal(readout(hidden).detach(), before)


def test_cache_rejects_mismatched_native_logits(cache, tmp_path):
    _, packet, _ = cache
    packet["neutral_logits"][0, 0] += 1
    path = tmp_path / "broken.pt"
    torch.save(packet, path)
    with pytest.raises(ValueError, match="neutral logits"):
        product.load_readout_file(path, cache=True)


def test_no_ema_is_enforced_before_update(cache):
    path, _, _ = cache
    with pytest.raises(ValueError, match="no EMA"):
        session(path, ema_decay=.9)
    with session(path) as run:
        assert run.config.options["ema_decay"] is None
        assert run.ema.values is None
        assert run.game.prior.z is run.game.backend.particles
        assert run.game._signature["generator_arm"] == "routed_particle_gan"
        assert run.game.recipe.ema_decay == 0.


def test_cpu_and_immutable_source_gates(cache, monkeypatch):
    path, packet, _ = cache
    with pytest.raises(ValueError, match="CPU-only"):
        session(path, device="cuda:0")
    with pytest.raises(ValueError, match="model_id"):
        product.ReadoutSlider(product.readout_from_packet(packet), model_identity={"qualification": "unknown"})
    class FakeDistribution:
        def read_text(self, _):
            return '{"url":"https://github.com/HyperGAN/particle-sliders.git","vcs_info":{"commit_id":"wrong"}}'
    monkeypatch.setattr(product, "distribution", lambda _: FakeDistribution())
    with pytest.raises(ValueError, match="Git source pin"):
        product._runtime()


def test_inactive_readout_rows_remain_unchanged_after_shared_updates(cache):
    path, _, _ = cache
    with session(path) as run:
        slider = run.game.backend.slider
        before = slider.model.weight.detach().clone()
        run.train(stop_after=2)
        assert torch.equal(slider.model.weight, before)
        assert torch.count_nonzero(slider.adapter.lora_up.weight[:2]).item() > 0
        assert torch.count_nonzero(slider.adapter.lora_up.weight[2:]).item() == 0


def test_shared_checkpoint_resume_matches_uninterrupted_next_update(cache):
    path, _, _ = cache
    with session(path) as direct:
        direct.train(stop_after=2)
        checkpoint = direct.checkpoint()
        assert checkpoint["payload"]["ema"]["values"] is None
        direct.step()
        expected = direct.checkpoint()
    with session(path) as resumed:
        resumed.restore(checkpoint)
        assert resumed.game.completed_steps == 2
        resumed.step()
        same(expected, resumed.checkpoint())


def test_native_only_export_loads_without_training_cache(cache):
    path, packet, model_path = cache
    with session(path) as run:
        run.step()
        artifact_path = path.parent / "adapter.pt"
        artifact = run.export_inference(artifact_path, selection="raw")
        expected = run.evaluate(0, strength=.5, split="test")
        assert expected.shape == (1, 2)
    path.unlink()
    with open_inference(artifact, options=dict(readout_file=str(model_path)),
                        backend=product.INFERENCE, operation="response") as native:
        actual = native.response(hidden=packet["neutral_hidden"][4:5], strength=.5)
        assert torch.equal(actual, expected)
    # A fresh process reconstructs only the frozen native readout and adapter.
    # No encoder, cache, teacher rows, game, critic or optimizer is available.
    hidden_path, result_path = path.parent / "hidden.pt", path.parent / "fresh-response.pt"
    torch.save(packet["neutral_hidden"][4:5], hidden_path)
    code = '''import sys, torch
from glide_particle_sliders.binding import INFERENCE
from particle_sliders.inference import open_inference
with open_inference(sys.argv[1], options={"readout_file":sys.argv[2]}, backend=INFERENCE, operation="response") as run:
    value=run.response(hidden=torch.load(sys.argv[3],weights_only=True),strength=.5)
    torch.save(value,sys.argv[4])
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(product.__file__).parents[1]),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", code, str(artifact_path), str(model_path), str(hidden_path), str(result_path)],
                   env=environment, check=True, capture_output=True, text=True)
    assert torch.equal(torch.load(result_path, weights_only=True), expected)
    bad = torch.load(model_path, weights_only=True)
    bad["readout_weight"][0, 0] += 1
    torch.save(bad, model_path)
    with pytest.raises(ValueError, match="identity differs"):
        open_inference(artifact, options=dict(readout_file=str(model_path)),
                       backend=product.INFERENCE, operation="response")


def test_close_does_not_overwrite_foreign_hook(cache):
    _, packet, _ = cache
    readout = product.readout_from_packet(packet)
    slider = product.ReadoutSlider(readout, model_identity=packet["model_identity"])
    owned = readout.forward
    foreign = lambda hidden: hidden[:, :255]
    readout.forward = foreign
    with pytest.raises(RuntimeError, match="ownership changed"):
        slider.close()
    assert readout.forward is foreign and not slider.closed
    readout.forward = owned
    slider.close()
