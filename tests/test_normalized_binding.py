import torch

from particle_sliders.inference import open_inference
from glide_particle_sliders import MODEL_ID, MODEL_REVISION
from glide_particle_sliders import binding_normalized_v3 as normalized


def fixture_cache(tmp_path):
    g = torch.Generator().manual_seed(34)
    h = torch.randn(8, 1024, generator=g) * .1 + 3
    w = torch.randn(255, 1024, generator=g) / 32
    n = torch.nn.functional.linear(h, w)[:, :2]
    identity = dict(model_id=MODEL_ID, revision=MODEL_REVISION, qualification="synthetic fixture")
    packet = dict(schema="glide-readout-cache-v1", model_identity=identity, neutral_hidden=h,
        teacher_logits=n + .2, neutral_logits=n, readout_weight=w, readout_bias=None,
        row_ids=[str(i) for i in range(8)], splits=["train"] * 4 + ["dev"] * 4)
    path = tmp_path / "cache.pt"; torch.save(packet, path)
    base = tmp_path / "readout.pt"
    torch.save(dict(schema="glide-readout-model-v1", model_identity=identity,
                    readout_weight=w, readout_bias=None), base)
    return path, base, packet


def test_statistics_are_train_only(tmp_path):
    _, _, packet = fixture_cache(tmp_path)
    center, rms = normalized.train_statistics(packet)
    packet["neutral_hidden"][4:] += 1000
    new_center, new_rms = normalized.train_statistics(packet)
    assert torch.equal(center, new_center) and torch.equal(rms, new_rms)


def test_normalized_full_state_and_native_only_buffers(tmp_path):
    cache, readout, packet = fixture_cache(tmp_path)
    cfg = normalized.config(cache_file=str(cache), steps=8, batch=2, lr=.001, critic="residual64")
    with normalized.BACKEND.open(cfg) as run:
        initial_center = run.game.network.input_center.clone()
        initial_rms = run.game.network.input_rms.clone()
        off = run.evaluate(0, strength=0.)
        direct = torch.nn.functional.linear(packet["neutral_hidden"][:1], packet["readout_weight"])[:, :2]
        assert torch.equal(off, direct)
        run.train(stop_after=4); saved = run.checkpoint()
        run.step(); expected = run.checkpoint()["sha256"]
        export = run.export_inference(selection="raw")
        hidden = packet["neutral_hidden"]
        response = run.game.backend.slider.response(hidden).detach().clone()
        assert torch.equal(run.game.network.input_center, initial_center)
        assert torch.equal(run.game.network.input_rms, initial_rms)
    with normalized.BACKEND.open(cfg) as resumed:
        resumed.restore(saved)
        resumed.step()
        assert resumed.checkpoint()["sha256"] == expected
    cache.unlink()
    with open_inference(export, backend=normalized.INFERENCE, operation="response",
                        options={"readout_file": str(readout)}) as inference:
        assert torch.equal(inference.response(hidden=hidden), response)
        assert torch.equal(inference.binding.network.input_center, initial_center)
        assert torch.equal(inference.binding.network.input_rms, initial_rms)


def test_both_declared_critics_update_with_no_ema(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    for critic in ("conditional16", "residual64"):
        with normalized.BACKEND.open(normalized.config(cache_file=str(cache), critic=critic,
                                                       steps=2, batch=2)) as run:
            run.train()
            assert run.ema.decay is None and run.ema.values is None
            assert run.game.completed_steps == 2
