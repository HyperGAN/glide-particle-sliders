import torch

from glide_particle_sliders import MODEL_ID, MODEL_REVISION
from glide_particle_sliders.cli import run_experiment


def test_shared_experiment_selects_dev_then_reloads_raw_export(tmp_path):
    rng = torch.Generator().manual_seed(12)
    hidden = torch.randn(6, 1024, generator=rng)
    weight = torch.randn(255, 1024, generator=rng) / 32
    neutral = torch.nn.functional.linear(hidden, weight)[:, :2]
    # Synthetic numerical fixture; this does not execute the GLiDE encoder.
    identity = dict(model_id=MODEL_ID, revision=MODEL_REVISION, qualification="synthetic fixture")
    rows = [dict(id=str(i), group=str(i // 2), concise_slot=i % 2,
                 preferred_slot=i % 2, control=False) for i in range(6)]
    packet = dict(schema="glide-readout-cache-v1", model_identity=identity, neutral_hidden=hidden,
                  neutral_logits=neutral, teacher_logits=neutral + .1,
                  readout_weight=weight, readout_bias=None,
                  row_ids=[str(i) for i in range(6)], splits=["train"] * 2 + ["dev"] * 2 + ["test"] * 2,
                  metadata=dict(dataset=rows))
    cache, model = tmp_path / "cache.pt", tmp_path / "readout.pt"
    torch.save(packet, cache)
    torch.save(dict(schema="glide-readout-model-v1", model_identity=identity,
                    readout_weight=weight, readout_bias=None), model)
    result = run_experiment(cache, model, tmp_path / "experiment", steps=4, rates=(.0001, .001), batch=2, threads=1)
    assert result["resume_proof"]["exact_packet_match"]
    assert result["raw_export_exact"]
    assert result["actual_updates"]["sweep"] == 8
    assert result["actual_updates"]["total_physical"] == 9
    assert result["resume_proof"]["replayed_prefix_updates"] == 0
    assert result["resource"]["gpu_launches"] == 0
    assert result["winner"]["dev"]["mean"] == min(c["dev"]["mean"] for c in result["candidates"])
