from types import SimpleNamespace

import pytest
import torch

from glide_particle_sliders import native


def test_cache_refuses_dataset_mutation_during_encoder_work(tmp_path, monkeypatch):
    dataset = tmp_path / "rows.json"
    dataset.write_text('{"rows":[{"id":"a","split":"train","neutral":{},"positive":{}},'
                       '{"id":"b","split":"train","neutral":{},"positive":{}}]}')
    head = torch.nn.Linear(1024, 255, bias=False)
    hidden = torch.ones(4, 1024)
    logits = head(hidden).detach()
    public = [{"answers":{"preference":{"probabilities":dict(zip(["first", "second"],
               torch.softmax(logits[i, :2], -1).tolist()))}}} for i in range(2)]
    model = SimpleNamespace(network=SimpleNamespace(readout=head), decide_batch=lambda requests: public)
    monkeypatch.setattr(native, "load_glide", lambda *args, **kwargs: (model, {"fixture": True}))
    def encode(*args, **kwargs):
        dataset.write_text(dataset.read_text() + "\n")
        return hidden, logits, [1] * 4
    monkeypatch.setattr(native, "encode_requests", encode)
    with pytest.raises(ValueError, match="changed during encoding"):
        native.prepare_cache("unused", "unused", dataset, tmp_path / "cache.pt", tmp_path / "head.pt")
    assert not (tmp_path / "cache.pt").exists()
    assert not (tmp_path / "head.pt").exists()
