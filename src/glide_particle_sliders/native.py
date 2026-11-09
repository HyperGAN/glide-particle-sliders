"""Load the pinned upstream runtime locally and cache its exact native readout input.

The public decide API uses inference_mode. Training instead uses ordinary cached
encoder outputs and the same native Linear readout. No encoder is approximated.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

from . import MODEL_ID, MODEL_REVISION


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify_assets(model_dir, lock_file):
    root = Path(model_dir).resolve()
    lock = json.loads(Path(lock_file).read_text())
    if (lock.get("model_id"), lock.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("GLiDE model identity differs from the reviewed checkpoint")
    for name, digest in lock["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"native asset differs: {name}")
    decision = json.loads((root / "decision_config.json").read_text())
    cfg = json.loads((root / "config.json").read_text())
    if (decision.get("backbone"), decision.get("pooling"), len(decision.get("codes", [])),
            cfg.get("hidden_size"), cfg.get("num_hidden_layers")) != ("encoder", "last", 255, 1024, 28):
        raise ValueError("unsupported native GLiDE architecture")
    return lock


def load_glide(model_dir, lock_file, *, threads=4):
    """CPU FP32 evaluation cohort; does not allocate a GPU or download assets."""
    lock = verify_assets(model_dir, lock_file)
    root = Path(model_dir).resolve()
    torch.set_num_threads(threads)
    if "glide" in sys.modules and Path(sys.modules["glide"].__file__).resolve().parent != root / "glide":
        raise ValueError("another glide runtime is already imported")
    sys.path.insert(0, str(root))
    try:
        upstream = importlib.import_module("glide")
    finally:
        sys.path.remove(str(root))
    model = upstream.Glide.from_pretrained(str(root), device="cpu", dtype=torch.float32)
    if type(model.network.readout) is not torch.nn.Linear or model.network.readout.bias is not None:
        raise ValueError("expected native bias-free Linear decision readout")
    if model.network.training or any(p.requires_grad for p in model.network.parameters()):
        raise ValueError("native pretrained backbone must be frozen and in eval mode")
    from importlib.metadata import version
    identity = dict(model_id=MODEL_ID, revision=MODEL_REVISION,
                    assets_index_sha256=hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest(),
                    files=lock["files"], numerical=dict(torch=str(torch.__version__),
                    transformers=version("transformers"), tokenizers=version("tokenizers"),
                    device="cpu", dtype="float32", source_weights="bfloat16",
                    attention="sdpa", threads=threads),
                    response="first two native decision logits", pooling="native last real token before SEP")
    return model, identity


def encode_requests(model, requests, *, batch_size=2):
    """Return exact [N,1024] pre-readout hidden states and [N,255] logits.

    Uses upstream prompt renderer, dual tokenizer, head+tail truncation, left
    padding, position IDs and pooling by calling the native EncoderNetwork.
    """
    from glide.model import left_pad
    texts = []
    for request in requests:
        entries, vis = model._prepare(request)
        if vis or len(entries) != 1 or len(entries[0][2]) != 2:
            raise ValueError("this product supports exactly one text-only two-choice question")
        texts.append(entries[0][3])
    seqs = model.tokenizer(texts, add_special_tokens=False)["input_ids"]
    hidden, logits, lengths = [], [], []
    for start in range(0, len(seqs), batch_size):
        group = seqs[start:start + batch_size]
        if any(len(s) + 8 > model.max_len for s in group):
            raise ValueError("prompt exceeds upstream serving context")
        x, mask = left_pad(group, model.pad_id, "cpu")
        captured = []
        handle = model.network.readout.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                out = model.network(x, mask)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (len(group), 1024):
            raise RuntimeError("native readout input geometry differs")
        if not torch.isfinite(captured[0]).all() or not torch.isfinite(out).all():
            raise ValueError("nonfinite native output")
        hidden.append(captured[0]); logits.append(out.detach().clone())
        lengths.extend(map(len, group))
        print(json.dumps(dict(event="native_encoded", rows=min(start + batch_size, len(seqs)), total=len(seqs))), flush=True)
    return torch.cat(hidden), torch.cat(logits), lengths


def prepare_cache(model_dir, lock_file, dataset_file, cache_file, readout_file, *, threads=4, batch_size=2):
    dataset_bytes = Path(dataset_file).read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    rows = json.loads(dataset_bytes)["rows"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty dataset rows required")
    model, identity = load_glide(model_dir, lock_file, threads=threads)
    requests = [r[arm] for r in rows for arm in ("neutral", "positive")]
    h, logits, lengths = encode_requests(model, requests, batch_size=batch_size)
    # Public inference agreement is a separately exercised baseline, not a claim
    # based solely on sharing a loader. Reproduce the first two prompt shapes.
    public = model.decide_batch(requests[:2])
    errors = []
    for i, result in enumerate(public):
        p = torch.softmax(logits[i, :2], -1)
        observed = result["answers"]["preference"]["probabilities"]
        errors.append(max(abs(float(p[j]) - observed[k]) for j, k in enumerate(observed)))
    if max(errors) > 1e-5:
        raise ValueError("direct native path differs from public decide probabilities")
    weight = model.network.readout.weight.detach().clone()
    cache = dict(schema="glide-readout-cache-v1", model_identity=identity,
                 neutral_hidden=h[0::2], teacher_logits=logits[1::2, :2],
                 neutral_logits=logits[0::2, :2], readout_weight=weight,
                 readout_bias=None, row_ids=[r["id"] for r in rows],
                 splits=[r["split"] for r in rows],
                 metadata=dict(lengths=lengths, dataset_sha256=dataset_sha256, public_api_max_probability_error=max(errors),
                               dataset=rows, encoder_frozen=True))
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    if Path(cache_file).exists() or Path(readout_file).exists():
        raise FileExistsError("cache outputs are immutable; select new paths")
    if sha256(dataset_file) != dataset_sha256:
        raise ValueError("dataset changed during encoding; captured inputs preserved in memory, refusing a mislabeled cache")
    torch.save(cache, cache_file)
    torch.save(dict(schema="glide-readout-model-v1", model_identity=identity,
                    readout_weight=weight, readout_bias=None), readout_file)
    return dict(rows=len(rows), cache_sha256=sha256(cache_file),
                readout_sha256=sha256(readout_file), identity=identity,
                public_api_max_probability_error=max(errors))
