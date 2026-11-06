#!/usr/bin/env python3
"""Closed-form feature-capacity diagnostic, separate from the particle model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.cli import assess
from glide_particle_sliders.native import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    cache = load_readout_file(args.cache, cache=True)
    train = [i for i, split in enumerate(cache["splits"]) if split == "train"]
    features = cache["neutral_hidden"].double()
    target = cache["teacher_logits"].double() - cache["neutral_logits"].double()
    xmean, ymean = features[train].mean(0), target[train].mean(0)
    x, y = features[train] - xmean, target[train] - ymean
    gram = x @ x.T / len(train)
    candidates, coefficients = [], []
    for regularization in (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.):
        alpha = torch.linalg.solve(gram + regularization * torch.eye(len(train), dtype=torch.float64), y)
        beta = x.T @ alpha / len(train)
        pred = cache["neutral_logits"].double() + (features - xmean) @ beta + ymean
        candidates.append(dict(regularization=regularization, dev=assess(cache, "dev", pred)))
        coefficients.append(beta)
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["dev"]["mean"])
    # Fix the development choice before computing exploratory test metrics.
    (root / "selection.json").write_text(json.dumps(candidates[selected], indent=2, sort_keys=True) + "\n")
    beta = coefficients[selected]
    predicted = cache["neutral_logits"].double() + (features - xmean) @ beta + ymean
    result = dict(schema="glide-ridge-capacity-control-v1", cache_sha256=sha256(args.cache),
                  protocol="Closed-form affine residual on the same frozen encoder features; train centering only; dual solve; no PG/particle optimization",
                  target_protocol=cache["metadata"].get("teacher_protocol"),
                  candidates=candidates, selected=candidates[selected],
                  exploratory_test=assess(cache, "test", predicted),
                  training=assess(cache, "train", predicted), test_previously_viewed=True,
                  source_sha256=sha256(__file__), finished=datetime.now(timezone.utc).isoformat())
    torch.save(dict(beta=beta, feature_train_mean=xmean, residual_train_mean=ymean), root / "local-coefficients.pt")
    (root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(regularization=candidates[selected]["regularization"], dev=candidates[selected]["dev"]["mean"],
                          exploratory_test=result["exploratory_test"]["mean"],
                          concise_choice_rate=result["exploratory_test"]["concise_choice_rate_main"])), flush=True)


if __name__ == "__main__":
    main()
