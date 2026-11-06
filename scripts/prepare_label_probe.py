#!/usr/bin/env python3
"""Immutable corrected input and explicitly synthetic label-target derivative.

Preserves the initial run. Re-encodes only scenarios changed while its dataset
was being finalized. This is not a native instruction-teacher result.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.native import load_glide, encode_requests, sha256


def main():
    output = Path("artifacts/label-probe-inputs-v1")
    output.mkdir(exist_ok=False)
    source = Path("artifacts/concise-cache-v2.pt")
    cache = load_readout_file(source, cache=True)
    dataset_path = Path("data/concise-preference-v1.json")
    dataset_bytes = dataset_path.read_bytes()
    rows = json.loads(dataset_bytes)["rows"]
    previous = cache["metadata"]["dataset"]
    if len(rows) != len(previous):
        raise ValueError("this derivative requires exactly the same row count")
    changed = [i for i, (a, b) in enumerate(zip(previous, rows)) if a != b]
    if any(rows[i]["split"] != previous[i]["split"] for i in changed):
        raise ValueError("split changes require a distinct complete experiment design")
    model, identity = load_glide("artifacts/native-model", "native-model.lock.json", threads=4)
    if identity != cache["model_identity"]:
        raise ValueError("native model differs from the original cached model")
    requests = [rows[i][arm] for i in changed for arm in ("neutral", "positive")]
    hidden, logits, lengths = encode_requests(model, requests, batch_size=2)
    fixed = deepcopy(cache)
    for k, index in enumerate(changed):
        fixed["neutral_hidden"][index] = hidden[2*k]
        fixed["neutral_logits"][index] = logits[2*k, :2]
        fixed["teacher_logits"][index] = logits[2*k+1, :2]
        fixed["metadata"]["lengths"][2*index:2*index+2] = lengths[2*k:2*k+2]
    fixed["row_ids"] = [r["id"] for r in rows]
    fixed["metadata"]["dataset"] = rows
    fixed["metadata"]["dataset_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
    fixed["metadata"]["source_cache_sha256"] = sha256(source)
    fixed["metadata"]["changed_rows_reencoded"] = changed
    if sha256(dataset_path) != hashlib.sha256(dataset_bytes).hexdigest():
        raise ValueError("dataset changed during correction")
    unchanged = [i for i in range(len(rows)) if i not in changed]
    preserved = {key: torch.equal(cache[key][unchanged], fixed[key][unchanged]) for key in
                 ("neutral_hidden", "neutral_logits", "teacher_logits")}
    if not all(preserved.values()):
        raise ValueError("unaffected native tensors changed")
    torch.save(fixed, output / "corrected-native-instruction-cache.pt")
    labels = deepcopy(fixed)
    labels["native_instruction_teacher_logits"] = fixed["teacher_logits"].clone()
    delta = torch.full_like(labels["neutral_logits"], -1.5)
    for index, row in enumerate(rows):
        delta[index, row["preferred_slot"]] = 1.5
    labels["teacher_logits"] = labels["neutral_logits"] + delta
    labels["metadata"]["teacher_protocol"] = (
        "synthetic-known-label capacity probe: T=N+delta, preferred slot+1.5/other-1.5; "
        "frozen native encoder/readout particle training; NOT native instruction distillation; "
        "previously viewed test examples, exploratory")
    labels["metadata"]["target_rule"] = dict(preferred_logit_delta=1.5, other_logit_delta=-1.5,
        source="original instructor-authored synthetic preferred_slot labels", previous_test_viewed=True)
    torch.save(labels, output / "synthetic-label-cache.pt")
    report = dict(source_cache_sha256=sha256(source), final_dataset_sha256=sha256(dataset_path),
                  changed_rows=[dict(index=i, old_id=previous[i]["id"], new_id=rows[i]["id"], split=rows[i]["split"]) for i in changed],
                  native_forward_requests=len(requests), unchanged_tensors_exact=preserved,
                  corrected_native_cache_sha256=sha256(output / "corrected-native-instruction-cache.pt"),
                  synthetic_label_cache_sha256=sha256(output / "synthetic-label-cache.pt"),
                  target_protocol=labels["metadata"]["teacher_protocol"],
                  source_code_sha256=sha256(__file__))
    (output / "derivation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
