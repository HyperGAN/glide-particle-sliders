#!/usr/bin/env python3
"""Create explicit synthetic preferred-slot targets from a verified native cache."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from glide_particle_sliders.binding import load_readout_file
from glide_particle_sliders.native import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-cache", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    packet = load_readout_file(args.native_cache, cache=True)
    rows = packet["metadata"]["dataset"]
    if [r["id"] for r in rows] != packet["row_ids"] or [r["split"] for r in rows] != packet["splits"]:
        raise ValueError("embedded dataset does not match cached row identities")
    result = deepcopy(packet)
    result["native_instruction_teacher_logits"] = packet["teacher_logits"].clone()
    delta = torch.full_like(packet["neutral_logits"], -1.5)
    for i, row in enumerate(rows):
        if type(row["preferred_slot"]) is not int or row["preferred_slot"] not in (0, 1):
            raise ValueError("preferred_slot must be0 or1")
        delta[i, row["preferred_slot"]] = 1.5
    result["teacher_logits"] = packet["neutral_logits"] + delta
    result["metadata"]["teacher_protocol"] = "synthetic-known-label T=N+delta; preferred_slot+1.5/other-1.5; no native teacher claim"
    result["metadata"]["source_native_cache_sha256"] = sha256(args.native_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(json.dumps(dict(output=str(output), sha256=sha256(output), rows=len(rows))), flush=True)


if __name__ == "__main__":
    main()
