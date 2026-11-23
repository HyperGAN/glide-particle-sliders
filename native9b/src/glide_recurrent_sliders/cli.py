"""CLI for an explicitly chosen GLiDE-9B recurrent replay."""

import argparse
import json
from pathlib import Path

from .identity import decode_json
from .runtime import RecurrentChoice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--full-receipt", type=Path, required=True)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--calibration", type=Path, help="Selected Block 23 calibration")
    provenance.add_argument("--development-evidence", type=Path, help="Verified Block 17 development rank receipt")
    parser.add_argument("--mode", choices=("adaptive", "off", "matched"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--match-side", choices=("nearest", "lower", "upper"),
                        help="Binary32 control for a matched Block 17 replay")
    args = parser.parse_args()
    if (args.mode == "matched") != (args.reference is not None):
        parser.error("--reference is required for matched mode")
    if args.match_side is not None and (args.mode != "matched" or args.development_evidence is None):
        parser.error("--match-side requires a matched Block 17 development replay")
    request = decode_json(args.request.read_bytes())
    reference = decode_json(args.reference.read_bytes())["trace"] if args.reference else None
    scorer = RecurrentChoice.open(
        args.model_dir, plan_file=args.plan, plan_sha256=args.plan_sha256, raw_file=args.raw,
        full_file=args.full, full_receipt_file=args.full_receipt, calibration_file=args.calibration,
        development_evidence_file=args.development_evidence,
    )
    print(json.dumps(scorer.score(request, mode=args.mode, reference=reference, match_side=args.match_side), allow_nan=False))
