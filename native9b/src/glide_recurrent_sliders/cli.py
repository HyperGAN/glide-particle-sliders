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
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=("adaptive", "off", "matched"), required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if (args.mode == "matched") != (args.reference is not None):
        parser.error("--reference is required for matched mode")
    request = decode_json(args.request.read_bytes())
    reference = decode_json(args.reference.read_bytes())["trace"] if args.reference else None
    scorer = RecurrentChoice.open(
        args.model_dir, plan_file=args.plan, plan_sha256=args.plan_sha256, raw_file=args.raw,
        full_file=args.full, full_receipt_file=args.full_receipt, calibration_file=args.calibration,
    )
    print(json.dumps(scorer.score(request, mode=args.mode, reference=reference), allow_nan=False))
