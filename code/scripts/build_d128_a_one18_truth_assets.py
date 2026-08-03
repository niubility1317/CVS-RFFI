#!/usr/bin/env python3
"""Build D128-A-ONE18 scorer-only truth and formal-D92 assets after opening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.stage2_d128_a_one18_scorer import build_d128_a_one18_truth_assets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--prepared-plan", required=True, type=Path)
    parser.add_argument("--prepared-plan-sha256", required=True)
    parser.add_argument("--method-lock", required=True, type=Path)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--truth-open-event", required=True, type=Path)
    parser.add_argument("--truth-open-event-sha256", required=True)
    parser.add_argument("--d92-retry2-root", required=True, type=Path)
    parser.add_argument("--d92-retry2-manifest", required=True, type=Path)
    parser.add_argument("--d92-retry2-manifest-sha256", required=True)
    parser.add_argument("--truth-catalog-output", required=True, type=Path)
    parser.add_argument("--formal-d92-reference-output", required=True, type=Path)
    parser.add_argument("--build-receipt-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = build_d128_a_one18_truth_assets(
        prediction_path=args.prediction,
        expected_prediction_sha256=args.prediction_sha256,
        prepared_plan_path=args.prepared_plan,
        expected_prepared_plan_sha256=args.prepared_plan_sha256,
        method_lock_path=args.method_lock,
        expected_method_lock_sha256=args.method_lock_sha256,
        truth_open_event_path=args.truth_open_event,
        expected_truth_open_event_sha256=args.truth_open_event_sha256,
        d92_retry2_root=args.d92_retry2_root,
        d92_retry2_manifest_path=args.d92_retry2_manifest,
        expected_d92_retry2_manifest_sha256=args.d92_retry2_manifest_sha256,
        truth_catalog_output=args.truth_catalog_output,
        formal_d92_reference_output=args.formal_d92_reference_output,
        build_receipt_output=args.build_receipt_output,
    )
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
