#!/usr/bin/env python3
"""Build the complete machine-readable B0-B3 M2.5 full125 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m25_anchored_residual import B1, B2, B3
from scripts import summarize_m24_d1_refit_full125 as shared


ARMS = (D1, B1, B2, B3)
PARITY_ARM = None


def _write_summary_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shared.ARMS = ARMS
    shared.REFERENCE_ARM = D1
    shared.PARITY_ARM = PARITY_ARM
    shared.EXPECTED_INPUT_IDENTITIES = 125
    shared.SUMMARY_SCHEMA = "cvs.erbt_idr.m25.anchored_residual_full125.results_summary.v1"
    shared.SUMMARY_VERDICT = "B0_B3_FULL125_MEASURED"
    result = shared.build_summary(Path(args.prediction_root), Path(args.score_root))
    output = Path(args.output)
    _write_summary_exclusive(output, result)
    print(json.dumps({"status": result["status"], "row_count": result["matrix"]["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
