#!/usr/bin/env python3
"""Build the complete machine-readable G0-G4 full-125 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvsrffi.stage2_m24_invariance_breaking import G1, G2, G3, G4
from cvsrffi.stage2_m24_safe_residual import D0
from scripts import summarize_m24_d1_refit_full125 as shared


ARMS = (D0, G1, G2, G3, G4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shared.ARMS = ARMS
    shared.REFERENCE_ARM = D0
    shared.PARITY_ARM = None
    shared.EXPECTED_INPUT_IDENTITIES = 125
    shared.SUMMARY_SCHEMA = "cvs.erbt_idr.m24.invariance_breaking_full125.results_summary.v1"
    shared.SUMMARY_VERDICT = "G0_G4_FULL125_MEASURED"
    result = shared.build_summary(Path(args.prediction_root), Path(args.score_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "row_count": result["matrix"]["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
