#!/usr/bin/env python3
"""Score sealed SOMP-H Stage2-B or matched Stage2-C prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_metric_scorer import (  # noqa: E402
    score_somph_registration_pair,
    score_somph_stage2b,
    write_somph_scoring_outputs_exclusive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage2b", "stage2c"), required=True)
    parser.add_argument("--before-prediction-artifact", required=True)
    parser.add_argument("--expected-before-artifact-sha256", required=True)
    parser.add_argument("--expected-before-seal-sha256", required=True)
    parser.add_argument("--after-prediction-artifact")
    parser.add_argument("--expected-after-artifact-sha256")
    parser.add_argument("--expected-after-seal-sha256")
    parser.add_argument("--scoring-manifest", required=True)
    parser.add_argument("--expected-scoring-manifest-sha256", required=True)
    parser.add_argument("--formal-rows", required=True)
    parser.add_argument("--formal-predictions", required=True)
    parser.add_argument("--scoring-receipt", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "stage2b":
        forbidden = (
            args.after_prediction_artifact,
            args.expected_after_artifact_sha256,
            args.expected_after_seal_sha256,
        )
        if any(value is not None for value in forbidden):
            raise ValueError("Stage2-B scoring must not receive an after-registration artifact")
        rows, predictions, receipt = score_somph_stage2b(
            args.before_prediction_artifact,
            args.scoring_manifest,
            expected_prediction_artifact_sha256=args.expected_before_artifact_sha256,
            expected_prediction_seal_sha256=args.expected_before_seal_sha256,
            expected_scoring_manifest_sha256=args.expected_scoring_manifest_sha256,
        )
    else:
        required = (
            args.after_prediction_artifact,
            args.expected_after_artifact_sha256,
            args.expected_after_seal_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("Stage2-C scoring requires the sealed after-registration artifact")
        rows, predictions, receipt = score_somph_registration_pair(
            args.before_prediction_artifact,
            args.after_prediction_artifact,
            args.scoring_manifest,
            expected_before_artifact_sha256=args.expected_before_artifact_sha256,
            expected_before_seal_sha256=args.expected_before_seal_sha256,
            expected_after_artifact_sha256=args.expected_after_artifact_sha256,
            expected_after_seal_sha256=args.expected_after_seal_sha256,
            expected_scoring_manifest_sha256=args.expected_scoring_manifest_sha256,
        )
    write_somph_scoring_outputs_exclusive(
        formal_rows_path=Path(args.formal_rows),
        formal_predictions_path=Path(args.formal_predictions),
        scoring_receipt_path=Path(args.scoring_receipt),
        formal_rows=rows,
        formal_predictions=predictions,
        scoring_receipt=receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
