#!/usr/bin/env python3
"""Score a sealed CVS Stage2-B/C prediction in a truth-side isolated process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_metric_scorer import (  # noqa: E402
    score_sealed_prediction,
    write_scoring_outputs_exclusive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--expected-prediction-artifact-sha256", required=True)
    parser.add_argument("--expected-prediction-seal-sha256", required=True)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--expected-scoring-manifest-sha256", required=True)
    parser.add_argument("--formal-rows", type=Path, required=True)
    parser.add_argument("--formal-predictions", type=Path, required=True)
    parser.add_argument("--scoring-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, predictions, receipt = score_sealed_prediction(
        args.prediction_artifact,
        args.scoring_manifest,
        expected_prediction_artifact_sha256=(
            args.expected_prediction_artifact_sha256
        ),
        expected_prediction_seal_sha256=args.expected_prediction_seal_sha256,
        expected_scoring_manifest_sha256=args.expected_scoring_manifest_sha256,
    )
    write_scoring_outputs_exclusive(
        formal_rows_path=args.formal_rows,
        formal_predictions_path=args.formal_predictions,
        scoring_receipt_path=args.scoring_receipt,
        formal_rows=rows,
        formal_predictions=predictions,
        scoring_receipt=receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
