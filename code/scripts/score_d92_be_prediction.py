#!/usr/bin/env python3
"""Join truth only after immutable D92-BE before/after predictions exist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--before-prediction", required=True)
    result.add_argument("--after-prediction", required=True)
    result.add_argument("--truth-sidecar", required=True)
    result.add_argument("--candidate", required=True)
    result.add_argument("--output-path", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    score = score_diag_cosine_pair(
        before_prediction_path=args.before_prediction,
        after_prediction_path=args.after_prediction,
        truth_sidecar_path=args.truth_sidecar,
        output_path=args.output_path,
        candidate=args.candidate,
    )
    print(
        json.dumps(
            {
                "status": "D92_BE_POST_PREDICTION_SCORE_COMPLETE",
                "candidate": score["candidate"],
                "score_artifact_sha256": score["score_artifact_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
