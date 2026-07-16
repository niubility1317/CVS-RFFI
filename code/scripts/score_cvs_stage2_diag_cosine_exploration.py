#!/usr/bin/env python3
"""Score a frozen diag-cosine Stage2-B/C prediction pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-prediction", required=True)
    parser.add_argument("--after-prediction", required=True)
    parser.add_argument("--truth-sidecar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = score_diag_cosine_pair(
        before_prediction_path=args.before_prediction,
        after_prediction_path=args.after_prediction,
        truth_sidecar_path=args.truth_sidecar,
        output_path=args.output,
        candidate=args.candidate,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
