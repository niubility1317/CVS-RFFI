"""Predict or truth-last score one existing SF-TAPFT clean-single bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_tapft_query_closure import (  # noqa: E402
    run_clean_query_prediction,
    score_clean_query_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--support", type=Path, required=True)
    predict.add_argument("--query", type=Path, required=True)
    predict.add_argument("--data-handle", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--device", required=True)
    score = commands.add_parser("score")
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        result = run_clean_query_prediction(
            bundle_path=args.bundle,
            support_path=args.support,
            query_path=args.query,
            data_handle_path=args.data_handle,
            output_root=args.output_root,
            device=args.device,
        )
    else:
        result = score_clean_query_prediction(
            prediction_root=args.prediction_root,
            truth_path=args.truth,
            data_handle_path=args.data_handle,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
