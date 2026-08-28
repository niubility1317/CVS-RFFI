#!/usr/bin/env python3
"""Predict and truth-last score D3 plus ERBT-IDR four-state rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_erbt_four_state import (  # noqa: E402
    run_four_state_prediction,
    score_four_state_predictions,
)
from cvsrffi.stage2_sf_d3_erbt_plan import build_d3_config  # noqa: E402
from cvsrffi.target_only_progressive_runner import (  # noqa: E402
    run_sf_tapft_deploy_no_query,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    adapt = commands.add_parser("adapt")
    adapt.add_argument("--plan", type=Path, required=True)
    adapt.add_argument("--scenario", required=True)
    adapt.add_argument("--output-root", type=Path, required=True)
    adapt.add_argument("--device", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--base-checkpoint", type=Path, required=True)
    predict.add_argument("--d3-delta", type=Path, required=True)
    predict.add_argument("--old-support", type=Path, required=True)
    predict.add_argument("--registered-support", type=Path, required=True)
    predict.add_argument("--query", type=Path, required=True)
    predict.add_argument("--data-handle", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--seed", type=int, required=True)
    predict.add_argument("--device", required=True)

    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "adapt":
        plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
        result = run_sf_tapft_deploy_no_query(
            build_d3_config(plan, args.scenario),
            args.output_root,
            device=args.device,
            deployment_inplace=True,
            emit_clean_single_bundle=False,
        )
    elif args.command == "predict":
        result = run_four_state_prediction(
            base_checkpoint_path=args.base_checkpoint,
            d3_delta_path=args.d3_delta,
            old_support_path=args.old_support,
            registered_support_path=args.registered_support,
            query_path=args.query,
            data_handle_path=args.data_handle,
            output_root=args.output_root,
            seed=args.seed,
            device=args.device,
        )
    else:
        result = score_four_state_predictions(
            args.predictions,
            args.truth,
            args.prediction_receipt,
            args.data_handle,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
