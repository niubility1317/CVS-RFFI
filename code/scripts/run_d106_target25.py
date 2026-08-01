#!/usr/bin/env python3
"""Prepare, predict, or independently score the frozen D106 Target25."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d106_target25_runner import (  # noqa: E402
    predict_d106_target25,
    prepare_d106_target25_run,
    score_d106_target25,
)


def _add_prepared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest-sha256", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="project sealed D92 locators into immutable D106 inputs"
    )
    prepare.add_argument("--matrix-index", type=Path, required=True)
    prepare.add_argument("--matrix-index-sha256", required=True)
    prepare.add_argument("--split-locator", type=Path, required=True)
    prepare.add_argument("--split-locator-sha256", required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--rdce-wire", type=Path, required=True)
    prepare.add_argument("--rdce-wire-sha256", required=True)
    prepare.add_argument("--rdce-lock", type=Path, required=True)
    prepare.add_argument("--rdce-lock-sha256", required=True)
    prepare.add_argument("--rcmr-lock", type=Path, required=True)
    prepare.add_argument("--rcmr-lock-sha256", required=True)
    prepare.add_argument("--kcr-route-lock", type=Path, required=True)
    prepare.add_argument("--kcr-route-lock-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    predict = commands.add_parser(
        "predict", help="run the complete truth-free four-arm prediction matrix"
    )
    _add_prepared_inputs(predict)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--checkpoint-sha256", required=True)
    predict.add_argument("--rdce-wire", type=Path, required=True)
    predict.add_argument("--rdce-wire-sha256", required=True)
    predict.add_argument("--rcmr-lock", type=Path, required=True)
    predict.add_argument("--rcmr-lock-sha256", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--feature-batch-size", type=int, default=64)

    score = commands.add_parser(
        "score", help="validate predictions, record truth opening, then score"
    )
    _add_prepared_inputs(score)
    score.add_argument("--prediction-manifest", type=Path, required=True)
    score.add_argument("--prediction-manifest-sha256", required=True)
    score.add_argument("--truth-catalog", type=Path, required=True)
    score.add_argument("--truth-catalog-sha256", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_d106_target25_run(
            matrix_index_path=args.matrix_index,
            expected_matrix_index_sha256=args.matrix_index_sha256,
            split_locator_path=args.split_locator,
            expected_split_locator_sha256=args.split_locator_sha256,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            rdce_wire_path=args.rdce_wire,
            expected_rdce_wire_sha256=args.rdce_wire_sha256,
            rdce_lock_path=args.rdce_lock,
            expected_rdce_lock_sha256=args.rdce_lock_sha256,
            rcmr_lock_path=args.rcmr_lock,
            expected_rcmr_lock_sha256=args.rcmr_lock_sha256,
            kcr_route_lock_path=args.kcr_route_lock,
            expected_kcr_route_lock_sha256=args.kcr_route_lock_sha256,
            output_dir=args.output_dir,
        )
    elif args.command == "predict":
        result = predict_d106_target25(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            rdce_wire_path=args.rdce_wire,
            expected_rdce_wire_sha256=args.rdce_wire_sha256,
            rcmr_lock_path=args.rcmr_lock,
            expected_rcmr_lock_sha256=args.rcmr_lock_sha256,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
        )
    else:
        result = score_d106_target25(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_file_sha256=args.prediction_manifest_sha256,
            truth_catalog_path=args.truth_catalog,
            expected_truth_catalog_file_sha256=args.truth_catalog_sha256,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
