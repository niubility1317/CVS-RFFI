#!/usr/bin/env python3
"""Prepare, smoke, shard, merge, truth-build, or score D109 Target125."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d109_target125 import (  # noqa: E402
    build_d109_target125_truth_catalog,
    predict_d109_target125,
    prepare_d109_target125_run,
    score_d109_target125,
    smoke_d109_target125_prepared_state,
    validate_d109_target125_prediction_manifest,
)


def _add_prepared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest-sha256", required=True)


def _add_prediction_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest-sha256", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="reuse sealed D108/D92 inputs")
    prepare.add_argument("--d92-matrix-manifest", type=Path, required=True)
    prepare.add_argument("--d92-matrix-manifest-sha256", required=True)
    prepare.add_argument("--d92-output-root", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--d108-method-lock", type=Path, required=True)
    prepare.add_argument("--d108-method-lock-sha256", required=True)
    prepare.add_argument("--ground-component-dir", type=Path, required=True)
    prepare.add_argument("--ground-manifest-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    smoke = commands.add_parser("smoke", help="exercise one row and all four arms")
    _add_prepared_inputs(smoke)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--row-index", type=int, default=0)
    smoke.add_argument("--scene-index", type=int, default=0)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--feature-batch-size", type=int, default=64)

    shard = commands.add_parser("predict-shard", help="seal one modulo-8 shard")
    _add_prepared_inputs(shard)
    shard.add_argument("--output-dir", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, choices=range(8), required=True)
    shard.add_argument("--device", required=True)
    shard.add_argument("--feature-batch-size", type=int, default=64)

    merge = commands.add_parser("merge", help="merge exactly eight D109 shards")
    merge.add_argument("--shard-manifest", type=Path, action="append", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate the sealed D109 manifest")
    validate.add_argument("--prediction-manifest", type=Path, required=True)
    validate.add_argument("--prediction-manifest-sha256")

    truth = commands.add_parser("build-truth", help="join sealed D92 truth sidecars")
    _add_prediction_input(truth)
    _add_prepared_inputs(truth)
    truth.add_argument("--truth-catalog", type=Path, required=True)

    score = commands.add_parser("score", help="score the complete D109 matrix")
    _add_prediction_input(score)
    score.add_argument("--truth-catalog", type=Path, required=True)
    score.add_argument("--truth-catalog-sha256", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_d109_target125_run(
            d92_matrix_manifest_path=args.d92_matrix_manifest,
            expected_d92_matrix_manifest_sha256=args.d92_matrix_manifest_sha256,
            d92_output_root=args.d92_output_root,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            d108_method_lock_path=args.d108_method_lock,
            expected_d108_method_lock_sha256=args.d108_method_lock_sha256,
            ground_component_dir=args.ground_component_dir,
            expected_ground_manifest_sha256=args.ground_manifest_sha256,
            output_dir=args.output_dir,
        )
    elif args.command == "smoke":
        result = smoke_d109_target125_prepared_state(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_dir=args.output_dir,
            row_index=args.row_index,
            scene_index=args.scene_index,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
        )
    elif args.command == "predict-shard":
        result = predict_d109_target125(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
        )
    elif args.command == "merge":
        result = predict_d109_target125(
            shard_manifest_paths=args.shard_manifest, output_dir=args.output_dir
        )
    elif args.command == "validate":
        result = validate_d109_target125_prediction_manifest(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=(
                args.prediction_manifest_sha256
            ),
        )
    elif args.command == "build-truth":
        result = build_d109_target125_truth_catalog(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=(
                args.prediction_manifest_sha256
            ),
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_path=args.truth_catalog,
        )
    else:
        result = score_d109_target125(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=(
                args.prediction_manifest_sha256
            ),
            truth_catalog_path=args.truth_catalog,
            expected_truth_catalog_file_sha256=args.truth_catalog_sha256,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
