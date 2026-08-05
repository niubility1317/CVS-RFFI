#!/usr/bin/env python3
"""Prepare, smoke, predict, merge, and score NEXT-R5 FA-RDCE3+qKNN Target125."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_next_r5_fa_target125_matrix as matrix  # noqa: E402
from cvsrffi import stage2_next_r5_fa_target125_runtime as runtime  # noqa: E402
from cvsrffi.stage2_next_r5_fa_target125 import (  # noqa: E402
    build_target125_truth_catalog,
    score_target125_from_files,
)


def _json_plain(value: object) -> object:
    """Convert immutable runtime containers into JSON-compatible plain values."""

    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(item) for item in value]
    return value


def _add_prepared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest-sha256", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="pin the sealed D108 inputs and FA asset")
    prepare.add_argument("--d108-plan-manifest", type=Path, required=True)
    prepare.add_argument("--d108-plan-manifest-sha256", required=True)
    prepare.add_argument("--d108-context-manifest", type=Path, required=True)
    prepare.add_argument("--d108-context-manifest-sha256", required=True)
    prepare.add_argument("--fa-asset", type=Path, required=True)
    prepare.add_argument("--fa-asset-sha256", required=True)
    prepare.add_argument("--method-lock", type=Path, required=True)
    prepare.add_argument("--method-lock-sha256", required=True)
    prepare.add_argument("--pr160-extractor-runtime", type=Path, required=True)
    prepare.add_argument("--pr160-extractor-runtime-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    smoke = commands.add_parser("smoke", help="run one truth-free real-checkpoint outer/scene")
    _add_prepared_inputs(smoke)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--row-index", type=int, default=0, choices=range(matrix.OUTER_JOB_COUNT))
    smoke.add_argument("--scene-index", type=int, default=0, choices=range(len(matrix.SCENES)))
    smoke.add_argument("--device", default="cpu")

    shard = commands.add_parser("predict-shard", help="seal one immutable modulo-8 prediction shard")
    _add_prepared_inputs(shard)
    shard.add_argument("--output-dir", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True, choices=range(runtime.SHARD_COUNT))
    shard.add_argument("--device", required=True)

    merge = commands.add_parser("merge", help="merge exactly eight completed prediction shards")
    merge.add_argument("--shard-manifest", type=Path, action="append", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    truth_open = commands.add_parser(
        "truth-open",
        help="open the D92 truth side only after the complete prediction manifest is sealed",
    )
    truth_open.add_argument("--prediction-manifest", type=Path, required=True)
    truth_open.add_argument("--prediction-manifest-sha256", required=True)
    _add_prepared_inputs(truth_open)
    truth_open.add_argument("--truth-catalog", type=Path, required=True)

    score = commands.add_parser("score", help="score a sealed prediction manifest with an independent truth catalog")
    score.add_argument("--prediction-manifest", type=Path, required=True)
    score.add_argument("--prediction-manifest-sha256", required=True)
    score.add_argument("--truth-catalog", type=Path, required=True)
    score.add_argument("--truth-catalog-sha256", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    """Exercise exactly one frozen row/scene without touching truth or scoring."""

    plan, context, source_plan, source_context = runtime._load_prepared_next_r5_inputs(  # type: ignore[attr-defined]
        plan_manifest_path=args.plan_manifest,
        expected_plan_file_sha256=args.plan_manifest_sha256,
        context_manifest_path=args.context_manifest,
        expected_context_file_sha256=args.context_manifest_sha256,
    )
    asset = runtime._load_target_asset(plan)  # type: ignore[attr-defined]
    pr160_extractor = runtime._prepared_pr160_extractor_runtime(  # type: ignore[attr-defined]
        plan=plan, source_plan=source_plan
    )
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    outer = frozen.outer_rows[args.row_index]
    target_row = context["rows"][args.row_index]
    source_row = source_context["rows"][target_row["source_row_index"]]
    scene = matrix.SCENES[args.scene_index]
    materializer = runtime.D108ZID160Materializer(
        source_plan=source_plan,
        pr160_extractor_runtime=pr160_extractor,
        device=args.device,
    )
    condition = materializer.materialize_condition(outer_row=outer, source_row=source_row, scene=scene)
    result = runtime.execute_target125_condition(
        condition,
        executor=runtime.FAqKNNCoreExecutor(source_plan=source_plan, fa_asset=asset),
    )
    states = result["core_result"]["states"]
    receipt = {
        "schema": "cvs.phase2.next_r5.fa_rdce3_qknn.target125.smoke_receipt.v1",
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "status": "REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "outer_id": outer.outer_id,
        "scene": scene,
        "states": list(matrix.STATES),
        "state_prediction_counts": {state: len(states[state]["predictions"]) for state in matrix.STATES},
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }
    receipt["smoke_receipt_sha256"] = matrix.canonical_sha256(receipt)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"immutable smoke output already exists: {args.output_dir}")
    if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
        raise ValueError("unsafe smoke output parent")
    args.output_dir.mkdir()
    path = args.output_dir / "smoke_receipt.json"
    raw = matrix.canonical_bytes(receipt) + b"\n"
    path.write_bytes(raw)
    return {
        **receipt,
        "smoke_receipt": str(path),
        "smoke_receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = runtime.prepare_next_r5_fa_target125_inputs(
            d108_plan_manifest_path=args.d108_plan_manifest,
            expected_d108_plan_file_sha256=args.d108_plan_manifest_sha256,
            d108_context_manifest_path=args.d108_context_manifest,
            expected_d108_context_file_sha256=args.d108_context_manifest_sha256,
            fa_asset_path=args.fa_asset,
            expected_fa_asset_sha256=args.fa_asset_sha256,
            method_lock_path=args.method_lock,
            expected_method_lock_sha256=args.method_lock_sha256,
            pr160_extractor_runtime_path=args.pr160_extractor_runtime,
            expected_pr160_extractor_runtime_sha256=args.pr160_extractor_runtime_sha256,
            output_dir=args.output_dir,
        )
    elif args.command == "smoke":
        result = _smoke(args)
    elif args.command == "predict-shard":
        result = runtime.predict_next_r5_fa_target125_shard(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            device=args.device,
        )
    elif args.command == "merge":
        result = runtime.merge_next_r5_fa_target125_shards(
            shard_manifest_paths=args.shard_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "truth-open":
        result = build_target125_truth_catalog(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=args.prediction_manifest_sha256,
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_path=args.truth_catalog,
        )
    else:
        result = score_target125_from_files(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=args.prediction_manifest_sha256,
            truth_catalog_path=args.truth_catalog,
            expected_truth_catalog_file_sha256=args.truth_catalog_sha256,
            output_dir=args.output_dir,
        )
    print(json.dumps(_json_plain(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
