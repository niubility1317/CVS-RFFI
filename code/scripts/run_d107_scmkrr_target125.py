#!/usr/bin/env python3
"""Prepare, smoke, or seal the truth-free D107-SCMKRR Target125 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d107_target125_runner import (  # noqa: E402
    predict_d107_target125,
    prepare_d107_target125_run,
    smoke_d107_target125_prepared_state,
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
        "prepare", help="locate and pin all 125 legal D92 package rows"
    )
    prepare.add_argument("--d92-matrix-manifest", type=Path, required=True)
    prepare.add_argument("--d92-matrix-manifest-sha256", required=True)
    prepare.add_argument("--d92-output-root", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--d107-method-lock", type=Path, required=True)
    prepare.add_argument("--d107-method-lock-sha256", required=True)
    prepare.add_argument("--rdce-asset-dir", type=Path, required=True)
    prepare.add_argument("--rdce-wire-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    smoke = commands.add_parser(
        "smoke", help="run one sealed outer/scene through both phases and all arms"
    )
    _add_prepared_inputs(smoke)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--row-index", type=int, default=0)
    smoke.add_argument("--scene-index", type=int, default=0)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--feature-batch-size", type=int, default=64)

    predict = commands.add_parser(
        "predict", help="write and seal all 3,000 pre-truth prediction surfaces"
    )
    _add_prepared_inputs(predict)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_d107_target125_run(
            d92_matrix_manifest_path=args.d92_matrix_manifest,
            expected_d92_matrix_manifest_sha256=args.d92_matrix_manifest_sha256,
            d92_output_root=args.d92_output_root,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            d107_method_lock_path=args.d107_method_lock,
            expected_d107_method_lock_sha256=args.d107_method_lock_sha256,
            rdce_asset_dir=args.rdce_asset_dir,
            expected_rdce_wire_sha256=args.rdce_wire_sha256,
            output_dir=args.output_dir,
        )
    elif args.command == "smoke":
        result = smoke_d107_target125_prepared_state(
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
    else:
        result = predict_d107_target125(
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_dir=args.output_dir,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
