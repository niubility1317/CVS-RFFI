#!/usr/bin/env python3
"""Build or score the independently held D107-SCMKRR Target125 truth plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d107_truth_scorer import (  # noqa: E402
    build_d107_target125_truth_catalog,
    score_d107_target125,
)


def _add_prediction_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest-sha256", required=True)


def _add_prepared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest-sha256", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-truth",
        help="verify sealed predictions, then join the 125 D92 truth sidecars",
    )
    _add_prediction_input(build)
    _add_prepared_inputs(build)
    build.add_argument("--truth-catalog", type=Path, required=True)

    score = commands.add_parser(
        "score", help="open an immutable D107 truth catalog and score the full matrix"
    )
    _add_prediction_input(score)
    score.add_argument("--truth-catalog", type=Path, required=True)
    score.add_argument("--truth-catalog-sha256", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-truth":
        result = build_d107_target125_truth_catalog(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=args.prediction_manifest_sha256,
            plan_manifest_path=args.plan_manifest,
            expected_plan_file_sha256=args.plan_manifest_sha256,
            context_manifest_path=args.context_manifest,
            expected_context_file_sha256=args.context_manifest_sha256,
            output_path=args.truth_catalog,
        )
    else:
        result = score_d107_target125(
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_file_sha256=args.prediction_manifest_sha256,
            truth_catalog_path=args.truth_catalog,
            expected_truth_catalog_file_sha256=args.truth_catalog_sha256,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
