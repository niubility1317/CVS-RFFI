#!/usr/bin/env python3
"""Open and score a sealed D128-A-ONE18 prediction without rerunning it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi import stage2_d128_a_one18_scorer as scorer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_truth_free_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--prepared-plan", required=True, type=Path)
    parser.add_argument("--prepared-plan-sha256", required=True)
    parser.add_argument("--method-lock", required=True, type=Path)
    parser.add_argument("--method-lock-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    open_parser = actions.add_parser("open", help="validate truth-free prediction and write an immutable truth-open event")
    _add_truth_free_inputs(open_parser)
    open_parser.add_argument("--truth-open-event-output", required=True, type=Path)
    score_parser = actions.add_parser("score", help="score after independent D128 truth/formal assets exist")
    _add_truth_free_inputs(score_parser)
    score_parser.add_argument("--truth-open-event", required=True, type=Path)
    score_parser.add_argument("--truth-open-event-sha256", required=True)
    score_parser.add_argument("--truth-catalog", required=True, type=Path)
    score_parser.add_argument("--truth-catalog-sha256", required=True)
    score_parser.add_argument("--formal-d92-reference", required=True, type=Path)
    score_parser.add_argument("--formal-d92-reference-sha256", required=True)
    score_parser.add_argument("--score-output", required=True, type=Path)
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    return scorer.prepare_d128_a_one18_scoring_inputs(
        prediction_path=args.prediction,
        expected_prediction_sha256=args.prediction_sha256,
        prepared_plan_path=args.prepared_plan,
        expected_prepared_plan_sha256=args.prepared_plan_sha256,
        method_lock_path=args.method_lock,
        expected_method_lock_sha256=args.method_lock_sha256,
    )


def _open(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    normalized = prepared["normalized_prediction"]
    event = scorer.build_d128_a_one18_truth_open_event(normalized)
    output = scorer.write_d128_a_one18_truth_open_event_exclusive(
        args.truth_open_event_output, event, normalized_prediction=normalized
    )
    return {
        "status": "D128_A_ONE18_TRUTH_OPENED",
        "candidate_id": normalized["candidate_id"],
        "row_count": normalized["row_count"],
        "truth_open_event": str(output.resolve()),
        "truth_open_event_sha256": _sha256_file(output),
        "prediction_sha256": normalized["prediction_sha256"],
    }


def _score(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    normalized = prepared["normalized_prediction"]
    event, _event_file_sha = scorer._read_pinned_json(
        args.truth_open_event, expected_sha256=args.truth_open_event_sha256, name="D128 truth-open event"
    )
    scorer._validate_truth_open_event(event, normalized_prediction=normalized)
    # No truth/formal path is opened until the preceding durable event validates.
    truth, _truth_file_sha = scorer._read_pinned_json(
        args.truth_catalog, expected_sha256=args.truth_catalog_sha256, name="D128 truth catalog"
    )
    formal, _formal_file_sha = scorer._read_pinned_json(
        args.formal_d92_reference,
        expected_sha256=args.formal_d92_reference_sha256,
        name="D128 formal D92 reference",
    )
    score = scorer.score_d128_a_one18(
        normalized_prediction=normalized,
        truth_open_event=event,
        truth_catalog=truth,
        formal_d92_reference=formal,
    )
    output = scorer.write_d128_a_one18_score_exclusive(args.score_output, score)
    return {
        "status": "D128_A_ONE18_SCORED",
        "candidate_id": score["candidate_id"],
        "row_count": score["row_count"],
        "metric_row_count": score["metric_row_count"],
        "score_output": str(output.resolve()),
        "score_output_sha256": _sha256_file(output),
        "promotion_action": "NONE_REPORT_ONLY",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = _open(args) if args.mode == "open" else _score(args)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
