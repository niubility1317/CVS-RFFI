"""Open and score one sealed D127 S0 paired prediction without re-running it.

``open`` validates only truth-free inputs and writes the durable event consumed
by the independent truth-assets builder.  ``score`` revalidates those inputs,
then validates the already-written event by file SHA and content binding,
before it is allowed to read runner-supplied truth/formal artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[2]
_CODE = _ROOT / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from cvsrffi import stage2_d127_s0_scorer as scorer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    prepared = scorer.prepare_d127_s0_scoring_inputs(
        paired_prediction_path=args.paired_prediction,
        expected_paired_prediction_sha256=args.paired_prediction_sha256,
        prepared_plan_path=args.prepared_plan,
        expected_prepared_plan_sha256=args.prepared_plan_sha256,
        method_lock_path=args.method_lock,
        expected_method_lock_sha256=args.method_lock_sha256,
    )
    return prepared


def _open(args: argparse.Namespace) -> dict[str, Any]:
    """Create the one immutable authority required before truth construction."""

    normalized = _prepare(args)["normalized_prediction"]
    event = scorer.build_d127_s0_truth_open_event(normalized)
    event_path = scorer.write_d127_s0_truth_open_event_exclusive(args.truth_open_event_output, event)
    return {
        "status": "D127_S0_TRUTH_OPENED",
        "row_count": normalized["row_count"],
        "truth_open_event": str(event_path.resolve()),
        "truth_open_event_sha256": _sha256_file(event_path),
        "paired_prediction_sha256": normalized["paired_prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
    }


def _score(args: argparse.Namespace) -> dict[str, Any]:
    # This performs all truth-free reads/checks first.  Do not move the event,
    # truth or formal-reference handling above this boundary.
    normalized = _prepare(args)["normalized_prediction"]
    event, _event_file_sha = scorer._load_pinned_json(
        args.truth_open_event,
        expected_sha256=args.truth_open_event_sha256,
        name="D127 truth-open event",
    )
    scorer._validate_truth_open_event(event, normalized_prediction=normalized)

    # The event was written by ``open`` and independently hash-validated.  Only
    # now may runner-side truth and historic formal-D92 files be statted/read.
    truth, _truth_file_sha = scorer._load_pinned_json(
        args.truth_catalog,
        expected_sha256=args.truth_catalog_sha256,
        name="D127 runner truth catalog",
    )
    formal, _formal_file_sha = scorer._load_pinned_json(
        args.formal_d92_reference,
        expected_sha256=args.formal_d92_reference_sha256,
        name="D127 runner formal D92 reference",
    )
    score = scorer.score_d127_s0_paired(
        normalized_prediction=normalized,
        truth_open_event=event,
        truth_catalog=truth,
        formal_d92_reference=formal,
    )
    score_path = scorer.write_d127_s0_paired_score_exclusive(args.score_output, score)
    return {
        "status": "D127_S0_SCORED",
        "row_count": score["row_count"],
        "metric_row_count": score["metric_row_count"],
        "truth_open_event": str(Path(args.truth_open_event).resolve()),
        "truth_open_event_sha256": args.truth_open_event_sha256,
        "score_output": str(score_path.resolve()),
        "score_output_sha256": _sha256_file(score_path),
        "formal_d92_is_same_row_reference_only": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)

    def add_truth_free_inputs(target: argparse.ArgumentParser) -> None:
        target.add_argument("--paired-prediction", required=True, type=Path)
        target.add_argument("--paired-prediction-sha256", required=True)
        target.add_argument("--prepared-plan", required=True, type=Path)
        target.add_argument("--prepared-plan-sha256", required=True)
        target.add_argument("--method-lock", required=True, type=Path)
        target.add_argument("--method-lock-sha256", required=True)

    open_parser = actions.add_parser("open", help="validate truth-free pairs and write immutable truth-open event")
    add_truth_free_inputs(open_parser)
    open_parser.add_argument("--truth-open-event-output", required=True, type=Path)
    open_parser.set_defaults(handler=_open)

    score_parser = actions.add_parser("score", help="score after an independently built truth/formal artifact pair exists")
    add_truth_free_inputs(score_parser)
    score_parser.add_argument("--truth-open-event", required=True, type=Path)
    score_parser.add_argument("--truth-open-event-sha256", required=True)
    score_parser.add_argument("--truth-catalog", required=True, type=Path)
    score_parser.add_argument("--truth-catalog-sha256", required=True)
    score_parser.add_argument("--formal-d92-reference", required=True, type=Path)
    score_parser.add_argument("--formal-d92-reference-sha256", required=True)
    score_parser.add_argument("--score-output", required=True, type=Path)
    score_parser.set_defaults(handler=_score)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _emit(args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
