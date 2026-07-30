"""Fail closed unless the exact Stage2 states run is artifact-complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_ablation_release import (
    RUNNER_SUMMARY_SCHEMA,
    TERMINAL_ROW_SCHEMA,
    validate_sealed_stage2_plan,
)
from scripts.run_full_ablation_stage2 import (
    _validate_row_execution_receipt,
    _validate_score_completion,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_regular_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("expected SHA256 must be a lowercase 64-hex value")
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"completion artifact is not a regular file: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"completion artifact SHA256 drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"completion artifact must be a JSON object: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-sealed-plan", type=Path, required=True)
    parser.add_argument("--expected-states-plan-sha256", required=True)
    parser.add_argument("--states-runner-summary", type=Path, required=True)
    parser.add_argument("--expected-states-summary-sha256", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-logical", type=int, required=True)
    parser.add_argument("--expected-physical", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan_path = args.states_sealed_plan.absolute()
    summary_path = args.states_runner_summary.absolute()
    plan = _load_regular_json(plan_path, args.expected_states_plan_sha256)
    validate_sealed_stage2_plan(plan)
    summary = _load_regular_json(
        summary_path,
        args.expected_states_summary_sha256,
    )

    if (
        plan.get("run_id") != args.expected_run_id
        or plan.get("git_commit") != args.expected_git_commit
        or int(plan.get("logical_row_count", -1)) != args.expected_logical
        or int(plan.get("physical_execution_count", -1)) != args.expected_physical
        or Path(str(plan.get("log_root", ""))) != summary_path.parent
    ):
        raise ValueError("states sealed-plan identity drift")

    statuses = list(summary.get("statuses") or [])
    expected_physical_ids = {
        str(row["physical_execution_id"]) for row in plan["physical_rows"]
    }
    actual_physical_ids = {
        str(row.get("physical_execution_id", "")) for row in statuses
    }
    if (
        summary.get("schema") != RUNNER_SUMMARY_SCHEMA
        or summary.get("run_id") != args.expected_run_id
        or int(summary.get("logical_row_count", -1)) != args.expected_logical
        or int(summary.get("physical_execution_count", -1)) != args.expected_physical
        or int(summary.get("launched_physical_count", -1)) != args.expected_physical
        or int(summary.get("completed_physical_count", -1)) != args.expected_physical
        or int(summary.get("completed_logical_score_count", -1)) != args.expected_logical
        or int(summary.get("failed_physical_count", -1)) != 0
        or int(summary.get("not_launched_systemic_stop_count", -1)) != 0
        or int(summary.get("reused_physical_count", -1))
        != int(plan.get("reused_physical_count", -2))
        or summary.get("systemic_stop") is not False
        or summary.get("performance_values_visible_to_scheduler") is not False
        or summary.get("failure_fingerprints") != {}
        or summary.get("thread_errors") != []
        or len(statuses) != args.expected_physical
        or actual_physical_ids != expected_physical_ids
    ):
        raise ValueError("states runner summary is not artifact-complete")

    physical_by_id = {
        str(row["physical_execution_id"]): row for row in plan["physical_rows"]
    }
    logical_total = 0
    for status in statuses:
        physical_id = str(status["physical_execution_id"])
        physical = physical_by_id[physical_id]
        expected_logical = len(list(physical["logical_rows"]))
        status_path = Path(str(physical["status_path"]))
        scorer_return_codes = status.get("scorer_return_codes")
        if (
            status.get("schema") != TERMINAL_ROW_SCHEMA
            or status.get("run_id") != args.expected_run_id
            or status.get("physical_execution_id") != physical_id
            or status.get("representative_logical_row_key")
            != physical["representative_logical_row_key"]
            or status.get("mode") != physical["mode"]
            or int(status.get("gpu", -1)) != int(physical["worker"]["gpu"])
            or int(status.get("slot", -1)) != int(physical["worker"]["slot"])
            or status.get("status") != "COMPLETE"
            or status.get("prediction_complete") is not True
            or status.get("scores_complete") is not True
            or int(status.get("predictor_return_code", -1)) != 0
            or not isinstance(scorer_return_codes, list)
            or len(scorer_return_codes) != expected_logical
            or any(int(value) != 0 for value in scorer_return_codes)
            or int(status.get("logical_score_count", -1)) != expected_logical
            or int(status.get("expected_logical_score_count", -1)) != expected_logical
            or not status_path.is_file()
            or status_path.is_symlink()
            or json.loads(status_path.read_text(encoding="utf-8")) != status
        ):
            raise ValueError(f"states physical completion drift: {physical_id}")
        _validate_row_execution_receipt(
            physical["row_execution_receipt"],
            physical,
        )
        for logical in physical["logical_rows"]:
            _validate_score_completion(
                logical["score_completion_path"],
                logical,
                physical,
            )
        logical_total += expected_logical
    if logical_total != args.expected_logical:
        raise ValueError("states logical completion coverage drift")

    print(
        json.dumps(
            {
                "status": "ARTIFACTS_COMPLETE",
                "run_id": args.expected_run_id,
                "git_commit": args.expected_git_commit,
                "logical_row_count": args.expected_logical,
                "physical_execution_count": args.expected_physical,
                "failed_physical_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
