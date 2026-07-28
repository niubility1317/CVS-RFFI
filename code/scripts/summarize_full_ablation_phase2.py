#!/usr/bin/env python3
"""Summarize Phase2 same-row evidence without counting aliases twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_ablation_truth_scorer import (  # noqa: E402
    FAILED_ROW_SCHEMA,
    SAME_ROW_SCORE_SCHEMA,
)
from cvsrffi.stage2_metric_scorer import canonical_json_bytes  # noqa: E402


SUMMARY_SCHEMA = "cvs.full_ablation.phase2.summary.v1"
_METRICS = ("A_o_pre", "A_o_post", "A_n", "H", "F", "min_old", "min_new")
_SHA256_FIELDS = (
    "effective_config_hash",
    "before_prediction_hash",
    "after_prediction_hash",
    "behavior_receipt_sha256",
    "quantization_receipt_sha256",
    "resource_receipt_sha256",
    "same_row_metrics_sha256",
    "scorer_receipt_sha256",
)


class Phase2SummaryError(ValueError):
    """Raised when row evidence cannot be summarized without ambiguity."""


def _load_row(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase2SummaryError(f"cannot read row record: {source}") from exc
    if not isinstance(value, dict):
        raise Phase2SummaryError(f"row record root must be an object: {source}")
    return value


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") == "PASS":
        if row.get("schema") != SAME_ROW_SCORE_SCHEMA:
            raise Phase2SummaryError("PASS row schema drift")
        scenario_rows = row.get("scenario_rows")
        if not isinstance(scenario_rows, list) or not scenario_rows:
            raise Phase2SummaryError("PASS row has no scenario metrics")
        if row.get("truth_opened_after_prediction_commit") is not True:
            raise Phase2SummaryError("PASS row lacks truth-after-prediction proof")
        if not isinstance(row.get("scorer_receipt"), Mapping):
            raise Phase2SummaryError("PASS row lacks scorer receipt")
        receipt_sha256 = hashlib.sha256(
            canonical_json_bytes(row["scorer_receipt"])
        ).hexdigest()
        if receipt_sha256 != row.get("scorer_receipt_sha256"):
            raise Phase2SummaryError("scorer receipt hash mismatch")
        if (
            row["scorer_receipt"].get("scorer_output_must_not_feed_predictor")
            is not True
        ):
            raise Phase2SummaryError("scorer feedback guard missing")
        for field in (
            "logical_row_key",
            "ablation_id",
            "physical_execution_id",
            "effective_config_hash",
            "alias_of",
            "independent_observation",
            "stage",
            "receiver",
            "k_shot",
            "before_prediction_hash",
            "after_prediction_hash",
            "behavior_receipt_sha256",
            "quantization_receipt_sha256",
            "resource_receipt_sha256",
            "same_row_metrics_sha256",
        ):
            if row["scorer_receipt"].get(field) != row.get(field):
                raise Phase2SummaryError(
                    f"scorer receipt does not bind top-level {field}"
                )
        for field in _SHA256_FIELDS:
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise Phase2SummaryError(f"PASS row has invalid {field}")
        if (
            hashlib.sha256(canonical_json_bytes(row.get("behavior"))).hexdigest()
            != row["behavior_receipt_sha256"]
            or hashlib.sha256(
                canonical_json_bytes(row.get("quantization"))
            ).hexdigest()
            != row["quantization_receipt_sha256"]
            or hashlib.sha256(canonical_json_bytes(row.get("resource"))).hexdigest()
            != row["resource_receipt_sha256"]
            or hashlib.sha256(canonical_json_bytes(scenario_rows)).hexdigest()
            != row["same_row_metrics_sha256"]
        ):
            raise Phase2SummaryError("PASS row receipt/metric payload hash mismatch")
        for scenario in scenario_rows:
            if not isinstance(scenario, Mapping):
                raise Phase2SummaryError("scenario row must be an object")
            for metric in _METRICS:
                value = scenario.get(metric)
                if row.get("stage") == "stage2c" and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise Phase2SummaryError(
                        f"Stage2-C row has invalid same-row metric: {metric}"
                    )
    elif row.get("status") == "FAILED":
        if row.get("schema") != FAILED_ROW_SCHEMA:
            raise Phase2SummaryError("FAILED row schema drift")
        if row.get("scenario_rows") != []:
            raise Phase2SummaryError("FAILED row cannot contain performance metrics")
        failure_receipt = row.get("failure_receipt")
        failure_receipt_sha256 = row.get("failure_receipt_sha256")
        if not isinstance(failure_receipt, Mapping):
            raise Phase2SummaryError("FAILED row lacks failure receipt")
        if (
            not isinstance(failure_receipt_sha256, str)
            or hashlib.sha256(
                canonical_json_bytes(failure_receipt)
            ).hexdigest()
            != failure_receipt_sha256
        ):
            raise Phase2SummaryError("failure receipt hash mismatch")
        for field in (
            "logical_row_key",
            "ablation_id",
            "physical_execution_id",
            "effective_config_hash",
            "alias_of",
            "independent_observation",
            "stage",
            "receiver",
            "k_shot",
            "failure_code",
            "failure_fingerprint",
            "zero_prediction",
        ):
            if failure_receipt.get(field) != row.get(field):
                raise Phase2SummaryError(
                    f"failure receipt does not bind top-level {field}"
                )
    else:
        raise Phase2SummaryError("row status must be PASS or FAILED")

    for field in (
        "logical_row_key",
        "ablation_id",
        "physical_execution_id",
        "effective_config_hash",
        "independent_observation",
        "alias_of",
    ):
        if field not in row:
            raise Phase2SummaryError(f"row identity missing {field}")
    if not isinstance(row["logical_row_key"], str) or not row["logical_row_key"]:
        raise Phase2SummaryError("logical_row_key must be nonempty")
    if not isinstance(row["ablation_id"], str) or not row["ablation_id"]:
        raise Phase2SummaryError("ablation_id must be nonempty")
    if (
        not isinstance(row["physical_execution_id"], str)
        or not row["physical_execution_id"]
    ):
        raise Phase2SummaryError("physical_execution_id must be nonempty")
    effective_hash = row["effective_config_hash"]
    if (
        not isinstance(effective_hash, str)
        or len(effective_hash) != 64
        or any(character not in "0123456789abcdef" for character in effective_hash)
    ):
        raise Phase2SummaryError("effective_config_hash must be lowercase SHA256")
    expected_independent = row["alias_of"] is None
    if row["independent_observation"] is not expected_independent:
        raise Phase2SummaryError("alias/independent_observation contradiction")
    return dict(row)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain all rows while excluding failed/alias rows from statistics."""

    validated = [_validate_row(row) for row in rows]
    logical_keys = [row["logical_row_key"] for row in validated]
    if len(logical_keys) != len(set(logical_keys)):
        raise Phase2SummaryError("duplicate logical_row_key")

    by_logical = {row["logical_row_key"]: row for row in validated}
    independent_physical: dict[str, str] = {}
    for row in validated:
        if row["alias_of"] is None:
            physical = row["physical_execution_id"]
            if physical in independent_physical:
                raise Phase2SummaryError(
                    "multiple independent rows reuse one physical execution"
                )
            independent_physical[physical] = row["logical_row_key"]
            continue
        canonical = by_logical.get(row["alias_of"])
        if canonical is None or canonical["alias_of"] is not None:
            raise Phase2SummaryError("alias_of must reference a canonical logical row")
        if (
            canonical["physical_execution_id"] != row["physical_execution_id"]
            or canonical["effective_config_hash"] != row["effective_config_hash"]
        ):
            raise Phase2SummaryError(
                "alias does not bind the canonical physical execution/config"
            )

    failed = [
        {
            "logical_row_key": row["logical_row_key"],
            "ablation_id": row["ablation_id"],
            "physical_execution_id": row["physical_execution_id"],
            "failure_code": row["failure_code"],
            "failure_fingerprint": row["failure_fingerprint"],
            "zero_prediction": row["zero_prediction"],
        }
        for row in validated
        if row["status"] == "FAILED"
    ]
    aliases = [
        {
            "logical_row_key": row["logical_row_key"],
            "ablation_id": row["ablation_id"],
            "physical_execution_id": row["physical_execution_id"],
            "alias_of": row["alias_of"],
            "effective_config_hash": row["effective_config_hash"],
        }
        for row in validated
        if row["alias_of"] is not None
    ]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        groups.setdefault(row["ablation_id"], []).append(row)
    arm_summaries: list[dict[str, Any]] = []
    for ablation_id, group in sorted(groups.items()):
        included = [
            row
            for row in group
            if row["status"] == "PASS" and row["independent_observation"] is True
        ]
        metric_values = {
            metric: [
                float(scenario[metric])
                for row in included
                for scenario in row["scenario_rows"]
                if scenario.get(metric) is not None
            ]
            for metric in _METRICS
        }
        arm_summaries.append(
            {
                "ablation_id": ablation_id,
                "logical_row_count": len(group),
                "independent_pass_row_count": len(included),
                "failed_row_count": sum(
                    row["status"] == "FAILED" for row in group
                ),
                "alias_row_count": sum(
                    row["independent_observation"] is False for row in group
                ),
                "independent_scenario_observation_count": sum(
                    len(row["scenario_rows"]) for row in included
                ),
                "metric_mean": {
                    metric: _mean(values)
                    for metric, values in metric_values.items()
                },
                "metric_min": {
                    metric: min(values) if values else None
                    for metric, values in metric_values.items()
                },
            }
        )

    return {
        "schema": SUMMARY_SCHEMA,
        "logical_row_count": len(validated),
        "independent_physical_execution_count": len(independent_physical),
        "failed_row_count": len(failed),
        "alias_row_count": len(aliases),
        "excluded_failed_rows": failed,
        "excluded_alias_rows": aliases,
        "arm_summaries": arm_summaries,
        "all_rows": sorted(validated, key=lambda row: row["logical_row_key"]),
        "statistics_policy": (
            "PASS_AND_INDEPENDENT_OBSERVATION_ONLY;"
            "FAILED_AND_ALIAS_ROWS_RETAINED_IN_ALL_ROWS"
        ),
        "performance_feedback_to_scheduler_forbidden": True,
    }


def write_summary_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        data = canonical_json_bytes(payload) + b"\n"
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing Phase2 summary")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-record", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = summarize_rows([_load_row(path) for path in args.row_record])
    write_summary_exclusive(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
