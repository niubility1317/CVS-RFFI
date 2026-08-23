"""Truth-last paired DA0_REG0/DA1_REG0 scorer for late-block adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class StructuredLateBlockPairScoringError(ValueError):
    """Raised when a frozen prediction pair cannot be scored exactly."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise StructuredLateBlockPairScoringError(f"JSON object required: {path}")
    return value


def _validate_prediction(
    path: str | Path, *, expected_state: str
) -> tuple[dict[str, str], Mapping[str, Any]]:
    value = _read_json(Path(path))
    if (
        value.get("status") != "PREDICTIONS_COMPLETE"
        or value.get("state") != expected_state
        or value.get("source_input_count") != 0
        or value.get("query_truth_loaded") is not False
        or value.get("query_role_loaded") is not False
        or value.get("query_batch_state_updated") is not False
    ):
        raise StructuredLateBlockPairScoringError(
            f"{expected_state} prediction boundary is invalid"
        )
    rows = value.get("predictions")
    if not isinstance(rows, list) or len(rows) != int(value.get("query_input_count", -1)):
        raise StructuredLateBlockPairScoringError("prediction rows are incomplete")
    by_token: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("sample_index") != index:
            raise StructuredLateBlockPairScoringError("prediction order is invalid")
        token = row.get("query_token")
        predicted = row.get("predicted_class_id")
        scores = np.asarray(row.get("scores"), dtype=np.float64)
        if (
            not isinstance(token, str)
            or not token
            or token in by_token
            or not isinstance(predicted, str)
            or not predicted
            or scores.ndim != 1
            or scores.size < 2
            or not np.isfinite(scores).all()
        ):
            raise StructuredLateBlockPairScoringError("prediction row is invalid")
        by_token[token] = predicted
    audit = value.get("audit")
    if not isinstance(audit, Mapping):
        raise StructuredLateBlockPairScoringError("prediction audit is missing")
    return by_token, audit


def _metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    true_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    per_class: dict[str, float] = {}
    for class_id in sorted(set(truth)):
        mask = true_array == class_id
        per_class[class_id] = float(np.mean(predicted_array[mask] == true_array[mask]))
    return {
        "old_class_mean_accuracy": float(np.mean(list(per_class.values()))),
        "old_class_floor_accuracy": float(min(per_class.values())),
        "old_class_micro_accuracy": float(np.mean(predicted_array == true_array)),
        "per_old_class_accuracy": per_class,
        "old_query_count": len(truth),
    }


def score_prediction_pair(
    da0_path: str | Path,
    da1_path: str | Path,
    truth_path: str | Path,
    *,
    scenario: str,
    output_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise StructuredLateBlockPairScoringError(
            f"refusing to overwrite score: {destination}"
        )
    da0, da0_audit = _validate_prediction(da0_path, expected_state="DA0_REG0")
    da1, da1_audit = _validate_prediction(da1_path, expected_state="DA1_REG0")
    if set(da0) != set(da1):
        raise StructuredLateBlockPairScoringError("DA0/DA1 opaque query IDs differ")
    for field in ("protocol_schema", "phase2_data_status", "capsule_id", "split_id"):
        if da0_audit.get(field) != da1_audit.get(field):
            raise StructuredLateBlockPairScoringError(f"DA0/DA1 {field} differs")

    truth_value = _read_json(Path(truth_path))
    rows = truth_value.get("rows")
    if not isinstance(rows, list):
        raise StructuredLateBlockPairScoringError("truth rows are missing")
    scenario_rows = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("scenario") == scenario
    ]
    truth_tokens = {str(row.get("query_token")) for row in scenario_rows}
    if truth_tokens != set(da0) or len(scenario_rows) != len(truth_tokens):
        raise StructuredLateBlockPairScoringError(
            "truth/prediction opaque query ID join is not exact"
        )
    old_rows = [row for row in scenario_rows if row.get("evaluation_role") == "target_old"]
    if not old_rows or any(
        not isinstance(row.get("true_class_handle"), str) for row in old_rows
    ):
        raise StructuredLateBlockPairScoringError("old-class truth rows are invalid")
    old_tokens = [str(row["query_token"]) for row in old_rows]
    old_truth = [str(row["true_class_handle"]) for row in old_rows]
    da0_metrics = _metrics(old_truth, [da0[token] for token in old_tokens])
    da1_metrics = _metrics(old_truth, [da1[token] for token in old_tokens])
    result = {
        "status": "ANALYZED",
        "scenario": scenario,
        "protocol_schema": da1_audit["protocol_schema"],
        "phase2_data_status": da1_audit["phase2_data_status"],
        "capsule_id": da1_audit["capsule_id"],
        "split_id": da1_audit["split_id"],
        "prediction_rows_verified_before_truth_open": len(da0),
        "truth_rows_joined": len(scenario_rows),
        "DA0_REG0": da0_metrics,
        "DA1_REG0": da1_metrics,
        "DA1_REG0_minus_DA0_REG0": {
            "old_class_mean_delta_pp": 100.0 * (
                da1_metrics["old_class_mean_accuracy"]
                - da0_metrics["old_class_mean_accuracy"]
            ),
            "old_class_floor_delta_pp": 100.0 * (
                da1_metrics["old_class_floor_accuracy"]
                - da0_metrics["old_class_floor_accuracy"]
            ),
        },
        "REG0_new_class_metrics": "N/A",
        "mrior_comparison": "UNKNOWN_NO_COMPLIANT_SAME_ROW_BASELINE",
        "scorer_output_must_not_feed_predictor": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result

