"""Canonical multi-dimensional summaries over truth-last Phase2 scored rows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


CANONICAL_SUMMARY_SCHEMA = "cvs.phase2.canonical_summary.v1"
FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
REQUIRED_ROW_FIELDS = (
    "true_class_index",
    "predicted_class_index",
    "receiver_label",
    "day_label",
    "scenario",
    "query_token",
)


class CanonicalSummaryError(ValueError):
    """Raised when scored-row input cannot produce a canonical summary."""


def _materialize_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise CanonicalSummaryError("rows must be a nonempty iterable of mappings")
    try:
        materialized = list(rows)
    except TypeError as exc:
        raise CanonicalSummaryError(
            "rows must be a nonempty iterable of mappings"
        ) from exc
    if not materialized:
        raise CanonicalSummaryError("rows must be nonempty")
    return materialized


def _nonnegative_python_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise CanonicalSummaryError(
            f"{field} must be an exact nonnegative Python integer"
        )
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalSummaryError(f"{field} must be a nonempty string")
    return value


def _metric_records(
    groups: Mapping[Any, tuple[int, int]],
    *,
    key_name: str,
    ordered_keys: list[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ordered_keys:
        sample_count, correct_count = groups[key]
        records.append(
            {
                key_name: key,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "accuracy": float(correct_count / sample_count),
            }
        )
    return records


def _macro(records: list[dict[str, Any]]) -> float:
    return float(sum(record["accuracy"] for record in records) / len(records))


def summarize_scored_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and summarize independently scored Phase2 prediction rows."""

    materialized = _materialize_rows(rows)
    class_groups: dict[int, tuple[int, int]] = {}
    receiver_groups: dict[str, tuple[int, int]] = {}
    day_groups: dict[str, tuple[int, int]] = {}
    scene_groups: dict[str, tuple[int, int]] = {}
    cell_groups: dict[tuple[int, str, str, str], tuple[int, int]] = {}
    seen_prediction_keys: set[tuple[str, str]] = set()
    correct_count = 0

    for position, raw_row in enumerate(materialized):
        if not isinstance(raw_row, Mapping):
            raise CanonicalSummaryError(f"row {position} must be a mapping")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in raw_row]
        if missing:
            raise CanonicalSummaryError(
                f"row {position} missing required fields: {','.join(missing)}"
            )
        true_class = _nonnegative_python_int(
            raw_row["true_class_index"], field="true_class_index"
        )
        predicted_class = _nonnegative_python_int(
            raw_row["predicted_class_index"], field="predicted_class_index"
        )
        receiver = _nonempty_string(
            raw_row["receiver_label"], field="receiver_label"
        )
        day = _nonempty_string(raw_row["day_label"], field="day_label")
        scenario = _nonempty_string(raw_row["scenario"], field="scenario")
        if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CanonicalSummaryError(f"unsupported formal scenario: {scenario!r}")
        query_token = _nonempty_string(
            raw_row["query_token"], field="query_token"
        )
        prediction_key = (scenario, query_token)
        if prediction_key in seen_prediction_keys:
            raise CanonicalSummaryError(
                "duplicate (scenario,query_token) prediction key"
            )
        seen_prediction_keys.add(prediction_key)
        correct = int(predicted_class == true_class)
        correct_count += correct

        group_keys = (
            (class_groups, true_class),
            (receiver_groups, receiver),
            (day_groups, day),
            (scene_groups, scenario),
            (cell_groups, (true_class, receiver, day, scenario)),
        )
        for group, key in group_keys:
            sample_total, correct_total = group.get(key, (0, 0))
            group[key] = (sample_total + 1, correct_total + correct)

    scenario_order = {
        scenario: index for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }
    class_metrics = _metric_records(
        class_groups,
        key_name="true_class_index",
        ordered_keys=sorted(class_groups),
    )
    receiver_metrics = _metric_records(
        receiver_groups,
        key_name="receiver_label",
        ordered_keys=sorted(receiver_groups),
    )
    day_metrics = _metric_records(
        day_groups,
        key_name="day_label",
        ordered_keys=sorted(day_groups),
    )
    scene_metrics = _metric_records(
        scene_groups,
        key_name="scenario",
        ordered_keys=sorted(scene_groups, key=scenario_order.__getitem__),
    )
    ordered_cells = sorted(
        cell_groups,
        key=lambda key: (key[0], key[1], key[2], scenario_order[key[3]]),
    )
    cell_metrics = [
        {
            "true_class_index": key[0],
            "receiver_label": key[1],
            "day_label": key[2],
            "scenario": key[3],
            "sample_count": cell_groups[key][0],
            "correct_count": cell_groups[key][1],
            "accuracy": float(cell_groups[key][1] / cell_groups[key][0]),
        }
        for key in ordered_cells
    ]
    summary = {
        "schema": CANONICAL_SUMMARY_SCHEMA,
        "sample_count": len(materialized),
        "correct_count": correct_count,
        "sample_micro_accuracy": float(correct_count / len(materialized)),
        "class_macro_accuracy": _macro(class_metrics),
        "receiver_macro_accuracy": _macro(receiver_metrics),
        "day_macro_accuracy": _macro(day_metrics),
        "scene_macro_accuracy": _macro(scene_metrics),
        "class_group_count": len(class_metrics),
        "receiver_group_count": len(receiver_metrics),
        "day_group_count": len(day_metrics),
        "scene_group_count": len(scene_metrics),
        "observed_cell_count": len(cell_metrics),
        "class_metrics": class_metrics,
        "receiver_metrics": receiver_metrics,
        "day_metrics": day_metrics,
        "scene_metrics": scene_metrics,
        "cell_metrics": cell_metrics,
    }
    if any(
        not math.isfinite(value)
        for value in (
            summary["sample_micro_accuracy"],
            summary["class_macro_accuracy"],
            summary["receiver_macro_accuracy"],
            summary["day_macro_accuracy"],
            summary["scene_macro_accuracy"],
            *(record["accuracy"] for record in class_metrics),
            *(record["accuracy"] for record in receiver_metrics),
            *(record["accuracy"] for record in day_metrics),
            *(record["accuracy"] for record in scene_metrics),
            *(record["accuracy"] for record in cell_metrics),
        )
    ):
        raise CanonicalSummaryError("canonical summary contains a non-finite metric")
    return summary
