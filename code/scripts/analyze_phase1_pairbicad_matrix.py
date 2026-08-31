#!/usr/bin/env python3
"""Read-only analysis of the formal source-only PairBiCAD matrix.

The analyzer intentionally only visits the direct row directories under the
given run root and the source-only artifacts declared by the matrix contract.
It never loads a checkpoint or any target/Phase2/support/query/truth data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCENARIOS: tuple[str, str, str, str] = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
LEO_SCENARIOS: tuple[str, str, str] = SCENARIOS[1:]
ACCESS_FLAGS: tuple[str, ...] = (
    "source_only",
    "target_access",
    "phase2_access",
    "support_access",
    "query_access",
    "truth_access",
)
EXPECTED_ACCESS: dict[str, bool] = {
    "source_only": True,
    "target_access": False,
    "phase2_access": False,
    "support_access": False,
    "query_access": False,
    "truth_access": False,
}
AGGREGATE_METRICS: tuple[str, ...] = (
    "source_sat_hmean",
    "leo_mean",
    "leo_scenario_floor",
    "leo_class_floor",
    "clean_accuracy",
    "clean_floor_accuracy",
)
RECONSTRUCTION_ALIASES: dict[str, tuple[str, ...]] = {
    "missing": ("missing", "missing_keys"),
    "unexpected": ("unexpected", "unexpected_keys"),
    "shape_mismatch": (
        "shape_mismatch",
        "shape_mismatches",
        "shape_mismatch_keys",
    ),
}
_NONFINITE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


class MatrixAnalysisError(ValueError):
    """A validation failure tied to a matrix row and artifact field."""


def _row_error(row_id: str, field: str, message: str) -> MatrixAnalysisError:
    return MatrixAnalysisError(f"row={row_id} field={field}: {message}")


def _ensure_finite(value: Any, *, row_id: str, field: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _row_error(row_id, field, "value must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite(item, row_id=row_id, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite(item, row_id=row_id, field=f"{field}[{index}]")


def _read_json(path: Path, *, row_id: str, field: str) -> Any:
    if not path.is_file():
        raise _row_error(row_id, field, f"missing file: {path.name}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise _row_error(row_id, field, f"cannot read {path.name}: {exc}") from exc
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _row_error(row_id, field, f"invalid JSON in {path.name}: {exc}") from exc
    _ensure_finite(payload, row_id=row_id, field=field)
    return payload


def _read_jsonl(path: Path, *, row_id: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise _row_error(row_id, "metrics_epoch.jsonl", "missing file")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise _row_error(row_id, "metrics_epoch.jsonl", f"cannot read: {exc}") from exc
    lines = text.splitlines()
    if not lines:
        raise _row_error(row_id, "metrics_epoch.jsonl", "file is empty")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        field = f"metrics_epoch.jsonl.line[{line_number}]"
        if not line.strip():
            raise _row_error(row_id, field, "blank JSONL line")
        try:
            payload = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _row_error(row_id, field, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise _row_error(row_id, field, "record must be a JSON object")
        _ensure_finite(payload, row_id=row_id, field=field)
        records.append(payload)
    return records


def _reject_csv_nonfinite(value: str, *, row_id: str, field: str) -> None:
    if _NONFINITE_TOKEN.search(value):
        raise _row_error(row_id, field, "value must be finite")


def _read_csv(path: Path, *, row_id: str) -> list[Mapping[str, str]]:
    if not path.is_file():
        raise _row_error(row_id, "metrics_epoch.csv", "missing file")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise _row_error(row_id, "metrics_epoch.csv.header", "missing header")
            if len(fieldnames) != len(set(fieldnames)):
                raise _row_error(row_id, "metrics_epoch.csv.header", "duplicate field")
            records: list[Mapping[str, str]] = []
            for row_number, record in enumerate(reader, start=2):
                field = f"metrics_epoch.csv.row[{row_number}]"
                if None in record:
                    raise _row_error(row_id, field, "unexpected extra column")
                if any(value is None for value in record.values()):
                    raise _row_error(row_id, field, "missing column value")
                for name, value in record.items():
                    assert value is not None
                    _reject_csv_nonfinite(value, row_id=row_id, field=f"{field}.{name}")
                records.append(dict(record))
    except csv.Error as exc:
        raise _row_error(row_id, "metrics_epoch.csv", f"invalid CSV: {exc}") from exc
    except OSError as exc:
        raise _row_error(row_id, "metrics_epoch.csv", f"cannot read: {exc}") from exc
    if not records:
        raise _row_error(row_id, "metrics_epoch.csv", "file has no data rows")
    return records


def _strict_int(value: Any, *, row_id: str, field: str) -> int:
    if isinstance(value, bool):
        raise _row_error(row_id, field, "must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise _row_error(row_id, field, "must be an integer")


def _strict_bool(value: Any, *, row_id: str, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "true":
            return True
        if token == "false":
            return False
    raise _row_error(row_id, field, "must be boolean")


def _strict_number(value: Any, *, row_id: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _row_error(row_id, field, "must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _row_error(row_id, field, "value must be finite")
    return numeric


def _bounded_accuracy(value: Any, *, row_id: str, field: str) -> float:
    numeric = _strict_number(value, row_id=row_id, field=field)
    if not 0.0 <= numeric <= 1.0:
        raise _row_error(row_id, field, "accuracy must be between 0 and 1")
    return numeric


def _require_mapping(value: Any, *, row_id: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _row_error(row_id, field, "must be a JSON object")
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    name: str,
    expected: Any,
    *,
    row_id: str,
    field_prefix: str,
) -> None:
    field = f"{field_prefix}.{name}"
    if name not in mapping:
        raise _row_error(row_id, field, "missing field")
    actual = mapping[name]
    if isinstance(expected, bool):
        if actual is not expected:
            raise _row_error(row_id, field, f"expected {expected!r}, got {actual!r}")
    elif isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise _row_error(row_id, field, f"expected {expected!r}, got {actual!r}")
    elif actual != expected or type(actual) is not type(expected):
        raise _row_error(row_id, field, f"expected {expected!r}, got {actual!r}")


def _validate_access_flags(
    mapping: Mapping[str, Any], *, row_id: str, field_prefix: str, required: bool
) -> None:
    for name in ACCESS_FLAGS:
        if name not in mapping:
            if required:
                raise _row_error(row_id, f"{field_prefix}.{name}", "missing field")
            continue
        actual = _strict_bool(mapping[name], row_id=row_id, field=f"{field_prefix}.{name}")
        if actual is not EXPECTED_ACCESS[name]:
            raise _row_error(
                row_id,
                f"{field_prefix}.{name}",
                f"forbidden access flag; expected {EXPECTED_ACCESS[name]!r}",
            )


def _validate_telemetry_record(
    record: Mapping[str, Any],
    *,
    row_id: str,
    field_prefix: str,
    candidate: str,
    fold: int,
    seed: int,
) -> int:
    update = _strict_int(
        record.get("optimizer_update"),
        row_id=row_id,
        field=f"{field_prefix}.optimizer_update",
    )
    if "phase1_method" in record:
        _require_exact(
            record,
            "phase1_method",
            "bicad_xr",
            row_id=row_id,
            field_prefix=field_prefix,
        )
    for name, expected in (
        ("candidate_id", candidate),
        ("fold", fold),
        ("seed", seed),
    ):
        if name in record:
            field = f"{field_prefix}.{name}"
            actual = record[name]
            if isinstance(expected, int):
                actual = _strict_int(actual, row_id=row_id, field=field)
            elif actual != expected:
                raise _row_error(
                    row_id, field, f"expected {expected!r}, got {actual!r}"
                )
            if isinstance(expected, int) and actual != expected:
                raise _row_error(
                    row_id, field, f"expected {expected!r}, got {actual!r}"
                )
    _validate_access_flags(record, row_id=row_id, field_prefix=field_prefix, required=False)
    return update


def _validate_telemetry(
    row_root: Path,
    *,
    row_id: str,
    candidate: str,
    fold: int,
    seed: int,
    expected_updates: int,
) -> None:
    jsonl_records = _read_jsonl(row_root / "metrics_epoch.jsonl", row_id=row_id)
    csv_records = _read_csv(row_root / "metrics_epoch.csv", row_id=row_id)
    jsonl_updates = [
        _validate_telemetry_record(
            record,
            row_id=row_id,
            field_prefix=f"metrics_epoch.jsonl.line[{index}]",
            candidate=candidate,
            fold=fold,
            seed=seed,
        )
        for index, record in enumerate(jsonl_records, start=1)
    ]
    csv_updates = [
        _validate_telemetry_record(
            record,
            row_id=row_id,
            field_prefix=f"metrics_epoch.csv.row[{index + 1}]",
            candidate=candidate,
            fold=fold,
            seed=seed,
        )
        for index, record in enumerate(csv_records, start=1)
    ]
    if len(jsonl_updates) != len(csv_updates):
        raise _row_error(
            row_id,
            "metrics_epoch.row_count",
            f"JSONL has {len(jsonl_updates)} rows but CSV has {len(csv_updates)}",
        )
    for index, (jsonl_update, csv_update) in enumerate(
        zip(jsonl_updates, csv_updates, strict=True), start=1
    ):
        if jsonl_update != csv_update:
            raise _row_error(
                row_id,
                f"metrics_epoch.optimizer_update[{index}]",
                f"JSONL={jsonl_update} differs from CSV={csv_update}",
            )
    for name, updates in (("jsonl", jsonl_updates), ("csv", csv_updates)):
        if updates[-1] != expected_updates:
            raise _row_error(
                row_id,
                f"metrics_epoch.{name}.optimizer_update",
                f"expected final update {expected_updates}, got {updates[-1]}",
            )


def _empty_reconstruction(
    mapping: Mapping[str, Any], *, row_id: str, field_prefix: str
) -> None:
    for canonical, aliases in RECONSTRUCTION_ALIASES.items():
        selected_name = next((name for name in aliases if name in mapping), None)
        if selected_name is None:
            raise _row_error(
                row_id,
                f"{field_prefix}.{canonical}",
                "missing reconstruction field",
            )
        value = mapping[selected_name]
        if not isinstance(value, list):
            raise _row_error(
                row_id,
                f"{field_prefix}.{selected_name}",
                "must be a list",
            )
        if value:
            raise _row_error(
                row_id,
                f"{field_prefix}.{selected_name}",
                "strict reconstruction mismatch is not empty",
            )


def _validate_checkpoint_runtime(
    row_root: Path,
    *,
    row_id: str,
    candidate: str,
    fold: int,
    seed: int,
    expected_updates: int,
) -> None:
    payload = _read_json(
        row_root / "checkpoint_runtime.json",
        row_id=row_id,
        field="checkpoint_runtime",
    )
    artifact = _require_mapping(payload, row_id=row_id, field="checkpoint_runtime")
    runtime = _require_mapping(
        artifact.get("runtime"), row_id=row_id, field="checkpoint_runtime.runtime"
    )
    for name, expected in (
        ("phase1_method", "bicad_xr"),
        ("candidate_id", candidate),
        ("fold", fold),
        ("seed", seed),
        ("optimizer_update", expected_updates),
    ):
        _require_exact(
            runtime,
            name,
            expected,
            row_id=row_id,
            field_prefix="checkpoint_runtime.runtime",
        )
    if "total_updates" in runtime:
        _require_exact(
            runtime,
            "total_updates",
            expected_updates,
            row_id=row_id,
            field_prefix="checkpoint_runtime.runtime",
        )
    _validate_access_flags(
        runtime,
        row_id=row_id,
        field_prefix="checkpoint_runtime.runtime",
        required=True,
    )
    _empty_reconstruction(
        _require_mapping(
            artifact.get("reconstruction"),
            row_id=row_id,
            field="checkpoint_runtime.reconstruction",
        ),
        row_id=row_id,
        field_prefix="checkpoint_runtime.reconstruction",
    )
    for name in ("strict_reconstruction", "trainer_runtime_strict"):
        _require_exact(
            artifact,
            name,
            True,
            row_id=row_id,
            field_prefix="checkpoint_runtime",
        )
    for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        if name not in artifact:
            raise _row_error(row_id, f"checkpoint_runtime.{name}", "missing field")
        value = artifact[name]
        if not isinstance(value, list) or value:
            raise _row_error(
                row_id,
                f"checkpoint_runtime.{name}",
                "strict reconstruction mismatch is not empty",
            )

    checkpoint_name = artifact.get("checkpoint_path")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise _row_error(row_id, "checkpoint_runtime.checkpoint_path", "missing path")
    checkpoint_path = Path(checkpoint_name)
    if checkpoint_path.is_absolute() or len(checkpoint_path.parts) != 1:
        raise _row_error(
            row_id,
            "checkpoint_runtime.checkpoint_path",
            "checkpoint must be a direct child of the row",
        )
    checkpoint = row_root / checkpoint_path
    try:
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise _row_error(
                row_id,
                "checkpoint_runtime.checkpoint_path",
                "checkpoint is missing or empty",
            )
    except OSError as exc:
        raise _row_error(
            row_id,
            "checkpoint_runtime.checkpoint_path",
            f"cannot inspect checkpoint: {exc}",
        ) from exc


def _validate_diagnostics(row_root: Path, *, row_id: str) -> None:
    payload = _read_json(
        row_root / "diagnostics.json", row_id=row_id, field="diagnostics"
    )
    _require_mapping(payload, row_id=row_id, field="diagnostics")


def _validate_marker(
    row_root: Path, *, row_id: str, scenarios: Mapping[str, Mapping[str, Any]]
) -> None:
    payload = _read_json(
        row_root / "ARTIFACTS_COMPLETE.json",
        row_id=row_id,
        field="ARTIFACTS_COMPLETE",
    )
    marker = _require_mapping(payload, row_id=row_id, field="ARTIFACTS_COMPLETE")
    _require_exact(
        marker,
        "complete",
        True,
        row_id=row_id,
        field_prefix="ARTIFACTS_COMPLETE",
    )
    _require_exact(
        marker,
        "status",
        "ARTIFACTS_COMPLETE",
        row_id=row_id,
        field_prefix="ARTIFACTS_COMPLETE",
    )
    if marker.get("missing") != []:
        raise _row_error(
            row_id,
            "ARTIFACTS_COMPLETE.missing",
            "completion marker contains missing artifacts",
        )
    _empty_reconstruction(
        _require_mapping(
            marker.get("reconstruction"),
            row_id=row_id,
            field="ARTIFACTS_COMPLETE.reconstruction",
        ),
        row_id=row_id,
        field_prefix="ARTIFACTS_COMPLETE.reconstruction",
    )
    evaluations = _require_mapping(
        marker.get("evaluations"),
        row_id=row_id,
        field="ARTIFACTS_COMPLETE.evaluations",
    )
    for scenario in SCENARIOS:
        if scenario not in evaluations:
            raise _row_error(
                row_id,
                f"ARTIFACTS_COMPLETE.evaluations.{scenario}",
                "missing scenario completion",
            )
        if scenario not in scenarios:
            raise _row_error(
                row_id,
                f"ARTIFACTS_COMPLETE.evaluations.{scenario}",
                "scenario was not parsed",
            )


def _validate_scenario(
    row_root: Path, *, row_id: str, scenario: str
) -> dict[str, Any]:
    json_path = row_root / "evaluations" / f"{scenario}.json"
    log_path = row_root / "evaluations" / f"{scenario}.log"
    payload = _read_json(
        json_path,
        row_id=row_id,
        field=f"evaluations.{scenario}",
    )
    metrics = _require_mapping(
        payload, row_id=row_id, field=f"evaluations.{scenario}"
    )
    recorded_scenario = metrics.get("scenario", metrics.get("scene"))
    if recorded_scenario != scenario:
        raise _row_error(
            row_id,
            f"evaluations.{scenario}.scenario",
            f"expected {scenario!r}, got {recorded_scenario!r}",
        )
    _require_exact(
        metrics,
        "checkpoint_load_strict",
        True,
        row_id=row_id,
        field_prefix=f"evaluations.{scenario}",
    )
    for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        value = metrics.get(name)
        if not isinstance(value, list) or value:
            raise _row_error(
                row_id,
                f"evaluations.{scenario}.{name}",
                "strict reconstruction mismatch is not empty",
            )
    accuracy = _bounded_accuracy(
        metrics.get("accuracy"),
        row_id=row_id,
        field=f"evaluations.{scenario}.accuracy",
    )
    floor = _bounded_accuracy(
        metrics.get("floor_accuracy"),
        row_id=row_id,
        field=f"evaluations.{scenario}.floor_accuracy",
    )
    per_class_value = metrics.get("per_class_accuracy")
    per_class = _require_mapping(
        per_class_value,
        row_id=row_id,
        field=f"evaluations.{scenario}.per_class_accuracy",
    )
    if not per_class:
        raise _row_error(
            row_id,
            f"evaluations.{scenario}.per_class_accuracy",
            "must not be empty",
        )
    normalized_per_class: dict[str, float] = {}
    for class_id, class_accuracy in per_class.items():
        if not isinstance(class_id, str) or not class_id:
            raise _row_error(
                row_id,
                f"evaluations.{scenario}.per_class_accuracy",
                "class IDs must be non-empty strings",
            )
        normalized_per_class[class_id] = _bounded_accuracy(
            class_accuracy,
            row_id=row_id,
            field=f"evaluations.{scenario}.per_class_accuracy.{class_id}",
        )
    if not log_path.is_file():
        raise _row_error(row_id, f"evaluations.{scenario}.log", "missing file")
    try:
        log_text = log_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise _row_error(
            row_id, f"evaluations.{scenario}.log", f"cannot read: {exc}"
        ) from exc
    if not log_text.strip():
        raise _row_error(row_id, f"evaluations.{scenario}.log", "log is empty")
    return {
        "accuracy": accuracy,
        "floor_accuracy": floor,
        "per_class_accuracy": dict(sorted(normalized_per_class.items())),
    }


def _hmean(left: float, right: float) -> float:
    denominator = left + right
    return 0.0 if denominator == 0.0 else 2.0 * left * right / denominator


def _analyze_row(
    row_root: Path,
    *,
    candidate: str,
    fold: int,
    seed: int,
    expected_updates: int,
) -> dict[str, Any]:
    row_id = row_root.name
    _validate_telemetry(
        row_root,
        row_id=row_id,
        candidate=candidate,
        fold=fold,
        seed=seed,
        expected_updates=expected_updates,
    )
    _validate_checkpoint_runtime(
        row_root,
        row_id=row_id,
        candidate=candidate,
        fold=fold,
        seed=seed,
        expected_updates=expected_updates,
    )
    _validate_diagnostics(row_root, row_id=row_id)
    scenarios = {
        scenario: _validate_scenario(row_root, row_id=row_id, scenario=scenario)
        for scenario in SCENARIOS
    }
    class_keys = {
        scenario: tuple(scenarios[scenario]["per_class_accuracy"])
        for scenario in SCENARIOS
    }
    expected_class_keys = class_keys["clean"]
    for scenario in SCENARIOS[1:]:
        if class_keys[scenario] != expected_class_keys:
            raise _row_error(
                row_id,
                f"evaluations.{scenario}.per_class_accuracy",
                "dimension mismatch; class IDs differ from clean: "
                f"expected={list(expected_class_keys)!r} "
                f"actual={list(class_keys[scenario])!r}",
            )
    _validate_marker(row_root, row_id=row_id, scenarios=scenarios)

    leo_mean = sum(scenarios[scenario]["accuracy"] for scenario in LEO_SCENARIOS) / 3.0
    leo_scenario_floor = min(
        scenarios[scenario]["accuracy"] for scenario in LEO_SCENARIOS
    )
    leo_class_floor = min(
        class_accuracy
        for scenario in LEO_SCENARIOS
        for class_accuracy in scenarios[scenario]["per_class_accuracy"].values()
    )
    clean_accuracy = scenarios["clean"]["accuracy"]
    clean_floor_accuracy = scenarios["clean"]["floor_accuracy"]
    row: dict[str, Any] = {
        "row_id": row_id,
        "candidate_id": candidate,
        "fold": fold,
        "seed": seed,
        "optimizer_updates": expected_updates,
        "source_only": True,
        "scenarios": scenarios,
        "clean_accuracy": clean_accuracy,
        "clean_floor_accuracy": clean_floor_accuracy,
        "leo_mean": leo_mean,
        "leo_scenario_floor": leo_scenario_floor,
        "leo_class_floor": leo_class_floor,
        "source_sat_hmean": _hmean(clean_accuracy, leo_scenario_floor),
    }
    for scenario in SCENARIOS:
        row[f"{scenario}_accuracy"] = scenarios[scenario]["accuracy"]
        row[f"{scenario}_floor_accuracy"] = scenarios[scenario]["floor_accuracy"]
        row[f"{scenario}_per_class_accuracy"] = scenarios[scenario][
            "per_class_accuracy"
        ]
    return row


def _population_std(values: Sequence[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _candidate_summary(candidate: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 6:
        raise MatrixAnalysisError(
            f"candidate={candidate} field=row_count: expected 6 rows, got {len(rows)}"
        )
    summary: dict[str, Any] = {
        "candidate_id": candidate,
        "row_count": len(rows),
        "metrics": {},
    }
    for metric in AGGREGATE_METRICS:
        values = [float(row[metric]) for row in rows]
        mean = sum(values) / len(values)
        stats = {
            "mean": mean,
            "population_std": _population_std(values, mean),
            "minimum": min(values),
        }
        summary["metrics"][metric] = stats
        summary[f"{metric}_mean"] = stats["mean"]
        summary[f"{metric}_population_std"] = stats["population_std"]
        summary[f"{metric}_minimum"] = stats["minimum"]
    worst = min(
        rows,
        key=lambda row: (float(row["source_sat_hmean"]), str(row["row_id"])),
    )
    summary["worst_row"] = worst["row_id"]
    summary["worst_row_metrics"] = {
        metric: worst[metric]
        for metric in AGGREGATE_METRICS
        if metric in worst
    }
    return summary


def _parse_csv_text(raw: str, *, name: str, integer: bool) -> tuple[Any, ...]:
    values = tuple(part.strip() for part in raw.split(","))
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must be a non-empty comma-separated unique list")
    if not integer:
        if any(any(char in value for char in "\\/\0") for value in values):
            raise ValueError(f"{name} contains an invalid path component")
        return values
    result: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integers") from exc
        result.append(parsed)
    return tuple(result)


def _expected_rows(
    candidates: Sequence[str], folds: Sequence[int], seeds: Sequence[int]
) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    for candidate in candidates:
        for fold in folds:
            for seed in seeds:
                rows.append((candidate, fold, seed, f"{candidate}-F{fold}-S{seed}"))
    return rows


def analyze_matrix(
    run_root: str | Path,
    *,
    expected_candidates: Sequence[str],
    expected_folds: Sequence[int],
    expected_seeds: Sequence[int],
    expected_updates: int,
) -> dict[str, Any]:
    """Validate and aggregate one complete 30-row PairBiCAD matrix."""

    root = Path(run_root)
    if not root.is_dir():
        raise MatrixAnalysisError(f"row=<run-root> field=run_root: not a directory: {root}")
    if isinstance(expected_updates, bool) or not isinstance(expected_updates, int):
        raise MatrixAnalysisError("row=<run-root> field=expected_updates: must be an integer")
    if expected_updates <= 0:
        raise MatrixAnalysisError("row=<run-root> field=expected_updates: must be positive")
    expected = _expected_rows(expected_candidates, expected_folds, expected_seeds)
    if len(expected) != 30:
        raise MatrixAnalysisError(
            "row=<run-root> field=matrix_shape: expected exactly 30 rows from "
            f"the candidate/fold/seed product, got {len(expected)}"
        )
    expected_names = {row[3] for row in expected}
    try:
        actual_names = {path.name for path in root.iterdir() if path.is_dir()}
    except OSError as exc:
        raise MatrixAnalysisError(
            f"row=<run-root> field=row_directories: cannot enumerate: {exc}"
        ) from exc
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if unexpected:
            detail.append(f"unexpected={unexpected}")
        first_row = missing[0] if missing else unexpected[0]
        raise _row_error(first_row, "row_directories", "; ".join(detail))

    analyzed_rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[Mapping[str, Any]]] = {
        candidate: [] for candidate in expected_candidates
    }
    for candidate, fold, seed, row_id in expected:
        analyzed = _analyze_row(
            root / row_id,
            candidate=candidate,
            fold=fold,
            seed=seed,
            expected_updates=expected_updates,
        )
        analyzed_rows.append(analyzed)
        by_candidate[candidate].append(analyzed)

    summaries = [
        _candidate_summary(candidate, by_candidate[candidate])
        for candidate in expected_candidates
    ]
    by_id = {summary["candidate_id"]: summary for summary in summaries}
    ranking = sorted(
        expected_candidates,
        key=lambda candidate: (
            -float(by_id[candidate]["metrics"]["source_sat_hmean"]["mean"]),
            -float(by_id[candidate]["metrics"]["leo_mean"]["mean"]),
            -float(by_id[candidate]["metrics"]["leo_scenario_floor"]["minimum"]),
            -float(by_id[candidate]["metrics"]["clean_accuracy"]["mean"]),
            candidate,
        ),
    )
    ranked_summaries = [by_id[candidate] for candidate in ranking]
    return {
        "schema": "pairbicad_matrix_analysis_v1",
        "run_root": str(root.resolve()),
        "expected": {
            "candidates": list(expected_candidates),
            "folds": list(expected_folds),
            "seeds": list(expected_seeds),
            "optimizer_updates": expected_updates,
        },
        "row_count": len(analyzed_rows),
        "rows": analyzed_rows,
        "ranking": ranking,
        "candidates": ranked_summaries,
        "candidate_summaries": summaries,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


CSV_FIELDS: tuple[str, ...] = (
    "row_id",
    "candidate_id",
    "fold",
    "seed",
    "optimizer_updates",
    "clean_accuracy",
    "clean_floor_accuracy",
    "leo_clear_weak_accuracy",
    "leo_clear_weak_floor_accuracy",
    "leo_low_elev_weak_accuracy",
    "leo_low_elev_weak_floor_accuracy",
    "leo_rain_weak_accuracy",
    "leo_rain_weak_floor_accuracy",
    "leo_mean",
    "leo_scenario_floor",
    "leo_class_floor",
    "source_sat_hmean",
    "clean_per_class_accuracy",
    "leo_clear_weak_per_class_accuracy",
    "leo_low_elev_weak_per_class_accuracy",
    "leo_rain_weak_per_class_accuracy",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in CSV_FIELDS}
            for scenario in SCENARIOS:
                field = f"{scenario}_per_class_accuracy"
                output[field] = json.dumps(
                    row[field], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            writer.writerow(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-candidates", required=True)
    parser.add_argument("--expected-folds", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-updates", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates = _parse_csv_text(
            args.expected_candidates, name="expected_candidates", integer=False
        )
        folds = _parse_csv_text(args.expected_folds, name="expected_folds", integer=True)
        seeds = _parse_csv_text(args.expected_seeds, name="expected_seeds", integer=True)
        analysis = analyze_matrix(
            args.run_root,
            expected_candidates=candidates,
            expected_folds=folds,
            expected_seeds=seeds,
            expected_updates=args.expected_updates,
        )
        _write_json(args.output_json, analysis)
        _write_csv(args.output_csv, analysis["rows"])
    except (MatrixAnalysisError, OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
