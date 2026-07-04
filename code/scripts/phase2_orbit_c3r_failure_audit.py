#!/usr/bin/env python
"""Class-floor and confusion audit for ORBIT-C3R result JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_float(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, "missing"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "invalid"
    if not math.isfinite(parsed):
        return None, "invalid"
    return parsed, "ok"


def _float(value: Any, default: float = 0.0) -> float:
    parsed, status = _parse_float(value)
    if status != "ok" or parsed is None:
        return float(default)
    return float(parsed)


def _parse_count(value: Any) -> tuple[int, str]:
    parsed, status = _parse_float(value)
    if status != "ok" or parsed is None:
        return 0, status
    if parsed < 0:
        return 0, "invalid"
    return int(parsed), "ok"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _collab_sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
    label = str(item[0])
    try:
        return (0, int(label))
    except ValueError:
        match = re.search(r"\d+", label)
        if match:
            return (1, int(match.group(0)))
        return (2, label)


def _flatten_counts(prefix: str, counts: Mapping[str, Any], warnings: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in sorted(_dict(counts).items()):
        count, status = _parse_count(value)
        if status != "ok":
            warnings.append(f"invalid_{prefix}_count:{key}={value!r}")
        out[f"{prefix}_{key}"] = count
    return out


def build_failure_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_errors: list[str] = []
    schema_warnings: list[str] = []
    profile_results_raw = payload.get("profile_results")
    if not isinstance(profile_results_raw, dict) or not profile_results_raw:
        schema_errors.append("missing_or_empty_profile_results")
    profile_results = _dict(profile_results_raw)
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for profile_name, profile_result in sorted(profile_results.items()):
        if not isinstance(profile_result, dict):
            schema_errors.append(f"profile_not_object:{profile_name}")
            continue
        counts_raw = profile_result.get("counts")
        if not isinstance(counts_raw, dict) or not counts_raw:
            schema_errors.append(f"missing_or_empty_counts:{profile_name}")
            continue
        counts_by_k = _dict(counts_raw)
        for collab_count, metrics_any in sorted(counts_by_k.items(), key=_collab_sort_key):
            collab_count_value, collab_status = _parse_count(collab_count)
            if collab_status != "ok":
                schema_warnings.append(f"non_numeric_collab_count:{collab_count!r}")
            metrics = _dict(metrics_any)
            if not metrics:
                schema_warnings.append(f"empty_metrics:{profile_name}:{collab_count}")
            for role, acc_key, total_key, decision_key, output_key in [
                (
                    "old",
                    "per_old_class_acc",
                    "per_old_class_total",
                    "per_old_class_decision_counts",
                    "per_old_class_output_counts",
                ),
                (
                    "seen_new",
                    "per_seen_new_class_acc",
                    "per_seen_new_class_total",
                    "per_seen_new_class_decision_counts",
                    "per_seen_new_class_output_counts",
                ),
            ]:
                accs = _dict(metrics.get(acc_key))
                totals = _dict(metrics.get(total_key))
                decisions = _dict(metrics.get(decision_key))
                outputs = _dict(metrics.get(output_key))
                for label in sorted(set(accs) | set(totals) | set(decisions) | set(outputs)):
                    acc_value, acc_status = _parse_float(accs.get(label))
                    class_total, total_status = _parse_count(totals.get(label))
                    if total_status == "invalid":
                        schema_warnings.append(
                            f"invalid_total:{profile_name}:{collab_count}:{role}:{label}={totals.get(label)!r}"
                        )
                    row = {
                        "profile": profile_name,
                        "collab_count": collab_count_value,
                        "collab_count_label": str(collab_count),
                        "role": role,
                        "label": str(label),
                        "class_acc": acc_value,
                        "acc_status": acc_status,
                        "class_total": class_total,
                        "total_status": total_status,
                    }
                    row["is_no_event"] = row["class_total"] <= 0
                    row["is_real_floor_failure"] = (
                        acc_status == "ok" and (not row["is_no_event"]) and float(row["class_acc"]) <= 0.0
                    )
                    row["is_floor_failure"] = row["is_real_floor_failure"]
                    row.update(_flatten_counts("decision", _dict(decisions.get(label)), schema_warnings))
                    row.update(_flatten_counts("output", _dict(outputs.get(label)), schema_warnings))
                    rows.append(row)
            for transition, count in sorted(_dict(metrics.get("open_set_confusion")).items()):
                parsed_count, count_status = _parse_count(count)
                if count_status != "ok":
                    schema_warnings.append(
                        f"invalid_confusion_count:{profile_name}:{collab_count}:{transition}={count!r}"
                    )
                confusion_rows.append(
                    {
                        "profile": profile_name,
                        "collab_count": collab_count_value,
                        "collab_count_label": str(collab_count),
                        "transition": str(transition),
                        "count": parsed_count,
                    }
                )

    floor_rows = [row for row in rows if row["is_real_floor_failure"]]
    no_event_rows = [row for row in rows if row["is_no_event"]]
    return {
        "algorithm": "ORBIT-C3R class-floor failure audit",
        "source_algorithm": payload.get("algorithm", ""),
        "feature_npz": payload.get("feature_npz", ""),
        "target_gates": payload.get("target_gates", {}),
        "class_rows": rows,
        "confusion_rows": confusion_rows,
        "floor_failure_count": len(floor_rows),
        "floor_failures": floor_rows,
        "no_event_count": len(no_event_rows),
        "no_event_rows": no_event_rows,
        "schema_errors": schema_errors,
        "schema_warnings": schema_warnings,
    }


CSV_LEADING_FIELDS = [
    "profile",
    "collab_count",
    "collab_count_label",
    "role",
    "label",
    "class_acc",
    "acc_status",
    "class_total",
    "total_status",
    "is_no_event",
    "is_real_floor_failure",
    "is_floor_failure",
    "transition",
    "count",
]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_fields = sorted({str(key) for row in rows for key in row.keys()} - set(CSV_LEADING_FIELDS))
    fields = [field for field in CSV_LEADING_FIELDS if field in dynamic_fields or any(field in row for row in rows)]
    fields.extend(dynamic_fields)
    if not rows:
        path.write_text(",".join(CSV_LEADING_FIELDS) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_class_csv", type=Path)
    parser.add_argument("--output_confusion_csv", type=Path)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    result = build_failure_audit(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_class_csv:
        _write_csv(args.output_class_csv, result["class_rows"])
    if args.output_confusion_csv:
        _write_csv(args.output_confusion_csv, result["confusion_rows"])
    if result["schema_errors"] and not args.allow_empty:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
