#!/usr/bin/env python
"""Audit Stage2-C evidence candidate distributions before event fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCORE_FIELDS = (
    "known_score",
    "prototype_score",
    "old_anchor_score",
    "margin",
    "unknown_score",
    "quality",
    "rmd_score",
    "density_score",
    "shell_risk",
)


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return sum(clean) / float(len(clean)) if clean else 0.0


def _p95(values: Sequence[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    idx = int(math.ceil(0.95 * len(clean))) - 1
    return clean[max(0, min(idx, len(clean) - 1))]


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    top_label_set_counts = Counter(_str(row, "top_label_set", "unknown") for row in rows)
    top_label_counts = Counter(_str(row, "top_label", "unknown") for row in rows)
    receiver_counts = Counter(_str(row, "receiver_id", "unknown") for row in rows)
    match = sum(1 for row in rows if _str(row, "top_label") == _str(row, "true_label"))
    out: dict[str, Any] = {
        "row_count": int(total),
        "event_count": int(len({_str(row, "event_id") for row in rows})),
        "receiver_count": int(len(receiver_counts)),
        "top_label_match_rate": float(match / total) if total else 0.0,
        "top_label_set_counts": dict(sorted(top_label_set_counts.items())),
        "top_label_counts": dict(sorted(top_label_counts.items())),
        "receiver_counts": dict(sorted(receiver_counts.items())),
    }
    for field in SCORE_FIELDS:
        values = [_float(row, field) for row in rows if field in row]
        if values:
            out[f"{field}_mean"] = _mean(values)
            out[f"{field}_p95"] = _p95(values)
    return out


def audit_evidence_rows(rows: Sequence[Mapping[str, Any]], *, algorithm: str) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_role_label: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_role_receiver: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        role = _str(row, "role", "unknown")
        label = _str(row, "true_label", "unknown")
        receiver = _str(row, "receiver_id", "unknown")
        by_role[role].append(row)
        by_role_label[(role, label)].append(row)
        by_role_receiver[(role, receiver)].append(row)

    return {
        "algorithm": str(algorithm),
        "row_count": int(len(materialized)),
        "event_count": int(len({_str(row, "event_id") for row in materialized})),
        "receiver_count": int(len({_str(row, "receiver_id") for row in materialized})),
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "role_summary": {role: _summarize_group(group) for role, group in sorted(by_role.items())},
        "role_label_summary": {
            f"{role}|{label}": _summarize_group(group)
            for (role, label), group in sorted(by_role_label.items())
        },
        "role_receiver_summary": {
            f"{role}|{receiver}": _summarize_group(group)
            for (role, receiver), group in sorted(by_role_receiver.items())
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _flatten_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, item in summary.get("role_label_summary", {}).items():
        role, label = str(key).split("|", 1)
        row = {
            "role": role,
            "true_label": label,
            "row_count": item.get("row_count", 0),
            "event_count": item.get("event_count", 0),
            "receiver_count": item.get("receiver_count", 0),
            "top_label_match_rate": item.get("top_label_match_rate", 0.0),
            "top_label_set_counts_json": json.dumps(item.get("top_label_set_counts", {}), sort_keys=True),
            "top_label_counts_json": json.dumps(item.get("top_label_counts", {}), sort_keys=True),
            "receiver_counts_json": json.dumps(item.get("receiver_counts", {}), sort_keys=True),
        }
        for field in SCORE_FIELDS:
            for suffix in ("mean", "p95"):
                name = f"{field}_{suffix}"
                if name in item:
                    row[name] = item[name]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row.keys()})
    preferred = [
        "role",
        "true_label",
        "row_count",
        "event_count",
        "receiver_count",
        "top_label_match_rate",
        "top_label_set_counts_json",
        "top_label_counts_json",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_by_role_label_csv", type=Path)
    parser.add_argument("--algorithm", default="unknown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(args.evidence_csv)
    result = audit_evidence_rows(rows, algorithm=str(args.algorithm))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_by_role_label_csv:
        _write_csv(args.output_by_role_label_csv, _flatten_summary_rows(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
