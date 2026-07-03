#!/usr/bin/env python
"""Diagnose whether existing evidence risk fields can separate unknown queries.

This is an offline diagnostic. It may sweep thresholds against held-out query
labels to estimate an upper bound, so its result must not be reported as a
deployable threshold-selection procedure.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

UNKNOWN_LABEL = "__unknown__"

DEFAULT_RISK_FIELDS = [
    "unknown_risk",
    "score_risk",
    "radius_risk",
    "margin_risk",
    "mahalanobis_risk",
    "evt_risk",
    "oldness_risk",
    "class_shell_risk",
    "class_negative_risk",
    "virtual_unknown_risk",
    "class_evidence_top1_unknown_risk",
    "class_evidence_top1_radius_risk",
    "class_evidence_top1_margin_risk",
    "class_evidence_top1_mahalanobis_risk",
    "class_evidence_top1_evt_risk",
    "class_evidence_top1_oldness_risk",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_evidence_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        out: list[dict[str, Any]] = []
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            out.append(parsed)
        return out


def _risk_value(row: Mapping[str, Any], fields: Sequence[str], mode: str) -> float:
    values = [_to_float(row.get(field, 0.0)) for field in fields]
    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "min":
        return min(values)
    raise ValueError("mode must be max, mean, or min")


def _thresholds(values: Iterable[float], *, max_thresholds: int) -> list[float]:
    vals = sorted({float(v) for v in values})
    if not vals:
        return [0.0]
    probes = {0.0, 1.0}
    probes.update(vals)
    for a, b in zip(vals, vals[1:]):
        probes.add((a + b) / 2.0)
    ordered = sorted(probes)
    limit = int(max_thresholds)
    if limit > 0 and len(ordered) > limit:
        step = (len(ordered) - 1) / float(limit - 1)
        ordered = [ordered[int(round(i * step))] for i in range(limit)]
        ordered = sorted(set(ordered))
    return ordered


def evaluate_gate(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], threshold: float, mode: str) -> dict[str, Any]:
    old_total = old_correct = 0
    seen_total = seen_correct = 0
    unknown_total = unknown_rejected = unknown_false_accept = 0
    defer_total = 0
    per_old: dict[str, list[int]] = {}
    per_seen: dict[str, list[int]] = {}
    for row in rows:
        role = str(row.get("role", ""))
        true_label = str(row.get("true_label", UNKNOWN_LABEL if role == "unknown" else ""))
        predicted = str(row.get("predicted_label", ""))
        risk = _risk_value(row, fields, mode)
        reject = risk >= float(threshold)
        if role == "old":
            old_total += 1
            per_old.setdefault(true_label, [0, 0])[1] += 1
            if not reject and predicted == true_label:
                old_correct += 1
                per_old[true_label][0] += 1
            if reject:
                defer_total += 1
        elif role == "seen_new":
            seen_total += 1
            per_seen.setdefault(true_label, [0, 0])[1] += 1
            if not reject and predicted == true_label:
                seen_correct += 1
                per_seen[true_label][0] += 1
            if reject:
                defer_total += 1
        elif role == "unknown":
            unknown_total += 1
            if reject:
                unknown_rejected += 1
            else:
                unknown_false_accept += 1
    total = old_total + seen_total + unknown_total
    known_total = old_total + seen_total
    known_correct = old_correct + seen_correct
    known_accepted = sum(
        1
        for row in rows
        if str(row.get("role", "")) in {"old", "seen_new"}
        and _risk_value(row, fields, mode) < float(threshold)
    )
    min_old = min((c / t for c, t in per_old.values() if t), default=0.0)
    min_seen = min((c / t for c, t in per_seen.values() if t), default=0.0)
    return {
        "fields": list(fields),
        "mode": mode,
        "threshold": float(threshold),
        "old_acc": old_correct / old_total if old_total else 0.0,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_correct / seen_total if seen_total else 0.0,
        "min_seen_new_class_acc": min_seen,
        "known_full_accuracy": known_correct / known_total if known_total else 0.0,
        "known_coverage": known_accepted / known_total if known_total else 0.0,
        "unknown_reject_rate": unknown_rejected / unknown_total if unknown_total else 0.0,
        "unknown_FAR": unknown_false_accept / unknown_total if unknown_total else 0.0,
        "defer_rate": defer_total / total if total else 0.0,
    }


def scan_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    risk_fields: Sequence[str],
    max_combo_size: int,
    modes: Sequence[str],
    far_targets: Sequence[float],
    max_thresholds: int,
) -> dict[str, Any]:
    present = [field for field in risk_fields if any(field in row for row in rows)]
    candidates: list[dict[str, Any]] = []
    for size in range(1, max(1, int(max_combo_size)) + 1):
        for fields in itertools.combinations(present, size):
            for mode in modes:
                risks = [_risk_value(row, fields, mode) for row in rows]
                for threshold in _thresholds(risks, max_thresholds=max_thresholds):
                    candidates.append(evaluate_gate(rows, fields, threshold, mode))
    by_far: dict[str, dict[str, Any]] = {}
    for target in far_targets:
        feasible = [item for item in candidates if item["unknown_FAR"] <= float(target)]
        feasible.sort(
            key=lambda item: (
                item["old_acc"] + item["seen_new_acc"],
                item["min_old_class_acc"] + item["min_seen_new_class_acc"],
                item["unknown_reject_rate"],
                item["known_coverage"],
            ),
            reverse=True,
        )
        by_far[str(float(target))] = feasible[0] if feasible else {"reason": "no_feasible_gate"}
    candidates.sort(
        key=lambda item: (
            item["unknown_reject_rate"],
            item["old_acc"] + item["seen_new_acc"],
            item["min_old_class_acc"] + item["min_seen_new_class_acc"],
        ),
        reverse=True,
    )
    return {
        "diagnostic_only": True,
        "uses_query_labels_for_oracle_sweep": True,
        "row_count": len(rows),
        "risk_fields_present": present,
        "best_by_far_target": by_far,
        "top_unknown_reject_candidates": candidates[:25],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence_csv", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--risk_fields", default=",".join(DEFAULT_RISK_FIELDS))
    p.add_argument("--max_combo_size", type=int, default=2)
    p.add_argument("--modes", default="max")
    p.add_argument("--far_targets", default="0.01,0.05,0.10")
    p.add_argument("--max_thresholds", type=int, default=256)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_evidence_csv(args.evidence_csv)
    risk_fields = [field.strip() for field in str(args.risk_fields).split(",") if field.strip()]
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    far_targets = [float(value) for value in str(args.far_targets).split(",") if str(value).strip()]
    result = scan_fields(
        rows,
        risk_fields=risk_fields,
        max_combo_size=int(args.max_combo_size),
        modes=modes,
        far_targets=far_targets,
        max_thresholds=int(args.max_thresholds),
    )
    result["evidence_csv"] = str(args.evidence_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
