#!/usr/bin/env python
"""Target-label oracle diagnostics for existing Phase1 rejection score tables.

This is diagnostic-only: it uses target query labels to measure whether an
existing scalar rejection score has any threshold that can satisfy the target.
It must not be reported as deployable threshold selection evidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = np.asarray(scores[np.isfinite(scores)], dtype=np.float64)
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    values = np.unique(finite)
    lo = np.nextafter(values[0], -np.inf)
    hi = np.nextafter(values[-1], np.inf)
    return np.concatenate(([lo], values, [hi]))


def _eval_threshold(scores: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed_correct: np.ndarray, threshold: float) -> dict:
    accepted = scores <= float(threshold)
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed_correct = int((known & closed_correct).sum())
    known_correct_after = int((known & closed_correct & accepted).sum())
    known_accepted = int((known & accepted).sum())
    unknown_accepted = int((unknown & accepted).sum())
    known_closed_acc = _safe_rate(known_closed_correct, known_total)
    full_old_acc = _safe_rate(known_correct_after, known_total)
    old_drop_pp = 100.0 * (known_closed_acc - full_old_acc)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    return {
        "threshold": float(threshold),
        "unknown_FAR": unknown_far,
        "old_drop_pp_vs_closed": old_drop_pp,
        "known_closed_accuracy_no_reject": known_closed_acc,
        "known_full_accuracy_after_reject": full_old_acc,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
    }


def _best_rows(path: Path, far_target: float, drop_target: float) -> dict:
    scores = []
    known = []
    unknown = []
    closed_correct = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not (_truthy(row.get("is_known_query")) or _truthy(row.get("is_unknown_query"))):
                continue
            scores.append(float(row["unknown_score"]))
            known.append(_truthy(row.get("is_known_query")))
            unknown.append(_truthy(row.get("is_unknown_query")))
            closed_correct.append(_truthy(row.get("closed_correct_known")))
    scores_a = np.asarray(scores, dtype=np.float64)
    known_a = np.asarray(known, dtype=bool)
    unknown_a = np.asarray(unknown, dtype=bool)
    closed_a = np.asarray(closed_correct, dtype=bool)
    thresholds = _candidate_thresholds(scores_a)
    if thresholds.size == 0:
        return {"oracle_status": "empty_score_table"}
    evaluated = [_eval_threshold(scores_a, known_a, unknown_a, closed_a, t) for t in thresholds]
    dual = [
        r for r in evaluated
        if float(r["unknown_FAR"]) <= far_target and float(r["old_drop_pp_vs_closed"]) <= drop_target
    ]
    old_ok = [r for r in evaluated if float(r["old_drop_pp_vs_closed"]) <= drop_target]
    far_ok = [r for r in evaluated if float(r["unknown_FAR"]) <= far_target]
    if dual:
        best_dual = sorted(dual, key=lambda r: (r["unknown_FAR"], r["old_drop_pp_vs_closed"]))[0]
        status = "dual_possible"
    else:
        best_dual = None
        status = "dual_impossible_for_scalar_score"
    best_far_given_old = sorted(old_ok, key=lambda r: (r["unknown_FAR"], r["old_drop_pp_vs_closed"]))[0] if old_ok else None
    best_old_given_far = sorted(far_ok, key=lambda r: (r["old_drop_pp_vs_closed"], r["unknown_FAR"]))[0] if far_ok else None
    nearest = sorted(
        evaluated,
        key=lambda r: (
            max(0.0, float(r["unknown_FAR"]) - far_target) * 100.0
            + max(0.0, float(r["old_drop_pp_vs_closed"]) - drop_target),
            r["unknown_FAR"],
            r["old_drop_pp_vs_closed"],
        ),
    )[0]
    return {
        "oracle_status": status,
        "dual": best_dual,
        "best_far_given_old": best_far_given_old,
        "best_old_given_far": best_old_given_far,
        "nearest": nearest,
    }


def _flatten(prefix: str, data: dict | None) -> dict:
    if not data:
        return {}
    return {f"{prefix}_{k}": v for k, v in data.items()}


def iter_score_tables(runs_root: Path, run_glob: str, adapters: Iterable[str]) -> Iterable[tuple[str, str, str, Path]]:
    for run_dir in sorted(runs_root.glob(run_glob)):
        for adapter in adapters:
            adapter_dir = run_dir / adapter
            if not adapter_dir.is_dir():
                continue
            for score_table in sorted(adapter_dir.glob("ADAPT3_*/score_table.csv")):
                yield run_dir.name, adapter, score_table.parent.name, score_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    parser.add_argument("--adapters", default="LEOADAPT3_IDENTITY,LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapters = [x.strip() for x in str(args.adapters).split(",") if x.strip()]
    rows = []
    for run_id, adapter, policy, path in iter_score_tables(args.runs_root, args.run_glob, adapters):
        diag = _best_rows(path, float(args.unknown_far_target), float(args.max_old_drop_pp))
        row = {
            "run_id": run_id,
            "adapter": adapter,
            "reject_policy": policy,
            "score_table": str(path),
            "oracle_status": diag.get("oracle_status"),
        }
        row.update(_flatten("dual", diag.get("dual")))
        row.update(_flatten("best_far_given_old", diag.get("best_far_given_old")))
        row.update(_flatten("best_old_given_far", diag.get("best_old_given_far")))
        row.update(_flatten("nearest", diag.get("nearest")))
        rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["run_id", "adapter", "reject_policy"]
    leading = ["run_id", "adapter", "reject_policy", "oracle_status", "score_table"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dual_count = sum(1 for row in rows if row.get("oracle_status") == "dual_possible")
    print({"rows": len(rows), "dual_possible": dual_count, "out_csv": str(args.out_csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
