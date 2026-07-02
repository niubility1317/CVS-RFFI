#!/usr/bin/env python
"""Class-conditional recalibration for Phase1 rejection score tables.

This reads score tables produced by eval_phase1_multiview_reject.py and
recomputes accept/reject decisions with thresholds calibrated by predicted old
class. Only source-old and proxy-unknown rows are used for thresholds; target
old and target unknown rows remain evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _quantile(values: Sequence[float], q: float, fallback: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float(fallback)
    return float(np.quantile(np.asarray(vals, dtype=np.float64), float(q)))


def _class_thresholds(
    rows: Sequence[Mapping[str, str]],
    *,
    source_known: set[str],
    train_known_roles: set[str],
    proxy_roles: set[str],
    source_quantile: float,
    proxy_quantile: float,
    policy: str,
    correct_source_only: bool,
    max_source_quantile: float,
) -> dict[str, Any]:
    source_by_pred: dict[str, list[float]] = defaultdict(list)
    proxy_by_pred: dict[str, list[float]] = defaultdict(list)
    source_all: list[float] = []
    proxy_all: list[float] = []
    classes: set[str] = set()
    for row in rows:
        role = str(row.get("role", ""))
        tx = canonical_tx_id(row.get("tx_id", ""))
        pred = canonical_tx_id(row.get("pred_tx_id", ""))
        score = _safe_float(row.get("unknown_score"))
        if not math.isfinite(score):
            continue
        classes.add(pred)
        if role in train_known_roles and tx in source_known:
            if correct_source_only and pred != tx:
                continue
            source_by_pred[pred].append(score)
            source_all.append(score)
        if role in proxy_roles:
            proxy_by_pred[pred].append(score)
            proxy_all.append(score)

    if not source_all:
        raise ValueError("no source calibration rows found in score table")
    if not proxy_all:
        raise ValueError("no proxy calibration rows found in score table")

    global_source = _quantile(source_all, float(source_quantile), fallback=max(source_all))
    global_proxy = _quantile(proxy_all, float(proxy_quantile), fallback=max(proxy_all))
    global_source_cap = _quantile(source_all, float(max_source_quantile), fallback=max(source_all))
    policy_text = str(policy)
    thresholds: dict[str, float] = {}
    detail: dict[str, dict[str, Any]] = {}
    for cls in sorted(classes):
        source_t = _quantile(source_by_pred.get(cls, []), float(source_quantile), fallback=global_source)
        proxy_t = _quantile(proxy_by_pred.get(cls, []), float(proxy_quantile), fallback=global_proxy)
        source_cap = _quantile(source_by_pred.get(cls, []), float(max_source_quantile), fallback=global_source_cap)
        source_t = min(float(source_t), float(source_cap))
        if policy_text == "source_class_accept":
            threshold = source_t
        elif policy_text == "proxy_class_far":
            threshold = proxy_t
        elif policy_text == "min_class_source_proxy":
            threshold = min(source_t, proxy_t)
        elif policy_text == "max_class_source_proxy":
            threshold = max(source_t, proxy_t)
        else:
            raise ValueError(f"unknown threshold_policy={policy!r}")
        thresholds[cls] = float(threshold)
        detail[cls] = {
            "source_threshold": float(source_t),
            "proxy_threshold": float(proxy_t),
            "threshold": float(threshold),
            "source_count": int(len(source_by_pred.get(cls, []))),
            "proxy_count": int(len(proxy_by_pred.get(cls, []))),
        }
    return {
        "threshold_policy": policy_text,
        "source_quantile": float(source_quantile),
        "proxy_quantile": float(proxy_quantile),
        "max_source_quantile": float(max_source_quantile),
        "correct_source_only": bool(correct_source_only),
        "global_source_threshold": float(global_source),
        "global_proxy_threshold": float(global_proxy),
        "class_thresholds": detail,
        "threshold_map": thresholds,
        "source_calibration_count": int(len(source_all)),
        "proxy_calibration_count": int(len(proxy_all)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows_in = _read_rows(args.score_table_csv)
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}
    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    train_roles = _parse_roles(args.train_known_roles)
    proxy_roles = _parse_roles(args.proxy_unknown_roles)
    threshold_info = _class_thresholds(
        rows_in,
        source_known=source_known,
        train_known_roles=train_roles,
        proxy_roles=proxy_roles,
        source_quantile=float(args.source_quantile),
        proxy_quantile=float(args.proxy_quantile),
        policy=str(args.threshold_policy),
        correct_source_only=bool(args.correct_source_only),
        max_source_quantile=float(args.max_source_quantile),
    )
    th_map: dict[str, float] = threshold_info["threshold_map"]
    global_fallback = float(threshold_info["global_source_threshold"])

    rows_out: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    for i, row in enumerate(rows_in):
        role = str(row.get("role", ""))
        tx = canonical_tx_id(row.get("tx_id", ""))
        pred = canonical_tx_id(row.get("pred_tx_id", ""))
        score = _safe_float(row.get("unknown_score"))
        threshold = float(th_map.get(pred, global_fallback))
        accepted = bool(score <= threshold)
        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and pred == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            known_correct_accepted += int(accepted and closed_correct)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(float(score))
            accepted_flags.append(bool(accepted))
        out = dict(row)
        out.update(
            {
                "classcond_group": i,
                "classcond_threshold": f"{threshold:.8f}",
                "classcond_accepted": int(accepted),
                "classcond_closed_correct_known": int(closed_correct),
                "classcond_accepted_correct_known": int(accepted and closed_correct),
            }
        )
        rows_out.append(out)

    known_closed_accuracy = _safe_rate(known_closed_correct, known_total)
    known_full_accuracy = _safe_rate(known_correct_accepted, known_total)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    old_drop_pp = None
    if known_closed_accuracy is not None and known_full_accuracy is not None:
        old_drop_pp = 100.0 * (float(known_closed_accuracy) - float(known_full_accuracy))
    metrics = {
        "phase": "phase1_only_scoretable_class_conditional_reject",
        "threshold_scope": "source_old_and_source_proxy_unknown_only_no_target_support_no_unknown_query_tuning",
        "score_table_csv": str(args.score_table_csv),
        "source_tx_ids": source_tx_ids,
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "threshold": {k: v for k, v in threshold_info.items() if k != "threshold_map"},
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": known_closed_accuracy,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy_after_reject": known_full_accuracy,
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "old_drop_pp_vs_closed": old_drop_pp,
        "max_old_drop_pp": float(args.max_old_drop_pp),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": unknown_far,
        "unknown_reject_rate": None if unknown_total == 0 else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "passes_unknown_far_target": None if unknown_far is None else float(unknown_far) <= float(args.unknown_far_target),
        "passes_old_drop_target": None if old_drop_pp is None else float(old_drop_pp) <= float(args.max_old_drop_pp),
        "passes_dual_target": None
        if unknown_far is None or old_drop_pp is None
        else (float(unknown_far) <= float(args.unknown_far_target) and float(old_drop_pp) <= float(args.max_old_drop_pp)),
    }
    if y_unknown:
        metrics.update(
            binary_reject_metrics(
                torch.tensor(y_unknown, dtype=torch.bool),
                torch.tensor(reject_scores, dtype=torch.float32),
                torch.tensor(accepted_flags, dtype=torch.bool),
            )
        )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_score_table_csv:
        _write_rows(args.output_score_table_csv, rows_out)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_table_csv", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--train_known_roles", default="source")
    parser.add_argument("--proxy_unknown_roles", default="proxy_unknown")
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument(
        "--threshold_policy",
        default="min_class_source_proxy",
        choices=["source_class_accept", "proxy_class_far", "min_class_source_proxy", "max_class_source_proxy"],
    )
    parser.add_argument("--source_quantile", type=float, default=0.999)
    parser.add_argument("--proxy_quantile", type=float, default=0.05)
    parser.add_argument("--max_source_quantile", type=float, default=1.0)
    parser.add_argument("--correct_source_only", action="store_true")
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=2.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
