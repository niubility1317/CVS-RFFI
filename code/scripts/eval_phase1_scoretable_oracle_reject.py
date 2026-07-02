#!/usr/bin/env python
"""Oracle separability audit for Phase1 rejection score tables.

This is a diagnostic upper bound only. It scans target-query labels after the
fact to test whether a score table contains any thresholding solution that can
meet the requested old-retention and unknown-FAR targets. Because target query
labels are used, rows from this script must not be reported as deployable
Stage2-A evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _query_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_known: set[str],
    known_roles: set[str],
    unknown_roles: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        role = str(row.get("role", ""))
        tx = canonical_tx_id(row.get("tx_id", ""))
        pred = canonical_tx_id(row.get("pred_tx_id", ""))
        score = _safe_float(row.get("unknown_score"))
        if not math.isfinite(score):
            continue
        is_known = bool(role in known_roles and tx in source_known)
        is_unknown = bool(role in unknown_roles)
        if not (is_known or is_unknown):
            continue
        out.append(
            {
                "role": role,
                "tx_id": tx,
                "pred_tx_id": pred,
                "score": float(score),
                "is_known": is_known,
                "is_unknown": is_unknown,
                "closed_correct": bool(is_known and pred == tx),
            }
        )
    return out


def _metrics_for_accept(query: Sequence[Mapping[str, Any]], accepted: Sequence[bool], *, unknown_far_target: float, max_old_drop_pp: float) -> dict[str, Any]:
    known_total = sum(1 for q in query if q["is_known"])
    unknown_total = sum(1 for q in query if q["is_unknown"])
    known_closed_correct = sum(1 for q in query if q["closed_correct"])
    known_correct_accepted = sum(1 for q, a in zip(query, accepted) if bool(a) and q["closed_correct"])
    known_accepted = sum(1 for q, a in zip(query, accepted) if bool(a) and q["is_known"])
    unknown_accepted = sum(1 for q, a in zip(query, accepted) if bool(a) and q["is_unknown"])
    known_closed_acc = _safe_rate(known_closed_correct, known_total)
    known_full_acc = _safe_rate(known_correct_accepted, known_total)
    old_drop_pp = None
    if known_closed_acc is not None and known_full_acc is not None:
        old_drop_pp = 100.0 * (float(known_closed_acc) - float(known_full_acc))
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    return {
        "known_query_count": int(known_total),
        "unknown_query_count": int(unknown_total),
        "known_closed_correct_count": int(known_closed_correct),
        "known_accepted_count": int(known_accepted),
        "known_correct_accepted_count": int(known_correct_accepted),
        "unknown_accepted_count": int(unknown_accepted),
        "known_closed_accuracy_no_reject": known_closed_acc,
        "known_full_accuracy_after_reject": known_full_acc,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "old_drop_pp_vs_closed": old_drop_pp,
        "unknown_FAR": unknown_far,
        "passes_unknown_far_target": None if unknown_far is None else float(unknown_far) <= float(unknown_far_target),
        "passes_old_drop_target": None if old_drop_pp is None else float(old_drop_pp) <= float(max_old_drop_pp),
        "passes_dual_target": None
        if unknown_far is None or old_drop_pp is None
        else (float(unknown_far) <= float(unknown_far_target) and float(old_drop_pp) <= float(max_old_drop_pp)),
    }


def _global_oracle(query: Sequence[Mapping[str, Any]], *, unknown_far_target: float, max_old_drop_pp: float) -> dict[str, Any]:
    candidates = sorted({float(q["score"]) for q in query})
    if not candidates:
        raise ValueError("no query scores found")
    thresholds = [min(candidates) - 1e-9] + candidates + [max(candidates) + 1e-9]
    best: dict[str, Any] | None = None
    feasible: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [float(q["score"]) <= float(threshold) for q in query]
        m = _metrics_for_accept(query, accepted, unknown_far_target=unknown_far_target, max_old_drop_pp=max_old_drop_pp)
        row = {"oracle_kind": "global_threshold", "threshold": float(threshold), **m}
        if bool(row.get("passes_unknown_far_target")):
            feasible.append(row)
        if best is None:
            best = row
            continue
        # Prefer dual pass, then more old correctness under FAR, then lower FAR.
        key = (
            int(bool(row.get("passes_dual_target"))),
            int(bool(row.get("passes_unknown_far_target"))),
            float(row.get("known_correct_accepted_count") or 0),
            -float(row.get("unknown_accepted_count") or 0),
        )
        best_key = (
            int(bool(best.get("passes_dual_target"))),
            int(bool(best.get("passes_unknown_far_target"))),
            float(best.get("known_correct_accepted_count") or 0),
            -float(best.get("unknown_accepted_count") or 0),
        )
        if key > best_key:
            best = row
    assert best is not None
    best_under_far = max(feasible, key=lambda r: (float(r.get("known_correct_accepted_count") or 0), -float(r.get("unknown_accepted_count") or 0)), default=None)
    return {"best": best, "best_under_far": best_under_far}


def _class_conditional_oracle(query: Sequence[Mapping[str, Any]], *, unknown_far_target: float, max_old_drop_pp: float) -> dict[str, Any]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for q in query:
        by_class[canonical_tx_id(q["pred_tx_id"])].append(q)
    unknown_total = sum(1 for q in query if q["is_unknown"])
    max_unknown_accept = int(math.floor(float(unknown_far_target) * float(unknown_total) + 1e-9))

    choices_by_class: list[tuple[str, list[dict[str, Any]]]] = []
    for cls, items in sorted(by_class.items()):
        scores = sorted({float(q["score"]) for q in items})
        thresholds = [min(scores) - 1e-9] + scores + [max(scores) + 1e-9]
        choices = []
        seen = set()
        for threshold in thresholds:
            accepted = [float(q["score"]) <= float(threshold) for q in items]
            unknown_accept = sum(1 for q, a in zip(items, accepted) if bool(a) and q["is_unknown"])
            known_correct = sum(1 for q, a in zip(items, accepted) if bool(a) and q["closed_correct"])
            known_accept = sum(1 for q, a in zip(items, accepted) if bool(a) and q["is_known"])
            key = (int(unknown_accept), int(known_correct), int(known_accept))
            if key in seen:
                continue
            seen.add(key)
            choices.append(
                {
                    "class": cls,
                    "threshold": float(threshold),
                    "unknown_accept": int(unknown_accept),
                    "known_correct": int(known_correct),
                    "known_accept": int(known_accept),
                }
            )
        choices_by_class.append((cls, choices))

    dp: dict[int, tuple[int, int, dict[str, float]]] = {0: (0, 0, {})}
    for cls, choices in choices_by_class:
        nxt: dict[int, tuple[int, int, dict[str, float]]] = {}
        for used_unknown, (known_correct, known_accept, thresholds) in dp.items():
            for choice in choices:
                new_unknown = int(used_unknown) + int(choice["unknown_accept"])
                if new_unknown > max_unknown_accept:
                    continue
                new_known_correct = int(known_correct) + int(choice["known_correct"])
                new_known_accept = int(known_accept) + int(choice["known_accept"])
                new_thresholds = dict(thresholds)
                new_thresholds[cls] = float(choice["threshold"])
                cur = nxt.get(new_unknown)
                if cur is None or (new_known_correct, new_known_accept) > (cur[0], cur[1]):
                    nxt[new_unknown] = (new_known_correct, new_known_accept, new_thresholds)
        dp = nxt
    if not dp:
        # FAR budget can be zero and all candidate choices may accept unknowns;
        # fall back to rejecting everything.
        dp = {0: (0, 0, {cls: -float("inf") for cls, _ in choices_by_class})}
    best_unknown, (best_known_correct, _best_known_accept, thresholds) = max(
        dp.items(), key=lambda item: (item[1][0], -item[0], item[1][1])
    )
    accepted = [float(q["score"]) <= float(thresholds.get(canonical_tx_id(q["pred_tx_id"]), -float("inf"))) for q in query]
    metrics = _metrics_for_accept(query, accepted, unknown_far_target=unknown_far_target, max_old_drop_pp=max_old_drop_pp)
    return {
        "best": {
            "oracle_kind": "class_conditional_threshold",
            "thresholds_json": json.dumps(thresholds, sort_keys=True),
            "far_budget_unknown_count": int(max_unknown_accept),
            "dp_unknown_count": int(best_unknown),
            "dp_known_correct_count": int(best_known_correct),
            **metrics,
        }
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.score_table_csv)
    source_known = {canonical_tx_id(x) for x in parse_tx_id_list(args.source_tx_ids)}
    query = _query_rows(
        rows,
        source_known=source_known,
        known_roles=_parse_roles(args.known_query_roles),
        unknown_roles=_parse_roles(args.unknown_query_roles),
    )
    if not query:
        raise ValueError("no target query rows found")
    global_result = _global_oracle(query, unknown_far_target=float(args.unknown_far_target), max_old_drop_pp=float(args.max_old_drop_pp))
    class_result = _class_conditional_oracle(query, unknown_far_target=float(args.unknown_far_target), max_old_drop_pp=float(args.max_old_drop_pp))
    base = {
        "phase": "phase1_scoretable_oracle_separability_diagnostic",
        "diagnostic_only": True,
        "not_deployment_evidence_reason": "target query labels are used to choose oracle thresholds",
        "score_table_csv": str(args.score_table_csv),
        "source_tx_ids": sorted(source_known),
        "unknown_far_target": float(args.unknown_far_target),
        "max_old_drop_pp": float(args.max_old_drop_pp),
    }
    results = {
        **base,
        "global_oracle": global_result["best"],
        "global_oracle_best_under_far": global_result["best_under_far"],
        "class_conditional_oracle": class_result["best"],
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_table_csv", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=2.0)
    parser.add_argument("--output_json", default="")
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
