#!/usr/bin/env python
"""Diagnose source-calibrated old-class rescue on SATPHY11 score tables."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = float(q) * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _eval_accept(rows: list[dict[str, str]], accepted: list[bool]) -> tuple[float, float, float]:
    known_idx = [i for i, row in enumerate(rows) if _int(row, "is_known_query") == 1]
    unknown_idx = [i for i, row in enumerate(rows) if _int(row, "is_unknown_query") == 1]
    closed_correct = sum(_int(rows[i], "closed_correct_known") for i in known_idx)
    known_correct_accepted = sum(_int(rows[i], "closed_correct_known") for i in known_idx if accepted[i])
    unknown_accepted = sum(1 for i in unknown_idx if accepted[i])
    known_closed_accuracy = closed_correct / max(1, len(known_idx))
    known_full_accuracy = known_correct_accepted / max(1, len(known_idx))
    unknown_far = unknown_accepted / max(1, len(unknown_idx))
    known_coverage = sum(1 for i in known_idx if accepted[i]) / max(1, len(known_idx))
    return unknown_far, 100.0 * (known_closed_accuracy - known_full_accuracy), known_coverage


def _format_table(rows: list[dict[str, object]], cols: list[str], *, limit: int = 25) -> str:
    view = rows[:limit]
    widths = {col: len(col) for col in cols}
    rendered: list[dict[str, str]] = []
    for row in view:
        item: dict[str, str] = {}
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            item[col] = text
            widths[col] = max(widths[col], len(text))
        rendered.append(item)
    lines = [" ".join(col.ljust(widths[col]) for col in cols)]
    lines.append(" ".join("-" * widths[col] for col in cols))
    for row in rendered:
        lines.append(" ".join(row[col].ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def main() -> int:
    base = Path("/home/szu2070436088/2510044040/CV-SincNet/runs")
    paths = sorted(base.glob("phase1_adv3b02_satphysmv11_*_20260702/SATPHY11_*/score_table.csv"))
    quantiles = [0.001, 0.01, 0.05, 0.10, 0.20]
    out: list[dict[str, object]] = []
    for path in paths:
        rows = _read_rows(path)
        source = [row for row in rows if row.get("role") == "source"]
        if not source:
            continue
        original_accepted = [_int(row, "accepted") == 1 for row in rows]
        original_far, original_drop, original_coverage = _eval_accept(rows, original_accepted)
        if original_far > 0.15:
            continue
        best: tuple[float, float, float, float, float, float, float] | None = None
        source_conf = [_float(row, "mean_confidence") for row in source]
        source_vote = [_float(row, "vote_frac") for row in source]
        source_cosine = [_float(row, "mean_cosine") for row in source]
        conf = [_float(row, "mean_confidence") for row in rows]
        vote = [_float(row, "vote_frac") for row in rows]
        cosine = [_float(row, "mean_cosine") for row in rows]
        for conf_q in quantiles:
            conf_t = _quantile(source_conf, conf_q)
            for vote_q in quantiles:
                vote_t = _quantile(source_vote, vote_q)
                for cosine_q in quantiles:
                    cosine_t = _quantile(source_cosine, cosine_q)
                    rescue = [c >= conf_t and v >= vote_t and m >= cosine_t for c, v, m in zip(conf, vote, cosine)]
                    accepted = [bool(a or r) for a, r in zip(original_accepted, rescue)]
                    far, drop, coverage = _eval_accept(rows, accepted)
                    score = max(0.0, far - 0.05) + max(0.0, drop - 2.0) / 100.0
                    cand = (score, far, drop, coverage, conf_q, vote_q, cosine_q)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            continue
        out.append(
            {
                "run_id": path.parent.parent.name,
                "policy": path.parent.name,
                "orig_far": original_far,
                "orig_drop": original_drop,
                "orig_coverage": original_coverage,
                "best_score": best[0],
                "best_far": best[1],
                "best_drop": best[2],
                "best_coverage": best[3],
                "conf_q": best[4],
                "vote_q": best[5],
                "cosine_q": best[6],
            }
        )
    print({"score_tables": len(paths), "rows": len(out)})
    dual_count = sum(1 for row in out if float(row["best_far"]) <= 0.05 and float(row["best_drop"]) <= 2.0)
    far_count = sum(1 for row in out if float(row["best_far"]) <= 0.05)
    drop_count = sum(1 for row in out if float(row["best_drop"]) <= 2.0)
    print({"dual_pass_after_rescue": dual_count, "far_pass_after_rescue": far_count, "drop_pass_after_rescue": drop_count})
    cols = [
        "run_id",
        "policy",
        "orig_far",
        "orig_drop",
        "best_far",
        "best_drop",
        "best_coverage",
        "conf_q",
        "vote_q",
        "cosine_q",
        "best_score",
    ]
    best_rows = sorted(out, key=lambda row: (float(row["best_score"]), float(row["best_far"]), float(row["best_drop"])))
    print(_format_table(best_rows, cols, limit=25))
    by_policy: dict[str, list[dict[str, object]]] = {}
    for row in out:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    agg: list[dict[str, object]] = []
    for policy, rows in by_policy.items():
        agg.append(
            {
                "policy": policy,
                "mean_far": mean(float(row["best_far"]) for row in rows),
                "max_far": max(float(row["best_far"]) for row in rows),
                "mean_drop": mean(float(row["best_drop"]) for row in rows),
                "max_drop": max(float(row["best_drop"]) for row in rows),
            }
        )
    agg = sorted(agg, key=lambda row: (float(row["mean_far"]), float(row["mean_drop"])))
    print(_format_table(agg, ["policy", "mean_far", "max_far", "mean_drop", "max_drop"], limit=50))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
