#!/usr/bin/env python
"""Fuse V3 rejection score tables without retraining features or heads."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


KEY_FIELDS = ["group", "role", "tx_id", "rx_id", "day_id", "sig_id"]
COMPONENTS = {
    "lin": "ADAPT3_LIN_SRC9999/score_table.csv",
    "mlp": "ADAPT3_MLP64_SRC9999/score_table.csv",
    "pcos": "ADAPT3_PROTO_COS_SRC9999/score_table.csv",
    "pmah": "ADAPT3_PROTO_MAH_MIN05/score_table.csv",
}
COMPONENT_SETS = {
    "lin_mlp": ["lin", "mlp"],
    "lin_pcos": ["lin", "pcos"],
    "mlp_pcos": ["mlp", "pcos"],
    "mlp_pmah": ["mlp", "pmah"],
    "lin_mlp_pcos": ["lin", "mlp", "pcos"],
    "lin_mlp_pmah": ["lin", "mlp", "pmah"],
    "all4": ["lin", "mlp", "pcos", "pmah"],
}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _rank_against_source(values: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    src = np.sort(np.asarray(source_values, dtype=np.float64))
    if src.size == 0:
        raise ValueError("empty source calibration values")
    vals = np.asarray(values, dtype=np.float64)
    return np.searchsorted(src, vals, side="right").astype(np.float64) / float(src.size)


def _load_component(adapter_dir: Path, name: str) -> dict[tuple[str, ...], dict]:
    path = adapter_dir / COMPONENTS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, ...], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(str(row[field]) for field in KEY_FIELDS)
            rows[key] = {
                **{field: str(row[field]) for field in KEY_FIELDS},
                "is_known_query": row.get("is_known_query", "0"),
                "is_unknown_query": row.get("is_unknown_query", "0"),
                "closed_correct_known": row.get("closed_correct_known", "0"),
                f"score_{name}": float(row.get("unknown_score", "nan")),
            }
    return rows


def _fuse(ranks: np.ndarray, method: str) -> np.ndarray:
    if method == "max":
        return np.nanmax(ranks, axis=1)
    if method == "mean":
        return np.nanmean(ranks, axis=1)
    if method == "min":
        return np.nanmin(ranks, axis=1)
    if method == "top2mean":
        ordered = np.sort(ranks, axis=1)
        return ordered[:, -2:].mean(axis=1)
    raise ValueError(f"unknown fusion method: {method}")


def _evaluate(
    merged: list[dict],
    component_set: str,
    method: str,
    threshold_policy: str,
    source_q: float,
    proxy_q: float,
) -> dict:
    comps = COMPONENT_SETS[component_set]
    roles = np.asarray([str(row["role"]) for row in merged])
    source_mask = roles == "source"
    proxy_mask = roles == "proxy_unknown"
    ranks = []
    for comp in comps:
        score = np.asarray([float(row[f"score_{comp}"]) for row in merged], dtype=np.float64)
        ranks.append(_rank_against_source(score, score[source_mask]))
    rank_mat = np.stack(ranks, axis=1)
    fused = _fuse(rank_mat, method)
    source_t = float(np.quantile(fused[source_mask], source_q)) if source_mask.any() else float(source_q)
    proxy_t = float(np.quantile(fused[proxy_mask], proxy_q)) if proxy_mask.any() else source_t
    if threshold_policy == "source_accept":
        threshold = source_t
    elif threshold_policy == "min_source_proxy":
        threshold = min(source_t, proxy_t)
    elif threshold_policy == "mean_source_proxy":
        threshold = 0.5 * (source_t + proxy_t)
    else:
        raise ValueError(f"unknown threshold policy: {threshold_policy}")
    accepted = fused <= threshold

    known_mask = np.asarray([_as_bool(row["is_known_query"]) for row in merged], dtype=bool)
    unknown_mask = np.asarray([_as_bool(row["is_unknown_query"]) for row in merged], dtype=bool)
    closed_correct = np.asarray([_as_bool(row["closed_correct_known"]) for row in merged], dtype=bool)
    known_total = int(known_mask.sum())
    unknown_total = int(unknown_mask.sum())
    known_closed_correct = int((known_mask & closed_correct).sum())
    known_correct_after = int((known_mask & closed_correct & accepted).sum())
    known_accepted = int((known_mask & accepted).sum())
    unknown_accepted = int((unknown_mask & accepted).sum())
    known_closed_acc = known_closed_correct / known_total if known_total else np.nan
    known_full_acc = known_correct_after / known_total if known_total else np.nan
    old_drop_pp = 100.0 * (known_closed_acc - known_full_acc) if known_total else np.nan
    unknown_far = unknown_accepted / unknown_total if unknown_total else np.nan
    return {
        "component_set": component_set,
        "fusion_method": method,
        "threshold_policy": threshold_policy,
        "source_accept_quantile": source_q,
        "proxy_far_quantile": proxy_q,
        "threshold": threshold,
        "source_threshold": source_t,
        "proxy_threshold": proxy_t,
        "unknown_FAR": unknown_far,
        "known_closed_accuracy_no_reject": known_closed_acc,
        "known_full_accuracy_after_reject": known_full_acc,
        "old_drop_pp_vs_closed": old_drop_pp,
        "passes_unknown_far_target": bool(unknown_far <= 0.05),
        "passes_old_drop_target": bool(old_drop_pp <= 2.0),
        "passes_dual_target": bool(unknown_far <= 0.05 and old_drop_pp <= 2.0),
        "known_coverage": known_accepted / known_total if known_total else np.nan,
        "known_accepted_accuracy": known_correct_after / known_accepted if known_accepted else np.nan,
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
    }


def _merge_components(adapter_dir: Path, comps: Iterable[str]) -> list[dict]:
    base: dict[tuple[str, ...], dict] | None = None
    for comp in comps:
        df = _load_component(adapter_dir, comp)
        if base is None:
            base = df
        else:
            common = sorted(set(base) & set(df))
            merged = {}
            for key in common:
                row = dict(base[key])
                row[f"score_{comp}"] = df[key][f"score_{comp}"]
                merged[key] = row
            base = merged
    if base is None:
        raise ValueError("empty component set")
    return [base[key] for key in sorted(base)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--adapters", default="LEOADAPT3_IDENTITY,LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    parser.add_argument("--component_sets", default=",".join(COMPONENT_SETS))
    parser.add_argument("--fusion_methods", default="max,mean,min,top2mean")
    parser.add_argument("--threshold_policies", default="source_accept,min_source_proxy,mean_source_proxy")
    parser.add_argument("--source_qs", default="0.95,0.98,0.99,0.995,0.999,0.9999")
    parser.add_argument("--proxy_qs", default="0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.30")
    parser.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapters = [x.strip() for x in str(args.adapters).split(",") if x.strip()]
    component_sets = [x.strip() for x in str(args.component_sets).split(",") if x.strip()]
    fusion_methods = [x.strip() for x in str(args.fusion_methods).split(",") if x.strip()]
    source_qs = [float(x.strip()) for x in str(args.source_qs).split(",") if x.strip()]
    proxy_qs = [float(x.strip()) for x in str(args.proxy_qs).split(",") if x.strip()]
    threshold_policies = [x.strip() for x in str(args.threshold_policies).split(",") if x.strip()]
    rows = []
    for run_dir in sorted(args.runs_root.glob(str(args.run_glob))):
        for adapter in adapters:
            adapter_dir = run_dir / adapter
            if not adapter_dir.is_dir():
                continue
            for component_set in component_sets:
                comps = COMPONENT_SETS[component_set]
                try:
                    merged = _merge_components(adapter_dir, comps)
                except FileNotFoundError:
                    continue
                for method in fusion_methods:
                    for policy in threshold_policies:
                        for source_q in source_qs:
                            pqs = proxy_qs if policy != "source_accept" else [0.05]
                            for proxy_q in pqs:
                                data = _evaluate(merged, component_set, method, policy, source_q, proxy_q)
                                data.update({"run_id": run_dir.name, "adapter": adapter})
                                rows.append(data)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["run_id", "adapter"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "out_csv": str(args.out_csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
