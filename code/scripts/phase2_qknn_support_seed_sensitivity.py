#!/usr/bin/env python3
"""Support-seed sensitivity audit for Stage2-C scenario-aware qKNN.

This diagnostic keeps the deployed head unchanged and varies only the K-shot
support/query split seed. It is useful for active-enrollment planning, but a
best seed selected with query labels must be reported as support sensitivity
evidence rather than as an independent blind deployment result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_source_guarded_qknn_sweep as qknn


def _evaluate_seed(
    seed: int,
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    logits: np.ndarray,
    scenarios: np.ndarray,
    old_labels: list[str],
    new_labels: list[str],
    k_old: int,
    k_new: int,
    query_per_old: int,
    query_per_new: int,
    topk: int,
    old_bias: float,
    radius_norm: float,
    scenario_aware: bool,
    old_target: float,
    old_floor: float,
    seen_new_target: float,
    seen_new_floor: float,
) -> dict[str, Any]:
    old_label_array = np.asarray(old_labels, dtype=object)
    source_idx = np.argmax(logits, axis=1)
    source_label = old_label_array[source_idx]
    source_conf, source_margin = qknn._softmax_confidence(logits)
    old_splits = qknn._prepare_class_splits(
        tx_ids, roles, old_labels, "target_old", k_old, query_per_old, seed
    )
    new_splits = qknn._prepare_class_splits(
        tx_ids, roles, new_labels, "target_new", k_new, query_per_new, seed
    )
    if set(old_splits) != set(old_labels):
        missing = sorted(set(old_labels) - set(old_splits))
        raise RuntimeError(f"Missing old splits for seed {seed}: {missing}")
    if set(new_splits) != set(new_labels):
        missing = sorted(set(new_labels) - set(new_splits))
        raise RuntimeError(f"Missing new splits for seed {seed}: {missing}")
    return qknn._evaluate_row(
        tuple(new_labels),
        features=features,
        tx_ids=tx_ids,
        source_label=source_label,
        source_conf=source_conf,
        source_margin=source_margin,
        scenarios=scenarios,
        old_splits=old_splits,
        new_splits=new_splits,
        old_labels=old_labels,
        topk=topk,
        old_bias=old_bias,
        radius_norm=radius_norm,
        source_guard_mode="none",
        source_conf_min=0.0,
        source_margin_min=0.0,
        scenario_aware=scenario_aware,
        old_target=old_target,
        old_floor=old_floor,
        new_target=seen_new_target,
        new_floor=seen_new_floor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--seed_start", type=int, default=422001)
    parser.add_argument("--seed_count", type=int, default=1000)
    parser.add_argument("--k_old", type=int, default=5)
    parser.add_argument("--k_new", type=int, default=5)
    parser.add_argument("--query_per_old", type=int, default=75)
    parser.add_argument("--query_per_new", type=int, default=75)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--old_bias", type=float, default=0.0)
    parser.add_argument("--radius_norm", type=float, default=0.0)
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.88)
    parser.add_argument("--old_floor", type=float, default=0.80)
    parser.add_argument("--seen_new_target", type=float, default=0.85)
    parser.add_argument("--seen_new_floor", type=float, default=0.80)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)

    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        row = _evaluate_seed(
            seed,
            features=features,
            tx_ids=tx_ids,
            roles=roles,
            logits=logits,
            scenarios=scenarios,
            old_labels=old_labels,
            new_labels=new_labels,
            k_old=args.k_old,
            k_new=args.k_new,
            query_per_old=args.query_per_old,
            query_per_new=args.query_per_new,
            topk=args.topk,
            old_bias=args.old_bias,
            radius_norm=args.radius_norm,
            scenario_aware=bool(args.scenario_aware),
            old_target=args.old_target,
            old_floor=args.old_floor,
            seen_new_target=args.seen_new_target,
            seen_new_floor=args.seen_new_floor,
        )
        row["seed"] = int(seed)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            min(
                row["old_acc"] / args.old_target,
                row["min_old_class_acc"] / args.old_floor,
                row["seen_new_acc"] / args.seen_new_target,
                row["min_seen_new_class_acc"] / args.seen_new_floor,
            ),
            row["old_acc"],
            row["seen_new_acc"],
        ),
        reverse=True,
    )
    summary = {
        "diagnostic_scope": "support_seed_sensitivity_not_blind_model_selection",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "method": (
            f"source_guarded_qknn8_k{args.topk}_oldbias{args.old_bias:g}"
            f"_rnorm{args.radius_norm:g}_sgnone_c0_m0"
            f"{'_scenario' if args.scenario_aware else ''}"
        ),
        "old_target": float(args.old_target),
        "old_floor": float(args.old_floor),
        "seen_new_target": float(args.seen_new_target),
        "seen_new_floor": float(args.seen_new_floor),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "best": rows[:20],
        "seed_rows": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "seed",
        "method",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "support_count",
        "old_query_count",
        "new_query_count",
        "stored_quantized_count",
        "stored_support_count",
        "scenario_aware",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)

    print(
        json.dumps(
            {
                "joint_pass_count": summary["joint_pass_count"],
                "best": rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
