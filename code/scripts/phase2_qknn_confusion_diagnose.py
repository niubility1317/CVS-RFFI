#!/usr/bin/env python3
"""Dump per-class confusion for a Phase2 active support-code qKNN row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _confusion(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for label in labels:
        mask = truth == label
        values, counts = np.unique(pred[mask], return_counts=True)
        pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-int(item[1]), str(item[0])))
        out[str(label)] = {str(value): int(count) for value, count in pairs}
    return out


def _per_class_acc(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {str(label): float(np.mean(pred[truth == label] == label)) for label in labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policy", default="stable_first")
    parser.add_argument("--seed", type=int, default=422947)
    parser.add_argument("--k_old", type=int, default=20)
    parser.add_argument("--k_new", type=int, default=20)
    parser.add_argument("--query_per_old", type=int, default=60)
    parser.add_argument("--query_per_new", type=int, default=60)
    parser.add_argument("--pool_per_old", type=int, default=50)
    parser.add_argument("--pool_per_new", type=int, default=50)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--old_bias", type=float, default=0.0)
    parser.add_argument("--radius_norm", type=float, default=0.2)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_new")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    all_labels = old_labels + new_labels

    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    common: dict[str, Any] = {
        "tx_ids": tx_ids,
        "roles": roles,
        "features": features,
        "scenarios": scenarios,
        "source_probs": source_probs,
        "source_label_to_idx": source_label_to_idx,
        "source_prototypes": source_prototypes,
        "policy": str(args.policy),
        "seed": int(args.seed),
        "exclude_pool_from_query": bool(args.exclude_pool_from_query),
    }
    old_raw = active._build_active_splits(
        labels=old_labels,
        role=str(args.old_role),
        k=int(args.k_old),
        query_per_class=int(args.query_per_old),
        pool_per_class=int(args.pool_per_old),
        **common,
    )
    new_raw = active._build_active_splits(
        labels=new_labels,
        role=str(args.new_role),
        k=int(args.k_new),
        query_per_class=int(args.query_per_new),
        pool_per_class=int(args.pool_per_new),
        **common,
    )
    if set(old_raw) != set(old_labels) or set(new_raw) != set(new_labels):
        raise RuntimeError(
            f"incomplete splits old={sorted(old_raw)} new={sorted(new_raw)} "
            f"expected_old={old_labels} expected_new={new_labels}"
        )

    old_splits = active._as_eval_splits(old_raw)
    new_splits = active._as_eval_splits(new_raw)
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query: list[int] = []
    new_query: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query.extend(query.tolist())
    for label in new_labels:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query.extend(query.tolist())

    bank = qknn._build_support_bank(
        features,
        support_indices,
        support_labels,
        set(old_labels),
        support_scenarios=scenarios[np.asarray(support_indices, dtype=int)] if bool(args.scenario_aware) else None,
    )
    query_idx = np.asarray(old_query + new_query, dtype=int)
    truth = tx_ids[query_idx]
    pred = qknn._predict_from_bank(
        bank,
        features[query_idx],
        topk=int(args.topk),
        old_bias=float(args.old_bias),
        radius_norm=float(args.radius_norm),
        query_scenarios=scenarios[query_idx] if bool(args.scenario_aware) else None,
        scenario_aware=bool(args.scenario_aware),
    )

    scenario_detail: dict[str, dict[str, Any]] = {}
    for label in all_labels:
        scenario_detail[label] = {}
        label_mask = truth == label
        for scenario in sorted({str(value) for value in scenarios[query_idx][label_mask].tolist()}):
            mask = label_mask & (scenarios[query_idx] == scenario)
            scenario_detail[label][scenario] = {
                "acc": float(np.mean(pred[mask] == label)),
                "confusion": _confusion(pred[mask], truth[mask], [label])[label],
            }

    summary = {
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "old_role": str(args.old_role),
        "new_role": str(args.new_role),
        "policy": str(args.policy),
        "seed": int(args.seed),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "pool_per_old": int(args.pool_per_old),
        "pool_per_new": int(args.pool_per_new),
        "topk": int(args.topk),
        "old_bias": float(args.old_bias),
        "radius_norm": float(args.radius_norm),
        "scenario_aware": bool(args.scenario_aware),
        "support_count": int(len(support_indices)),
        "old_query_count": int(len(old_query)),
        "new_query_count": int(len(new_query)),
        "per_class_acc": _per_class_acc(pred, truth, all_labels),
        "confusion": _confusion(pred, truth, all_labels),
        "scenario_detail": scenario_detail,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
