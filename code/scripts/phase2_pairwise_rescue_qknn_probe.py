#!/usr/bin/env python3
"""Probe pairwise prototype rescue on top of Phase2 active support-code qKNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in qknn._parse_csv(raw):
        left, right = item.split(":", 1)
        pairs.append((left.strip(), right.strip()))
    return pairs


def _collect_splits(old_splits, new_splits, old_labels, new_labels):
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
    return support_indices, support_labels, old_query, new_query


def _per_class_acc(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {label: float(np.mean(pred[truth == label] == label)) for label in labels}


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int, old_labels: list[str], new_labels: list[str]) -> dict[str, Any]:
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    old_per = _per_class_acc(old_pred, old_truth, old_labels)
    new_per = _per_class_acc(new_pred, new_truth, new_labels)
    return {
        "old_acc": float(np.mean(old_pred == old_truth)),
        "min_old_class_acc": min(old_per.values()),
        "seen_new_acc": float(np.mean(new_pred == new_truth)),
        "min_seen_new_class_acc": min(new_per.values()),
        "per_old_acc": old_per,
        "per_new_acc": new_per,
        "passes_joint_target": float(np.mean(old_pred == old_truth)) >= 0.80 and min(new_per.values()) >= 0.75,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--pairs", required=True, help="comma-separated left:right class pairs")
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
    pairs = _parse_pairs(args.pairs)

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
        "exclude_pool_from_query": False,
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
    old_splits = active._as_eval_splits(old_raw)
    new_splits = active._as_eval_splits(new_raw)
    support_indices, support_labels, old_query, new_query = _collect_splits(old_splits, new_splits, old_labels, new_labels)
    support_idx = np.asarray(support_indices, dtype=int)
    support_labels_arr = np.asarray(support_labels, dtype=object).astype(str)

    bank = qknn._build_support_bank(
        features,
        support_indices,
        support_labels,
        set(old_labels),
        support_scenarios=scenarios[support_idx] if bool(args.scenario_aware) else None,
    )
    query_idx = np.asarray(old_query + new_query, dtype=int)
    truth = tx_ids[query_idx]
    base_pred = qknn._predict_from_bank(
        bank,
        features[query_idx],
        topk=int(args.topk),
        old_bias=float(args.old_bias),
        radius_norm=float(args.radius_norm),
        query_scenarios=scenarios[query_idx] if bool(args.scenario_aware) else None,
        scenario_aware=bool(args.scenario_aware),
    )

    prototypes: dict[str, np.ndarray] = {}
    for label in all_labels:
        idx = support_idx[support_labels_arr == label]
        prototypes[label] = qknn._normalize_rows(features[idx].mean(axis=0, keepdims=True))[0]
    rescue_pred = base_pred.copy()
    rescue_count = 0
    for left, right in pairs:
        pair_mask = np.isin(base_pred, np.asarray([left, right], dtype=object))
        if not bool(np.any(pair_mask)):
            continue
        q = qknn._normalize_rows(features[query_idx[pair_mask]])
        scores = np.stack([q @ prototypes[left], q @ prototypes[right]], axis=1)
        pair_pred = np.where(scores[:, 0] >= scores[:, 1], left, right).astype(object)
        rescue_count += int(np.sum(pair_pred != rescue_pred[pair_mask]))
        rescue_pred[pair_mask] = pair_pred

    summary = {
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "pairs": pairs,
        "seed": int(args.seed),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "base": _metrics(base_pred, truth, len(old_query), old_labels, new_labels),
        "pairwise_rescue": _metrics(rescue_pred, truth, len(old_query), old_labels, new_labels),
        "rescue_changed_predictions": rescue_count,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
