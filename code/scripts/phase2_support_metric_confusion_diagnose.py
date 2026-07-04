#!/usr/bin/env python3
"""Dump confusion for one support-metric compressed qKNN configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn
import phase2_support_metric_qknn_probe as metric_qknn


def _confusion(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for label in labels:
        mask = truth == label
        values, counts = np.unique(pred[mask], return_counts=True)
        pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-int(item[1]), str(item[0])))
        out[label] = {str(value): int(count) for value, count in pairs}
    return out


def _per_class_acc(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {label: qknn._accuracy(pred[truth == label], truth[truth == label]) for label in labels}


def _collect(
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        np.asarray(support_indices, dtype=int),
        np.asarray(support_labels, dtype=object).astype(str),
        np.asarray(old_query, dtype=int),
        np.asarray(new_query, dtype=int),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--policy", default="stable_first")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_old", type=int, required=True)
    parser.add_argument("--k_new", type=int, required=True)
    parser.add_argument("--query_per_old", type=int, required=True)
    parser.add_argument("--query_per_new", type=int, required=True)
    parser.add_argument("--pool_per_old", type=int, required=True)
    parser.add_argument("--pool_per_new", type=int, required=True)
    parser.add_argument("--transform_mode", default="identity")
    parser.add_argument("--transform_strength", type=float, default=0.0)
    parser.add_argument("--topm", type=int, default=4)
    parser.add_argument("--proto_mix", type=float, default=0.25)
    parser.add_argument("--radius_norm", type=float, default=0.0)
    parser.add_argument("--old_bias", type=float, default=0.0)
    parser.add_argument("--neg_lambda", type=float, default=0.0)
    parser.add_argument("--neg_threshold", type=float, default=0.75)
    parser.add_argument("--neg_margin", type=float, default=0.0)
    parser.add_argument("--mutual_only", action="store_true")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--balanced_assignment", action="store_true")
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
        raise RuntimeError(f"incomplete split old={sorted(old_raw)} new={sorted(new_raw)}")
    old_splits = active._as_eval_splits(old_raw)
    new_splits = active._as_eval_splits(new_raw)
    support_indices, support_labels, old_query, new_query = _collect(old_splits, new_splits, old_labels, new_labels)
    query_indices = np.asarray(old_query.tolist() + new_query.tolist(), dtype=int)
    old_count = int(old_query.size)

    transform = metric_qknn.metric._fit_transform(
        features[support_indices],
        support_labels,
        str(args.transform_mode),
        float(args.transform_strength),
    )
    adapted = metric_qknn.metric._apply_transform(features, transform)
    scores, radii, proto_sim = metric_qknn.base._class_scores(
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        scenarios=scenarios,
        class_labels=all_labels,
        old_labels=set(old_labels),
        topm=int(args.topm),
        proto_mix=float(args.proto_mix),
        radius_norm=float(args.radius_norm),
        old_bias=float(args.old_bias),
        neg_lambda=float(args.neg_lambda),
        neg_threshold=float(args.neg_threshold),
        neg_margin=float(args.neg_margin),
        mutual_only=bool(args.mutual_only),
        scenario_aware=bool(args.scenario_aware),
    )
    if bool(args.balanced_assignment):
        pred = metric_qknn.base._balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    else:
        pred = metric_qknn.base._predict(scores, all_labels)
    truth = tx_ids[query_indices]
    per_old = _per_class_acc(pred[:old_count], truth[:old_count], old_labels)
    per_new = _per_class_acc(pred[old_count:], truth[old_count:], new_labels)
    close_pairs: list[dict[str, Any]] = []
    for i, left in enumerate(all_labels):
        for j, right in enumerate(all_labels):
            if j <= i:
                continue
            close_pairs.append({"left": left, "right": right, "prototype_similarity": float(proto_sim[i, j])})
    close_pairs.sort(key=lambda row: row["prototype_similarity"], reverse=True)
    summary = {
        "feature_npz": str(args.feature_npz),
        "seed": int(args.seed),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "support_selection_policy": str(args.policy),
        "config": {
            "transform_mode": str(args.transform_mode),
            "transform_strength": float(args.transform_strength),
            "topm": int(args.topm),
            "proto_mix": float(args.proto_mix),
            "radius_norm": float(args.radius_norm),
            "old_bias": float(args.old_bias),
            "neg_lambda": float(args.neg_lambda),
            "neg_threshold": float(args.neg_threshold),
            "neg_margin": float(args.neg_margin),
            "mutual_only": bool(args.mutual_only),
            "scenario_aware": bool(args.scenario_aware),
            "balanced_assignment": bool(args.balanced_assignment),
        },
        "old_acc": qknn._accuracy(pred[:old_count], truth[:old_count]),
        "new_acc": qknn._accuracy(pred[old_count:], truth[old_count:]),
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "confusion": _confusion(pred, truth, all_labels),
        "class_radii": radii,
        "top_prototype_similarity_pairs": close_pairs[:30],
        "stored_quantized_support_code_count": int(support_indices.size),
        "stored_raw_support_count": 0,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
