#!/usr/bin/env python3
"""Probe transductive prototype-refined qKNN for Phase2-C many-new stability.

The deployed memory remains a compressed support-code bank plus per-class
prototype statistics. At inference, qKNN scores initialize a closed-set
balanced assignment over the unlabeled target query batch. Class prototypes are
then refined with the assigned query centroids and used to rescore the same
batch. Query labels are never used for fitting or selection in the algorithm;
they are used only for the reported audit metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_confusion_aware_qknn_probe as base
import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def _support_prototypes(
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    class_labels: list[str],
) -> np.ndarray:
    support = qknn._normalize_rows(features[support_indices])
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[support_labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    return np.stack(prototypes, axis=0)


def _prototype_scores(
    features: np.ndarray,
    query_indices: np.ndarray,
    prototypes: np.ndarray,
    old_labels: list[str],
    old_bias: float,
) -> np.ndarray:
    query = qknn._normalize_rows(features[query_indices])
    scores = query @ qknn._normalize_rows(prototypes).T
    if float(old_bias) != 0.0:
        scores[:, : len(old_labels)] += float(old_bias)
    return scores


def _refine_scores(
    *,
    features: np.ndarray,
    query_indices: np.ndarray,
    base_scores: np.ndarray,
    support_proto: np.ndarray,
    old_labels: list[str],
    new_labels: list[str],
    old_count: int,
    query_mix: float,
    score_mix: float,
    iterations: int,
    proto_old_bias: float,
    confidence_margin: float,
) -> np.ndarray:
    class_labels = old_labels + new_labels
    query = qknn._normalize_rows(features[query_indices])
    current_scores = np.asarray(base_scores, dtype=np.float64).copy()
    prototypes = qknn._normalize_rows(support_proto)
    for _ in range(max(0, int(iterations))):
        pred = base._balanced_predict(
            current_scores,
            old_count=int(old_count),
            old_labels=old_labels,
            new_labels=new_labels,
        )
        assigned = np.asarray([class_labels.index(str(label)) for label in pred.tolist()], dtype=int)
        if float(confidence_margin) > 0.0:
            sorted_scores = np.sort(current_scores, axis=1)
            confident = (sorted_scores[:, -1] - sorted_scores[:, -2]) >= float(confidence_margin)
        else:
            confident = np.ones(query.shape[0], dtype=bool)
        refined = prototypes.copy()
        for class_index, _label in enumerate(class_labels):
            mask = (assigned == class_index) & confident
            if not bool(np.any(mask)):
                continue
            query_proto = qknn._normalize_rows(query[mask].mean(axis=0, keepdims=True))[0]
            refined[class_index] = qknn._normalize_rows(
                (
                    (1.0 - float(query_mix)) * prototypes[class_index]
                    + float(query_mix) * query_proto
                ).reshape(1, -1)
            )[0]
        proto_scores = _prototype_scores(
            features,
            query_indices,
            refined,
            old_labels,
            old_bias=float(proto_old_bias),
        )
        current_scores = (1.0 - float(score_mix)) * base_scores + float(score_mix) * proto_scores
        prototypes = refined
    return current_scores


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "support_selection_policy",
        "k_old",
        "k_new",
        "topm",
        "proto_mix",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "query_mix",
        "score_mix",
        "iterations",
        "proto_old_bias",
        "confidence_margin",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_new_floor75",
        "query_passes_joint_target",
        "query_per_old_acc",
        "query_per_new_acc",
        "stored_quantized_support_code_count",
        "stored_raw_support_count",
        "stored_class_prototype_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--seed_start", type=int, default=421000)
    parser.add_argument("--seed_count", type=int, default=120)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=70)
    parser.add_argument("--query_per_new", type=int, default=70)
    parser.add_argument("--pool_per_old", type=int, default=10)
    parser.add_argument("--pool_per_new", type=int, default=10)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--topm", type=int, default=4)
    parser.add_argument("--proto_mix", type=float, default=0.25)
    parser.add_argument("--radius_norm", type=float, default=0.0)
    parser.add_argument("--old_bias", type=float, default=0.001)
    parser.add_argument("--neg_lambda", type=float, default=0.7)
    parser.add_argument("--neg_threshold", type=float, default=0.75)
    parser.add_argument("--neg_margin", type=float, default=0.01)
    parser.add_argument("--mutual_only", action="store_true")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--query_mix_grid", default="0,0.1,0.2,0.35,0.5")
    parser.add_argument("--score_mix_grid", default="0,0.15,0.3,0.5,0.7")
    parser.add_argument("--iterations_grid", default="1,2,3")
    parser.add_argument("--proto_old_bias_grid", default="0,0.001,0.005")
    parser.add_argument("--confidence_margin_grid", default="0")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--old_floor", type=float, default=0.75)
    parser.add_argument("--seen_new_target", type=float, default=0.75)
    parser.add_argument("--seen_new_floor", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    class_labels = old_labels + new_labels
    old_label_set = set(old_labels)
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    rows: list[dict[str, Any]] = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.seed_count)):
        for policy in qknn._parse_csv(args.policies):
            common = {
                "tx_ids": tx_ids,
                "roles": roles,
                "features": features,
                "scenarios": scenarios,
                "source_probs": source_probs,
                "source_label_to_idx": source_label_to_idx,
                "source_prototypes": source_prototypes,
                "policy": policy,
                "seed": seed,
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
                continue
            support_indices, support_labels, old_query, new_query = base._collect(
                active._as_eval_splits(old_raw),
                active._as_eval_splits(new_raw),
                old_labels,
                new_labels,
            )
            query_indices = np.concatenate([old_query, new_query])
            truth = tx_ids[query_indices]
            base_scores, _, _ = base._class_scores(
                features=features,
                support_indices=support_indices,
                support_labels=support_labels,
                query_indices=query_indices,
                scenarios=scenarios,
                class_labels=class_labels,
                old_labels=old_label_set,
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
            support_proto = _support_prototypes(features, support_indices, support_labels, class_labels)
            for query_mix in _parse_float_csv(args.query_mix_grid):
                for score_mix in _parse_float_csv(args.score_mix_grid):
                    for iterations in _parse_int_csv(args.iterations_grid):
                        for proto_old_bias in _parse_float_csv(args.proto_old_bias_grid):
                            for confidence_margin in _parse_float_csv(args.confidence_margin_grid):
                                scores = _refine_scores(
                                    features=features,
                                    query_indices=query_indices,
                                    base_scores=base_scores,
                                    support_proto=support_proto,
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                    old_count=int(old_query.size),
                                    query_mix=float(query_mix),
                                    score_mix=float(score_mix),
                                    iterations=int(iterations),
                                    proto_old_bias=float(proto_old_bias),
                                    confidence_margin=float(confidence_margin),
                                )
                                pred = base._balanced_predict(
                                    scores,
                                    old_count=int(old_query.size),
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                )
                                metrics = base._metrics(
                                    pred,
                                    truth,
                                    old_count=int(old_query.size),
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                    old_target=float(args.old_target),
                                    old_floor=float(args.old_floor),
                                    new_target=float(args.seen_new_target),
                                    new_floor=float(args.seen_new_floor),
                                )
                                row: dict[str, Any] = {
                                    "seed": int(seed),
                                    "support_selection_policy": policy,
                                    "k_old": int(args.k_old),
                                    "k_new": int(args.k_new),
                                    "topm": int(args.topm),
                                    "proto_mix": float(args.proto_mix),
                                    "radius_norm": float(args.radius_norm),
                                    "old_bias": float(args.old_bias),
                                    "neg_lambda": float(args.neg_lambda),
                                    "neg_threshold": float(args.neg_threshold),
                                    "neg_margin": float(args.neg_margin),
                                    "mutual_only": bool(args.mutual_only),
                                    "scenario_aware": bool(args.scenario_aware),
                                    "query_mix": float(query_mix),
                                    "score_mix": float(score_mix),
                                    "iterations": int(iterations),
                                    "proto_old_bias": float(proto_old_bias),
                                    "confidence_margin": float(confidence_margin),
                                    "stored_quantized_support_code_count": int(support_indices.size),
                                    "stored_raw_support_count": 0,
                                    "stored_class_prototype_count": int(len(class_labels)),
                                }
                                row.update({f"query_{key}": value for key, value in metrics.items()})
                                row["query_rank_score"] = base._rank(metrics)
                                rows.append(row)

    rows.sort(key=lambda row: tuple(row["query_rank_score"]), reverse=True)
    summary = {
        "diagnostic_scope": "TRANSDUCTIVE_PROTO_REFINED_QKNN_NO_QUERY_LABEL_FIT",
        "selection_note": "Unlabeled query features refine prototypes after qKNN initialization; query labels are audit-only.",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows_count": int(len(rows)),
        "best_by_query": rows[:20],
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(Path(args.output_csv), rows)
    print(json.dumps({"rows_count": len(rows), "best_by_query": rows[:3], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
