#!/usr/bin/env python3
"""Probe support-only confusion-aware qKNN heads for Phase2-C many-new.

The deployed state is still a compressed support-code bank plus small class
prototype statistics. Calibration uses target support leave-one-out only; query
labels are reported only as an audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


Split = tuple[np.ndarray, np.ndarray]


def _collect(
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
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


def _topm_mean(scores: np.ndarray, topm: int) -> np.ndarray:
    k = max(1, min(int(topm), int(scores.shape[1])))
    part = np.partition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
    return np.mean(part, axis=1)


def _class_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    scenarios: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    topm: int,
    proto_mix: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if scenario_aware:
        query_scenarios = np.asarray(scenarios[query_indices], dtype=object).astype(str)
        support_scenarios = np.asarray(scenarios[support_indices], dtype=object).astype(str)
        out = np.full((query_indices.size, len(class_labels)), -1e9, dtype=np.float64)
        radii: dict[str, float] = {}
        proto_sim = np.zeros((len(class_labels), len(class_labels)), dtype=np.float64)
        for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
            query_mask = query_scenarios == scenario
            support_mask = support_scenarios == scenario
            if int(np.sum(support_mask)) < max(1, int(topm)) or len(set(support_labels[support_mask].tolist())) < 2:
                support_mask = np.ones_like(support_mask, dtype=bool)
            sub_scores, sub_radii, sub_proto_sim = _class_scores(
                features=features,
                support_indices=support_indices[support_mask],
                support_labels=support_labels[support_mask],
                query_indices=query_indices[query_mask],
                scenarios=scenarios,
                class_labels=class_labels,
                old_labels=old_labels,
                topm=topm,
                proto_mix=proto_mix,
                radius_norm=radius_norm,
                old_bias=old_bias,
                neg_lambda=neg_lambda,
                neg_threshold=neg_threshold,
                neg_margin=neg_margin,
                mutual_only=mutual_only,
                scenario_aware=False,
            )
            out[query_mask] = sub_scores
            radii.update(sub_radii)
            proto_sim = sub_proto_sim
        return out, radii, proto_sim

    query = qknn._normalize_rows(features[query_indices])
    support = qknn._normalize_rows(features[support_indices])
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    pos_scores: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[support_labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
            radii.append(1.0)
            pos_scores.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        prototype = qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0]
        prototypes.append(prototype)
        radii.append(float(np.mean(1.0 - class_support @ prototype)))
        sims = query @ class_support.T
        local = _topm_mean(sims, int(topm))
        proto = query @ prototype
        mixed = (1.0 - float(proto_mix)) * local + float(proto_mix) * proto
        if float(radius_norm) != 0.0:
            mixed = 1.0 - ((1.0 - mixed) / (max(radii[-1], 1e-4) ** float(radius_norm)))
        if label in old_labels:
            mixed = mixed + float(old_bias)
        pos_scores.append(mixed)
    score_matrix = np.stack(pos_scores, axis=1)
    proto_matrix = np.stack(prototypes, axis=0)
    proto_sim = proto_matrix @ proto_matrix.T
    if float(neg_lambda) > 0.0:
        penalties = np.zeros_like(score_matrix)
        for class_i in range(len(class_labels)):
            close_mask = proto_sim[class_i] >= float(neg_threshold)
            close_mask[class_i] = False
            if mutual_only:
                close_mask = close_mask & (proto_sim[:, class_i] >= float(neg_threshold))
            if not bool(np.any(close_mask)):
                continue
            other = score_matrix[:, close_mask]
            # Penalize a class only when a close support-defined neighbor has
            # competitive positive evidence. This uses no query label.
            penalties[:, class_i] = np.maximum(0.0, np.max(other, axis=1) - score_matrix[:, class_i] + float(neg_margin))
        score_matrix = score_matrix - float(neg_lambda) * penalties
    radius_by_label = {label: radii[i] for i, label in enumerate(class_labels)}
    return score_matrix, radius_by_label, proto_sim


def _predict(scores: np.ndarray, class_labels: list[str]) -> np.ndarray:
    labels = np.asarray(class_labels, dtype=object)
    return labels[np.argmax(scores, axis=1)]


def _metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: qknn._accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in new_labels}
    old_acc = qknn._accuracy(old_pred, old_truth)
    new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_new_floor75": min_new >= float(new_floor),
        "passes_joint_target": (
            old_acc >= float(old_target)
            and min_old >= float(old_floor)
            and new_acc >= float(new_target)
            and min_new >= float(new_floor)
        ),
    }


def _rank(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    ratio = min(
        float(row["old_acc"]) / 0.80,
        float(row["min_old_class_acc"]) / 0.75,
        float(row["seen_new_acc"]) / 0.75,
        float(row["min_seen_new_class_acc"]) / 0.75,
    )
    return (
        float(bool(row["passes_joint_target"])),
        ratio,
        float(row["min_seen_new_class_acc"]),
        float(row["seen_new_acc"]),
        float(row["old_acc"]),
        float(row["min_old_class_acc"]),
    )


def _support_loo_order(support_labels: np.ndarray, old_labels: list[str], new_labels: list[str]) -> np.ndarray:
    old_set = set(old_labels)
    new_set = set(new_labels)
    old_pos = [i for i, label in enumerate(support_labels.tolist()) if label in old_set]
    new_pos = [i for i, label in enumerate(support_labels.tolist()) if label in new_set]
    return np.asarray(old_pos + new_pos, dtype=int)


def _support_loo_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    scenarios: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    topm: int,
    proto_mix: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row, _label in enumerate(support_labels.tolist()):
        keep = np.ones(support_indices.size, dtype=bool)
        keep[row] = False
        row_scores, _, _ = _class_scores(
            features=features,
            support_indices=support_indices[keep],
            support_labels=support_labels[keep],
            query_indices=support_indices[row : row + 1],
            scenarios=scenarios,
            class_labels=class_labels,
            old_labels=old_labels,
            topm=topm,
            proto_mix=proto_mix,
            radius_norm=radius_norm,
            old_bias=old_bias,
            neg_lambda=neg_lambda,
            neg_threshold=neg_threshold,
            neg_margin=neg_margin,
            mutual_only=mutual_only,
            scenario_aware=scenario_aware,
        )
        rows.append(row_scores[0])
    return np.stack(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--seed_start", type=int, default=421077)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=50)
    parser.add_argument("--k_new", type=int, default=50)
    parser.add_argument("--query_per_old", type=int, default=30)
    parser.add_argument("--query_per_new", type=int, default=30)
    parser.add_argument("--pool_per_old", type=int, default=80)
    parser.add_argument("--pool_per_new", type=int, default=80)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--topm_grid", default="1,3,5,9,15,25")
    parser.add_argument("--proto_mix_grid", default="0,0.15,0.3,0.5,0.7")
    parser.add_argument("--radius_norm_grid", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--old_bias_grid", default="-0.05,0,0.03,0.06,0.1")
    parser.add_argument("--neg_lambda_grid", default="0,0.1,0.2,0.4,0.7,1.0")
    parser.add_argument("--neg_threshold_grid", default="0.65,0.7,0.75,0.8,0.85")
    parser.add_argument("--neg_margin_grid", default="0,0.01,0.02,0.04")
    parser.add_argument("--mutual_only_grid", default="true,false")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--skip_support_loo", action="store_true")
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
    best_proto_sim: list[dict[str, Any]] = []
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
            support_indices, support_labels, old_query, new_query = _collect(
                active._as_eval_splits(old_raw),
                active._as_eval_splits(new_raw),
                old_labels,
                new_labels,
            )
            query_indices = np.concatenate([old_query, new_query])
            truth = tx_ids[query_indices]
            loo_order = _support_loo_order(support_labels, old_labels, new_labels)
            loo_truth = support_labels[loo_order]
            for topm in qknn._parse_int_csv(args.topm_grid):
                for proto_mix in qknn._parse_float_csv(args.proto_mix_grid):
                    for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                        for old_bias in qknn._parse_float_csv(args.old_bias_grid):
                            for neg_lambda in qknn._parse_float_csv(args.neg_lambda_grid):
                                for neg_threshold in qknn._parse_float_csv(args.neg_threshold_grid):
                                    for neg_margin in qknn._parse_float_csv(args.neg_margin_grid):
                                        for mutual_raw in qknn._parse_csv(args.mutual_only_grid):
                                            mutual_only = str(mutual_raw).lower() in {"1", "true", "yes", "y"}
                                            query_scores, radii, proto_sim = _class_scores(
                                                features=features,
                                                support_indices=support_indices,
                                                support_labels=support_labels,
                                                query_indices=query_indices,
                                                scenarios=scenarios,
                                                class_labels=class_labels,
                                                old_labels=old_label_set,
                                                topm=topm,
                                                proto_mix=proto_mix,
                                                radius_norm=radius_norm,
                                                old_bias=old_bias,
                                                neg_lambda=neg_lambda,
                                                neg_threshold=neg_threshold,
                                                neg_margin=neg_margin,
                                                mutual_only=mutual_only,
                                                scenario_aware=bool(args.scenario_aware),
                                            )
                                            if bool(args.skip_support_loo):
                                                loo_metrics = {
                                                    "old_acc": 0.0,
                                                    "min_old_class_acc": 0.0,
                                                    "seen_new_acc": 0.0,
                                                    "min_seen_new_class_acc": 0.0,
                                                    "per_old_acc": {label: 0.0 for label in old_labels},
                                                    "per_new_acc": {label: 0.0 for label in new_labels},
                                                    "passes_new_floor75": False,
                                                    "passes_joint_target": False,
                                                }
                                            else:
                                                loo_scores = _support_loo_scores(
                                                    features=features,
                                                    support_indices=support_indices,
                                                    support_labels=support_labels,
                                                    scenarios=scenarios,
                                                    class_labels=class_labels,
                                                    old_labels=old_label_set,
                                                    topm=topm,
                                                    proto_mix=proto_mix,
                                                    radius_norm=radius_norm,
                                                    old_bias=old_bias,
                                                    neg_lambda=neg_lambda,
                                                    neg_threshold=neg_threshold,
                                                    neg_margin=neg_margin,
                                                    mutual_only=mutual_only,
                                                    scenario_aware=bool(args.scenario_aware),
                                                )
                                                loo_metrics = _metrics(
                                                    _predict(loo_scores[loo_order], class_labels),
                                                    loo_truth,
                                                    old_count=sum(1 for label in loo_truth.tolist() if label in old_label_set),
                                                    old_labels=old_labels,
                                                    new_labels=new_labels,
                                                    old_target=float(args.old_target),
                                                    old_floor=float(args.old_floor),
                                                    new_target=float(args.seen_new_target),
                                                    new_floor=float(args.seen_new_floor),
                                                )
                                            query_metrics = _metrics(
                                                _predict(query_scores, class_labels),
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
                                                "topm": int(topm),
                                                "proto_mix": float(proto_mix),
                                                "radius_norm": float(radius_norm),
                                                "old_bias": float(old_bias),
                                                "neg_lambda": float(neg_lambda),
                                                "neg_threshold": float(neg_threshold),
                                                "neg_margin": float(neg_margin),
                                                "mutual_only": bool(mutual_only),
                                                "scenario_aware": bool(args.scenario_aware),
                                                "k_old": int(args.k_old),
                                                "k_new": int(args.k_new),
                                                "pool_per_old": int(args.pool_per_old),
                                                "pool_per_new": int(args.pool_per_new),
                                                "old_role": str(args.old_role),
                                                "new_role": str(args.new_role),
                                                "stored_quantized_support_code_count": int(support_indices.size),
                                                "stored_raw_support_count": 0,
                                                "stored_class_prototype_count": int(len(class_labels)),
                                                "support_loo_skipped": bool(args.skip_support_loo),
                                                "class_radii": radii,
                                            }
                                            row.update({f"query_{key}": value for key, value in query_metrics.items()})
                                            row.update({f"support_loo_{key}": value for key, value in loo_metrics.items()})
                                            row["query_rank_score"] = _rank(query_metrics)
                                            row["support_loo_rank_score"] = _rank(loo_metrics)
                                            rows.append(row)
                                            if not best_proto_sim:
                                                best_proto_sim = [
                                                    {
                                                        "left": class_labels[i],
                                                        "right": class_labels[j],
                                                        "prototype_cosine": float(proto_sim[i, j]),
                                                    }
                                                    for i in range(len(class_labels))
                                                    for j in range(i + 1, len(class_labels))
                                                ]

    rows.sort(key=lambda row: tuple(row["query_rank_score"]), reverse=True)
    best_by_support = sorted(rows, key=lambda row: tuple(row["support_loo_rank_score"]), reverse=True)
    best_proto_sim.sort(key=lambda row: row["prototype_cosine"], reverse=True)
    summary = {
        "diagnostic_scope": "SUPPORT_ONLY_CONFUSION_AWARE_QKNN_NO_QUERY_LABEL_FIT",
        "selection_note": "best_by_query is audit only; deployable selection must use support_loo or fixed hyperparameters.",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "old_role": str(args.old_role),
        "new_role": str(args.new_role),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "rows_count": int(len(rows)),
        "best_by_query": rows[:20],
        "best_by_support_loo": best_by_support[:20],
        "closest_support_prototype_pairs": best_proto_sim[:30],
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "topm",
        "proto_mix",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "neg_margin",
        "mutual_only",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_new_floor75",
        "query_passes_joint_target",
        "query_per_old_acc",
        "query_per_new_acc",
        "support_loo_old_acc",
        "support_loo_min_old_class_acc",
        "support_loo_seen_new_acc",
        "support_loo_min_seen_new_class_acc",
        "support_loo_passes_new_floor75",
        "support_loo_passes_joint_target",
        "support_loo_per_old_acc",
        "support_loo_per_new_acc",
        "stored_quantized_support_code_count",
        "stored_raw_support_count",
        "stored_class_prototype_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            for key in ("query_per_old_acc", "query_per_new_acc", "support_loo_per_old_acc", "support_loo_per_new_acc"):
                out[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(
        json.dumps(
            {
                "rows_count": len(rows),
                "best_by_support_loo": best_by_support[:3],
                "best_by_query": rows[:3],
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
