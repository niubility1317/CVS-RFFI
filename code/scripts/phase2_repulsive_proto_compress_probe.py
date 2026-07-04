#!/usr/bin/env python3
"""Probe support-only repulsive compressed prototypes for Phase2-C.

The head freezes exported z_id features. Target support is used to estimate one
compressed prototype per class, then the prototypes are lightly separated on
the unit sphere using only support labels. Deployment stores int8 prototypes and
does not retain raw support IQ or per-sample support embeddings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _build_support(
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    combo: tuple[str, ...],
) -> tuple[list[int], list[str], list[int], list[int]]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for label, (support, query) in old_splits.items():
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query_indices.extend(query.tolist())
    for label in combo:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query_indices.extend(query.tolist())
    return support_indices, support_labels, old_query_indices, new_query_indices


def _fit_repulsive_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    repel_lambda: float,
    anchor_lambda: float,
    margin: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(sorted({str(label) for label in support_labels.tolist()}), dtype=object)
    centroids = []
    radii = []
    for label in labels.tolist():
        cls = support_features[support_labels == label]
        centroid = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        radii.append(float(np.mean(1.0 - (cls @ centroid))))
    anchors = qknn._normalize_rows(np.asarray(centroids, dtype=np.float64))
    proto = anchors.copy()
    if proto.shape[0] >= 2 and float(repel_lambda) > 0.0 and int(steps) > 0:
        for _ in range(int(steps)):
            sim = proto @ proto.T
            np.fill_diagonal(sim, -np.inf)
            active_pairs = sim > float(margin)
            grad = float(anchor_lambda) * (anchors - proto)
            for i in range(proto.shape[0]):
                close = np.where(active_pairs[i])[0]
                if close.size:
                    excess = (sim[i, close] - float(margin))[:, None]
                    grad[i] -= float(repel_lambda) * np.sum(excess * proto[close], axis=0)
            proto = qknn._normalize_rows(proto + grad)
    return labels, proto, np.asarray(radii, dtype=np.float64)


def _quantize_prototypes(proto: np.ndarray) -> np.ndarray:
    scale = 127.0
    quantized = np.clip(np.rint(proto * scale), -scale, scale).astype(np.int8)
    return qknn._normalize_rows(quantized.astype(np.float64) / scale)


def _predict(
    query_features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    old_labels: set[str],
    radii: np.ndarray,
    *,
    old_bias: float,
    radius_norm: float,
) -> np.ndarray:
    scores = qknn._normalize_rows(query_features) @ prototypes.T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(radii, 1e-4), float(radius_norm))[None, :]
        scores = 1.0 - ((1.0 - scores) / denom)
    if float(old_bias) != 0.0:
        is_old = np.asarray([str(label) in old_labels for label in labels.tolist()], dtype=np.float64)
        scores = scores + is_old[None, :] * float(old_bias)
    return np.asarray([labels[int(idx)] for idx in np.argmax(scores, axis=1)], dtype=object)


def _evaluate(
    combo: tuple[str, ...],
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    repel_lambda: float,
    anchor_lambda: float,
    margin: float,
    steps: int,
    old_bias: float,
    radius_norm: float,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    support_indices, support_labels, old_query_indices, new_query_indices = _build_support(old_splits, new_splits, combo)
    support_idx = np.asarray(support_indices, dtype=int)
    support_label_arr = np.asarray(support_labels, dtype=object).astype(str)
    labels, proto, radii = _fit_repulsive_prototypes(
        qknn._normalize_rows(features[support_idx]),
        support_label_arr,
        repel_lambda=repel_lambda,
        anchor_lambda=anchor_lambda,
        margin=margin,
        steps=steps,
    )
    proto = _quantize_prototypes(proto)
    query_idx = np.asarray(old_query_indices + new_query_indices, dtype=int)
    pred = _predict(
        features[query_idx],
        labels,
        proto,
        set(old_labels),
        radii,
        old_bias=old_bias,
        radius_norm=radius_norm,
    )
    truth = tx_ids[query_idx]
    old_count = len(old_query_indices)
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: qknn._accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in combo}
    old_acc = qknn._accuracy(old_pred, old_truth)
    seen_new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "method": "repulsive_proto_compress",
        "repel_lambda": float(repel_lambda),
        "anchor_lambda": float(anchor_lambda),
        "repel_margin": float(margin),
        "repel_steps": int(steps),
        "old_bias": float(old_bias),
        "radius_norm": float(radius_norm),
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_goal_floor75": min_new >= new_floor,
        "passes_joint_target": old_acc >= old_target and min_old >= old_floor and seen_new_acc >= new_target and min_new >= new_floor,
        "support_sample_count_used_for_compression": int(len(support_indices)),
        "old_query_count": int(len(old_query_indices)),
        "new_query_count": int(len(new_query_indices)),
        "stored_quantized_count": int(proto.shape[0]),
        "stored_support_count": 0,
    }


def _rank_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    return float(
        min(
            row["old_acc"] / args.old_target,
            row["min_old_class_acc"] / args.old_floor,
            row["seen_new_acc"] / args.seen_new_target,
            row["min_seen_new_class_acc"] / args.seen_new_floor,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="stable_first,centroid,scenario_centroid,scenario_diverse,source_proto_ranked_diverse")
    parser.add_argument("--seed_start", type=int, default=422931)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=5)
    parser.add_argument("--k_new", type=int, default=5)
    parser.add_argument("--query_per_old", type=int, default=75)
    parser.add_argument("--query_per_new", type=int, default=75)
    parser.add_argument("--pool_per_old", type=int, default=5)
    parser.add_argument("--pool_per_new", type=int, default=5)
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--repel_lambdas", default="0,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--anchor_lambdas", default="0.2,0.5,1.0")
    parser.add_argument("--repel_margins", default="0.65,0.7,0.75,0.8,0.85")
    parser.add_argument("--repel_steps", default="0,3,8,16")
    parser.add_argument("--old_bias_grid", default="0,0.001,0.003,0.005")
    parser.add_argument("--radius_norm_grid", default="0,0.1,0.2")
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
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    rows: list[dict[str, Any]] = []
    combo = tuple(new_labels)
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for policy in qknn._parse_csv(args.policies):
            old_raw = active._build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=old_labels,
                role="target_old",
                k=args.k_old,
                query_per_class=args.query_per_old,
                pool_per_class=args.pool_per_old,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            new_raw = active._build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=new_labels,
                role="target_new",
                k=args.k_new,
                query_per_class=args.query_per_new,
                pool_per_class=args.pool_per_new,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            if set(old_raw) != set(old_labels) or set(new_raw) != set(new_labels):
                continue
            old_splits = active._as_eval_splits(old_raw)
            new_splits = active._as_eval_splits(new_raw)
            for repel_lambda, anchor_lambda, margin, steps, old_bias, radius_norm in itertools.product(
                qknn._parse_float_csv(args.repel_lambdas),
                qknn._parse_float_csv(args.anchor_lambdas),
                qknn._parse_float_csv(args.repel_margins),
                qknn._parse_int_csv(args.repel_steps),
                qknn._parse_float_csv(args.old_bias_grid),
                qknn._parse_float_csv(args.radius_norm_grid),
            ):
                row = _evaluate(
                    combo,
                    features=features,
                    tx_ids=tx_ids,
                    old_splits=old_splits,
                    new_splits=new_splits,
                    old_labels=old_labels,
                    repel_lambda=repel_lambda,
                    anchor_lambda=anchor_lambda,
                    margin=margin,
                    steps=steps,
                    old_bias=old_bias,
                    radius_norm=radius_norm,
                    old_target=args.old_target,
                    old_floor=args.old_floor,
                    new_target=args.seen_new_target,
                    new_floor=args.seen_new_floor,
                )
                row["seed"] = int(seed)
                row["support_selection_policy"] = policy
                row["k_old"] = int(args.k_old)
                row["k_new"] = int(args.k_new)
                row["pool_per_old"] = int(args.pool_per_old)
                row["pool_per_new"] = int(args.pool_per_new)
                rows.append(row)

    rows.sort(key=lambda row: (bool(row["passes_joint_target"]), _rank_score(row, args), row["min_seen_new_class_acc"]), reverse=True)
    summary = {
        "diagnostic_scope": "REPULSIVE_PROTO_COMPRESS_NO_RAW_SUPPORT_STORAGE",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "pool_per_old": int(args.pool_per_old),
        "pool_per_new": int(args.pool_per_new),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "floor75_pass_count": int(sum(1 for row in rows if row["passes_goal_floor75"])),
        "best": rows[:20],
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "repel_lambda",
        "anchor_lambda",
        "repel_margin",
        "repel_steps",
        "old_bias",
        "radius_norm",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_goal_floor75",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "support_sample_count_used_for_compression",
        "stored_quantized_count",
        "stored_support_count",
        "old_query_count",
        "new_query_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            out["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"joint_pass_count": summary["joint_pass_count"], "best": rows[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
