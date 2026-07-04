#!/usr/bin/env python3
"""Probe compact subspace prototypes for Phase2-C many-new enrollment.

The head freezes exported ADV3B02 features. Target support labels are used only
to build a compact per-class prototype set: centroids, optional k-means centers,
and optional PCA virtual anchors. The deployed state stores quantized synthetic
prototypes and small transform metadata, not raw support IQ or support
embeddings.
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


def _safe_mean_center(features: np.ndarray) -> np.ndarray:
    return qknn._normalize_rows(features.mean(axis=0, keepdims=True))[0]


def _farthest_kmeans(features: np.ndarray, k: int, iterations: int) -> np.ndarray:
    x = qknn._normalize_rows(features)
    k = max(1, min(int(k), int(x.shape[0])))
    centers = [_safe_mean_center(x)]
    while len(centers) < k:
        center_arr = np.asarray(centers, dtype=np.float64)
        distances = 1.0 - np.max(x @ center_arr.T, axis=1)
        pick = int(np.argmax(distances))
        candidate = x[pick]
        if any(float(candidate @ center) > 0.999999 for center in centers):
            break
        centers.append(candidate)
    centers_arr = qknn._normalize_rows(np.asarray(centers, dtype=np.float64))
    for _ in range(max(0, int(iterations))):
        sims = x @ centers_arr.T
        assign = np.argmax(sims, axis=1)
        next_centers = []
        for idx in range(centers_arr.shape[0]):
            members = x[assign == idx]
            next_centers.append(_safe_mean_center(members) if members.size else centers_arr[idx])
        centers_arr = qknn._normalize_rows(np.asarray(next_centers, dtype=np.float64))
    return centers_arr


def _pca_virtual_anchors(features: np.ndarray, rank: int, scale: float, include_center: bool) -> np.ndarray:
    x = qknn._normalize_rows(features)
    center = _safe_mean_center(x)
    if rank <= 0 or x.shape[0] < 3:
        return center.reshape(1, -1) if include_center else np.empty((0, x.shape[1]), dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    _, svals, vh = np.linalg.svd(xc, full_matrices=False)
    max_rank = min(int(rank), int(vh.shape[0]), int(x.shape[0] - 1))
    anchors = [center] if include_center else []
    for pos in range(max_rank):
        step = float(scale) * float(svals[pos] / max(1, x.shape[0] - 1)) * vh[pos]
        anchors.append(center + step)
        anchors.append(center - step)
    if not anchors:
        return np.empty((0, x.shape[1]), dtype=np.float64)
    return qknn._normalize_rows(np.asarray(anchors, dtype=np.float64))


def _class_compact_prototypes(
    features: np.ndarray,
    *,
    mode: str,
    kmeans_count: int,
    pca_rank: int,
    pca_scale: float,
    include_centroid: bool,
    kmeans_iterations: int,
) -> np.ndarray:
    x = qknn._normalize_rows(features)
    pieces: list[np.ndarray] = []
    if mode in {"centroid", "hybrid"} or include_centroid:
        pieces.append(_safe_mean_center(x).reshape(1, -1))
    if mode in {"kmeans", "hybrid"} and int(kmeans_count) > 0:
        pieces.append(_farthest_kmeans(x, k=int(kmeans_count), iterations=int(kmeans_iterations)))
    if mode in {"subspace", "hybrid"} and int(pca_rank) > 0:
        pieces.append(
            _pca_virtual_anchors(
                x,
                rank=int(pca_rank),
                scale=float(pca_scale),
                include_center=False,
            )
        )
    if not pieces:
        pieces.append(_safe_mean_center(x).reshape(1, -1))
    proto = qknn._normalize_rows(np.concatenate([piece for piece in pieces if piece.size], axis=0))
    keep: list[int] = []
    for idx, row in enumerate(proto):
        if all(float(row @ proto[kept]) < 0.999 for kept in keep):
            keep.append(idx)
    return proto[np.asarray(keep, dtype=int)]


def _build_compact_bank(
    features: np.ndarray,
    scenarios: np.ndarray,
    support_indices: list[int],
    support_labels: list[str],
    old_labels: set[str],
    *,
    mode: str,
    kmeans_count: int,
    pca_rank: int,
    pca_scale: float,
    include_centroid: bool,
    scenario_aware: bool,
    quantize: bool,
    kmeans_iterations: int,
) -> dict[str, np.ndarray]:
    support_idx = np.asarray(support_indices, dtype=int)
    support_features = qknn._normalize_rows(features[support_idx])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    support_scenarios = np.asarray(scenarios[support_idx], dtype=object).astype(str)
    keys = (
        sorted({(str(label), str(scenario)) for label, scenario in zip(labels.tolist(), support_scenarios.tolist())})
        if scenario_aware
        else [(str(label), "") for label in sorted(set(labels.tolist()))]
    )

    proto_rows: list[np.ndarray] = []
    proto_labels: list[str] = []
    proto_scenarios: list[str] = []
    proto_radii: list[float] = []
    for label, scenario in keys:
        mask = labels == label
        if scenario_aware:
            mask = mask & (support_scenarios == scenario)
        cls = support_features[mask]
        if cls.size == 0:
            continue
        proto = _class_compact_prototypes(
            cls,
            mode=mode,
            kmeans_count=kmeans_count,
            pca_rank=pca_rank,
            pca_scale=pca_scale,
            include_centroid=include_centroid,
            kmeans_iterations=kmeans_iterations,
        )
        nearest = np.max(cls @ proto.T, axis=1)
        radius = float(np.mean(1.0 - nearest))
        proto_rows.extend([row for row in proto])
        proto_labels.extend([label] * int(proto.shape[0]))
        proto_scenarios.extend([scenario] * int(proto.shape[0]))
        proto_radii.extend([radius] * int(proto.shape[0]))

    proto_arr = np.asarray(proto_rows, dtype=np.float64)
    if quantize:
        scale = 127.0
        proto_arr = qknn._normalize_rows(np.clip(np.rint(proto_arr * scale), -scale, scale).astype(np.int8).astype(np.float64) / scale)
    return {
        "features": proto_arr,
        "labels": np.asarray(proto_labels, dtype=object),
        "is_old": np.asarray([str(label) in old_labels for label in proto_labels], dtype=bool),
        "radii_by_support": np.asarray(proto_radii, dtype=np.float64),
        "class_labels": np.asarray(sorted(set(proto_labels)), dtype=object),
        "scenarios": None if not scenario_aware else np.asarray(proto_scenarios, dtype=object),
    }


def _collect_indices(
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[list[int], list[str], list[int], list[int]]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query_indices.extend(query.tolist())
    for label in new_labels:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query_indices.extend(query.tolist())
    return support_indices, support_labels, old_query_indices, new_query_indices


def _evaluate(
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
    mode: str,
    kmeans_count: int,
    pca_rank: int,
    pca_scale: float,
    include_centroid: bool,
    scenario_aware: bool,
    topk: int,
    old_bias: float,
    radius_norm: float,
    kmeans_iterations: int,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    support_indices, support_labels, old_query_indices, new_query_indices = _collect_indices(
        old_splits,
        new_splits,
        old_labels,
        new_labels,
    )
    bank = _build_compact_bank(
        features,
        scenarios,
        support_indices,
        support_labels,
        set(old_labels),
        mode=mode,
        kmeans_count=kmeans_count,
        pca_rank=pca_rank,
        pca_scale=pca_scale,
        include_centroid=include_centroid,
        scenario_aware=scenario_aware,
        quantize=True,
        kmeans_iterations=kmeans_iterations,
    )
    query_idx = np.asarray(old_query_indices + new_query_indices, dtype=int)
    pred = qknn._predict_from_bank(
        bank,
        features[query_idx],
        topk=int(topk),
        old_bias=float(old_bias),
        radius_norm=float(radius_norm),
        query_scenarios=scenarios[query_idx] if scenario_aware else None,
        scenario_aware=scenario_aware,
    )
    truth = tx_ids[query_idx]
    old_count = len(old_query_indices)
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: qknn._accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in new_labels}
    old_acc = qknn._accuracy(old_pred, old_truth)
    seen_new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_goal_floor75": min_new >= float(new_floor),
        "passes_joint_target": old_acc >= float(old_target)
        and min_old >= float(old_floor)
        and seen_new_acc >= float(new_target)
        and min_new >= float(new_floor),
        "old_query_count": int(len(old_query_indices)),
        "new_query_count": int(len(new_query_indices)),
        "support_sample_count_used_for_fit": int(len(support_indices)),
        "stored_support_count": 0,
        "stored_quantized_prototype_count": int(bank["features"].shape[0]),
    }


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _rank_score(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}_min_seen_new_class_acc"]),
        float(row[f"{prefix}_seen_new_acc"]),
        float(row[f"{prefix}_min_old_class_acc"]),
        float(row[f"{prefix}_old_acc"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="source_proto_ranked_diverse")
    parser.add_argument("--modes", default="centroid,kmeans,subspace,hybrid")
    parser.add_argument("--kmeans_counts", default="2,4,6,8")
    parser.add_argument("--pca_ranks", default="0,2,4,6")
    parser.add_argument("--pca_scales", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--topk_grid", default="1,3,5,8")
    parser.add_argument("--old_bias_grid", default="0")
    parser.add_argument("--radius_norm_grid", default="0,0.1,0.2,0.3")
    parser.add_argument("--seed_start", type=int, default=422931)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=20)
    parser.add_argument("--k_new", type=int, default=20)
    parser.add_argument("--query_per_old", type=int, default=60)
    parser.add_argument("--query_per_new", type=int, default=60)
    parser.add_argument("--pool_per_old", type=int, default=50)
    parser.add_argument("--pool_per_new", type=int, default=50)
    parser.add_argument("--kmeans_iterations", type=int, default=6)
    parser.add_argument("--scenario_aware", action="store_true")
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
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    rows: list[dict[str, Any]] = []
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
            query_old_splits = active._as_eval_splits(old_raw)
            query_new_splits = active._as_eval_splits(new_raw)
            enroll_old_splits = active._as_eval_splits(old_raw, use_enrollment_val=True)
            enroll_new_splits = active._as_eval_splits(new_raw, use_enrollment_val=True)
            for mode in qknn._parse_csv(args.modes):
                for kmeans_count in qknn._parse_int_csv(args.kmeans_counts):
                    for pca_rank in qknn._parse_int_csv(args.pca_ranks):
                        if mode == "kmeans" and pca_rank != 0:
                            continue
                        if mode == "subspace" and kmeans_count != qknn._parse_int_csv(args.kmeans_counts)[0]:
                            continue
                        for pca_scale in qknn._parse_float_csv(args.pca_scales):
                            if mode in {"centroid", "kmeans"} and pca_scale != qknn._parse_float_csv(args.pca_scales)[0]:
                                continue
                            for topk in qknn._parse_int_csv(args.topk_grid):
                                for old_bias in qknn._parse_float_csv(args.old_bias_grid):
                                    for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                                        common = {
                                            "features": features,
                                            "tx_ids": tx_ids,
                                            "scenarios": scenarios,
                                            "old_labels": old_labels,
                                            "new_labels": new_labels,
                                            "mode": mode,
                                            "kmeans_count": int(kmeans_count),
                                            "pca_rank": int(pca_rank),
                                            "pca_scale": float(pca_scale),
                                            "include_centroid": True,
                                            "scenario_aware": bool(args.scenario_aware),
                                            "topk": int(topk),
                                            "old_bias": float(old_bias),
                                            "radius_norm": float(radius_norm),
                                            "kmeans_iterations": int(args.kmeans_iterations),
                                            "old_target": float(args.old_target),
                                            "old_floor": float(args.old_floor),
                                            "new_target": float(args.seen_new_target),
                                            "new_floor": float(args.seen_new_floor),
                                        }
                                        query_metrics = _evaluate(
                                            old_splits=query_old_splits,
                                            new_splits=query_new_splits,
                                            **common,
                                        )
                                        enroll_metrics = _evaluate(
                                            old_splits=enroll_old_splits,
                                            new_splits=enroll_new_splits,
                                            **common,
                                        )
                                        row: dict[str, Any] = {
                                            "seed": int(seed),
                                            "support_selection_policy": policy,
                                            "mode": mode,
                                            "kmeans_count": int(kmeans_count),
                                            "pca_rank": int(pca_rank),
                                            "pca_scale": float(pca_scale),
                                            "topk": int(topk),
                                            "old_bias": float(old_bias),
                                            "radius_norm": float(radius_norm),
                                            "scenario_aware": bool(args.scenario_aware),
                                            "k_old": int(args.k_old),
                                            "k_new": int(args.k_new),
                                            "pool_per_old": int(args.pool_per_old),
                                            "pool_per_new": int(args.pool_per_new),
                                            "exclude_pool_from_query": bool(args.exclude_pool_from_query),
                                        }
                                        row.update(_prefix("query", query_metrics))
                                        row.update(_prefix("enroll_val", enroll_metrics))
                                        rows.append(row)

    rows_by_query = sorted(rows, key=lambda row: _rank_score(row, "query"), reverse=True)
    rows_by_enroll = sorted(rows, key=lambda row: _rank_score(row, "enroll_val"), reverse=True)
    summary = {
        "diagnostic_scope": "COMPACT_SUBSPACE_PROTOTYPE_NO_RAW_SUPPORT_STORAGE",
        "selection_note": "best_by_enrollment uses labeled enrollment-pool leftovers only; best_by_query is an audit and must not be used as deployment hyperparameter selection.",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows": rows_by_query,
        "best_by_query": rows_by_query[:20],
        "best_by_enrollment": rows_by_enroll[:20],
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "mode",
        "kmeans_count",
        "pca_rank",
        "pca_scale",
        "topk",
        "old_bias",
        "radius_norm",
        "scenario_aware",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_goal_floor75",
        "query_passes_joint_target",
        "query_per_old_acc",
        "query_per_new_acc",
        "query_old_query_count",
        "query_new_query_count",
        "query_stored_quantized_prototype_count",
        "query_stored_support_count",
        "enroll_val_old_acc",
        "enroll_val_min_old_class_acc",
        "enroll_val_seen_new_acc",
        "enroll_val_min_seen_new_class_acc",
        "enroll_val_passes_goal_floor75",
        "enroll_val_passes_joint_target",
        "enroll_val_per_old_acc",
        "enroll_val_per_new_acc",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows_by_query:
            out = {key: row.get(key) for key in fields}
            for key in ("query_per_old_acc", "query_per_new_acc", "enroll_val_per_old_acc", "enroll_val_per_new_acc"):
                out[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(
        json.dumps(
            {
                "best_by_enrollment": rows_by_enroll[:3],
                "best_by_query": rows_by_query[:3],
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
