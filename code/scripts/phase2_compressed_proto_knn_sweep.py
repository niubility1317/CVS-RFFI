#!/usr/bin/env python3
"""Evaluate exemplar-free compressed prototype KNN heads for Stage2-C.

The head consumes support features once, then keeps only class prototypes,
prototype radii, class counts, and old/new class flags. It does not retain raw
support samples or per-support embeddings for inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CompressedMemory:
    prototype_matrix: np.ndarray
    prototype_labels: np.ndarray
    prototype_radii: np.ndarray
    prototype_weights: np.ndarray
    prototype_is_old: np.ndarray
    counts: dict[str, int]


@dataclass(frozen=True)
class QuantizedKnnMemory:
    quantized_matrix: np.ndarray
    scale: float
    labels: np.ndarray
    is_old: np.ndarray
    class_prototype_matrix: np.ndarray
    class_prototype_labels: np.ndarray
    counts: dict[str, int]
    quant_bits: int
    stored_support_count: int = 0

    @property
    def stored_quantized_count(self) -> int:
        return int(self.quantized_matrix.shape[0])


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None) -> list[float]:
    if not value:
        return [0.0]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == truth))


def _harmonic(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def _classwise_topk_predict(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(scores.shape[1])))
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    pred: list[str] = []
    for row_idx, candidates in enumerate(top_idx):
        class_scores: dict[str, float] = {}
        for candidate in candidates.tolist():
            label = str(labels[candidate])
            class_scores[label] = class_scores.get(label, 0.0) + float(scores[row_idx, candidate])
        pred.append(max(class_scores.items(), key=lambda item: (item[1], item[0]))[0])
    return np.asarray(pred, dtype=object)


def _medoid_anchors(features: np.ndarray, count: int) -> list[np.ndarray]:
    features = _normalize_rows(features)
    count = max(1, min(int(count), int(features.shape[0])))
    mean = _normalize_rows(features.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(features @ mean))]
    while len(selected) < count:
        sims = features @ features[np.asarray(selected, dtype=int)].T
        farthest = int(np.argmin(np.max(sims, axis=1)))
        if farthest in selected:
            remaining = [idx for idx in range(features.shape[0]) if idx not in selected]
            if not remaining:
                break
            farthest = remaining[0]
        selected.append(farthest)
    return [features[idx] for idx in selected]


def _boundary_medoid_anchors(
    features: np.ndarray,
    label: str,
    class_means: dict[str, np.ndarray],
    count: int,
) -> list[np.ndarray]:
    features = _normalize_rows(features)
    count = max(1, min(int(count), int(features.shape[0])))
    anchors = _medoid_anchors(features, 1)
    if count == 1:
        return anchors

    own_mean = class_means[label]
    other_means = [mean for other_label, mean in class_means.items() if other_label != label]
    if not other_means:
        return _medoid_anchors(features, count)

    other_matrix = np.vstack(other_means)
    margins = (features @ own_mean) - np.max(features @ other_matrix.T, axis=1)
    selected = [int(np.argmax(features @ anchors[0]))]
    for candidate in np.argsort(margins).tolist():
        if candidate not in selected:
            selected.append(int(candidate))
        if len(selected) >= count:
            break
    while len(selected) < count:
        sims = features @ features[np.asarray(selected, dtype=int)].T
        candidate = int(np.argmin(np.max(sims, axis=1)))
        if candidate in selected:
            remaining = [idx for idx in range(features.shape[0]) if idx not in selected]
            if not remaining:
                break
            candidate = remaining[0]
        selected.append(candidate)
    return [features[idx] for idx in selected]


def _subprototypes(features: np.ndarray, count: int, seed: int, mode: str) -> list[np.ndarray]:
    features = _normalize_rows(features)
    count = max(1, min(int(count), int(features.shape[0])))
    if mode == "medoid":
        return _medoid_anchors(features, count)
    if mode != "mean":
        raise ValueError(f"Unsupported prototype_mode: {mode}")
    if count == 1:
        return [_normalize_rows(features.mean(axis=0, keepdims=True))[0]]

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, features.shape[0]))
    centers = [features[first]]
    while len(centers) < count:
        sims = features @ np.vstack(centers).T
        farthest = int(np.argmin(np.max(sims, axis=1)))
        centers.append(features[farthest])
    center_matrix = _normalize_rows(np.vstack(centers))
    for _ in range(6):
        assignments = np.argmax(features @ center_matrix.T, axis=1)
        updated = []
        for idx in range(count):
            members = features[assignments == idx]
            if members.size == 0:
                updated.append(center_matrix[idx])
            else:
                updated.append(_normalize_rows(members.mean(axis=0, keepdims=True))[0])
        center_matrix = _normalize_rows(np.vstack(updated))
    return [center_matrix[idx] for idx in range(count)]


def _loo_knn1_teacher_labels(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if features.shape[0] <= 1:
        return labels.copy()
    sims = features @ features.T
    np.fill_diagonal(sims, -np.inf)
    return labels[np.argmax(sims, axis=1)]


def _loo_knn1_agreement_weights(
    features: np.ndarray,
    labels: np.ndarray,
    prototype_matrix: np.ndarray,
    prototype_labels: np.ndarray,
) -> np.ndarray:
    teacher_labels = _loo_knn1_teacher_labels(features, labels)
    winners = np.argmax(features @ prototype_matrix.T, axis=1)
    weights: list[float] = []
    for idx, prototype_label in enumerate(prototype_labels):
        assigned = winners == idx
        if not np.any(assigned):
            weights.append(1.0)
            continue
        agree = int(np.sum(teacher_labels[assigned] == prototype_label))
        disagree = int(np.sum(teacher_labels[assigned] != prototype_label))
        weights.append(float((agree + 1.0) / (disagree + 1.0)))
    return np.asarray(weights, dtype=np.float64)


def build_compressed_memory(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    old_labels: set[str] | None = None,
    prototypes_per_class: int = 1,
    prototype_mode: str = "mean",
    prototype_weight_mode: str = "uniform",
    seed: int = 0,
) -> CompressedMemory:
    features = _normalize_rows(support_features)
    labels = np.asarray(support_labels, dtype=object).astype(str)
    old = set(old_labels or set())
    prototypes: list[np.ndarray] = []
    prototype_labels: list[str] = []
    prototype_radii: list[float] = []
    prototype_weights: list[float] = []
    prototype_is_old: list[bool] = []
    counts: dict[str, int] = {}
    sorted_labels = sorted({str(label) for label in labels.tolist()})
    class_means = {
        label: _normalize_rows(features[labels == label].mean(axis=0, keepdims=True))[0] for label in sorted_labels
    }

    for label in sorted_labels:
        class_features = features[labels == label]
        counts[label] = int(class_features.shape[0])
        if prototype_mode == "boundary_medoid":
            centers = _boundary_medoid_anchors(class_features, label, class_means, prototypes_per_class)
        else:
            centers = _subprototypes(class_features, prototypes_per_class, _stable_seed(seed, label), prototype_mode)
        center_matrix = np.vstack(centers)
        assignments = np.argmax(class_features @ center_matrix.T, axis=1)
        for idx, center in enumerate(centers):
            members = class_features[assignments == idx]
            if members.size == 0:
                radius = 0.0
                weight = 1.0
            else:
                radius = float(np.mean(1.0 - (members @ center)))
                if prototype_weight_mode == "uniform":
                    weight = 1.0
                elif prototype_weight_mode == "assigned_count":
                    weight = float(members.shape[0])
                elif prototype_weight_mode == "loo_knn1_agreement":
                    weight = 1.0
                else:
                    raise ValueError(f"Unsupported prototype_weight_mode: {prototype_weight_mode}")
            prototypes.append(center)
            prototype_labels.append(label)
            prototype_radii.append(radius)
            prototype_weights.append(weight)
            prototype_is_old.append(label in old)

    prototype_matrix = _normalize_rows(np.vstack(prototypes))
    prototype_label_array = np.asarray(prototype_labels, dtype=object)
    if prototype_weight_mode == "loo_knn1_agreement":
        prototype_weight_array = _loo_knn1_agreement_weights(features, labels, prototype_matrix, prototype_label_array)
    else:
        prototype_weight_array = np.asarray(prototype_weights, dtype=np.float64)

    return CompressedMemory(
        prototype_matrix=prototype_matrix,
        prototype_labels=prototype_label_array,
        prototype_radii=np.asarray(prototype_radii, dtype=np.float64),
        prototype_weights=prototype_weight_array,
        prototype_is_old=np.asarray(prototype_is_old, dtype=bool),
        counts=counts,
    )


def build_quantized_knn_memory(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    old_labels: set[str] | None = None,
    quant_bits: int = 8,
) -> QuantizedKnnMemory:
    if int(quant_bits) != 8:
        raise ValueError("Only int8 quantized KNN memory is currently supported")
    features = _normalize_rows(support_features)
    labels = np.asarray(support_labels, dtype=object).astype(str)
    scale = float((2 ** (int(quant_bits) - 1)) - 1)
    quantized = np.clip(np.rint(features * scale), -scale, scale).astype(np.int8)
    old = set(old_labels or set())
    sorted_labels = sorted({str(label) for label in labels.tolist()})
    counts = {label: int(np.sum(labels == label)) for label in sorted_labels}
    prototypes = np.vstack([
        _normalize_rows(features[labels == label].mean(axis=0, keepdims=True))[0] for label in sorted_labels
    ])
    return QuantizedKnnMemory(
        quantized_matrix=quantized,
        scale=scale,
        labels=labels,
        is_old=np.asarray([str(label) in old for label in labels], dtype=bool),
        class_prototype_matrix=prototypes,
        class_prototype_labels=np.asarray(sorted_labels, dtype=object),
        counts=counts,
        quant_bits=int(quant_bits),
    )


def predict_quantized_knn_memory(
    memory: QuantizedKnnMemory,
    query_features: np.ndarray,
    *,
    k: int = 1,
    old_bias: float = 0.0,
    prototype_blend: float = 0.0,
) -> np.ndarray:
    query = _normalize_rows(query_features)
    support = memory.quantized_matrix.astype(np.float64) / float(memory.scale)
    support = _normalize_rows(support)
    scores = query @ support.T
    scores = scores + (memory.is_old.astype(np.float64) * float(old_bias))[None, :]
    if float(prototype_blend) != 0.0:
        proto_scores = query @ memory.class_prototype_matrix.T
        proto_by_support = np.zeros_like(scores)
        for idx, label in enumerate(memory.class_prototype_labels):
            proto_by_support[:, memory.labels == label] = proto_scores[:, idx][:, None]
        scores = scores + proto_by_support * float(prototype_blend)
    return _classwise_topk_predict(scores, memory.labels, k)


def predict_compressed_memory(
    memory: CompressedMemory,
    query_features: np.ndarray,
    *,
    old_bias: float = 0.0,
    radius_weight: float = 0.0,
    weight_scale: float = 0.0,
) -> np.ndarray:
    query = _normalize_rows(query_features)
    scores = query @ memory.prototype_matrix.T
    scores = scores + (memory.prototype_is_old.astype(np.float64) * float(old_bias))[None, :]
    scores = scores - (memory.prototype_radii * float(radius_weight))[None, :]
    scores = scores + (np.log(np.maximum(memory.prototype_weights, 1e-12)) * float(weight_scale))[None, :]
    best = np.argmax(scores, axis=1)
    return memory.prototype_labels[best]


def _split_indices(
    tx_ids: np.ndarray,
    roles: np.ndarray,
    tx: str,
    role: str,
    support_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    available = np.where((tx_ids == tx) & (roles == role))[0]
    required = int(support_count) + int(query_count)
    if available.size < required:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    rng = np.random.default_rng(_stable_seed(seed, f"{role}:{tx}"))
    shuffled = available[rng.permutation(available.size)]
    return shuffled[:support_count].astype(int), shuffled[support_count : support_count + query_count].astype(int)


def _eligible_tx(
    tx_ids: np.ndarray,
    roles: np.ndarray,
    candidates: Iterable[str],
    roles_to_match: Iterable[str],
    support_count: int,
    query_count: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tx in candidates:
        for role in roles_to_match:
            support, query = _split_indices(tx_ids, roles, tx, role, support_count, query_count, seed)
            if support.size and query.size:
                out[str(tx)] = (support, query)
                break
    return out


def _evaluate_combo(
    combo: tuple[str, ...],
    features: np.ndarray,
    tx_ids: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_target: float,
    new_target: float,
    prototypes_per_class: int,
    prototype_mode: str,
    prototype_weight_mode: str,
    old_bias: float,
    radius_weight: float,
    weight_scale: float,
    seed: int,
    quant_bits: int,
) -> dict[str, Any]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    old_labels = set(old_splits)

    for tx, (support, query) in old_splits.items():
        support_indices.extend(support.tolist())
        support_labels.extend([tx] * int(support.size))
        old_query_indices.extend(query.tolist())
    for tx in combo:
        support, query = new_splits[tx]
        support_indices.extend(support.tolist())
        support_labels.extend([tx] * int(support.size))
        new_query_indices.extend(query.tolist())

    support_features = features[np.asarray(support_indices, dtype=int)]
    support_label_array = np.asarray(support_labels, dtype=object)
    old_query_idx = np.asarray(old_query_indices, dtype=int)
    new_query_idx = np.asarray(new_query_indices, dtype=int)
    query_idx = np.concatenate([old_query_idx, new_query_idx])
    truth = tx_ids[query_idx]
    if prototype_mode == "quantized_knn":
        memory = build_quantized_knn_memory(
            support_features,
            support_label_array,
            old_labels=old_labels,
            quant_bits=quant_bits,
        )
        pred = predict_quantized_knn_memory(
            memory,
            features[query_idx],
            k=prototypes_per_class,
            old_bias=old_bias,
            prototype_blend=radius_weight,
        )
        method = f"qknn{quant_bits}_k{prototypes_per_class}_oldbias{old_bias:g}_pblend{radius_weight:g}"
        stored_prototype_count = int(len(memory.class_prototype_labels)) if float(radius_weight) != 0.0 else 0
        stored_weight_count = 0
        stored_quantized_count = memory.stored_quantized_count
    else:
        memory = build_compressed_memory(
            support_features,
            support_label_array,
            old_labels=old_labels,
            prototypes_per_class=prototypes_per_class,
            prototype_mode=prototype_mode,
            prototype_weight_mode=prototype_weight_mode,
            seed=seed,
        )
        pred = predict_compressed_memory(
            memory,
            features[query_idx],
            old_bias=old_bias,
            radius_weight=radius_weight,
            weight_scale=weight_scale,
        )
        method = f"cproto_{prototype_mode}_{prototype_weight_mode}_p{prototypes_per_class}_oldbias{old_bias:g}_rad{radius_weight:g}_w{weight_scale:g}"
        stored_prototype_count = int(memory.prototype_matrix.shape[0])
        stored_weight_count = int(memory.prototype_weights.shape[0])
        stored_quantized_count = 0
    old_pred = pred[: old_query_idx.size]
    old_truth = truth[: old_query_idx.size]
    new_pred = pred[old_query_idx.size :]
    new_truth = truth[old_query_idx.size :]

    old_acc = _accuracy(old_pred, old_truth)
    per_new: dict[str, float] = {}
    for tx in combo:
        mask = new_truth == tx
        per_new[tx] = _accuracy(new_pred[mask], new_truth[mask])
    seen_new_acc = float(np.mean(list(per_new.values()))) if per_new else 0.0
    min_new_acc = min(per_new.values()) if per_new else 0.0

    return {
        "new_tx_ids": list(combo),
        "method": method,
        "old_acc": old_acc,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new_acc,
        "H_old_new": _harmonic(old_acc, seen_new_acc),
        "per_new_acc": per_new,
        "passes_old_target": old_acc >= old_target,
        "passes_seen_new_average_target": seen_new_acc >= 0.80,
        "passes_seen_new_per_class_target": min_new_acc >= new_target,
        "passes_joint_target": old_acc >= old_target and seen_new_acc >= 0.80 and min_new_acc >= new_target,
        "support_count": int(len(support_indices)),
        "old_query_count": int(old_query_idx.size),
        "new_query_count": int(new_query_idx.size),
        "stored_prototype_count": stored_prototype_count,
        "stored_weight_count": stored_weight_count,
        "stored_quantized_count": stored_quantized_count,
        "stored_support_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--old_roles", default="target_old")
    parser.add_argument("--new_roles", default="target_new,new")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=422001)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=40)
    parser.add_argument("--query_per_new", type=int, default=40)
    parser.add_argument("--old_target", type=float, default=0.85)
    parser.add_argument("--seen_new_target", type=float, default=0.85)
    parser.add_argument("--max_pair_candidates", type=int, default=50)
    parser.add_argument("--prototypes_per_class", default="1,2")
    parser.add_argument("--prototype_mode_grid", default="mean")
    parser.add_argument("--prototype_weight_mode_grid", default="uniform")
    parser.add_argument("--old_bias_grid", default="0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16")
    parser.add_argument("--radius_weight_grid", default="0,0.25,0.5")
    parser.add_argument("--weight_scale_grid", default="0")
    parser.add_argument("--quant_bits", type=int, default=8)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    old_candidates = _parse_csv(args.old_tx_ids)
    new_candidates = _parse_csv(args.candidate_new_tx_ids)
    if not new_candidates:
        role_mask = np.isin(roles, np.asarray(_parse_csv(args.new_roles), dtype=object))
        new_candidates = sorted({str(tx) for tx in tx_ids[role_mask].tolist()})
    old_splits = _eligible_tx(
        tx_ids, roles, old_candidates, _parse_csv(args.old_roles), args.k_old, args.query_per_old, args.seed
    )
    new_splits = _eligible_tx(
        tx_ids, roles, new_candidates, _parse_csv(args.new_roles), args.k_new, args.query_per_new, args.seed
    )
    if not old_splits:
        raise RuntimeError("No eligible old TX splits found")
    if not new_splits:
        raise RuntimeError("No eligible new TX splits found")

    selected = sorted(new_splits)[: int(args.max_pair_candidates)]
    rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(selected, int(args.combo_size)):
        for prototype_mode in _parse_csv(args.prototype_mode_grid):
            for prototype_weight_mode in _parse_csv(args.prototype_weight_mode_grid):
                for prototypes_per_class in [int(v) for v in _parse_csv(args.prototypes_per_class)]:
                    for old_bias in _parse_float_csv(args.old_bias_grid):
                        for radius_weight in _parse_float_csv(args.radius_weight_grid):
                            for weight_scale in _parse_float_csv(args.weight_scale_grid):
                                rows.append(
                                    _evaluate_combo(
                                        tuple(combo),
                                        features,
                                        tx_ids,
                                        old_splits,
                                        new_splits,
                                        args.old_target,
                                        args.seen_new_target,
                                        prototypes_per_class,
                                        prototype_mode,
                                        prototype_weight_mode,
                                        old_bias,
                                        radius_weight,
                                        weight_scale,
                                        args.seed,
                                        args.quant_bits,
                                    )
                                )
    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            min(row["old_acc"] / args.old_target, row["seen_new_acc"] / 0.80, row["min_seen_new_class_acc"] / args.seen_new_target),
            float(row["min_seen_new_class_acc"]),
            float(row["old_acc"]),
        ),
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": sorted(old_splits),
        "eligible_new_tx_count": len(new_splits),
        "selected_pair_candidate_count": len(selected),
        "selected_pair_candidates": selected,
        "combo_size": int(args.combo_size),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "old_target": float(args.old_target),
        "seen_new_target_per_class": float(args.seen_new_target),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "combo_rows": rows,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "method",
        "new_tx_ids",
        "old_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "H_old_new",
        "passes_joint_target",
        "per_new_acc",
        "support_count",
        "stored_prototype_count",
        "stored_weight_count",
        "stored_quantized_count",
        "stored_support_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["new_tx_ids"] = ",".join(row["new_tx_ids"])
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)

    print(
        json.dumps(
            {
                "eligible_new_tx_count": len(new_splits),
                "selected_pair_candidate_count": len(selected),
                "joint_pass_count": summary["joint_pass_count"],
                "best": rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
