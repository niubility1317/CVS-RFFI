#!/usr/bin/env python3
"""Evaluate closed-form support-set ridge heads for Stage2-C."""

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
class RidgeHead:
    weight_matrix: np.ndarray
    class_labels: np.ndarray
    class_is_old: np.ndarray


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _augment_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([_normalize_rows(features), np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def train_ridge_head(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    old_labels: set[str] | None = None,
    l2: float = 1.0,
) -> RidgeHead:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    class_labels = np.asarray(sorted({str(label) for label in labels.tolist()}), dtype=object)
    y = np.zeros((labels.shape[0], class_labels.shape[0]), dtype=np.float64)
    label_to_idx = {label: idx for idx, label in enumerate(class_labels.tolist())}
    for row_idx, label in enumerate(labels.tolist()):
        y[row_idx, label_to_idx[str(label)]] = 1.0
    x = _augment_bias(np.asarray(support_features, dtype=np.float64))
    reg = np.eye(x.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    old = set(old_labels or set())
    return RidgeHead(
        weight_matrix=weights,
        class_labels=class_labels,
        class_is_old=np.asarray([label in old for label in class_labels.tolist()], dtype=bool),
    )


def predict_ridge_head(model: RidgeHead, query_features: np.ndarray, *, old_bias: float = 0.0) -> np.ndarray:
    scores = _augment_bias(np.asarray(query_features, dtype=np.float64)) @ model.weight_matrix
    scores = scores + (model.class_is_old.astype(np.float64) * float(old_bias))[None, :]
    return model.class_labels[np.argmax(scores, axis=1)]


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
            support, query = _split_indices(tx_ids, roles, str(tx), str(role), support_count, query_count, seed)
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
    l2: float,
    old_bias: float,
) -> dict[str, Any]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for tx, (support, query) in old_splits.items():
        support_indices.extend(support.tolist())
        support_labels.extend([tx] * int(support.size))
        old_query_indices.extend(query.tolist())
    for tx in combo:
        support, query = new_splits[tx]
        support_indices.extend(support.tolist())
        support_labels.extend([tx] * int(support.size))
        new_query_indices.extend(query.tolist())

    model = train_ridge_head(
        features[np.asarray(support_indices, dtype=int)],
        np.asarray(support_labels, dtype=object),
        old_labels=set(old_splits),
        l2=l2,
    )
    old_query_idx = np.asarray(old_query_indices, dtype=int)
    new_query_idx = np.asarray(new_query_indices, dtype=int)
    query_idx = np.concatenate([old_query_idx, new_query_idx])
    pred = predict_ridge_head(model, features[query_idx], old_bias=old_bias)
    truth = tx_ids[query_idx]
    old_acc = _accuracy(pred[: old_query_idx.size], truth[: old_query_idx.size])
    new_pred = pred[old_query_idx.size :]
    new_truth = truth[old_query_idx.size :]
    per_new: dict[str, float] = {}
    for tx in combo:
        mask = new_truth == tx
        per_new[tx] = _accuracy(new_pred[mask], new_truth[mask])
    seen_new_acc = float(np.mean(list(per_new.values()))) if per_new else 0.0
    min_new_acc = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "method": f"ridge_l2{l2:g}_oldbias{old_bias:g}",
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
        "stored_weight_count": int(model.weight_matrix.size),
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
    parser.add_argument("--l2_grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--old_bias_grid", default="0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2")
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    old_splits = _eligible_tx(
        tx_ids, roles, _parse_csv(args.old_tx_ids), _parse_csv(args.old_roles), args.k_old, args.query_per_old, args.seed
    )
    new_candidates = _parse_csv(args.candidate_new_tx_ids)
    if not new_candidates:
        role_mask = np.isin(roles, np.asarray(_parse_csv(args.new_roles), dtype=object))
        new_candidates = sorted({str(tx) for tx in tx_ids[role_mask].tolist()})
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
        for l2 in _parse_float_csv(args.l2_grid):
            for old_bias in _parse_float_csv(args.old_bias_grid):
                rows.append(
                    _evaluate_combo(
                        tuple(combo),
                        features,
                        tx_ids,
                        old_splits,
                        new_splits,
                        args.old_target,
                        args.seen_new_target,
                        l2,
                        old_bias,
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
        "stored_weight_count",
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
    print(json.dumps({"joint_pass_count": summary["joint_pass_count"], "best": rows[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
