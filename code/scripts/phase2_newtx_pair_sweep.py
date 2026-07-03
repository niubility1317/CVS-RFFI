#!/usr/bin/env python3
"""Search Stage2-C new-TX combinations with target-support prototype/kNN heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == truth))


def _harmonic(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def _knn_predict(query: np.ndarray, support: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    similarities = query @ support.T
    k_eff = min(max(1, int(k)), int(support.shape[0]))
    predictions: list[str] = []
    for row in similarities:
        top_idx = np.argsort(row)[-k_eff:][::-1]
        counts: dict[str, int] = {}
        scores: dict[str, float] = {}
        for idx in top_idx:
            label = str(labels[int(idx)])
            counts[label] = counts.get(label, 0) + 1
            scores[label] = scores.get(label, 0.0) + float(row[int(idx)])
        predictions.append(max(scores, key=lambda label: (counts[label], scores[label])))
    return np.asarray(predictions, dtype=object)


def _prototype_predict(query: np.ndarray, support: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_values = sorted({str(label) for label in labels.tolist()})
    prototypes = []
    for label in label_values:
        prototypes.append(support[labels == label].mean(axis=0))
    prototype_matrix = _normalize_rows(np.vstack(prototypes))
    similarities = query @ prototype_matrix.T
    return np.asarray([label_values[int(idx)] for idx in np.argmax(similarities, axis=1)], dtype=object)


def _predict(method: str, query: np.ndarray, support: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if method == "proto":
        return _prototype_predict(query, support, labels)
    if method.startswith("knn"):
        return _knn_predict(query, support, labels, int(method[3:]))
    raise ValueError(f"Unsupported method: {method}")


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
    support = shuffled[:support_count]
    query = shuffled[support_count : support_count + query_count]
    return np.asarray(support, dtype=int), np.asarray(query, dtype=int)


def _role_values(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "dataset_role" not in data.files:
        raise KeyError("feature NPZ missing dataset_role")
    return np.asarray(data["dataset_role"], dtype=object).astype(str)


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
    role_values = [str(role) for role in roles_to_match]
    for tx in candidates:
        for role in role_values:
            support, query = _split_indices(tx_ids, roles, tx, role, support_count, query_count, seed)
            if support.size and query.size:
                out[tx] = (support, query)
                break
    return out


def _evaluate_combo(
    combo: tuple[str, ...],
    method: str,
    features: np.ndarray,
    tx_ids: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_target: float,
    new_target: float,
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

    support = features[np.asarray(support_indices, dtype=int)]
    labels = np.asarray(support_labels, dtype=object)
    old_query_idx = np.asarray(old_query_indices, dtype=int)
    new_query_idx = np.asarray(new_query_indices, dtype=int)
    query_idx = np.concatenate([old_query_idx, new_query_idx])
    query = features[query_idx]
    truth = tx_ids[query_idx]
    pred = _predict(method, query, support, labels)

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
        "passes_seen_new_average_target": seen_new_acc >= new_target,
        "passes_seen_new_per_class_target": min_new_acc >= new_target,
        "passes_joint_target": old_acc >= old_target and min_new_acc >= new_target,
        "support_count": int(len(support_indices)),
        "old_query_count": int(old_query_idx.size),
        "new_query_count": int(new_query_idx.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--methods", default="proto,knn1,knn3,knn5")
    parser.add_argument("--old_roles", default="target_old")
    parser.add_argument("--new_roles", default="target_new,new")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=422001)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=40)
    parser.add_argument("--query_per_new", type=int, default=40)
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--seen_new_target", type=float, default=0.65)
    parser.add_argument("--max_pair_candidates", type=int, default=50)
    parser.add_argument("--single_new_floor", type=float, default=0.50)
    parser.add_argument("--single_old_floor", type=float, default=0.55)
    args = parser.parse_args()

    feature_npz = Path(args.feature_npz)
    data = np.load(feature_npz, allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = _role_values(data)

    old_candidates = _parse_csv(args.old_tx_ids)
    explicit_new = _parse_csv(args.candidate_new_tx_ids)
    old_roles = _parse_csv(args.old_roles)
    new_roles = _parse_csv(args.new_roles)
    if explicit_new:
        new_candidates = explicit_new
    else:
        role_mask = np.isin(roles, np.asarray(new_roles, dtype=object))
        new_candidates = sorted({str(tx) for tx in tx_ids[role_mask].tolist()})
    old_splits = _eligible_tx(
        tx_ids, roles, old_candidates, old_roles, args.k_old, args.query_per_old, args.seed
    )
    new_splits = _eligible_tx(
        tx_ids, roles, new_candidates, new_roles, args.k_new, args.query_per_new, args.seed
    )
    if not old_splits:
        raise RuntimeError(f"No eligible old TX splits found for roles={old_roles}")
    if not new_splits:
        raise RuntimeError(f"No eligible new TX splits found for roles={new_roles}")
    methods = _parse_csv(args.methods)

    single_rows: list[dict[str, Any]] = []
    for tx in sorted(new_splits):
        for method in methods:
            single_rows.append(
                _evaluate_combo(
                    (tx,),
                    method,
                    features,
                    tx_ids,
                    old_splits,
                    new_splits,
                    args.old_target,
                    args.seen_new_target,
                )
            )
    single_rows.sort(
        key=lambda row: (
            bool(row["old_acc"] >= args.single_old_floor and row["min_seen_new_class_acc"] >= args.single_new_floor),
            float(row["H_old_new"]),
            float(row["min_seen_new_class_acc"]),
            float(row["old_acc"]),
        ),
        reverse=True,
    )

    selected: list[str] = []
    for row in single_rows:
        for tx in row["new_tx_ids"]:
            if tx not in selected:
                selected.append(tx)
        if len(selected) >= int(args.max_pair_candidates):
            break
    selected = sorted(selected)

    combo_rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(selected, int(args.combo_size)):
        for method in methods:
            combo_rows.append(
                _evaluate_combo(
                    tuple(combo),
                    method,
                    features,
                    tx_ids,
                    old_splits,
                    new_splits,
                    args.old_target,
                    args.seen_new_target,
                )
            )
    combo_rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            float(row["min_seen_new_class_acc"]),
            float(row["old_acc"]),
            float(row["H_old_new"]),
        ),
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_npz": str(feature_npz),
        "old_tx_ids": sorted(old_splits),
        "eligible_new_tx_count": len(new_splits),
        "selected_pair_candidate_count": len(selected),
        "selected_pair_candidates": selected,
        "methods": methods,
        "old_roles": old_roles,
        "new_roles": new_roles,
        "combo_size": int(args.combo_size),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "old_target": float(args.old_target),
        "seen_new_target_per_class": float(args.seen_new_target),
        "joint_pass_count": int(sum(1 for row in combo_rows if row["passes_joint_target"])),
        "single_rows": single_rows[:200],
        "combo_rows": combo_rows,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

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
        "old_query_count",
        "new_query_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combo_rows:
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
                "best": combo_rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
