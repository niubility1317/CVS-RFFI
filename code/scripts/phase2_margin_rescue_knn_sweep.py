#!/usr/bin/env python3
"""Sweep margin-gated old rescue for class-max KNN Stage2-C heads."""

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


def _parse_float_csv(value: str | None) -> list[float]:
    return [float(part) for part in _parse_csv(value)]


def _parse_combo_specs(value: str | None) -> list[tuple[str, ...]]:
    combos = []
    for spec in _parse_csv(value):
        parts = tuple(part.strip() for part in spec.replace("+", "|").split("|") if part.strip())
        if len(parts) >= 2:
            combos.append(parts)
    return combos


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
            support, query = _split_indices(tx_ids, roles, tx, str(role), support_count, query_count, seed)
            if support.size and query.size:
                out[str(tx)] = (support, query)
                break
    return out


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == truth))


def _harmonic(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def _score_matrix(query: np.ndarray, support_by_class: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    labels = sorted(support_by_class)
    scores = [np.max(query @ support_by_class[label].T, axis=1) for label in labels]
    return labels, np.vstack(scores).T


def _evaluate(
    combo: tuple[str, ...],
    old_min_score: float,
    rescue_margin: float,
    features: np.ndarray,
    tx_ids: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_target: float,
    new_target: float,
) -> dict[str, Any]:
    support_by_class: dict[str, np.ndarray] = {}
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for tx, (support, query) in old_splits.items():
        support_by_class[tx] = features[support]
        old_query_indices.extend(query.tolist())
    for tx in combo:
        support, query = new_splits[tx]
        support_by_class[tx] = features[support]
        new_query_indices.extend(query.tolist())

    old_query_idx = np.asarray(old_query_indices, dtype=int)
    new_query_idx = np.asarray(new_query_indices, dtype=int)
    query_idx = np.concatenate([old_query_idx, new_query_idx])
    labels, scores = _score_matrix(features[query_idx], support_by_class)
    old_positions = np.asarray([idx for idx, label in enumerate(labels) if label in old_splits], dtype=int)
    new_positions = np.asarray([idx for idx, label in enumerate(labels) if label not in old_splits], dtype=int)
    top_idx = np.argmax(scores, axis=1)
    pred = np.asarray([labels[int(idx)] for idx in top_idx], dtype=object)

    old_score_matrix = scores[:, old_positions]
    new_score_matrix = scores[:, new_positions]
    best_old_local = np.argmax(old_score_matrix, axis=1)
    best_new_local = np.argmax(new_score_matrix, axis=1)
    best_old_score = old_score_matrix[np.arange(scores.shape[0]), best_old_local]
    best_new_score = new_score_matrix[np.arange(scores.shape[0]), best_new_local]
    best_old_label = np.asarray([labels[int(old_positions[idx])] for idx in best_old_local], dtype=object)

    top_is_new = np.asarray([labels[int(idx)] not in old_splits for idx in top_idx], dtype=bool)
    rescue_mask = top_is_new & (best_old_score >= float(old_min_score)) & ((best_new_score - best_old_score) <= float(rescue_margin))
    pred[rescue_mask] = best_old_label[rescue_mask]

    truth = tx_ids[query_idx]
    old_pred = pred[: old_query_idx.size]
    old_truth = truth[: old_query_idx.size]
    new_pred = pred[old_query_idx.size :]
    new_truth = truth[old_query_idx.size :]
    old_acc = _accuracy(old_pred, old_truth)
    per_new = {tx: _accuracy(new_pred[new_truth == tx], new_truth[new_truth == tx]) for tx in combo}
    seen_new_acc = float(np.mean(list(per_new.values()))) if per_new else 0.0
    min_new_acc = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "method": "classmax_knn_margin_old_rescue",
        "old_min_score": float(old_min_score),
        "rescue_margin": float(rescue_margin),
        "old_acc": old_acc,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new_acc,
        "H_old_new": _harmonic(old_acc, seen_new_acc),
        "per_new_acc": per_new,
        "rescued_count": int(np.sum(rescue_mask)),
        "passes_old_target": old_acc >= old_target,
        "passes_seen_new_per_class_target": min_new_acc >= new_target,
        "passes_joint_target": old_acc >= old_target and min_new_acc >= new_target,
        "support_count": int(sum(value[0].size for value in old_splits.values()) + sum(new_splits[tx][0].size for tx in combo)),
        "old_query_count": int(old_query_idx.size),
        "new_query_count": int(new_query_idx.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--combo_specs", default="")
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--old_roles", default="target_old")
    parser.add_argument("--new_roles", default="target_new,new")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=422001)
    parser.add_argument("--k_old", type=int, default=20)
    parser.add_argument("--k_new", type=int, default=20)
    parser.add_argument("--query_per_old", type=int, default=40)
    parser.add_argument("--query_per_new", type=int, default=40)
    parser.add_argument("--old_target", type=float, default=0.88)
    parser.add_argument("--seen_new_target", type=float, default=0.85)
    parser.add_argument("--old_min_score_values", default="0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98")
    parser.add_argument("--rescue_margin_values", default="0,0.001,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03")
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    old_roles = _parse_csv(args.old_roles)
    new_roles = _parse_csv(args.new_roles)
    old_splits = _eligible_tx(
        tx_ids, roles, _parse_csv(args.old_tx_ids), old_roles, args.k_old, args.query_per_old, args.seed
    )
    explicit_new = _parse_csv(args.candidate_new_tx_ids)
    if explicit_new:
        new_candidates = explicit_new
    else:
        new_candidates = sorted({str(tx) for tx in tx_ids[np.isin(roles, np.asarray(new_roles, dtype=object))].tolist()})
    new_splits = _eligible_tx(
        tx_ids, roles, new_candidates, new_roles, args.k_new, args.query_per_new, args.seed
    )
    combos = _parse_combo_specs(args.combo_specs)
    if not combos:
        combos = list(itertools.combinations(sorted(new_splits), int(args.combo_size)))
    combos = [tuple(tx for tx in combo if tx in new_splits) for combo in combos]
    combos = [combo for combo in combos if len(combo) == int(args.combo_size)]
    old_min_score_values = _parse_float_csv(args.old_min_score_values)
    rescue_margin_values = _parse_float_csv(args.rescue_margin_values)

    rows = []
    for combo in combos:
        for old_min_score in old_min_score_values:
            for rescue_margin in rescue_margin_values:
                rows.append(
                    _evaluate(
                        combo,
                        old_min_score,
                        rescue_margin,
                        features,
                        tx_ids,
                        old_splits,
                        new_splits,
                        args.old_target,
                        args.seen_new_target,
                    )
                )
    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            float(row["min_seen_new_class_acc"]),
            float(row["old_acc"]),
            float(row["H_old_new"]),
        ),
        reverse=True,
    )
    summary = {
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": sorted(old_splits),
        "combo_count": len(combos),
        "old_min_score_values": old_min_score_values,
        "rescue_margin_values": rescue_margin_values,
        "combo_size": int(args.combo_size),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "old_target": float(args.old_target),
        "seen_new_target_per_class": float(args.seen_new_target),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    fields = [
        "method",
        "new_tx_ids",
        "old_min_score",
        "rescue_margin",
        "old_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "H_old_new",
        "rescued_count",
        "passes_joint_target",
        "per_new_acc",
        "support_count",
        "old_query_count",
        "new_query_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fields}
            csv_row["new_tx_ids"] = ",".join(row["new_tx_ids"])
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)
    print(json.dumps({"joint_pass_count": summary["joint_pass_count"], "best": rows[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
