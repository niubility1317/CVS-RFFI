#!/usr/bin/env python3
"""Active support selection diagnostic for Stage2-C scenario-aware qKNN.

This script selects the deployed K-shot qKNN memory from a larger labeled
enrollment pool without using query correctness. It is an active-enrollment
diagnostic, not a strict K-shot label-budget claim when the pool size is larger
than K.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_source_guarded_qknn_sweep as qknn


Split = tuple[np.ndarray, np.ndarray]
EnrollmentSplit = tuple[np.ndarray, np.ndarray, np.ndarray]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _stable_order(indices: np.ndarray, *, label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(qknn._stable_seed(seed, label))
    return indices[rng.permutation(indices.size)].astype(int)


def _nearest_to_centroid(features: np.ndarray, indices: np.ndarray, k: int) -> list[int]:
    if indices.size == 0 or k <= 0:
        return []
    subset = features[indices]
    centroid = qknn._normalize_rows(subset.mean(axis=0, keepdims=True))[0]
    scores = subset @ centroid
    return indices[np.argsort(-scores)[:k]].astype(int).tolist()


def _farthest_fill(features: np.ndarray, candidates: np.ndarray, selected: list[int], k: int) -> list[int]:
    remaining = [int(idx) for idx in candidates.tolist() if int(idx) not in set(selected)]
    while len(selected) < k and remaining:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        sel_feat = features[np.asarray(selected, dtype=int)]
        rem_feat = features[np.asarray(remaining, dtype=int)]
        # Cosine distance to the nearest already selected support code.
        min_dist = np.min(1.0 - rem_feat @ sel_feat.T, axis=1)
        pick_pos = int(np.argmax(min_dist))
        selected.append(int(remaining.pop(pick_pos)))
    return selected


def _scenario_centroid(
    features: np.ndarray,
    indices: np.ndarray,
    scenarios: np.ndarray,
    k: int,
    *,
    diverse_fill: bool,
) -> list[int]:
    if indices.size == 0:
        return []
    selected: list[int] = []
    scenario_values = sorted(
        {str(value) for value in scenarios[indices].tolist()},
        key=lambda value: (-int(np.sum(scenarios[indices] == value)), value),
    )
    for scenario in scenario_values:
        if len(selected) >= k:
            break
        scenario_idx = indices[scenarios[indices] == scenario]
        for idx in _nearest_to_centroid(features, scenario_idx, 1):
            if idx not in selected:
                selected.append(int(idx))
                break
    remaining = np.asarray([int(idx) for idx in indices.tolist() if int(idx) not in set(selected)], dtype=int)
    if len(selected) < k and remaining.size:
        if diverse_fill:
            selected = _farthest_fill(features, remaining, selected, k)
        else:
            selected.extend(_nearest_to_centroid(features, remaining, k - len(selected)))
    return selected[:k]


def _scenario_ranked_select(
    features: np.ndarray,
    ordered: np.ndarray,
    scenarios: np.ndarray,
    k: int,
    *,
    diverse_fill: bool,
) -> list[int]:
    if ordered.size == 0:
        return []
    selected: list[int] = []
    for scenario in sorted({str(value) for value in scenarios[ordered].tolist()}):
        if len(selected) >= k:
            break
        scenario_idx = [int(idx) for idx in ordered.tolist() if str(scenarios[int(idx)]) == scenario]
        if scenario_idx:
            selected.append(scenario_idx[0])
    remaining = np.asarray([int(idx) for idx in ordered.tolist() if int(idx) not in set(selected)], dtype=int)
    if len(selected) < k and remaining.size:
        if diverse_fill:
            selected = _farthest_fill(features, remaining, selected, k)
        else:
            selected.extend(remaining[: k - len(selected)].astype(int).tolist())
    return selected[:k]


def _select_support(
    *,
    policy: str,
    label: str,
    role: str,
    candidates: np.ndarray,
    features: np.ndarray,
    scenarios: np.ndarray,
    source_probs: np.ndarray,
    source_label_to_idx: dict[str, int],
    source_prototypes: dict[str, np.ndarray],
    k: int,
    seed: int,
) -> list[int]:
    ordered = _stable_order(candidates, label=f"{role}:{label}", seed=seed)
    if policy == "old_stable_new_scenario_centroid":
        if role == "target_old":
            return ordered[:k].astype(int).tolist()
        return _scenario_centroid(features, ordered, scenarios, k, diverse_fill=False)
    if policy == "stable_first":
        return ordered[:k].astype(int).tolist()
    if policy == "centroid":
        return _nearest_to_centroid(features, ordered, k)
    if policy == "scenario_centroid":
        return _scenario_centroid(features, ordered, scenarios, k, diverse_fill=False)
    if policy == "scenario_diverse":
        return _scenario_centroid(features, ordered, scenarios, k, diverse_fill=True)
    if policy == "source_score_scenario_centroid" and role == "target_old":
        class_pos = source_label_to_idx[str(label)]
        ranked = ordered[np.argsort(-source_probs[ordered, class_pos])]
        return _scenario_centroid(features, ranked, scenarios, k, diverse_fill=False)
    if policy == "source_correct_scenario_diverse" and role == "target_old":
        class_pos = source_label_to_idx[str(label)]
        correct = ordered[np.argmax(source_probs[ordered], axis=1) == class_pos]
        fallback = np.asarray([int(idx) for idx in ordered.tolist() if int(idx) not in set(correct.tolist())], dtype=int)
        ranked = np.concatenate([correct, fallback])
        return _scenario_centroid(features, ranked, scenarios, k, diverse_fill=True)
    if policy == "source_proto_ranked_diverse" and role == "target_old" and str(label) in source_prototypes:
        prototype = source_prototypes[str(label)]
        ranked = ordered[np.argsort(-(features[ordered] @ prototype))]
        return _scenario_ranked_select(features, ranked, scenarios, k, diverse_fill=True)
    if policy == "source_score_ranked_diverse" and role == "target_old":
        class_pos = source_label_to_idx[str(label)]
        ranked = ordered[np.argsort(-source_probs[ordered, class_pos])]
        return _scenario_ranked_select(features, ranked, scenarios, k, diverse_fill=True)
    if policy in {
        "source_score_scenario_centroid",
        "source_correct_scenario_diverse",
        "source_proto_ranked_diverse",
        "source_score_ranked_diverse",
    }:
        return _scenario_centroid(features, ordered, scenarios, k, diverse_fill=False)
    raise ValueError(f"Unsupported policy: {policy}")


def _build_active_splits(
    *,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    features: np.ndarray,
    scenarios: np.ndarray,
    source_probs: np.ndarray,
    source_label_to_idx: dict[str, int],
    source_prototypes: dict[str, np.ndarray],
    labels: list[str],
    role: str,
    k: int,
    query_per_class: int,
    pool_per_class: int,
    policy: str,
    seed: int,
    exclude_pool_from_query: bool,
) -> dict[str, EnrollmentSplit]:
    splits: dict[str, EnrollmentSplit] = {}
    for label in labels:
        available = np.where((tx_ids == label) & (roles == role))[0].astype(int)
        if available.size < k + query_per_class:
            continue
        ordered = _stable_order(available, label=f"pool:{role}:{label}", seed=seed)
        pool_n = min(int(pool_per_class), int(available.size))
        pool = ordered[:pool_n]
        support = _select_support(
            policy=policy,
            label=label,
            role=role,
            candidates=pool,
            features=features,
            scenarios=scenarios,
            source_probs=source_probs,
            source_label_to_idx=source_label_to_idx,
            source_prototypes=source_prototypes,
            k=k,
            seed=seed,
        )
        support_set = set(support)
        if exclude_pool_from_query:
            query = ordered[pool_n:].astype(int)
        else:
            query = np.asarray([int(idx) for idx in ordered.tolist() if int(idx) not in support_set], dtype=int)
        enrollment_val = np.asarray([int(idx) for idx in pool.tolist() if int(idx) not in support_set], dtype=int)
        query = query[:query_per_class]
        if len(support) == k and query.size == query_per_class:
            splits[str(label)] = (np.asarray(support, dtype=int), query.astype(int), enrollment_val)
    return splits


def _as_eval_splits(splits: dict[str, EnrollmentSplit], *, use_enrollment_val: bool = False) -> dict[str, Split]:
    if use_enrollment_val:
        return {label: (support, enrollment_val) for label, (support, _, enrollment_val) in splits.items()}
    return {label: (support, query) for label, (support, query, _) in splits.items()}


def _prefixed_metrics(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_old_acc": row["old_acc"],
        f"{prefix}_min_old_class_acc": row["min_old_class_acc"],
        f"{prefix}_seen_new_acc": row["seen_new_acc"],
        f"{prefix}_min_seen_new_class_acc": row["min_seen_new_class_acc"],
        f"{prefix}_passes_joint_target": row["passes_joint_target"],
        f"{prefix}_per_old_acc": row["per_old_acc"],
        f"{prefix}_per_new_acc": row["per_new_acc"],
        f"{prefix}_old_query_count": row["old_query_count"],
        f"{prefix}_new_query_count": row["new_query_count"],
    }


def _joint_rank_score(row: dict[str, Any], *, old_target: float, old_floor: float, new_target: float, new_floor: float) -> float:
    return float(
        min(
            row["old_acc"] / old_target,
            row["min_old_class_acc"] / old_floor,
            row["seen_new_acc"] / new_target,
            row["min_seen_new_class_acc"] / new_floor,
        )
    )


def _mean_pairwise_cosine_distance(features: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return 0.0
    subset = features[indices]
    sim = subset @ subset.T
    upper = sim[np.triu_indices(indices.size, k=1)]
    return float(np.mean(np.clip(1.0 - upper, 0.0, 2.0)) / 2.0)


def _mean_centroid_similarity(features: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    subset = features[indices]
    centroid = qknn._normalize_rows(subset.mean(axis=0, keepdims=True))[0]
    return float(np.mean(subset @ centroid))


def _support_quality_metrics(
    *,
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    new_labels: tuple[str, ...],
    features: np.ndarray,
    scenarios: np.ndarray,
    source_prototypes: dict[str, np.ndarray],
    k_old: int,
    k_new: int,
) -> dict[str, float]:
    old_proto_scores: list[float] = []
    scenario_coverages: list[float] = []
    diversities: list[float] = []
    new_compactness: list[float] = []
    all_scenario_count = max(1, len({str(value) for value in scenarios.tolist()}))

    def add_common(indices: np.ndarray, k: int) -> None:
        scenario_cap = max(1, min(int(k), all_scenario_count))
        scenario_coverages.append(float(len({str(value) for value in scenarios[indices].tolist()}) / scenario_cap))
        diversities.append(_mean_pairwise_cosine_distance(features, indices))

    for label, (support, _) in old_splits.items():
        add_common(support, k_old)
        prototype = source_prototypes.get(str(label))
        if prototype is not None and support.size:
            old_proto_scores.append(float(np.mean(features[support] @ prototype)))

    for label in new_labels:
        support, _ = new_splits[str(label)]
        add_common(support, k_new)
        new_compactness.append(_mean_centroid_similarity(features, support))

    old_proto_mean = float(np.mean(old_proto_scores)) if old_proto_scores else 0.0
    old_proto_min = float(np.min(old_proto_scores)) if old_proto_scores else 0.0
    scenario_coverage_mean = float(np.mean(scenario_coverages)) if scenario_coverages else 0.0
    support_diversity_mean = float(np.mean(diversities)) if diversities else 0.0
    new_compactness_mean = float(np.mean(new_compactness)) if new_compactness else 0.0
    old_proto_mean_norm = (old_proto_mean + 1.0) / 2.0
    old_proto_min_norm = (old_proto_min + 1.0) / 2.0
    new_compactness_norm = (new_compactness_mean + 1.0) / 2.0
    score = (
        0.35 * old_proto_mean_norm
        + 0.20 * old_proto_min_norm
        + 0.25 * scenario_coverage_mean
        + 0.10 * support_diversity_mean
        + 0.10 * new_compactness_norm
    )
    return {
        "support_quality_score": float(score),
        "support_old_proto_mean": old_proto_mean,
        "support_old_proto_min": old_proto_min,
        "support_scenario_coverage_mean": scenario_coverage_mean,
        "support_diversity_mean": support_diversity_mean,
        "support_new_compactness_mean": new_compactness_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", default="")
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--policies", default="stable_first,centroid,scenario_centroid,scenario_diverse,source_score_scenario_centroid,source_correct_scenario_diverse,source_proto_ranked_diverse,source_score_ranked_diverse")
    parser.add_argument("--seed_start", type=int, default=422001)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=5)
    parser.add_argument("--k_new", type=int, default=5)
    parser.add_argument("--query_per_old", type=int, default=75)
    parser.add_argument("--query_per_new", type=int, default=75)
    parser.add_argument("--pool_per_old", type=int, default=80)
    parser.add_argument("--pool_per_new", type=int, default=80)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_new")
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--old_bias", type=float, default=0.0)
    parser.add_argument("--radius_norm", type=float, default=0.0)
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.88)
    parser.add_argument("--old_floor", type=float, default=0.80)
    parser.add_argument("--seen_new_target", type=float, default=0.85)
    parser.add_argument("--seen_new_floor", type=float, default=0.80)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    explicit_new_labels = qknn._parse_csv(args.new_tx_ids)
    candidate_new_labels = qknn._parse_csv(args.candidate_new_tx_ids)
    if explicit_new_labels:
        requested_new_labels = explicit_new_labels
    elif candidate_new_labels:
        requested_new_labels = candidate_new_labels
    else:
        requested_new_labels = sorted({str(label) for label in tx_ids[roles == "target_new"].tolist()})
    policies = qknn._parse_csv(args.policies)

    old_label_array = np.asarray(old_labels, dtype=object)
    source_idx = np.argmax(logits, axis=1)
    source_label = old_label_array[source_idx]
    source_conf, source_margin = qknn._softmax_confidence(logits)
    source_probs = _softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx_for_label = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx_for_label.size:
            source_prototypes[label] = qknn._normalize_rows(
                features[source_idx_for_label].mean(axis=0, keepdims=True)
            )[0]

    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for policy in policies:
            old_splits = _build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=old_labels,
                role=str(args.old_role),
                k=args.k_old,
                query_per_class=args.query_per_old,
                pool_per_class=args.pool_per_old,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            new_splits = _build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=requested_new_labels,
                role=str(args.new_role),
                k=args.k_new,
                query_per_class=args.query_per_new,
                pool_per_class=args.pool_per_new,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            if set(old_splits) != set(old_labels):
                continue
            if explicit_new_labels:
                combos = [tuple(explicit_new_labels)] if set(explicit_new_labels).issubset(new_splits) else []
            else:
                combos = list(itertools.combinations(sorted(new_splits), int(args.combo_size)))
            old_eval_splits = _as_eval_splits(old_splits)
            old_enroll_val_splits = _as_eval_splits(old_splits, use_enrollment_val=True)
            new_eval_splits = _as_eval_splits(new_splits)
            new_enroll_val_splits = _as_eval_splits(new_splits, use_enrollment_val=True)
            for combo in combos:
                support_quality = _support_quality_metrics(
                    old_splits=old_eval_splits,
                    new_splits=new_eval_splits,
                    new_labels=tuple(combo),
                    features=features,
                    scenarios=scenarios,
                    source_prototypes=source_prototypes,
                    k_old=args.k_old,
                    k_new=args.k_new,
                )
                enroll_val_row = qknn._evaluate_row(
                    tuple(combo),
                    features=features,
                    tx_ids=tx_ids,
                    source_label=source_label,
                    source_conf=source_conf,
                    source_margin=source_margin,
                    scenarios=scenarios,
                    old_splits=old_enroll_val_splits,
                    new_splits=new_enroll_val_splits,
                    old_labels=old_labels,
                    topk=args.topk,
                    old_bias=args.old_bias,
                    radius_norm=args.radius_norm,
                    source_guard_mode="none",
                    source_conf_min=0.0,
                    source_margin_min=0.0,
                    scenario_aware=bool(args.scenario_aware),
                    old_target=args.old_target,
                    old_floor=args.old_floor,
                    new_target=args.seen_new_target,
                    new_floor=args.seen_new_floor,
                )
                row = qknn._evaluate_row(
                    tuple(combo),
                    features=features,
                    tx_ids=tx_ids,
                    source_label=source_label,
                    source_conf=source_conf,
                    source_margin=source_margin,
                    scenarios=scenarios,
                    old_splits=old_eval_splits,
                    new_splits=new_eval_splits,
                    old_labels=old_labels,
                    topk=args.topk,
                    old_bias=args.old_bias,
                    radius_norm=args.radius_norm,
                    source_guard_mode="none",
                    source_conf_min=0.0,
                    source_margin_min=0.0,
                    scenario_aware=bool(args.scenario_aware),
                    old_target=args.old_target,
                    old_floor=args.old_floor,
                    new_target=args.seen_new_target,
                    new_floor=args.seen_new_floor,
                )
                row["seed"] = int(seed)
                row["support_selection_policy"] = policy
                row["pool_per_old"] = int(args.pool_per_old)
                row["pool_per_new"] = int(args.pool_per_new)
                row["old_role"] = str(args.old_role)
                row["new_role"] = str(args.new_role)
                row["exclude_pool_from_query"] = bool(args.exclude_pool_from_query)
                row["diagnostic_scope"] = "active_enrollment_pool_selection_no_query_correctness"
                row.update(support_quality)
                row.update(_prefixed_metrics("enroll_val", enroll_val_row))
                row["enroll_val_rank_score"] = _joint_rank_score(
                    enroll_val_row,
                    old_target=args.old_target,
                    old_floor=args.old_floor,
                    new_target=args.seen_new_target,
                    new_floor=args.seen_new_floor,
                )
                rows.append(row)

    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            min(
                row["old_acc"] / args.old_target,
                row["min_old_class_acc"] / args.old_floor,
                row["seen_new_acc"] / args.seen_new_target,
                row["min_seen_new_class_acc"] / args.seen_new_floor,
            ),
            row["old_acc"],
            row["seen_new_acc"],
        ),
        reverse=True,
    )

    summary = {
        "diagnostic_scope": "ACTIVE_ENROLLMENT_DIAGNOSTIC_not_strict_Kshot_when_pool_gt_K",
        "selection_uses_query_correctness": False,
        "selection_inputs": ["support/enrollment labels", "z_id feature geometry", "sat_scenarios", "source old-class logits", "source old-class prototypes"],
        "enrollment_validation": "Final support code is chosen from a labeled pool; unused pool samples are scored as enrollment validation, while final query can exclude the full pool.",
        "support_quality_sort_uses_query_correctness": False,
        "support_quality_score": "0.35*old_proto_mean_norm + 0.20*old_proto_min_norm + 0.25*scenario_coverage_mean + 0.10*support_diversity_mean + 0.10*new_compactness_norm",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "explicit_new_tx_ids": explicit_new_labels,
        "candidate_new_tx_ids": requested_new_labels,
        "eligible_new_tx_count": int(len(requested_new_labels)),
        "combo_size": int(args.combo_size),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "pool_per_old": int(args.pool_per_old),
        "pool_per_new": int(args.pool_per_new),
        "old_role": str(args.old_role),
        "new_role": str(args.new_role),
        "exclude_pool_from_query": bool(args.exclude_pool_from_query),
        "method": (
            f"active_select_qknn8_k{args.topk}_oldbias{args.old_bias:g}"
            f"_rnorm{args.radius_norm:g}{'_scenario' if args.scenario_aware else ''}"
        ),
        "old_target": float(args.old_target),
        "old_floor": float(args.old_floor),
        "seen_new_target": float(args.seen_new_target),
        "seen_new_floor": float(args.seen_new_floor),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "best": rows[:20],
        "best_by_support_quality": sorted(rows, key=lambda row: row["support_quality_score"], reverse=True)[:20],
        "best_by_enrollment_validation": sorted(
            rows,
            key=lambda row: (
                bool(row["enroll_val_passes_joint_target"]),
                row["enroll_val_rank_score"],
                row["enroll_val_old_acc"],
                row["enroll_val_seen_new_acc"],
            ),
            reverse=True,
        )[:20],
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "seed",
        "support_selection_policy",
        "method",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "support_count",
        "old_query_count",
        "new_query_count",
        "stored_quantized_count",
        "stored_support_count",
        "scenario_aware",
        "pool_per_old",
        "pool_per_new",
        "exclude_pool_from_query",
        "support_quality_score",
        "enroll_val_rank_score",
        "enroll_val_old_acc",
        "enroll_val_min_old_class_acc",
        "enroll_val_seen_new_acc",
        "enroll_val_min_seen_new_class_acc",
        "enroll_val_passes_joint_target",
        "enroll_val_per_old_acc",
        "enroll_val_per_new_acc",
        "support_old_proto_mean",
        "support_old_proto_min",
        "support_scenario_coverage_mean",
        "support_diversity_mean",
        "support_new_compactness_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["enroll_val_per_old_acc"] = json.dumps(
                row["enroll_val_per_old_acc"], ensure_ascii=False, sort_keys=True
            )
            csv_row["enroll_val_per_new_acc"] = json.dumps(
                row["enroll_val_per_new_acc"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(csv_row)

    print(
        json.dumps(
            {
                "joint_pass_count": summary["joint_pass_count"],
                "best": rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
