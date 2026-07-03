#!/usr/bin/env python3
"""Probe compressed prototype-code heads for Phase2-C qKNN enrollment.

The head uses K-shot support only to compute quantized class/scenario
prototype codes. It does not retain raw support IQ, full-precision support
embeddings, or individual support-sample codes.
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


def _build_prototype_bank(
    *,
    features: np.ndarray,
    support_indices: list[int],
    support_labels: list[str],
    old_labels: set[str],
    scenarios: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray]:
    support_idx = np.asarray(support_indices, dtype=int)
    support_features = qknn._normalize_rows(features[support_idx])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    support_scenarios = np.asarray(scenarios[support_idx], dtype=object).astype(str)
    proto_features: list[np.ndarray] = []
    proto_labels: list[str] = []
    proto_scenarios: list[str] = []
    proto_radii: list[float] = []

    if mode == "class_centroid":
        keys = [(str(label), "") for label in sorted(set(labels.tolist()))]
    elif mode == "scenario_centroid":
        keys = sorted({(str(label), str(scenario)) for label, scenario in zip(labels.tolist(), support_scenarios.tolist())})
    else:
        raise ValueError(f"Unsupported prototype mode: {mode}")

    for label, scenario in keys:
        mask = labels == label
        if mode == "scenario_centroid":
            mask = mask & (support_scenarios == scenario)
        class_features = support_features[mask]
        if class_features.size == 0:
            continue
        prototype = qknn._normalize_rows(class_features.mean(axis=0, keepdims=True))[0]
        proto_features.append(prototype)
        proto_labels.append(label)
        proto_scenarios.append(scenario)
        proto_radii.append(float(np.mean(1.0 - (class_features @ prototype))))

    proto = np.asarray(proto_features, dtype=np.float64)
    scale = 127.0
    quantized = np.clip(np.rint(proto * scale), -scale, scale).astype(np.int8)
    dequant = qknn._normalize_rows(quantized.astype(np.float64) / scale)
    return {
        "features": dequant,
        "labels": np.asarray(proto_labels, dtype=object),
        "is_old": np.asarray([str(label) in old_labels for label in proto_labels], dtype=bool),
        "radii_by_support": np.asarray(proto_radii, dtype=np.float64),
        "class_labels": np.asarray(sorted(set(proto_labels)), dtype=object),
        "scenarios": None if mode == "class_centroid" else np.asarray(proto_scenarios, dtype=object),
        "stored_quantized_count": int(dequant.shape[0]),
    }


def _evaluate_prototype_row(
    combo: tuple[str, ...],
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    prototype_mode: str,
    topk: int,
    old_bias: float,
    radius_norm: float,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query_indices.extend(query.tolist())
    for label in combo:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query_indices.extend(query.tolist())

    bank = _build_prototype_bank(
        features=features,
        support_indices=support_indices,
        support_labels=support_labels,
        old_labels=set(old_labels),
        scenarios=scenarios,
        mode=prototype_mode,
    )
    query_idx = np.asarray(old_query_indices + new_query_indices, dtype=int)
    scenario_aware = prototype_mode == "scenario_centroid"
    pred = qknn._predict_from_bank(
        bank,
        features[query_idx],
        topk=topk,
        old_bias=old_bias,
        radius_norm=radius_norm,
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
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in combo}
    old_acc = qknn._accuracy(old_pred, old_truth)
    seen_new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "method": f"prototype_compress_{prototype_mode}_topk{topk}_oldbias{old_bias:g}_rnorm{radius_norm:g}",
        "prototype_mode": prototype_mode,
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_joint_target": old_acc >= old_target and min_old >= old_floor and seen_new_acc >= new_target and min_new >= new_floor,
        "support_sample_count_used_for_compression": int(len(support_indices)),
        "old_query_count": int(len(old_query_indices)),
        "new_query_count": int(len(new_query_indices)),
        "stored_quantized_count": int(bank["stored_quantized_count"]),
        "stored_support_count": 0,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", default="")
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--policies", default="stable_first,centroid,scenario_centroid,scenario_diverse,source_proto_ranked_diverse,source_score_ranked_diverse")
    parser.add_argument("--prototype_modes", default="class_centroid,scenario_centroid")
    parser.add_argument("--seed_start", type=int, default=422001)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=5)
    parser.add_argument("--k_new", type=int, default=5)
    parser.add_argument("--query_per_old", type=int, default=65)
    parser.add_argument("--query_per_new", type=int, default=65)
    parser.add_argument("--pool_per_old", type=int, default=15)
    parser.add_argument("--pool_per_new", type=int, default=15)
    parser.add_argument("--topk_grid", default="1")
    parser.add_argument("--old_bias_grid", default="0")
    parser.add_argument("--radius_norm_grid", default="0")
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
    requested_new_labels = explicit_new_labels or candidate_new_labels or sorted({str(label) for label in tx_ids[roles == "target_new"].tolist()})
    old_label_array = np.asarray(old_labels, dtype=object)
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx_for_label = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx_for_label.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx_for_label].mean(axis=0, keepdims=True))[0]

    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for policy in qknn._parse_csv(args.policies):
            old_splits_raw = active._build_active_splits(
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
            new_splits_raw = active._build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=requested_new_labels,
                role="target_new",
                k=args.k_new,
                query_per_class=args.query_per_new,
                pool_per_class=args.pool_per_new,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            if set(old_splits_raw) != set(old_labels):
                continue
            old_splits = active._as_eval_splits(old_splits_raw)
            old_enroll_val_splits = active._as_eval_splits(old_splits_raw, use_enrollment_val=True)
            new_splits = active._as_eval_splits(new_splits_raw)
            new_enroll_val_splits = active._as_eval_splits(new_splits_raw, use_enrollment_val=True)
            if explicit_new_labels:
                combos = [tuple(explicit_new_labels)] if set(explicit_new_labels).issubset(new_splits) else []
            else:
                combos = list(itertools.combinations(sorted(new_splits), int(args.combo_size)))
            for combo in combos:
                for prototype_mode in qknn._parse_csv(args.prototype_modes):
                    for topk in qknn._parse_int_csv(args.topk_grid):
                        for old_bias in qknn._parse_float_csv(args.old_bias_grid):
                            for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                                enroll_val_row = _evaluate_prototype_row(
                                    tuple(combo),
                                    features=features,
                                    tx_ids=tx_ids,
                                    scenarios=scenarios,
                                    old_splits=old_enroll_val_splits,
                                    new_splits=new_enroll_val_splits,
                                    old_labels=old_labels,
                                    prototype_mode=prototype_mode,
                                    topk=topk,
                                    old_bias=old_bias,
                                    radius_norm=radius_norm,
                                    old_target=args.old_target,
                                    old_floor=args.old_floor,
                                    new_target=args.seen_new_target,
                                    new_floor=args.seen_new_floor,
                                )
                                row = _evaluate_prototype_row(
                                    tuple(combo),
                                    features=features,
                                    tx_ids=tx_ids,
                                    scenarios=scenarios,
                                    old_splits=old_splits,
                                    new_splits=new_splits,
                                    old_labels=old_labels,
                                    prototype_mode=prototype_mode,
                                    topk=topk,
                                    old_bias=old_bias,
                                    radius_norm=radius_norm,
                                    old_target=args.old_target,
                                    old_floor=args.old_floor,
                                    new_target=args.seen_new_target,
                                    new_floor=args.seen_new_floor,
                                )
                                row["seed"] = int(seed)
                                row["support_selection_policy"] = policy
                                row["pool_per_old"] = int(args.pool_per_old)
                                row["pool_per_new"] = int(args.pool_per_new)
                                row["exclude_pool_from_query"] = bool(args.exclude_pool_from_query)
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
        "diagnostic_scope": "PROTOTYPE_COMPRESS_PROBE_not_raw_support_storage",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "explicit_new_tx_ids": explicit_new_labels,
        "eligible_new_tx_count": int(len(requested_new_labels)),
        "combo_size": int(args.combo_size),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "pool_per_old": int(args.pool_per_old),
        "pool_per_new": int(args.pool_per_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "enroll_val_pass_count": int(sum(1 for row in rows if row["enroll_val_passes_joint_target"])),
        "enroll_val_and_query_pass_count": int(
            sum(1 for row in rows if row["enroll_val_passes_joint_target"] and row["passes_joint_target"])
        ),
        "best": rows[:20],
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
        "prototype_mode",
        "method",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "enroll_val_rank_score",
        "enroll_val_old_acc",
        "enroll_val_min_old_class_acc",
        "enroll_val_seen_new_acc",
        "enroll_val_min_seen_new_class_acc",
        "enroll_val_passes_joint_target",
        "enroll_val_per_old_acc",
        "enroll_val_per_new_acc",
        "support_sample_count_used_for_compression",
        "stored_quantized_count",
        "stored_support_count",
        "old_query_count",
        "new_query_count",
        "pool_per_old",
        "pool_per_new",
        "exclude_pool_from_query",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["enroll_val_per_old_acc"] = json.dumps(row["enroll_val_per_old_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["enroll_val_per_new_acc"] = json.dumps(row["enroll_val_per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)

    print(json.dumps({"joint_pass_count": summary["joint_pass_count"], "best": rows[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
