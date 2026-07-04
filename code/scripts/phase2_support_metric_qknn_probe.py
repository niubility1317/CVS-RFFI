#!/usr/bin/env python3
"""Probe support-only metric transforms before compressed qKNN.

The transform is fitted from target support only. Query labels are used only
for audit metrics. The deployed state stores transform scalars plus quantized
support codes, not raw IQ or raw support samples.
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
import phase2_metric_adapter_probe as metric
import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


Split = tuple[np.ndarray, np.ndarray]


def _collect_support(
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    for label in old_labels:
        support, _query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
    for label in new_labels:
        support, _query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
    return np.asarray(support_indices, dtype=int), np.asarray(support_labels, dtype=object).astype(str)


def _evaluate_metric_qknn(
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
    transform_mode: str,
    transform_strength: float,
    topm: int,
    proto_mix: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
    balanced_assignment: bool,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    support_indices, support_labels = _collect_support(old_splits, new_splits, old_labels, new_labels)
    old_query: list[int] = []
    new_query: list[int] = []
    for label in old_labels:
        _support, query = old_splits[label]
        old_query.extend(query.tolist())
    for label in new_labels:
        _support, query = new_splits[label]
        new_query.extend(query.tolist())
    query_indices = np.asarray(old_query + new_query, dtype=int)
    old_count = len(old_query)

    transform = metric._fit_transform(
        features[support_indices],
        support_labels,
        str(transform_mode),
        float(transform_strength),
    )
    adapted = metric._apply_transform(features, transform)
    scores, radii, proto_sim = base._class_scores(
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        scenarios=scenarios,
        class_labels=old_labels + new_labels,
        old_labels=set(old_labels),
        topm=int(topm),
        proto_mix=float(proto_mix),
        radius_norm=float(radius_norm),
        old_bias=float(old_bias),
        neg_lambda=float(neg_lambda),
        neg_threshold=float(neg_threshold),
        neg_margin=float(neg_margin),
        mutual_only=bool(mutual_only),
        scenario_aware=bool(scenario_aware),
    )
    if balanced_assignment:
        pred = base._balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    else:
        pred = base._predict(scores, old_labels + new_labels)
    truth = tx_ids[query_indices]
    metrics = base._metrics(
        pred,
        truth,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        old_target=old_target,
        old_floor=old_floor,
        new_target=new_target,
        new_floor=new_floor,
    )
    row: dict[str, Any] = {
        "transform_mode": str(transform_mode),
        "transform_strength": float(transform_strength),
        "topm": int(topm),
        "proto_mix": float(proto_mix),
        "radius_norm": float(radius_norm),
        "old_bias": float(old_bias),
        "neg_lambda": float(neg_lambda),
        "neg_threshold": float(neg_threshold),
        "neg_margin": float(neg_margin),
        "mutual_only": bool(mutual_only),
        "scenario_aware": bool(scenario_aware),
        "balanced_assignment": bool(balanced_assignment),
        "stored_quantized_support_code_count": int(support_indices.size),
        "stored_raw_support_count": 0,
        "stored_class_prototype_count": int(len(old_labels) + len(new_labels)),
        "stored_transform_scalars": int(2 * features.shape[1]),
        "transform_scale_min": float(np.min(transform["scale"])),
        "transform_scale_max": float(np.max(transform["scale"])),
        "transform_scale_mean": float(np.mean(transform["scale"])),
        "class_radii": radii,
        "max_offdiag_proto_sim": float(np.max(proto_sim - np.eye(proto_sim.shape[0]) * 2.0)) if proto_sim.size else 0.0,
    }
    row.update({f"query_{key}": value for key, value in metrics.items()})
    row["query_rank_score"] = base._rank(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--seed_start", type=int, default=421000)
    parser.add_argument("--seed_count", type=int, default=120)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=70)
    parser.add_argument("--query_per_new", type=int, default=70)
    parser.add_argument("--pool_per_old", type=int, default=10)
    parser.add_argument("--pool_per_new", type=int, default=10)
    parser.add_argument("--transform_modes", default="identity,diag_fisher,diag_whiten_fisher")
    parser.add_argument("--transform_strengths", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--topm_grid", default="4")
    parser.add_argument("--proto_mix_grid", default="0.25")
    parser.add_argument("--radius_norm_grid", default="0")
    parser.add_argument("--old_bias_grid", default="0.001")
    parser.add_argument("--neg_lambda_grid", default="0.7")
    parser.add_argument("--neg_threshold_grid", default="0.75")
    parser.add_argument("--neg_margin_grid", default="0.01")
    parser.add_argument("--mutual_only_grid", default="true")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--balanced_assignment", action="store_true")
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
            old_splits = active._as_eval_splits(old_raw)
            new_splits = active._as_eval_splits(new_raw)
            for mode in qknn._parse_csv(args.transform_modes):
                for strength in qknn._parse_float_csv(args.transform_strengths):
                    for topm in qknn._parse_int_csv(args.topm_grid):
                        for proto_mix in qknn._parse_float_csv(args.proto_mix_grid):
                            for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                                for old_bias in qknn._parse_float_csv(args.old_bias_grid):
                                    for neg_lambda in qknn._parse_float_csv(args.neg_lambda_grid):
                                        for neg_threshold in qknn._parse_float_csv(args.neg_threshold_grid):
                                            for neg_margin in qknn._parse_float_csv(args.neg_margin_grid):
                                                for mutual_raw in qknn._parse_csv(args.mutual_only_grid):
                                                    row = _evaluate_metric_qknn(
                                                        features=features,
                                                        tx_ids=tx_ids,
                                                        scenarios=scenarios,
                                                        old_splits=old_splits,
                                                        new_splits=new_splits,
                                                        old_labels=old_labels,
                                                        new_labels=new_labels,
                                                        transform_mode=mode,
                                                        transform_strength=float(strength),
                                                        topm=int(topm),
                                                        proto_mix=float(proto_mix),
                                                        radius_norm=float(radius_norm),
                                                        old_bias=float(old_bias),
                                                        neg_lambda=float(neg_lambda),
                                                        neg_threshold=float(neg_threshold),
                                                        neg_margin=float(neg_margin),
                                                        mutual_only=str(mutual_raw).lower() == "true",
                                                        scenario_aware=bool(args.scenario_aware),
                                                        balanced_assignment=bool(args.balanced_assignment),
                                                        old_target=float(args.old_target),
                                                        old_floor=float(args.old_floor),
                                                        new_target=float(args.seen_new_target),
                                                        new_floor=float(args.seen_new_floor),
                                                    )
                                                    row["seed"] = int(seed)
                                                    row["support_selection_policy"] = policy
                                                    row["k_old"] = int(args.k_old)
                                                    row["k_new"] = int(args.k_new)
                                                    row["pool_per_old"] = int(args.pool_per_old)
                                                    row["pool_per_new"] = int(args.pool_per_new)
                                                    rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["query_passes_new_floor75"]),
            float(row["query_min_seen_new_class_acc"]),
            float(row["query_seen_new_acc"]),
            float(row["query_old_acc"]),
            float(row["query_min_old_class_acc"]),
        ),
        reverse=True,
    )
    summary = {
        "diagnostic_scope": "SUPPORT_ONLY_METRIC_QKNN_COMPRESSED_NO_RAW_SUPPORT",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows_count": len(rows),
        "best": rows[:20],
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "transform_mode",
        "transform_strength",
        "topm",
        "proto_mix",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "neg_margin",
        "mutual_only",
        "scenario_aware",
        "balanced_assignment",
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
        "stored_transform_scalars",
        "transform_scale_min",
        "transform_scale_max",
        "transform_scale_mean",
        "max_offdiag_proto_sim",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["query_per_old_acc"] = json.dumps(row["query_per_old_acc"], ensure_ascii=False, sort_keys=True)
            out["query_per_new_acc"] = json.dumps(row["query_per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"best": rows[:5], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
