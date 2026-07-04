#!/usr/bin/env python3
"""Probe support-only metric adapters for Phase2-C many-new enrollment.

The evaluated heads freeze the exported backbone features. Target support is
used only at enrollment time to estimate a lightweight feature transform and
compressed class prototypes. The deployed head stores transform parameters and
quantized prototypes, not raw support IQ or full-precision support embeddings.
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


def _class_stats(features: np.ndarray, labels: np.ndarray, class_labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    counts = []
    for label in class_labels:
        cls = features[labels == label]
        means.append(cls.mean(axis=0))
        counts.append(cls.shape[0])
    return np.asarray(means, dtype=np.float64), np.asarray(counts, dtype=np.float64)


def _fit_transform(features: np.ndarray, labels: np.ndarray, mode: str, strength: float) -> dict[str, Any]:
    class_labels = sorted({str(label) for label in labels.tolist()})
    x = qknn._normalize_rows(features)
    if mode == "identity":
        return {"mode": mode, "center": np.zeros(x.shape[1]), "scale": np.ones(x.shape[1])}

    center = x.mean(axis=0)
    xc = x - center
    means, counts = _class_stats(xc, labels, class_labels)
    global_mean = np.average(means, axis=0, weights=counts)
    between = np.average((means - global_mean) ** 2, axis=0, weights=counts)
    within_parts = []
    for label in class_labels:
        cls = xc[labels == label]
        cls_mean = cls.mean(axis=0, keepdims=True)
        within_parts.append((cls - cls_mean) ** 2)
    within = np.concatenate(within_parts, axis=0).mean(axis=0)

    if mode == "diag_fisher":
        raw = between / np.maximum(within, 1e-6)
        raw = raw / np.maximum(np.median(raw), 1e-6)
        scale = np.power(np.clip(raw, 0.05, 20.0), float(strength))
        return {"mode": mode, "center": center, "scale": scale}
    if mode == "diag_whiten_fisher":
        raw = between / np.maximum(within, 1e-6)
        raw = raw / np.maximum(np.median(raw), 1e-6)
        whiten = 1.0 / np.sqrt(np.maximum(xc.var(axis=0), 1e-5))
        scale = np.power(np.clip(raw, 0.05, 20.0), float(strength)) * np.power(whiten / np.median(whiten), 0.5)
        scale = np.clip(scale, 0.05, 20.0)
        return {"mode": mode, "center": center, "scale": scale}
    raise ValueError(f"Unsupported transform mode: {mode}")


def _apply_transform(features: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    x = qknn._normalize_rows(features)
    center = np.asarray(transform["center"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    return qknn._normalize_rows((x - center) * scale)


def _build_prototypes(
    features: np.ndarray,
    scenarios: np.ndarray,
    support_indices: list[int],
    support_labels: list[str],
    old_labels: set[str],
    *,
    scenario_aware: bool,
    quantize: bool,
) -> dict[str, np.ndarray]:
    support_idx = np.asarray(support_indices, dtype=int)
    support_features = qknn._normalize_rows(features[support_idx])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    support_scenarios = np.asarray(scenarios[support_idx], dtype=object).astype(str)
    proto_features: list[np.ndarray] = []
    proto_labels: list[str] = []
    proto_scenarios: list[str] = []
    proto_radii: list[float] = []
    if scenario_aware:
        keys = sorted({(str(label), str(scenario)) for label, scenario in zip(labels.tolist(), support_scenarios.tolist())})
    else:
        keys = [(str(label), "") for label in sorted(set(labels.tolist()))]
    for label, scenario in keys:
        mask = labels == label
        if scenario_aware:
            mask = mask & (support_scenarios == scenario)
        cls = support_features[mask]
        if cls.size == 0:
            continue
        prototype = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]
        proto_features.append(prototype)
        proto_labels.append(label)
        proto_scenarios.append(scenario)
        proto_radii.append(float(np.mean(1.0 - (cls @ prototype))))
    proto = np.asarray(proto_features, dtype=np.float64)
    if quantize:
        scale = 127.0
        quantized = np.clip(np.rint(proto * scale), -scale, scale).astype(np.int8)
        proto = qknn._normalize_rows(quantized.astype(np.float64) / scale)
    return {
        "features": proto,
        "labels": np.asarray(proto_labels, dtype=object),
        "is_old": np.asarray([str(label) in old_labels for label in proto_labels], dtype=bool),
        "radii_by_support": np.asarray(proto_radii, dtype=np.float64),
        "class_labels": np.asarray(sorted(set(proto_labels)), dtype=object),
        "scenarios": None if not scenario_aware else np.asarray(proto_scenarios, dtype=object),
        "stored_quantized_count": int(proto.shape[0]),
    }


def _evaluate(
    combo: list[str],
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    transform_mode: str,
    transform_strength: float,
    scenario_aware: bool,
    radius_norm: float,
    quantize: bool,
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

    support_arr = np.asarray(support_indices, dtype=int)
    transform = _fit_transform(features[support_arr], np.asarray(support_labels, dtype=object), transform_mode, transform_strength)
    adapted = _apply_transform(features, transform)
    bank = _build_prototypes(
        adapted,
        scenarios,
        support_indices,
        support_labels,
        set(old_labels),
        scenario_aware=scenario_aware,
        quantize=quantize,
    )
    query_idx = np.asarray(old_query_indices + new_query_indices, dtype=int)
    pred = qknn._predict_from_bank(
        bank,
        adapted[query_idx],
        topk=1,
        old_bias=0.0,
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
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in combo}
    old_acc = qknn._accuracy(old_pred, old_truth)
    seen_new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "transform_mode": transform_mode,
        "transform_strength": float(transform_strength),
        "scenario_aware": bool(scenario_aware),
        "radius_norm": float(radius_norm),
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_goal_floor75": min_new >= new_floor,
        "passes_joint_target": old_acc >= old_target and min_old >= old_floor and seen_new_acc >= new_target and min_new >= new_floor,
        "support_sample_count_used_for_adapter": int(len(support_indices)),
        "old_query_count": int(len(old_query_indices)),
        "new_query_count": int(len(new_query_indices)),
        "stored_quantized_count": int(bank["stored_quantized_count"]),
        "stored_support_count": 0,
        "stored_transform_scalars": int(2 * features.shape[1]),
        "transform_scale_min": float(np.min(transform["scale"])),
        "transform_scale_max": float(np.max(transform["scale"])),
        "transform_scale_mean": float(np.mean(transform["scale"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="source_proto_ranked_diverse")
    parser.add_argument("--transform_modes", default="identity,diag_fisher,diag_whiten_fisher")
    parser.add_argument("--transform_strengths", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--radius_norm_grid", default="0,0.1,0.2,0.3")
    parser.add_argument("--seed_start", type=int, default=422931)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=5)
    parser.add_argument("--k_new", type=int, default=5)
    parser.add_argument("--query_per_old", type=int, default=65)
    parser.add_argument("--query_per_new", type=int, default=65)
    parser.add_argument("--pool_per_old", type=int, default=15)
    parser.add_argument("--pool_per_new", type=int, default=15)
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
            old_splits = active._as_eval_splits(old_raw)
            new_splits = active._as_eval_splits(new_raw)
            for mode in qknn._parse_csv(args.transform_modes):
                for strength in qknn._parse_float_csv(args.transform_strengths):
                    for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                        row = _evaluate(
                            new_labels,
                            features=features,
                            tx_ids=tx_ids,
                            scenarios=scenarios,
                            old_splits=old_splits,
                            new_splits=new_splits,
                            old_labels=old_labels,
                            transform_mode=mode,
                            transform_strength=strength,
                            scenario_aware=bool(args.scenario_aware),
                            radius_norm=radius_norm,
                            quantize=True,
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
                        row["exclude_pool_from_query"] = bool(args.exclude_pool_from_query)
                        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["min_seen_new_class_acc"],
            row["seen_new_acc"],
            row["min_old_class_acc"],
            row["old_acc"],
        ),
        reverse=True,
    )
    summary = {
        "diagnostic_scope": "SUPPORT_ONLY_METRIC_ADAPTER_NO_RAW_SUPPORT_STORAGE",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows": rows,
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
        "radius_norm",
        "scenario_aware",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_goal_floor75",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "support_sample_count_used_for_adapter",
        "stored_quantized_count",
        "stored_transform_scalars",
        "stored_support_count",
        "old_query_count",
        "new_query_count",
        "transform_scale_min",
        "transform_scale_max",
        "transform_scale_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            out["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"best": rows[:5], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
