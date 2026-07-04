#!/usr/bin/env python3
"""Probe support-only class-bias calibrated qKNN for Phase2-C many-new.

This keeps the qKNN route but adds one deployable scalar per enrolled class.
Biases are fitted from target support leave-one-out predictions only, so query
labels are not used for calibration. The deployed head stores int8 support
codes plus class biases, not raw support IQ.
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


def _collect(
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[list[int], list[str], list[int], list[int]]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query: list[int] = []
    new_query: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query.extend(query.tolist())
    for label in new_labels:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query.extend(query.tolist())
    return support_indices, support_labels, old_query, new_query


def _predict_biased(
    *,
    bank: dict[str, np.ndarray],
    query_features: np.ndarray,
    query_scenarios: np.ndarray | None,
    class_bias: dict[str, float],
    topk: int,
    radius_norm: float,
    scenario_aware: bool,
    exclude_bank_positions: np.ndarray | None = None,
) -> np.ndarray:
    query = qknn._normalize_rows(query_features)
    if not scenario_aware:
        return _predict_biased_block(
            bank,
            query,
            class_bias=class_bias,
            topk=topk,
            radius_norm=radius_norm,
            exclude_bank_positions=exclude_bank_positions,
        )
    support_scenarios = bank.get("scenarios")
    if support_scenarios is None or query_scenarios is None:
        raise ValueError("scenario_aware requires support and query scenarios")
    query_scenarios = np.asarray(query_scenarios, dtype=object).astype(str)
    pred = np.empty(query.shape[0], dtype=object)
    for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
        query_mask = query_scenarios == scenario
        support_mask = support_scenarios == scenario
        if int(np.sum(support_mask)) < max(1, int(topk)) or len(set(bank["labels"][support_mask].tolist())) < 2:
            support_mask = np.ones_like(support_mask, dtype=bool)
        positions = np.where(support_mask)[0].astype(int)
        sub_exclude = None
        if exclude_bank_positions is not None:
            pos_to_sub = {int(pos): sub for sub, pos in enumerate(positions.tolist())}
            sub_exclude = np.asarray(
                [pos_to_sub.get(int(pos), -1) for pos in exclude_bank_positions[query_mask].tolist()],
                dtype=int,
            )
        sub_bank = {
            "features": bank["features"][support_mask],
            "labels": bank["labels"][support_mask],
            "is_old": bank["is_old"][support_mask],
            "radii_by_support": bank["radii_by_support"][support_mask],
        }
        pred[query_mask] = _predict_biased_block(
            sub_bank,
            query[query_mask],
            class_bias=class_bias,
            topk=topk,
            radius_norm=radius_norm,
            exclude_bank_positions=sub_exclude,
        )
    return pred


def _predict_biased_block(
    bank: dict[str, np.ndarray],
    query_features: np.ndarray,
    *,
    class_bias: dict[str, float],
    topk: int,
    radius_norm: float,
    exclude_bank_positions: np.ndarray | None,
) -> np.ndarray:
    scores = qknn._normalize_rows(query_features) @ bank["features"].T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(bank["radii_by_support"], 1e-4), float(radius_norm))[None, :]
        scores = 1.0 - ((1.0 - scores) / denom)
    bias = np.asarray([float(class_bias.get(str(label), 0.0)) for label in bank["labels"].tolist()], dtype=np.float64)
    scores = scores + bias[None, :]
    if exclude_bank_positions is not None:
        for row, pos in enumerate(exclude_bank_positions.tolist()):
            if 0 <= int(pos) < scores.shape[1]:
                scores[row, int(pos)] = -1e9
    return qknn._classwise_topk_predict(scores, bank["labels"], int(topk))


def _metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: qknn._accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in new_labels}
    old_acc = qknn._accuracy(old_pred, old_truth)
    new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_goal_floor75": min_new >= float(new_floor),
        "passes_joint_target": old_acc >= float(old_target)
        and min_old >= float(old_floor)
        and new_acc >= float(new_target)
        and min_new >= float(new_floor),
    }


def _score(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    joint_ratio = min(
        float(metrics["old_acc"]) / 0.80,
        float(metrics["min_old_class_acc"]) / 0.75,
        float(metrics["seen_new_acc"]) / 0.75,
        float(metrics["min_seen_new_class_acc"]) / 0.75,
    )
    return (
        float(bool(metrics["passes_joint_target"])),
        joint_ratio,
        float(metrics["min_seen_new_class_acc"]),
        float(metrics["seen_new_acc"]),
        float(metrics["min_old_class_acc"]),
        float(metrics["old_acc"]),
    )


def _fit_bias_support_loo(
    *,
    bank: dict[str, np.ndarray],
    support_features: np.ndarray,
    support_labels: list[str],
    support_scenarios: np.ndarray,
    class_labels: list[str],
    topk: int,
    radius_norm: float,
    scenario_aware: bool,
    bias_grid: list[float],
    rounds: int,
    old_labels: list[str],
    new_labels: list[str],
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    bias = {label: 0.0 for label in class_labels}
    truth = np.asarray(support_labels, dtype=object)
    old_count = sum(1 for label in support_labels if label in set(old_labels))
    # Reorder support calibration rows as old first, new second for metric reporting.
    old_pos = [idx for idx, label in enumerate(support_labels) if label in set(old_labels)]
    new_pos = [idx for idx, label in enumerate(support_labels) if label in set(new_labels)]
    order = np.asarray(old_pos + new_pos, dtype=int)
    ordered_features = support_features[order]
    ordered_truth = truth[order]
    ordered_scenarios = support_scenarios[order]
    ordered_exclude = order.astype(int)

    def evaluate(candidate: dict[str, float]) -> dict[str, Any]:
        pred = _predict_biased(
            bank=bank,
            query_features=ordered_features,
            query_scenarios=ordered_scenarios,
            class_bias=candidate,
            topk=topk,
            radius_norm=radius_norm,
            scenario_aware=scenario_aware,
            exclude_bank_positions=ordered_exclude,
        )
        return _metrics(
            pred,
            ordered_truth,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            old_target=old_target,
            old_floor=old_floor,
            new_target=new_target,
            new_floor=new_floor,
        )

    best_metrics = evaluate(bias)
    for _ in range(max(1, int(rounds))):
        changed = False
        for label in class_labels:
            label_best = best_metrics
            label_value = bias[label]
            for value in bias_grid:
                trial = dict(bias)
                trial[label] = float(value)
                trial_metrics = evaluate(trial)
                if _score(trial_metrics) > _score(label_best):
                    label_best = trial_metrics
                    label_value = float(value)
            if label_value != bias[label]:
                bias[label] = label_value
                best_metrics = label_best
                changed = True
        if not changed:
            break
    return bias, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="source_proto_ranked_diverse")
    parser.add_argument("--topk_grid", default="1,3,5")
    parser.add_argument("--radius_norm_grid", default="0,0.1,0.2,0.3")
    parser.add_argument("--bias_grid", default="-0.10,-0.06,-0.03,0,0.03,0.06,0.10,0.14,0.18,0.22")
    parser.add_argument("--bias_rounds", type=int, default=4)
    parser.add_argument("--seed_start", type=int, default=422931)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=20)
    parser.add_argument("--k_new", type=int, default=20)
    parser.add_argument("--query_per_old", type=int, default=60)
    parser.add_argument("--query_per_new", type=int, default=60)
    parser.add_argument("--pool_per_old", type=int, default=50)
    parser.add_argument("--pool_per_new", type=int, default=50)
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
    class_labels = old_labels + new_labels
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
            support_indices, support_labels, old_query, new_query = _collect(old_splits, new_splits, old_labels, new_labels)
            bank = qknn._build_support_bank(
                features,
                support_indices,
                support_labels,
                set(old_labels),
                support_scenarios=scenarios[np.asarray(support_indices, dtype=int)] if bool(args.scenario_aware) else None,
            )
            query_idx = np.asarray(old_query + new_query, dtype=int)
            truth = tx_ids[query_idx]
            for topk in qknn._parse_int_csv(args.topk_grid):
                for radius_norm in qknn._parse_float_csv(args.radius_norm_grid):
                    bias, support_loo = _fit_bias_support_loo(
                        bank=bank,
                        support_features=features[np.asarray(support_indices, dtype=int)],
                        support_labels=support_labels,
                        support_scenarios=scenarios[np.asarray(support_indices, dtype=int)],
                        class_labels=class_labels,
                        topk=int(topk),
                        radius_norm=float(radius_norm),
                        scenario_aware=bool(args.scenario_aware),
                        bias_grid=qknn._parse_float_csv(args.bias_grid),
                        rounds=int(args.bias_rounds),
                        old_labels=old_labels,
                        new_labels=new_labels,
                        old_target=float(args.old_target),
                        old_floor=float(args.old_floor),
                        new_target=float(args.seen_new_target),
                        new_floor=float(args.seen_new_floor),
                    )
                    pred = _predict_biased(
                        bank=bank,
                        query_features=features[query_idx],
                        query_scenarios=scenarios[query_idx],
                        class_bias=bias,
                        topk=int(topk),
                        radius_norm=float(radius_norm),
                        scenario_aware=bool(args.scenario_aware),
                    )
                    query_metrics = _metrics(
                        pred,
                        truth,
                        old_count=len(old_query),
                        old_labels=old_labels,
                        new_labels=new_labels,
                        old_target=float(args.old_target),
                        old_floor=float(args.old_floor),
                        new_target=float(args.seen_new_target),
                        new_floor=float(args.seen_new_floor),
                    )
                    row: dict[str, Any] = {
                        "seed": int(seed),
                        "support_selection_policy": policy,
                        "topk": int(topk),
                        "radius_norm": float(radius_norm),
                        "scenario_aware": bool(args.scenario_aware),
                        "k_old": int(args.k_old),
                        "k_new": int(args.k_new),
                        "pool_per_old": int(args.pool_per_old),
                        "pool_per_new": int(args.pool_per_new),
                        "exclude_pool_from_query": bool(args.exclude_pool_from_query),
                        "bias": bias,
                        "stored_support_count": int(len(support_indices)),
                        "stored_bias_scalars": int(len(class_labels)),
                        "stored_quantized_support_code_count": int(len(support_indices)),
                    }
                    row.update({f"query_{key}": value for key, value in query_metrics.items()})
                    row.update({f"support_loo_{key}": value for key, value in support_loo.items()})
                    rows.append(row)

    rows.sort(
        key=lambda row: (
            row["query_min_seen_new_class_acc"],
            row["query_seen_new_acc"],
            row["query_min_old_class_acc"],
            row["query_old_acc"],
        ),
        reverse=True,
    )
    summary = {
        "diagnostic_scope": "SUPPORT_LOO_CLASS_BIAS_QKNN_NO_RAW_SUPPORT_IQ",
        "calibration_source": "support_leave_one_out_only",
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
        "topk",
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
        "support_loo_old_acc",
        "support_loo_min_old_class_acc",
        "support_loo_seen_new_acc",
        "support_loo_min_seen_new_class_acc",
        "bias",
        "stored_support_count",
        "stored_bias_scalars",
        "stored_quantized_support_code_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            for key in ("query_per_old_acc", "query_per_new_acc", "bias"):
                out[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"best": rows[:5], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
