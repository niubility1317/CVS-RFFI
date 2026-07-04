#!/usr/bin/env python3
"""Probe graph-smoothed qKNN for Phase2-C many-new stability.

This variant keeps the deployable memory as quantized support codes plus class
prototype statistics. At inference time it builds a transient unlabeled query
graph, smooths qKNN class scores on that graph, then applies the same closed-set
balanced assignment used by the existing qKNN probe. It does not store raw
support samples.
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
import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def _smooth_scores(
    *,
    scores: np.ndarray,
    nbr: np.ndarray | None,
    weights: np.ndarray | None,
    alpha: float,
    steps: int,
) -> np.ndarray:
    if nbr is None or weights is None or alpha <= 0.0 or steps <= 0:
        return scores
    out = np.asarray(scores, dtype=np.float64).copy()
    unary = out.copy()
    for _ in range(int(steps)):
        out = (1.0 - float(alpha)) * unary + float(alpha) * np.sum(weights[:, :, None] * out[nbr], axis=1)
    return out


def _graph_weights(
    *,
    features: np.ndarray,
    query_indices: np.ndarray,
    graph_k: int,
    temperature: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if graph_k <= 0:
        return None, None
    query = qknn._normalize_rows(features[query_indices])
    sim = query @ query.T
    np.fill_diagonal(sim, -np.inf)
    k = min(int(graph_k), max(1, sim.shape[0] - 1))
    nbr = np.argpartition(sim, kth=sim.shape[1] - k, axis=1)[:, -k:]
    row = np.arange(sim.shape[0])[:, None]
    weights = np.exp(np.clip(sim[row, nbr], -1.0, 1.0) / max(float(temperature), 1e-6))
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    return nbr.astype(int), weights.astype(np.float64)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "support_selection_policy",
        "k_old",
        "k_new",
        "topm",
        "proto_mix",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "neg_margin",
        "graph_k",
        "graph_alpha",
        "graph_steps",
        "graph_temperature",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--seed_start", type=int, default=421000)
    parser.add_argument("--seed_count", type=int, default=120)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=70)
    parser.add_argument("--query_per_new", type=int, default=70)
    parser.add_argument("--pool_per_old", type=int, default=10)
    parser.add_argument("--pool_per_new", type=int, default=10)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--topm", type=int, default=4)
    parser.add_argument("--proto_mix", type=float, default=0.25)
    parser.add_argument("--radius_norm", type=float, default=0.0)
    parser.add_argument("--old_bias", type=float, default=0.001)
    parser.add_argument("--neg_lambda", type=float, default=0.4)
    parser.add_argument("--neg_threshold", type=float, default=0.7)
    parser.add_argument("--neg_margin", type=float, default=0.01)
    parser.add_argument("--mutual_only", action="store_true")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--balanced_assignment", action="store_true")
    parser.add_argument("--graph_k_grid", default="5,10,20")
    parser.add_argument("--graph_alpha_grid", default="0,0.1,0.2,0.35")
    parser.add_argument("--graph_steps_grid", default="1,2,4")
    parser.add_argument("--graph_temperature_grid", default="0.03,0.05,0.08")
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
    old_label_set = set(old_labels)
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
            support_indices, support_labels, old_query, new_query = base._collect(
                active._as_eval_splits(old_raw),
                active._as_eval_splits(new_raw),
                old_labels,
                new_labels,
            )
            query_indices = np.concatenate([old_query, new_query])
            truth = tx_ids[query_indices]
            query_scores, _, _ = base._class_scores(
                features=features,
                support_indices=support_indices,
                support_labels=support_labels,
                query_indices=query_indices,
                scenarios=scenarios,
                class_labels=class_labels,
                old_labels=old_label_set,
                topm=int(args.topm),
                proto_mix=float(args.proto_mix),
                radius_norm=float(args.radius_norm),
                old_bias=float(args.old_bias),
                neg_lambda=float(args.neg_lambda),
                neg_threshold=float(args.neg_threshold),
                neg_margin=float(args.neg_margin),
                mutual_only=bool(args.mutual_only),
                scenario_aware=bool(args.scenario_aware),
            )
            graph_cache: dict[tuple[int, float], tuple[np.ndarray | None, np.ndarray | None]] = {}
            for graph_k in _parse_int_csv(args.graph_k_grid):
                for temperature in _parse_float_csv(args.graph_temperature_grid):
                    graph_cache[(int(graph_k), float(temperature))] = _graph_weights(
                        features=features,
                        query_indices=query_indices,
                        graph_k=int(graph_k),
                        temperature=float(temperature),
                    )
            for graph_k in _parse_int_csv(args.graph_k_grid):
                for temperature in _parse_float_csv(args.graph_temperature_grid):
                    nbr, weights = graph_cache[(int(graph_k), float(temperature))]
                    for alpha in _parse_float_csv(args.graph_alpha_grid):
                        for steps in _parse_int_csv(args.graph_steps_grid):
                            smoothed = _smooth_scores(
                                scores=query_scores,
                                nbr=nbr,
                                weights=weights,
                                alpha=alpha,
                                steps=steps,
                            )
                            if bool(args.balanced_assignment):
                                pred = base._balanced_predict(
                                    smoothed,
                                    old_count=int(old_query.size),
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                )
                            else:
                                pred = base._predict(smoothed, class_labels)
                            metrics = base._metrics(
                                pred,
                                truth,
                                old_count=int(old_query.size),
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
                                "k_old": int(args.k_old),
                                "k_new": int(args.k_new),
                                "topm": int(args.topm),
                                "proto_mix": float(args.proto_mix),
                                "radius_norm": float(args.radius_norm),
                                "old_bias": float(args.old_bias),
                                "neg_lambda": float(args.neg_lambda),
                                "neg_threshold": float(args.neg_threshold),
                                "neg_margin": float(args.neg_margin),
                                "mutual_only": bool(args.mutual_only),
                                "scenario_aware": bool(args.scenario_aware),
                                "balanced_assignment": bool(args.balanced_assignment),
                                "graph_k": int(graph_k),
                                "graph_alpha": float(alpha),
                                "graph_steps": int(steps),
                                "graph_temperature": float(temperature),
                                "stored_quantized_support_code_count": int(support_indices.size),
                                "stored_raw_support_count": 0,
                                "stored_class_prototype_count": int(len(class_labels)),
                            }
                            row.update({f"query_{key}": value for key, value in metrics.items()})
                            row["query_rank_score"] = base._rank(metrics)
                            rows.append(row)

    rows.sort(key=lambda row: tuple(row["query_rank_score"]), reverse=True)
    summary = {
        "diagnostic_scope": "GRAPH_SMOOTHED_QKNN_NO_QUERY_LABEL_FIT",
        "selection_note": "Query graph uses unlabeled query features only; best_by_query is still audit-ranked.",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows_count": int(len(rows)),
        "best_by_query": rows[:20],
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(Path(args.output_csv), rows)
    print(json.dumps({"rows_count": len(rows), "best_by_query": rows[:3], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
