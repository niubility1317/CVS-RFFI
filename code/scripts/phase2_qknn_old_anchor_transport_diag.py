#!/usr/bin/env python3
"""Diagnose old-anchor feature transport for Phase2 qKNN.

This diagnostic fits a compressed feature-space transport from target-old
support prototypes to source-old prototypes, then runs the existing qKNN head.
It stores no raw support samples; persistent transport state is limited to a
shift vector, diagonal scale, or low-rank orthogonal basis summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn
import phase2_support_metric_qknn_probe as probe


OLD_LABELS = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
NEW10 = ["10-10", "11-10", "18-5", "19-3", "2-13", "2-5", "3-8", "4-10", "8-18", "8-3"]
NEW20 = ["1-1", "1-10", "1-11", "1-12", "1-14", "1-15", "1-16", "1-18", "1-19", "1-2"] + NEW10


BASE_DEFAULTS: dict[str, Any] = {
    "transform_mode": "identity",
    "transform_strength": 0.0,
    "topm": 4,
    "proto_mix": 0.25,
    "aux_score_weight": 0.0,
    "radius_norm": 0.0,
    "old_bias": 0.001,
    "neg_lambda": 0.7,
    "neg_threshold": 0.75,
    "neg_margin": 0.01,
    "mutual_only": True,
    "scenario_aware": False,
    "balanced_assignment": False,
    "role_balanced_assignment": False,
    "fast_role_balanced_assignment": False,
    "scenario_balanced_assignment": False,
    "proto_repel_lambda": 0.0,
    "proto_repel_margin": 0.85,
    "proto_repel_steps": 0,
    "proto_repel_anchor": 0.5,
    "pair_refine_similarity": 1.1,
    "pair_axis_similarity": 1.1,
    "pair_axis_weight": 0.0,
    "pair_axis_clip": 1.0,
    "pair_gaussian_similarity": 1.1,
    "pair_gaussian_weight": 0.0,
    "pair_gaussian_clip": 5.0,
    "pair_fisher_similarity": 1.1,
    "pair_fisher_weight": 0.0,
    "pair_fisher_alpha": 1.0,
    "pair_fisher_clip": 5.0,
    "support_guided_proxy_rows": [],
    "support_guided_proxy_weight": 0.0,
    "support_guided_proxy_top_pairs": 0,
    "support_guided_proxy_clip": 2.0,
    "pair_logreg_similarity": 1.1,
    "pair_logreg_weight": 0.0,
    "pair_logreg_alpha": 1.0,
    "pair_logreg_clip": 2.0,
    "pair_logreg_scope": "all",
    "new_old_conflict_bias_threshold": 1.1,
    "new_old_conflict_bias_weight": 0.0,
    "bootstrap_proto_mix": 0.0,
    "bootstrap_proto_drop": 1,
    "bootstrap_proto_topm": 1,
    "core_proto_weight": 0.0,
    "core_proto_count": 3,
    "core_proto_topm": 1,
    "core_proto_mode": "centroid",
    "ridge_head_weight": 0.0,
    "ridge_head_alpha": 1.0,
    "ridge_head_clip": 2.0,
    "subspace_proto_weight": 0.0,
    "subspace_proto_rank": 8,
    "subspace_proto_power": 0.0,
    "subspace_proto_clip": 3.0,
    "old_residual_new_weight": 0.0,
    "old_residual_new_rank": 2,
    "old_residual_new_proto_mix": 0.4,
    "old_residual_new_clip": 2.0,
    "domain_refine_key": "none",
    "domain_refine_weight": 0.0,
    "domain_refine_scope": "all",
    "class_diag_metric_weight": 0.0,
    "class_diag_metric_similarity": 0.9,
    "class_diag_metric_alpha": 0.01,
    "class_diag_metric_power": 0.5,
    "class_diag_metric_clip": 2.0,
    "support_bias_weight": 0.0,
    "support_bias_step": 0.01,
    "support_bias_rounds": 4,
    "support_loo_pair_rescue_weight": 0.0,
    "support_loo_pair_rescue_top_pairs": 0,
    "support_loo_pair_rescue_min_errors": 1,
    "support_loo_pair_rescue_alpha": 0.1,
    "support_loo_pair_rescue_clip": 2.0,
    "support_loo_pair_rescue_scope": "new",
    "mahal_proto_weight": 0.0,
    "mahal_proto_alpha": 1.0,
    "mahal_proto_diag_mix": 0.5,
    "mahal_proto_clip": 3.0,
    "score_calibration": "none",
    "assignment_margin_weight": 0.0,
    "assignment_margin_clip": 1.0,
    "labelprop_weight": 0.0,
    "labelprop_k": 8,
    "labelprop_alpha": 0.8,
    "labelprop_temperature": 0.05,
    "labelprop_rounds": 10,
    "labelprop_clip": 2.0,
    "labelprop_scope": "all",
    "query_graph_weight": 0.0,
    "query_graph_k": 8,
    "query_graph_temperature": 0.05,
    "query_graph_rounds": 1,
    "query_graph_scope": "all",
    "local_competition_weight": 0.0,
    "local_competition_k": 4,
    "local_competition_clip": 1.0,
    "local_competition_scope": "role",
    "query_proto_refine_weight": 0.0,
    "query_proto_refine_topm": 0,
    "query_proto_refine_clip": 2.0,
    "transductive_proto_weight": 0.0,
    "transductive_proto_rounds": 0,
    "transductive_proto_query_topm": 50,
    "transductive_proto_support_weight": 1.0,
    "transductive_proto_query_weight": 1.0,
    "transductive_proto_clip": 1.5,
    "dense_cluster_weight": 0.0,
    "dense_cluster_similarity": 0.9,
    "dense_cluster_neighbor_k": 0,
    "dense_cluster_rounds": 1,
    "dense_cluster_candidate_topn": 3,
    "dense_cluster_query_topm": 50,
    "dense_cluster_clip": 1.5,
    "dense_cluster_scope": "new",
    "source_guard_mode": "none",
    "source_guard_weight": 0.0,
    "source_guard_conf_min": 0.0,
    "source_guard_margin_min": 0.0,
    "source_proto_anchor_mode": "none",
    "source_proto_anchor_weight": 0.0,
    "source_proto_anchor_center": 0.0,
    "adaptive_qknn_policy": "none",
    "support_guided_proxy_min_errors": 1,
    "support_guided_proxy_scope": "new",
    "support_guided_proxy_balance": False,
    "support_guided_proxy_bundle_rows": 1,
    "support_guided_proxy_analogy": False,
    "support_guided_proxy_gate": False,
    "support_guided_proxy_gate_floor_tol": 0.0,
    "support_guided_proxy_gate_mean_tol": 0.0,
    "support_quality_weight": 0.0,
    "support_quality_floor": 0.25,
    "support_quality_margin_scale": 0.1,
    "support_loo_pair_rescue_proto_neighbors": 0,
    "support_loo_pair_rescue_proto_min_sim": 1.1,
    "support_loo_pair_linear_weight": 0.0,
    "support_loo_pair_linear_top_pairs": 0,
    "support_loo_pair_linear_min_errors": 1,
    "support_loo_pair_linear_alpha": 0.1,
    "support_loo_pair_linear_clip": 1.5,
    "support_loo_pair_linear_scope": "new",
    "query_cluster_weight": 0.0,
    "query_cluster_rounds": 3,
    "query_cluster_support_weight": 0.6,
    "query_cluster_temperature": 0.08,
    "query_cluster_clip": 2.0,
    "query_cluster_scope": "new",
    "query_cluster_agreement_min": 0.70,
    "query_cluster_margin_min": -1.0,
    "old_new_runnerup_rescue_similarity": 1.1,
    "old_new_runnerup_rescue_margin": 0.0,
    "old_new_runnerup_rescue_weight": 0.0,
    "old_target": 0.8,
    "old_floor": 0.75,
    "new_target": 0.65,
    "new_floor": 0.75,
    "collect_predictions": False,
}


def _old_anchor_transport(
    features: np.ndarray,
    *,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    old_splits: dict[str, probe.Split],
    mode: str,
    weight: float,
    ridge: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    x = qknn._normalize_rows(features)
    if float(weight) <= 0.0:
        return x, 0
    source_proto: list[np.ndarray] = []
    target_proto: list[np.ndarray] = []
    for label in OLD_LABELS:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        target_idx = old_splits[label][0]
        if source_idx.size and target_idx.size:
            source_proto.append(qknn._normalize_rows(x[source_idx].mean(axis=0, keepdims=True))[0])
            target_proto.append(qknn._normalize_rows(x[target_idx].mean(axis=0, keepdims=True))[0])
    if len(source_proto) < 2:
        return x, 0
    source = np.asarray(source_proto, dtype=np.float64)
    target = np.asarray(target_proto, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    if mode == "shift":
        out = x + float(weight) * (source_center - target_center)
        stored_scalars = int(x.shape[1])
    elif mode == "diag":
        target_c = target - target_center
        source_c = source - source_center
        scale = (target_c * source_c).sum(axis=0) / (np.square(target_c).sum(axis=0) + float(ridge))
        scale = np.clip(scale, -float(clip), float(clip))
        out = (x - target_center) * ((1.0 - float(weight)) + float(weight) * scale)
        out = out + target_center + float(weight) * (source_center - target_center)
        stored_scalars = int(2 * x.shape[1])
    elif mode == "procrustes":
        target_c = target - target_center
        source_c = source - source_center
        u, _singular, vt = np.linalg.svd(target_c.T @ source_c, full_matrices=False)
        rotation = u @ vt
        rotated = (x - target_center) @ rotation + source_center
        out = (1.0 - float(weight)) * x + float(weight) * rotated
        stored_scalars = int(x.shape[1] * min(len(source_proto) - 1, x.shape[1]) + x.shape[1])
    else:
        raise ValueError(f"unsupported old_anchor mode: {mode}")
    return qknn._normalize_rows(out), stored_scalars


def _query_indices(old_splits: dict[str, probe.Split], new_splits: dict[str, probe.Split], new_labels: list[str]) -> np.ndarray:
    query: list[int] = []
    for label in OLD_LABELS:
        query.extend(old_splits[label][1].tolist())
    for label in new_labels:
        query.extend(new_splits[label][1].tolist())
    return np.asarray(query, dtype=int)


def _unlabeled_batch_transport(
    features: np.ndarray,
    *,
    fit_indices: np.ndarray,
    mode: str,
    weight: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    x = qknn._normalize_rows(features)
    if str(mode) == "none" or float(weight) <= 0.0:
        return x, 0
    batch = x[np.asarray(fit_indices, dtype=int)]
    center = batch.mean(axis=0)
    if mode == "center":
        transported = qknn._normalize_rows(x - center)
        stored_scalars = int(x.shape[1])
    elif mode == "diag_whiten":
        scale = 1.0 / np.sqrt(np.maximum(batch.var(axis=0), 1e-5))
        scale = scale / max(float(np.median(scale)), 1e-6)
        scale = np.clip(scale, 1.0 / float(clip), float(clip))
        transported = qknn._normalize_rows((x - center) * scale)
        stored_scalars = int(2 * x.shape[1])
    else:
        raise ValueError(f"unsupported batch mode: {mode}")
    return qknn._normalize_rows((1.0 - float(weight)) * x + float(weight) * transported), stored_scalars


def _build_splits(
    *,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    features: np.ndarray,
    scenarios: np.ndarray,
    logits: np.ndarray,
    new_labels: list[str],
    k: int,
    query_per_class: int,
    seed: int,
) -> tuple[dict[str, probe.Split], dict[str, probe.Split]]:
    source_probs = active._softmax(logits)
    source_prototypes: dict[str, np.ndarray] = {}
    for label in OLD_LABELS:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]
    common = {
        "tx_ids": tx_ids,
        "roles": roles,
        "features": features,
        "scenarios": scenarios,
        "source_probs": source_probs,
        "source_label_to_idx": {label: index for index, label in enumerate(OLD_LABELS)},
        "source_prototypes": source_prototypes,
        "policy": "stable_first",
        "seed": int(seed),
        "exclude_pool_from_query": False,
    }
    old_raw = active._build_active_splits(
        labels=OLD_LABELS,
        role="target_old",
        k=int(k),
        query_per_class=int(query_per_class),
        pool_per_class=int(k),
        **common,
    )
    new_raw = active._build_active_splits(
        labels=new_labels,
        role="target_unknown",
        k=int(k),
        query_per_class=int(query_per_class),
        pool_per_class=int(k),
        **common,
    )
    return active._as_eval_splits(old_raw), active._as_eval_splits(new_raw)


def _evaluate_case(
    *,
    features: np.ndarray,
    aux_features: np.ndarray | None,
    logits: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    scenarios: np.ndarray,
    rx_ids: np.ndarray,
    new_labels: list[str],
    scope: str,
    k: int,
    query_per_class: int,
    seed: int,
    mode: str,
    weight: float,
    batch_mode: str,
    batch_weight: float,
    ridge: float,
    clip: float,
) -> dict[str, Any]:
    old_splits, new_splits = _build_splits(
        tx_ids=tx_ids,
        roles=roles,
        features=features,
        scenarios=scenarios,
        logits=logits,
        new_labels=new_labels,
        k=k,
        query_per_class=query_per_class,
        seed=seed,
    )
    adapted, stored_scalars = _old_anchor_transport(
        features,
        tx_ids=tx_ids,
        roles=roles,
        old_splits=old_splits,
        mode=mode,
        weight=weight,
        ridge=ridge,
        clip=clip,
    )
    support_indices, support_labels = probe._collect_support(old_splits, new_splits, OLD_LABELS, new_labels)
    query_indices = _query_indices(old_splits, new_splits, new_labels)
    fit_indices = np.concatenate([support_indices, query_indices]).astype(int)
    adapted, stored_batch_scalars = _unlabeled_batch_transport(
        adapted,
        fit_indices=fit_indices,
        mode=batch_mode,
        weight=batch_weight,
        clip=clip,
    )
    aux_adapted = None
    if aux_features is not None:
        aux_adapted, _aux_scalars = _old_anchor_transport(
            aux_features,
            tx_ids=tx_ids,
            roles=roles,
            old_splits=old_splits,
            mode=mode,
            weight=weight,
            ridge=ridge,
            clip=clip,
        )
        aux_adapted, _aux_batch_scalars = _unlabeled_batch_transport(
            aux_adapted,
            fit_indices=fit_indices,
            mode=batch_mode,
            weight=batch_weight,
            clip=clip,
        )
    geometry = probe._support_geometry_summary(
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        old_labels=OLD_LABELS,
        new_labels=new_labels,
        k_old=k,
        k_new=k,
    )
    overrides = probe._adaptive_qknn_overrides(
        policy="dualview_support_v8",
        geometry=geometry,
        aux_available=aux_adapted is not None,
    )
    params = BASE_DEFAULTS.copy()
    params.update({key: value for key, value in overrides.items() if key in params})
    row = probe._evaluate_metric_qknn(
        features=adapted,
        aux_features=aux_adapted,
        logits=logits,
        tx_ids=tx_ids,
        roles=roles,
        scenarios=scenarios,
        domain_values=rx_ids,
        old_splits=old_splits,
        new_splits=new_splits,
        old_labels=OLD_LABELS,
        new_labels=new_labels,
        **params,
    )
    row.update(
        {
            "scope": scope,
            "K": int(k),
            "query_per_class": int(query_per_class),
            "old_anchor_mode": mode,
            "old_anchor_weight": float(weight),
            "old_anchor_ridge": float(ridge),
            "old_anchor_clip": float(clip),
            "batch_transport_mode": str(batch_mode),
            "batch_transport_weight": float(batch_weight),
            "stored_old_anchor_scalars": int(stored_scalars),
            "stored_batch_transport_scalars": int(stored_batch_scalars),
            "support_index_sha16": probe._split_fingerprint(old_splits, new_splits, OLD_LABELS, new_labels)[
                "support_index_sha16"
            ],
            "new_query_index_sha16": probe._split_fingerprint(old_splits, new_splits, OLD_LABELS, new_labels)[
                "new_query_index_sha16"
            ],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--aux_feature_npz", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, default=421023)
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--modes", default="shift,diag,procrustes")
    parser.add_argument("--batch_modes", default="none")
    parser.add_argument("--batch_weights", default="0")
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=3.0)
    args = parser.parse_args()

    data = np.load(args.feature_npz, allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    rx_ids = np.asarray(data["rx_ids"], dtype=object).astype(str)
    aux_features = None
    if str(args.aux_feature_npz).strip():
        aux = np.load(args.aux_feature_npz, allow_pickle=True)
        aux_features = qknn._normalize_rows(aux["features"])

    rows: list[dict[str, Any]] = []
    mode_values = qknn._parse_csv(args.modes)
    weights = qknn._parse_float_csv(args.weights)
    batch_modes = qknn._parse_csv(args.batch_modes)
    batch_weights = qknn._parse_float_csv(args.batch_weights)
    for scope, new_labels in (("N10", NEW10), ("N20", NEW20)):
        for k, query_per_class in ((5, 75), (10, 70)):
            rows.append(
                _evaluate_case(
                    features=features,
                    aux_features=aux_features,
                    logits=logits,
                    tx_ids=tx_ids,
                    roles=roles,
                    scenarios=scenarios,
                    rx_ids=rx_ids,
                    new_labels=new_labels,
                    scope=scope,
                    k=k,
                    query_per_class=query_per_class,
                    seed=int(args.seed),
                    mode="shift",
                    weight=0.0,
                    batch_mode="none",
                    batch_weight=0.0,
                    ridge=float(args.ridge),
                    clip=float(args.clip),
                )
            )
            for mode in mode_values:
                for weight in weights:
                    if float(weight) <= 0.0:
                        continue
                    for batch_mode in batch_modes:
                        for batch_weight in batch_weights:
                            rows.append(
                                _evaluate_case(
                                    features=features,
                                    aux_features=aux_features,
                                    logits=logits,
                                    tx_ids=tx_ids,
                                    roles=roles,
                                    scenarios=scenarios,
                                    rx_ids=rx_ids,
                                    new_labels=new_labels,
                                    scope=scope,
                                    k=k,
                                    query_per_class=query_per_class,
                                    seed=int(args.seed),
                                    mode=str(mode),
                                    weight=float(weight),
                                    batch_mode=str(batch_mode),
                                    batch_weight=float(batch_weight),
                                    ridge=float(args.ridge),
                                    clip=float(args.clip),
                                )
                            )
            for batch_mode in batch_modes:
                for batch_weight in batch_weights:
                    if str(batch_mode) == "none" or float(batch_weight) <= 0.0:
                        continue
                    rows.append(
                        _evaluate_case(
                            features=features,
                            aux_features=aux_features,
                            logits=logits,
                            tx_ids=tx_ids,
                            roles=roles,
                            scenarios=scenarios,
                            rx_ids=rx_ids,
                            new_labels=new_labels,
                            scope=scope,
                            k=k,
                            query_per_class=query_per_class,
                            seed=int(args.seed),
                            mode="shift",
                            weight=0.0,
                            batch_mode=str(batch_mode),
                            batch_weight=float(batch_weight),
                            ridge=float(args.ridge),
                            clip=float(args.clip),
                        )
                    )

    output = {"rows": rows}
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for scope in ("N10", "N20"):
        for k in (5, 10):
            subset = [row for row in rows if row["scope"] == scope and int(row["K"]) == k]
            best = sorted(
                subset,
                key=lambda row: (row["query_min_seen_new_class_acc"], row["query_seen_new_acc"]),
                reverse=True,
            )[0]
            print(
                f"{scope} K={k} best mode={best['old_anchor_mode']} weight={best['old_anchor_weight']} "
                f"old={best['query_old_acc']:.4f} min_old={best['query_min_old_class_acc']:.4f} "
                f"new={best['query_seen_new_acc']:.4f} min_new={best['query_min_seen_new_class_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
