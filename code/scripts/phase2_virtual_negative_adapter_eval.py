#!/usr/bin/env python
"""Evaluate a support-only virtual-negative boundary adapter for Stage2-C.

The frozen Phase1 features stay unchanged. Each target receiver fits a compact
multiclass ridge head and a binary known-vs-virtual-negative boundary from its
own target_old and target_new support rows only. Unknown rows are eval-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
from scripts.phase2_collaborative_open_set_qknn_eval import (
    KNOWN_ROLES,
    UNKNOWN_ROLE,
    _event_key,
    _scenario_of,
    _split_support_query_selected,
    _stable_score,
    load_feature_npz,
)
from scripts.phase2_support_ridge_adapter_eval import (
    _fit_ridge_head,
    _normalize_rows,
    _protocol_sets,
    _require_split,
    _score_ridge_head,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_binary_ridge(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    if positive_features.size == 0 or negative_features.size == 0:
        raise RuntimeError("virtual-negative adapter requires positive support and generated negatives")
    x = _normalize_rows(np.concatenate([positive_features, negative_features], axis=0))
    y = np.concatenate(
        [
            np.ones((positive_features.shape[0], 1), dtype=np.float64),
            np.zeros((negative_features.shape[0], 1), dtype=np.float64),
        ],
        axis=0,
    )
    x_aug = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = max(float(ridge_lambda), 0.0) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def _score_binary_ridge(features: np.ndarray, weights: np.ndarray, *, temperature: float) -> np.ndarray:
    x = _normalize_rows(features)
    x_aug = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    temp = max(float(temperature), 1e-6)
    return _sigmoid((x_aug @ weights).reshape(-1) / temp)


def _class_centroids(features: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    x = _normalize_rows(features)
    centroids: dict[str, np.ndarray] = {}
    for label in sorted({str(v) for v in labels}):
        idx = [i for i, value in enumerate(labels) if str(value) == label]
        centroids[label] = _normalize_rows(np.mean(x[np.asarray(idx, dtype=int)], axis=0, keepdims=True))[0]
    return centroids


def _generate_virtual_negatives(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    policy: str,
    shell_scale: float,
    max_mix_pairs_per_class: int,
) -> tuple[np.ndarray, dict[str, int]]:
    x = _normalize_rows(support_features)
    labels = [str(v) for v in support_labels]
    centroids = _class_centroids(x, labels)
    policy_key = str(policy or "shell_mix").strip().lower()
    if policy_key not in {"shell", "midpoint", "mix", "shell_mix"}:
        raise ValueError("virtual_negative_policy must be shell, midpoint, mix, or shell_mix")

    parts: list[np.ndarray] = []
    counts = {"shell": 0, "midpoint": 0, "mix": 0}

    if policy_key in {"shell", "shell_mix"}:
        shell_rows = []
        for i, label in enumerate(labels):
            centroid = centroids[label]
            delta = x[i] - centroid
            if float(np.linalg.norm(delta)) < 1e-8:
                delta = x[i]
            shell_rows.append(x[i] + float(shell_scale) * delta)
        if shell_rows:
            arr = _normalize_rows(np.stack(shell_rows, axis=0))
            parts.append(arr)
            counts["shell"] = int(arr.shape[0])

    if policy_key in {"midpoint", "shell_mix"}:
        midpoint_rows = []
        centroid_items = sorted(centroids.items())
        for left_i, (_, left) in enumerate(centroid_items):
            for _, right in centroid_items[left_i + 1 :]:
                midpoint_rows.append(0.5 * (left + right))
        if midpoint_rows:
            arr = _normalize_rows(np.stack(midpoint_rows, axis=0))
            parts.append(arr)
            counts["midpoint"] = int(arr.shape[0])

    if policy_key in {"mix", "shell_mix"}:
        mix_rows = []
        by_label: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        label_values = sorted(by_label)
        limit = max(1, int(max_mix_pairs_per_class))
        for left_i, left_label in enumerate(label_values):
            for right_label in label_values[left_i + 1 :]:
                for pair_i, left_idx in enumerate(by_label[left_label][:limit]):
                    right_idx = by_label[right_label][pair_i % len(by_label[right_label])]
                    mix_rows.append(0.5 * (x[left_idx] + x[right_idx]))
        if mix_rows:
            arr = _normalize_rows(np.stack(mix_rows, axis=0))
            parts.append(arr)
            counts["mix"] = int(arr.shape[0])

    if not parts:
        raise RuntimeError("virtual-negative generation produced no negatives; need at least two support classes")
    return np.concatenate(parts, axis=0).astype(np.float32), counts


def _support_known_threshold(
    support_features: np.ndarray,
    class_weights: np.ndarray,
    binary_weights: np.ndarray,
    *,
    class_temperature: float,
    boundary_temperature: float,
    quantile: float,
) -> float:
    class_probs = _score_ridge_head(support_features, class_weights, temperature=class_temperature)
    top = np.max(class_probs, axis=1)
    boundary = _score_binary_ridge(support_features, binary_weights, temperature=boundary_temperature)
    known_scores = np.sqrt(np.maximum(top, 0.0) * np.maximum(boundary, 0.0))
    if known_scores.size == 0:
        return 0.0
    return float(np.quantile(known_scores, max(0.0, min(1.0, float(quantile)))))


def build_virtual_negative_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int = 8,
    query_per_class: int = 20,
    seed: int = 4070303,
    ridge_lambda: float = 0.1,
    boundary_ridge_lambda: float = 0.1,
    class_temperature: float = 0.05,
    boundary_temperature: float = 0.25,
    support_threshold_quantile: float = 0.05,
    virtual_negative_policy: str = "shell_mix",
    virtual_negative_shell_scale: float = 1.5,
    virtual_negative_mix_pairs_per_class: int = 4,
    event_alignment_policy: str = "receiver_domain_ranked",
    support_selection_policy: str = "stable_first",
    evidence_packet_bytes: float = 112.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_labels, new_labels, unknown_labels, target_receivers, source_receivers = _protocol_sets(payload)
    alignment_policy = str(event_alignment_policy or "receiver_domain_ranked").strip().lower()
    if alignment_policy not in {"strict_event_key", "receiver_domain_ranked"}:
        raise ValueError("event_alignment_policy must be strict_event_key or receiver_domain_ranked")

    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
    receiver_query: dict[str, dict[str, list[int]]] = {}
    receiver_models: dict[str, dict[str, Any]] = {}
    receiver_support_counts: dict[str, int] = {}
    receiver_virtual_counts: dict[str, int] = {}
    receiver_thresholds: dict[str, float] = {}
    receiver_negative_breakdown: dict[str, dict[str, int]] = {}

    start = time.perf_counter()
    for rx in target_receivers:
        support_indices: list[int] = []
        support_labels: list[str] = []
        receiver_query[rx] = {"old": [], "seen_new": [], "unknown": []}
        for label in old_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_old",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            _require_split(rx, "target_old", label, support, query, k_shot=k_shot, query_per_class=query_per_class)
            support_indices.extend(support)
            support_labels.extend([label] * len(support))
            receiver_query[rx]["old"].extend(query)
        for label in new_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_new",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            _require_split(rx, "target_new", label, support, query, k_shot=k_shot, query_per_class=query_per_class)
            support_indices.extend(support)
            support_labels.extend([label] * len(support))
            receiver_query[rx]["seen_new"].extend(query)
        for label in unknown_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role=UNKNOWN_ROLE,
                tx_id=label,
                rx_id=rx,
                k_shot=0,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            if support:
                raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: unknown rows must not enter virtual-negative support")
            if len(query) < int(query_per_class):
                raise RuntimeError(
                    "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete unknown query coverage for "
                    f"receiver={rx}, tx_id={label}, query={len(query)}/{int(query_per_class)}"
                )
            receiver_query[rx]["unknown"].extend(query)

        support_arr = features[np.asarray(support_indices, dtype=int)]
        class_labels, class_weights = _fit_ridge_head(
            support_arr,
            support_labels,
            ridge_lambda=float(ridge_lambda),
        )
        virtual_negatives, breakdown = _generate_virtual_negatives(
            support_arr,
            support_labels,
            policy=virtual_negative_policy,
            shell_scale=float(virtual_negative_shell_scale),
            max_mix_pairs_per_class=int(virtual_negative_mix_pairs_per_class),
        )
        binary_weights = _fit_binary_ridge(
            support_arr,
            virtual_negatives,
            ridge_lambda=float(boundary_ridge_lambda),
        )
        threshold = _support_known_threshold(
            support_arr,
            class_weights,
            binary_weights,
            class_temperature=float(class_temperature),
            boundary_temperature=float(boundary_temperature),
            quantile=float(support_threshold_quantile),
        )
        receiver_models[rx] = {
            "class_labels": class_labels,
            "class_weights": class_weights,
            "binary_weights": binary_weights,
        }
        receiver_support_counts[rx] = len(support_indices)
        receiver_virtual_counts[rx] = int(virtual_negatives.shape[0])
        receiver_negative_breakdown[rx] = breakdown
        receiver_thresholds[rx] = threshold

    total_query_rows = sum(len(v[role]) for v in receiver_query.values() for role in ["old", "seen_new", "unknown"])
    per_row_ms = (time.perf_counter() - start) * 1000.0 / max(total_query_rows, 1)
    evidence: list[dict[str, Any]] = []
    row_alignment = ""

    for role_name, role_label_set, raw_role in [
        ("old", old_labels, "target_old"),
        ("seen_new", new_labels, "target_new"),
        ("unknown", unknown_labels, UNKNOWN_ROLE),
    ]:
        for label in role_label_set:
            by_rx_key: dict[str, dict[str, int]] = {}
            for rx in target_receivers:
                keyed: dict[str, int] = {}
                for idx in receiver_query[rx][role_name]:
                    if str(tx_ids[idx]) == str(label):
                        keyed[_event_key(payload, idx, role_name, label)] = int(idx)
                by_rx_key[rx] = keyed
            if alignment_policy == "strict_event_key":
                common_event_ids = sorted(set.intersection(*(set(by_rx_key[rx]) for rx in target_receivers)))
                event_groups = [(event_id, {rx: by_rx_key[rx][event_id] for rx in target_receivers}) for event_id in common_event_ids]
                row_alignment = "role_tx_day_sig_scenario"
            else:
                by_rx_scenario: dict[str, dict[str, list[int]]] = {}
                for rx in target_receivers:
                    by_rx_scenario[rx] = defaultdict(list)
                    for idx in receiver_query[rx][role_name]:
                        if str(tx_ids[idx]) == str(label):
                            by_rx_scenario[rx][_scenario_of(payload, idx)].append(int(idx))
                    for scenario in by_rx_scenario[rx]:
                        by_rx_scenario[rx][scenario] = sorted(
                            by_rx_scenario[rx][scenario],
                            key=lambda i: (
                                str(payload["day_ids"][i]),
                                str(payload["sig_ids"][i]),
                                _stable_score((rx, role_name, label, i), seed),
                            ),
                        )
                common_scenarios = sorted(set().union(*(set(by_rx_scenario[rx]) for rx in target_receivers)))
                event_groups = []
                for scenario in common_scenarios:
                    n = max(len(by_rx_scenario[rx].get(scenario, [])) for rx in target_receivers)
                    for event_i in range(n):
                        rx_to_idx = {
                            rx: by_rx_scenario[rx][scenario][event_i]
                            for rx in target_receivers
                            if event_i < len(by_rx_scenario[rx].get(scenario, []))
                        }
                        if rx_to_idx:
                            event_groups.append((f"{role_name}|{label}|{scenario}|rank{event_i:05d}", rx_to_idx))
                row_alignment = "receiver_domain_ranked_by_role_tx_scenario"
            for event_id, rx_to_idx in event_groups:
                for rx in sorted(rx_to_idx):
                    idx = rx_to_idx[rx]
                    model = receiver_models[rx]
                    class_probs = _score_ridge_head(
                        features[[idx]],
                        model["class_weights"],
                        temperature=float(class_temperature),
                    )[0]
                    boundary_known = float(
                        _score_binary_ridge(
                            features[[idx]],
                            model["binary_weights"],
                            temperature=float(boundary_temperature),
                        )[0]
                    )
                    order = np.argsort(-class_probs)
                    top_i = int(order[0])
                    second_i = int(order[1]) if len(order) > 1 else top_i
                    top_score = float(class_probs[top_i])
                    second_score = float(class_probs[second_i]) if second_i != top_i else 0.0
                    margin = float(top_score - second_score)
                    known_score = float(np.sqrt(max(top_score, 0.0) * max(boundary_known, 0.0)))
                    threshold = float(receiver_thresholds[rx])
                    threshold_risk = 0.0 if threshold <= 0.0 else max(0.0, min(1.0, (threshold - known_score) / threshold))
                    boundary_risk = max(0.0, min(1.0, 1.0 - boundary_known))
                    margin_risk = max(0.0, min(1.0, 1.0 - margin))
                    unknown_risk = float(max(threshold_risk, boundary_risk))
                    evidence.append(
                        {
                            "event_id": event_id,
                            "receiver_id": rx,
                            "role": role_name,
                            "true_label": "__unknown__" if role_name == "unknown" else str(tx_ids[idx]),
                            "predicted_label": str(model["class_labels"][top_i]),
                            "second_label": str(model["class_labels"][second_i]) if second_i != top_i else "",
                            "second_score": second_score,
                            "label_score_gap": margin,
                            "known_score": known_score,
                            "known_margin": margin,
                            "boundary_known_probability": boundary_known,
                            "effective_score_threshold": threshold,
                            "receiver_score_threshold": threshold,
                            "base_receiver_score_threshold": threshold,
                            "score_threshold_source": "support_virtual_negative_quantile",
                            "unknown_risk": unknown_risk,
                            "score_risk": threshold_risk,
                            "radius_risk": threshold_risk,
                            "margin_risk": margin_risk,
                            "mahalanobis_risk": 0.0,
                            "evt_risk": 0.0,
                            "oldness_risk": 0.0,
                            "virtual_unknown_risk": boundary_risk,
                            "class_negative_risk": boundary_risk,
                            "class_shell_risk": boundary_risk,
                            "class_radius": 0.0,
                            "class_radius_z": 0.0,
                            "reliability": float(max(0.0, min(1.0, 1.0 - boundary_risk))),
                            "reliability_source": "support_virtual_negative_boundary",
                            "receiver_deployment_prior": 1.0,
                            "receiver_deployment_prior_source": "support_virtual_negative_boundary",
                            "support_neighbor_count": int(receiver_support_counts[rx]),
                            "support_density": 1.0,
                            "candidate_class_count": len(model["class_labels"]),
                            "ridge_lambda": float(ridge_lambda),
                            "boundary_ridge_lambda": float(boundary_ridge_lambda),
                            "class_temperature": float(class_temperature),
                            "boundary_temperature": float(boundary_temperature),
                            "support_threshold_quantile": float(support_threshold_quantile),
                            "virtual_negative_count": int(receiver_virtual_counts[rx]),
                            "latency_ms": float(per_row_ms),
                            "bytes": float(evidence_packet_bytes),
                            "threshold_selection_label_scope": "support_virtual_unknown",
                            "calibration_role": "query",
                            "sat_scenario": _scenario_of(payload, idx),
                            "raw_role": raw_role,
                            "event_alignment": row_alignment,
                        }
                    )

    if not evidence:
        raise RuntimeError(
            "NO_ALIGNED_COLLABORATIVE_EVENTS: no shared evidence rows; use "
            "--event_alignment_policy receiver_domain_ranked for domain-ensemble diagnostics"
        )

    state_size_bytes = int(
        sum(
            model["class_weights"].nbytes
            + model["binary_weights"].nbytes
            + sum(len(label.encode("utf-8")) for label in model["class_labels"])
            for model in receiver_models.values()
        )
    )
    metadata = {
        "source_receiver_ids": source_receivers,
        "target_receiver_ids": target_receivers,
        "old_tx_ids": old_labels,
        "seen_new_tx_ids": new_labels,
        "unknown_tx_ids": unknown_labels,
        "target_channel_view": ",".join(sorted({row["sat_scenario"] for row in evidence if row["sat_scenario"]})),
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "adapter_type": "support_virtual_negative_boundary",
        "adapter_update_scope": "support_old_seen_new_only",
        "unknown_query_eval_only": True,
        "threshold_scope": "support_virtual_unknown",
        "threshold_scope_detail": "support_old_seen_new_virtual_negatives_only",
        "ridge_lambda": float(ridge_lambda),
        "boundary_ridge_lambda": float(boundary_ridge_lambda),
        "class_temperature": float(class_temperature),
        "boundary_temperature": float(boundary_temperature),
        "support_threshold_quantile": float(support_threshold_quantile),
        "virtual_negative_policy": str(virtual_negative_policy),
        "virtual_negative_shell_scale": float(virtual_negative_shell_scale),
        "virtual_negative_mix_pairs_per_class": int(virtual_negative_mix_pairs_per_class),
        "virtual_negative_count": receiver_virtual_counts,
        "virtual_negative_breakdown": receiver_negative_breakdown,
        "state_size_bytes": state_size_bytes,
        "evidence_bytes_per_receiver_event": float(evidence_packet_bytes),
        "event_alignment": row_alignment,
        "event_alignment_policy": alignment_policy,
        "strict_same_event_collaboration": alignment_policy == "strict_event_key",
        "support_selection_policy": str(support_selection_policy),
        "receiver_support_counts": receiver_support_counts,
        "receiver_score_thresholds": receiver_thresholds,
        "non_deployment_diagnostic": True,
    }
    return evidence, metadata


def _write_evidence_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_evidence_csv", type=Path, default=None)
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="available_up_to_k")
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--event_alignment_policy", choices=["strict_event_key", "receiver_domain_ranked"], default="receiver_domain_ranked")
    p.add_argument("--support_selection_policy", choices=["stable_first", "centroid", "scenario_diverse"], default="stable_first")
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--seed", type=int, default=4070303)
    p.add_argument("--ridge_lambda", type=float, default=0.1)
    p.add_argument("--boundary_ridge_lambda", type=float, default=0.1)
    p.add_argument("--class_temperature", type=float, default=0.05)
    p.add_argument("--boundary_temperature", type=float, default=0.25)
    p.add_argument("--support_threshold_quantile", type=float, default=0.05)
    p.add_argument("--virtual_negative_policy", choices=["shell", "midpoint", "mix", "shell_mix"], default="shell_mix")
    p.add_argument("--virtual_negative_shell_scale", type=float, default=1.5)
    p.add_argument("--virtual_negative_mix_pairs_per_class", type=int, default=4)
    p.add_argument("--evidence_packet_bytes", type=float, default=112.0)
    p.add_argument("--unknown_risk_threshold", type=float, default=0.65)
    p.add_argument("--accept_margin_threshold", type=float, default=0.02)
    p.add_argument("--fusion_policy", default="risk_margin")
    p.add_argument("--label_fusion_policy", default="weighted_vote_margin")
    p.add_argument("--latency_budget_ms", type=float, default=0.0)
    args = p.parse_args(argv)

    evidence, metadata = build_virtual_negative_evidence(
        load_feature_npz(args.feature_npz),
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        boundary_ridge_lambda=float(args.boundary_ridge_lambda),
        class_temperature=float(args.class_temperature),
        boundary_temperature=float(args.boundary_temperature),
        support_threshold_quantile=float(args.support_threshold_quantile),
        virtual_negative_policy=str(args.virtual_negative_policy),
        virtual_negative_shell_scale=float(args.virtual_negative_shell_scale),
        virtual_negative_mix_pairs_per_class=int(args.virtual_negative_mix_pairs_per_class),
        event_alignment_policy=str(args.event_alignment_policy),
        support_selection_policy=str(args.support_selection_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
    )
    result = evaluate_collaborative_open_set_evidence(
        evidence,
        collab_counts=args.collab_counts,
        threshold_selection_label_scope=metadata["threshold_scope"],
        unknown_query_eval_only=True,
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
        unknown_risk_threshold=float(args.unknown_risk_threshold),
        accept_margin_threshold=float(args.accept_margin_threshold),
        fusion_policy=str(args.fusion_policy),
        label_fusion_policy=str(args.label_fusion_policy),
        collab_group_policy=str(args.collab_group_policy),
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        latency_budget_ms=float(args.latency_budget_ms),
    )
    result["virtual_negative_adapter_metadata"] = metadata
    result["evidence_row_count"] = len(evidence)
    result["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    result["cwd"] = str(Path.cwd())
    result["python_executable"] = sys.executable
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_evidence_csv is not None:
        _write_evidence_csv(args.output_evidence_csv, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
