#!/usr/bin/env python
"""Evaluate a support-only ridge adapter head for Stage2-C collaborative inference.

The Phase1 backbone stays frozen. Each target receiver fits a closed-form
linear ridge head from its own target_old and target_new support rows only.
Unknown rows are evaluation-only and are never used for threshold selection.
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
    _parse_csv,
    _scenario_of,
    _split_support_query_selected,
    _stable_score,
    canonical_tx_id,
    load_feature_npz,
    validate_required_roles,
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-12)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _fit_ridge_head(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    ridge_lambda: float,
) -> tuple[list[str], np.ndarray]:
    label_values = sorted({str(label) for label in labels})
    if not label_values:
        raise RuntimeError("support ridge head requires at least one support label")
    label_to_col = {label: i for i, label in enumerate(label_values)}
    x = _normalize_rows(features)
    y = np.zeros((x.shape[0], len(label_values)), dtype=np.float64)
    for row_i, label in enumerate(labels):
        y[row_i, label_to_col[str(label)]] = 1.0
    x_aug = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = max(float(ridge_lambda), 0.0) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return label_values, weights


def _score_ridge_head(features: np.ndarray, weights: np.ndarray, *, temperature: float) -> np.ndarray:
    x = _normalize_rows(features)
    x_aug = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return _softmax(x_aug @ weights, temperature)


def _require_split(
    receiver_id: str,
    role: str,
    tx_id: str,
    support: Sequence[int],
    query: Sequence[int],
    *,
    k_shot: int,
    query_per_class: int,
) -> None:
    if len(support) < int(k_shot) or len(query) < int(query_per_class):
        raise RuntimeError(
            "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete Stage2-C coverage for "
            f"receiver={receiver_id}, role={role}, tx_id={tx_id}, "
            f"support={len(support)}/{int(k_shot)}, query={len(query)}/{int(query_per_class)}"
        )


def _protocol_sets(payload: Mapping[str, Any]) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    old_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_old")[0].tolist()})
    new_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_new")[0].tolist()})
    unknown_labels = sorted({str(tx_ids[i]) for i in np.where(roles == UNKNOWN_ROLE)[0].tolist()})
    source_labels = {str(tx_ids[i]) for i in np.where(roles == "source")[0].tolist()}
    manifest = payload.get("manifest", {})
    if isinstance(manifest, Mapping):
        manifest_source_tx = manifest.get("source_tx_ids", [])
        if isinstance(manifest_source_tx, str):
            source_labels.update(_parse_csv(manifest_source_tx))
        else:
            source_labels.update(_parse_csv(",".join(str(v) for v in manifest_source_tx)))
    label_overlaps = {
        "old_new": sorted(set(old_labels) & set(new_labels)),
        "old_unknown": sorted(set(old_labels) & set(unknown_labels)),
        "new_unknown": sorted(set(new_labels) & set(unknown_labels)),
    }
    if any(label_overlaps.values()):
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: target TX sets must be mutually disjoint: {label_overlaps}")
    old_not_source = sorted(set(old_labels) - source_labels)
    non_old_in_source = sorted((set(new_labels) | set(unknown_labels)) & source_labels)
    if old_not_source or non_old_in_source:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: Stage2-C TX split violates source/target semantics; "
            f"old_not_in_source={old_not_source}, non_old_in_source={non_old_in_source}"
        )
    target_receivers = sorted({str(rx_ids[i]) for i in np.where(np.isin(roles, [*KNOWN_ROLES, UNKNOWN_ROLE]))[0].tolist()})
    source_receivers = sorted({str(rx_ids[i]) for i in np.where(roles == "source")[0].tolist()})
    if not source_receivers:
        raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: source receiver metadata is required to verify R_s/R_t disjointness")
    overlap_receivers = sorted(set(source_receivers) & set(target_receivers))
    if overlap_receivers:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: R_s and R_t overlap: {overlap_receivers}")
    return old_labels, new_labels, unknown_labels, target_receivers, source_receivers


def build_support_ridge_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int = 8,
    query_per_class: int = 20,
    seed: int = 4070303,
    ridge_lambda: float = 0.1,
    ridge_score_threshold: float = 0.2,
    ridge_temperature: float = 0.05,
    event_alignment_policy: str = "receiver_domain_ranked",
    support_selection_policy: str = "stable_first",
    evidence_packet_bytes: float = 96.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_labels, new_labels, unknown_labels, target_receivers, source_receivers = _protocol_sets(payload)
    alignment_policy = str(event_alignment_policy or "receiver_domain_ranked").strip().lower()
    if alignment_policy not in {"strict_event_key", "receiver_domain_ranked"}:
        raise ValueError("event_alignment_policy must be strict_event_key or receiver_domain_ranked")
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
    receiver_query: dict[str, dict[str, list[int]]] = {}
    receiver_heads: dict[str, tuple[list[str], np.ndarray]] = {}
    receiver_support_counts: dict[str, int] = {}
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
                raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: unknown rows must not enter ridge support")
            if len(query) < int(query_per_class):
                raise RuntimeError(
                    "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete unknown query coverage for "
                    f"receiver={rx}, tx_id={label}, query={len(query)}/{int(query_per_class)}"
                )
            receiver_query[rx]["unknown"].extend(query)
        receiver_heads[rx] = _fit_ridge_head(features[np.asarray(support_indices, dtype=int)], support_labels, ridge_lambda=ridge_lambda)
        receiver_support_counts[rx] = len(support_indices)
    total_query_rows = sum(len(v[role]) for v in receiver_query.values() for role in ["old", "seen_new", "unknown"])
    per_row_ms = (time.perf_counter() - start) * 1000.0 / max(total_query_rows, 1)
    evidence: list[dict[str, Any]] = []
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
                    class_labels, weights = receiver_heads[rx]
                    probs = _score_ridge_head(features[[idx]], weights, temperature=ridge_temperature)[0]
                    order = np.argsort(-probs)
                    top_i = int(order[0])
                    second_i = int(order[1]) if len(order) > 1 else top_i
                    top_score = float(probs[top_i])
                    second_score = float(probs[second_i]) if second_i != top_i else 0.0
                    margin = float(top_score - second_score)
                    unknown_risk = float(max(0.0, min(1.0, (float(ridge_score_threshold) - top_score) / max(float(ridge_score_threshold), 1e-6))))
                    score_risk = unknown_risk
                    evidence.append(
                        {
                            "event_id": event_id,
                            "receiver_id": rx,
                            "role": role_name,
                            "true_label": "__unknown__" if role_name == "unknown" else str(tx_ids[idx]),
                            "predicted_label": str(class_labels[top_i]),
                            "second_label": str(class_labels[second_i]) if second_i != top_i else "",
                            "second_score": second_score,
                            "label_score_gap": margin,
                            "known_score": top_score,
                            "known_margin": margin,
                            "effective_score_threshold": float(ridge_score_threshold),
                            "receiver_score_threshold": float(ridge_score_threshold),
                            "base_receiver_score_threshold": float(ridge_score_threshold),
                            "score_threshold_source": "support_ridge_fixed_arg",
                            "unknown_risk": unknown_risk,
                            "score_risk": score_risk,
                            "radius_risk": score_risk,
                            "margin_risk": 0.0 if margin > 0.0 else 1.0,
                            "mahalanobis_risk": 0.0,
                            "evt_risk": 0.0,
                            "oldness_risk": 0.0,
                            "virtual_unknown_risk": 0.0,
                            "class_negative_risk": 0.0,
                            "class_shell_risk": 0.0,
                            "class_radius": 0.0,
                            "class_radius_z": 0.0,
                            "reliability": 1.0,
                            "reliability_source": "support_ridge_receiver_head",
                            "receiver_deployment_prior": 1.0,
                            "receiver_deployment_prior_source": "support_ridge_receiver_head",
                            "support_neighbor_count": int(receiver_support_counts[rx]),
                            "support_density": 1.0,
                            "candidate_class_count": len(class_labels),
                            "ridge_lambda": float(ridge_lambda),
                            "ridge_temperature": float(ridge_temperature),
                            "ridge_score_threshold": float(ridge_score_threshold),
                            "latency_ms": float(per_row_ms),
                            "bytes": float(evidence_packet_bytes),
                            "threshold_selection_label_scope": "support_known_only",
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
        sum(weights.nbytes + sum(len(label.encode("utf-8")) for label in labels) for labels, weights in receiver_heads.values())
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
        "adapter_type": "support_ridge_linear_head",
        "adapter_update_scope": "support_old_seen_new_only",
        "unknown_query_eval_only": True,
        "threshold_scope": "support_known_only",
        "ridge_threshold_scope_detail": "support_known_ridge_only",
        "ridge_lambda": float(ridge_lambda),
        "ridge_score_threshold": float(ridge_score_threshold),
        "ridge_temperature": float(ridge_temperature),
        "state_size_bytes": state_size_bytes,
        "evidence_bytes_per_receiver_event": float(evidence_packet_bytes),
        "event_alignment": row_alignment,
        "event_alignment_policy": alignment_policy,
        "strict_same_event_collaboration": alignment_policy == "strict_event_key",
        "support_selection_policy": str(support_selection_policy),
        "receiver_support_counts": receiver_support_counts,
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
    p.add_argument("--ridge_score_threshold", type=float, default=0.2)
    p.add_argument("--ridge_temperature", type=float, default=0.05)
    p.add_argument("--evidence_packet_bytes", type=float, default=96.0)
    p.add_argument("--unknown_risk_threshold", type=float, default=0.65)
    p.add_argument("--accept_margin_threshold", type=float, default=0.02)
    p.add_argument("--fusion_policy", default="risk_margin")
    p.add_argument("--label_fusion_policy", default="weighted_vote_margin")
    p.add_argument("--latency_budget_ms", type=float, default=0.0)
    args = p.parse_args(argv)

    evidence, metadata = build_support_ridge_evidence(
        load_feature_npz(args.feature_npz),
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        ridge_score_threshold=float(args.ridge_score_threshold),
        ridge_temperature=float(args.ridge_temperature),
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
    result["support_ridge_metadata"] = metadata
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
