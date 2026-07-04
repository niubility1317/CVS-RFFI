#!/usr/bin/env python
"""Evaluate support-only metric/energy calibration for satellite CI.

SMEC-CI keeps the frozen qKNN8 route as the only label authority. It builds
receiver-local calibration from target-old and seen-new support only, then may
raise unknown/defer risk for weak query rows that are far from the support
metric shell. Unknown query rows are never used for threshold fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from scripts.phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    KNOWN_ROLES,
    UNKNOWN_ROLE,
    _normalize_rows,
    _require_split,
    _scenario_of,
    _split_support_query,
    _split_support_query_selected,
    _stable_score,
    load_feature_npz,
    validate_required_roles,
)
from scripts.phase2_old_protected_unknown_confirm_ci_eval import (  # noqa: E402
    POLICIES,
    OpuPolicy,
    _evaluate_policy,
    _summary_row,
)
from scripts.phase2_socapr_qknn8_pareto_eval import _run_route  # noqa: E402


@dataclass(frozen=True)
class SmecConfig:
    proto_weight: float = 0.70
    knn_weight: float = 0.50
    energy_weight: float = 0.08
    proto_temperature: float = 0.03
    knn_temperature: float = 0.03
    energy_temperature: float = 0.20
    proto_quantile: float = 0.90
    knn_quantile: float = 0.90
    energy_quantile: float = 0.90
    proto_slack: float = 0.01
    knn_slack: float = 0.01
    energy_slack: float = 0.0
    strong_score: float = 0.60
    strong_margin: float = 0.08
    strong_support_density: float = 0.35
    strong_reliability: float = 0.60
    weak_score_anchor: float = 0.70
    weak_margin_anchor: float = 0.20
    weak_support_anchor: float = 0.45
    weak_reliability_anchor: float = 0.70
    strong_aux_cap: float = 0.15
    aux_bytes_per_receiver: float = 24.0
    aux_latency_ms: float = 0.03
    old_label_aux_policy: str = "strong_cap"
    old_lift_max_label_agreement: float = 0.60
    old_lift_min_weakness: float = 0.50
    old_boundary_weight: float = 0.50
    old_boundary_temperature: float = 0.03
    old_boundary_quantile: float = 0.05
    old_boundary_slack: float = 0.00
    old_boundary_min_risk: float = 0.95
    obace_conformal_weight: float = 0.0
    obace_conformal_min_risk: float = 0.80
    obace_absolute_risk_gate: float = 0.70
    obace_old_min_abs_failures: int = 2
    obace_nonold_min_abs_failures: int = 2
    obace_event_weight: float = 0.0
    obace_event_vote_min_risk: float = 0.60
    obace_event_min_votes: int = 2
    obace_event_min_mean_risk: float = 0.60
    obace_event_old_min_local_failures: int = 0
    obace_event_nonold_min_local_failures: int = 0


@dataclass(frozen=True)
class SmecReceiverModel:
    receiver_id: str
    labels: tuple[str, ...]
    old_labels: frozenset[str]
    support_features: np.ndarray
    support_labels: tuple[str, ...]
    centroids: dict[str, np.ndarray]
    proto_thresholds: dict[str, float]
    global_proto_threshold: float
    knn_threshold: float
    energy_threshold: float | None
    old_boundary_margin_thresholds: dict[str, float]
    global_old_boundary_margin_threshold: float
    obace_conformal_scores: dict[str, tuple[float, ...]]
    global_obace_conformal_scores: tuple[float, ...]


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    z = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-z))


def _deficit(value: float, anchor: float) -> float:
    return _unit((float(anchor) - float(value)) / max(float(anchor), 1e-6))


def _top_label(row: Mapping[str, Any]) -> str:
    return str(row.get("class_evidence_top1_label") or row.get("predicted_label") or "")


def _quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, min(1.0, max(0.0, float(q)))))


def _energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    scaled = values / temp
    maxes = np.max(scaled, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(scaled - maxes), axis=1)) + maxes[:, 0]
    return -temp * logsumexp


def build_support_model(
    *,
    receiver_id: str,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    support_logits: np.ndarray | None,
    old_labels: set[str],
    config: SmecConfig,
) -> SmecReceiverModel:
    labels = tuple(str(label) for label in support_labels)
    features = _normalize_rows(np.asarray(support_features, dtype=np.float32))
    if features.shape[0] != len(labels):
        raise ValueError("support_features and support_labels must have matching lengths")
    unique_labels = tuple(sorted(set(labels)))
    centroids: dict[str, np.ndarray] = {}
    proto_thresholds: dict[str, float] = {}
    all_proto_dists: list[float] = []
    for label in unique_labels:
        idx = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
        centroid = _normalize_rows(features[idx].mean(axis=0, keepdims=True))[0]
        centroids[label] = centroid
        dists = (1.0 - np.clip(features[idx] @ centroid, -1.0, 1.0)).astype(np.float64)
        all_proto_dists.extend(float(v) for v in dists.tolist())
        proto_thresholds[label] = _quantile(dists.tolist(), config.proto_quantile) + float(config.proto_slack)
    global_proto_threshold = _quantile(all_proto_dists, config.proto_quantile) + float(config.proto_slack)

    loo_dists: list[float] = []
    if features.shape[0] >= 2:
        sim = np.clip(features @ features.T, -1.0, 1.0)
        for i, label in enumerate(labels):
            same = [j for j, value in enumerate(labels) if value == label and j != i]
            if same:
                loo_dists.append(float(1.0 - np.max(sim[i, same])))
    knn_threshold = _quantile(loo_dists or all_proto_dists, config.knn_quantile) + float(config.knn_slack)

    energy_threshold: float | None = None
    if support_logits is not None and np.asarray(support_logits).size:
        old_idx = [i for i, label in enumerate(labels) if label in old_labels]
        if old_idx:
            old_energy = _energy(np.asarray(support_logits, dtype=np.float32)[old_idx])
            energy_threshold = _quantile(old_energy.tolist(), config.energy_quantile) + float(config.energy_slack)

    old_boundary_margin_thresholds: dict[str, float] = {}
    all_old_boundary_margins: list[float] = []
    if len(centroids) >= 2:
        for label in unique_labels:
            if label not in old_labels:
                continue
            other_centroids = [centroid for other, centroid in centroids.items() if other != label]
            if not other_centroids:
                continue
            other_matrix = np.stack(other_centroids, axis=0)
            idx = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
            own_sim = np.clip(features[idx] @ centroids[label], -1.0, 1.0)
            other_sim = np.max(np.clip(features[idx] @ other_matrix.T, -1.0, 1.0), axis=1)
            margins = (own_sim - other_sim).astype(np.float64).tolist()
            all_old_boundary_margins.extend(float(v) for v in margins)
            old_boundary_margin_thresholds[label] = (
                _quantile(margins, config.old_boundary_quantile) - float(config.old_boundary_slack)
            )
    global_old_boundary_margin_threshold = _quantile(
        all_old_boundary_margins,
        config.old_boundary_quantile,
        default=0.0,
    ) - float(config.old_boundary_slack)

    obace_conformal_scores: dict[str, tuple[float, ...]] = {}
    all_obace_scores: list[float] = []
    if features.shape[0] >= 1:
        sim = np.clip(features @ features.T, -1.0, 1.0)
        for label in unique_labels:
            idx = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
            if idx.size == 0:
                continue
            centroid = centroids[label]
            proto_threshold = max(float(proto_thresholds.get(label, global_proto_threshold)), 1e-6)
            label_scores: list[float] = []
            for i in idx.tolist():
                proto_dist = float(1.0 - np.clip(float(features[i] @ centroid), -1.0, 1.0))
                same = [j for j in idx.tolist() if j != i]
                if same:
                    knn_dist = float(1.0 - np.max(sim[i, same]))
                else:
                    knn_dist = proto_dist
                score = (proto_dist / proto_threshold) + (
                    knn_dist / max(float(knn_threshold), 1e-6)
                )
                label_scores.append(float(score))
            obace_conformal_scores[label] = tuple(label_scores)
            all_obace_scores.extend(label_scores)

    return SmecReceiverModel(
        receiver_id=str(receiver_id),
        labels=unique_labels,
        old_labels=frozenset(str(label) for label in old_labels),
        support_features=features,
        support_labels=labels,
        centroids=centroids,
        proto_thresholds=proto_thresholds,
        global_proto_threshold=float(global_proto_threshold),
        knn_threshold=float(knn_threshold),
        energy_threshold=energy_threshold,
        old_boundary_margin_thresholds=old_boundary_margin_thresholds,
        global_old_boundary_margin_threshold=float(global_old_boundary_margin_threshold),
        obace_conformal_scores=obace_conformal_scores,
        global_obace_conformal_scores=tuple(all_obace_scores),
    )


def _base_strength(row: Mapping[str, Any], config: SmecConfig) -> tuple[bool, float]:
    score = max(_float(row, "known_score"), _float(row, "class_evidence_top1_score"))
    margin = max(_float(row, "known_margin"), _float(row, "class_evidence_top1_margin"))
    support = _float(row, "support_density")
    reliability = max(_float(row, "reliability"), _float(row, "receiver_class_reliability"))
    strong = (
        score >= config.strong_score
        and margin >= config.strong_margin
        and support >= config.strong_support_density
        and reliability >= config.strong_reliability
    )
    weakness = max(
        _deficit(score, config.weak_score_anchor),
        _deficit(margin, config.weak_margin_anchor),
        _deficit(support, config.weak_support_anchor),
        _deficit(reliability, config.weak_reliability_anchor),
    )
    return strong, weakness


def _smec_risks(
    *,
    model: SmecReceiverModel,
    label: str,
    feature: np.ndarray,
    logits: np.ndarray | None,
    config: SmecConfig,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, int]:
    feat = _normalize_rows(np.asarray(feature, dtype=np.float32).reshape(1, -1))[0]
    centroid = model.centroids.get(str(label))
    if centroid is None:
        proto_dist = model.global_proto_threshold + float(config.proto_temperature)
        proto_threshold = model.global_proto_threshold
    else:
        proto_dist = float(1.0 - np.clip(float(feat @ centroid), -1.0, 1.0))
        proto_threshold = float(model.proto_thresholds.get(str(label), model.global_proto_threshold))
    proto_risk = _sigmoid((proto_dist - proto_threshold) / max(float(config.proto_temperature), 1e-6))

    sims = np.clip(model.support_features @ feat, -1.0, 1.0)
    knn_dist = float(1.0 - np.max(sims)) if sims.size else float(model.knn_threshold)
    knn_risk = _sigmoid((knn_dist - model.knn_threshold) / max(float(config.knn_temperature), 1e-6))

    energy_risk = 0.0
    if logits is not None and model.energy_threshold is not None and str(label) in model.old_labels:
        query_energy = float(_energy(np.asarray(logits, dtype=np.float32).reshape(1, -1))[0])
        energy_risk = _sigmoid(
            (query_energy - float(model.energy_threshold)) / max(float(config.energy_temperature), 1e-6)
        )
    old_boundary_margin = math.inf
    old_boundary_threshold = float(model.global_old_boundary_margin_threshold)
    old_boundary_risk = 0.0
    if str(label) in model.old_labels and len(model.centroids) >= 2:
        own_centroid = model.centroids.get(str(label))
        other_centroids = [centroid for other, centroid in model.centroids.items() if other != str(label)]
        if own_centroid is not None and other_centroids:
            other_matrix = np.stack(other_centroids, axis=0)
            own_sim = float(np.clip(float(feat @ own_centroid), -1.0, 1.0))
            other_sim = float(np.max(np.clip(other_matrix @ feat, -1.0, 1.0)))
            old_boundary_margin = own_sim - other_sim
            old_boundary_threshold = float(
                model.old_boundary_margin_thresholds.get(str(label), model.global_old_boundary_margin_threshold)
            )
            old_boundary_risk = _sigmoid(
                (old_boundary_threshold - old_boundary_margin)
                / max(float(config.old_boundary_temperature), 1e-6)
            )
    conformal_scores = model.obace_conformal_scores.get(str(label), model.global_obace_conformal_scores)
    obace_nonconformity = (proto_dist / max(float(proto_threshold), 1e-6)) + (
        knn_dist / max(float(model.knn_threshold), 1e-6)
    )
    if conformal_scores:
        obace_conformal_pvalue = (1.0 + sum(1 for value in conformal_scores if value >= obace_nonconformity)) / (
            len(conformal_scores) + 1.0
        )
    else:
        obace_conformal_pvalue = 1.0
    obace_conformal_risk = 1.0 - _unit(obace_conformal_pvalue)
    absolute_fail_count = 0
    gate = float(config.obace_absolute_risk_gate)
    if float(config.proto_weight) > 0.0 and proto_risk >= gate:
        absolute_fail_count += 1
    if float(config.knn_weight) > 0.0 and knn_risk >= gate:
        absolute_fail_count += 1
    if float(config.energy_weight) > 0.0 and energy_risk >= gate:
        absolute_fail_count += 1
    if float(config.old_boundary_weight) > 0.0 and old_boundary_risk >= float(config.old_boundary_min_risk):
        absolute_fail_count += 1
    if float(config.obace_conformal_weight) > 0.0 and obace_conformal_risk >= float(config.obace_conformal_min_risk):
        absolute_fail_count += 1
    aux = _unit(
        float(config.proto_weight) * proto_risk
        + float(config.knn_weight) * knn_risk
        + float(config.energy_weight) * energy_risk
        + float(config.old_boundary_weight) * old_boundary_risk
        + float(config.obace_conformal_weight) * obace_conformal_risk
    )
    return (
        proto_risk,
        knn_risk,
        energy_risk,
        old_boundary_risk,
        obace_conformal_risk,
        aux,
        max(proto_dist, knn_dist),
        old_boundary_margin,
        old_boundary_threshold,
        obace_conformal_pvalue,
        obace_nonconformity,
        absolute_fail_count,
    )


def augment_smec_evidence(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, SmecReceiverModel],
    query_features: Mapping[tuple[str, str], np.ndarray],
    query_logits: Mapping[tuple[str, str], np.ndarray],
    config: SmecConfig,
    *,
    old_labels: set[str],
) -> list[dict[str, Any]]:
    event_stats: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for source in rows:
        grouped.setdefault(str(source.get("event_id", "")), []).append(source)
    for event_id, group in grouped.items():
        labels = [_top_label(row) for row in group]
        counts = {label: labels.count(label) for label in set(labels)}
        n = max(len(labels), 1)
        event_stats[event_id] = {
            "receiver_count": float(n),
            "label_agreement": max(counts.values()) / float(n) if counts else 0.0,
        }
    risk_cache: dict[
        tuple[str, str],
        tuple[float, float, float, float, float, float, float, float, float, float, float, int],
    ] = {}
    event_obace_stats: dict[str, dict[str, float]] = {}

    def _risk_for(
        source: Mapping[str, Any],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, int] | None:
        rx = str(source.get("receiver_id", ""))
        event_id = str(source.get("event_id", ""))
        key = (event_id, rx)
        if key in risk_cache:
            return risk_cache[key]
        model = models.get(rx)
        feature = query_features.get(key)
        if model is None or feature is None:
            return None
        risk_cache[key] = _smec_risks(
            model=model,
            label=_top_label(source),
            feature=feature,
            logits=query_logits.get(key),
            config=config,
        )
        return risk_cache[key]

    def _event_obace_for(event_id: str) -> dict[str, float]:
        if event_id in event_obace_stats:
            return event_obace_stats[event_id]
        conformal_risks: list[float] = []
        fail_counts: list[int] = []
        for peer in grouped.get(event_id, []):
            risks = _risk_for(peer)
            if risks is None:
                continue
            conformal_risks.append(float(risks[4]))
            fail_counts.append(int(risks[11]))
        n = max(len(conformal_risks), 1)
        vote_count = sum(1 for risk in conformal_risks if risk >= float(config.obace_event_vote_min_risk))
        fail_vote_count = sum(1 for count in fail_counts if count > 0)
        mean_risk = sum(conformal_risks) / float(n) if conformal_risks else 0.0
        vote_ratio = vote_count / float(n)
        fail_vote_ratio = fail_vote_count / float(n)
        event_risk = _unit(mean_risk + 0.15 * vote_ratio + 0.10 * fail_vote_ratio)
        stats = {
            "risk": event_risk,
            "mean_risk": mean_risk,
            "vote_count": float(vote_count),
            "fail_vote_count": float(fail_vote_count),
            "receiver_count": float(len(conformal_risks)),
        }
        event_obace_stats[event_id] = stats
        return stats

    out: list[dict[str, Any]] = []
    for source in rows:
        rx = str(source.get("receiver_id", ""))
        event_id = str(source.get("event_id", ""))
        key = (event_id, rx)
        row = dict(source)
        base_risk = _unit(_float(source, "unknown_risk"))
        model = models.get(rx)
        feature = query_features.get(key)
        label = _top_label(source)
        if model is None or feature is None:
            row["smec_missing_feature"] = 1
            row["smec_label_authority"] = "base_qknn_only"
            out.append(row)
            continue
        risk_values = _risk_for(source)
        if risk_values is None:
            row["smec_missing_feature"] = 1
            row["smec_label_authority"] = "base_qknn_only"
            out.append(row)
            continue
        (
            proto_risk,
            knn_risk,
            energy_risk,
            old_boundary_risk,
            obace_conformal_risk,
            aux_raw,
            distance,
            old_boundary_margin,
            old_boundary_threshold,
            obace_conformal_pvalue,
            obace_nonconformity,
            obace_absolute_fail_count,
        ) = risk_values
        obace_event = _event_obace_for(event_id)
        obace_event_risk = float(obace_event.get("risk", 0.0))
        obace_event_pass = (
            float(config.obace_event_weight) > 0.0
            and obace_event_risk >= float(config.obace_event_min_mean_risk)
            and float(obace_event.get("vote_count", 0.0)) >= float(config.obace_event_min_votes)
        )
        strong_known, weakness = _base_strength(source, config)
        aux_component = _unit(aux_raw * max(weakness, 0.20))
        if obace_event_pass:
            aux_component = max(
                aux_component,
                _unit(float(config.obace_event_weight) * obace_event_risk * max(weakness, 0.20)),
            )
        old_label_lift_blocked = 0
        old_policy = str(config.old_label_aux_policy or "strong_cap").strip().lower()
        if old_policy not in {
            "strong_cap",
            "never",
            "consensus_guard",
            "old_boundary_guard",
            "obace_guard",
            "obace_event_guard",
        }:
            raise ValueError(
                "old_label_aux_policy must be strong_cap, never, consensus_guard, old_boundary_guard, obace_guard, or obace_event_guard"
            )
        if old_policy == "never" and label in old_labels:
            aux_component = min(aux_component, base_risk)
            old_label_lift_blocked = 1
        if old_policy == "consensus_guard" and label in old_labels:
            stats = event_stats.get(event_id, {})
            label_agreement = float(stats.get("label_agreement", 1.0))
            allow_old_lift = (
                label_agreement <= float(config.old_lift_max_label_agreement)
                and weakness >= float(config.old_lift_min_weakness)
                and not strong_known
            )
            if not allow_old_lift:
                aux_component = min(aux_component, base_risk)
                old_label_lift_blocked = 1
        if old_policy == "old_boundary_guard" and label in old_labels:
            stats = event_stats.get(event_id, {})
            label_agreement = float(stats.get("label_agreement", 1.0))
            allow_by_disagreement = label_agreement <= float(config.old_lift_max_label_agreement)
            allow_by_boundary = old_boundary_risk >= float(config.old_boundary_min_risk)
            allow_old_lift = (
                (allow_by_disagreement or allow_by_boundary)
                and weakness >= float(config.old_lift_min_weakness)
                and not strong_known
            )
            if not allow_old_lift:
                aux_component = min(aux_component, base_risk)
                old_label_lift_blocked = 1
        if old_policy == "obace_guard" and label in old_labels:
            allow_old_lift = (
                int(obace_absolute_fail_count) >= int(config.obace_old_min_abs_failures)
                and obace_conformal_risk >= float(config.obace_conformal_min_risk)
                and weakness >= float(config.old_lift_min_weakness)
                and not strong_known
            )
            if not allow_old_lift:
                aux_component = min(aux_component, base_risk)
                old_label_lift_blocked = 1
        if old_policy == "obace_guard" and label not in old_labels:
            allow_nonold_lift = (
                int(obace_absolute_fail_count) >= int(config.obace_nonold_min_abs_failures)
                and obace_conformal_risk >= float(config.obace_conformal_min_risk)
                and weakness >= float(config.old_lift_min_weakness)
            )
            if not allow_nonold_lift:
                aux_component = min(aux_component, base_risk)
                old_label_lift_blocked = 1
        if old_policy == "obace_event_guard":
            local_min_failures = (
                int(config.obace_event_old_min_local_failures)
                if label in old_labels
                else int(config.obace_event_nonold_min_local_failures)
            )
            allow_event_lift = (
                obace_event_pass
                and int(obace_absolute_fail_count) >= local_min_failures
                and weakness >= float(config.old_lift_min_weakness)
                and not strong_known
            )
            if not allow_event_lift:
                aux_component = min(aux_component, base_risk)
                old_label_lift_blocked = 1
        if strong_known and label in old_labels:
            aux_component = min(aux_component, float(config.strong_aux_cap))
        fused = max(base_risk, aux_component)
        row["unknown_risk"] = fused
        row["class_evidence_top1_unknown_risk"] = fused
        row["smec_proto_risk"] = proto_risk
        row["smec_knn_risk"] = knn_risk
        row["smec_energy_risk"] = energy_risk
        row["smec_old_boundary_risk"] = old_boundary_risk
        row["smec_obace_conformal_risk"] = obace_conformal_risk
        row["smec_obace_conformal_pvalue"] = obace_conformal_pvalue
        row["smec_obace_nonconformity"] = obace_nonconformity
        row["smec_obace_absolute_fail_count"] = int(obace_absolute_fail_count)
        row["smec_obace_event_risk"] = obace_event_risk
        row["smec_obace_event_mean_risk"] = float(obace_event.get("mean_risk", 0.0))
        row["smec_obace_event_vote_count"] = float(obace_event.get("vote_count", 0.0))
        row["smec_obace_event_fail_vote_count"] = float(obace_event.get("fail_vote_count", 0.0))
        row["smec_old_boundary_margin"] = old_boundary_margin
        row["smec_old_boundary_threshold"] = old_boundary_threshold
        row["smec_aux_raw"] = aux_raw
        row["smec_aux_component"] = aux_component
        row["smec_base_unknown_risk"] = base_risk
        row["smec_base_weakness"] = weakness
        row["smec_distance"] = distance
        row["smec_strong_known_candidate"] = int(strong_known and label in old_labels)
        row["smec_old_label_aux_policy"] = old_policy
        row["smec_old_label_lift_blocked"] = old_label_lift_blocked
        row["smec_event_label_agreement"] = float(event_stats.get(event_id, {}).get("label_agreement", 0.0))
        row["smec_event_receiver_count"] = float(event_stats.get(event_id, {}).get("receiver_count", 0.0))
        row["smec_missing_feature"] = 0
        row["smec_unknown_query_used_for_threshold"] = "false"
        row["smec_label_authority"] = "base_qknn_only"
        row["bytes"] = _float(source, "bytes") + max(0.0, float(config.aux_bytes_per_receiver))
        row["latency_ms"] = _float(source, "latency_ms") + max(0.0, float(config.aux_latency_ms))
        row["reliability_source"] = "smec_ci_support_metric_energy"
        out.append(row)
    return out


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _stage2_sets(payload: Mapping[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    old_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_old")[0].tolist()})
    new_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_new")[0].tolist()})
    unknown_labels = sorted({str(tx_ids[i]) for i in np.where(roles == UNKNOWN_ROLE)[0].tolist()})
    target_receivers = sorted({str(rx_ids[i]) for i in np.where(np.isin(roles, [*KNOWN_ROLES, UNKNOWN_ROLE]))[0].tolist()})
    return old_labels, new_labels, unknown_labels, target_receivers


def build_stage2_smec_inputs(
    payload: Mapping[str, Any],
    config: SmecConfig,
    *,
    k_shot: int = 8,
    query_per_class: int = 20,
    seed: int = 4070303,
    support_selection_policy: str = "stable_first",
) -> tuple[dict[str, SmecReceiverModel], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
    logits = np.asarray(payload["tx_logits"], dtype=np.float32) if "tx_logits" in payload else None
    old_labels, new_labels, unknown_labels, target_receivers = _stage2_sets(payload)
    models: dict[str, SmecReceiverModel] = {}
    receiver_query: dict[str, dict[str, list[int]]] = {}

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
            _, query = _split_support_query(
                payload,
                role=UNKNOWN_ROLE,
                tx_id=label,
                rx_id=rx,
                k_shot=0,
                query_per_class=query_per_class,
                seed=seed,
            )
            _require_split(rx, UNKNOWN_ROLE, label, [], query, k_shot=0, query_per_class=query_per_class)
            receiver_query[rx]["unknown"].extend(query)
        support_logits = logits[support_indices] if logits is not None and support_indices else None
        models[rx] = build_support_model(
            receiver_id=rx,
            support_features=features[support_indices],
            support_labels=support_labels,
            support_logits=support_logits,
            old_labels=set(old_labels),
            config=config,
        )

    query_features: dict[tuple[str, str], np.ndarray] = {}
    query_logits: dict[tuple[str, str], np.ndarray] = {}
    for role_name in ["old", "seen_new", "unknown"]:
        label_set = old_labels if role_name == "old" else new_labels if role_name == "seen_new" else unknown_labels
        for label in label_set:
            by_rx_scenario: dict[str, dict[str, list[int]]] = {}
            for rx in target_receivers:
                by_rx_scenario[rx] = {}
                for idx in receiver_query[rx][role_name]:
                    if str(tx_ids[int(idx)]) != label:
                        continue
                    scenario = _scenario_of(payload, int(idx))
                    by_rx_scenario[rx].setdefault(scenario, []).append(int(idx))
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
            for scenario in common_scenarios:
                n = max(len(by_rx_scenario[rx].get(scenario, [])) for rx in target_receivers)
                for event_i in range(n):
                    event_id = f"{role_name}|{label}|{scenario}|rank{event_i:05d}"
                    for rx in target_receivers:
                        rows = by_rx_scenario[rx].get(scenario, [])
                        if event_i >= len(rows):
                            continue
                        idx = int(rows[event_i])
                        query_features[(event_id, rx)] = features[idx]
                        if logits is not None:
                            query_logits[(event_id, rx)] = logits[idx]
    metadata = {
        "target_receiver_ids": target_receivers,
        "old_tx_ids": old_labels,
        "seen_new_tx_ids": new_labels,
        "unknown_tx_ids": unknown_labels,
        "support_model_count": len(models),
        "query_feature_count": len(query_features),
        "unknown_query_used_for_threshold": False,
    }
    return models, query_features, query_logits, metadata


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("qknn_metadata"), dict):
        return dict(data["qknn_metadata"])
    if isinstance(data.get("metadata"), dict):
        return dict(data["metadata"])
    return dict(data)


def _select_policies(spec: str) -> list[OpuPolicy]:
    if str(spec).strip().lower() in {"all", "*"}:
        return list(POLICIES)
    wanted = {part.strip() for part in str(spec).split(",") if part.strip()}
    selected = [policy for policy in POLICIES if policy.name in wanted]
    missing = sorted(wanted - {policy.name for policy in selected})
    if missing:
        raise ValueError(f"unknown policies: {missing}")
    return selected


def _flatten_counts(
    *,
    algorithm: str,
    policy: OpuPolicy,
    metrics: Mapping[str, Any],
    base_counts: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count, count_metrics in sorted(metrics.get("counts", {}).items(), key=lambda item: int(item[0])):
        row = _summary_row(policy=policy, count=str(count), metrics=count_metrics)
        row.update(
            {
                "algorithm": algorithm,
                "threshold_fit_scope": metrics.get("threshold_selection_label_scope", ""),
                "unknown_query_used_for_threshold": "false",
            }
        )
        if base_counts is not None:
            base = base_counts[str(count)]
            row["delta_old_acc"] = float(row["old_acc"]) - float(base.get("old_acc", 0.0))
            row["delta_seen_new_acc"] = float(row["seen_new_acc"]) - float(base.get("seen_new_acc", 0.0))
            row["delta_unknown_reject_rate"] = float(row["unknown_reject_rate"]) - float(
                base.get("unknown_reject_rate", 0.0)
            )
            row["delta_unknown_FAR"] = float(row["unknown_FAR"]) - float(base.get("unknown_FAR", 0.0))
            row["old_not_drop_pass"] = float(row["delta_old_acc"]) >= -1e-12
            row["verdict"] = (
                "candidate"
                if row["old_not_drop_pass"]
                and float(row["old_acc"]) >= 0.80
                and float(row["delta_unknown_reject_rate"]) > 0.0
                and float(row["delta_unknown_FAR"]) <= 0.0
                else "diagnostic_only"
            )
        rows.append(row)
    return rows


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    parser.add_argument("--max_event_bytes", type=float, default=900.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--write_evidence", action="store_true")
    parser.add_argument("--profiles", default="standard,old_lossless")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    base_rows = read_csv_rows(known_csv)
    metadata = _load_metadata(known_json)
    requested_profiles = [part.strip() for part in str(args.profiles).split(",") if part.strip()]
    if not requested_profiles:
        raise ValueError("at least one --profiles value is required")
    profile_configs: dict[str, SmecConfig] = {}
    for profile in requested_profiles:
        if profile == "standard":
            profile_configs["standard"] = SmecConfig()
        elif profile == "old_lossless":
            profile_configs["old_lossless"] = SmecConfig(old_label_aux_policy="never")
        elif profile == "consensus_guard":
            profile_configs["consensus_guard"] = SmecConfig(
                old_label_aux_policy="consensus_guard",
                old_lift_max_label_agreement=0.60,
                old_lift_min_weakness=0.50,
            )
        elif profile == "old_boundary_guard":
            profile_configs["old_boundary_guard"] = SmecConfig(
                old_label_aux_policy="old_boundary_guard",
                old_lift_max_label_agreement=0.60,
                old_lift_min_weakness=0.50,
                old_boundary_weight=0.60,
                old_boundary_min_risk=0.98,
            )
        elif profile == "obace":
            profile_configs["obace"] = SmecConfig(
                old_label_aux_policy="obace_guard",
                old_lift_min_weakness=0.50,
                proto_weight=0.60,
                knn_weight=0.50,
                energy_weight=0.08,
                old_boundary_weight=0.25,
                old_boundary_min_risk=0.98,
                obace_conformal_weight=0.70,
                obace_conformal_min_risk=0.85,
                obace_absolute_risk_gate=0.70,
                obace_old_min_abs_failures=3,
                obace_nonold_min_abs_failures=3,
            )
        elif profile == "obace_event":
            profile_configs["obace_event"] = SmecConfig(
                old_label_aux_policy="obace_event_guard",
                old_lift_min_weakness=0.50,
                proto_weight=0.55,
                knn_weight=0.45,
                energy_weight=0.08,
                old_boundary_weight=0.20,
                old_boundary_min_risk=0.98,
                obace_conformal_weight=0.55,
                obace_conformal_min_risk=0.85,
                obace_absolute_risk_gate=0.70,
                obace_old_min_abs_failures=3,
                obace_nonold_min_abs_failures=3,
                obace_event_weight=0.60,
                obace_event_vote_min_risk=0.55,
                obace_event_min_votes=4,
                obace_event_min_mean_risk=0.75,
                obace_event_old_min_local_failures=2,
                obace_event_nonold_min_local_failures=2,
            )
        else:
            raise ValueError(
                "unknown SMEC profile; expected standard, old_lossless, consensus_guard, old_boundary_guard, obace, or obace_event"
            )
    build_config = next(iter(profile_configs.values()))
    payload = load_feature_npz(args.feature_npz)
    models, query_features, query_logits, smec_metadata = build_stage2_smec_inputs(
        payload,
        build_config,
        k_shot=int(metadata.get("k_shot", 8)),
        query_per_class=int(metadata.get("query_per_class", 20)),
        seed=4070303,
        support_selection_policy=str(metadata.get("support_selection_policy", "stable_first")),
    )

    policies = _select_policies(args.policies)
    summary_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    base_metrics: dict[str, dict[str, Any]] = {
        policy.name: _evaluate_policy(
            base_rows,
            metadata,
            policy,
            max_event_bytes=args.max_event_bytes,
            max_event_latency_ms=args.max_event_latency_ms,
        )
        for policy in policies
    }
    for policy in policies:
        summary_rows.extend(
            _flatten_counts(algorithm="base_known_route", policy=policy, metrics=base_metrics[policy.name])
        )
    profile_audits: dict[str, Any] = {}
    for profile, config in profile_configs.items():
        algorithm = "smec_old_lossless_ci" if profile == "old_lossless" else "smec_ci"
        if profile == "consensus_guard":
            algorithm = "smec_consensus_guard_ci"
        if profile == "old_boundary_guard":
            algorithm = "smec_old_boundary_guard_ci"
        if profile == "obace":
            algorithm = "obace_ci"
        if profile == "obace_event":
            algorithm = "obace_event_ci"
        smec_rows = augment_smec_evidence(
            base_rows,
            models,
            query_features,
            query_logits,
            config,
            old_labels=set(metadata.get("old_tx_ids", smec_metadata["old_tx_ids"])),
        )
        if args.write_evidence:
            write_csv_rows(args.output_dir / f"{algorithm}_evidence.csv", smec_rows)
        profile_audits[profile] = {
            "algorithm": algorithm,
            "config": config.__dict__,
            "row_count": len(smec_rows),
            "old_label_lift_blocked_count": sum(
                int(row.get("smec_old_label_lift_blocked", 0) or 0) for row in smec_rows
            ),
        }
        for policy in policies:
            metrics = _evaluate_policy(
                smec_rows,
                metadata,
                policy,
                max_event_bytes=args.max_event_bytes,
                max_event_latency_ms=args.max_event_latency_ms,
            )
            rows = _flatten_counts(
                algorithm=algorithm,
                policy=policy,
                metrics=metrics,
                base_counts=base_metrics[policy.name]["counts"],
            )
            summary_rows.extend(rows)
            best_rows.extend(rows)

    best_rows = sorted(
        best_rows,
        key=lambda row: (
            row.get("verdict") == "candidate",
            float(row.get("old_acc", 0.0)),
            float(row.get("delta_unknown_reject_rate", 0.0)),
            float(row.get("unknown_reject_rate", 0.0)),
            -float(row.get("unknown_FAR", 1.0)),
        ),
        reverse=True,
    )
    write_csv_rows(args.output_dir / "smec_ci_summary.csv", summary_rows)
    write_csv_rows(args.output_dir / "smec_ci_best_rows.csv", best_rows[:50])
    (args.output_dir / "smec_ci_audit.json").write_text(
        json.dumps(
            {
                "algorithm": "SMEC-CI",
                "base_evidence_csv": str(known_csv),
                "metadata_json": str(known_json),
                "config": config.__dict__,
                "profile_audits": profile_audits,
                "support_only_thresholds": True,
                "unknown_query_used_for_threshold": False,
                "label_authority": "base_qknn_only",
                "smec_metadata": smec_metadata,
                "summary_rows": len(summary_rows),
                "candidate_count": sum(1 for row in best_rows if row.get("verdict") == "candidate"),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_rows": len(summary_rows),
                "candidate_count": sum(1 for row in best_rows if row.get("verdict") == "candidate"),
                "summary_csv": str(args.output_dir / "smec_ci_summary.csv"),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
