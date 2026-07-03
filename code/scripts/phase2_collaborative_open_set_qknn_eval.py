#!/usr/bin/env python
"""Build qknn8 collaborative open-set evidence from Stage2-C feature NPZ files.

The script keeps the Phase1 backbone frozen and performs only CPU-side support
memory updates. It requires target_old, target_new, and target_unknown rows.
Unknown query rows are evaluation-only and never used to set thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

KNOWN_ROLES = {"target_old", "target_new"}
UNKNOWN_ROLE = "target_unknown"
PROXY_UNKNOWN_ROLE = "proxy_unknown"


@dataclass(frozen=True)
class QknnMemory:
    qfeatures: np.ndarray
    labels: np.ndarray
    old_labels: set[str]
    support_scenarios: np.ndarray
    radii_by_support: np.ndarray
    scale: float
    prototype_storage_bytes: int
    centroid_labels: np.ndarray
    centroids: np.ndarray
    class_radius_thresholds: dict[str, float]
    class_mahalanobis_inv_vars: dict[str, np.ndarray]
    class_mahalanobis_thresholds: dict[str, float]
    class_evt_thresholds: dict[str, float]
    class_evt_scales: dict[str, float]
    class_oldness_weights: dict[str, np.ndarray]
    class_oldness_thresholds: dict[str, float]
    margin_threshold: float
    score_threshold: float


@dataclass(frozen=True)
class FeatureAdapter:
    policy: str
    center: np.ndarray
    scale: np.ndarray
    strength: float


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def _parse_csv(text: str | None) -> list[str]:
    return [canonical_tx_id(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _stable_score(parts: Sequence[object], seed: int) -> float:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64 - 1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def _fit_feature_adapter(
    support_features: np.ndarray,
    *,
    policy: str,
    strength: float,
    variance_floor: float,
) -> FeatureAdapter:
    policy = str(policy or "none").strip().lower()
    if policy not in {"none", "support_center", "support_bn_affine"}:
        raise ValueError("feature_adapter_policy must be none, support_center, or support_bn_affine")
    variance_floor_value = max(float(variance_floor), 1e-8)
    normalized = _normalize_rows(support_features)
    dim = int(normalized.shape[1])
    strength_value = float(np.clip(float(strength), 0.0, 1.0))
    if policy == "none" or normalized.shape[0] <= 0 or strength_value <= 0.0:
        return FeatureAdapter(
            policy="none",
            center=np.zeros((dim,), dtype=np.float32),
            scale=np.ones((dim,), dtype=np.float32),
            strength=0.0,
        )
    center = normalized.mean(axis=0).astype(np.float32)
    if policy == "support_bn_affine":
        var = np.var(normalized, axis=0).astype(np.float32)
        support_scale = np.sqrt(np.maximum(var, variance_floor_value)).astype(np.float32)
        scale = ((1.0 - strength_value) + strength_value * support_scale).astype(np.float32)
        scale = np.maximum(scale, variance_floor_value).astype(np.float32)
    else:
        scale = np.ones((dim,), dtype=np.float32)
    return FeatureAdapter(policy=policy, center=center, scale=scale, strength=strength_value)


def _apply_feature_adapter(features: np.ndarray, adapter: FeatureAdapter) -> np.ndarray:
    values = _normalize_rows(features)
    if adapter.policy == "none" or float(adapter.strength) <= 0.0:
        return values
    centered = values - float(adapter.strength) * adapter.center[None, :]
    if adapter.policy == "support_bn_affine":
        centered = centered / adapter.scale[None, :]
    return _normalize_rows(centered.astype(np.float32))


def _as_str(data: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    if key not in data:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if int(arr.shape[0]) != int(n):
        raise ValueError(f"{key} length mismatch: expected {n}, got {arr.shape[0]}")
    return arr.astype(str)


def load_feature_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        features = _normalize_rows(np.asarray(data["features"], dtype=np.float32))
        n = int(features.shape[0])
        payload = {
            "features": features,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": np.asarray([canonical_tx_id(v) for v in _as_str(data, "tx_ids", n)], dtype=str),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
            "sat_scenarios": _as_str(data, "sat_scenarios", n),
            "channel_views": _as_str(data, "channel_views", n),
            "manifest": {},
        }
        if "manifest_json" in data.files:
            raw = np.asarray(data["manifest_json"]).item()
            payload["manifest"] = json.loads(str(raw))
        return payload


def validate_required_roles(payload: Mapping[str, Any]) -> None:
    roles = {str(role) for role in np.asarray(payload["dataset_role"]).tolist()}
    missing = sorted((KNOWN_ROLES | {UNKNOWN_ROLE}) - roles)
    if missing:
        raise RuntimeError(
            "LOCAL_DATASET_EXTENSION_REQUIRED: feature NPZ must contain "
            f"{sorted(KNOWN_ROLES | {UNKNOWN_ROLE})}; missing={missing}"
        )


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


def _class_centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_values = sorted({str(label) for label in labels.tolist()})
    centroids = []
    for label in label_values:
        centroids.append(_normalize_rows(features[labels == label].mean(axis=0, keepdims=True))[0])
    return np.asarray(label_values, dtype=object), _normalize_rows(np.vstack(centroids))


def _calibrate_seen_new_centroids(
    centroid_labels: np.ndarray,
    centroids: np.ndarray,
    *,
    old_labels: set[str],
    policy: str,
    alpha: float,
    top_m: int,
) -> tuple[np.ndarray, dict[str, float]]:
    policy = str(policy or "none").strip().lower()
    if policy not in {"none", "teen_blend", "teen_separate"}:
        raise ValueError("prototype_calibration_policy must be none, teen_blend, or teen_separate")
    if policy == "none":
        return centroids, {}
    alpha_value = float(np.clip(float(alpha), 0.0, 1.0))
    if alpha_value <= 0.0:
        return centroids, {}
    labels = [str(label) for label in centroid_labels.tolist()]
    old_positions = [i for i, label in enumerate(labels) if label in old_labels]
    seen_positions = [i for i, label in enumerate(labels) if label not in old_labels]
    if not old_positions or not seen_positions:
        return centroids, {}
    calibrated = np.asarray(centroids, dtype=np.float32).copy()
    old_matrix = _normalize_rows(calibrated[old_positions])
    applied: dict[str, float] = {}
    m = max(1, int(top_m))
    for pos in seen_positions:
        original = calibrated[pos]
        sims = old_matrix @ original
        chosen_local = np.argsort(-sims)[: min(m, sims.shape[0])]
        chosen_sims = sims[chosen_local]
        weights = np.exp(chosen_sims - np.max(chosen_sims))
        weights = weights / max(float(np.sum(weights)), 1e-12)
        old_mix = _normalize_rows((weights[:, None] * old_matrix[chosen_local]).sum(axis=0, keepdims=True))[0]
        if policy == "teen_blend":
            updated = (1.0 - alpha_value) * original + alpha_value * old_mix
        else:
            updated = original + alpha_value * (original - old_mix)
        calibrated[pos] = _normalize_rows(updated.reshape(1, -1))[0]
        applied[labels[pos]] = alpha_value
    return _normalize_rows(calibrated), applied


def _centroid_scores(memory: QknnMemory, query_features: np.ndarray) -> np.ndarray:
    return _normalize_rows(query_features) @ memory.centroids.T


def _mahalanobis_known_scores(memory: QknnMemory, query_features: np.ndarray, *, temperature: float) -> np.ndarray:
    query = _normalize_rows(query_features)
    temp = max(float(temperature), 1e-6)
    out = np.zeros((query.shape[0], memory.centroid_labels.shape[0]), dtype=np.float64)
    for pos, label_value in enumerate(memory.centroid_labels.tolist()):
        label = str(label_value)
        inv_var = memory.class_mahalanobis_inv_vars.get(label)
        if inv_var is None:
            continue
        centroid = memory.centroids[pos]
        distances = np.sqrt(np.mean(((query - centroid[None, :]) ** 2) * inv_var[None, :], axis=1))
        threshold = float(memory.class_mahalanobis_thresholds.get(label, 1.0))
        z = np.clip((distances - threshold) / temp, -60.0, 60.0)
        out[:, pos] = 1.0 / (1.0 + np.exp(z))
    return np.clip(out, 0.0, 1.0)


def _virtual_unknown_features(
    memory: QknnMemory,
    *,
    samples_per_class: int,
    seed: int,
    mix_alpha: float,
    noise_scale: float,
    neighbor_count: int,
) -> np.ndarray:
    samples = max(0, int(samples_per_class))
    if samples == 0 or int(memory.centroids.shape[0]) < 2:
        return np.zeros((0, int(memory.centroids.shape[1])), dtype=np.float32)
    alpha = float(np.clip(float(mix_alpha), 0.05, 0.95))
    noise = max(0.0, float(noise_scale))
    n_neighbors = max(1, int(neighbor_count))
    centroids = _normalize_rows(memory.centroids)
    dim = int(centroids.shape[1])
    rng = np.random.default_rng(int(seed))
    rows: list[np.ndarray] = []
    similarity = centroids @ centroids.T
    for class_i in range(int(centroids.shape[0])):
        order = np.argsort(-similarity[class_i])
        neighbors = [int(pos) for pos in order.tolist() if int(pos) != class_i][:n_neighbors]
        if not neighbors:
            continue
        for sample_i in range(samples):
            other_i = neighbors[sample_i % len(neighbors)]
            local_alpha = alpha
            if samples > 1:
                span = (sample_i / max(samples - 1, 1)) - 0.5
                local_alpha = float(np.clip(alpha + 0.20 * span, 0.05, 0.95))
            point = (1.0 - local_alpha) * centroids[class_i] + local_alpha * centroids[other_i]
            if noise > 0.0:
                direction = rng.normal(0.0, noise, size=dim).astype(np.float32)
                direction = direction - point * float(direction @ point)
                point = point + direction
            rows.append(point.astype(np.float32))
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return _normalize_rows(np.vstack(rows)).astype(np.float32)


def _support_envelope(
    features: np.ndarray,
    labels: np.ndarray,
    centroid_labels: np.ndarray,
    centroids: np.ndarray,
    *,
    radius_quantile: float,
    margin_quantile: float,
    score_quantile: float,
    radius_slack: float,
    margin_slack: float,
    score_slack: float,
) -> tuple[dict[str, float], float, float]:
    scores = _normalize_rows(features) @ centroids.T
    class_to_pos = {str(label): int(i) for i, label in enumerate(centroid_labels.tolist())}
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2] if scores.shape[1] >= 2 else sorted_scores[:, -1]
    best_scores = np.max(scores, axis=1)
    radii: dict[str, float] = {}
    for label in sorted(class_to_pos):
        idx = np.where(labels == label)[0]
        if idx.size:
            label_score = scores[idx, class_to_pos[label]]
            radii[label] = float(np.quantile(1.0 - label_score, float(radius_quantile)) + float(radius_slack))
    margin_t = float(max(0.0, np.quantile(margins, float(margin_quantile)) - float(margin_slack)))
    score_t = float(max(0.0, np.quantile(best_scores, float(score_quantile)) - float(score_slack)))
    return radii, margin_t, score_t


def _mahalanobis_envelope(
    features: np.ndarray,
    labels: np.ndarray,
    centroid_labels: np.ndarray,
    centroids: np.ndarray,
    *,
    mahalanobis_quantile: float,
    mahalanobis_slack: float,
    variance_floor: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    normalized = _normalize_rows(features)
    class_to_pos = {str(label): int(i) for i, label in enumerate(centroid_labels.tolist())}
    inv_vars: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    support_distances: dict[str, np.ndarray] = {}
    global_var = np.var(normalized, axis=0) + float(variance_floor)
    for label in sorted(class_to_pos):
        idx = np.where(labels == label)[0]
        if idx.size <= 0:
            continue
        class_features = normalized[idx]
        centroid = centroids[class_to_pos[label]]
        class_var = np.var(class_features, axis=0) if idx.size > 1 else global_var
        var = np.maximum(0.5 * class_var + 0.5 * global_var, float(variance_floor))
        inv_var = 1.0 / var
        distances = np.sqrt(np.mean(((class_features - centroid) ** 2) * inv_var[None, :], axis=1))
        inv_vars[label] = inv_var.astype(np.float32)
        thresholds[label] = float(np.quantile(distances, float(mahalanobis_quantile)) + float(mahalanobis_slack))
        support_distances[label] = distances.astype(np.float64)
    return inv_vars, thresholds, support_distances


def _evt_tail_envelope(
    support_distances: Mapping[str, np.ndarray],
    *,
    evt_tail_quantile: float,
    evt_tail_slack: float,
    evt_min_scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    thresholds: dict[str, float] = {}
    scales: dict[str, float] = {}
    for label, distances in support_distances.items():
        values = np.asarray(distances, dtype=np.float64)
        if values.size <= 0:
            continue
        threshold = float(np.quantile(values, float(evt_tail_quantile)) + float(evt_tail_slack))
        excess = values[values >= threshold] - threshold
        scale = float(np.mean(excess)) if excess.size else 0.0
        if scale <= 0.0:
            scale = float(np.std(values) + float(evt_min_scale))
        thresholds[str(label)] = threshold
        scales[str(label)] = max(scale, float(evt_min_scale))
    return thresholds, scales


def _oldness_gate_envelope(
    features: np.ndarray,
    labels: np.ndarray,
    centroid_labels: np.ndarray,
    centroids: np.ndarray,
    *,
    oldness_quantile: float,
    oldness_slack: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    normalized = _normalize_rows(features)
    class_to_pos = {str(label): int(i) for i, label in enumerate(centroid_labels.tolist())}
    weights: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    global_centroid = _normalize_rows(normalized.mean(axis=0, keepdims=True))[0]
    for label in sorted(class_to_pos):
        idx = np.where(labels == label)[0]
        other_idx = np.where(labels != label)[0]
        if idx.size <= 0:
            continue
        pos_centroid = centroids[class_to_pos[label]]
        if other_idx.size:
            neg_centroid = _normalize_rows(normalized[other_idx].mean(axis=0, keepdims=True))[0]
        else:
            neg_centroid = global_centroid
        direction = pos_centroid - neg_centroid
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = pos_centroid
            norm = float(np.linalg.norm(direction))
        weight = direction / max(norm, 1e-8)
        pos_scores = normalized[idx] @ weight
        weights[label] = weight.astype(np.float32)
        thresholds[label] = float(np.quantile(pos_scores, float(oldness_quantile)) - float(oldness_slack))
    return weights, thresholds


def build_qknn_memory(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    old_labels: set[str],
    support_scenarios: Sequence[str] | None = None,
    radius_quantile: float = 0.95,
    margin_quantile: float = 0.05,
    score_quantile: float = 0.05,
    mahalanobis_quantile: float = 0.95,
    evt_tail_quantile: float = 0.80,
    oldness_quantile: float = 0.05,
    radius_slack: float = 0.02,
    margin_slack: float = 0.0,
    score_slack: float = 0.0,
    mahalanobis_slack: float = 0.0,
    evt_tail_slack: float = 0.0,
    oldness_slack: float = 0.0,
    mahalanobis_variance_floor: float = 1e-4,
    evt_min_scale: float = 1e-3,
    prototype_calibration_policy: str = "none",
    prototype_calibration_alpha: float = 0.0,
    prototype_calibration_top_m: int = 2,
) -> QknnMemory:
    normalized = _normalize_rows(features)
    scale = 127.0
    qfeatures = np.clip(np.rint(normalized * scale), -127, 127).astype(np.int8)
    labels_arr = np.asarray([canonical_tx_id(v) for v in labels], dtype=object)
    if support_scenarios is None:
        scenario_arr = np.asarray([""] * labels_arr.size, dtype=object)
    else:
        scenario_arr = np.asarray([str(v) for v in support_scenarios], dtype=object)
        if scenario_arr.shape[0] != labels_arr.shape[0]:
            raise ValueError("support_scenarios length must match labels")
    centroid_labels, centroids = _class_centroids(normalized, labels_arr)
    centroids, _ = _calibrate_seen_new_centroids(
        centroid_labels,
        centroids,
        old_labels=set(old_labels),
        policy=prototype_calibration_policy,
        alpha=prototype_calibration_alpha,
        top_m=prototype_calibration_top_m,
    )
    class_to_radius: dict[str, float] = {}
    for label in centroid_labels.tolist():
        label = str(label)
        class_features = normalized[labels_arr == label]
        prototype = centroids[np.where(centroid_labels == label)[0][0]]
        class_to_radius[label] = float(np.mean(1.0 - (class_features @ prototype))) if class_features.size else 0.0
    radii_by_support = np.asarray(
        [max(class_to_radius.get(str(label), 0.0), 1e-4) for label in labels_arr.tolist()],
        dtype=np.float64,
    )
    radius_thresholds, margin_threshold, score_threshold = _support_envelope(
        normalized,
        labels_arr,
        centroid_labels,
        centroids,
        radius_quantile=radius_quantile,
        margin_quantile=margin_quantile,
        score_quantile=score_quantile,
        radius_slack=radius_slack,
        margin_slack=margin_slack,
        score_slack=score_slack,
    )
    mahalanobis_inv_vars, mahalanobis_thresholds, mahalanobis_support_distances = _mahalanobis_envelope(
        normalized,
        labels_arr,
        centroid_labels,
        centroids,
        mahalanobis_quantile=mahalanobis_quantile,
        mahalanobis_slack=mahalanobis_slack,
        variance_floor=mahalanobis_variance_floor,
    )
    evt_thresholds, evt_scales = _evt_tail_envelope(
        mahalanobis_support_distances,
        evt_tail_quantile=evt_tail_quantile,
        evt_tail_slack=evt_tail_slack,
        evt_min_scale=evt_min_scale,
    )
    oldness_weights, oldness_thresholds = _oldness_gate_envelope(
        normalized,
        labels_arr,
        centroid_labels,
        centroids,
        oldness_quantile=oldness_quantile,
        oldness_slack=oldness_slack,
    )
    storage_bytes = int(qfeatures.nbytes + labels_arr.size * 4)
    return QknnMemory(
        qfeatures=qfeatures,
        labels=labels_arr,
        old_labels=set(old_labels),
        support_scenarios=scenario_arr,
        radii_by_support=radii_by_support,
        scale=scale,
        prototype_storage_bytes=storage_bytes,
        centroid_labels=centroid_labels,
        centroids=centroids,
        class_radius_thresholds=radius_thresholds,
        class_mahalanobis_inv_vars=mahalanobis_inv_vars,
        class_mahalanobis_thresholds=mahalanobis_thresholds,
        class_evt_thresholds=evt_thresholds,
        class_evt_scales=evt_scales,
        class_oldness_weights=oldness_weights,
        class_oldness_thresholds=oldness_thresholds,
        margin_threshold=margin_threshold,
        score_threshold=score_threshold,
    )


def qknn_scores(
    memory: QknnMemory,
    query_features: np.ndarray,
    *,
    top_k: int = 8,
    query_scenarios: Sequence[str] | None = None,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    exclude_support_indices: Sequence[int] | None = None,
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proto_blend = float(prototype_score_blend)
    if proto_blend < 0.0:
        raise ValueError("prototype_score_blend must be >= 0")
    maha_blend = float(mahalanobis_score_blend)
    if maha_blend < 0.0:
        raise ValueError("mahalanobis_score_blend must be >= 0")
    maha_temp = max(float(mahalanobis_score_temperature), 1e-6)
    query = _normalize_rows(query_features)
    support = _normalize_rows(memory.qfeatures.astype(np.float32) / float(memory.scale))
    centroid_scores = _centroid_scores(memory, query)
    mahalanobis_scores = _mahalanobis_known_scores(memory, query, temperature=maha_temp) if maha_blend > 0.0 else None
    class_to_pos = {str(label): int(pos) for pos, label in enumerate(memory.centroid_labels.tolist())}
    class_top_m = int(candidate_class_top_m)
    if class_top_m < 0:
        raise ValueError("candidate_class_top_m must be >= 0")
    query_scenarios_arr = None
    if scenario_aware:
        if query_scenarios is None:
            raise ValueError("query_scenarios is required when scenario_aware=True")
        query_scenarios_arr = np.asarray([str(v) for v in query_scenarios], dtype=object)
        if query_scenarios_arr.shape[0] != query.shape[0]:
            raise ValueError("query_scenarios length must match query_features")
    exclude_arr = None
    if exclude_support_indices is not None:
        exclude_arr = np.asarray(list(exclude_support_indices), dtype=int)
        if exclude_arr.shape[0] != query.shape[0]:
            raise ValueError("exclude_support_indices length must match query_features")
    scores = query @ support.T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(memory.radii_by_support, 1e-4), float(radius_norm))[None, :]
        scores = 1.0 - ((1.0 - scores) / denom)
    if float(old_bias) != 0.0:
        is_old = np.asarray([str(label) in memory.old_labels for label in memory.labels], dtype=np.float64)
        scores = scores + is_old[None, :] * float(old_bias)
    k = max(1, min(int(top_k), int(scores.shape[1])))
    out_labels: list[str] = []
    out_scores: list[float] = []
    out_margins: list[float] = []
    out_candidate_counts: list[int] = []
    out_support_neighbor_counts: list[int] = []
    out_support_densities: list[float] = []
    out_second_labels: list[str] = []
    out_second_scores: list[float] = []
    all_support_mask = np.ones(memory.labels.shape[0], dtype=bool)
    for row_i in range(scores.shape[0]):
        support_mask = all_support_mask.copy()
        if class_top_m > 0:
            m = max(1, min(class_top_m, int(memory.centroid_labels.shape[0])))
            candidate_class_pos = np.argpartition(centroid_scores[row_i], -m)[-m:]
            candidate_labels = set(str(memory.centroid_labels[pos]) for pos in candidate_class_pos.tolist())
            candidate_mask = np.asarray([str(label) in candidate_labels for label in memory.labels], dtype=bool)
            if int(np.sum(candidate_mask)) >= k:
                support_mask &= candidate_mask
        if scenario_aware and query_scenarios_arr is not None:
            candidate_mask = memory.support_scenarios.astype(str) == str(query_scenarios_arr[row_i])
            if int(np.sum(candidate_mask)) >= k and len(set(memory.labels[candidate_mask].tolist())) >= 2:
                support_mask = candidate_mask.copy()
        if exclude_arr is not None and 0 <= int(exclude_arr[row_i]) < support_mask.shape[0]:
            support_mask[int(exclude_arr[row_i])] = False
        support_positions = np.where(support_mask)[0]
        if support_positions.size == 0:
            raise ValueError("qknn_scores has no support rows after exclusions")
        row_scores = scores[row_i, support_positions]
        row_k = max(1, min(k, int(row_scores.shape[0])))
        idx = support_positions[np.argpartition(row_scores, -row_k)[-row_k:]]
        per_label: dict[str, float] = defaultdict(float)
        per_label_count: dict[str, int] = defaultdict(int)
        for j in idx.tolist():
            score = float(scores[row_i, j])
            per_label[str(memory.labels[j])] += max(0.0, score)
            per_label_count[str(memory.labels[j])] += 1
        if proto_blend > 0.0:
            candidate_labels = sorted({str(label) for label in memory.labels[support_mask].tolist()})
            for label in candidate_labels:
                pos = class_to_pos.get(label)
                if pos is not None:
                    per_label[label] += proto_blend * max(0.0, float(centroid_scores[row_i, pos]))
                    per_label_count.setdefault(label, 0)
        if maha_blend > 0.0 and mahalanobis_scores is not None:
            candidate_labels = sorted({str(label) for label in memory.labels[support_mask].tolist()})
            for label in candidate_labels:
                pos = class_to_pos.get(label)
                if pos is not None:
                    per_label[label] += maha_blend * max(0.0, float(mahalanobis_scores[row_i, pos]))
                    per_label_count.setdefault(label, 0)
        ranked = sorted(per_label.items(), key=lambda item: (item[1], item[0]), reverse=True)
        best_label, best_score = ranked[0]
        second_label = ranked[1][0] if len(ranked) > 1 else ""
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        best_count = int(per_label_count.get(best_label, 0))
        score_denom = max(float(row_k) + proto_blend + maha_blend, 1.0)
        out_labels.append(best_label)
        out_scores.append(float(best_score / score_denom))
        out_margins.append(float((best_score - second) / score_denom))
        out_candidate_counts.append(len(set(memory.labels[support_mask].tolist())))
        out_support_neighbor_counts.append(best_count)
        out_support_densities.append(float(best_count / max(row_k, 1)))
        out_second_labels.append(str(second_label))
        out_second_scores.append(float(second / score_denom))
    return (
        np.asarray(out_labels, dtype=object),
        np.asarray(out_scores, dtype=np.float64),
        np.asarray(out_margins, dtype=np.float64),
        np.asarray(out_candidate_counts, dtype=np.int64),
        np.asarray(out_support_neighbor_counts, dtype=np.int64),
        np.asarray(out_support_densities, dtype=np.float64),
        np.asarray(out_second_labels, dtype=object),
        np.asarray(out_second_scores, dtype=np.float64),
    )


def _qknn_label_score_matrix(
    memory: QknnMemory,
    query_features: np.ndarray,
    *,
    top_k: int = 8,
    query_scenarios: Sequence[str] | None = None,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    exclude_support_indices: Sequence[int] | None = None,
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    proto_blend = float(prototype_score_blend)
    if proto_blend < 0.0:
        raise ValueError("prototype_score_blend must be >= 0")
    maha_blend = float(mahalanobis_score_blend)
    if maha_blend < 0.0:
        raise ValueError("mahalanobis_score_blend must be >= 0")
    maha_temp = max(float(mahalanobis_score_temperature), 1e-6)
    query = _normalize_rows(query_features)
    support = _normalize_rows(memory.qfeatures.astype(np.float32) / float(memory.scale))
    centroid_scores = _centroid_scores(memory, query)
    mahalanobis_scores = _mahalanobis_known_scores(memory, query, temperature=maha_temp) if maha_blend > 0.0 else None
    class_to_pos = {str(label): int(pos) for pos, label in enumerate(memory.centroid_labels.tolist())}
    labels = np.asarray([str(label) for label in memory.centroid_labels.tolist()], dtype=object)
    label_to_col = {str(label): int(pos) for pos, label in enumerate(labels.tolist())}
    class_top_m = int(candidate_class_top_m)
    if class_top_m < 0:
        raise ValueError("candidate_class_top_m must be >= 0")
    query_scenarios_arr = None
    if scenario_aware:
        if query_scenarios is None:
            raise ValueError("query_scenarios is required when scenario_aware=True")
        query_scenarios_arr = np.asarray([str(v) for v in query_scenarios], dtype=object)
        if query_scenarios_arr.shape[0] != query.shape[0]:
            raise ValueError("query_scenarios length must match query_features")
    exclude_arr = None
    if exclude_support_indices is not None:
        exclude_arr = np.asarray(list(exclude_support_indices), dtype=int)
        if exclude_arr.shape[0] != query.shape[0]:
            raise ValueError("exclude_support_indices length must match query_features")
    scores = query @ support.T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(memory.radii_by_support, 1e-4), float(radius_norm))[None, :]
        scores = 1.0 - ((1.0 - scores) / denom)
    if float(old_bias) != 0.0:
        is_old = np.asarray([str(label) in memory.old_labels for label in memory.labels], dtype=np.float64)
        scores = scores + is_old[None, :] * float(old_bias)
    k = max(1, min(int(top_k), int(scores.shape[1])))
    out = np.zeros((query.shape[0], labels.shape[0]), dtype=np.float64)
    all_support_mask = np.ones(memory.labels.shape[0], dtype=bool)
    for row_i in range(scores.shape[0]):
        support_mask = all_support_mask.copy()
        if class_top_m > 0:
            m = max(1, min(class_top_m, int(memory.centroid_labels.shape[0])))
            candidate_class_pos = np.argpartition(centroid_scores[row_i], -m)[-m:]
            candidate_labels = set(str(memory.centroid_labels[pos]) for pos in candidate_class_pos.tolist())
            candidate_mask = np.asarray([str(label) in candidate_labels for label in memory.labels], dtype=bool)
            if int(np.sum(candidate_mask)) >= k:
                support_mask &= candidate_mask
        if scenario_aware and query_scenarios_arr is not None:
            candidate_mask = memory.support_scenarios.astype(str) == str(query_scenarios_arr[row_i])
            if int(np.sum(candidate_mask)) >= k and len(set(memory.labels[candidate_mask].tolist())) >= 2:
                support_mask = candidate_mask.copy()
        if exclude_arr is not None and 0 <= int(exclude_arr[row_i]) < support_mask.shape[0]:
            support_mask[int(exclude_arr[row_i])] = False
        support_positions = np.where(support_mask)[0]
        if support_positions.size == 0:
            raise ValueError("qknn_scores has no support rows after exclusions")
        row_scores = scores[row_i, support_positions]
        row_k = max(1, min(k, int(row_scores.shape[0])))
        idx = support_positions[np.argpartition(row_scores, -row_k)[-row_k:]]
        per_label: dict[str, float] = defaultdict(float)
        for j in idx.tolist():
            per_label[str(memory.labels[j])] += max(0.0, float(scores[row_i, j]))
        candidate_labels = sorted({str(label) for label in memory.labels[support_mask].tolist()})
        if proto_blend > 0.0:
            for label in candidate_labels:
                pos = class_to_pos.get(label)
                if pos is not None:
                    per_label[label] += proto_blend * max(0.0, float(centroid_scores[row_i, pos]))
        if maha_blend > 0.0 and mahalanobis_scores is not None:
            for label in candidate_labels:
                pos = class_to_pos.get(label)
                if pos is not None:
                    per_label[label] += maha_blend * max(0.0, float(mahalanobis_scores[row_i, pos]))
        score_denom = max(float(row_k) + proto_blend + maha_blend, 1.0)
        for label, value in per_label.items():
            col = label_to_col.get(str(label))
            if col is not None:
                out[row_i, col] = float(value / score_denom)
    return labels, out


def _split_support_query(
    payload: Mapping[str, Any],
    *,
    role: str,
    tx_id: str,
    rx_id: str,
    k_shot: int,
    query_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    idx = [
        int(i)
        for i in np.where((roles == role) & (tx_ids == tx_id) & (rx_ids == rx_id))[0].tolist()
    ]
    ordered = sorted(
        idx,
        key=lambda i: _stable_score(
            (role, tx_id, rx_id, payload["day_ids"][i], payload["sig_ids"][i], payload["sat_scenarios"][i]),
            seed,
        ),
    )
    support = ordered[: min(int(k_shot), len(ordered))]
    query = ordered[len(support) : len(support) + int(query_per_class)]
    return support, query


def _nearest_to_centroid(features: np.ndarray, indices: Sequence[int], k: int) -> list[int]:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0 or int(k) <= 0:
        return []
    subset = _normalize_rows(features[idx])
    centroid = _normalize_rows(subset.mean(axis=0, keepdims=True))[0]
    return idx[np.argsort(-(subset @ centroid))[: int(k)]].astype(int).tolist()


def _scenario_diverse_support(
    payload: Mapping[str, Any],
    features: np.ndarray,
    indices: Sequence[int],
    k: int,
) -> list[int]:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return []
    scenarios = np.asarray(payload["sat_scenarios"]).astype(str)
    selected: list[int] = []
    for scenario in sorted({str(scenarios[i]) for i in idx.tolist()}):
        scenario_idx = idx[scenarios[idx] == scenario]
        selected.extend(_nearest_to_centroid(features, scenario_idx, 1))
        if len(selected) >= int(k):
            return selected[: int(k)]
    remaining = [int(i) for i in idx.tolist() if int(i) not in set(selected)]
    selected.extend(_nearest_to_centroid(features, remaining, int(k) - len(selected)))
    return selected[: int(k)]


def _select_support_indices(
    payload: Mapping[str, Any],
    features: np.ndarray,
    ordered: Sequence[int],
    *,
    k_shot: int,
    policy: str,
) -> list[int]:
    ordered = [int(i) for i in ordered]
    policy = str(policy or "stable_first").strip().lower()
    if policy == "stable_first":
        return ordered[: min(int(k_shot), len(ordered))]
    if policy == "centroid":
        return _nearest_to_centroid(features, ordered, int(k_shot))
    if policy == "scenario_diverse":
        return _scenario_diverse_support(payload, features, ordered, int(k_shot))
    raise ValueError("support_selection_policy must be stable_first, centroid, or scenario_diverse")


def _split_support_query_selected(
    payload: Mapping[str, Any],
    *,
    features: np.ndarray,
    role: str,
    tx_id: str,
    rx_id: str,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
) -> tuple[list[int], list[int]]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    idx = [
        int(i)
        for i in np.where((roles == role) & (tx_ids == tx_id) & (rx_ids == rx_id))[0].tolist()
    ]
    ordered = sorted(
        idx,
        key=lambda i: _stable_score(
            (role, tx_id, rx_id, payload["day_ids"][i], payload["sig_ids"][i], payload["sat_scenarios"][i]),
            seed,
        ),
    )
    support_candidates = ordered[: min(int(k_shot), len(ordered))]
    support = _select_support_indices(
        payload,
        features,
        support_candidates,
        k_shot=int(k_shot),
        policy=support_selection_policy,
    )
    support_set = set(support)
    query = [int(i) for i in ordered if int(i) not in support_set][: int(query_per_class)]
    return support, query


def _threshold_from_calibration(
    memory: QknnMemory,
    support_features: np.ndarray,
    proxy_features: np.ndarray | None,
    *,
    top_k: int,
    support_quantile: float,
    proxy_quantile: float,
    support_scenarios: Sequence[str] | None = None,
    proxy_scenarios: Sequence[str] | None = None,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    support_calibration_mode: str = "self",
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
) -> tuple[float, str]:
    calibration_mode = str(support_calibration_mode or "self").strip().lower()
    if calibration_mode not in {"self", "leave_one_out", "loo"}:
        raise ValueError("support_calibration_mode must be self or leave_one_out")
    exclude = range(int(support_features.shape[0])) if calibration_mode in {"leave_one_out", "loo"} else None
    _, support_scores, _, _, _, _, _, _ = qknn_scores(
        memory,
        support_features,
        top_k=top_k,
        query_scenarios=support_scenarios,
        scenario_aware=scenario_aware,
        radius_norm=radius_norm,
        old_bias=old_bias,
        candidate_class_top_m=candidate_class_top_m,
        exclude_support_indices=exclude,
        prototype_score_blend=prototype_score_blend,
        mahalanobis_score_blend=mahalanobis_score_blend,
        mahalanobis_score_temperature=mahalanobis_score_temperature,
    )
    threshold = float(np.quantile(support_scores, float(support_quantile))) if support_scores.size else 0.0
    scope = "support_known_only"
    if proxy_features is not None and int(proxy_features.shape[0]) > 0:
        _, proxy_scores, _, _, _, _, _, _ = qknn_scores(
            memory,
            proxy_features,
            top_k=top_k,
            query_scenarios=proxy_scenarios,
            scenario_aware=scenario_aware and proxy_scenarios is not None,
            radius_norm=radius_norm,
            old_bias=old_bias,
            candidate_class_top_m=candidate_class_top_m,
            prototype_score_blend=prototype_score_blend,
            mahalanobis_score_blend=mahalanobis_score_blend,
            mahalanobis_score_temperature=mahalanobis_score_temperature,
        )
        if proxy_scores.size:
            threshold = max(threshold, float(np.quantile(proxy_scores, float(proxy_quantile))))
            scope = "source_only"
    return threshold, scope


def _label_thresholds_from_calibration(
    memory: QknnMemory,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    proxy_features: np.ndarray | None,
    *,
    top_k: int,
    support_quantile: float,
    proxy_quantile: float,
    support_scenarios: Sequence[str] | None = None,
    proxy_scenarios: Sequence[str] | None = None,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    support_calibration_mode: str = "self",
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
    min_support: int = 1,
) -> dict[str, float]:
    labels_arr = np.asarray([canonical_tx_id(label) for label in support_labels], dtype=object)
    calibration_mode = str(support_calibration_mode or "self").strip().lower()
    if calibration_mode not in {"self", "leave_one_out", "loo"}:
        raise ValueError("support_calibration_mode must be self or leave_one_out")
    exclude = range(int(support_features.shape[0])) if calibration_mode in {"leave_one_out", "loo"} else None
    score_labels, support_score_matrix = _qknn_label_score_matrix(
        memory,
        support_features,
        top_k=top_k,
        query_scenarios=support_scenarios,
        scenario_aware=scenario_aware,
        radius_norm=radius_norm,
        old_bias=old_bias,
        candidate_class_top_m=candidate_class_top_m,
        exclude_support_indices=exclude,
        prototype_score_blend=prototype_score_blend,
        mahalanobis_score_blend=mahalanobis_score_blend,
        mahalanobis_score_temperature=mahalanobis_score_temperature,
    )
    label_to_col = {str(label): int(pos) for pos, label in enumerate(score_labels.tolist())}
    thresholds: dict[str, float] = {}
    for label in sorted({str(label) for label in labels_arr.tolist()}):
        col = label_to_col.get(label)
        values = support_score_matrix[labels_arr == label, col] if col is not None else np.asarray([], dtype=np.float64)
        if values.size >= max(1, int(min_support)):
            thresholds[label] = float(np.quantile(values, float(support_quantile)))
    if proxy_features is not None and int(proxy_features.shape[0]) > 0:
        proxy_score_labels, proxy_score_matrix = _qknn_label_score_matrix(
            memory,
            proxy_features,
            top_k=top_k,
            query_scenarios=proxy_scenarios,
            scenario_aware=scenario_aware and proxy_scenarios is not None,
            radius_norm=radius_norm,
            old_bias=old_bias,
            candidate_class_top_m=candidate_class_top_m,
            prototype_score_blend=prototype_score_blend,
            mahalanobis_score_blend=mahalanobis_score_blend,
            mahalanobis_score_temperature=mahalanobis_score_temperature,
        )
        proxy_label_to_col = {str(label): int(pos) for pos, label in enumerate(proxy_score_labels.tolist())}
        proxy_pred, proxy_scores, _, _, _, _, _, _ = qknn_scores(
            memory,
            proxy_features,
            top_k=top_k,
            query_scenarios=proxy_scenarios,
            scenario_aware=scenario_aware and proxy_scenarios is not None,
            radius_norm=radius_norm,
            old_bias=old_bias,
            candidate_class_top_m=candidate_class_top_m,
            prototype_score_blend=prototype_score_blend,
            mahalanobis_score_blend=mahalanobis_score_blend,
            mahalanobis_score_temperature=mahalanobis_score_temperature,
        )
        for label in sorted({str(label) for label in proxy_pred.tolist()}):
            col = proxy_label_to_col.get(label)
            values = proxy_score_matrix[proxy_pred == label, col] if col is not None else proxy_scores[proxy_pred == label]
            if values.size:
                thresholds[label] = max(
                    thresholds.get(label, 0.0),
                    float(np.quantile(values, float(proxy_quantile))),
                )
    return thresholds


def _label_score_samples_from_calibration(
    memory: QknnMemory,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    top_k: int,
    support_scenarios: Sequence[str] | None = None,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    support_calibration_mode: str = "self",
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
    min_support: int = 1,
) -> dict[str, list[float]]:
    labels_arr = np.asarray([canonical_tx_id(label) for label in support_labels], dtype=object)
    calibration_mode = str(support_calibration_mode or "self").strip().lower()
    if calibration_mode not in {"self", "leave_one_out", "loo"}:
        raise ValueError("support_calibration_mode must be self or leave_one_out")
    exclude = range(int(support_features.shape[0])) if calibration_mode in {"leave_one_out", "loo"} else None
    score_labels, support_score_matrix = _qknn_label_score_matrix(
        memory,
        support_features,
        top_k=top_k,
        query_scenarios=support_scenarios,
        scenario_aware=scenario_aware,
        radius_norm=radius_norm,
        old_bias=old_bias,
        candidate_class_top_m=candidate_class_top_m,
        exclude_support_indices=exclude,
        prototype_score_blend=prototype_score_blend,
        mahalanobis_score_blend=mahalanobis_score_blend,
        mahalanobis_score_temperature=mahalanobis_score_temperature,
    )
    label_to_col = {str(label): int(pos) for pos, label in enumerate(score_labels.tolist())}
    out: dict[str, list[float]] = {}
    for label in sorted({str(label) for label in labels_arr.tolist()}):
        col = label_to_col.get(label)
        values = support_score_matrix[labels_arr == label, col] if col is not None else np.asarray([], dtype=np.float64)
        if values.size >= max(1, int(min_support)):
            out[label] = [float(value) for value in values.tolist()]
    return out


def _conformal_pvalue(score: float, calibration_scores: Sequence[float] | None) -> float:
    values = [float(value) for value in (calibration_scores or []) if np.isfinite(float(value))]
    if not values:
        return 0.0
    leq = sum(value <= float(score) for value in values)
    return float(leq + 1) / float(len(values) + 1)


def _receiver_class_reliability_from_support(
    calibration_scores: Sequence[float] | None,
    *,
    threshold: float,
    min_support: int,
) -> float:
    values = [float(value) for value in (calibration_scores or []) if np.isfinite(float(value))]
    if not values:
        return 1.0
    threshold_value = float(threshold) if np.isfinite(float(threshold)) else 0.0
    support_scale = min(1.0, len(values) / max(float(min_support), 1.0))
    pass_rate = sum(value >= threshold_value for value in values) / float(len(values))
    mean_score = sum(values) / float(len(values))
    if threshold_value > 1e-12:
        margin_scale = max(0.0, min(1.0, mean_score / threshold_value))
    else:
        margin_scale = max(0.0, min(1.0, mean_score))
    reliability = (0.25 + 0.75 * support_scale) * (0.35 + 0.65 * pass_rate) * (0.50 + 0.50 * margin_scale)
    return float(max(0.05, min(1.0, reliability)))


def _unknown_risk(scores: np.ndarray, threshold: float, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = (np.asarray(scores, dtype=np.float64) - float(threshold)) / temp
    known_prob = 1.0 / (1.0 + np.exp(-z))
    return np.clip(1.0 - known_prob, 0.0, 1.0)


def _virtual_unknown_boundary_risk(
    query_features: np.ndarray,
    known_scores: np.ndarray,
    virtual_features: np.ndarray | None,
    *,
    temperature: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(known_scores, dtype=np.float64)
    if virtual_features is None or int(np.asarray(virtual_features).shape[0]) == 0:
        return np.zeros_like(scores, dtype=np.float64), np.zeros_like(scores, dtype=np.float64)
    query = _normalize_rows(query_features)
    virtual = _normalize_rows(np.asarray(virtual_features, dtype=np.float32))
    virtual_scores = np.max(query @ virtual.T, axis=1).astype(np.float64)
    z = np.clip((virtual_scores - scores + float(margin)) / max(float(temperature), 1e-6), -60.0, 60.0)
    risk = 1.0 / (1.0 + np.exp(-z))
    return np.clip(risk, 0.0, 1.0), np.asarray(virtual_scores, dtype=np.float64)


def _class_shell_boundary_risk(
    memory: QknnMemory,
    query_features: np.ndarray,
    predicted_labels: Sequence[str],
    *,
    radius_scale: float,
    temperature: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    query = _normalize_rows(query_features)
    class_to_pos = {str(label): int(i) for i, label in enumerate(memory.centroid_labels.tolist())}
    risks: list[float] = []
    distances: list[float] = []
    scale = max(float(radius_scale), 1e-6)
    temp = max(float(temperature), 1e-6)
    for row_i, raw_label in enumerate(predicted_labels):
        label = str(raw_label)
        pos = class_to_pos.get(label)
        if pos is None:
            risks.append(1.0)
            distances.append(1.0)
            continue
        distance = float(1.0 - float(query[row_i] @ memory.centroids[pos]))
        threshold = max(0.0, float(memory.class_radius_thresholds.get(label, 1.0))) * scale
        z = np.clip((distance - threshold + float(margin)) / temp, -60.0, 60.0)
        risks.append(float(1.0 / (1.0 + np.exp(-z))))
        distances.append(distance)
    return np.clip(np.asarray(risks, dtype=np.float64), 0.0, 1.0), np.asarray(distances, dtype=np.float64)


def _combined_unknown_risk(
    memory: QknnMemory,
    query_features: np.ndarray,
    predicted_labels: Sequence[str],
    known_scores: np.ndarray,
    known_margins: np.ndarray,
    score_threshold: float,
    *,
    temperature: float,
    gate_mode: str,
    radius_temperature: float,
    margin_temperature: float,
    mahalanobis_temperature: float,
    evt_temperature: float,
    oldness_temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = str(gate_mode or "score").strip().lower()
    score_risk = _unknown_risk(known_scores, score_threshold, temperature=temperature)
    if mode == "score":
        zeros = np.zeros_like(score_risk)
        return score_risk, score_risk, zeros, zeros, zeros, zeros, zeros, zeros, zeros
    centroid_scores = _centroid_scores(memory, query_features)
    class_to_pos = {str(label): int(i) for i, label in enumerate(memory.centroid_labels.tolist())}
    radius_risks = []
    radius_values = []
    radius_z_values = []
    for row_i, label in enumerate(predicted_labels):
        label = str(label)
        pos = class_to_pos.get(label)
        if pos is None:
            radius_values.append(1.0)
            radius_z_values.append(1.0)
            radius_risks.append(1.0)
            continue
        radius = float(1.0 - centroid_scores[row_i, pos])
        threshold = float(memory.class_radius_thresholds.get(label, 1.0))
        z = (radius - threshold) / max(float(radius_temperature), 1e-6)
        radius_values.append(radius)
        radius_z_values.append(float(z))
        radius_risks.append(float(1.0 / (1.0 + np.exp(-z))))
    radius_risk = np.asarray(radius_risks, dtype=np.float64)
    margin_t = float(memory.margin_threshold)
    margin_z = (margin_t - np.asarray(known_margins, dtype=np.float64)) / max(float(margin_temperature), 1e-6)
    margin_risk = 1.0 / (1.0 + np.exp(-margin_z))
    mahalanobis_risks = []
    mahalanobis_distances = []
    evt_risks = []
    oldness_risks = []
    normalized_query = _normalize_rows(query_features)
    for row_i, label in enumerate(predicted_labels):
        label = str(label)
        pos = class_to_pos.get(label)
        inv_var = memory.class_mahalanobis_inv_vars.get(label)
        if pos is None or inv_var is None:
            mahalanobis_distances.append(1.0)
            mahalanobis_risks.append(1.0)
            evt_risks.append(1.0)
            oldness_risks.append(1.0)
            continue
        centroid = memory.centroids[pos]
        distance = float(np.sqrt(np.mean(((normalized_query[row_i] - centroid) ** 2) * inv_var)))
        threshold = float(memory.class_mahalanobis_thresholds.get(label, 1.0))
        z = (distance - threshold) / max(float(mahalanobis_temperature), 1e-6)
        mahalanobis_distances.append(distance)
        mahalanobis_risks.append(float(1.0 / (1.0 + np.exp(-z))))
        evt_threshold = float(memory.class_evt_thresholds.get(label, threshold))
        evt_scale = max(float(memory.class_evt_scales.get(label, evt_temperature)), float(evt_temperature), 1e-12)
        excess = max(0.0, distance - evt_threshold)
        evt_risks.append(float(1.0 - np.exp(-excess / evt_scale)))
        oldness_weight = memory.class_oldness_weights.get(label)
        if oldness_weight is None:
            oldness_risks.append(1.0)
        else:
            oldness_score = float(normalized_query[row_i] @ oldness_weight)
            oldness_threshold = float(memory.class_oldness_thresholds.get(label, 0.0))
            z = (oldness_threshold - oldness_score) / max(float(oldness_temperature), 1e-6)
            oldness_risks.append(float(1.0 / (1.0 + np.exp(-z))))
    mahalanobis_risk = np.asarray(mahalanobis_risks, dtype=np.float64)
    evt_risk = np.asarray(evt_risks, dtype=np.float64)
    oldness_risk = np.asarray(oldness_risks, dtype=np.float64)
    if mode == "support_envelope":
        risk = np.maximum.reduce([score_risk, radius_risk, margin_risk])
    elif mode == "support_envelope_mahalanobis":
        risk = np.maximum.reduce([score_risk, radius_risk, margin_risk, mahalanobis_risk])
    elif mode == "support_envelope_evt":
        risk = np.maximum.reduce([score_risk, radius_risk, margin_risk, evt_risk])
    elif mode == "support_envelope_oldness":
        risk = np.maximum.reduce([score_risk, radius_risk, margin_risk, oldness_risk])
    elif mode == "support_envelope_full":
        risk = np.maximum.reduce(
            [
                score_risk,
                radius_risk,
                margin_risk,
                mahalanobis_risk,
                evt_risk,
                oldness_risk,
            ]
        )
    elif mode == "support_envelope_consensus":
        stacked = np.vstack(
            [
                score_risk,
                radius_risk,
                margin_risk,
                mahalanobis_risk,
                evt_risk,
                oldness_risk,
            ]
        )
        risk = np.mean(np.sort(stacked, axis=0)[-3:, :], axis=0)
    elif mode == "radius":
        risk = np.maximum(score_risk, radius_risk)
    elif mode == "margin":
        risk = np.maximum(score_risk, margin_risk)
    elif mode == "mahalanobis":
        risk = np.maximum(score_risk, mahalanobis_risk)
    elif mode == "evt":
        risk = np.maximum(score_risk, evt_risk)
    elif mode == "oldness":
        risk = np.maximum(score_risk, oldness_risk)
    else:
        raise ValueError(
            "unknown_gate_mode must be score, radius, margin, mahalanobis, evt, oldness, support_envelope, "
            "support_envelope_mahalanobis, support_envelope_evt, support_envelope_oldness, "
            "support_envelope_full, or support_envelope_consensus"
        )
    return (
        np.clip(risk, 0.0, 1.0),
        np.asarray(score_risk, dtype=np.float64),
        radius_risk,
        margin_risk,
        mahalanobis_risk,
        evt_risk,
        oldness_risk,
        np.asarray(radius_values, dtype=np.float64),
        np.asarray(radius_z_values, dtype=np.float64),
    )


def _active_risk_components_for_gate_mode(gate_mode: str) -> list[str]:
    mode = str(gate_mode or "score").strip().lower()
    mapping = {
        "score": ["score"],
        "radius": ["score", "radius"],
        "margin": ["score", "margin"],
        "mahalanobis": ["score", "mahalanobis"],
        "evt": ["score", "evt"],
        "oldness": ["score", "oldness"],
        "support_envelope": ["score", "radius", "margin"],
        "support_envelope_mahalanobis": ["score", "radius", "margin", "mahalanobis"],
        "support_envelope_evt": ["score", "radius", "margin", "evt"],
        "support_envelope_oldness": ["score", "radius", "margin", "oldness"],
        "support_envelope_full": ["score", "radius", "margin", "mahalanobis", "evt", "oldness"],
        "support_envelope_consensus": ["score", "radius", "margin", "mahalanobis", "evt", "oldness"],
    }
    if mode not in mapping:
        raise ValueError(f"unknown unknown_gate_mode {gate_mode!r}")
    return list(mapping[mode])


def _combine_score_threshold(qknn_threshold: float, centroid_threshold: float, mode: str) -> float:
    mode = str(mode or "max").strip().lower()
    if mode == "max":
        return max(float(qknn_threshold), float(centroid_threshold))
    if mode == "qknn_only":
        return float(qknn_threshold)
    if mode == "centroid_only":
        return float(centroid_threshold)
    if mode == "min":
        return min(float(qknn_threshold), float(centroid_threshold))
    if mode == "mean":
        return 0.5 * (float(qknn_threshold) + float(centroid_threshold))
    raise ValueError("score_threshold_combine must be max, qknn_only, centroid_only, min, or mean")


def _support_quality_class_verifier(
    memory: QknnMemory,
    query_feature: np.ndarray,
    *,
    rx: str,
    scenario: str,
    qknn_k: int,
    scenario_aware: bool,
    radius_norm: float,
    old_bias: float,
    candidate_class_top_m: int,
    class_verifier_top_m: int,
    prototype_blend: float,
    mahalanobis_blend: float,
    mahalanobis_score_temp: float,
    receiver_threshold: float,
    receiver_class_thresholds: Mapping[str, Mapping[str, float]],
    receiver_class_conformal_scores: Mapping[str, Mapping[str, Sequence[float]]],
    receiver_class_reliabilities: Mapping[str, Mapping[str, float]],
    score_threshold_combine: str,
    class_score_threshold_enabled: bool,
    class_conformal_enabled: bool,
    unknown_gate_mode: str,
    risk_temperature: float,
    radius_temperature: float,
    margin_temperature: float,
    mahalanobis_temperature: float,
    evt_temperature: float,
    oldness_temperature: float,
    class_shell_unknown_risk_enabled: bool,
    class_shell_radius_scale: float,
    class_shell_risk_temperature: float,
    class_shell_risk_margin: float,
    pvalue_weight: float,
    reliability_weight: float,
    risk_weight: float,
) -> dict[str, Any]:
    score_labels, score_matrix = _qknn_label_score_matrix(
        memory,
        query_feature,
        top_k=qknn_k,
        query_scenarios=[scenario],
        scenario_aware=bool(scenario_aware),
        radius_norm=float(radius_norm),
        old_bias=float(old_bias),
        candidate_class_top_m=int(candidate_class_top_m),
        prototype_score_blend=float(prototype_blend),
        mahalanobis_score_blend=float(mahalanobis_blend),
        mahalanobis_score_temperature=float(mahalanobis_score_temp),
    )
    label_scores = [
        (str(label), float(score_value))
        for label, score_value in zip(score_labels.tolist(), score_matrix[0].tolist())
        if np.isfinite(float(score_value))
    ]
    label_scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    top_m = int(class_verifier_top_m)
    if top_m > 0:
        label_scores = label_scores[:top_m]
    if not label_scores:
        raise RuntimeError("support_quality class verifier has no candidate labels")

    receiver_conformal = receiver_class_conformal_scores.get(rx, {})
    receiver_reliabilities = receiver_class_reliabilities.get(rx, {})
    verified: list[dict[str, float | str | int]] = []
    for label_name, label_score in label_scores:
        label_second_score = max(
            (score_value for other_label, score_value in label_scores if other_label != label_name),
            default=0.0,
        )
        label_margin = float(label_score - label_second_score)
        label_class_threshold_value = receiver_class_thresholds.get(rx, {}).get(label_name)
        if bool(class_score_threshold_enabled) and label_class_threshold_value is not None:
            label_receiver_threshold = float(label_class_threshold_value)
            label_threshold_source = "class"
        elif bool(class_score_threshold_enabled):
            label_receiver_threshold = float(receiver_threshold)
            label_threshold_source = "receiver_fallback"
        else:
            label_receiver_threshold = float(receiver_threshold)
            label_threshold_source = "receiver_global"
        label_effective_threshold = _combine_score_threshold(
            label_receiver_threshold,
            memory.score_threshold,
            str(score_threshold_combine),
        )
        (
            label_risk,
            _label_score_risk,
            _label_radius_risk,
            _label_margin_risk,
            _label_mahalanobis_risk,
            _label_evt_risk,
            _label_oldness_risk,
            _label_class_radius,
            _label_class_radius_z,
        ) = _combined_unknown_risk(
            memory,
            query_feature,
            [label_name],
            np.asarray([label_score], dtype=np.float64),
            np.asarray([label_margin], dtype=np.float64),
            label_effective_threshold,
            temperature=float(risk_temperature),
            gate_mode=str(unknown_gate_mode),
            radius_temperature=float(radius_temperature),
            margin_temperature=float(margin_temperature),
            mahalanobis_temperature=float(mahalanobis_temperature),
            evt_temperature=float(evt_temperature),
            oldness_temperature=float(oldness_temperature),
        )
        label_shell_risk, _label_shell_distance = _class_shell_boundary_risk(
            memory,
            query_feature,
            [label_name],
            radius_scale=float(class_shell_radius_scale),
            temperature=float(class_shell_risk_temperature),
            margin=float(class_shell_risk_margin),
        )
        pvalue = (
            _conformal_pvalue(label_score, receiver_conformal.get(label_name))
            if bool(class_conformal_enabled)
            else 1.0
        )
        reliability = float(receiver_reliabilities.get(label_name, 1.0))
        combined_risk = max(
            float(label_risk[0]),
            float(label_shell_risk[0]) if bool(class_shell_unknown_risk_enabled) else 0.0,
        )
        p_factor = (0.50 + 0.50 * max(0.0, min(1.0, float(pvalue)))) ** max(float(pvalue_weight), 0.0)
        r_factor = (0.50 + 0.50 * max(0.0, min(1.0, reliability))) ** max(float(reliability_weight), 0.0)
        risk_factor = max(0.0, 1.0 - combined_risk) ** max(float(risk_weight), 0.0)
        verified_score = float(max(0.0, label_score) * p_factor * r_factor * risk_factor)
        verified.append(
            {
                "label": label_name,
                "raw_score": float(label_score),
                "raw_margin": float(label_margin),
                "verified_score": verified_score,
                "pvalue": float(pvalue),
                "receiver_class_reliability": reliability,
                "unknown_risk": float(label_risk[0]),
                "class_shell_risk": float(label_shell_risk[0]) if bool(class_shell_unknown_risk_enabled) else 0.0,
                "combined_risk": float(combined_risk),
                "effective_score_threshold": float(label_effective_threshold),
                "score_threshold_source": label_threshold_source,
                "support_count": int(len(receiver_conformal.get(label_name, []))),
            }
        )
    verified.sort(
        key=lambda item: (
            float(item["verified_score"]),
            float(item["raw_score"]),
            str(item["label"]),
        ),
        reverse=True,
    )
    best = verified[0]
    second: Mapping[str, float | str] = (
        verified[1] if len(verified) > 1 else {"label": "", "raw_score": 0.0, "verified_score": 0.0}
    )
    return {
        "top1_label": str(best["label"]),
        "top1_raw_score": float(best["raw_score"]),
        "top1_raw_margin": float(best["raw_margin"]),
        "top1_verified_score": float(best["verified_score"]),
        "top1_pvalue": float(best["pvalue"]),
        "top1_receiver_class_reliability": float(best["receiver_class_reliability"]),
        "top1_unknown_risk": float(best["unknown_risk"]),
        "top1_class_shell_risk": float(best["class_shell_risk"]),
        "top1_combined_risk": float(best["combined_risk"]),
        "top1_support_count": int(best["support_count"]),
        "top1_effective_score_threshold": float(best["effective_score_threshold"]),
        "top1_score_threshold_source": str(best["score_threshold_source"]),
        "second_label": str(second["label"]),
        "second_raw_score": float(second["raw_score"]),
        "second_verified_score": float(second["verified_score"]),
        "candidate_count": int(len(verified)),
        "all": verified,
    }


def _scenario_of(payload: Mapping[str, Any], idx: int) -> str:
    scenario = str(payload["sat_scenarios"][idx])
    return scenario if scenario else str(payload["channel_views"][idx])


def _event_key(payload: Mapping[str, Any], idx: int, role_name: str, label: str) -> str:
    return "|".join(
        [
            str(role_name),
            str(label),
            str(payload["day_ids"][idx]),
            str(payload["sig_ids"][idx]),
            _scenario_of(payload, idx),
        ]
    )


def build_collaborative_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int = 8,
    query_per_class: int = 20,
    qknn_k: int = 8,
    seed: int = 4070303,
    support_quantile: float = 0.05,
    proxy_quantile: float = 0.95,
    risk_temperature: float = 0.035,
    event_alignment_policy: str = "strict_event_key",
    support_selection_policy: str = "stable_first",
    unknown_gate_mode: str = "score",
    radius_quantile: float = 0.95,
    margin_quantile: float = 0.05,
    score_quantile: float = 0.05,
    mahalanobis_quantile: float = 0.95,
    evt_tail_quantile: float = 0.80,
    oldness_quantile: float = 0.05,
    radius_slack: float = 0.02,
    margin_slack: float = 0.0,
    score_slack: float = 0.0,
    mahalanobis_slack: float = 0.0,
    evt_tail_slack: float = 0.0,
    oldness_slack: float = 0.0,
    radius_temperature: float = 0.02,
    margin_temperature: float = 0.02,
    mahalanobis_temperature: float = 0.20,
    evt_temperature: float = 0.05,
    oldness_temperature: float = 0.05,
    mahalanobis_variance_floor: float = 1e-4,
    evt_min_scale: float = 1e-3,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    candidate_class_top_m: int = 0,
    support_calibration_mode: str = "self",
    score_threshold_combine: str = "max",
    class_score_threshold_enabled: bool = False,
    class_score_threshold_quantile: float | None = None,
    class_score_threshold_min_support: int = 1,
    class_conformal_enabled: bool = False,
    class_conformal_min_support: int = 2,
    class_evidence_top_m: int = 0,
    virtual_unknown_calibration_enabled: bool = False,
    virtual_unknown_samples_per_class: int = 0,
    virtual_unknown_mix_alpha: float = 0.50,
    virtual_unknown_noise_scale: float = 0.02,
    virtual_unknown_neighbor_count: int = 2,
    virtual_unknown_risk_enabled: bool = False,
    virtual_unknown_risk_samples_per_class: int = 2,
    virtual_unknown_risk_temperature: float = 0.05,
    virtual_unknown_risk_margin: float = 0.0,
    class_shell_unknown_risk_enabled: bool = False,
    class_shell_radius_scale: float = 1.25,
    class_shell_risk_temperature: float = 0.05,
    class_shell_risk_margin: float = 0.0,
    evidence_packet_bytes: float = 40.0,
    receiver_reliability_policy: str = "deployment_prior",
    receiver_class_reliability_policy: str = "none",
    prototype_score_blend: float = 0.0,
    mahalanobis_score_blend: float = 0.0,
    mahalanobis_score_temperature: float = 0.20,
    prototype_calibration_policy: str = "none",
    prototype_calibration_alpha: float = 0.0,
    prototype_calibration_top_m: int = 2,
    feature_adapter_policy: str = "none",
    feature_adapter_strength: float = 0.0,
    feature_adapter_variance_floor: float = 1e-4,
    candidate_audit_unknown_risk_enabled: bool = False,
    candidate_audit_disagreement_risk: float = 1.0,
    candidate_audit_min_gap: float = 0.0,
    candidate_audit_gap_risk: float = 0.0,
    class_verifier_policy: str = "none",
    class_verifier_top_m: int = 0,
    class_verifier_pvalue_weight: float = 1.0,
    class_verifier_reliability_weight: float = 1.0,
    class_verifier_risk_weight: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
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
    if not target_receivers:
        raise RuntimeError("no target receivers found")
    if not source_receivers:
        raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: source receiver metadata is required to verify R_s/R_t disjointness")
    overlap_receivers = sorted(set(source_receivers) & set(target_receivers))
    if overlap_receivers:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: R_s and R_t overlap: {overlap_receivers}")

    proxy_idx = np.where(roles == PROXY_UNKNOWN_ROLE)[0].astype(int)
    if proxy_idx.size:
        proxy_receivers = sorted({str(rx_ids[i]) for i in proxy_idx.tolist()})
        proxy_labels = sorted({str(tx_ids[i]) for i in proxy_idx.tolist()})
        proxy_target_overlap = sorted(set(proxy_receivers) & set(target_receivers))
        proxy_label_overlap = sorted(set(proxy_labels) & (set(old_labels) | set(new_labels) | set(unknown_labels)))
        if proxy_target_overlap or proxy_label_overlap:
            raise RuntimeError(
                "LOCAL_PROTOCOL_REPAIR_REQUIRED: proxy_unknown calibration rows must be source-only "
                f"and disjoint from target known/unknown labels; receiver_overlap={proxy_target_overlap}, "
                f"label_overlap={proxy_label_overlap}"
            )
    receiver_memories: dict[str, QknnMemory] = {}
    receiver_thresholds: dict[str, float] = {}
    receiver_class_thresholds: dict[str, dict[str, float]] = {}
    receiver_class_conformal_scores: dict[str, dict[str, list[float]]] = {}
    receiver_class_reliabilities: dict[str, dict[str, float]] = {}
    receiver_deployment_priors: dict[str, float] = {}
    receiver_virtual_unknowns: dict[str, np.ndarray] = {}
    receiver_virtual_unknown_counts: dict[str, int] = {}
    receiver_feature_adapters: dict[str, FeatureAdapter] = {}
    evidence: list[dict[str, Any]] = []
    receiver_query: dict[str, dict[str, list[int]]] = {}
    threshold_scope = "support_known_only"
    reliability_policy = str(receiver_reliability_policy or "deployment_prior").strip().lower()
    if reliability_policy not in {"deployment_prior", "support_density", "margin_density"}:
        raise ValueError("receiver_reliability_policy must be deployment_prior, support_density, or margin_density")
    receiver_class_policy = str(receiver_class_reliability_policy or "none").strip().lower()
    if receiver_class_policy not in {"none", "support_calibrated"}:
        raise ValueError("receiver_class_reliability_policy must be none or support_calibrated")
    if receiver_class_policy == "support_calibrated" and not bool(class_conformal_enabled):
        raise ValueError("receiver_class_reliability_policy=support_calibrated requires --class_conformal_enabled")
    prototype_blend = float(prototype_score_blend)
    if prototype_blend < 0.0:
        raise ValueError("prototype_score_blend must be >= 0")
    mahalanobis_blend = float(mahalanobis_score_blend)
    if mahalanobis_blend < 0.0:
        raise ValueError("mahalanobis_score_blend must be >= 0")
    mahalanobis_score_temp = max(float(mahalanobis_score_temperature), 1e-6)
    class_verifier = str(class_verifier_policy or "none").strip().lower()
    if class_verifier not in {"none", "support_quality"}:
        raise ValueError("class_verifier_policy must be none or support_quality")
    verifier_top_m = int(class_verifier_top_m)
    if verifier_top_m < 0:
        raise ValueError("class_verifier_top_m must be >= 0")
    verifier_candidate_top_m = verifier_top_m if verifier_top_m > 0 else int(candidate_class_top_m)
    verifier_pvalue_weight = max(float(class_verifier_pvalue_weight), 0.0)
    verifier_reliability_weight = max(float(class_verifier_reliability_weight), 0.0)
    verifier_risk_weight = max(float(class_verifier_risk_weight), 0.0)
    proto_cal_policy = str(prototype_calibration_policy or "none").strip().lower()
    if proto_cal_policy not in {"none", "teen_blend", "teen_separate"}:
        raise ValueError("prototype_calibration_policy must be none, teen_blend, or teen_separate")
    proto_cal_alpha = float(np.clip(float(prototype_calibration_alpha), 0.0, 1.0))
    proto_cal_top_m = max(1, int(prototype_calibration_top_m))
    adapter_policy = str(feature_adapter_policy or "none").strip().lower()
    if adapter_policy not in {"none", "support_center", "support_bn_affine"}:
        raise ValueError("feature_adapter_policy must be none, support_center, or support_bn_affine")
    adapter_strength = float(np.clip(float(feature_adapter_strength), 0.0, 1.0))
    adapter_var_floor = max(float(feature_adapter_variance_floor), 1e-8)
    t0 = time.perf_counter()

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
            _require_split(
                rx,
                "target_old",
                label,
                support,
                query,
                k_shot=k_shot,
                query_per_class=query_per_class,
            )
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
            _require_split(
                rx,
                "target_new",
                label,
                support,
                query,
                k_shot=k_shot,
                query_per_class=query_per_class,
            )
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
            _require_split(
                rx,
                UNKNOWN_ROLE,
                label,
                [],
                query,
                k_shot=0,
                query_per_class=query_per_class,
            )
            receiver_query[rx]["unknown"].extend(query)
        if not support_indices:
            raise RuntimeError(f"receiver {rx} has no known-class support rows")
        feature_adapter = _fit_feature_adapter(
            features[np.asarray(support_indices, dtype=int)],
            policy=adapter_policy,
            strength=adapter_strength,
            variance_floor=adapter_var_floor,
        )
        support_features_for_memory = _apply_feature_adapter(
            features[np.asarray(support_indices, dtype=int)],
            feature_adapter,
        )
        memory = build_qknn_memory(
            support_features_for_memory,
            support_labels,
            old_labels=set(old_labels),
            support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
            radius_quantile=radius_quantile,
            margin_quantile=margin_quantile,
            score_quantile=score_quantile,
            mahalanobis_quantile=mahalanobis_quantile,
            evt_tail_quantile=evt_tail_quantile,
            oldness_quantile=oldness_quantile,
            radius_slack=radius_slack,
            margin_slack=margin_slack,
            score_slack=score_slack,
            mahalanobis_slack=mahalanobis_slack,
            evt_tail_slack=evt_tail_slack,
            oldness_slack=oldness_slack,
            mahalanobis_variance_floor=mahalanobis_variance_floor,
            evt_min_scale=evt_min_scale,
            prototype_calibration_policy=proto_cal_policy,
            prototype_calibration_alpha=proto_cal_alpha,
            prototype_calibration_top_m=proto_cal_top_m,
        )
        virtual_sample_count = 0
        if bool(virtual_unknown_calibration_enabled):
            virtual_sample_count = max(virtual_sample_count, int(virtual_unknown_samples_per_class))
        if bool(virtual_unknown_risk_enabled):
            virtual_sample_count = max(virtual_sample_count, int(virtual_unknown_risk_samples_per_class))
        virtual_features = _virtual_unknown_features(
            memory,
            samples_per_class=virtual_sample_count,
            seed=int(seed) + 7919 * (len(receiver_memories) + 1),
            mix_alpha=float(virtual_unknown_mix_alpha),
            noise_scale=float(virtual_unknown_noise_scale),
            neighbor_count=int(virtual_unknown_neighbor_count),
        )
        proxy_parts: list[np.ndarray] = []
        if proxy_idx.size:
            proxy_parts.append(_apply_feature_adapter(features[proxy_idx], feature_adapter))
        if bool(virtual_unknown_calibration_enabled) and int(virtual_features.shape[0]) > 0:
            proxy_parts.append(virtual_features)
        proxy_features = np.vstack(proxy_parts).astype(np.float32) if proxy_parts else None
        source_proxy_scenarios = [_scenario_of(payload, int(i)) for i in proxy_idx.tolist()] if proxy_idx.size else []
        virtual_proxy_scenarios = (
            [""] * int(virtual_features.shape[0]) if bool(virtual_unknown_calibration_enabled) else []
        )
        proxy_scenarios = source_proxy_scenarios + virtual_proxy_scenarios
        threshold, scope = _threshold_from_calibration(
            memory,
            support_features_for_memory,
            proxy_features,
            top_k=qknn_k,
            support_quantile=support_quantile,
            proxy_quantile=proxy_quantile,
            support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
            proxy_scenarios=proxy_scenarios if proxy_scenarios else None,
            scenario_aware=bool(scenario_aware),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            candidate_class_top_m=int(candidate_class_top_m),
            support_calibration_mode=str(support_calibration_mode),
            prototype_score_blend=prototype_blend,
            mahalanobis_score_blend=mahalanobis_blend,
            mahalanobis_score_temperature=mahalanobis_score_temp,
        )
        if scope == "source_only" and proxy_idx.size:
            threshold_scope = scope
        elif bool(virtual_unknown_calibration_enabled) and int(virtual_features.shape[0]) > 0 and threshold_scope != "source_only":
            threshold_scope = "support_virtual_unknown"
        class_thresholds: dict[str, float] = {}
        if bool(class_score_threshold_enabled):
            class_thresholds = _label_thresholds_from_calibration(
                memory,
                support_features_for_memory,
                support_labels,
                proxy_features,
                top_k=qknn_k,
                support_quantile=float(
                    support_quantile if class_score_threshold_quantile is None else class_score_threshold_quantile
                ),
                proxy_quantile=proxy_quantile,
                support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
                proxy_scenarios=proxy_scenarios if proxy_scenarios else None,
                scenario_aware=bool(scenario_aware),
                radius_norm=float(radius_norm),
                old_bias=float(old_bias),
                candidate_class_top_m=int(candidate_class_top_m),
                support_calibration_mode=str(support_calibration_mode),
                prototype_score_blend=prototype_blend,
                mahalanobis_score_blend=mahalanobis_blend,
                mahalanobis_score_temperature=mahalanobis_score_temp,
                min_support=int(class_score_threshold_min_support),
            )
        conformal_scores: dict[str, list[float]] = {}
        if bool(class_conformal_enabled):
            conformal_scores = _label_score_samples_from_calibration(
                memory,
                support_features_for_memory,
                support_labels,
                top_k=qknn_k,
                support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
                scenario_aware=bool(scenario_aware),
                radius_norm=float(radius_norm),
                old_bias=float(old_bias),
                candidate_class_top_m=int(candidate_class_top_m),
                support_calibration_mode=str(support_calibration_mode),
                prototype_score_blend=prototype_blend,
                mahalanobis_score_blend=mahalanobis_blend,
                mahalanobis_score_temperature=mahalanobis_score_temp,
                min_support=int(class_conformal_min_support),
            )
        if receiver_class_policy == "support_calibrated" and not conformal_scores:
            raise RuntimeError(
                f"receiver_class_reliability_policy=support_calibrated produced no support calibration scores for receiver {rx}"
            )
        class_reliabilities: dict[str, float] = {}
        for label_name in sorted(set(old_labels) | set(new_labels)):
            class_reliabilities[label_name] = _receiver_class_reliability_from_support(
                conformal_scores.get(label_name),
                threshold=float(class_thresholds.get(label_name, threshold)),
                min_support=int(class_conformal_min_support),
            )
        receiver_memories[rx] = memory
        receiver_feature_adapters[rx] = feature_adapter
        receiver_thresholds[rx] = threshold
        receiver_class_thresholds[rx] = class_thresholds
        receiver_class_conformal_scores[rx] = conformal_scores
        receiver_class_reliabilities[rx] = class_reliabilities
        prior_values = [float(value) for value in class_reliabilities.values()]
        receiver_deployment_priors[rx] = (
            float(sum(prior_values) / max(len(prior_values), 1)) if prior_values else 1.0
        )
        receiver_virtual_unknown_counts[rx] = int(virtual_features.shape[0])
        receiver_virtual_unknowns[rx] = virtual_features if bool(virtual_unknown_risk_enabled) else np.zeros(
            (0, int(memory.centroids.shape[1])),
            dtype=np.float32,
        )

    elapsed_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)
    total_query_rows = sum(len(v[role]) for v in receiver_query.values() for role in ["old", "seen_new", "unknown"])
    per_row_ms = elapsed_ms / max(total_query_rows, 1)

    alignment_policy = str(event_alignment_policy or "strict_event_key").strip().lower()
    if alignment_policy not in {"strict_event_key", "receiver_domain_ranked"}:
        raise ValueError("event_alignment_policy must be strict_event_key or receiver_domain_ranked")

    for role_name, eval_role in [("old", "target_old"), ("seen_new", "target_new"), ("unknown", UNKNOWN_ROLE)]:
        label_set = old_labels if role_name == "old" else new_labels if role_name == "seen_new" else unknown_labels
        for label in label_set:
            by_rx_key: dict[str, dict[str, int]] = {}
            for rx in target_receivers:
                keyed: dict[str, int] = {}
                for idx in receiver_query[rx][role_name]:
                    if tx_ids[idx] == label:
                        keyed[_event_key(payload, idx, role_name, label)] = int(idx)
                by_rx_key[rx] = keyed
            if alignment_policy == "strict_event_key":
                common_event_ids = sorted(set.intersection(*(set(by_rx_key[rx]) for rx in target_receivers)))
                event_groups = [
                    (event_id, {rx: by_rx_key[rx][event_id] for rx in target_receivers})
                    for event_id in common_event_ids
                ]
                row_alignment = "role_tx_day_sig_scenario"
            else:
                by_rx_scenario: dict[str, dict[str, list[int]]] = {}
                for rx in target_receivers:
                    by_rx_scenario[rx] = defaultdict(list)
                    for idx in receiver_query[rx][role_name]:
                        if tx_ids[idx] == label:
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
                        if not rx_to_idx:
                            continue
                        event_groups.append(
                            (
                                f"{role_name}|{label}|{scenario}|rank{event_i:05d}",
                                rx_to_idx,
                            )
                        )
                row_alignment = "receiver_domain_ranked_by_role_tx_scenario"
            for event_id, rx_to_idx in event_groups:
                for rx in sorted(rx_to_idx):
                    idx = rx_to_idx[rx]
                    memory = receiver_memories[rx]
                    query_feature = _apply_feature_adapter(features[[idx]], receiver_feature_adapters[rx])
                    (
                        pred,
                        score,
                        margin,
                        candidate_counts,
                        support_neighbor_counts,
                        support_densities,
                        second_labels,
                        second_scores,
                    ) = qknn_scores(
                        memory,
                        query_feature,
                        top_k=qknn_k,
                        query_scenarios=[_scenario_of(payload, idx)],
                        scenario_aware=bool(scenario_aware),
                        radius_norm=float(radius_norm),
                        old_bias=float(old_bias),
                        candidate_class_top_m=int(candidate_class_top_m),
                        prototype_score_blend=prototype_blend,
                        mahalanobis_score_blend=mahalanobis_blend,
                        mahalanobis_score_temperature=mahalanobis_score_temp,
                    )
                    if int(candidate_class_top_m) > 0:
                        (
                            audit_pred,
                            audit_score,
                            _audit_margin,
                            _audit_candidate_counts,
                            _audit_support_neighbor_counts,
                            _audit_support_densities,
                            audit_second_labels,
                            audit_second_scores,
                        ) = qknn_scores(
                            memory,
                            query_feature,
                            top_k=qknn_k,
                            query_scenarios=[_scenario_of(payload, idx)],
                            scenario_aware=bool(scenario_aware),
                            radius_norm=float(radius_norm),
                            old_bias=float(old_bias),
                            candidate_class_top_m=0,
                            prototype_score_blend=prototype_blend,
                            mahalanobis_score_blend=mahalanobis_blend,
                            mahalanobis_score_temperature=mahalanobis_score_temp,
                        )
                    else:
                        audit_pred = pred
                        audit_score = score
                        audit_second_labels = second_labels
                        audit_second_scores = second_scores
                    base_pred = str(pred[0])
                    base_score = float(score[0])
                    base_margin = float(margin[0])
                    base_second_label = str(second_labels[0])
                    base_second_score = float(second_scores[0])
                    verifier_fields: dict[str, float | int | str] = {
                        "class_verifier_policy": class_verifier,
                        "class_verifier_top_m": int(verifier_top_m),
                        "class_verifier_candidate_top_m": int(verifier_candidate_top_m),
                        "class_verifier_changed": 0,
                        "class_verifier_base_label": base_pred,
                        "class_verifier_base_score": base_score,
                        "class_verifier_base_margin": base_margin,
                        "class_verifier_base_second_label": base_second_label,
                        "class_verifier_base_second_score": base_second_score,
                        "class_verifier_top1_label": base_pred,
                        "class_verifier_top1_raw_score": base_score,
                        "class_verifier_top1_raw_margin": base_margin,
                        "class_verifier_top1_verified_score": base_score,
                        "class_verifier_top1_pvalue": 1.0 if not bool(class_conformal_enabled) else 0.0,
                        "class_verifier_top1_receiver_class_reliability": 1.0,
                        "class_verifier_top1_unknown_risk": 0.0,
                        "class_verifier_top1_class_shell_risk": 0.0,
                        "class_verifier_top1_combined_risk": 0.0,
                        "class_verifier_top1_support_count": 0,
                        "class_verifier_top1_effective_score_threshold": 0.0,
                        "class_verifier_top1_score_threshold_source": "not_applied",
                        "class_verifier_second_label": base_second_label,
                        "class_verifier_second_raw_score": base_second_score,
                        "class_verifier_second_verified_score": base_second_score,
                    }
                    if class_verifier == "support_quality":
                        verifier = _support_quality_class_verifier(
                            memory,
                            query_feature,
                            rx=rx,
                            scenario=_scenario_of(payload, idx),
                            qknn_k=qknn_k,
                            scenario_aware=bool(scenario_aware),
                            radius_norm=float(radius_norm),
                            old_bias=float(old_bias),
                            candidate_class_top_m=int(verifier_candidate_top_m),
                            class_verifier_top_m=int(verifier_top_m),
                            prototype_blend=prototype_blend,
                            mahalanobis_blend=mahalanobis_blend,
                            mahalanobis_score_temp=mahalanobis_score_temp,
                            receiver_threshold=float(receiver_thresholds[rx]),
                            receiver_class_thresholds=receiver_class_thresholds,
                            receiver_class_conformal_scores=receiver_class_conformal_scores,
                            receiver_class_reliabilities=receiver_class_reliabilities,
                            score_threshold_combine=str(score_threshold_combine),
                            class_score_threshold_enabled=bool(class_score_threshold_enabled),
                            class_conformal_enabled=bool(class_conformal_enabled),
                            unknown_gate_mode=str(unknown_gate_mode),
                            risk_temperature=float(risk_temperature),
                            radius_temperature=float(radius_temperature),
                            margin_temperature=float(margin_temperature),
                            mahalanobis_temperature=float(mahalanobis_temperature),
                            evt_temperature=float(evt_temperature),
                            oldness_temperature=float(oldness_temperature),
                            class_shell_unknown_risk_enabled=bool(class_shell_unknown_risk_enabled),
                            class_shell_radius_scale=float(class_shell_radius_scale),
                            class_shell_risk_temperature=float(class_shell_risk_temperature),
                            class_shell_risk_margin=float(class_shell_risk_margin),
                            pvalue_weight=verifier_pvalue_weight,
                            reliability_weight=verifier_reliability_weight,
                            risk_weight=verifier_risk_weight,
                        )
                        pred = np.asarray([str(verifier["top1_label"])], dtype=object)
                        score = np.asarray([float(verifier["top1_raw_score"])], dtype=np.float64)
                        margin = np.asarray([float(verifier["top1_raw_margin"])], dtype=np.float64)
                        second_labels = np.asarray([str(verifier["second_label"])], dtype=object)
                        second_scores = np.asarray([float(verifier["second_raw_score"])], dtype=np.float64)
                        verifier_fields.update(
                            {
                                "class_verifier_changed": int(str(verifier["top1_label"]) != base_pred),
                                "class_verifier_top1_label": str(verifier["top1_label"]),
                                "class_verifier_top1_raw_score": float(verifier["top1_raw_score"]),
                                "class_verifier_top1_raw_margin": float(verifier["top1_raw_margin"]),
                                "class_verifier_top1_verified_score": float(verifier["top1_verified_score"]),
                                "class_verifier_top1_pvalue": float(verifier["top1_pvalue"]),
                                "class_verifier_top1_receiver_class_reliability": float(
                                    verifier["top1_receiver_class_reliability"]
                                ),
                                "class_verifier_top1_unknown_risk": float(verifier["top1_unknown_risk"]),
                                "class_verifier_top1_class_shell_risk": float(verifier["top1_class_shell_risk"]),
                                "class_verifier_top1_combined_risk": float(verifier["top1_combined_risk"]),
                                "class_verifier_top1_support_count": int(verifier["top1_support_count"]),
                                "class_verifier_top1_effective_score_threshold": float(
                                    verifier["top1_effective_score_threshold"]
                                ),
                                "class_verifier_top1_score_threshold_source": str(
                                    verifier["top1_score_threshold_source"]
                                ),
                                "class_verifier_second_label": str(verifier["second_label"]),
                                "class_verifier_second_raw_score": float(verifier["second_raw_score"]),
                                "class_verifier_second_verified_score": float(verifier["second_verified_score"]),
                            }
                        )
                    audit_gap = float(audit_score[0] - audit_second_scores[0])
                    candidate_audit_disagreement = str(audit_pred[0]) != str(pred[0])
                    support_density = float(support_densities[0])
                    if reliability_policy == "support_density":
                        reliability = support_density
                    elif reliability_policy == "margin_density":
                        reliability = support_density * max(0.0, min(1.0, float(margin[0]) / max(memory.margin_threshold, 1e-6)))
                    else:
                        reliability = 1.0
                    class_threshold_value = receiver_class_thresholds.get(rx, {}).get(str(pred[0]))
                    threshold_source = "receiver_global"
                    if bool(class_score_threshold_enabled):
                        if class_threshold_value is not None:
                            receiver_score_threshold = float(class_threshold_value)
                            threshold_source = "class"
                        else:
                            receiver_score_threshold = float(receiver_thresholds[rx])
                            threshold_source = "receiver_fallback"
                    else:
                        receiver_score_threshold = float(receiver_thresholds[rx])
                    effective_score_threshold = _combine_score_threshold(
                        receiver_score_threshold,
                        memory.score_threshold,
                        str(score_threshold_combine),
                    )
                    class_conformal_pvalue = _conformal_pvalue(
                        float(score[0]),
                        receiver_class_conformal_scores.get(rx, {}).get(str(pred[0])),
                    )
                    class_evidence_fields: dict[str, float | int | str] = {}
                    if int(class_evidence_top_m) > 0:
                        score_labels_for_event, score_matrix_for_event = _qknn_label_score_matrix(
                            memory,
                            query_feature,
                            top_k=qknn_k,
                            query_scenarios=[_scenario_of(payload, idx)],
                            scenario_aware=bool(scenario_aware),
                            radius_norm=float(radius_norm),
                            old_bias=float(old_bias),
                            candidate_class_top_m=int(candidate_class_top_m),
                            prototype_score_blend=prototype_blend,
                            mahalanobis_score_blend=mahalanobis_blend,
                            mahalanobis_score_temperature=mahalanobis_score_temp,
                        )
                        label_scores_for_event = [
                            (str(label), float(score_value))
                            for label, score_value in zip(
                                score_labels_for_event.tolist(),
                                score_matrix_for_event[0].tolist(),
                            )
                        ]
                        label_scores_for_event.sort(key=lambda item: (item[1], item[0]), reverse=True)
                        top_label_scores = label_scores_for_event[: int(class_evidence_top_m)]
                        for rank, (label_name, label_score) in enumerate(
                            top_label_scores,
                            start=1,
                        ):
                            label_second_score = max(
                                (score_value for other_label, score_value in label_scores_for_event if other_label != label_name),
                                default=0.0,
                            )
                            label_margin = float(label_score - label_second_score)
                            label_class_threshold_value = receiver_class_thresholds.get(rx, {}).get(label_name)
                            if bool(class_score_threshold_enabled) and label_class_threshold_value is not None:
                                label_receiver_threshold = float(label_class_threshold_value)
                                label_threshold_source = "class"
                            elif bool(class_score_threshold_enabled):
                                label_receiver_threshold = float(receiver_thresholds[rx])
                                label_threshold_source = "receiver_fallback"
                            else:
                                label_receiver_threshold = float(receiver_thresholds[rx])
                                label_threshold_source = "receiver_global"
                            label_effective_threshold = _combine_score_threshold(
                                label_receiver_threshold,
                                memory.score_threshold,
                                str(score_threshold_combine),
                            )
                            (
                                label_risk,
                                label_score_risk,
                                label_radius_risk,
                                label_margin_risk,
                                label_mahalanobis_risk,
                                label_evt_risk,
                                label_oldness_risk,
                                label_class_radius,
                                label_class_radius_z,
                            ) = _combined_unknown_risk(
                                memory,
                                query_feature,
                                [label_name],
                                np.asarray([label_score], dtype=np.float64),
                                np.asarray([label_margin], dtype=np.float64),
                                label_effective_threshold,
                                temperature=risk_temperature,
                                gate_mode=unknown_gate_mode,
                                radius_temperature=radius_temperature,
                                margin_temperature=margin_temperature,
                                mahalanobis_temperature=mahalanobis_temperature,
                                evt_temperature=evt_temperature,
                                oldness_temperature=oldness_temperature,
                            )
                            label_shell_risk, label_shell_distance = _class_shell_boundary_risk(
                                memory,
                                query_feature,
                                [label_name],
                                radius_scale=float(class_shell_radius_scale),
                                temperature=float(class_shell_risk_temperature),
                                margin=float(class_shell_risk_margin),
                            )
                            label_pvalue = _conformal_pvalue(
                                label_score,
                                receiver_class_conformal_scores.get(rx, {}).get(label_name),
                            )
                            label_support_count = len(receiver_class_conformal_scores.get(rx, {}).get(label_name, []))
                            label_receiver_class_reliability = receiver_class_reliabilities.get(rx, {}).get(
                                label_name,
                                1.0,
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_label"] = label_name
                            class_evidence_fields[f"class_evidence_top{rank}_score"] = float(label_score)
                            class_evidence_fields[f"class_evidence_top{rank}_margin"] = float(label_margin)
                            class_evidence_fields[f"class_evidence_top{rank}_conformal_pvalue"] = (
                                float(label_pvalue) if bool(class_conformal_enabled) else 0.0
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_support_count"] = int(label_support_count)
                            class_evidence_fields[f"class_evidence_top{rank}_effective_score_threshold"] = float(
                                label_effective_threshold
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_score_threshold_source"] = label_threshold_source
                            class_evidence_fields[f"class_evidence_top{rank}_unknown_risk"] = float(label_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_score_risk"] = float(label_score_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_radius_risk"] = float(label_radius_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_margin_risk"] = float(label_margin_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_mahalanobis_risk"] = float(
                                label_mahalanobis_risk[0]
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_evt_risk"] = float(label_evt_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_oldness_risk"] = float(label_oldness_risk[0])
                            class_evidence_fields[f"class_evidence_top{rank}_class_shell_risk"] = (
                                float(label_shell_risk[0]) if bool(class_shell_unknown_risk_enabled) else 0.0
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_class_shell_distance"] = float(
                                label_shell_distance[0]
                            )
                            class_evidence_fields[f"class_evidence_top{rank}_class_radius"] = float(label_class_radius[0])
                            class_evidence_fields[f"class_evidence_top{rank}_class_radius_z"] = float(label_class_radius_z[0])
                            class_evidence_fields[f"class_evidence_top{rank}_receiver_class_reliability"] = float(
                                label_receiver_class_reliability
                            )
                    (
                        risk,
                        score_risk,
                        radius_risk,
                        margin_risk,
                        mahalanobis_risk,
                        evt_risk,
                        oldness_risk,
                        class_radius,
                        class_radius_z,
                    ) = _combined_unknown_risk(
                        memory,
                        query_feature,
                        pred,
                        score,
                        margin,
                        effective_score_threshold,
                        temperature=risk_temperature,
                        gate_mode=unknown_gate_mode,
                        radius_temperature=radius_temperature,
                        margin_temperature=margin_temperature,
                        mahalanobis_temperature=mahalanobis_temperature,
                        evt_temperature=evt_temperature,
                        oldness_temperature=oldness_temperature,
                    )
                    virtual_unknown_risk, virtual_unknown_score = _virtual_unknown_boundary_risk(
                        query_feature,
                        score,
                        receiver_virtual_unknowns.get(rx),
                        temperature=float(virtual_unknown_risk_temperature),
                        margin=float(virtual_unknown_risk_margin),
                    )
                    class_shell_risk, class_shell_distance = _class_shell_boundary_risk(
                        memory,
                        query_feature,
                        pred,
                        radius_scale=float(class_shell_radius_scale),
                        temperature=float(class_shell_risk_temperature),
                        margin=float(class_shell_risk_margin),
                    )
                    candidate_audit_risk = 0.0
                    if bool(candidate_audit_unknown_risk_enabled):
                        if candidate_audit_disagreement:
                            candidate_audit_risk = max(
                                candidate_audit_risk,
                                float(candidate_audit_disagreement_risk),
                            )
                        if float(candidate_audit_min_gap) > 0.0 and audit_gap < float(candidate_audit_min_gap):
                            candidate_audit_risk = max(candidate_audit_risk, float(candidate_audit_gap_risk))
                    unknown_risk_value = max(
                        float(risk[0]),
                        float(virtual_unknown_risk[0]) if bool(virtual_unknown_risk_enabled) else 0.0,
                        float(class_shell_risk[0]) if bool(class_shell_unknown_risk_enabled) else 0.0,
                        float(candidate_audit_risk),
                    )
                    score_risk_value = max(float(score_risk[0]), float(candidate_audit_risk))
                    evidence.append(
                        {
                            "event_id": event_id,
                            "receiver_id": rx,
                            "role": role_name,
                            "true_label": "__unknown__" if role_name == "unknown" else str(tx_ids[idx]),
                            "predicted_label": str(pred[0]),
                            "second_label": str(second_labels[0]),
                            "second_score": float(second_scores[0]),
                            "label_score_gap": float(score[0] - second_scores[0]),
                            "audit_full_top1_label": str(audit_pred[0]),
                            "audit_full_top1_score": float(audit_score[0]),
                            "audit_full_second_label": str(audit_second_labels[0]),
                            "audit_full_second_score": float(audit_second_scores[0]),
                            "audit_full_label_score_gap": audit_gap,
                            "candidate_audit_disagreement": int(candidate_audit_disagreement),
                            "candidate_audit_risk": float(candidate_audit_risk),
                            "known_score": float(score[0]),
                            "known_margin": float(margin[0]),
                            "effective_score_threshold": float(effective_score_threshold),
                            "receiver_score_threshold": float(receiver_score_threshold),
                            "base_receiver_score_threshold": float(receiver_thresholds[rx]),
                            "class_score_threshold": float(class_threshold_value) if class_threshold_value is not None else 0.0,
                            "class_score_threshold_enabled": int(bool(class_score_threshold_enabled)),
                            "score_threshold_source": threshold_source,
                            "class_conformal_enabled": int(bool(class_conformal_enabled)),
                            "class_conformal_pvalue": float(class_conformal_pvalue) if bool(class_conformal_enabled) else 0.0,
                            "class_conformal_support_count": int(
                                len(receiver_class_conformal_scores.get(rx, {}).get(str(pred[0]), []))
                            ),
                            "receiver_class_reliability": float(
                                receiver_class_reliabilities.get(rx, {}).get(str(pred[0]), 1.0)
                            ),
                            "class_evidence_top_m": int(class_evidence_top_m),
                            **verifier_fields,
                            **class_evidence_fields,
                            "virtual_unknown_calibration_enabled": int(bool(virtual_unknown_calibration_enabled)),
                            "virtual_unknown_risk_enabled": int(bool(virtual_unknown_risk_enabled)),
                            "virtual_unknown_count": int(
                                receiver_virtual_unknown_counts.get(rx, 0)
                            ),
                            "class_shell_unknown_risk_enabled": int(bool(class_shell_unknown_risk_enabled)),
                            "candidate_class_count": int(candidate_counts[0]),
                            "support_neighbor_count": int(support_neighbor_counts[0]),
                            "support_density": support_density,
                            "receiver_deployment_prior": float(receiver_deployment_priors.get(rx, 1.0)),
                            "receiver_deployment_prior_source": "support_calibrated_receiver_class_mean",
                            "prototype_score_blend": prototype_blend,
                            "prototype_assisted": int(prototype_blend > 0.0),
                            "prototype_calibration_policy": proto_cal_policy,
                            "prototype_calibration_alpha": proto_cal_alpha,
                            "prototype_calibration_top_m": proto_cal_top_m,
                            "prototype_only_top1": int(prototype_blend > 0.0 and int(support_neighbor_counts[0]) == 0),
                            "mahalanobis_score_blend": mahalanobis_blend,
                            "mahalanobis_score_temperature": mahalanobis_score_temp,
                            "mahalanobis_score_assisted": int(mahalanobis_blend > 0.0),
                            "feature_adapter_policy": feature_adapter.policy,
                            "feature_adapter_strength": float(feature_adapter.strength),
                            "feature_adapter_variance_floor": adapter_var_floor,
                            "unknown_risk": float(unknown_risk_value),
                            "score_risk": float(score_risk_value),
                            "radius_risk": float(radius_risk[0]),
                            "margin_risk": float(margin_risk[0]),
                            "mahalanobis_risk": float(mahalanobis_risk[0]),
                            "evt_risk": float(evt_risk[0]),
                            "oldness_risk": float(oldness_risk[0]),
                            "virtual_unknown_risk": float(virtual_unknown_risk[0]) if bool(virtual_unknown_risk_enabled) else 0.0,
                            "virtual_unknown_score": float(virtual_unknown_score[0]) if bool(virtual_unknown_risk_enabled) else 0.0,
                            "class_shell_risk": float(class_shell_risk[0]) if bool(class_shell_unknown_risk_enabled) else 0.0,
                            "class_shell_distance": float(class_shell_distance[0]),
                            "class_shell_radius_scale": float(class_shell_radius_scale),
                            "class_radius": float(class_radius[0]),
                            "class_radius_z": float(class_radius_z[0]),
                            "reliability": float(reliability),
                            "reliability_source": reliability_policy,
                            "latency_ms": float(per_row_ms),
                            "bytes": float(evidence_packet_bytes),
                            "threshold_selection_label_scope": threshold_scope,
                            "calibration_role": "query",
                            "sat_scenario": _scenario_of(payload, idx),
                            "raw_role": eval_role,
                            "event_alignment": row_alignment,
                        }
                    )
    if not evidence:
        raise RuntimeError(
            "NO_ALIGNED_COLLABORATIVE_EVENTS: target receiver query rows do not share "
            "role+tx+day+sig+scenario keys; use --event_alignment_policy receiver_domain_ranked "
            "only for explicitly marked receiver-domain ensemble diagnostics"
        )

    metadata = {
        "source_receiver_ids": source_receivers,
        "target_receiver_ids": target_receivers,
        "old_tx_ids": old_labels,
        "seen_new_tx_ids": new_labels,
        "unknown_tx_ids": unknown_labels,
        "target_channel_view": ",".join(sorted({row["sat_scenario"] for row in evidence if row["sat_scenario"]})),
        "qknn_k": int(qknn_k),
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "prototype_storage_bytes": int(sum(memory.prototype_storage_bytes for memory in receiver_memories.values())),
        "evidence_bytes_per_receiver_event": float(evidence_packet_bytes),
        "event_alignment": row_alignment if alignment_policy == "receiver_domain_ranked" else "role_tx_day_sig_scenario",
        "event_alignment_policy": alignment_policy,
        "strict_same_event_collaboration": alignment_policy == "strict_event_key",
        "receiver_thresholds": receiver_thresholds,
        "receiver_class_thresholds": receiver_class_thresholds,
        "receiver_class_reliability_policy": receiver_class_policy,
        "receiver_class_reliabilities": receiver_class_reliabilities,
        "threshold_scope": threshold_scope,
        "support_calibration_mode": str(support_calibration_mode),
        "score_threshold_combine": str(score_threshold_combine),
        "class_score_threshold_enabled": bool(class_score_threshold_enabled),
        "class_score_threshold_quantile": float(
            support_quantile if class_score_threshold_quantile is None else class_score_threshold_quantile
        ),
        "class_score_threshold_min_support": int(class_score_threshold_min_support),
        "class_conformal_enabled": bool(class_conformal_enabled),
        "class_conformal_min_support": int(class_conformal_min_support),
        "class_evidence_top_m": int(class_evidence_top_m),
        "receiver_class_conformal_counts": {
            str(rx): {str(label): int(len(values)) for label, values in scores.items()}
            for rx, scores in receiver_class_conformal_scores.items()
        },
        "virtual_unknown_calibration_enabled": bool(virtual_unknown_calibration_enabled),
        "virtual_unknown_samples_per_class": int(virtual_unknown_samples_per_class),
        "virtual_unknown_mix_alpha": float(virtual_unknown_mix_alpha),
        "virtual_unknown_noise_scale": float(virtual_unknown_noise_scale),
        "virtual_unknown_neighbor_count": int(virtual_unknown_neighbor_count),
        "virtual_unknown_risk_enabled": bool(virtual_unknown_risk_enabled),
        "virtual_unknown_risk_samples_per_class": int(virtual_unknown_risk_samples_per_class),
        "virtual_unknown_risk_temperature": float(virtual_unknown_risk_temperature),
        "virtual_unknown_risk_margin": float(virtual_unknown_risk_margin),
        "class_shell_unknown_risk_enabled": bool(class_shell_unknown_risk_enabled),
        "class_shell_radius_scale": float(class_shell_radius_scale),
        "class_shell_risk_temperature": float(class_shell_risk_temperature),
        "class_shell_risk_margin": float(class_shell_risk_margin),
        "receiver_reliability_policy": reliability_policy,
        "prototype_score_blend": prototype_blend,
        "prototype_assisted_qknn": prototype_blend > 0.0,
        "prototype_calibration_policy": proto_cal_policy,
        "prototype_calibration_alpha": proto_cal_alpha,
        "prototype_calibration_top_m": proto_cal_top_m,
        "mahalanobis_score_blend": mahalanobis_blend,
        "mahalanobis_score_temperature": mahalanobis_score_temp,
        "mahalanobis_score_assisted_qknn": mahalanobis_blend > 0.0,
        "feature_adapter_policy": adapter_policy,
        "feature_adapter_strength": adapter_strength,
        "feature_adapter_variance_floor": adapter_var_floor,
        "candidate_audit_unknown_risk_enabled": bool(candidate_audit_unknown_risk_enabled),
        "candidate_audit_disagreement_risk": float(candidate_audit_disagreement_risk),
        "candidate_audit_min_gap": float(candidate_audit_min_gap),
        "candidate_audit_gap_risk": float(candidate_audit_gap_risk),
        "class_verifier_policy": class_verifier,
        "class_verifier_top_m": int(verifier_top_m),
        "class_verifier_candidate_top_m": int(verifier_candidate_top_m),
        "class_verifier_pvalue_weight": float(verifier_pvalue_weight),
        "class_verifier_reliability_weight": float(verifier_reliability_weight),
        "class_verifier_risk_weight": float(verifier_risk_weight),
        "support_selection_policy": str(support_selection_policy),
        "unknown_gate_mode": str(unknown_gate_mode),
        "active_risk_components": _active_risk_components_for_gate_mode(unknown_gate_mode)
        + (["virtual_unknown"] if bool(virtual_unknown_risk_enabled) else [])
        + (["class_shell"] if bool(class_shell_unknown_risk_enabled) else []),
        "scenario_aware": bool(scenario_aware),
        "radius_norm": float(radius_norm),
        "old_bias": float(old_bias),
        "candidate_class_top_m": int(candidate_class_top_m),
        "radius_quantile": float(radius_quantile),
        "margin_quantile": float(margin_quantile),
        "score_quantile": float(score_quantile),
        "mahalanobis_quantile": float(mahalanobis_quantile),
        "evt_tail_quantile": float(evt_tail_quantile),
        "oldness_quantile": float(oldness_quantile),
        "radius_slack": float(radius_slack),
        "margin_slack": float(margin_slack),
        "score_slack": float(score_slack),
        "mahalanobis_slack": float(mahalanobis_slack),
        "evt_tail_slack": float(evt_tail_slack),
        "oldness_slack": float(oldness_slack),
        "mahalanobis_temperature": float(mahalanobis_temperature),
        "evt_temperature": float(evt_temperature),
        "oldness_temperature": float(oldness_temperature),
        "mahalanobis_variance_floor": float(mahalanobis_variance_floor),
        "evt_min_scale": float(evt_min_scale),
    }
    return evidence, metadata


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(Path(args.feature_npz))
    evidence, metadata = build_collaborative_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        qknn_k=int(args.qknn_k),
        seed=int(args.seed),
        support_quantile=float(args.support_quantile),
        proxy_quantile=float(args.proxy_quantile),
        risk_temperature=float(args.risk_temperature),
        event_alignment_policy=str(args.event_alignment_policy),
        support_selection_policy=str(args.support_selection_policy),
        unknown_gate_mode=str(args.unknown_gate_mode),
        radius_quantile=float(args.radius_quantile),
        margin_quantile=float(args.margin_quantile),
        score_quantile=float(args.score_quantile),
        mahalanobis_quantile=float(args.mahalanobis_quantile),
        evt_tail_quantile=float(args.evt_tail_quantile),
        oldness_quantile=float(args.oldness_quantile),
        radius_slack=float(args.radius_slack),
        margin_slack=float(args.margin_slack),
        score_slack=float(args.score_slack),
        mahalanobis_slack=float(args.mahalanobis_slack),
        evt_tail_slack=float(args.evt_tail_slack),
        oldness_slack=float(args.oldness_slack),
        radius_temperature=float(args.radius_temperature),
        margin_temperature=float(args.margin_temperature),
        mahalanobis_temperature=float(args.mahalanobis_temperature),
        evt_temperature=float(args.evt_temperature),
        oldness_temperature=float(args.oldness_temperature),
        mahalanobis_variance_floor=float(args.mahalanobis_variance_floor),
        evt_min_scale=float(args.evt_min_scale),
        scenario_aware=bool(args.scenario_aware),
        radius_norm=float(args.radius_norm),
        old_bias=float(args.old_bias),
        candidate_class_top_m=int(args.candidate_class_top_m),
        support_calibration_mode=str(args.support_calibration_mode),
        score_threshold_combine=str(args.score_threshold_combine),
        class_score_threshold_enabled=bool(args.class_score_threshold_enabled),
        class_score_threshold_quantile=args.class_score_threshold_quantile,
        class_score_threshold_min_support=int(args.class_score_threshold_min_support),
        class_conformal_enabled=bool(args.class_conformal_enabled),
        class_conformal_min_support=int(args.class_conformal_min_support),
        class_evidence_top_m=int(args.class_evidence_top_m),
        virtual_unknown_calibration_enabled=bool(args.virtual_unknown_calibration_enabled),
        virtual_unknown_samples_per_class=int(args.virtual_unknown_samples_per_class),
        virtual_unknown_mix_alpha=float(args.virtual_unknown_mix_alpha),
        virtual_unknown_noise_scale=float(args.virtual_unknown_noise_scale),
        virtual_unknown_neighbor_count=int(args.virtual_unknown_neighbor_count),
        virtual_unknown_risk_enabled=bool(args.virtual_unknown_risk_enabled),
        virtual_unknown_risk_samples_per_class=int(args.virtual_unknown_risk_samples_per_class),
        virtual_unknown_risk_temperature=float(args.virtual_unknown_risk_temperature),
        virtual_unknown_risk_margin=float(args.virtual_unknown_risk_margin),
        class_shell_unknown_risk_enabled=bool(args.class_shell_unknown_risk_enabled),
        class_shell_radius_scale=float(args.class_shell_radius_scale),
        class_shell_risk_temperature=float(args.class_shell_risk_temperature),
        class_shell_risk_margin=float(args.class_shell_risk_margin),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        receiver_reliability_policy=str(args.receiver_reliability_policy),
        receiver_class_reliability_policy=str(args.receiver_class_reliability_policy),
        prototype_score_blend=float(args.prototype_score_blend),
        mahalanobis_score_blend=float(args.mahalanobis_score_blend),
        mahalanobis_score_temperature=float(args.mahalanobis_score_temperature),
        prototype_calibration_policy=str(args.prototype_calibration_policy),
        prototype_calibration_alpha=float(args.prototype_calibration_alpha),
        prototype_calibration_top_m=int(args.prototype_calibration_top_m),
        feature_adapter_policy=str(args.feature_adapter_policy),
        feature_adapter_strength=float(args.feature_adapter_strength),
        feature_adapter_variance_floor=float(args.feature_adapter_variance_floor),
        candidate_audit_unknown_risk_enabled=bool(args.candidate_audit_unknown_risk_enabled),
        candidate_audit_disagreement_risk=float(args.candidate_audit_disagreement_risk),
        candidate_audit_min_gap=float(args.candidate_audit_min_gap),
        candidate_audit_gap_risk=float(args.candidate_audit_gap_risk),
        class_verifier_policy=str(args.class_verifier_policy),
        class_verifier_top_m=int(args.class_verifier_top_m),
        class_verifier_pvalue_weight=float(args.class_verifier_pvalue_weight),
        class_verifier_reliability_weight=float(args.class_verifier_reliability_weight),
        class_verifier_risk_weight=float(args.class_verifier_risk_weight),
    )
    result = evaluate_collaborative_open_set_evidence(
        evidence,
        collab_counts=args.collab_counts,
        unknown_risk_threshold=float(args.unknown_risk_threshold),
        accept_margin_threshold=float(args.accept_margin_threshold),
        unknown_quantile=float(args.unknown_quantile),
        fusion_policy=str(args.fusion_policy),
        consensus_gap_threshold=float(args.consensus_gap_threshold),
        consensus_score_threshold=float(args.consensus_score_threshold),
        scorer_component_vote_threshold=float(args.scorer_component_vote_threshold),
        collaboration_policy=str(args.collaboration_policy),
        label_fusion_policy=str(args.label_fusion_policy),
        class_reliability_policy=str(args.class_reliability_policy),
        receiver_class_reliability_policy=str(args.receiver_class_reliability_policy),
        latency_budget_ms=float(args.latency_budget_ms),
        max_event_bytes=float(args.max_event_bytes),
        max_event_latency_ms=float(args.max_event_latency_ms),
        adaptive_gain_min_risk=float(args.adaptive_gain_min_risk),
        adaptive_gain_latency_weight=float(args.adaptive_gain_latency_weight),
        adaptive_gain_bytes_weight=float(args.adaptive_gain_bytes_weight),
        adaptive_gain_disagreement_weight=float(args.adaptive_gain_disagreement_weight),
        rb_capr_utility_min_delta=float(args.rb_capr_utility_min_delta),
        rb_capr_seen_new_balance_weight=float(args.rb_capr_seen_new_balance_weight),
        rb_capr_old_floor_weight=float(args.rb_capr_old_floor_weight),
        rb_capr_unknown_confirm_weight=float(args.rb_capr_unknown_confirm_weight),
        rb_capr_max_avg_rx_target=float(args.rb_capr_max_avg_rx_target),
        seen_new_rescue_enabled=bool(args.seen_new_rescue_enabled),
        seen_new_rescue_risk_scale=float(args.seen_new_rescue_risk_scale),
        seen_new_rescue_min_score=float(args.seen_new_rescue_min_score),
        seen_new_rescue_min_margin=float(args.seen_new_rescue_min_margin),
        seen_new_rescue_min_agreement=float(args.seen_new_rescue_min_agreement),
        conformal_rescue_enabled=bool(args.conformal_rescue_enabled),
        conformal_rescue_min_pvalue=float(args.conformal_rescue_min_pvalue),
        conformal_rescue_risk_scale=float(args.conformal_rescue_risk_scale),
        conformal_rescue_min_agreement=float(args.conformal_rescue_min_agreement),
        class_set_gate_enabled=bool(args.class_set_gate_enabled),
        old_gate_min_receivers=int(args.old_gate_min_receivers),
        old_gate_max_effective_unknown_risk=float(args.old_gate_max_effective_unknown_risk),
        old_gate_max_component_agreement=float(args.old_gate_max_component_agreement),
        old_gate_min_support_density=float(args.old_gate_min_support_density),
        old_gate_max_radius_z=float(args.old_gate_max_radius_z),
        seen_new_gate_min_receivers=int(args.seen_new_gate_min_receivers),
        seen_new_gate_max_effective_unknown_risk=float(args.seen_new_gate_max_effective_unknown_risk),
        seen_new_gate_max_component_agreement=float(args.seen_new_gate_max_component_agreement),
        seen_new_gate_min_support_density=float(args.seen_new_gate_min_support_density),
        seen_new_gate_max_radius_z=float(args.seen_new_gate_max_radius_z),
        candidate_set_min_receivers=int(args.candidate_set_min_receivers),
        candidate_set_min_top1_receivers=int(args.candidate_set_min_top1_receivers),
        candidate_set_min_conformal_pvalue=float(args.candidate_set_min_conformal_pvalue),
        candidate_set_max_label_unknown_risk=float(args.candidate_set_max_label_unknown_risk),
        candidate_set_max_event_unknown_risk=float(args.candidate_set_max_event_unknown_risk),
        candidate_set_max_label_risk_component_agreement=float(
            args.candidate_set_max_label_risk_component_agreement
        ),
        candidate_set_max_label_shell_risk=float(args.candidate_set_max_label_shell_risk),
        candidate_set_shell_reject_risk=float(args.candidate_set_shell_reject_risk),
        candidate_set_event_high_unknown_risk_veto=float(args.candidate_set_event_high_unknown_risk_veto),
        candidate_set_max_label_high_unknown_risk_fraction=float(
            args.candidate_set_max_label_high_unknown_risk_fraction
        ),
        candidate_set_high_unknown_risk_threshold=float(args.candidate_set_high_unknown_risk_threshold),
        candidate_set_min_score_gap=float(args.candidate_set_min_score_gap),
        candidate_set_unknown_reject_risk=float(args.candidate_set_unknown_reject_risk),
        candidate_set_max_receiver_pair_label_disagreement=float(
            args.candidate_set_max_receiver_pair_label_disagreement
        ),
        candidate_set_max_receiver_pair_unknown_risk_range=float(
            args.candidate_set_max_receiver_pair_unknown_risk_range
        ),
        candidate_set_min_label_receiver_class_reliability=float(
            args.candidate_set_min_label_receiver_class_reliability
        ),
        candidate_set_require_label_shell_observed=bool(args.candidate_set_require_label_shell_observed),
        candidate_set_pairguard_mode=str(args.candidate_set_pairguard_mode),
        candidate_set_pairguard_min_event_unknown_risk=float(
            args.candidate_set_pairguard_min_event_unknown_risk
        ),
        candidate_set_pairguard_min_label_unknown_risk=float(
            args.candidate_set_pairguard_min_label_unknown_risk
        ),
        candidate_set_pairguard_min_shell_risk=float(args.candidate_set_pairguard_min_shell_risk),
        candidate_set_pairguard_labels=str(args.candidate_set_pairguard_labels),
        candidate_set_pairguard_receiver_sets=str(args.candidate_set_pairguard_receiver_sets),
        candidate_set_pairguard_action=str(args.candidate_set_pairguard_action),
        candidate_set_pairguard_soft_penalty=float(args.candidate_set_pairguard_soft_penalty),
        candidate_set_pairguard_soft_floor=float(args.candidate_set_pairguard_soft_floor),
        candidate_set_pairguard_soft_min_margin=float(args.candidate_set_pairguard_soft_min_margin),
        candidate_set_pairguard_soft_min_agreement=float(args.candidate_set_pairguard_soft_min_agreement),
        candidate_set_pairguard_soft_min_pvalue=float(args.candidate_set_pairguard_soft_min_pvalue),
        candidate_set_pairguard_soft_min_reliability=float(
            args.candidate_set_pairguard_soft_min_reliability
        ),
        orbit_latency_weight=float(args.orbit_latency_weight),
        orbit_radius_risk_weight=float(args.orbit_radius_risk_weight),
        orbit_staleness_weight=float(args.orbit_staleness_weight),
        orbit_min_trust=float(args.orbit_min_trust),
        orbit_unknown_veto_risk=float(args.orbit_unknown_veto_risk),
        orbit_old_floor_rescue_enabled=bool(args.orbit_old_floor_rescue_enabled),
        orbit_old_floor_max_rank=int(args.orbit_old_floor_max_rank),
        orbit_old_floor_min_receivers=int(args.orbit_old_floor_min_receivers),
        orbit_old_floor_min_pvalue=float(args.orbit_old_floor_min_pvalue),
        orbit_old_floor_min_receiver_class_reliability=float(
            args.orbit_old_floor_min_receiver_class_reliability
        ),
        orbit_old_floor_min_support_density=float(args.orbit_old_floor_min_support_density),
        orbit_old_floor_min_margin=float(args.orbit_old_floor_min_margin),
        orbit_old_floor_max_label_unknown_risk=float(args.orbit_old_floor_max_label_unknown_risk),
        orbit_old_floor_max_event_unknown_risk=float(args.orbit_old_floor_max_event_unknown_risk),
        orbit_old_floor_max_shell_risk=float(args.orbit_old_floor_max_shell_risk),
        orbit_old_floor_max_component_agreement=float(args.orbit_old_floor_max_component_agreement),
        orbit_old_floor_min_trust=float(args.orbit_old_floor_min_trust),
        dual_route_rescue_min_pvalue=float(args.dual_route_rescue_min_pvalue),
        dual_route_rescue_min_receiver_class_reliability=float(
            args.dual_route_rescue_min_receiver_class_reliability
        ),
        dual_route_rescue_max_label_unknown_risk=float(args.dual_route_rescue_max_label_unknown_risk),
        dual_route_rescue_max_shell_risk=float(args.dual_route_rescue_max_shell_risk),
        dual_route_rescue_max_component_agreement=float(args.dual_route_rescue_max_component_agreement),
        dual_route_rescue_max_disagreement=float(args.dual_route_rescue_max_disagreement),
        dual_route_rescue_max_unknown_risk_range=float(args.dual_route_rescue_max_unknown_risk_range),
        dual_route_rescue_max_safety_unknown_risk=float(args.dual_route_rescue_max_safety_unknown_risk),
        threshold_selection_label_scope=str(metadata["threshold_scope"]),
        unknown_query_eval_only=True,
        receiver_selection_policy=str(args.receiver_selection_policy),
        collab_group_policy=str(args.collab_group_policy),
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
        scorer_risk_components=metadata["active_risk_components"],
        include_event_results=bool(args.include_event_results),
    )
    result["feature_npz"] = str(args.feature_npz)
    result["run_command_argv"] = [str(item) for item in sys.argv]
    result["run_cwd"] = str(Path.cwd())
    result["python_executable"] = str(sys.executable)
    result["output_json"] = str(args.output_json)
    result["output_evidence_csv"] = str(args.output_evidence_csv) if args.output_evidence_csv else ""
    result["qknn_metadata"] = metadata
    result["evidence_row_count"] = len(evidence)
    if args.output_evidence_csv:
        output_csv = Path(args.output_evidence_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted(set().union(*(row.keys() for row in evidence)))
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(evidence)
        result["evidence_csv"] = str(output_csv)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_evidence_csv", type=Path, default=None)
    p.add_argument("--collab_counts", default="all")
    p.add_argument(
        "--collab_group_policy",
        default="exact_k",
        choices=["exact_k", "available_up_to_k", "same_max_budget"],
        help=(
            "exact_k evaluates only events observed by at least k receivers; "
            "available_up_to_k treats k as a maximum collaboration budget and includes "
            "events with at least --partial_collab_min_receivers observations; "
            "same_max_budget evaluates every k on the same groups available at the maximum requested budget."
        ),
    )
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070303)
    p.add_argument("--support_quantile", type=float, default=0.05)
    p.add_argument("--proxy_quantile", type=float, default=0.95)
    p.add_argument("--risk_temperature", type=float, default=0.035)
    p.add_argument("--radius_temperature", type=float, default=0.02)
    p.add_argument("--margin_temperature", type=float, default=0.02)
    p.add_argument("--mahalanobis_temperature", type=float, default=0.20)
    p.add_argument("--evt_temperature", type=float, default=0.05)
    p.add_argument("--oldness_temperature", type=float, default=0.05)
    p.add_argument("--radius_quantile", type=float, default=0.95)
    p.add_argument("--margin_quantile", type=float, default=0.05)
    p.add_argument("--score_quantile", type=float, default=0.05)
    p.add_argument("--mahalanobis_quantile", type=float, default=0.95)
    p.add_argument("--evt_tail_quantile", type=float, default=0.80)
    p.add_argument("--oldness_quantile", type=float, default=0.05)
    p.add_argument("--radius_slack", type=float, default=0.02)
    p.add_argument("--margin_slack", type=float, default=0.0)
    p.add_argument("--score_slack", type=float, default=0.0)
    p.add_argument("--mahalanobis_slack", type=float, default=0.0)
    p.add_argument("--evt_tail_slack", type=float, default=0.0)
    p.add_argument("--oldness_slack", type=float, default=0.0)
    p.add_argument("--mahalanobis_variance_floor", type=float, default=1e-4)
    p.add_argument("--evt_min_scale", type=float, default=1e-3)
    p.add_argument("--scenario_aware", action="store_true")
    p.add_argument("--radius_norm", type=float, default=0.0)
    p.add_argument("--old_bias", type=float, default=0.0)
    p.add_argument("--candidate_class_top_m", type=int, default=0)
    p.add_argument("--prototype_score_blend", type=float, default=0.0)
    p.add_argument("--mahalanobis_score_blend", type=float, default=0.0)
    p.add_argument("--mahalanobis_score_temperature", type=float, default=0.20)
    p.add_argument(
        "--prototype_calibration_policy",
        default="none",
        choices=["none", "teen_blend", "teen_separate"],
    )
    p.add_argument("--prototype_calibration_alpha", type=float, default=0.0)
    p.add_argument("--prototype_calibration_top_m", type=int, default=2)
    p.add_argument(
        "--feature_adapter_policy",
        default="none",
        choices=["none", "support_center", "support_bn_affine"],
    )
    p.add_argument("--feature_adapter_strength", type=float, default=0.0)
    p.add_argument("--feature_adapter_variance_floor", type=float, default=1e-4)
    p.add_argument("--candidate_audit_unknown_risk_enabled", action="store_true")
    p.add_argument("--candidate_audit_disagreement_risk", type=float, default=1.0)
    p.add_argument("--candidate_audit_min_gap", type=float, default=0.0)
    p.add_argument("--candidate_audit_gap_risk", type=float, default=0.0)
    p.add_argument("--class_verifier_policy", default="none", choices=["none", "support_quality"])
    p.add_argument("--class_verifier_top_m", type=int, default=0)
    p.add_argument("--class_verifier_pvalue_weight", type=float, default=1.0)
    p.add_argument("--class_verifier_reliability_weight", type=float, default=1.0)
    p.add_argument("--class_verifier_risk_weight", type=float, default=1.0)
    p.add_argument("--support_calibration_mode", default="self", choices=["self", "leave_one_out", "loo"])
    p.add_argument(
        "--score_threshold_combine",
        default="max",
        choices=["max", "qknn_only", "centroid_only", "min", "mean"],
    )
    p.add_argument("--class_score_threshold_enabled", action="store_true")
    p.add_argument("--class_score_threshold_quantile", type=float, default=None)
    p.add_argument("--class_score_threshold_min_support", type=int, default=1)
    p.add_argument("--class_conformal_enabled", action="store_true")
    p.add_argument("--class_conformal_min_support", type=int, default=2)
    p.add_argument("--class_evidence_top_m", type=int, default=0)
    p.add_argument("--virtual_unknown_calibration_enabled", action="store_true")
    p.add_argument("--virtual_unknown_samples_per_class", type=int, default=0)
    p.add_argument("--virtual_unknown_mix_alpha", type=float, default=0.50)
    p.add_argument("--virtual_unknown_noise_scale", type=float, default=0.02)
    p.add_argument("--virtual_unknown_neighbor_count", type=int, default=2)
    p.add_argument("--virtual_unknown_risk_enabled", action="store_true")
    p.add_argument("--virtual_unknown_risk_samples_per_class", type=int, default=2)
    p.add_argument("--virtual_unknown_risk_temperature", type=float, default=0.05)
    p.add_argument("--virtual_unknown_risk_margin", type=float, default=0.0)
    p.add_argument("--class_shell_unknown_risk_enabled", action="store_true")
    p.add_argument("--class_shell_radius_scale", type=float, default=1.25)
    p.add_argument("--class_shell_risk_temperature", type=float, default=0.05)
    p.add_argument("--class_shell_risk_margin", type=float, default=0.0)
    p.add_argument("--unknown_risk_threshold", type=float, default=0.80)
    p.add_argument("--accept_margin_threshold", type=float, default=0.10)
    p.add_argument("--unknown_quantile", type=float, default=0.75)
    p.add_argument(
        "--fusion_policy",
        default="risk_margin",
        choices=[
            "risk_margin",
            "consensus_veto",
            "scorer_cvs",
            "cp_set_cvs",
            "candidate_set_cvs",
            "support_router_cvs",
            "orbit_coproto",
        ],
    )
    p.add_argument(
        "--collaboration_policy",
        default="fixed_k",
        choices=[
            "fixed_k",
            "progressive_budget",
            "adaptive_gain",
            "support_utility",
            "rb_capr_utility",
            "dual_route_cvs",
        ],
    )
    p.add_argument("--consensus_gap_threshold", type=float, default=0.0)
    p.add_argument("--consensus_score_threshold", type=float, default=0.0)
    p.add_argument("--scorer_component_vote_threshold", type=float, default=0.5)
    p.add_argument(
        "--label_fusion_policy",
        default="score_sum",
        choices=["score_sum", "vote_sum", "vote_margin", "weighted_vote_margin", "max_score"],
    )
    p.add_argument(
        "--class_reliability_policy",
        default="none",
        choices=["none", "conformal_margin_risk"],
    )
    p.add_argument(
        "--receiver_class_reliability_policy",
        default="none",
        choices=["none", "support_calibrated"],
    )
    p.add_argument("--latency_budget_ms", type=float, default=0.0)
    p.add_argument("--max_event_bytes", type=float, default=0.0)
    p.add_argument("--max_event_latency_ms", type=float, default=0.0)
    p.add_argument("--adaptive_gain_min_risk", type=float, default=0.80)
    p.add_argument("--adaptive_gain_latency_weight", type=float, default=0.0)
    p.add_argument("--adaptive_gain_bytes_weight", type=float, default=0.0)
    p.add_argument("--adaptive_gain_disagreement_weight", type=float, default=0.5)
    p.add_argument("--rb_capr_utility_min_delta", type=float, default=0.02)
    p.add_argument("--rb_capr_seen_new_balance_weight", type=float, default=0.50)
    p.add_argument("--rb_capr_old_floor_weight", type=float, default=0.35)
    p.add_argument("--rb_capr_unknown_confirm_weight", type=float, default=0.60)
    p.add_argument("--rb_capr_max_avg_rx_target", type=float, default=2.50)
    p.add_argument("--seen_new_rescue_enabled", action="store_true")
    p.add_argument("--seen_new_rescue_risk_scale", type=float, default=1.0)
    p.add_argument("--seen_new_rescue_min_score", type=float, default=0.0)
    p.add_argument("--seen_new_rescue_min_margin", type=float, default=0.0)
    p.add_argument("--seen_new_rescue_min_agreement", type=float, default=0.5)
    p.add_argument("--conformal_rescue_enabled", action="store_true")
    p.add_argument("--conformal_rescue_min_pvalue", type=float, default=0.05)
    p.add_argument("--conformal_rescue_risk_scale", type=float, default=0.5)
    p.add_argument("--conformal_rescue_min_agreement", type=float, default=0.5)
    p.add_argument("--class_set_gate_enabled", action="store_true")
    p.add_argument("--old_gate_min_receivers", type=int, default=1)
    p.add_argument("--old_gate_max_effective_unknown_risk", type=float, default=1.0)
    p.add_argument("--old_gate_max_component_agreement", type=float, default=1.0)
    p.add_argument("--old_gate_min_support_density", type=float, default=0.0)
    p.add_argument("--old_gate_max_radius_z", type=float, default=1.0e12)
    p.add_argument("--seen_new_gate_min_receivers", type=int, default=1)
    p.add_argument("--seen_new_gate_max_effective_unknown_risk", type=float, default=1.0)
    p.add_argument("--seen_new_gate_max_component_agreement", type=float, default=1.0)
    p.add_argument("--seen_new_gate_min_support_density", type=float, default=0.0)
    p.add_argument("--seen_new_gate_max_radius_z", type=float, default=1.0e12)
    p.add_argument("--candidate_set_min_receivers", type=int, default=2)
    p.add_argument("--candidate_set_min_top1_receivers", type=int, default=0)
    p.add_argument("--candidate_set_min_conformal_pvalue", type=float, default=0.0)
    p.add_argument("--candidate_set_max_label_unknown_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_max_event_unknown_risk", type=float, default=0.95)
    p.add_argument("--candidate_set_max_label_risk_component_agreement", type=float, default=1.0)
    p.add_argument("--candidate_set_max_label_shell_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_shell_reject_risk", type=float, default=1.0e12)
    p.add_argument("--candidate_set_event_high_unknown_risk_veto", type=float, default=1.0e12)
    p.add_argument("--candidate_set_max_label_high_unknown_risk_fraction", type=float, default=1.0)
    p.add_argument("--candidate_set_high_unknown_risk_threshold", type=float, default=0.80)
    p.add_argument("--candidate_set_min_score_gap", type=float, default=0.0)
    p.add_argument("--candidate_set_unknown_reject_risk", type=float, default=0.80)
    p.add_argument("--candidate_set_max_receiver_pair_label_disagreement", type=float, default=1.0)
    p.add_argument("--candidate_set_max_receiver_pair_unknown_risk_range", type=float, default=1.0)
    p.add_argument("--candidate_set_min_label_receiver_class_reliability", type=float, default=0.0)
    p.add_argument("--candidate_set_require_label_shell_observed", action="store_true")
    p.add_argument(
        "--candidate_set_pairguard_mode",
        default="accept_gate",
        choices=["accept_gate", "boundary_veto", "support_calibrated"],
    )
    p.add_argument("--candidate_set_pairguard_min_event_unknown_risk", type=float, default=0.80)
    p.add_argument("--candidate_set_pairguard_min_label_unknown_risk", type=float, default=0.80)
    p.add_argument("--candidate_set_pairguard_min_shell_risk", type=float, default=0.90)
    p.add_argument("--candidate_set_pairguard_labels", default="")
    p.add_argument("--candidate_set_pairguard_receiver_sets", default="")
    p.add_argument(
        "--candidate_set_pairguard_action",
        default="veto",
        choices=["veto", "request_more", "soft_penalty"],
    )
    p.add_argument("--candidate_set_pairguard_soft_penalty", type=float, default=0.0)
    p.add_argument("--candidate_set_pairguard_soft_floor", type=float, default=0.0)
    p.add_argument("--candidate_set_pairguard_soft_min_margin", type=float, default=0.0)
    p.add_argument("--candidate_set_pairguard_soft_min_agreement", type=float, default=0.0)
    p.add_argument("--candidate_set_pairguard_soft_min_pvalue", type=float, default=0.0)
    p.add_argument("--candidate_set_pairguard_soft_min_reliability", type=float, default=0.0)
    p.add_argument("--orbit_latency_weight", type=float, default=0.0)
    p.add_argument("--orbit_radius_risk_weight", type=float, default=0.5)
    p.add_argument("--orbit_staleness_weight", type=float, default=0.0)
    p.add_argument("--orbit_min_trust", type=float, default=0.10)
    p.add_argument("--orbit_unknown_veto_risk", type=float, default=0.80)
    p.add_argument("--orbit_old_floor_rescue_enabled", action="store_true")
    p.add_argument("--orbit_old_floor_max_rank", type=int, default=3)
    p.add_argument("--orbit_old_floor_min_receivers", type=int, default=2)
    p.add_argument("--orbit_old_floor_min_pvalue", type=float, default=0.25)
    p.add_argument("--orbit_old_floor_min_receiver_class_reliability", type=float, default=0.30)
    p.add_argument("--orbit_old_floor_min_support_density", type=float, default=0.20)
    p.add_argument("--orbit_old_floor_min_margin", type=float, default=0.03)
    p.add_argument("--orbit_old_floor_max_label_unknown_risk", type=float, default=0.55)
    p.add_argument("--orbit_old_floor_max_event_unknown_risk", type=float, default=0.75)
    p.add_argument("--orbit_old_floor_max_shell_risk", type=float, default=0.65)
    p.add_argument("--orbit_old_floor_max_component_agreement", type=float, default=0.50)
    p.add_argument("--orbit_old_floor_min_trust", type=float, default=0.0)
    p.add_argument("--include_event_results", action="store_true")
    p.add_argument("--dual_route_rescue_min_pvalue", type=float, default=0.75)
    p.add_argument("--dual_route_rescue_min_receiver_class_reliability", type=float, default=0.75)
    p.add_argument("--dual_route_rescue_max_label_unknown_risk", type=float, default=0.60)
    p.add_argument("--dual_route_rescue_max_shell_risk", type=float, default=0.80)
    p.add_argument("--dual_route_rescue_max_component_agreement", type=float, default=0.34)
    p.add_argument("--dual_route_rescue_max_disagreement", type=float, default=0.50)
    p.add_argument("--dual_route_rescue_max_unknown_risk_range", type=float, default=0.50)
    p.add_argument("--dual_route_rescue_max_safety_unknown_risk", type=float, default=0.80)
    p.add_argument("--evidence_packet_bytes", type=float, default=40.0)
    p.add_argument(
        "--receiver_reliability_policy",
        default="deployment_prior",
        choices=["deployment_prior", "support_density", "margin_density"],
    )
    p.add_argument(
        "--receiver_selection_policy",
        default="fixed_receiver_order",
        choices=["fixed_receiver_order", "reliability_prior", "support_quality_prior"],
    )
    p.add_argument(
        "--support_selection_policy",
        default="stable_first",
        choices=["stable_first", "centroid", "scenario_diverse"],
    )
    p.add_argument(
        "--unknown_gate_mode",
        default="score",
        choices=[
            "score",
            "radius",
            "margin",
            "mahalanobis",
            "evt",
            "oldness",
            "support_envelope",
            "support_envelope_mahalanobis",
            "support_envelope_evt",
            "support_envelope_oldness",
            "support_envelope_full",
            "support_envelope_consensus",
        ],
    )
    p.add_argument(
        "--event_alignment_policy",
        default="strict_event_key",
        choices=["strict_event_key", "receiver_domain_ranked"],
        help=(
            "strict_event_key requires shared role+tx+day+sig+scenario across receivers. "
            "receiver_domain_ranked is an explicit dataset diagnostic when no same-event key exists."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_evaluation(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["receiver_count", "group_count", "evidence_row_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
