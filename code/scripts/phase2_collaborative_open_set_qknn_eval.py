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
    margin_threshold: float
    score_threshold: float


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


def _centroid_scores(memory: QknnMemory, query_features: np.ndarray) -> np.ndarray:
    return _normalize_rows(query_features) @ memory.centroids.T


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


def build_qknn_memory(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    old_labels: set[str],
    support_scenarios: Sequence[str] | None = None,
    radius_quantile: float = 0.95,
    margin_quantile: float = 0.05,
    score_quantile: float = 0.05,
    radius_slack: float = 0.02,
    margin_slack: float = 0.0,
    score_slack: float = 0.0,
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
    exclude_support_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = _normalize_rows(query_features)
    support = _normalize_rows(memory.qfeatures.astype(np.float32) / float(memory.scale))
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
    all_support_mask = np.ones(memory.labels.shape[0], dtype=bool)
    for row_i in range(scores.shape[0]):
        support_mask = all_support_mask.copy()
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
            score = float(scores[row_i, j])
            per_label[str(memory.labels[j])] += max(0.0, score)
        ranked = sorted(per_label.items(), key=lambda item: (item[1], item[0]), reverse=True)
        best_label, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        out_labels.append(best_label)
        out_scores.append(float(best_score / max(row_k, 1)))
        out_margins.append(float((best_score - second) / max(row_k, 1)))
    return np.asarray(out_labels, dtype=object), np.asarray(out_scores, dtype=np.float64), np.asarray(out_margins, dtype=np.float64)


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
    support_calibration_mode: str = "self",
) -> tuple[float, str]:
    calibration_mode = str(support_calibration_mode or "self").strip().lower()
    if calibration_mode not in {"self", "leave_one_out", "loo"}:
        raise ValueError("support_calibration_mode must be self or leave_one_out")
    exclude = range(int(support_features.shape[0])) if calibration_mode in {"leave_one_out", "loo"} else None
    _, support_scores, _ = qknn_scores(
        memory,
        support_features,
        top_k=top_k,
        query_scenarios=support_scenarios,
        scenario_aware=scenario_aware,
        radius_norm=radius_norm,
        old_bias=old_bias,
        exclude_support_indices=exclude,
    )
    threshold = float(np.quantile(support_scores, float(support_quantile))) if support_scores.size else 0.0
    scope = "support_known_only"
    if proxy_features is not None and int(proxy_features.shape[0]) > 0:
        _, proxy_scores, _ = qknn_scores(
            memory,
            proxy_features,
            top_k=top_k,
            query_scenarios=proxy_scenarios,
            scenario_aware=scenario_aware and proxy_scenarios is not None,
            radius_norm=radius_norm,
            old_bias=old_bias,
        )
        if proxy_scores.size:
            threshold = max(threshold, float(np.quantile(proxy_scores, float(proxy_quantile))))
            scope = "source_only"
    return threshold, scope


def _unknown_risk(scores: np.ndarray, threshold: float, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = (np.asarray(scores, dtype=np.float64) - float(threshold)) / temp
    known_prob = 1.0 / (1.0 + np.exp(-z))
    return np.clip(1.0 - known_prob, 0.0, 1.0)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = str(gate_mode or "score").strip().lower()
    score_risk = _unknown_risk(known_scores, score_threshold, temperature=temperature)
    if mode == "score":
        zeros = np.zeros_like(score_risk)
        return score_risk, zeros, zeros, zeros
    centroid_scores = _centroid_scores(memory, query_features)
    class_to_pos = {str(label): int(i) for i, label in enumerate(memory.centroid_labels.tolist())}
    radius_risks = []
    radius_values = []
    for row_i, label in enumerate(predicted_labels):
        label = str(label)
        pos = class_to_pos.get(label)
        if pos is None:
            radius_values.append(1.0)
            radius_risks.append(1.0)
            continue
        radius = float(1.0 - centroid_scores[row_i, pos])
        threshold = float(memory.class_radius_thresholds.get(label, 1.0))
        z = (radius - threshold) / max(float(radius_temperature), 1e-6)
        radius_values.append(radius)
        radius_risks.append(float(1.0 / (1.0 + np.exp(-z))))
    radius_risk = np.asarray(radius_risks, dtype=np.float64)
    margin_t = float(memory.margin_threshold)
    margin_z = (margin_t - np.asarray(known_margins, dtype=np.float64)) / max(float(margin_temperature), 1e-6)
    margin_risk = 1.0 / (1.0 + np.exp(-margin_z))
    if mode == "support_envelope":
        risk = np.maximum.reduce([score_risk, radius_risk, margin_risk])
    elif mode == "radius":
        risk = np.maximum(score_risk, radius_risk)
    elif mode == "margin":
        risk = np.maximum(score_risk, margin_risk)
    else:
        raise ValueError("unknown_gate_mode must be score, radius, margin, or support_envelope")
    return np.clip(risk, 0.0, 1.0), radius_risk, margin_risk, np.asarray(radius_values, dtype=np.float64)


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
    radius_slack: float = 0.02,
    margin_slack: float = 0.0,
    score_slack: float = 0.0,
    radius_temperature: float = 0.02,
    margin_temperature: float = 0.02,
    scenario_aware: bool = False,
    radius_norm: float = 0.0,
    old_bias: float = 0.0,
    support_calibration_mode: str = "self",
    score_threshold_combine: str = "max",
    evidence_packet_bytes: float = 40.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
    old_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_old")[0].tolist()})
    new_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_new")[0].tolist()})
    unknown_labels = sorted({str(tx_ids[i]) for i in np.where(roles == UNKNOWN_ROLE)[0].tolist()})
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
    evidence: list[dict[str, Any]] = []
    receiver_query: dict[str, dict[str, list[int]]] = {}
    threshold_scope = "support_known_only"
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
        memory = build_qknn_memory(
            features[np.asarray(support_indices, dtype=int)],
            support_labels,
            old_labels=set(old_labels),
            support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
            radius_quantile=radius_quantile,
            margin_quantile=margin_quantile,
            score_quantile=score_quantile,
            radius_slack=radius_slack,
            margin_slack=margin_slack,
            score_slack=score_slack,
        )
        proxy_features = features[proxy_idx] if proxy_idx.size else None
        threshold, scope = _threshold_from_calibration(
            memory,
            features[np.asarray(support_indices, dtype=int)],
            proxy_features,
            top_k=qknn_k,
            support_quantile=support_quantile,
            proxy_quantile=proxy_quantile,
            support_scenarios=[_scenario_of(payload, int(i)) for i in support_indices],
            proxy_scenarios=[_scenario_of(payload, int(i)) for i in proxy_idx.tolist()] if proxy_idx.size else None,
            scenario_aware=bool(scenario_aware),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            support_calibration_mode=str(support_calibration_mode),
        )
        threshold_scope = scope if scope == "source_only" else threshold_scope
        receiver_memories[rx] = memory
        receiver_thresholds[rx] = threshold

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
                all_event_ids = sorted(set().union(*(set(by_rx_key[rx]) for rx in target_receivers)))
                event_groups = [
                    (event_id, {rx: by_rx_key[rx][event_id] for rx in target_receivers if event_id in by_rx_key[rx]})
                    for event_id in all_event_ids
                    if any(event_id in by_rx_key[rx] for rx in target_receivers)
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
                    pred, score, margin = qknn_scores(
                        memory,
                        features[[idx]],
                        top_k=qknn_k,
                        query_scenarios=[_scenario_of(payload, idx)],
                        scenario_aware=bool(scenario_aware),
                        radius_norm=float(radius_norm),
                        old_bias=float(old_bias),
                    )
                    risk, radius_risk, margin_risk, class_radius = _combined_unknown_risk(
                        memory,
                        features[[idx]],
                        pred,
                        score,
                        margin,
                        _combine_score_threshold(
                            receiver_thresholds[rx],
                            memory.score_threshold,
                            str(score_threshold_combine),
                        ),
                        temperature=risk_temperature,
                        gate_mode=unknown_gate_mode,
                        radius_temperature=radius_temperature,
                        margin_temperature=margin_temperature,
                    )
                    evidence.append(
                        {
                            "event_id": event_id,
                            "receiver_id": rx,
                            "role": role_name,
                            "true_label": "__unknown__" if role_name == "unknown" else str(tx_ids[idx]),
                            "predicted_label": str(pred[0]),
                            "known_score": float(score[0]),
                            "known_margin": float(margin[0]),
                            "unknown_risk": float(risk[0]),
                            "radius_risk": float(radius_risk[0]),
                            "margin_risk": float(margin_risk[0]),
                            "class_radius": float(class_radius[0]),
                            "reliability": 1.0,
                            "reliability_source": "deployment_prior",
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
        "threshold_scope": threshold_scope,
        "support_calibration_mode": str(support_calibration_mode),
        "score_threshold_combine": str(score_threshold_combine),
        "support_selection_policy": str(support_selection_policy),
        "unknown_gate_mode": str(unknown_gate_mode),
        "scenario_aware": bool(scenario_aware),
        "radius_norm": float(radius_norm),
        "old_bias": float(old_bias),
        "radius_quantile": float(radius_quantile),
        "margin_quantile": float(margin_quantile),
        "score_quantile": float(score_quantile),
        "radius_slack": float(radius_slack),
        "margin_slack": float(margin_slack),
        "score_slack": float(score_slack),
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
        radius_slack=float(args.radius_slack),
        margin_slack=float(args.margin_slack),
        score_slack=float(args.score_slack),
        radius_temperature=float(args.radius_temperature),
        margin_temperature=float(args.margin_temperature),
        scenario_aware=bool(args.scenario_aware),
        radius_norm=float(args.radius_norm),
        old_bias=float(args.old_bias),
        support_calibration_mode=str(args.support_calibration_mode),
        score_threshold_combine=str(args.score_threshold_combine),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
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
        latency_budget_ms=float(args.latency_budget_ms),
        threshold_selection_label_scope=str(metadata["threshold_scope"]),
        unknown_query_eval_only=True,
        receiver_selection_policy=str(args.receiver_selection_policy),
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
    )
    result["feature_npz"] = str(args.feature_npz)
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
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070303)
    p.add_argument("--support_quantile", type=float, default=0.05)
    p.add_argument("--proxy_quantile", type=float, default=0.95)
    p.add_argument("--risk_temperature", type=float, default=0.035)
    p.add_argument("--radius_temperature", type=float, default=0.02)
    p.add_argument("--margin_temperature", type=float, default=0.02)
    p.add_argument("--radius_quantile", type=float, default=0.95)
    p.add_argument("--margin_quantile", type=float, default=0.05)
    p.add_argument("--score_quantile", type=float, default=0.05)
    p.add_argument("--radius_slack", type=float, default=0.02)
    p.add_argument("--margin_slack", type=float, default=0.0)
    p.add_argument("--score_slack", type=float, default=0.0)
    p.add_argument("--scenario_aware", action="store_true")
    p.add_argument("--radius_norm", type=float, default=0.0)
    p.add_argument("--old_bias", type=float, default=0.0)
    p.add_argument("--support_calibration_mode", default="self", choices=["self", "leave_one_out", "loo"])
    p.add_argument(
        "--score_threshold_combine",
        default="max",
        choices=["max", "qknn_only", "centroid_only", "min", "mean"],
    )
    p.add_argument("--unknown_risk_threshold", type=float, default=0.80)
    p.add_argument("--accept_margin_threshold", type=float, default=0.10)
    p.add_argument("--unknown_quantile", type=float, default=0.75)
    p.add_argument("--fusion_policy", default="risk_margin", choices=["risk_margin", "consensus_veto", "scorer_cvs"])
    p.add_argument("--consensus_gap_threshold", type=float, default=0.0)
    p.add_argument("--consensus_score_threshold", type=float, default=0.0)
    p.add_argument("--latency_budget_ms", type=float, default=0.0)
    p.add_argument("--evidence_packet_bytes", type=float, default=40.0)
    p.add_argument("--receiver_selection_policy", default="fixed_receiver_order")
    p.add_argument(
        "--support_selection_policy",
        default="stable_first",
        choices=["stable_first", "centroid", "scenario_diverse"],
    )
    p.add_argument(
        "--unknown_gate_mode",
        default="score",
        choices=["score", "radius", "margin", "support_envelope"],
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
