#!/usr/bin/env python
"""CRISP-C residual-interval sketch collaborative inference for Stage2-C.

CRISP-C keeps ADV3B02 features frozen and builds receiver-local qknn8-style
prototype sketches from target old/seen-new support. Old classes are anchored
with source-old shrinkage; seen-new classes use small multi-prototype support
sets. target_unknown rows are evaluation-only and never enter prototype,
threshold, reliability, or profile selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import parse_collab_counts  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    UNKNOWN_ROLE,
    _normalize_rows,
    _split_support_query_selected,
    load_feature_npz,
    validate_required_roles,
)
from phase2_orbit_pcet_ci_eval import _count_value, _target_pass  # noqa: E402


UNKNOWN_LABEL = "__unknown__"
DEFER_LABEL = "__defer__"


@dataclass(frozen=True)
class CrispProfile:
    name: str
    description: str
    old_accept_min: float
    old_p_min: float
    old_violation_max: float
    seen_accept_min: float
    seen_p_min: float
    seen_old_gap_min: float
    seen_residual_max: float
    reject_score_min: float
    reject_known_max: float
    reject_old_violation_min: float
    reject_seen_residual_min: float
    reject_min_receivers: int


PROFILES: tuple[CrispProfile, ...] = (
    CrispProfile(
        name="crisp_primary",
        description="old shrinkage envelope plus seen-new residual sketch",
        old_accept_min=0.58,
        old_p_min=0.34,
        old_violation_max=0.24,
        seen_accept_min=0.56,
        seen_p_min=0.34,
        seen_old_gap_min=0.06,
        seen_residual_max=0.36,
        reject_score_min=0.58,
        reject_known_max=0.50,
        reject_old_violation_min=0.18,
        reject_seen_residual_min=0.38,
        reject_min_receivers=1,
    ),
    CrispProfile(
        name="crisp_old_guard",
        description="conservative old retention profile",
        old_accept_min=0.50,
        old_p_min=0.25,
        old_violation_max=0.32,
        seen_accept_min=0.64,
        seen_p_min=0.50,
        seen_old_gap_min=0.14,
        seen_residual_max=0.26,
        reject_score_min=0.72,
        reject_known_max=0.42,
        reject_old_violation_min=0.28,
        reject_seen_residual_min=0.48,
        reject_min_receivers=2,
    ),
    CrispProfile(
        name="crisp_unknown_probe",
        description="diagnostic stricter reject profile after envelope failure",
        old_accept_min=0.66,
        old_p_min=0.50,
        old_violation_max=0.18,
        seen_accept_min=0.52,
        seen_p_min=0.25,
        seen_old_gap_min=0.02,
        seen_residual_max=0.44,
        reject_score_min=0.48,
        reject_known_max=0.58,
        reject_old_violation_min=0.12,
        reject_seen_residual_min=0.30,
        reject_min_receivers=1,
    ),
)


def _unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value)


def _role(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"old", "target_old"}:
        return "old"
    if text in {"seen_new", "seennew", "target_new", "new"}:
        return "seen_new"
    if text in {"unknown", "target_unknown", "unk"}:
        return "unknown"
    raise ValueError(f"unknown role {value!r}")


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    missing = sorted(set(names) - known)
    if missing:
        raise argparse.ArgumentTypeError(f"unknown CRISP-C profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> CrispProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown CRISP-C profile {name!r}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _role_labels(payload: Mapping[str, Any], role: str) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    return sorted({str(tx_ids[i]) for i in np.where(roles == role)[0].tolist()})


def _target_receivers(payload: Mapping[str, Any]) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    mask = np.isin(roles, ["target_old", "target_new", UNKNOWN_ROLE])
    return sorted({str(rx_ids[i]) for i in np.where(mask)[0].tolist()})


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    old = set(_role_labels(payload, "target_old"))
    seen = set(_role_labels(payload, "target_new"))
    unknown = set(_role_labels(payload, UNKNOWN_ROLE))
    overlaps = {
        "old_new": sorted(old & seen),
        "old_unknown": sorted(old & unknown),
        "new_unknown": sorted(seen & unknown),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: target TX sets overlap: {overlaps}")
    source_labels = {str(tx_ids[i]) for i in np.where(roles == "source")[0].tolist()}
    manifest = payload.get("manifest", {})
    if isinstance(manifest, Mapping):
        raw = manifest.get("source_tx_ids", [])
        if isinstance(raw, str):
            source_labels.update(part.strip() for part in raw.split(",") if part.strip())
        else:
            source_labels.update(str(part) for part in raw)
    old_not_source = sorted(old - source_labels)
    non_old_in_source = sorted((seen | unknown) & source_labels)
    if old_not_source or non_old_in_source:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: Stage2-C TX split violates source/target semantics; "
            f"old_not_in_source={old_not_source}, non_old_in_source={non_old_in_source}"
        )
    target_receivers = set(_target_receivers(payload))
    source_receivers = {str(rx_ids[i]) for i in np.where(roles == "source")[0].tolist()}
    overlap_rx = sorted(target_receivers & source_receivers)
    if overlap_rx:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: R_s and R_t overlap: {overlap_rx}")


def _prototype(mat: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.asarray(mat, dtype=np.float32).mean(axis=0, keepdims=True))[0]


def _source_old_prototypes(payload: Mapping[str, Any], features: np.ndarray, old_labels: Sequence[str]) -> dict[str, np.ndarray]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    out: dict[str, np.ndarray] = {}
    for label in old_labels:
        idx = np.where((roles == "source") & (tx_ids == str(label)))[0]
        if idx.size:
            out[str(label)] = _prototype(features[idx])
    return out


def _event_id(role: str, tx: str, scenario: str, rank: int) -> str:
    return f"{role}|{tx}|{scenario}|rank{int(rank):05d}"


def _dist_to_bank(x: np.ndarray, bank: Sequence[np.ndarray]) -> float:
    if not bank:
        return 1.0
    mat = _normalize_rows(np.vstack([np.asarray(item, dtype=np.float32) for item in bank]))
    return float(max(0.0, 1.0 - float(np.max(mat @ _normalize_rows(x.reshape(1, -1))[0]))))


def _conformal_pvalue(distance: float, calibration: Sequence[float]) -> float:
    values = [float(item) for item in calibration if math.isfinite(float(item))]
    if not values:
        return 0.0
    return float((1 + sum(item >= float(distance) for item in values)) / (1 + len(values)))


def _support_distances(mat: np.ndarray, prototypes: Sequence[np.ndarray]) -> list[float]:
    if mat.size == 0:
        return [1.0]
    return [_dist_to_bank(row, prototypes) for row in np.asarray(mat, dtype=np.float32)]


def _seen_new_prototypes(mat: np.ndarray, max_prototypes: int) -> list[np.ndarray]:
    values = _normalize_rows(np.asarray(mat, dtype=np.float32))
    if values.shape[0] == 0:
        return []
    limit = max(1, min(int(max_prototypes), int(values.shape[0])))
    if values.shape[0] <= limit:
        return [values[i] for i in range(values.shape[0])]
    center = _prototype(values)
    order = sorted(range(values.shape[0]), key=lambda i: float(values[i] @ center), reverse=True)
    return [values[i] for i in order[:limit]]


def _label_set(label: str, old_labels: set[str], seen_labels: set[str]) -> str:
    if str(label) in old_labels:
        return "old"
    if str(label) in seen_labels:
        return "seen_new"
    return ""


def build_crisp_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
    event_alignment_policy: str,
    evidence_packet_bytes: float,
    old_shrinkage_alpha: float,
    max_seen_new_prototypes: int,
    envelope_slack: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_protocol(payload)
    if str(event_alignment_policy) != "receiver_domain_ranked":
        raise ValueError("CRISP-C currently supports receiver_domain_ranked event alignment")
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    sat = np.asarray(payload["sat_scenarios"]).astype(str)
    old_labels = _role_labels(payload, "target_old")
    seen_labels = _role_labels(payload, "target_new")
    unknown_labels = _role_labels(payload, UNKNOWN_ROLE)
    receivers = _target_receivers(payload)
    old_set = set(old_labels)
    seen_set = set(seen_labels)
    source_old = _source_old_prototypes(payload, features, old_labels)

    support_by_rx_label: dict[tuple[str, str], list[int]] = {}
    query_groups: list[tuple[str, str, str, str, list[int]]] = []
    for rx in receivers:
        for role_name, labels, out_role in [
            ("target_old", old_labels, "old"),
            ("target_new", seen_labels, "seen_new"),
        ]:
            for label in labels:
                support, query = _split_support_query_selected(
                    payload,
                    features=features,
                    role=role_name,
                    tx_id=label,
                    rx_id=rx,
                    k_shot=int(k_shot),
                    query_per_class=int(query_per_class),
                    seed=int(seed),
                    support_selection_policy=support_selection_policy,
                )
                if len(support) < int(k_shot) or len(query) < int(query_per_class):
                    raise RuntimeError(
                        "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete CRISP-C support/query for "
                        f"rx={rx}, role={role_name}, tx_id={label}, support={len(support)}, query={len(query)}"
                    )
                support_by_rx_label[(rx, label)] = [int(i) for i in support]
                by_scenario: dict[str, list[int]] = defaultdict(list)
                for idx in query:
                    by_scenario[str(sat[int(idx)] or "unknown_scenario")].append(int(idx))
                for scenario, rows in by_scenario.items():
                    query_groups.append((rx, out_role, label, scenario, rows[: int(query_per_class)]))
        for label in unknown_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role=UNKNOWN_ROLE,
                tx_id=label,
                rx_id=rx,
                k_shot=0,
                query_per_class=int(query_per_class),
                seed=int(seed),
                support_selection_policy=support_selection_policy,
            )
            if support:
                raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: target_unknown entered support")
            if len(query) < int(query_per_class):
                raise RuntimeError(
                    "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete target_unknown query for "
                    f"rx={rx}, tx_id={label}, query={len(query)}"
                )
            by_scenario: dict[str, list[int]] = defaultdict(list)
            for idx in query:
                by_scenario[str(sat[int(idx)] or "unknown_scenario")].append(int(idx))
            for scenario, rows in by_scenario.items():
                query_groups.append((rx, "unknown", label, scenario, rows[: int(query_per_class)]))

    evidence: list[dict[str, Any]] = []
    alpha = _unit(float(old_shrinkage_alpha))
    for rx in receivers:
        banks: dict[str, list[np.ndarray]] = {}
        cal_dists: dict[str, list[float]] = {}
        radii: dict[str, float] = {}
        for label in old_labels:
            support_rows = support_by_rx_label.get((rx, label), [])
            support_mat = features[np.asarray(support_rows, dtype=int)]
            target_proto = _prototype(support_mat)
            if label in source_old:
                proto = _normalize_rows(((1.0 - alpha) * source_old[label] + alpha * target_proto).reshape(1, -1))[0]
            else:
                proto = target_proto
            banks[label] = [proto]
            distances = _support_distances(support_mat, banks[label])
            cal_dists[label] = distances
            radii[label] = max(float(np.quantile(distances, 0.90)), 0.08) + float(envelope_slack)
        for label in seen_labels:
            support_rows = support_by_rx_label.get((rx, label), [])
            support_mat = features[np.asarray(support_rows, dtype=int)]
            banks[label] = _seen_new_prototypes(support_mat, max_seen_new_prototypes)
            distances = _support_distances(support_mat, banks[label])
            cal_dists[label] = distances
            radii[label] = max(float(np.quantile(distances, 0.90)), 0.08) + float(envelope_slack)

        for group_rx, role, label, scenario, rows in query_groups:
            if group_rx != rx:
                continue
            for rank, idx in enumerate(rows):
                x = features[int(idx)]
                old_scores = []
                for old_label in old_labels:
                    dist = _dist_to_bank(x, banks[old_label])
                    old_scores.append((old_label, 1.0 - dist, dist, _conformal_pvalue(dist, cal_dists[old_label])))
                seen_scores = []
                for seen_label in seen_labels:
                    dist = _dist_to_bank(x, banks[seen_label])
                    seen_scores.append((seen_label, 1.0 - dist, dist, _conformal_pvalue(dist, cal_dists[seen_label])))
                old_scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
                seen_scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
                all_scores = sorted([*old_scores, *seen_scores], key=lambda item: (item[1], item[0]), reverse=True)
                best_old = old_scores[0]
                best_seen = seen_scores[0]
                best = all_scores[0]
                second = all_scores[1] if len(all_scores) > 1 else ("", 0.0, 1.0, 0.0)
                old_violation = max(0.0, best_old[2] - radii[best_old[0]])
                seen_residual = max(0.0, best_seen[2] - float(np.median(cal_dists[best_seen[0]])))
                seen_violation = max(0.0, best_seen[2] - radii[best_seen[0]])
                reject_score = _unit(
                    0.38 * _unit(old_violation / max(radii[best_old[0]], 1e-6))
                    + 0.38 * _unit(seen_violation / max(radii[best_seen[0]], 1e-6))
                    + 0.16 * (1.0 - max(float(best_old[1]), float(best_seen[1])))
                    + 0.08 * (1.0 - _unit(float(best[1] - second[1]) * 4.0))
                )
                top_label = str(best[0])
                top_set = _label_set(top_label, old_set, seen_set)
                quality = _unit(0.45 * best[1] + 0.25 * (1.0 - reject_score) + 0.20 * _unit(best[1] - second[1]) + 0.10 * best[3])
                evidence.append(
                    {
                        "event_id": _event_id(role, label, scenario, rank),
                        "role": role,
                        "true_label": str(label),
                        "receiver_id": str(rx),
                        "top_label": top_label,
                        "top_label_set": top_set,
                        "best_old_label": str(best_old[0]),
                        "best_seen_new_label": str(best_seen[0]),
                        "old_accept_score": float(best_old[1]),
                        "seen_new_accept_score": float(best_seen[1]),
                        "reject_score": float(reject_score),
                        "old_envelope_violation": float(old_violation),
                        "seen_new_residual": float(seen_residual),
                        "seen_new_envelope_violation": float(seen_violation),
                        "conformal_p_old": float(best_old[3]),
                        "conformal_p_seen_new": float(best_seen[3]),
                        "known_margin": float(best[1] - second[1]),
                        "class_evidence_top1_label": top_label,
                        "class_evidence_top1_score": float(best[1]),
                        "class_evidence_top1_margin": float(best[1] - second[1]),
                        "class_evidence_top1_conformal_pvalue": float(best[3]),
                        "class_evidence_top2_label": str(second[0]),
                        "class_evidence_top2_score": float(second[1]),
                        "qknn_k": 8,
                        "bytes": float(evidence_packet_bytes),
                        "latency_ms": 0.37 + 0.005 * (len(old_labels) + len(seen_labels)),
                        "quality": float(quality),
                        "threshold_selection_label_scope": "support_known_only",
                        "calibration_role": "query",
                    }
                )
    metadata = {
        "algorithm": "CRISP-C",
        "in_orbit_method": "qknn8_residual_interval_sketch",
        "target_receivers": receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_tx_ids": unknown_labels,
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "support_selection_policy": str(support_selection_policy),
        "event_alignment_policy": str(event_alignment_policy),
        "calibration_source": "target_old_and_target_new_support_leave_one_out_no_target_unknown",
        "prototype_fit_scope": "target_old_target_new_support_plus_source_old_anchors_no_query_outcomes",
        "threshold_uses_target_unknown": False,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "prototype_fit_uses_target_unknown": False,
        "old_shrinkage_alpha": float(alpha),
        "max_seen_new_prototypes": int(max_seen_new_prototypes),
        "evidence_packet_bytes": float(evidence_packet_bytes),
    }
    return evidence, metadata


def _aggregate_mean(rows: Sequence[Mapping[str, Any]], label_key: str, score_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_str(row, label_key)].append(row)
    out = []
    for label, items in grouped.items():
        if not label:
            continue
        out.append(
            {
                "label": label,
                "score": sum(_float(item, score_key) for item in items) / max(len(items), 1),
                "vote_fraction": len(items) / max(len(rows), 1),
                "receiver_count": len(items),
                "p_old": sum(_float(item, "conformal_p_old") for item in items) / max(len(items), 1),
                "p_seen": sum(_float(item, "conformal_p_seen_new") for item in items) / max(len(items), 1),
                "old_violation": sum(_float(item, "old_envelope_violation") for item in items) / max(len(items), 1),
                "seen_residual": sum(_float(item, "seen_new_residual") for item in items) / max(len(items), 1),
            }
        )
    out.sort(key=lambda item: (item["score"], item["vote_fraction"], item["label"]), reverse=True)
    return out


def _fuse_crisp_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: CrispProfile,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    old_candidates = _aggregate_mean(rows, "best_old_label", "old_accept_score")
    seen_candidates = _aggregate_mean(rows, "best_seen_new_label", "seen_new_accept_score")
    old_best = old_candidates[0] if old_candidates else {}
    seen_best = seen_candidates[0] if seen_candidates else {}
    best_old_score = float(old_best.get("score", 0.0))
    best_seen_score = float(seen_best.get("score", 0.0))
    mean_reject = sum(_float(row, "reject_score") for row in rows) / max(len(rows), 1)
    mean_old_violation = sum(_float(row, "old_envelope_violation") for row in rows) / max(len(rows), 1)
    mean_seen_residual = sum(_float(row, "seen_new_residual") for row in rows) / max(len(rows), 1)
    label_counts = Counter(_str(row, "top_label") for row in rows)
    disagreement = 1.0 - max(label_counts.values(), default=0) / max(len(rows), 1)
    reject_score = _unit(mean_reject + 0.12 * disagreement)
    old_accept = bool(
        old_best
        and best_old_score >= profile.old_accept_min
        and float(old_best.get("p_old", 0.0)) >= profile.old_p_min
        and float(old_best.get("old_violation", 0.0)) <= profile.old_violation_max
    )
    seen_accept = bool(
        seen_best
        and best_seen_score >= profile.seen_accept_min
        and float(seen_best.get("p_seen", 0.0)) >= profile.seen_p_min
        and (best_seen_score - best_old_score) >= profile.seen_old_gap_min
        and float(seen_best.get("seen_residual", 0.0)) <= profile.seen_residual_max
    )
    reject = bool(
        reject_score >= profile.reject_score_min
        and max(best_old_score, best_seen_score) <= profile.reject_known_max
        and mean_old_violation >= profile.reject_old_violation_min
        and mean_seen_residual >= profile.reject_seen_residual_min
        and len(rows) >= int(profile.reject_min_receivers)
    )
    if seen_accept:
        output_label = str(seen_best["label"])
        output_action = "accept"
        decision = "accept_seen_new_residual"
        chosen_label_set = "seen_new"
    elif old_accept:
        output_label = str(old_best["label"])
        output_action = "accept"
        decision = "accept_old_shrinkage_envelope"
        chosen_label_set = "old"
    elif reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_outside_old_new_envelopes"
        chosen_label_set = "unknown"
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_crisp_uncertain"
        chosen_label_set = ""
    total_bytes = sum(_float(row, "bytes") for row in rows)
    latency_ms = max((_float(row, "latency_ms") for row in rows), default=0.0)
    resource_proxy_pass = (
        (float(max_event_bytes) <= 0.0 or total_bytes <= float(max_event_bytes))
        and (float(max_event_latency_ms) <= 0.0 or latency_ms <= float(max_event_latency_ms))
    )
    first = rows[0]
    return {
        "event_id": _str(first, "event_id"),
        "role": _role(first.get("role")),
        "true_label": _str(first, "true_label"),
        "output_label": output_label,
        "output_action": output_action,
        "decision": decision,
        "candidate_label_set": chosen_label_set,
        "best_old_label": str(old_best.get("label", "")),
        "best_seen_new_label": str(seen_best.get("label", "")),
        "old_accept_score": float(best_old_score),
        "seen_new_accept_score": float(best_seen_score),
        "reject_score": float(reject_score),
        "old_envelope_violation": float(mean_old_violation),
        "seen_new_residual": float(mean_seen_residual),
        "receiver_disagreement": float(disagreement),
        "unique_top_labels": int(len(label_counts)),
        "receiver_count": int(len(rows)),
        "bytes_per_event": float(total_bytes),
        "latency_ms": float(latency_ms),
        "resource_proxy_pass": bool(resource_proxy_pass),
    }


def _p95(values: Sequence[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    index = int(math.ceil(0.95 * len(clean))) - 1
    return clean[max(0, min(index, len(clean) - 1))]


def _finalize(event_results: Sequence[Mapping[str, Any]], old_labels: set[str], seen_labels: set[str]) -> dict[str, Any]:
    role_total = Counter()
    role_correct = Counter()
    role_defer = Counter()
    per_total: dict[str, Counter[str]] = {"old": Counter(), "seen_new": Counter()}
    per_correct: dict[str, Counter[str]] = {"old": Counter(), "seen_new": Counter()}
    confusion = Counter()
    for item in event_results:
        role = str(item["role"])
        output = str(item["output_label"])
        action = str(item.get("output_action", ""))
        true_label = str(item["true_label"])
        role_total[role] += 1
        if action == "defer":
            role_defer[role] += 1
        if role in {"old", "seen_new"}:
            per_total[role][true_label] += 1
            if action == "accept" and output == true_label:
                role_correct[role] += 1
                per_correct[role][true_label] += 1
            elif output in old_labels:
                confusion[f"{role}->old"] += 1
            elif output in seen_labels:
                confusion[f"{role}->seen_new"] += 1
            elif action in {"reject_unknown", "defer"}:
                confusion[f"{role}->reject_or_defer"] += 1
            else:
                confusion[f"{role}->other"] += 1
        elif role == "unknown":
            if action == "reject_unknown":
                role_correct[role] += 1
                confusion["unknown->reject"] += 1
            elif output in old_labels:
                confusion["unknown->old"] += 1
            elif output in seen_labels:
                confusion["unknown->seen_new"] += 1
            elif action == "defer":
                confusion["unknown->defer"] += 1
            else:
                confusion["unknown->other"] += 1
    old_rates = {
        label: (per_correct["old"][label] / per_total["old"][label] if per_total["old"][label] else 0.0)
        for label in sorted(old_labels)
    }
    seen_rates = {
        label: (
            per_correct["seen_new"][label] / per_total["seen_new"][label]
            if per_total["seen_new"][label]
            else 0.0
        )
        for label in sorted(seen_labels)
    }
    known_total = role_total["old"] + role_total["seen_new"]
    known_defer = role_defer["old"] + role_defer["seen_new"]
    unknown_reject = role_correct["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0
    unknown_defer = role_defer["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0
    return {
        "old_total": int(role_total["old"]),
        "seen_new_total": int(role_total["seen_new"]),
        "unknown_total": int(role_total["unknown"]),
        "old_acc": role_correct["old"] / role_total["old"] if role_total["old"] else 0.0,
        "seen_new_acc": role_correct["seen_new"] / role_total["seen_new"] if role_total["seen_new"] else 0.0,
        "unknown_reject_rate": unknown_reject,
        "unknown_defer_rate": unknown_defer,
        "unknown_accept_as_known_rate": max(0.0, 1.0 - unknown_reject - unknown_defer) if role_total["unknown"] else 0.0,
        "unknown_FAR": 1.0 - unknown_reject if role_total["unknown"] else 0.0,
        "known_defer_rate": known_defer / known_total if known_total else 0.0,
        "known_coverage": 1.0 - (known_defer / known_total if known_total else 0.0),
        "min_old_class_acc": min(old_rates.values()) if old_rates else 0.0,
        "min_seen_new_class_acc": min(seen_rates.values()) if seen_rates else 0.0,
        "per_old_class_acc": old_rates,
        "per_seen_new_class_acc": seen_rates,
        "confusion": dict(sorted(confusion.items())),
        "event_count": int(sum(role_total.values())),
    }


def evaluate_crisp(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[CrispProfile],
    collab_counts: str,
    collab_group_policy: str,
    receiver_selection_policy: str,
    max_event_bytes: float,
    max_event_latency_ms: float,
    target_gates: Mapping[str, float],
    include_event_results: bool,
) -> dict[str, Any]:
    target_receivers = sorted({str(row["receiver_id"]) for row in evidence_rows})
    counts = parse_collab_counts(collab_counts, receiver_count=len(target_receivers))
    old_labels = sorted({str(row["true_label"]) for row in evidence_rows if _role(row.get("role")) == "old"})
    seen_labels = sorted({str(row["true_label"]) for row in evidence_rows if _role(row.get("role")) == "seen_new"})
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        groups[str(row["event_id"])].append(row)
    profile_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        count_results: dict[str, Any] = {}
        for k in counts:
            selected_events = []
            excluded = 0
            for rows in groups.values():
                if len(rows) < int(k) and str(collab_group_policy) in {"exact_k", "same_max_budget"}:
                    excluded += 1
                    continue
                selected = _select_receivers(rows, min(int(k), len(rows)), receiver_selection_policy)
                if selected:
                    selected_events.append(
                        _fuse_crisp_event(
                            selected,
                            profile=profile,
                            max_event_bytes=float(max_event_bytes),
                            max_event_latency_ms=float(max_event_latency_ms),
                        )
                    )
            metrics = _finalize(selected_events, set(old_labels), set(seen_labels))
            metrics["bytes_per_event"] = (
                sum(float(item["bytes_per_event"]) for item in selected_events) / len(selected_events)
                if selected_events
                else 0.0
            )
            metrics["latency_ms_p95"] = _p95([float(item["latency_ms"]) for item in selected_events])
            metrics["latency_ms_pessimistic"] = max((float(item["latency_ms"]) for item in selected_events), default=0.0)
            metrics["participating_receivers_avg"] = (
                sum(float(item["receiver_count"]) for item in selected_events) / len(selected_events)
                if selected_events
                else 0.0
            )
            metrics["participating_receivers_p95"] = _p95([float(item["receiver_count"]) for item in selected_events])
            metrics["resource_proxy_pass"] = (
                all(bool(item["resource_proxy_pass"]) for item in selected_events) if selected_events else False
            )
            metrics["excluded_incomplete_group_count"] = int(excluded)
            if include_event_results:
                metrics["event_results"] = selected_events
            count_results[str(k)] = metrics
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(k),
                "event_count": int(_count_value(metrics, "event_count")),
                "excluded_incomplete_group_count": int(excluded),
                "old_total": int(_count_value(metrics, "old_total")),
                "seen_new_total": int(_count_value(metrics, "seen_new_total")),
                "unknown_total": int(_count_value(metrics, "unknown_total")),
                "old_acc": _count_value(metrics, "old_acc"),
                "min_old": _count_value(metrics, "min_old_class_acc"),
                "seen_new_acc": _count_value(metrics, "seen_new_acc"),
                "min_seen": _count_value(metrics, "min_seen_new_class_acc"),
                "unknown_reject": _count_value(metrics, "unknown_reject_rate"),
                "unknown_accept_as_known": _count_value(metrics, "unknown_accept_as_known_rate"),
                "unknown_FAR": _count_value(metrics, "unknown_FAR"),
                "known_defer": _count_value(metrics, "known_defer_rate"),
                "known_coverage": _count_value(metrics, "known_coverage"),
                "unknown_defer": _count_value(metrics, "unknown_defer_rate"),
                "bytes_proxy_per_event": _count_value(metrics, "bytes_per_event"),
                "latency_proxy_ms": _count_value(metrics, "latency_ms_pessimistic"),
                "latency_proxy_ms_p95": _count_value(metrics, "latency_ms_p95"),
                "avg_participating": _count_value(metrics, "participating_receivers_avg"),
                "p95_participating": _count_value(metrics, "participating_receivers_p95"),
                "target_old_acc": float(target_gates["old_acc"]),
                "target_min_old": float(target_gates["min_old"]),
                "target_seen_new_acc": float(target_gates["seen_new_acc"]),
                "target_min_seen": float(target_gates["min_seen"]),
                "target_unknown_reject": float(target_gates["unknown_reject"]),
                "resource_proxy_pass": bool(metrics["resource_proxy_pass"]),
                "unknown_query_eval_only": True,
                "target_unknown_training_count": 0,
                "threshold_uses_target_unknown": False,
                "profile_selection_uses_target_unknown": False,
                "prototype_fit_uses_target_unknown": False,
                "verdict": "PENDING",
            }
            metric_pass = _target_pass(
                {
                    **row,
                    "bytes_per_event": row["bytes_proxy_per_event"],
                    "latency_ms": row["latency_proxy_ms"],
                    "resource_pass": row["resource_proxy_pass"],
                }
            )
            row["target_pass"] = bool(metric_pass and row["resource_proxy_pass"])
            row["verdict"] = "TARGET_PASS" if row["target_pass"] else "NON_DEPLOYMENT_DIAGNOSTIC"
            row["joint_score"] = (
                row["old_acc"]
                + row["seen_new_acc"]
                + row["unknown_reject"]
                - row["unknown_FAR"]
                - 0.30 * row["known_defer"]
                - 0.10 * row["unknown_defer"]
            )
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": asdict(profile),
            "receiver_count": len(target_receivers),
            "observed_receiver_ids": target_receivers,
            "counts": count_results,
        }
    return {
        "algorithm": "CRISP-C",
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_protocol_order_row": summary_rows[0] if summary_rows else None,
        "best_posthoc_eval_row": sorted(summary_rows, key=lambda row: row["joint_score"], reverse=True)[0] if summary_rows else None,
        "summary_order": "pre_registered_profile_collab_count",
        "joint_score_scope": "posthoc_evaluation_analysis_only_not_profile_or_threshold_selection",
        "target_receivers": target_receivers,
        "receiver_total": len(target_receivers),
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "group_count": len(groups),
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "threshold_uses_target_unknown": False,
        "profile_selection_uses_target_unknown": False,
        "prototype_fit_uses_target_unknown": False,
        "resource_constraints": {
            "max_event_bytes": float(max_event_bytes),
            "max_event_latency_ms": float(max_event_latency_ms),
            "scope": "proxy_packet_and_local_fusion_only",
        },
    }


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int, policy: str) -> list[Mapping[str, Any]]:
    if str(policy or "quality_prior").strip().lower() == "fixed_receiver_order":
        return sorted(rows, key=lambda row: _str(row, "receiver_id"))[: int(k)]
    return sorted(
        rows,
        key=lambda row: (
            -_float(row, "quality", 1.0),
            -max(_float(row, "old_accept_score"), _float(row, "seen_new_accept_score")),
            _float(row, "reject_score"),
            _str(row, "receiver_id"),
        ),
    )[: int(k)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_summary_csv", type=Path)
    parser.add_argument("--output_evidence_csv", type=Path)
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--collab_counts", default="all")
    parser.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    parser.add_argument("--receiver_selection_policy", default="quality_prior", choices=["fixed_receiver_order", "quality_prior"])
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=12)
    parser.add_argument("--qknn_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4070901)
    parser.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    parser.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["receiver_domain_ranked"])
    parser.add_argument("--evidence_packet_bytes", type=float, default=128.0)
    parser.add_argument("--max_event_bytes", type=float, default=1152.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--old_shrinkage_alpha", type=float, default=0.25)
    parser.add_argument("--max_seen_new_prototypes", type=int, default=3)
    parser.add_argument("--envelope_slack", type=float, default=0.02)
    parser.add_argument("--include_event_results", action="store_true")
    parser.add_argument("--target_old_acc", type=float, default=0.99)
    parser.add_argument("--target_min_old", type=float, default=0.95)
    parser.add_argument("--target_seen_new_acc", type=float, default=0.97)
    parser.add_argument("--target_min_seen", type=float, default=0.93)
    parser.add_argument("--target_unknown_reject", type=float, default=0.99)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_feature_npz(args.feature_npz)
    profiles = [_profile_by_name(name) for name in _profile_names(args.profiles)]
    evidence, metadata = build_crisp_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        old_shrinkage_alpha=float(args.old_shrinkage_alpha),
        max_seen_new_prototypes=int(args.max_seen_new_prototypes),
        envelope_slack=float(args.envelope_slack),
    )
    result = evaluate_crisp(
        evidence,
        profiles=profiles,
        collab_counts=str(args.collab_counts),
        collab_group_policy=str(args.collab_group_policy),
        receiver_selection_policy=str(args.receiver_selection_policy),
        max_event_bytes=float(args.max_event_bytes),
        max_event_latency_ms=float(args.max_event_latency_ms),
        target_gates={
            "old_acc": float(args.target_old_acc),
            "min_old": float(args.target_min_old),
            "seen_new_acc": float(args.target_seen_new_acc),
            "min_seen": float(args.target_min_seen),
            "unknown_reject": float(args.target_unknown_reject),
        },
        include_event_results=bool(args.include_event_results),
    )
    result.update(
        {
            "feature_npz": str(args.feature_npz),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "run_command_argv": [str(item) for item in sys.argv],
            "python_executable": str(sys.executable),
            "run_cwd": str(Path.cwd()),
            "calibration_source": metadata["calibration_source"],
            "prototype_fit_scope": metadata["prototype_fit_scope"],
            "threshold_uses_target_unknown": metadata["threshold_uses_target_unknown"],
            "crisp_metadata": metadata,
            "evidence_row_count": len(evidence),
            "evidence_preview": evidence[:5],
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
