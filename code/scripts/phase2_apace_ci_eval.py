#!/usr/bin/env python
"""APACE-CI Stage2-C anchor-protected collaborative open-set inference.

APACE-CI is a decision-layer diagnostic for the frozen ADV3B02/qknn8 feature
path. It uses target_old/target_new K-shot support plus source old anchors to
calibrate known evidence. target_unknown rows are evaluation-only and never
enter calibration, reliability, profile selection, or threshold fitting.
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
from phase2_orbit_pcet_ci_eval import _count_value, _target_pass, _write_csv  # noqa: E402


UNKNOWN_LABEL = "__unknown__"
DEFER_LABEL = "__defer__"


@dataclass(frozen=True)
class ApaceProfile:
    name: str
    description: str
    old_anchor_min: float
    old_conformal_min: float
    old_density_min: float
    old_margin_min: float
    old_vote_min: float
    known_evidence_min: float
    known_vote_min: float
    known_min_receivers: int
    unknown_p_max: float
    unknown_density_max: float
    unknown_energy_min: float
    unknown_margin_max: float
    unknown_fraction_min: float
    no_consensus_vote_max: float


PROFILES: tuple[ApaceProfile, ...] = (
    ApaceProfile(
        name="apace_primary",
        description="anchor-protected conformal-density ensemble with selective reject/defer",
        old_anchor_min=0.70,
        old_conformal_min=0.20,
        old_density_min=0.45,
        old_margin_min=0.02,
        old_vote_min=0.67,
        known_evidence_min=0.48,
        known_vote_min=0.34,
        known_min_receivers=1,
        unknown_p_max=0.10,
        unknown_density_max=0.35,
        unknown_energy_min=0.60,
        unknown_margin_max=0.08,
        unknown_fraction_min=0.50,
        no_consensus_vote_max=0.67,
    ),
    ApaceProfile(
        name="apace_old_guard",
        description="stronger old-anchor guardrail; diagnostic old-retention upper bound",
        old_anchor_min=0.62,
        old_conformal_min=0.15,
        old_density_min=0.38,
        old_margin_min=0.00,
        old_vote_min=0.50,
        known_evidence_min=0.42,
        known_vote_min=0.25,
        known_min_receivers=1,
        unknown_p_max=0.06,
        unknown_density_max=0.28,
        unknown_energy_min=0.72,
        unknown_margin_max=0.05,
        unknown_fraction_min=0.66,
        no_consensus_vote_max=0.50,
    ),
    ApaceProfile(
        name="apace_unknown_probe",
        description="stricter multi-evidence unknown probe; non-deployment unless old guard holds",
        old_anchor_min=0.76,
        old_conformal_min=0.28,
        old_density_min=0.52,
        old_margin_min=0.04,
        old_vote_min=0.75,
        known_evidence_min=0.56,
        known_vote_min=0.50,
        known_min_receivers=2,
        unknown_p_max=0.16,
        unknown_density_max=0.44,
        unknown_energy_min=0.50,
        unknown_margin_max=0.12,
        unknown_fraction_min=0.50,
        no_consensus_vote_max=0.75,
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
        raise argparse.ArgumentTypeError(f"unknown APACE-CI profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> ApaceProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown APACE-CI profile {name!r}")


def _role_labels(payload: Mapping[str, Any], role: str) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    return sorted({str(tx_ids[i]) for i in np.where(roles == role)[0].tolist()})


def _target_receivers(payload: Mapping[str, Any]) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    mask = np.isin(roles, ["target_old", "target_new", UNKNOWN_ROLE])
    return sorted({str(rx_ids[i]) for i in np.where(mask)[0].tolist()})


def _event_id(role: str, tx: str, scenario: str, rank: int) -> str:
    return f"{role}|{tx}|{scenario}|rank{int(rank):05d}"


def _safe_unit_cos(value: float) -> float:
    return _unit((float(value) + 1.0) / 2.0)


def _positive_cos(value: float) -> float:
    return _unit(max(0.0, float(value)))


def _loo_nonconformity(features: np.ndarray, rows: Sequence[int]) -> list[float]:
    idx = [int(i) for i in rows]
    if len(idx) <= 1:
        return [0.0]
    x = _normalize_rows(features[np.asarray(idx, dtype=int)])
    sims = x @ x.T
    out = []
    for i in range(sims.shape[0]):
        others = np.delete(sims[i], i)
        out.append(float(1.0 - np.max(others)))
    return out


def _pvalue(nonconformity: float, calibration_scores: Sequence[float]) -> float:
    cal = np.asarray([float(v) for v in calibration_scores if math.isfinite(float(v))], dtype=np.float64)
    if cal.size == 0:
        return 0.0
    return float(np.sum(cal >= float(nonconformity)) / float(cal.size))


def _score_against_support(
    vector: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    target_prototypes: Mapping[str, np.ndarray],
    source_old_prototypes: Mapping[str, np.ndarray],
) -> tuple[str, str, float, float, float, float, float]:
    x = _normalize_rows(vector.reshape(1, -1).astype(np.float32))[0]
    scored = []
    for label, support in support_by_label.items():
        support_norm = _normalize_rows(support)
        support_score = float(np.max(support_norm @ x))
        target_proto_score = float(np.dot(target_prototypes[label], x))
        old_anchor_score = 0.0
        if label in source_old_prototypes:
            old_anchor_score = float(np.dot(source_old_prototypes[label], x))
        label_set = "old" if label in source_old_prototypes else "seen_new"
        combined = (
            0.45 * _positive_cos(support_score)
            + 0.35 * _positive_cos(target_proto_score)
            + 0.20 * _positive_cos(old_anchor_score if label_set == "old" else target_proto_score)
        )
        scored.append((label, label_set, support_score, target_proto_score, old_anchor_score, combined))
    scored.sort(key=lambda item: item[5], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else top
    margin = float(top[5] - second[5])
    return str(top[0]), str(top[1]), float(top[2]), float(top[3]), float(top[4]), float(top[5]), margin


def _source_old_prototypes(payload: Mapping[str, Any], features: np.ndarray, old_labels: Sequence[str]) -> dict[str, np.ndarray]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    out: dict[str, np.ndarray] = {}
    for label in old_labels:
        idx = np.where((roles == "source") & (tx_ids == str(label)))[0]
        if idx.size:
            out[str(label)] = _normalize_rows(features[idx].mean(axis=0, keepdims=True))[0]
    return out


def build_apace_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
    event_alignment_policy: str,
    evidence_packet_bytes: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_required_roles(payload)
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    sat = np.asarray(payload["sat_scenarios"]).astype(str)
    old_labels = _role_labels(payload, "target_old")
    seen_labels = _role_labels(payload, "target_new")
    unknown_labels = _role_labels(payload, UNKNOWN_ROLE)
    receivers = _target_receivers(payload)
    if str(event_alignment_policy) != "receiver_domain_ranked":
        raise ValueError("APACE-CI currently supports receiver_domain_ranked event alignment")

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
                        "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete support/query for "
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

    source_old = _source_old_prototypes(payload, features, old_labels)
    class_calibrators: dict[str, list[float]] = {}
    for label in sorted({*old_labels, *seen_labels}):
        rows: list[int] = []
        for rx in receivers:
            rows.extend(support_by_rx_label.get((rx, label), []))
        class_calibrators[label] = _loo_nonconformity(features, rows)

    evidence: list[dict[str, Any]] = []
    for rx in receivers:
        support_by_label: dict[str, np.ndarray] = {}
        target_prototypes: dict[str, np.ndarray] = {}
        for label in sorted({*old_labels, *seen_labels}):
            rows = support_by_rx_label.get((rx, label), [])
            if not rows:
                continue
            mat = features[np.asarray(rows, dtype=int)]
            support_by_label[label] = mat
            target_prototypes[label] = _normalize_rows(mat.mean(axis=0, keepdims=True))[0]
        for group_rx, role, label, scenario, rows in query_groups:
            if group_rx != rx:
                continue
            for rank, idx in enumerate(rows):
                top_label, top_set, support_score, target_score, old_anchor_score, proto_score, margin = (
                    _score_against_support(features[int(idx)], support_by_label, target_prototypes, source_old)
                )
                nonconformity = 1.0 - support_score
                conformal_p = _pvalue(nonconformity, class_calibrators.get(top_label, []))
                density_score = _positive_cos(support_score)
                energy = _unit(1.0 - max(_positive_cos(support_score), _positive_cos(target_score)))
                quality = _unit(0.50 * density_score + 0.30 * conformal_p + 0.20 * (1.0 - energy))
                evidence.append(
                    {
                        "event_id": _event_id(role, label, scenario, rank),
                        "role": role,
                        "true_label": str(label),
                        "receiver_id": str(rx),
                        "top_label": top_label,
                        "top_label_set": top_set,
                        "proto_score": float(proto_score),
                        "target_proto_score": float(_positive_cos(target_score)),
                        "old_anchor_score": float(_positive_cos(old_anchor_score)) if top_set == "old" else 0.0,
                        "support_score": float(_positive_cos(support_score)),
                        "density_score": float(density_score),
                        "conformal_p": float(conformal_p),
                        "open_energy": float(energy),
                        "margin": float(margin),
                        "quality": float(quality),
                        "bytes": float(evidence_packet_bytes),
                        "latency_ms": 0.54 + 0.014 * len(support_by_label),
                    }
                )
    metadata = {
        "algorithm": "APACE-CI",
        "in_orbit_method": "qknn8_anchor_protected_conformal_density",
        "target_receivers": receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_tx_ids": unknown_labels,
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "support_selection_policy": str(support_selection_policy),
        "event_alignment_policy": str(event_alignment_policy),
        "calibration_source": "target_old_and_target_new_support_plus_source_old_anchor",
        "threshold_uses_target_unknown": False,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "reliability_uses_target_unknown": False,
        "reliability_fit_scope": "support_quality_only_no_query_outcomes",
        "class_calibrators": {key: [float(v) for v in value] for key, value in class_calibrators.items()},
        "evidence_packet_bytes": float(evidence_packet_bytes),
    }
    return evidence, metadata


def _aggregate_label(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    items = [row for row in rows if _str(row, "top_label") == label]
    total = max(1, len({str(row.get("receiver_id", "")) for row in rows}))
    if not items:
        return {}
    weights = [_unit(_float(row, "quality", 1.0)) for row in items]
    weight_sum = sum(weights) or float(len(items))

    def avg(key: str) -> float:
        return sum(_float(row, key) * weight for row, weight in zip(items, weights)) / weight_sum

    receiver_count = len({str(row.get("receiver_id", "")) for row in items})
    evidence = (
        0.30 * avg("target_proto_score")
        + 0.25 * avg("conformal_p")
        + 0.25 * avg("density_score")
        + 0.10 * _unit(avg("margin") * 4.0)
        + 0.10 * avg("old_anchor_score")
    )
    return {
        "label": label,
        "label_set": _str(items[0], "top_label_set"),
        "receiver_count": int(receiver_count),
        "vote_fraction": float(receiver_count / float(total)),
        "evidence": float(evidence),
        "target_proto_score": avg("target_proto_score"),
        "old_anchor_score": avg("old_anchor_score"),
        "density_score": avg("density_score"),
        "conformal_p": avg("conformal_p"),
        "open_energy": avg("open_energy"),
        "margin": avg("margin"),
    }


def _fuse_apace_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: ApaceProfile,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    labels = sorted({_str(row, "top_label") for row in rows})
    candidates = [_aggregate_label(rows, label) for label in labels]
    candidates = [item for item in candidates if item]
    candidates.sort(
        key=lambda item: (item["evidence"], item["vote_fraction"], item["conformal_p"], item["density_score"]),
        reverse=True,
    )
    best = candidates[0] if candidates else {}
    old_candidates = [item for item in candidates if item.get("label_set") == "old"]
    old_candidates.sort(
        key=lambda item: (item["old_anchor_score"], item["conformal_p"], item["density_score"], item["vote_fraction"]),
        reverse=True,
    )
    old_best = old_candidates[0] if old_candidates else {}
    old_protect = bool(
        old_best
        and float(old_best["old_anchor_score"]) >= float(profile.old_anchor_min)
        and float(old_best["conformal_p"]) >= float(profile.old_conformal_min)
        and float(old_best["density_score"]) >= float(profile.old_density_min)
        and float(old_best["margin"]) >= float(profile.old_margin_min)
        and float(old_best["vote_fraction"]) >= float(profile.old_vote_min)
    )
    known_accept = bool(
        best
        and float(best["evidence"]) >= float(profile.known_evidence_min)
        and float(best["vote_fraction"]) >= float(profile.known_vote_min)
        and int(best["receiver_count"]) >= int(profile.known_min_receivers)
    )
    low_p = [_float(row, "conformal_p") <= float(profile.unknown_p_max) for row in rows]
    low_density = [_float(row, "density_score") <= float(profile.unknown_density_max) for row in rows]
    high_energy = [_float(row, "open_energy") >= float(profile.unknown_energy_min) for row in rows]
    low_margin = [_float(row, "margin") <= float(profile.unknown_margin_max) for row in rows]
    total = max(1, len(rows))
    low_p_frac = sum(float(item) for item in low_p) / total
    low_density_frac = sum(float(item) for item in low_density) / total
    high_energy_frac = sum(float(item) for item in high_energy) / total
    low_margin_frac = sum(float(item) for item in low_margin) / total
    no_consensus = bool(best and float(best["vote_fraction"]) <= float(profile.no_consensus_vote_max))
    strict_unknown_reject = bool(
        not old_protect
        and low_p_frac >= float(profile.unknown_fraction_min)
        and low_density_frac >= float(profile.unknown_fraction_min)
        and high_energy_frac >= float(profile.unknown_fraction_min)
        and (low_margin_frac >= float(profile.unknown_fraction_min) or no_consensus)
    )
    conflict_unknown_reject = bool(
        not old_protect
        and no_consensus
        and (
            low_p_frac >= float(profile.unknown_fraction_min)
            or low_density_frac >= float(profile.unknown_fraction_min)
            or high_energy_frac >= float(profile.unknown_fraction_min)
            or low_margin_frac >= float(profile.unknown_fraction_min)
        )
    )
    unknown_reject = strict_unknown_reject or conflict_unknown_reject

    if old_protect:
        output_label = str(old_best["label"])
        output_action = "accept"
        decision = "accept_old_anchor_protected"
        chosen = old_best
    elif unknown_reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_multi_evidence"
        chosen = best
    elif known_accept:
        output_label = str(best["label"])
        output_action = "accept"
        decision = f"accept_{best['label_set']}_apace_evidence"
        chosen = best
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_apace_selective"
        chosen = best

    total_bytes = sum(_float(row, "bytes", 0.0) for row in rows)
    latency_ms = max((_float(row, "latency_ms", 0.0) for row in rows), default=0.0)
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
        "candidate_label": str(chosen.get("label", "")),
        "candidate_label_set": str(chosen.get("label_set", "")),
        "candidate_evidence": float(chosen.get("evidence", 0.0)),
        "candidate_conformal_p": float(chosen.get("conformal_p", 0.0)),
        "candidate_density_score": float(chosen.get("density_score", 0.0)),
        "candidate_open_energy": float(chosen.get("open_energy", 0.0)),
        "candidate_margin": float(chosen.get("margin", 0.0)),
        "candidate_receiver_count": int(chosen.get("receiver_count", 0)),
        "candidate_vote_fraction": float(chosen.get("vote_fraction", 0.0)),
        "low_p_fraction": float(low_p_frac),
        "low_density_fraction": float(low_density_frac),
        "high_energy_fraction": float(high_energy_frac),
        "low_margin_fraction": float(low_margin_frac),
        "old_protect": bool(old_protect),
        "receiver_count": int(len(rows)),
        "bytes_per_event": float(total_bytes),
        "latency_ms": float(latency_ms),
        "resource_proxy_pass": bool(resource_proxy_pass),
    }


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
            if output == true_label and action == "accept":
                role_correct[role] += 1
                per_correct[role][true_label] += 1
            elif action in {"reject_unknown", "defer"}:
                confusion[f"{role}->reject_or_defer"] += 1
            elif output in old_labels:
                confusion[f"{role}->old"] += 1
            elif output in seen_labels:
                confusion[f"{role}->seen_new"] += 1
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
        label: (per_correct["seen_new"][label] / per_total["seen_new"][label] if per_total["seen_new"][label] else 0.0)
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


def _p95(values: Sequence[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    index = int(math.ceil(0.95 * len(clean))) - 1
    return clean[max(0, min(index, len(clean) - 1))]


def _joint_score(row: Mapping[str, Any]) -> float:
    return (
        _float(row, "old_acc")
        + _float(row, "seen_new_acc")
        + _float(row, "unknown_reject")
        - _float(row, "unknown_FAR")
        - 0.30 * _float(row, "known_defer")
        - 0.10 * _float(row, "unknown_defer")
    )


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int, policy: str) -> list[Mapping[str, Any]]:
    policy = str(policy or "quality_prior").strip().lower()
    if policy == "fixed_receiver_order":
        return sorted(rows, key=lambda row: _str(row, "receiver_id"))[: int(k)]
    return sorted(
        rows,
        key=lambda row: (
            -_float(row, "quality", 1.0),
            -_float(row, "conformal_p"),
            -_float(row, "density_score"),
            -_float(row, "margin"),
            _str(row, "receiver_id"),
        ),
    )[: int(k)]


def evaluate_apace(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[ApaceProfile],
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
            for rows in groups.values():
                if len(rows) < int(k) and str(collab_group_policy) in {"exact_k", "same_max_budget"}:
                    continue
                selected = _select_receivers(rows, min(int(k), len(rows)), receiver_selection_policy)
                if selected:
                    selected_events.append(
                        _fuse_apace_event(
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
            if include_event_results:
                metrics["event_results"] = selected_events
            count_results[str(k)] = metrics
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(k),
                "event_count": int(_count_value(metrics, "event_count")),
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
                "profile_selection_uses_target_unknown": False,
                "reliability_uses_target_unknown": False,
                "reliability_fit_scope": "support_quality_only_no_query_outcomes",
                "verdict": "PENDING",
            }
            row["target_pass"] = _target_pass(
                {
                    **row,
                    "bytes_per_event": row["bytes_proxy_per_event"],
                    "latency_ms": row["latency_proxy_ms"],
                    "resource_pass": row["resource_proxy_pass"],
                }
            )
            row["verdict"] = "TARGET_PASS" if row["target_pass"] else "NON_DEPLOYMENT_DIAGNOSTIC"
            row["joint_score"] = _joint_score(row)
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": asdict(profile),
            "receiver_count": len(target_receivers),
            "observed_receiver_ids": target_receivers,
            "counts": count_results,
        }
    return {
        "algorithm": "APACE-CI",
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_protocol_order_row": summary_rows[0] if summary_rows else None,
        "best_posthoc_eval_row": sorted(summary_rows, key=lambda row: row["joint_score"], reverse=True)[0] if summary_rows else None,
        "summary_order": "pre_registered_profile_collab_count",
        "joint_score_scope": "posthoc_evaluation_analysis_only_not_profile_or_threshold_selection",
        "target_receivers": target_receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "reliability_uses_target_unknown": False,
        "reliability_fit_scope": "support_quality_only_no_query_outcomes",
        "resource_constraints": {
            "max_event_bytes": float(max_event_bytes),
            "max_event_latency_ms": float(max_event_latency_ms),
            "scope": "proxy_packet_and_local_fusion_only",
        },
    }


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
    parser.add_argument("--query_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4070801)
    parser.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse"])
    parser.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["receiver_domain_ranked"])
    parser.add_argument("--evidence_packet_bytes", type=float, default=160.0)
    parser.add_argument("--max_event_bytes", type=float, default=1152.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
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
    evidence_rows, metadata = build_apace_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
    )
    profiles = [_profile_by_name(name) for name in _profile_names(args.profiles)]
    result = evaluate_apace(
        evidence_rows,
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
            "threshold_uses_target_unknown": metadata["threshold_uses_target_unknown"],
            "class_calibrators": metadata["class_calibrators"],
            "apace_metadata": metadata,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    print(
        json.dumps(
            {
                "best_protocol_order_row": result["best_protocol_order_row"],
                "best_posthoc_eval_row": result["best_posthoc_eval_row"],
                "target_receivers": result["target_receivers"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
