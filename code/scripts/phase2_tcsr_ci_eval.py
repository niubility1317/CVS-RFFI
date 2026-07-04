#!/usr/bin/env python
"""TCSR-CI target support envelope shrinkage for Stage2-C collaborative OSR.

TCSR-CI is a feature-level decision-layer diagnostic. It keeps the ADV3B02
feature extractor frozen and uses only target_old/target_new K-shot support to
build class-conditional support envelopes. target_unknown rows are evaluation
only and never enter thresholds, receiver reliability, or profile selection.
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
class TcsrProfile:
    name: str
    description: str
    accept_score_slack: float
    min_margin: float
    min_vote_fraction: float
    min_receivers: int
    unknown_score_slack: float
    unknown_high_fraction: float
    no_consensus_reject_vote_fraction: float
    defer_band: float


PROFILES: tuple[TcsrProfile, ...] = (
    TcsrProfile(
        name="tcsr_support_tight",
        description="target support envelope shrinkage with conservative known accept",
        accept_score_slack=-0.02,
        min_margin=0.02,
        min_vote_fraction=0.50,
        min_receivers=1,
        unknown_score_slack=-0.06,
        unknown_high_fraction=0.60,
        no_consensus_reject_vote_fraction=0.0,
        defer_band=0.04,
    ),
    TcsrProfile(
        name="tcsr_old_guard",
        description="old-favoring support envelope with low old rejection",
        accept_score_slack=-0.04,
        min_margin=0.00,
        min_vote_fraction=0.34,
        min_receivers=1,
        unknown_score_slack=-0.10,
        unknown_high_fraction=0.70,
        no_consensus_reject_vote_fraction=0.0,
        defer_band=0.05,
    ),
    TcsrProfile(
        name="tcsr_unknown_probe",
        description="stricter support envelope for unknown rejection diagnostics",
        accept_score_slack=0.04,
        min_margin=0.04,
        min_vote_fraction=0.60,
        min_receivers=2,
        unknown_score_slack=0.00,
        unknown_high_fraction=0.50,
        no_consensus_reject_vote_fraction=0.60,
        defer_band=0.02,
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
        raise argparse.ArgumentTypeError(f"unknown TCSR-CI profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> TcsrProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown TCSR-CI profile {name!r}")


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


def _support_threshold(values: Sequence[float], quantile: float, min_floor: float) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float(min_floor)
    return max(float(min_floor), float(np.quantile(arr, float(np.clip(quantile, 0.0, 1.0)))))


def _loo_support_scores(features: np.ndarray, rows: Sequence[int]) -> list[float]:
    idx = [int(i) for i in rows]
    if len(idx) <= 1:
        return [1.0]
    x = _normalize_rows(features[np.asarray(idx, dtype=int)])
    scores = x @ x.T
    out = []
    for i in range(scores.shape[0]):
        others = np.delete(scores[i], i)
        out.append(float(np.max(others)))
    return out


def _score_against_support(
    vector: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    prototypes: Mapping[str, np.ndarray],
) -> tuple[str, float, float, float]:
    x = _normalize_rows(vector.reshape(1, -1).astype(np.float32))[0]
    scored = []
    for label, support in support_by_label.items():
        support_score = float(np.max(_normalize_rows(support) @ x))
        proto_score = float(np.dot(prototypes[label], x))
        combined = 0.70 * support_score + 0.30 * proto_score
        scored.append((label, support_score, proto_score, combined))
    scored.sort(key=lambda item: item[3], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else top
    margin = float(top[3] - second[3])
    return str(top[0]), float(top[1]), float(top[2]), margin


def build_tcsr_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
    event_alignment_policy: str,
    threshold_quantile: float,
    min_class_threshold: float,
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
    event_policy = str(event_alignment_policy)
    if event_policy != "receiver_domain_ranked":
        raise ValueError("TCSR-CI currently supports receiver_domain_ranked event alignment")

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

    class_thresholds: dict[str, float] = {}
    class_threshold_scores: dict[str, list[float]] = {}
    for label in sorted({*old_labels, *seen_labels}):
        rows: list[int] = []
        for rx in receivers:
            rows.extend(support_by_rx_label.get((rx, label), []))
        scores = _loo_support_scores(features, rows)
        class_threshold_scores[label] = scores
        class_thresholds[label] = _support_threshold(scores, threshold_quantile, min_class_threshold)

    evidence: list[dict[str, Any]] = []
    for rx in receivers:
        support_by_label: dict[str, np.ndarray] = {}
        prototypes: dict[str, np.ndarray] = {}
        for label in sorted({*old_labels, *seen_labels}):
            rows = support_by_rx_label.get((rx, label), [])
            if not rows:
                continue
            mat = features[np.asarray(rows, dtype=int)]
            support_by_label[label] = mat
            prototypes[label] = _normalize_rows(mat.mean(axis=0, keepdims=True))[0]
        for group_rx, role, label, scenario, rows in query_groups:
            if group_rx != rx:
                continue
            for rank, idx in enumerate(rows):
                top_label, support_score, proto_score, margin = _score_against_support(
                    features[int(idx)], support_by_label, prototypes
                )
                top_set = "old" if top_label in old_labels else "seen_new"
                threshold = class_thresholds.get(top_label, float(min_class_threshold))
                evidence.append(
                    {
                        "event_id": _event_id(role, label, scenario, rank),
                        "role": role,
                        "true_label": str(label),
                        "receiver_id": str(rx),
                        "top_label": top_label,
                        "top_label_set": top_set,
                        "support_score": float(support_score),
                        "prototype_score": float(proto_score),
                        "margin": float(margin),
                        "class_threshold": float(threshold),
                        "score_above_threshold": float(support_score - threshold),
                        "bytes": float(evidence_packet_bytes),
                        "latency_ms": 0.50 + 0.012 * len(support_by_label),
                    }
                )
    metadata = {
        "algorithm": "TCSR-CI",
        "in_orbit_method": "qknn8_feature_support_envelope",
        "target_receivers": receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_tx_ids": unknown_labels,
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "support_selection_policy": str(support_selection_policy),
        "event_alignment_policy": event_policy,
        "threshold_source": "target_old_and_target_new_support_only",
        "threshold_uses_target_unknown": False,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "class_thresholds": class_thresholds,
        "class_threshold_score_counts": {key: len(value) for key, value in class_threshold_scores.items()},
        "evidence_packet_bytes": float(evidence_packet_bytes),
    }
    return evidence, metadata


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int, policy: str) -> list[Mapping[str, Any]]:
    policy = str(policy or "support_score_prior").strip().lower()
    if policy == "fixed_receiver_order":
        return sorted(rows, key=lambda row: _str(row, "receiver_id"))[: int(k)]
    return sorted(
        rows,
        key=lambda row: (-_float(row, "support_score"), -_float(row, "margin"), _str(row, "receiver_id")),
    )[: int(k)]


def _fuse_tcsr_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: TcsrProfile,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[_str(row, "top_label")].append(row)
    total_receivers = max(1, len({str(row.get("receiver_id", "")) for row in rows}))
    candidates = []
    for label, items in by_label.items():
        avg_score = sum(_float(item, "support_score") for item in items) / len(items)
        avg_margin = sum(_float(item, "margin") for item in items) / len(items)
        avg_threshold = sum(_float(item, "class_threshold") for item in items) / len(items)
        receivers = {str(item["receiver_id"]) for item in items}
        candidates.append(
            {
                "label": label,
                "label_set": _str(items[0], "top_label_set"),
                "support_score": float(avg_score),
                "margin": float(avg_margin),
                "class_threshold": float(avg_threshold),
                "receiver_count": len(receivers),
                "vote_fraction": len(receivers) / float(total_receivers),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["support_score"] - item["class_threshold"],
            item["vote_fraction"],
            item["margin"],
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else {}
    accept = bool(
        best
        and float(best["support_score"]) >= float(best["class_threshold"]) + float(profile.accept_score_slack)
        and float(best["margin"]) >= float(profile.min_margin)
        and float(best["vote_fraction"]) >= float(profile.min_vote_fraction)
        and int(best["receiver_count"]) >= int(profile.min_receivers)
    )
    low_votes = [
        float(row["support_score"]) < float(row["class_threshold"]) + float(profile.unknown_score_slack)
        for row in rows
    ]
    unknown_reject = bool(
        low_votes
        and sum(float(item) for item in low_votes) / len(low_votes) >= float(profile.unknown_high_fraction)
        and not accept
    )
    no_consensus_reject = bool(
        not accept
        and float(profile.no_consensus_reject_vote_fraction) > 0.0
        and best
        and float(best["vote_fraction"]) < float(profile.no_consensus_reject_vote_fraction)
    )
    if accept:
        output_label = str(best["label"])
        output_action = "accept"
        decision = f"accept_{best['label_set']}_support_envelope"
    elif unknown_reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_support_gap"
    elif no_consensus_reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_no_consensus"
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_support_band"
    total_bytes = sum(_float(row, "bytes", 0.0) for row in rows)
    latency_ms = max((_float(row, "latency_ms", 0.0) for row in rows), default=0.0)
    resource_pass = (
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
        "candidate_label": str(best.get("label", "")),
        "candidate_label_set": str(best.get("label_set", "")),
        "candidate_support_score": float(best.get("support_score", 0.0)),
        "candidate_margin": float(best.get("margin", 0.0)),
        "candidate_threshold": float(best.get("class_threshold", 0.0)),
        "candidate_receiver_count": int(best.get("receiver_count", 0)),
        "candidate_vote_fraction": float(best.get("vote_fraction", 0.0)),
        "receiver_count": int(len(rows)),
        "bytes_per_event": float(total_bytes),
        "latency_ms": float(latency_ms),
        "resource_pass": bool(resource_pass),
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
    return {
        "old_total": int(role_total["old"]),
        "seen_new_total": int(role_total["seen_new"]),
        "unknown_total": int(role_total["unknown"]),
        "old_acc": role_correct["old"] / role_total["old"] if role_total["old"] else 0.0,
        "seen_new_acc": role_correct["seen_new"] / role_total["seen_new"] if role_total["seen_new"] else 0.0,
        "unknown_reject_rate": role_correct["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0,
        "unknown_FAR": 1.0 - (role_correct["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0),
        "known_defer_rate": known_defer / known_total if known_total else 0.0,
        "unknown_defer_rate": role_defer["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0,
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


def evaluate_tcsr(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[TcsrProfile],
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
                        _fuse_tcsr_event(
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
            metrics["resource_pass"] = all(bool(item["resource_pass"]) for item in selected_events) if selected_events else False
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
                "unknown_FAR": _count_value(metrics, "unknown_FAR"),
                "known_defer": _count_value(metrics, "known_defer_rate"),
                "unknown_defer": _count_value(metrics, "unknown_defer_rate"),
                "bytes_per_event": _count_value(metrics, "bytes_per_event"),
                "latency_ms": _count_value(metrics, "latency_ms_pessimistic"),
                "latency_ms_p95": _count_value(metrics, "latency_ms_p95"),
                "avg_participating": _count_value(metrics, "participating_receivers_avg"),
                "p95_participating": _count_value(metrics, "participating_receivers_p95"),
                "target_old_acc": float(target_gates["old_acc"]),
                "target_min_old": float(target_gates["min_old"]),
                "target_seen_new_acc": float(target_gates["seen_new_acc"]),
                "target_min_seen": float(target_gates["min_seen"]),
                "target_unknown_reject": float(target_gates["unknown_reject"]),
                "resource_pass": bool(metrics["resource_pass"]),
                "unknown_query_eval_only": True,
                "target_unknown_training_count": 0,
                "profile_selection_uses_target_unknown": False,
            }
            row["target_pass"] = _target_pass(row)
            row["joint_score"] = _joint_score(row)
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": asdict(profile),
            "receiver_count": len(target_receivers),
            "observed_receiver_ids": target_receivers,
            "counts": count_results,
        }
    return {
        "algorithm": "TCSR-CI",
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
        "resource_constraints": {
            "max_event_bytes": float(max_event_bytes),
            "max_event_latency_ms": float(max_event_latency_ms),
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
    parser.add_argument("--receiver_selection_policy", default="support_score_prior", choices=["fixed_receiver_order", "support_score_prior"])
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4070801)
    parser.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse"])
    parser.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["receiver_domain_ranked"])
    parser.add_argument("--threshold_quantile", type=float, default=0.10)
    parser.add_argument("--min_class_threshold", type=float, default=0.20)
    parser.add_argument("--evidence_packet_bytes", type=float, default=128.0)
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
    evidence_rows, metadata = build_tcsr_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        threshold_quantile=float(args.threshold_quantile),
        min_class_threshold=float(args.min_class_threshold),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
    )
    profiles = [_profile_by_name(name) for name in _profile_names(args.profiles)]
    result = evaluate_tcsr(
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
            "threshold_source": metadata["threshold_source"],
            "threshold_uses_target_unknown": metadata["threshold_uses_target_unknown"],
            "class_thresholds": metadata["class_thresholds"],
            "tcsr_metadata": metadata,
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
