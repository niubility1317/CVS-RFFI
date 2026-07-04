#!/usr/bin/env python
"""RMD-CI receiver-conditioned relative Mahalanobis collaborative OSR.

RMD-CI is a lightweight decision-layer diagnostic for the frozen
ADV3B02/qknn8 feature path. It uses only target_old/target_new K-shot support
and support-shell negatives to calibrate receiver-conditioned relative density.
target_unknown rows are evaluation-only.
"""

from __future__ import annotations

import argparse
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
class RmdProfile:
    name: str
    description: str
    old_anchor_min: float
    old_density_min: float
    old_rmd_min: float
    old_vote_min: float
    known_rmd_min: float
    known_density_min: float
    known_shell_max: float
    known_vote_min: float
    known_min_receivers: int
    unknown_rmd_max: float
    unknown_density_max: float
    unknown_shell_min: float
    unknown_margin_max: float
    unknown_fraction_min: float


PROFILES: tuple[RmdProfile, ...] = (
    RmdProfile(
        name="rmd_primary",
        description="receiver-conditioned relative Mahalanobis density with shell-negative rejection",
        old_anchor_min=0.68,
        old_density_min=0.55,
        old_rmd_min=0.55,
        old_vote_min=0.50,
        known_rmd_min=0.50,
        known_density_min=0.45,
        known_shell_max=0.65,
        known_vote_min=0.34,
        known_min_receivers=1,
        unknown_rmd_max=0.42,
        unknown_density_max=0.38,
        unknown_shell_min=0.70,
        unknown_margin_max=0.08,
        unknown_fraction_min=0.50,
    ),
    RmdProfile(
        name="rmd_old_guard",
        description="old-retention relative density guardrail",
        old_anchor_min=0.58,
        old_density_min=0.45,
        old_rmd_min=0.45,
        old_vote_min=0.34,
        known_rmd_min=0.42,
        known_density_min=0.38,
        known_shell_max=0.78,
        known_vote_min=0.25,
        known_min_receivers=1,
        unknown_rmd_max=0.30,
        unknown_density_max=0.26,
        unknown_shell_min=0.84,
        unknown_margin_max=0.04,
        unknown_fraction_min=0.66,
    ),
    RmdProfile(
        name="rmd_unknown_probe",
        description="stricter shell-risk probe; diagnostic unless old and seen-new floors hold",
        old_anchor_min=0.74,
        old_density_min=0.62,
        old_rmd_min=0.62,
        old_vote_min=0.67,
        known_rmd_min=0.56,
        known_density_min=0.52,
        known_shell_max=0.55,
        known_vote_min=0.50,
        known_min_receivers=2,
        unknown_rmd_max=0.52,
        unknown_density_max=0.46,
        unknown_shell_min=0.58,
        unknown_margin_max=0.12,
        unknown_fraction_min=0.50,
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
        raise argparse.ArgumentTypeError(f"unknown RMD-CI profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> RmdProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown RMD-CI profile {name!r}")


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


def _positive_cos(value: float) -> float:
    return _unit(max(0.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 60:
        return 1.0
    if value <= -60:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-float(value))))


def _mahalanobis_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    diff = x.astype(np.float64) - mean.astype(np.float64)
    safe_var = np.maximum(var.astype(np.float64), 1e-4)
    return float(np.mean((diff * diff) / safe_var))


def _source_old_prototypes(payload: Mapping[str, Any], features: np.ndarray, old_labels: Sequence[str]) -> dict[str, np.ndarray]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    out: dict[str, np.ndarray] = {}
    for label in old_labels:
        idx = np.where((roles == "source") & (tx_ids == str(label)))[0]
        if idx.size:
            out[str(label)] = _normalize_rows(features[idx].mean(axis=0, keepdims=True))[0]
    return out


def _fit_class_models(
    features: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    all_support = np.concatenate([mat for mat in support_by_label.values()], axis=0)
    bg_mean = all_support.mean(axis=0)
    bg_var = all_support.var(axis=0) + 0.04
    models: dict[str, dict[str, np.ndarray]] = {}
    for label, mat in support_by_label.items():
        models[label] = {
            "mean": _normalize_rows(mat.mean(axis=0, keepdims=True))[0],
            "var": mat.var(axis=0) + 0.04,
            "support": mat,
        }
    return models, bg_mean, bg_var


def _score_vector(
    x: np.ndarray,
    models: Mapping[str, Mapping[str, np.ndarray]],
    bg_mean: np.ndarray,
    bg_var: np.ndarray,
    source_old: Mapping[str, np.ndarray],
) -> tuple[str, str, float, float, float, float, float]:
    x = _normalize_rows(x.reshape(1, -1).astype(np.float32))[0]
    scored = []
    for label, model in models.items():
        class_md = _mahalanobis_diag(x, model["mean"], model["var"])
        bg_md = _mahalanobis_diag(x, bg_mean, bg_var)
        rel_md = class_md - bg_md
        rmd_score = _sigmoid(-rel_md)
        density = _positive_cos(float(np.max(_normalize_rows(model["support"]) @ x)))
        old_anchor = _positive_cos(float(np.dot(source_old[label], x))) if label in source_old else 0.0
        label_set = "old" if label in source_old else "seen_new"
        combined = 0.42 * rmd_score + 0.34 * density + 0.16 * old_anchor + 0.08 * (1.0 - abs(rmd_score - density))
        scored.append((label, label_set, rmd_score, density, old_anchor, combined))
    scored.sort(key=lambda item: item[5], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else top
    margin = float(top[5] - second[5])
    return str(top[0]), str(top[1]), float(top[2]), float(top[3]), float(top[4]), float(top[5]), margin


def _support_shell_scores(
    models: Mapping[str, Mapping[str, np.ndarray]],
    bg_mean: np.ndarray,
    bg_var: np.ndarray,
    source_old: Mapping[str, np.ndarray],
) -> dict[str, list[float]]:
    labels = sorted(models)
    out: dict[str, list[float]] = {label: [] for label in labels}
    for label in labels:
        own = models[label]["mean"]
        for other in labels:
            if other == label:
                continue
            vec = _normalize_rows((0.55 * own + 0.45 * models[other]["mean"]).reshape(1, -1))[0]
            scored_label, _, rmd, density, _, _, _ = _score_vector(vec, models, bg_mean, bg_var, source_old)
            if scored_label == label:
                out[label].append(float(0.50 * rmd + 0.50 * density))
            else:
                out[label].append(float(0.50 * rmd))
    return out


def _shell_risk(score: float, shell_scores: Sequence[float]) -> float:
    if not shell_scores:
        return _unit(1.0 - score)
    arr = np.asarray(shell_scores, dtype=np.float64)
    return _unit(float(np.sum(arr >= float(score)) / float(arr.size)))


def build_rmd_evidence(
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
    sat = np.asarray(payload["sat_scenarios"]).astype(str)
    old_labels = _role_labels(payload, "target_old")
    seen_labels = _role_labels(payload, "target_new")
    unknown_labels = _role_labels(payload, UNKNOWN_ROLE)
    receivers = _target_receivers(payload)
    if str(event_alignment_policy) != "receiver_domain_ranked":
        raise ValueError("RMD-CI currently supports receiver_domain_ranked event alignment")

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
    shell_negative_counts: dict[str, int] = {}
    evidence: list[dict[str, Any]] = []
    for rx in receivers:
        support_by_label: dict[str, np.ndarray] = {}
        for label in sorted({*old_labels, *seen_labels}):
            rows = support_by_rx_label.get((rx, label), [])
            if rows:
                support_by_label[label] = features[np.asarray(rows, dtype=int)]
        models, bg_mean, bg_var = _fit_class_models(features, support_by_label)
        shell_scores = _support_shell_scores(models, bg_mean, bg_var, source_old)
        for label, scores in shell_scores.items():
            class_calibrators.setdefault(label, []).extend(scores)
            shell_negative_counts[label] = shell_negative_counts.get(label, 0) + len(scores)
        for group_rx, role, label, scenario, rows in query_groups:
            if group_rx != rx:
                continue
            for rank, idx in enumerate(rows):
                top_label, top_set, rmd_score, density, old_anchor, combined, margin = _score_vector(
                    features[int(idx)], models, bg_mean, bg_var, source_old
                )
                score_for_shell = 0.50 * rmd_score + 0.50 * density
                shell_risk = _shell_risk(score_for_shell, shell_scores.get(top_label, []))
                quality = _unit(0.40 * rmd_score + 0.35 * density + 0.15 * (1.0 - shell_risk) + 0.10 * _unit(margin * 4.0))
                evidence.append(
                    {
                        "event_id": _event_id(role, label, scenario, rank),
                        "role": role,
                        "true_label": str(label),
                        "receiver_id": str(rx),
                        "top_label": top_label,
                        "top_label_set": top_set,
                        "rmd_score": float(rmd_score),
                        "density_score": float(density),
                        "shell_risk": float(shell_risk),
                        "old_anchor_score": float(old_anchor),
                        "margin": float(margin),
                        "quality": float(quality),
                        "bytes": float(evidence_packet_bytes),
                        "latency_ms": 0.62 + 0.018 * len(support_by_label),
                    }
                )
    metadata = {
        "algorithm": "RMD-CI",
        "in_orbit_method": "qknn8_receiver_conditioned_relative_density",
        "target_receivers": receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_tx_ids": unknown_labels,
        "k_shot": int(k_shot),
        "query_per_class": int(query_per_class),
        "support_selection_policy": str(support_selection_policy),
        "event_alignment_policy": str(event_alignment_policy),
        "calibration_source": "target_old_and_target_new_support_plus_support_shell_negatives",
        "threshold_uses_target_unknown": False,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "reliability_uses_target_unknown": False,
        "shell_negatives_use_target_unknown": False,
        "class_calibrators": {key: [float(v) for v in value] for key, value in class_calibrators.items()},
        "shell_negative_counts": {key: int(value) for key, value in shell_negative_counts.items()},
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
        0.34 * avg("rmd_score")
        + 0.28 * avg("density_score")
        + 0.18 * (1.0 - avg("shell_risk"))
        + 0.12 * avg("old_anchor_score")
        + 0.08 * _unit(avg("margin") * 4.0)
    )
    return {
        "label": label,
        "label_set": _str(items[0], "top_label_set"),
        "receiver_count": int(receiver_count),
        "vote_fraction": float(receiver_count / float(total)),
        "evidence": float(evidence),
        "rmd_score": avg("rmd_score"),
        "density_score": avg("density_score"),
        "shell_risk": avg("shell_risk"),
        "old_anchor_score": avg("old_anchor_score"),
        "margin": avg("margin"),
    }


def _fuse_rmd_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: RmdProfile,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    labels = sorted({_str(row, "top_label") for row in rows})
    candidates = [_aggregate_label(rows, label) for label in labels]
    candidates = [item for item in candidates if item]
    candidates.sort(key=lambda item: (item["evidence"], item["vote_fraction"], item["rmd_score"]), reverse=True)
    best = candidates[0] if candidates else {}
    old_candidates = [item for item in candidates if item.get("label_set") == "old"]
    old_candidates.sort(key=lambda item: (item["old_anchor_score"], item["density_score"], item["rmd_score"]), reverse=True)
    old_best = old_candidates[0] if old_candidates else {}
    old_guard = bool(
        old_best
        and old_best["old_anchor_score"] >= profile.old_anchor_min
        and old_best["density_score"] >= profile.old_density_min
        and old_best["rmd_score"] >= profile.old_rmd_min
        and old_best["vote_fraction"] >= profile.old_vote_min
    )
    known_accept = bool(
        best
        and best["rmd_score"] >= profile.known_rmd_min
        and best["density_score"] >= profile.known_density_min
        and best["shell_risk"] <= profile.known_shell_max
        and best["vote_fraction"] >= profile.known_vote_min
        and best["receiver_count"] >= profile.known_min_receivers
    )
    low_rmd = [_float(row, "rmd_score") <= profile.unknown_rmd_max for row in rows]
    low_density = [_float(row, "density_score") <= profile.unknown_density_max for row in rows]
    high_shell = [_float(row, "shell_risk") >= profile.unknown_shell_min for row in rows]
    low_margin = [_float(row, "margin") <= profile.unknown_margin_max for row in rows]
    total = max(1, len(rows))
    low_rmd_frac = sum(float(item) for item in low_rmd) / total
    low_density_frac = sum(float(item) for item in low_density) / total
    high_shell_frac = sum(float(item) for item in high_shell) / total
    low_margin_frac = sum(float(item) for item in low_margin) / total
    unknown_reject = bool(
        not old_guard
        and high_shell_frac >= profile.unknown_fraction_min
        and (
            low_rmd_frac >= profile.unknown_fraction_min
            or low_density_frac >= profile.unknown_fraction_min
            or low_margin_frac >= profile.unknown_fraction_min
        )
    )
    if old_guard:
        output_label = str(old_best["label"])
        output_action = "accept"
        decision = "accept_old_rmd_anchor"
        chosen = old_best
    elif unknown_reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_rmd_shell"
        chosen = best
    elif known_accept:
        output_label = str(best["label"])
        output_action = "accept"
        decision = f"accept_{best['label_set']}_rmd_density"
        chosen = best
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_rmd_selective"
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
        "candidate_rmd_score": float(chosen.get("rmd_score", 0.0)),
        "candidate_density_score": float(chosen.get("density_score", 0.0)),
        "candidate_shell_risk": float(chosen.get("shell_risk", 0.0)),
        "candidate_margin": float(chosen.get("margin", 0.0)),
        "candidate_receiver_count": int(chosen.get("receiver_count", 0)),
        "candidate_vote_fraction": float(chosen.get("vote_fraction", 0.0)),
        "low_rmd_fraction": float(low_rmd_frac),
        "low_density_fraction": float(low_density_frac),
        "high_shell_fraction": float(high_shell_frac),
        "low_margin_fraction": float(low_margin_frac),
        "old_guard": bool(old_guard),
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
    if str(policy or "quality_prior").strip().lower() == "fixed_receiver_order":
        return sorted(rows, key=lambda row: _str(row, "receiver_id"))[: int(k)]
    return sorted(
        rows,
        key=lambda row: (
            -_float(row, "quality", 1.0),
            -_float(row, "rmd_score"),
            -_float(row, "density_score"),
            _float(row, "shell_risk"),
            _str(row, "receiver_id"),
        ),
    )[: int(k)]


def evaluate_rmd(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[RmdProfile],
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
                        _fuse_rmd_event(
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
                "shell_negatives_use_target_unknown": False,
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
            row["joint_score"] = _joint_score(row)
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": asdict(profile),
            "receiver_count": len(target_receivers),
            "observed_receiver_ids": target_receivers,
            "counts": count_results,
        }
    return {
        "algorithm": "RMD-CI",
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
        "shell_negatives_use_target_unknown": False,
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
    parser.add_argument("--evidence_packet_bytes", type=float, default=176.0)
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
    evidence_rows, metadata = build_rmd_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
    )
    profiles = [_profile_by_name(name) for name in _profile_names(args.profiles)]
    result = evaluate_rmd(
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
            "shell_negative_counts": metadata["shell_negative_counts"],
            "rmd_metadata": metadata,
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
