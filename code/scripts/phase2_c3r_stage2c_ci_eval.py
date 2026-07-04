#!/usr/bin/env python
"""C3R-Stage2C conservative collaborative conformal receiver fusion.

C3R keeps the ADV3B02/qknn8 feature path frozen. It reuses PCET support-only
evidence, then applies a positive-only conformal receiver fusion layer:

1. old labels are checked first with an old/source-anchor shield;
2. seen-new labels require stricter multi-receiver support;
3. unknown is rejected only when no known shield is active and multiple
   receiver risk components agree.

target_unknown rows remain evaluation-only; they are never used to fit
thresholds, choose profiles, or select collaborative receiver counts.
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

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import parse_collab_counts  # noqa: E402
from phase2_orbit_pcet_ci_eval import (  # noqa: E402
    _count_value,
    _target_pass,
    _write_csv,
    parse_args as parse_pcet_args,
    run_pcet_ci,
)


UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class C3RProfile:
    name: str
    description: str
    min_old_log_accept: float
    min_seen_log_accept: float
    min_old_vote_receivers: int
    min_seen_vote_receivers: int
    min_old_vote_fraction: float
    min_seen_vote_fraction: float
    max_old_anchor_risk: float
    max_seen_anchor_risk: float
    unknown_mean_risk: float
    unknown_high_fraction: float
    unknown_component_fraction: float
    risk_component_threshold: float
    max_known_risk_for_accept: float
    variance_penalty: float
    old_anchor_bonus: float


PROFILES: tuple[C3RProfile, ...] = (
    C3RProfile(
        name="c3r_old_anchor",
        description="old-first positive conformal shield with conservative unknown confirmation",
        min_old_log_accept=0.28,
        min_seen_log_accept=0.26,
        min_old_vote_receivers=1,
        min_seen_vote_receivers=2,
        min_old_vote_fraction=0.20,
        min_seen_vote_fraction=0.50,
        max_old_anchor_risk=0.96,
        max_seen_anchor_risk=0.86,
        unknown_mean_risk=0.88,
        unknown_high_fraction=0.72,
        unknown_component_fraction=0.66,
        risk_component_threshold=0.68,
        max_known_risk_for_accept=0.98,
        variance_penalty=0.18,
        old_anchor_bonus=0.20,
    ),
    C3RProfile(
        name="c3r_balanced",
        description="balanced old shield, seen-new registration, and unknown confirmation",
        min_old_log_accept=0.34,
        min_seen_log_accept=0.32,
        min_old_vote_receivers=1,
        min_seen_vote_receivers=2,
        min_old_vote_fraction=0.40,
        min_seen_vote_fraction=0.56,
        max_old_anchor_risk=0.68,
        max_seen_anchor_risk=0.56,
        unknown_mean_risk=0.78,
        unknown_high_fraction=0.60,
        unknown_component_fraction=0.56,
        risk_component_threshold=0.58,
        max_known_risk_for_accept=0.82,
        variance_penalty=0.24,
        old_anchor_bonus=0.16,
    ),
    C3RProfile(
        name="c3r_unknown_guarded",
        description="stricter unknown rejection; diagnostic unless old retention is non-degraded",
        min_old_log_accept=0.42,
        min_seen_log_accept=0.38,
        min_old_vote_receivers=2,
        min_seen_vote_receivers=2,
        min_old_vote_fraction=0.50,
        min_seen_vote_fraction=0.64,
        max_old_anchor_risk=0.58,
        max_seen_anchor_risk=0.48,
        unknown_mean_risk=0.66,
        unknown_high_fraction=0.50,
        unknown_component_fraction=0.50,
        risk_component_threshold=0.50,
        max_known_risk_for_accept=0.72,
        variance_penalty=0.30,
        old_anchor_bonus=0.10,
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
        raise argparse.ArgumentTypeError(f"unknown C3R profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> C3RProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown C3R profile {name!r}")


def _receiver_quality(row: Mapping[str, Any]) -> float:
    reliability = _unit(_float(row, "reliability", 1.0))
    support_density = _unit(_float(row, "support_density", reliability))
    class_rel = _unit(
        _float(
            row,
            "class_evidence_top1_receiver_class_reliability",
            _float(row, "receiver_class_reliability", reliability),
        )
    )
    pvalue = _unit(_float(row, "class_evidence_top1_conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
    return _unit(0.25 * reliability + 0.30 * support_density + 0.25 * class_rel + 0.20 * pvalue)


def _row_unknown_risk(row: Mapping[str, Any]) -> float:
    return max(
        _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
        _unit(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        _unit(_float(row, "class_negative_risk", 0.0)),
        _unit(_float(row, "class_shell_risk", 0.0)),
    )


def _tail_risk(row: Mapping[str, Any], rank: int) -> float:
    prefix = f"class_evidence_top{rank}_"
    return max(
        _unit(_float(row, prefix + "mahalanobis_risk", _float(row, "mahalanobis_risk", 0.0))),
        _unit(_float(row, prefix + "evt_risk", _float(row, "evt_risk", 0.0))),
        _unit(_float(row, prefix + "class_shell_risk", _float(row, "class_shell_risk", 0.0))),
        _unit(_float(row, prefix + "class_negative_risk", _float(row, "class_negative_risk", 0.0))),
    )


def _known_set(label: str, old_labels: set[str], seen_labels: set[str]) -> str:
    if label in old_labels:
        return "old"
    if label in seen_labels:
        return "seen_new"
    return ""


def _candidate_entries(row: Mapping[str, Any], *, top_m: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    rq = _receiver_quality(row)
    for rank in range(1, int(top_m) + 1):
        prefix = f"class_evidence_top{rank}_"
        label = _str(row, prefix + "label", "")
        if not label:
            continue
        score = _unit(_float(row, prefix + "score", 0.0))
        pvalue = _unit(_float(row, prefix + "conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
        margin = max(0.0, _float(row, prefix + "margin", _float(row, "known_margin", 0.0)))
        support_count = max(0.0, _float(row, prefix + "support_count", _float(row, "class_conformal_support_count", 0.0)))
        tail = _tail_risk(row, rank)
        unknown_risk = max(_row_unknown_risk(row), _unit(_float(row, prefix + "unknown_risk", 0.0)))
        conformal_accept = _unit(
            0.40 * pvalue
            + 0.24 * score
            + 0.16 * min(1.0, margin / 0.25)
            + 0.12 * rq
            + 0.08 * min(1.0, support_count / 4.0)
        )
        log_accept = math.log(max(conformal_accept, 1e-6)) - 0.55 * tail - 0.30 * unknown_risk
        entries.append(
            {
                "label": label,
                "rank": rank,
                "receiver_id": _str(row, "receiver_id"),
                "conformal_accept": conformal_accept,
                "log_accept": log_accept,
                "score": score,
                "pvalue": pvalue,
                "margin": margin,
                "support_count": support_count,
                "receiver_quality": rq,
                "tail_risk": tail,
                "unknown_risk": unknown_risk,
            }
        )
    return entries


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int, policy: str) -> list[Mapping[str, Any]]:
    policy = str(policy or "support_quality_prior").strip().lower()
    if policy == "fixed_receiver_order":
        return sorted(rows, key=lambda row: _str(row, "receiver_id"))[: int(k)]
    return sorted(
        rows,
        key=lambda row: (-_receiver_quality(row), _float(row, "latency_ms", 0.0), _str(row, "receiver_id")),
    )[: int(k)]


def _label_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: C3RProfile,
    top_m: int,
    old_labels: set[str],
    seen_labels: set[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.extend(_candidate_entries(row, top_m=top_m))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in entries:
        by_label[str(item["label"])].append(item)

    out: list[dict[str, Any]] = []
    for label, items in by_label.items():
        label_set = _known_set(label, old_labels, seen_labels)
        if not label_set:
            continue
        receivers = {str(item["receiver_id"]) for item in items}
        weights = [max(float(item["receiver_quality"]), 1e-6) for item in items]
        total_w = sum(weights) or 1.0
        mean_log_accept = sum(float(item["log_accept"]) * w for item, w in zip(items, weights)) / total_w
        mean_accept = sum(float(item["conformal_accept"]) * w for item, w in zip(items, weights)) / total_w
        accept_values = [float(item["conformal_accept"]) for item in items]
        variance = (
            sum((value - (sum(accept_values) / len(accept_values))) ** 2 for value in accept_values) / len(accept_values)
            if accept_values
            else 0.0
        )
        tail = sum(float(item["tail_risk"]) * w for item, w in zip(items, weights)) / total_w
        risk = sum(float(item["unknown_risk"]) * w for item, w in zip(items, weights)) / total_w
        vote_fraction = len(receivers) / max(len(rows), 1)
        score = mean_accept - float(profile.variance_penalty) * variance - 0.15 * tail - 0.10 * risk
        if label_set == "old":
            score += float(profile.old_anchor_bonus) * max(0.0, 1.0 - risk)
        out.append(
            {
                "label": label,
                "label_set": label_set,
                "score": score,
                "mean_log_accept": mean_log_accept,
                "mean_accept": mean_accept,
                "receiver_count": len(receivers),
                "vote_fraction": vote_fraction,
                "mean_tail_risk": tail,
                "mean_unknown_risk": risk,
                "accept_variance": variance,
            }
        )
    out.sort(
        key=lambda item: (
            float(item["score"]),
            str(item["label_set"]) == "old",
            int(item["receiver_count"]),
            -float(item["mean_unknown_risk"]),
            str(item["label"]),
        ),
        reverse=True,
    )
    return out


def _fuse_c3r_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: C3RProfile,
    top_m: int,
    old_labels: set[str],
    seen_labels: set[str],
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    candidates = _label_candidates(rows, profile=profile, top_m=top_m, old_labels=old_labels, seen_labels=seen_labels)
    best_old = next((item for item in candidates if item["label_set"] == "old"), None)
    best_seen = next((item for item in candidates if item["label_set"] == "seen_new"), None)
    selected_risks = [_row_unknown_risk(row) for row in rows]
    component_high = [
        max(
            _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
            _unit(_float(row, "class_negative_risk", 0.0)),
            _unit(_float(row, "class_shell_risk", 0.0)),
            _unit(_float(row, "mahalanobis_risk", 0.0)),
            _unit(_float(row, "evt_risk", 0.0)),
        )
        for row in rows
    ]
    mean_unknown_risk = sum(selected_risks) / max(len(selected_risks), 1)
    high_fraction = sum(v >= float(profile.unknown_mean_risk) for v in selected_risks) / max(len(selected_risks), 1)
    component_fraction = sum(v >= float(profile.risk_component_threshold) for v in component_high) / max(len(component_high), 1)

    old_accept = bool(
        best_old
        and float(best_old["score"]) >= float(profile.min_old_log_accept)
        and int(best_old["receiver_count"]) >= int(profile.min_old_vote_receivers)
        and float(best_old["vote_fraction"]) >= float(profile.min_old_vote_fraction)
        and float(best_old["mean_unknown_risk"]) <= float(profile.max_old_anchor_risk)
        and float(best_old["mean_unknown_risk"]) <= float(profile.max_known_risk_for_accept)
    )
    seen_accept = bool(
        (not old_accept)
        and best_seen
        and float(best_seen["score"]) >= float(profile.min_seen_log_accept)
        and int(best_seen["receiver_count"]) >= int(profile.min_seen_vote_receivers)
        and float(best_seen["vote_fraction"]) >= float(profile.min_seen_vote_fraction)
        and float(best_seen["mean_unknown_risk"]) <= float(profile.max_seen_anchor_risk)
        and float(best_seen["mean_unknown_risk"]) <= float(profile.max_known_risk_for_accept)
    )
    unknown_confirm = bool(
        (not old_accept)
        and (not seen_accept)
        and mean_unknown_risk >= float(profile.unknown_mean_risk)
        and high_fraction >= float(profile.unknown_high_fraction)
        and component_fraction >= float(profile.unknown_component_fraction)
    )

    if old_accept:
        chosen = best_old or {}
        output_label = str(chosen.get("label", UNKNOWN_LABEL))
        decision = "accept_old_shield"
    elif seen_accept:
        chosen = best_seen or {}
        output_label = str(chosen.get("label", UNKNOWN_LABEL))
        decision = "accept_seen_new"
    elif unknown_confirm:
        chosen = {}
        output_label = UNKNOWN_LABEL
        decision = "reject_unknown"
    else:
        chosen = best_old or best_seen or {}
        output_label = UNKNOWN_LABEL
        decision = "defer"

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
        "decision": decision,
        "candidate_label": str(chosen.get("label", "")),
        "candidate_label_set": str(chosen.get("label_set", "")),
        "candidate_score": float(chosen.get("score", 0.0)),
        "candidate_receiver_count": int(chosen.get("receiver_count", 0)),
        "candidate_vote_fraction": float(chosen.get("vote_fraction", 0.0)),
        "candidate_mean_unknown_risk": float(chosen.get("mean_unknown_risk", 1.0)),
        "mean_unknown_risk": float(mean_unknown_risk),
        "unknown_high_fraction": float(high_fraction),
        "risk_component_high_fraction": float(component_fraction),
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
        true_label = str(item["true_label"])
        output = str(item["output_label"])
        decision = str(item["decision"])
        role_total[role] += 1
        if decision == "defer":
            role_defer[role] += 1
        if role in {"old", "seen_new"}:
            per_total[role][true_label] += 1
            if output == true_label:
                role_correct[role] += 1
                per_correct[role][true_label] += 1
            elif output == UNKNOWN_LABEL:
                confusion[f"{role}->reject_or_defer"] += 1
            elif output in old_labels:
                confusion[f"{role}->old"] += 1
            elif output in seen_labels:
                confusion[f"{role}->seen_new"] += 1
            else:
                confusion[f"{role}->other"] += 1
        elif role == "unknown":
            if output == UNKNOWN_LABEL and decision == "reject_unknown":
                role_correct[role] += 1
                confusion["unknown->reject"] += 1
            elif output in old_labels:
                confusion["unknown->old"] += 1
            elif output in seen_labels:
                confusion["unknown->seen_new"] += 1
            else:
                confusion["unknown->defer"] += 1
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
    event_count = sum(role_total.values())
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
        "event_count": int(event_count),
    }


def evaluate_c3r(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[C3RProfile],
    collab_counts: str,
    collab_group_policy: str,
    receiver_selection_policy: str,
    top_m: int,
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
                if not selected:
                    continue
                selected_events.append(
                    _fuse_c3r_event(
                        selected,
                        profile=profile,
                        top_m=int(top_m),
                        old_labels=set(old_labels),
                        seen_labels=set(seen_labels),
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
        "algorithm": "C3R-Stage2C",
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_protocol_order_row": summary_rows[0] if summary_rows else None,
        "best_eval_row": sorted(summary_rows, key=lambda row: row["joint_score"], reverse=True)[0] if summary_rows else None,
        "summary_order": "pre_registered_profile_collab_count",
        "joint_score_scope": "evaluation_analysis_only_not_profile_selection",
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
        - 0.50 * _float(row, "known_defer")
        - 0.25 * _float(row, "unknown_defer")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_summary_csv", type=Path)
    p.add_argument("--output_evidence_csv", type=Path)
    p.add_argument("--profiles", default="all")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--receiver_selection_policy", default="support_quality_prior", choices=["fixed_receiver_order", "support_quality_prior"])
    p.add_argument("--top_m", type=int, default=3)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070721)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse"])
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--include_event_results", action="store_true")
    p.add_argument("--target_old_acc", type=float, default=0.99)
    p.add_argument("--target_min_old", type=float, default=0.95)
    p.add_argument("--target_seen_new_acc", type=float, default=0.97)
    p.add_argument("--target_min_seen", type=float, default=0.93)
    p.add_argument("--target_unknown_reject", type=float, default=0.99)
    return p.parse_args(argv)


def _pcet_argv(args: argparse.Namespace) -> list[str]:
    out = args.output_json.parent / "_c3r_pcet_base.json"
    return [
        "--feature_npz",
        str(args.feature_npz),
        "--output_json",
        str(out),
        "--profiles",
        "pcet_known_preserving",
        "--collab_counts",
        str(args.collab_counts),
        "--collab_group_policy",
        str(args.collab_group_policy),
        "--k_shot",
        str(args.k_shot),
        "--query_per_class",
        str(args.query_per_class),
        "--qknn_k",
        str(args.qknn_k),
        "--seed",
        str(args.seed),
        "--support_selection_policy",
        str(args.support_selection_policy),
        "--event_alignment_policy",
        str(args.event_alignment_policy),
        "--class_evidence_top_m",
        str(args.top_m),
        "--max_event_bytes",
        str(args.max_event_bytes),
        "--max_event_latency_ms",
        str(args.max_event_latency_ms),
        "--target_old_acc",
        str(args.target_old_acc),
        "--target_min_old",
        str(args.target_min_old),
        "--target_seen_new_acc",
        str(args.target_seen_new_acc),
        "--target_min_seen",
        str(args.target_min_seen),
        "--target_unknown_reject",
        str(args.target_unknown_reject),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pcet_args = parse_pcet_args(_pcet_argv(args))
    pcet = run_pcet_ci(pcet_args)
    evidence_rows = pcet.pop("_evidence_rows")
    profile_names = _profile_names(args.profiles)
    profiles = [_profile_by_name(name) for name in profile_names]
    result = evaluate_c3r(
        evidence_rows,
        profiles=profiles,
        collab_counts=str(args.collab_counts),
        collab_group_policy=str(args.collab_group_policy),
        receiver_selection_policy=str(args.receiver_selection_policy),
        top_m=int(args.top_m),
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
    result["feature_npz"] = str(args.feature_npz)
    result["base_pcet_known_preserving"] = pcet
    result["run_command_argv"] = [str(item) for item in sys.argv]
    result["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    print(
        json.dumps(
            {
                "best_protocol_order_row": result["best_protocol_order_row"],
                "best_eval_row": result["best_eval_row"],
                "target_receivers": result["target_receivers"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
