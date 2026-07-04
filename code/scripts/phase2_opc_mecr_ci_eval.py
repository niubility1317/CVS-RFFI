#!/usr/bin/env python
"""OPC-MECR Stage2-C collaborative open-set inference.

OPC-MECR keeps the ADV3B02/qknn8 evidence path frozen and adds a lightweight
decision layer for satellite-swarm inference:

* old-protected safety gate with strong-unknown override to defer/reject;
* class-conditional multi-envelope evidence from support conformal, tail,
  margin, reliability, and receiver agreement;
* consensus-before-accept fusion over M=1..R participating target receivers.

target_unknown rows are evaluation-only. They are never used for profile
selection, threshold fitting, receiver reliability, or envelope construction.
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
DEFER_LABEL = "__defer__"


@dataclass(frozen=True)
class OpcMecrProfile:
    name: str
    description: str
    min_old_envelope: float
    min_seen_envelope: float
    min_old_receivers: int
    min_seen_receivers: int
    min_old_vote_fraction: float
    min_seen_vote_fraction: float
    max_old_unknown_risk: float
    max_seen_unknown_risk: float
    strong_unknown_mean_risk: float
    strong_unknown_high_fraction: float
    strong_unknown_component_fraction: float
    risk_component_threshold: float
    old_override_unknown_risk: float
    old_override_component_fraction: float
    no_consensus_max_envelope: float
    seen_new_exclusion_envelope: float


PROFILES: tuple[OpcMecrProfile, ...] = (
    OpcMecrProfile(
        name="opc_old_guard",
        description="old-protected envelope with defer on strong unknown evidence",
        min_old_envelope=0.18,
        min_seen_envelope=0.36,
        min_old_receivers=1,
        min_seen_receivers=2,
        min_old_vote_fraction=0.20,
        min_seen_vote_fraction=0.50,
        max_old_unknown_risk=1.00,
        max_seen_unknown_risk=1.00,
        strong_unknown_mean_risk=0.86,
        strong_unknown_high_fraction=0.70,
        strong_unknown_component_fraction=0.66,
        risk_component_threshold=0.66,
        old_override_unknown_risk=0.985,
        old_override_component_fraction=0.88,
        no_consensus_max_envelope=0.30,
        seen_new_exclusion_envelope=0.30,
    ),
    OpcMecrProfile(
        name="mecr_balanced",
        description="class-conditional multi-envelope consensus with old floor",
        min_old_envelope=0.28,
        min_seen_envelope=0.42,
        min_old_receivers=1,
        min_seen_receivers=2,
        min_old_vote_fraction=0.30,
        min_seen_vote_fraction=0.56,
        max_old_unknown_risk=0.98,
        max_seen_unknown_risk=0.98,
        strong_unknown_mean_risk=0.76,
        strong_unknown_high_fraction=0.58,
        strong_unknown_component_fraction=0.56,
        risk_component_threshold=0.56,
        old_override_unknown_risk=0.88,
        old_override_component_fraction=0.66,
        no_consensus_max_envelope=0.34,
        seen_new_exclusion_envelope=0.34,
    ),
    OpcMecrProfile(
        name="mecr_unknown_probe",
        description="stricter open-set probe; diagnostic unless old floor holds",
        min_old_envelope=0.34,
        min_seen_envelope=0.48,
        min_old_receivers=2,
        min_seen_receivers=2,
        min_old_vote_fraction=0.40,
        min_seen_vote_fraction=0.62,
        max_old_unknown_risk=0.94,
        max_seen_unknown_risk=0.46,
        strong_unknown_mean_risk=0.66,
        strong_unknown_high_fraction=0.50,
        strong_unknown_component_fraction=0.50,
        risk_component_threshold=0.48,
        old_override_unknown_risk=0.80,
        old_override_component_fraction=0.58,
        no_consensus_max_envelope=0.38,
        seen_new_exclusion_envelope=0.38,
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
        raise argparse.ArgumentTypeError(f"unknown OPC-MECR profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> OpcMecrProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown OPC-MECR profile {name!r}")


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
    return _unit(0.25 * reliability + 0.25 * support_density + 0.25 * class_rel + 0.25 * pvalue)


def _row_unknown_risk(row: Mapping[str, Any]) -> float:
    return max(
        _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
        _unit(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        _unit(_float(row, "class_negative_risk", 0.0)),
        _unit(_float(row, "class_shell_risk", 0.0)),
        _unit(_float(row, "oldness_risk", 0.0)),
    )


def _risk_component_fraction(row: Mapping[str, Any], threshold: float) -> float:
    values = [
        _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
        _unit(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        _unit(_float(row, "class_negative_risk", 0.0)),
        _unit(_float(row, "class_shell_risk", 0.0)),
        _unit(_float(row, "mahalanobis_risk", 0.0)),
        _unit(_float(row, "evt_risk", 0.0)),
    ]
    return sum(float(value >= float(threshold)) for value in values) / len(values)


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
    row_risk = _row_unknown_risk(row)
    for rank in range(1, int(top_m) + 1):
        prefix = f"class_evidence_top{rank}_"
        label = _str(row, prefix + "label", "")
        if not label:
            continue
        score = _unit(_float(row, prefix + "score", 0.0))
        pvalue = _unit(_float(row, prefix + "conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
        margin = max(0.0, _float(row, prefix + "margin", _float(row, "known_margin", 0.0)))
        support_count = max(0.0, _float(row, prefix + "support_count", _float(row, "class_conformal_support_count", 0.0)))
        class_rel = _unit(_float(row, prefix + "receiver_class_reliability", _float(row, "receiver_class_reliability", rq)))
        tail = _tail_risk(row, rank)
        unknown_risk = max(row_risk, _unit(_float(row, prefix + "unknown_risk", 0.0)))
        envelope = _unit(
            (1.0 / float(rank))
            * (
                0.25 * pvalue
                + 0.20 * score
                + 0.17 * min(1.0, margin / 0.25)
                + 0.14 * rq
                + 0.10 * class_rel
                + 0.08 * min(1.0, support_count / 8.0)
                + 0.06 * (1.0 - tail)
            )
        )
        entries.append(
            {
                "label": label,
                "rank": rank,
                "receiver_id": _str(row, "receiver_id"),
                "envelope": envelope,
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
    top_m: int,
    old_labels: set[str],
    seen_labels: set[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.extend(_candidate_entries(row, top_m=top_m))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_label[str(entry["label"])].append(entry)
    total_receivers = max(1, len({str(row.get("receiver_id", "")) for row in rows}))
    candidates: list[dict[str, Any]] = []
    for label, items in by_label.items():
        receivers = {str(item["receiver_id"]) for item in items}
        top1_receivers = {str(item["receiver_id"]) for item in items if int(item["rank"]) == 1}
        envelope = sum(float(item["envelope"]) for item in items) / max(len(items), 1)
        candidates.append(
            {
                "label": label,
                "label_set": _known_set(label, old_labels, seen_labels),
                "envelope": float(envelope),
                "max_envelope": max(float(item["envelope"]) for item in items),
                "receiver_count": len(receivers),
                "top1_receiver_count": len(top1_receivers),
                "vote_fraction": len(receivers) / float(total_receivers),
                "mean_unknown_risk": sum(float(item["unknown_risk"]) for item in items) / max(len(items), 1),
                "min_unknown_risk": min(float(item["unknown_risk"]) for item in items),
                "mean_tail_risk": sum(float(item["tail_risk"]) for item in items) / max(len(items), 1),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            item["label_set"] == "old",
            item["envelope"],
            item["vote_fraction"],
            -item["mean_unknown_risk"],
        ),
        reverse=True,
    )


def _fuse_opc_mecr_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: OpcMecrProfile,
    top_m: int,
    old_labels: set[str],
    seen_labels: set[str],
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    candidates = _label_candidates(rows, top_m=top_m, old_labels=old_labels, seen_labels=seen_labels)
    old_candidates = [item for item in candidates if item["label_set"] == "old"]
    seen_candidates = [item for item in candidates if item["label_set"] == "seen_new"]
    best_old = old_candidates[0] if old_candidates else {}
    best_seen = seen_candidates[0] if seen_candidates else {}
    best_any = candidates[0] if candidates else {}

    row_risks = [_row_unknown_risk(row) for row in rows]
    component_fractions = [_risk_component_fraction(row, profile.risk_component_threshold) for row in rows]
    mean_unknown_risk = sum(row_risks) / max(len(row_risks), 1)
    high_fraction = sum(float(risk >= profile.strong_unknown_mean_risk) for risk in row_risks) / max(len(row_risks), 1)
    component_fraction = sum(component_fractions) / max(len(component_fractions), 1)
    strong_unknown = (
        mean_unknown_risk >= float(profile.strong_unknown_mean_risk)
        and high_fraction >= float(profile.strong_unknown_high_fraction)
        and component_fraction >= float(profile.strong_unknown_component_fraction)
    )
    no_known_consensus = (
        not candidates
        or float(best_any.get("envelope", 0.0)) < float(profile.no_consensus_max_envelope)
        or int(best_any.get("receiver_count", 0)) < 1
    )
    seen_new_excluded = (
        not best_seen
        or float(best_seen.get("envelope", 0.0)) < float(profile.seen_new_exclusion_envelope)
        or float(best_seen.get("vote_fraction", 0.0)) < float(profile.min_seen_vote_fraction)
    )

    old_safe = bool(
        best_old
        and float(best_old["envelope"]) >= float(profile.min_old_envelope)
        and int(best_old["receiver_count"]) >= int(profile.min_old_receivers)
        and float(best_old["vote_fraction"]) >= float(profile.min_old_vote_fraction)
        and float(best_old["mean_unknown_risk"]) <= float(profile.max_old_unknown_risk)
    )
    seen_safe = bool(
        best_seen
        and float(best_seen["envelope"]) >= float(profile.min_seen_envelope)
        and int(best_seen["receiver_count"]) >= int(profile.min_seen_receivers)
        and float(best_seen["vote_fraction"]) >= float(profile.min_seen_vote_fraction)
        and float(best_seen["mean_unknown_risk"]) <= float(profile.max_seen_unknown_risk)
    )
    strong_unknown_overrides_old = bool(
        old_safe
        and mean_unknown_risk >= float(profile.old_override_unknown_risk)
        and component_fraction >= float(profile.old_override_component_fraction)
    )

    seen_preempts_old = bool(
        seen_safe
        and (
            not old_safe
            or float(best_seen.get("envelope", 0.0)) >= float(best_old.get("envelope", 0.0)) + 0.05
            or float(best_seen.get("vote_fraction", 0.0)) > float(best_old.get("vote_fraction", 0.0))
        )
    )

    if seen_preempts_old and not (strong_unknown and no_known_consensus):
        output_label = str(best_seen["label"])
        output_action = "accept"
        decision = "accept_seen_new_envelope"
        chosen = best_seen
    elif old_safe and not strong_unknown_overrides_old:
        output_label = str(best_old["label"])
        output_action = "accept"
        decision = "accept_old_safe"
        chosen = best_old
    elif seen_safe and not (strong_unknown and no_known_consensus):
        output_label = str(best_seen["label"])
        output_action = "accept"
        decision = "accept_seen_new_envelope"
        chosen = best_seen
    elif strong_unknown and (no_known_consensus or seen_new_excluded):
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_no_consensus"
        chosen = best_any
    elif old_safe and strong_unknown_overrides_old:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_old_strong_unknown"
        chosen = best_old
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer"
        chosen = best_any

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
        "candidate_label": str(chosen.get("label", "")),
        "candidate_label_set": str(chosen.get("label_set", "")),
        "candidate_envelope": float(chosen.get("envelope", 0.0)),
        "candidate_receiver_count": int(chosen.get("receiver_count", 0)),
        "candidate_vote_fraction": float(chosen.get("vote_fraction", 0.0)),
        "candidate_mean_unknown_risk": float(chosen.get("mean_unknown_risk", 1.0)),
        "old_safe": bool(old_safe),
        "seen_safe": bool(seen_safe),
        "strong_unknown": bool(strong_unknown),
        "strong_unknown_overrides_old": bool(strong_unknown_overrides_old),
        "no_known_consensus": bool(no_known_consensus),
        "seen_new_excluded": bool(seen_new_excluded),
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
    old_safe_accept = 0
    old_reject_or_defer = 0
    known_consensus = 0
    unknown_no_consensus = 0
    for item in event_results:
        role = str(item["role"])
        true_label = str(item["true_label"])
        output = str(item["output_label"])
        decision = str(item["decision"])
        role_total[role] += 1
        action = str(item.get("output_action", ""))
        if action == "defer" or decision.startswith("defer"):
            role_defer[role] += 1
        if role in {"old", "seen_new"} and (bool(item.get("old_safe", False)) or bool(item.get("seen_safe", False))):
            known_consensus += 1
        if role == "unknown" and bool(item.get("no_known_consensus", False)):
            unknown_no_consensus += 1
        if role == "old" and decision == "accept_old_safe":
            old_safe_accept += 1
        if role == "old" and action in {"reject_unknown", "defer"}:
            old_reject_or_defer += 1
        if role in {"old", "seen_new"}:
            per_total[role][true_label] += 1
            if output == true_label:
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
            if action == "reject_unknown" and decision == "reject_unknown_no_consensus":
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
        "old_safe_accept_rate": old_safe_accept / role_total["old"] if role_total["old"] else 0.0,
        "old_reject_rate": old_reject_or_defer / role_total["old"] if role_total["old"] else 0.0,
        "known_consensus_rate": known_consensus / known_total if known_total else 0.0,
        "unknown_no_consensus_rate": unknown_no_consensus / role_total["unknown"] if role_total["unknown"] else 0.0,
        "min_old_class_acc": min(old_rates.values()) if old_rates else 0.0,
        "min_seen_new_class_acc": min(seen_rates.values()) if seen_rates else 0.0,
        "per_old_class_acc": old_rates,
        "per_seen_new_class_acc": seen_rates,
        "confusion": dict(sorted(confusion.items())),
        "event_count": int(event_count),
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
        + 0.10 * _float(row, "old_safe_accept_rate")
        - _float(row, "unknown_FAR")
        - 0.50 * _float(row, "old_reject_rate")
        - 0.40 * _float(row, "known_defer")
        - 0.20 * _float(row, "unknown_defer")
    )


def evaluate_opc_mecr(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[OpcMecrProfile],
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
                    _fuse_opc_mecr_event(
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
                "old_safe_accept_rate": _count_value(metrics, "old_safe_accept_rate"),
                "old_reject_rate": _count_value(metrics, "old_reject_rate"),
                "known_consensus_rate": _count_value(metrics, "known_consensus_rate"),
                "unknown_no_consensus_rate": _count_value(metrics, "unknown_no_consensus_rate"),
                "event_count": int(_count_value(metrics, "event_count")),
                "old_total": int(_count_value(metrics, "old_total")),
                "seen_new_total": int(_count_value(metrics, "seen_new_total")),
                "unknown_total": int(_count_value(metrics, "unknown_total")),
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
        "algorithm": "OPC-MECR-Stage2C",
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
    parser.add_argument("--receiver_selection_policy", default="support_quality_prior", choices=["fixed_receiver_order", "support_quality_prior"])
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=20)
    parser.add_argument("--qknn_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4070801)
    parser.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse"])
    parser.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    parser.add_argument("--max_event_bytes", type=float, default=1152.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--include_event_results", action="store_true")
    parser.add_argument("--target_old_acc", type=float, default=0.99)
    parser.add_argument("--target_min_old", type=float, default=0.95)
    parser.add_argument("--target_seen_new_acc", type=float, default=0.97)
    parser.add_argument("--target_min_seen", type=float, default=0.93)
    parser.add_argument("--target_unknown_reject", type=float, default=0.99)
    return parser.parse_args(argv)


def _pcet_argv(args: argparse.Namespace) -> list[str]:
    out = args.output_json.parent / "_opc_mecr_pcet_base.json"
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
    profiles = [_profile_by_name(name) for name in _profile_names(args.profiles)]
    result = evaluate_opc_mecr(
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
    result["run_cwd"] = str(Path.cwd())
    result["python_executable"] = str(sys.executable)
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
                "best_posthoc_eval_row": result["best_posthoc_eval_row"],
                "target_receivers": result["target_receivers"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
