#!/usr/bin/env python
"""ENPC-CI episode-negative collaborative inference for Stage2-C qknn8.

ENPC-CI keeps the ADV3B02/qknn8 feature evidence path frozen and changes only
the onboard collaborative decision layer. It estimates an episode-negative
pressure from support-derived verifier/prototype evidence and then applies a
conservative multi-receiver accept/reject rule. Unknown query rows remain
evaluation-only and are not used for calibration or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_orbit_pcet_ci_eval import (  # noqa: E402
    _count_value,
    _float,
    _positive_int,
    _str,
    _target_pass,
    _write_csv,
    parse_args as _parse_pcet_args,
    run_pcet_ci,
)


UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class EnpcProfile:
    name: str
    description: str
    accept_confidence: float
    accept_margin: float
    accept_max_pressure: float
    support_accept_confidence: float
    reject_pressure: float
    reject_min_high_fraction: float
    reject_min_disagreement: float
    min_accept_receivers: int


PROFILES: tuple[EnpcProfile, ...] = (
    EnpcProfile(
        name="enpc_known_anchor",
        description="loose known-anchor route, used to measure retention upper bound",
        accept_confidence=0.18,
        accept_margin=0.00,
        accept_max_pressure=1.00,
        support_accept_confidence=0.35,
        reject_pressure=1.10,
        reject_min_high_fraction=1.10,
        reject_min_disagreement=1.10,
        min_accept_receivers=1,
    ),
    EnpcProfile(
        name="enpc_balanced",
        description="episode-negative pressure with support-protected known acceptance",
        accept_confidence=0.30,
        accept_margin=0.025,
        accept_max_pressure=0.78,
        support_accept_confidence=0.58,
        reject_pressure=0.70,
        reject_min_high_fraction=0.50,
        reject_min_disagreement=0.55,
        min_accept_receivers=1,
    ),
    EnpcProfile(
        name="enpc_old80_unknown_probe",
        description="OLD80-preserving probe that raises unknown rejection without full known collapse",
        accept_confidence=0.42,
        accept_margin=0.00,
        accept_max_pressure=0.45,
        support_accept_confidence=0.55,
        reject_pressure=0.45,
        reject_min_high_fraction=0.20,
        reject_min_disagreement=1.20,
        min_accept_receivers=1,
    ),
    EnpcProfile(
        name="enpc_unknown_strict",
        description="strict episode-negative pressure rejection",
        accept_confidence=0.42,
        accept_margin=0.05,
        accept_max_pressure=0.62,
        support_accept_confidence=0.70,
        reject_pressure=0.58,
        reject_min_high_fraction=0.40,
        reject_min_disagreement=0.45,
        min_accept_receivers=2,
    ),
)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    unknown = sorted(set(names) - known)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown ENPC-CI profile(s): {', '.join(unknown)}")
    return names


def _role(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"target_old", "old"}:
        return "old"
    if text in {"target_new", "seen_new", "seennew", "new"}:
        return "seen_new"
    if text in {"target_unknown", "unknown", "unk"}:
        return "unknown"
    raise ValueError(f"unknown role {value!r}")


def _items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    try:
        return [str(part).strip() for part in value if str(part).strip()]
    except TypeError:
        return [str(value).strip()] if str(value).strip() else []


def _metadata_classes(metadata: Mapping[str, Any], key: str, observed: Iterable[str]) -> list[str]:
    values = _items(metadata.get(key))
    if values:
        return sorted(set(values))
    return sorted(set(str(item) for item in observed if str(item)))


def _parse_collab_counts(spec: str | Sequence[int] | None, receiver_count: int) -> list[int]:
    if spec is None or str(spec).strip().lower() in {"", "all", "*", "1..n"}:
        return list(range(1, int(receiver_count) + 1))
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.replace(";", ",").split(",") if part.strip()]
    else:
        parts = [str(part) for part in spec]
    out: list[int] = []
    for part in parts:
        k = int(part)
        if k < 1 or k > int(receiver_count):
            raise ValueError(f"collaborative receiver count {k} is outside 1..{receiver_count}")
        if k not in out:
            out.append(k)
    return out


def augment_enpc_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    gap_scale: float = 0.20,
    pressure_floor: float = 0.0,
) -> list[dict[str, Any]]:
    """Add support-only episode-negative pressure fields."""
    out: list[dict[str, Any]] = []
    gap_den = max(float(gap_scale), 1e-6)
    for source in evidence:
        row = dict(source)
        score = _clip01(_float(row, "known_score", _float(row, "class_evidence_top1_score", 0.0)))
        margin = _float(row, "known_margin", _float(row, "class_evidence_top1_margin", 0.0))
        pvalue = _clip01(_float(row, "class_verifier_top1_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
        reliability = _clip01(
            _float(
                row,
                "class_verifier_top1_receiver_class_reliability",
                _float(row, "receiver_class_reliability", 0.0),
            )
        )
        base_risk = max(
            _clip01(_float(row, "unknown_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        )
        verified = _clip01(_float(row, "class_verifier_top1_verified_score", score))
        second_verified = _clip01(_float(row, "class_verifier_second_verified_score", 0.0))
        verifier_margin = max(0.0, verified - second_verified)
        negative = _clip01(_float(row, "class_verifier_top1_class_negative_risk", 0.0))
        verifier_unknown = _clip01(_float(row, "class_verifier_top1_unknown_risk", 0.0))
        shell = max(
            _clip01(_float(row, "class_verifier_top1_class_shell_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_class_shell_risk", 0.0)),
            _clip01(_float(row, "class_shell_risk", 0.0)),
        )
        ambiguity = _clip01(1.0 - max(0.0, margin) / gap_den)
        verifier_ambiguity = _clip01(1.0 - verifier_margin / gap_den)
        changed = 1.0 if _str(row, "class_verifier_changed", "0").lower() in {"1", "true", "yes"} else 0.0
        support_confidence = _clip01(
            0.30 * score
            + 0.25 * pvalue
            + 0.20 * reliability
            + 0.15 * _clip01(max(0.0, margin) / gap_den)
            + 0.10 * verified
        )
        episode_negative_pressure = max(
            float(pressure_floor),
            _clip01(
                0.28 * base_risk
                + 0.18 * ambiguity
                + 0.16 * verifier_ambiguity
                + 0.14 * verifier_unknown
                + 0.10 * (1.0 - pvalue)
                + 0.06 * (1.0 - reliability)
                + 0.04 * shell
                + 0.03 * changed
                + 0.01 * negative
            ),
        )
        row["enpc_support_confidence"] = support_confidence
        row["enpc_episode_negative_pressure"] = episode_negative_pressure
        row["enpc_ambiguity"] = ambiguity
        row["enpc_verifier_ambiguity"] = verifier_ambiguity
        row["enpc_base_unknown_risk"] = base_risk
        row["enpc_verifier_margin"] = verifier_margin
        row["enpc_score"] = score
        row["enpc_margin"] = margin
        row["enpc_pvalue"] = pvalue
        row["enpc_reliability"] = reliability
        out.append(row)
    return out


def _receiver_quality(row: Mapping[str, Any]) -> float:
    return float(
        0.45 * _clip01(_float(row, "enpc_support_confidence", 0.0))
        + 0.25 * (1.0 - _clip01(_float(row, "enpc_episode_negative_pressure", 0.0)))
        + 0.20 * _clip01(_float(row, "receiver_class_reliability", 0.0))
        + 0.10 * _clip01(_float(row, "known_score", 0.0))
    )


def _select_rows(rows: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-_receiver_quality(row), _float(row, "latency_ms", 0.0), _str(row, "receiver_id")),
    )[: int(k)]


def _fuse_enpc_event(selected: Sequence[Mapping[str, Any]], profile: EnpcProfile) -> dict[str, Any]:
    label_scores: defaultdict[str, float] = defaultdict(float)
    label_counts: Counter[str] = Counter()
    pressures = []
    high_pressures = 0
    for row in selected:
        label = _str(row, "predicted_label")
        if label:
            weight = max(1e-6, _float(row, "enpc_support_confidence", 0.0)) * (
                1.0 - 0.45 * _clip01(_float(row, "enpc_episode_negative_pressure", 0.0))
            )
            label_scores[label] += max(0.0, weight)
            label_counts[label] += 1
        pressure = _clip01(_float(row, "enpc_episode_negative_pressure", 0.0))
        pressures.append(pressure)
        high_pressures += int(pressure >= float(profile.reject_pressure))
    if label_scores:
        ranked = sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))
        label = ranked[0][0]
        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    else:
        label, top_score, second_score = "", 0.0, 0.0
    total_score = sum(label_scores.values())
    confidence = 0.0 if total_score <= 0.0 else top_score / max(total_score, 1e-6)
    vote_margin = 0.0 if total_score <= 0.0 else (top_score - second_score) / max(total_score, 1e-6)
    mean_pressure = sum(pressures) / max(len(pressures), 1)
    max_pressure = max(pressures) if pressures else 0.0
    high_fraction = high_pressures / max(len(pressures), 1)
    top_fraction = label_counts[label] / max(len(selected), 1) if label else 0.0
    disagreement = 1.0 - top_fraction
    support_conf_floor = min((_float(row, "enpc_support_confidence", 0.0) for row in selected), default=0.0)
    support_conf_mean = sum(_float(row, "enpc_support_confidence", 0.0) for row in selected) / max(len(selected), 1)
    strong_known = bool(
        label
        and len(selected) >= int(profile.min_accept_receivers)
        and confidence >= float(profile.accept_confidence)
        and vote_margin >= float(profile.accept_margin)
        and (
            mean_pressure <= float(profile.accept_max_pressure)
            or support_conf_mean >= float(profile.support_accept_confidence)
        )
    )
    reject_ready = bool(
        mean_pressure >= float(profile.reject_pressure)
        or high_fraction >= float(profile.reject_min_high_fraction)
        or (disagreement >= float(profile.reject_min_disagreement) and mean_pressure >= 0.50)
    )
    if strong_known:
        decision = "accept"
        output_label = label
        stage = "support_protected_accept"
    elif reject_ready:
        decision = "unknown_reject"
        output_label = UNKNOWN_LABEL
        stage = "episode_negative_reject"
    else:
        decision = "defer"
        output_label = ""
        stage = "insufficient_known_or_unknown_evidence"
    return {
        "decision": decision,
        "output_label": output_label,
        "enpc_stage": stage,
        "enpc_label": label,
        "enpc_confidence": float(confidence),
        "enpc_vote_margin": float(vote_margin),
        "enpc_mean_pressure": float(mean_pressure),
        "enpc_max_pressure": float(max_pressure),
        "enpc_high_pressure_fraction": float(high_fraction),
        "enpc_label_disagreement": float(disagreement),
        "enpc_support_conf_floor": float(support_conf_floor),
        "enpc_support_conf_mean": float(support_conf_mean),
    }


def _safe_rate(num: int, den: int) -> float:
    return 0.0 if int(den) <= 0 else float(num) / float(den)


def _finalize(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_old_labels: Sequence[str],
    expected_seen_new_labels: Sequence[str],
    k: int,
    min_required_receivers: int,
    excluded: int,
) -> dict[str, Any]:
    role_totals: Counter[str] = Counter()
    role_correct: Counter[str] = Counter()
    unknown_rejected = 0
    unknown_false_accept = 0
    known_defer = 0
    unknown_defer = 0
    per_old_total: Counter[str] = Counter({label: 0 for label in expected_old_labels})
    per_old_correct: Counter[str] = Counter({label: 0 for label in expected_old_labels})
    per_seen_total: Counter[str] = Counter({label: 0 for label in expected_seen_new_labels})
    per_seen_correct: Counter[str] = Counter({label: 0 for label in expected_seen_new_labels})
    per_class_decisions: dict[str, defaultdict[str, Counter[str]]] = {
        "old": defaultdict(Counter),
        "seen_new": defaultdict(Counter),
    }
    per_class_outputs: dict[str, defaultdict[str, Counter[str]]] = {
        "old": defaultdict(Counter),
        "seen_new": defaultdict(Counter),
    }
    open_set_confusion: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for event in events:
        role = _role(event.get("role"))
        truth = _str(event, "true_label", UNKNOWN_LABEL if role == "unknown" else "")
        decision = _str(event, "decision")
        output = _str(event, "output_label")
        decision_counts[decision] += 1
        role_totals[role] += 1
        if role == "unknown":
            unknown_rejected += int(decision == "unknown_reject")
            unknown_false_accept += int(decision == "accept")
            unknown_defer += int(decision == "defer")
            if decision == "accept":
                open_set_confusion[f"unknown->{output or UNKNOWN_LABEL}"] += 1
            elif decision == "unknown_reject":
                open_set_confusion["unknown->reject"] += 1
            else:
                open_set_confusion[f"unknown->{decision}"] += 1
            continue
        if role == "old":
            per_old_total[truth] += 1
        if role == "seen_new":
            per_seen_total[truth] += 1
        if decision == "defer":
            known_defer += 1
        per_class_decisions[role][truth][decision] += 1
        if decision == "accept" and output:
            per_class_outputs[role][truth][output] += 1
            if output == truth:
                open_set_confusion[f"{role}->correct"] += 1
            else:
                out_set = "seen_new" if output in expected_seen_new_labels else "old"
                open_set_confusion[f"{role}->{out_set}"] += 1
        elif decision == "unknown_reject":
            open_set_confusion[f"{role}->reject"] += 1
        else:
            open_set_confusion[f"{role}->{decision}"] += 1
        correct = int(decision == "accept" and output == truth)
        role_correct[role] += correct
        if role == "old":
            per_old_correct[truth] += correct
        if role == "seen_new":
            per_seen_correct[truth] += correct
    per_old = {
        label: _safe_rate(per_old_correct[label], per_old_total[label]) for label in sorted(per_old_total)
    }
    per_seen = {
        label: _safe_rate(per_seen_correct[label], per_seen_total[label]) for label in sorted(per_seen_total)
    }
    bytes_per_event = 0.0
    latency_ms = 0.0
    if events:
        bytes_per_event = sum(float(event.get("bytes_per_event", 0.0)) for event in events) / len(events)
        latency_ms = max(float(event.get("latency_ms", 0.0)) for event in events)
    return {
        "k": int(k),
        "min_required_receivers": int(min_required_receivers),
        "excluded_incomplete_groups": int(excluded),
        "event_count": int(len(events)),
        "old_acc": _safe_rate(role_correct["old"], role_totals["old"]),
        "seen_new_acc": _safe_rate(role_correct["seen_new"], role_totals["seen_new"]),
        "unknown_reject_rate": _safe_rate(unknown_rejected, role_totals["unknown"]),
        "unknown_FAR": _safe_rate(unknown_false_accept, role_totals["unknown"]),
        "known_defer_rate": _safe_rate(known_defer, role_totals["old"] + role_totals["seen_new"]),
        "unknown_defer_rate": _safe_rate(unknown_defer, role_totals["unknown"]),
        "per_old_class_acc": per_old,
        "per_old_class_total": dict(sorted(per_old_total.items())),
        "per_old_class_decision_counts": {
            label: dict(sorted(per_class_decisions["old"][label].items())) for label in sorted(per_old_total)
        },
        "per_old_class_output_counts": {
            label: dict(sorted(per_class_outputs["old"][label].items())) for label in sorted(per_old_total)
        },
        "per_seen_new_class_acc": per_seen,
        "per_seen_new_class_total": dict(sorted(per_seen_total.items())),
        "per_seen_new_class_decision_counts": {
            label: dict(sorted(per_class_decisions["seen_new"][label].items()))
            for label in sorted(per_seen_total)
        },
        "per_seen_new_class_output_counts": {
            label: dict(sorted(per_class_outputs["seen_new"][label].items())) for label in sorted(per_seen_total)
        },
        "min_old_class_acc": min(per_old.values()) if per_old else 0.0,
        "min_seen_new_class_acc": min(per_seen.values()) if per_seen else 0.0,
        "open_set_confusion": dict(sorted(open_set_confusion.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "bytes_per_event": float(bytes_per_event),
        "latency_ms_pessimistic": float(latency_ms),
    }


def evaluate_enpc_collaborative_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    profile: EnpcProfile,
    collab_counts: str | Sequence[int] | None,
    collab_group_policy: str,
    partial_collab_min_receivers: int,
    max_event_bytes: float,
    max_event_latency_ms: float,
    metadata: Mapping[str, Any],
    include_event_results: bool = False,
) -> dict[str, Any]:
    receivers = sorted({_str(row, "receiver_id") for row in evidence if _str(row, "receiver_id")})
    counts = _parse_collab_counts(collab_counts, len(receivers))
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        groups[_str(row, "event_id")].append(row)
    observed_old = [_str(row, "true_label") for row in evidence if _role(row.get("role")) == "old"]
    observed_seen = [_str(row, "true_label") for row in evidence if _role(row.get("role")) == "seen_new"]
    expected_old = _metadata_classes(metadata, "old_tx_ids", observed_old)
    expected_seen = _metadata_classes(metadata, "seen_new_tx_ids", observed_seen)
    out_counts: dict[str, Any] = {}
    for k in counts:
        if str(collab_group_policy) == "available_up_to_k":
            min_required = min(int(k), max(int(partial_collab_min_receivers), 1))
        else:
            min_required = int(k)
        event_results = []
        excluded = 0
        for event_id, rows in groups.items():
            if len(rows) < min_required:
                excluded += 1
                continue
            selected = _select_rows(rows, min(int(k), len(rows)))
            bytes_used = sum(_float(row, "bytes", 0.0) for row in selected)
            latency_used = max((_float(row, "latency_ms", 0.0) for row in selected), default=0.0)
            fused = _fuse_enpc_event(selected, profile)
            first = selected[0]
            fused.update(
                {
                    "event_id": str(event_id),
                    "role": _role(first.get("role")),
                    "true_label": _str(first, "true_label", UNKNOWN_LABEL),
                    "selected_receiver_ids": ",".join(_str(row, "receiver_id") for row in selected),
                    "selected_receiver_predictions": ",".join(
                        f"{_str(row, 'receiver_id')}:{_str(row, 'predicted_label')}" for row in selected
                    ),
                    "bytes_per_event": float(bytes_used),
                    "latency_ms": float(latency_used),
                    "resource_pass": bool(
                        (float(max_event_bytes) <= 0.0 or bytes_used <= float(max_event_bytes))
                        and (float(max_event_latency_ms) <= 0.0 or latency_used <= float(max_event_latency_ms))
                    ),
                }
            )
            event_results.append(fused)
        metrics = _finalize(
            event_results,
            expected_old_labels=expected_old,
            expected_seen_new_labels=expected_seen,
            k=int(k),
            min_required_receivers=int(min_required),
            excluded=int(excluded),
        )
        if include_event_results:
            metrics["event_results"] = event_results
        out_counts[str(k)] = metrics
    return {
        "enabled": True,
        "protocol": "enpc_collaborative_open_set_qknn_evidence",
        "fusion_policy": "enpc_ci",
        "receiver_count": int(len(receivers)),
        "observed_receiver_ids": receivers,
        "collab_group_policy": str(collab_group_policy),
        "partial_collab_min_receivers": int(partial_collab_min_receivers),
        "threshold_selection_label_scope": str(metadata.get("threshold_scope", "support_virtual_unknown")),
        "unknown_query_eval_only": True,
        "profile": profile.__dict__,
        "counts": out_counts,
    }


def run_enpc_ci(args: argparse.Namespace) -> dict[str, Any]:
    base_args = argparse.Namespace(**vars(args))
    base_args.profiles = "pcet_known_preserving"
    base = run_pcet_ci(base_args)
    evidence = augment_enpc_evidence(
        base.pop("_evidence_rows"),
        gap_scale=float(args.enpc_gap_scale),
        pressure_floor=float(args.enpc_pressure_floor),
    )
    metadata = dict(base["qknn_metadata"])
    metadata["algorithm_wrapper"] = "ENPC-CI"
    metadata["unknown_query_eval_only"] = True
    metadata["labeled_unknown_support_used_for_boundary_fit"] = False
    metadata["in_orbit_method"] = "qknn8"
    metadata["enpc_components"] = [
        "episode_negative_pressure",
        "support_confidence",
        "multi_receiver_conservative_accept",
    ]
    requested = set(_profile_names(args.enpc_profiles))
    profile_map = {profile.name: profile for profile in PROFILES}
    profile_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for profile_name in [profile.name for profile in PROFILES if profile.name in requested]:
        profile = profile_map[profile_name]
        result = evaluate_enpc_collaborative_evidence(
            evidence,
            profile=profile,
            collab_counts=args.collab_counts,
            collab_group_policy=str(args.collab_group_policy),
            partial_collab_min_receivers=int(args.partial_collab_min_receivers),
            max_event_bytes=float(args.max_event_bytes),
            max_event_latency_ms=float(args.max_event_latency_ms),
            metadata=metadata,
            include_event_results=bool(args.include_event_results),
        )
        profile_results[profile.name] = result
        for collab_count, counts in sorted(result["counts"].items(), key=lambda item: int(item[0])):
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(collab_count),
                "old_acc": _count_value(counts, "old_acc"),
                "min_old": _count_value(counts, "min_old_class_acc", "min_old_acc"),
                "seen_new_acc": _count_value(counts, "seen_new_acc"),
                "min_seen": _count_value(counts, "min_seen_new_class_acc", "min_seen_new_acc"),
                "unknown_reject": _count_value(counts, "unknown_reject_rate", "unknown_reject_acc"),
                "unknown_FAR": _count_value(counts, "unknown_FAR", "unknown_far"),
                "known_defer": _count_value(counts, "known_defer_rate"),
                "unknown_defer": _count_value(counts, "unknown_defer_rate"),
                "bytes_per_event": _count_value(counts, "bytes_per_event", "mean_evidence_bytes"),
                "latency_ms": _count_value(counts, "latency_ms_pessimistic", "mean_latency_ms"),
                "target_old_acc": float(args.target_old_acc),
                "target_min_old": float(args.target_min_old),
                "target_seen_new_acc": float(args.target_seen_new_acc),
                "target_min_seen": float(args.target_min_seen),
                "target_unknown_reject": float(args.target_unknown_reject),
            }
            row["target_pass"] = _target_pass(row)
            row["resource_pass"] = (
                (float(args.max_event_bytes) <= 0.0 or row["bytes_per_event"] <= float(args.max_event_bytes))
                and (
                    float(args.max_event_latency_ms) <= 0.0
                    or row["latency_ms"] <= float(args.max_event_latency_ms)
                )
            )
            summary_rows.append(row)
    best_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["target_pass"],
            row["old_acc"] >= 0.80,
            row["unknown_reject"],
            row["old_acc"],
            row["seen_new_acc"],
            -row["known_defer"],
        ),
        reverse=True,
    )
    return {
        "algorithm": "ENPC-CI",
        "feature_npz": str(args.feature_npz),
        "profiles": [profile.__dict__ for profile in PROFILES if profile.name in requested],
        "base_pcet_known_preserving": base,
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_joint_row": best_rows[0] if best_rows else None,
        "qknn_metadata": metadata,
        "evidence_row_count": len(evidence),
        "target_gates": {
            "old_acc": float(args.target_old_acc),
            "min_old": float(args.target_min_old),
            "seen_new_acc": float(args.target_seen_new_acc),
            "min_seen": float(args.target_min_seen),
            "unknown_reject": float(args.target_unknown_reject),
        },
        "resource_constraints": {
            "evidence_packet_bytes": float(args.evidence_packet_bytes),
            "max_event_bytes": float(args.max_event_bytes),
            "max_event_latency_ms": float(args.max_event_latency_ms),
            "latency_budget_ms": float(args.latency_budget_ms),
        },
        "_evidence_rows": evidence,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    enpc_parser = argparse.ArgumentParser(add_help=False)
    enpc_parser.add_argument("--profiles", dest="enpc_profiles", default="all")
    enpc_parser.add_argument("--enpc_gap_scale", type=float, default=0.20)
    enpc_parser.add_argument("--enpc_pressure_floor", type=float, default=0.0)
    enpc_args, remaining = enpc_parser.parse_known_args(raw)
    args = _parse_pcet_args(remaining)
    for key, value in vars(enpc_args).items():
        setattr(args, key, value)
    _profile_names(args.enpc_profiles)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_enpc_ci(args)
    evidence_rows = result.pop("_evidence_rows")
    result["run_command_argv"] = [str(item) for item in sys.argv]
    result["run_cwd"] = str(Path.cwd())
    result["python_executable"] = str(sys.executable)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
