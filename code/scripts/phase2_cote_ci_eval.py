#!/usr/bin/env python
"""COTE-CI candidate-over-topM collaborative inference for Stage2-C qknn8.

COTE-CI is a deployable decision-layer experiment. It reuses PCET/qknn8
evidence, aggregates each receiver's top-M class evidence, protects known
old/seen-new candidates when support quality is sufficient, and rejects unknown
only when cross-receiver risk remains high without a known shield. target_unknown
rows are evaluation-only.
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
class CoteProfile:
    name: str
    description: str
    min_known_quality: float
    min_old_quality: float
    min_seen_quality: float
    min_label_receivers: int
    min_label_weight_fraction: float
    unknown_risk_threshold: float
    unknown_high_fraction: float
    risk_component_threshold: float
    max_accept_risk: float


PROFILES: tuple[CoteProfile, ...] = (
    CoteProfile(
        name="cote_known_anchor",
        description="known-retention candidate-over-topM fusion",
        min_known_quality=0.26,
        min_old_quality=0.24,
        min_seen_quality=0.22,
        min_label_receivers=1,
        min_label_weight_fraction=0.30,
        unknown_risk_threshold=0.92,
        unknown_high_fraction=0.80,
        risk_component_threshold=0.70,
        max_accept_risk=1.0,
    ),
    CoteProfile(
        name="cote_balanced",
        description="known shield plus cross-receiver unknown confirmation",
        min_known_quality=0.32,
        min_old_quality=0.30,
        min_seen_quality=0.28,
        min_label_receivers=2,
        min_label_weight_fraction=0.36,
        unknown_risk_threshold=0.82,
        unknown_high_fraction=0.60,
        risk_component_threshold=0.64,
        max_accept_risk=0.94,
    ),
    CoteProfile(
        name="cote_unknown_confirm",
        description="stricter unknown confirmation for diagnostic safety",
        min_known_quality=0.38,
        min_old_quality=0.36,
        min_seen_quality=0.34,
        min_label_receivers=2,
        min_label_weight_fraction=0.42,
        unknown_risk_threshold=0.72,
        unknown_high_fraction=0.50,
        risk_component_threshold=0.55,
        max_accept_risk=0.88,
    ),
)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value)


def _unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


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
        raise argparse.ArgumentTypeError(f"unknown COTE-CI profile(s): {', '.join(missing)}")
    return names


def _receiver_quality(row: Mapping[str, Any]) -> float:
    reliability = _unit(_float(row, "reliability", 1.0))
    support_density = _unit(_float(row, "support_density", reliability))
    receiver_class = _unit(_float(row, "receiver_class_reliability", _float(row, "class_evidence_top1_receiver_class_reliability", 1.0)))
    pvalue = _unit(_float(row, "class_conformal_pvalue", _float(row, "class_evidence_top1_conformal_pvalue", 0.0)))
    return _unit(0.30 * reliability + 0.25 * support_density + 0.25 * receiver_class + 0.20 * pvalue)


def _tail_risk(row: Mapping[str, Any], rank: int) -> float:
    prefix = f"class_evidence_top{rank}_"
    return max(
        _unit(_float(row, prefix + "mahalanobis_risk", _float(row, "mahalanobis_risk", 0.0))),
        _unit(_float(row, prefix + "evt_risk", _float(row, "evt_risk", 0.0))),
        _unit(_float(row, prefix + "class_shell_risk", _float(row, "class_shell_risk", 0.0))),
        _unit(_float(row, prefix + "class_negative_risk", _float(row, "class_negative_risk", 0.0))),
    )


def _row_unknown_risk(row: Mapping[str, Any]) -> float:
    return max(
        _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
        _unit(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        _unit(_float(row, "oldness_risk", 0.0)),
    )


def _candidate_entries(row: Mapping[str, Any], *, top_m: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    receiver_quality = _receiver_quality(row)
    for rank in range(1, int(top_m) + 1):
        prefix = f"class_evidence_top{rank}_"
        label = _str(row, prefix + "label", "")
        if not label:
            continue
        score = _unit(_float(row, prefix + "score", 0.0))
        margin = max(0.0, _float(row, prefix + "margin", _float(row, "known_margin", 0.0)))
        pvalue = _unit(_float(row, prefix + "conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
        receiver_class = _unit(_float(row, prefix + "receiver_class_reliability", _float(row, "receiver_class_reliability", 1.0)))
        tail = _tail_risk(row, rank)
        unknown_risk = _unit(_float(row, prefix + "unknown_risk", _row_unknown_risk(row)))
        rank_decay = 1.0 / float(rank)
        quality = _unit(
            rank_decay
            * (
                0.34 * score
                + 0.18 * min(1.0, margin / 0.30)
                + 0.18 * pvalue
                + 0.14 * receiver_quality
                + 0.10 * receiver_class
                + 0.06 * (1.0 - tail)
            )
        )
        entries.append(
            {
                "label": label,
                "rank": rank,
                "quality": quality,
                "score": score,
                "margin": margin,
                "pvalue": pvalue,
                "receiver_class_reliability": receiver_class,
                "tail_risk": tail,
                "unknown_risk": unknown_risk,
                "receiver_id": _str(row, "receiver_id"),
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


def _known_set(label: str, old_labels: set[str], seen_labels: set[str]) -> str:
    if label in old_labels:
        return "old"
    if label in seen_labels:
        return "seen_new"
    return ""


def _fuse_cote_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: CoteProfile,
    top_m: int,
    old_labels: set[str],
    seen_labels: set[str],
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.extend(_candidate_entries(row, top_m=top_m))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_label[str(entry["label"])].append(entry)
    total_quality = sum(float(entry["quality"]) for entry in entries) or 1.0
    candidates: list[dict[str, Any]] = []
    for label, items in by_label.items():
        label_quality = sum(float(item["quality"]) for item in items)
        receivers = {str(item["receiver_id"]) for item in items}
        best_risk = min(float(item["unknown_risk"]) for item in items)
        mean_tail = sum(float(item["tail_risk"]) for item in items) / max(len(items), 1)
        candidates.append(
            {
                "label": label,
                "label_set": _known_set(label, old_labels, seen_labels),
                "quality": label_quality,
                "quality_fraction": label_quality / total_quality,
                "receiver_count": len(receivers),
                "best_unknown_risk": best_risk,
                "mean_tail_risk": mean_tail,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["label_set"] in {"old", "seen_new"},
            float(item["quality"]),
            -float(item["best_unknown_risk"]),
            str(item["label"]),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else {
        "label": UNKNOWN_LABEL,
        "label_set": "",
        "quality": 0.0,
        "quality_fraction": 0.0,
        "receiver_count": 0,
        "best_unknown_risk": 1.0,
        "mean_tail_risk": 1.0,
    }
    selected_risks = [_row_unknown_risk(row) for row in rows]
    component_high = [
        max(
            _unit(_float(row, "pcet_unknown_risk", _float(row, "unknown_risk", 0.0))),
            _unit(_float(row, "class_negative_risk", 0.0)),
            _unit(_float(row, "oldness_risk", 0.0)),
            _unit(_float(row, "mahalanobis_risk", 0.0)),
        )
        for row in rows
    ]
    mean_unknown_risk = sum(selected_risks) / max(len(selected_risks), 1)
    high_fraction = sum(v >= float(profile.unknown_risk_threshold) for v in selected_risks) / max(len(selected_risks), 1)
    component_fraction = sum(v >= float(profile.risk_component_threshold) for v in component_high) / max(len(component_high), 1)
    label_set = str(best["label_set"])
    min_quality = float(profile.min_known_quality)
    if label_set == "old":
        min_quality = min(min_quality, float(profile.min_old_quality))
    elif label_set == "seen_new":
        min_quality = min(min_quality, float(profile.min_seen_quality))
    known_accept = bool(
        label_set in {"old", "seen_new"}
        and float(best["quality"]) >= min_quality
        and int(best["receiver_count"]) >= int(profile.min_label_receivers)
        and float(best["quality_fraction"]) >= float(profile.min_label_weight_fraction)
        and float(best["best_unknown_risk"]) <= float(profile.max_accept_risk)
    )
    unknown_confirm = bool(
        (not known_accept)
        and mean_unknown_risk >= float(profile.unknown_risk_threshold)
        and high_fraction >= float(profile.unknown_high_fraction)
        and component_fraction >= float(profile.unknown_high_fraction)
    )
    if known_accept:
        output_label = str(best["label"])
        decision = "accept_known"
    elif unknown_confirm:
        output_label = UNKNOWN_LABEL
        decision = "reject_unknown"
    else:
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
        "candidate_label": str(best["label"]),
        "candidate_label_set": label_set,
        "candidate_quality": float(best["quality"]),
        "candidate_quality_fraction": float(best["quality_fraction"]),
        "candidate_receiver_count": int(best["receiver_count"]),
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
        elif role == "unknown" and output == UNKNOWN_LABEL and decision == "reject_unknown":
            role_correct[role] += 1
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
    }


def evaluate_cote(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[CoteProfile],
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
                    _fuse_cote_event(
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
            metrics["latency_ms_pessimistic"] = max((float(item["latency_ms"]) for item in selected_events), default=0.0)
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
                "target_old_acc": float(target_gates["old_acc"]),
                "target_min_old": float(target_gates["min_old"]),
                "target_seen_new_acc": float(target_gates["seen_new_acc"]),
                "target_min_seen": float(target_gates["min_seen"]),
                "target_unknown_reject": float(target_gates["unknown_reject"]),
                "resource_pass": bool(metrics["resource_pass"]),
            }
            row["target_pass"] = _target_pass(row)
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": profile.__dict__,
            "receiver_count": len(target_receivers),
            "observed_receiver_ids": target_receivers,
            "counts": count_results,
        }
    best_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["target_pass"],
            row["old_acc"] >= 0.80,
            row["seen_new_acc"] >= 0.60,
            row["unknown_reject"],
            row["old_acc"],
            row["seen_new_acc"],
            -row["known_defer"],
        ),
        reverse=True,
    )
    return {
        "algorithm": "COTE-CI",
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_joint_row": best_rows[0] if best_rows else None,
        "target_receivers": target_receivers,
        "old_labels": old_labels,
        "seen_new_labels": seen_labels,
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "resource_constraints": {
            "max_event_bytes": float(max_event_bytes),
            "max_event_latency_ms": float(max_event_latency_ms),
        },
    }


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
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070501)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
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
    out = args.output_json.parent / "_cote_pcet_base.json"
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
    requested = set(_profile_names(args.profiles))
    profiles = [profile for profile in PROFILES if profile.name in requested]
    result = evaluate_cote(
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
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    print(json.dumps({"best_joint_row": result["best_joint_row"], "target_receivers": result["target_receivers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
