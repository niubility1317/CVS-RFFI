#!/usr/bin/env python
"""KERA-CI known-enrollment-repair collaborative inference for Stage2-C.

KERA-CI reuses the AOR support-only adapter/evidence builder, but changes the
fusion order. Enrolled seen-new classes get a known-enrollment gate before the
old-anchor guard, so old protection cannot automatically suppress registered
new classes. target_unknown remains evaluation-only.
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

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import parse_collab_counts  # noqa: E402
from phase2_aor_adapter_ci_eval import (  # noqa: E402
    DEFER_LABEL,
    UNKNOWN_LABEL,
    _aggregate_label,
    _finalize,
    _float,
    _joint_score,
    _p95,
    _role,
    _select_receivers,
    _str,
    _unit,
    build_aor_evidence,
)
from phase2_collaborative_open_set_qknn_eval import load_feature_npz  # noqa: E402
from phase2_orbit_pcet_ci_eval import _count_value, _target_pass, _write_csv  # noqa: E402


@dataclass(frozen=True)
class KeraProfile:
    name: str
    description: str
    adapter_alpha: float
    seen_new_known_min: float
    seen_new_margin_min: float
    seen_new_vote_min: float
    seen_new_old_gap_floor: float
    old_anchor_min: float
    old_known_min: float
    old_margin_min: float
    old_vote_min: float
    known_score_min: float
    known_margin_min: float
    known_vote_min: float
    known_min_receivers: int
    unknown_score_min: float
    unknown_fraction_min: float
    low_margin_fraction_min: float
    disagreement_min: float
    known_consensus_rescue_min: float


PROFILES: tuple[KeraProfile, ...] = (
    KeraProfile(
        name="kera_primary",
        description="seen-new enrollment gate before old-anchor guard",
        adapter_alpha=0.18,
        seen_new_known_min=0.48,
        seen_new_margin_min=0.00,
        seen_new_vote_min=0.25,
        seen_new_old_gap_floor=-0.10,
        old_anchor_min=0.66,
        old_known_min=0.58,
        old_margin_min=0.01,
        old_vote_min=0.34,
        known_score_min=0.48,
        known_margin_min=0.00,
        known_vote_min=0.25,
        known_min_receivers=1,
        unknown_score_min=0.58,
        unknown_fraction_min=0.50,
        low_margin_fraction_min=0.50,
        disagreement_min=0.50,
        known_consensus_rescue_min=0.72,
    ),
    KeraProfile(
        name="kera_old_guard",
        description="conservative old guard with explicit seen-new rescue",
        adapter_alpha=0.10,
        seen_new_known_min=0.54,
        seen_new_margin_min=0.01,
        seen_new_vote_min=0.34,
        seen_new_old_gap_floor=0.00,
        old_anchor_min=0.60,
        old_known_min=0.50,
        old_margin_min=0.00,
        old_vote_min=0.25,
        known_score_min=0.44,
        known_margin_min=0.00,
        known_vote_min=0.25,
        known_min_receivers=1,
        unknown_score_min=0.70,
        unknown_fraction_min=0.67,
        low_margin_fraction_min=0.67,
        disagreement_min=0.67,
        known_consensus_rescue_min=0.64,
    ),
    KeraProfile(
        name="kera_unknown_probe",
        description="unknown probe after known-enrollment gates fail",
        adapter_alpha=0.24,
        seen_new_known_min=0.46,
        seen_new_margin_min=0.00,
        seen_new_vote_min=0.25,
        seen_new_old_gap_floor=-0.15,
        old_anchor_min=0.72,
        old_known_min=0.66,
        old_margin_min=0.03,
        old_vote_min=0.50,
        known_score_min=0.54,
        known_margin_min=0.02,
        known_vote_min=0.34,
        known_min_receivers=1,
        unknown_score_min=0.50,
        unknown_fraction_min=0.50,
        low_margin_fraction_min=0.50,
        disagreement_min=0.34,
        known_consensus_rescue_min=0.80,
    ),
)


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    missing = sorted(set(names) - known)
    if missing:
        raise argparse.ArgumentTypeError(f"unknown KERA profile(s): {', '.join(missing)}")
    return names


def _profile_by_name(name: str) -> KeraProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown KERA profile {name!r}")


def _fuse_kera_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: KeraProfile,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    labels = sorted({_str(row, "top_label") for row in rows})
    candidates = [_aggregate_label(rows, label) for label in labels]
    candidates = [item for item in candidates if item]
    candidates.sort(key=lambda item: (item["evidence"], item["vote_fraction"], item["known_score"]), reverse=True)
    best = candidates[0] if candidates else {}

    seen_candidates = [item for item in candidates if item.get("label_set") == "seen_new"]
    seen_candidates.sort(key=lambda item: (item["known_score"], item["vote_fraction"], item["evidence"]), reverse=True)
    seen_best = seen_candidates[0] if seen_candidates else {}

    old_candidates = [item for item in candidates if item.get("label_set") == "old"]
    old_candidates.sort(key=lambda item: (item["old_anchor_score"], item["known_score"], item["vote_fraction"]), reverse=True)
    old_best = old_candidates[0] if old_candidates else {}

    old_evidence = float(old_best.get("evidence", 0.0)) if old_best else 0.0
    seen_enrollment = bool(
        seen_best
        and seen_best["known_score"] >= profile.seen_new_known_min
        and seen_best["margin"] >= profile.seen_new_margin_min
        and seen_best["vote_fraction"] >= profile.seen_new_vote_min
        and (float(seen_best["evidence"]) - old_evidence) >= profile.seen_new_old_gap_floor
    )
    old_guard = bool(
        old_best
        and old_best["old_anchor_score"] >= profile.old_anchor_min
        and old_best["known_score"] >= profile.old_known_min
        and old_best["margin"] >= profile.old_margin_min
        and old_best["vote_fraction"] >= profile.old_vote_min
    )
    known_accept = bool(
        best
        and best["known_score"] >= profile.known_score_min
        and best["margin"] >= profile.known_margin_min
        and best["vote_fraction"] >= profile.known_vote_min
        and best["receiver_count"] >= profile.known_min_receivers
    )

    total = max(1, len(rows))
    label_counts = Counter(str(row.get("top_label", "")) for row in rows)
    disagreement = 1.0 - (max(label_counts.values()) / float(total))
    unknown_fraction = sum(_float(row, "unknown_score") >= profile.unknown_score_min for row in rows) / float(total)
    low_margin_fraction = sum(_float(row, "margin") < profile.known_margin_min for row in rows) / float(total)
    mean_unknown = sum(_float(row, "unknown_score") for row in rows) / float(total)
    known_consensus = float(best.get("evidence", 0.0)) * float(best.get("vote_fraction", 0.0)) if best else 0.0
    unknown_reject = bool(
        not seen_enrollment
        and not old_guard
        and known_consensus < profile.known_consensus_rescue_min
        and (
            unknown_fraction >= profile.unknown_fraction_min
            or (
                disagreement >= profile.disagreement_min
                and low_margin_fraction >= profile.low_margin_fraction_min
                and mean_unknown >= (profile.unknown_score_min - 0.08)
            )
        )
    )

    if seen_enrollment:
        output_label = str(seen_best["label"])
        output_action = "accept"
        decision = "accept_seen_new_enrollment"
        chosen = seen_best
    elif old_guard:
        output_label = str(old_best["label"])
        output_action = "accept"
        decision = "accept_old_anchor_guard"
        chosen = old_best
    elif unknown_reject:
        output_label = UNKNOWN_LABEL
        output_action = "reject_unknown"
        decision = "reject_unknown_after_enrollment_fail"
        chosen = best
    elif known_accept:
        output_label = str(best["label"])
        output_action = "accept"
        decision = f"accept_{best['label_set']}_known_enrollment"
        chosen = best
    else:
        output_label = DEFER_LABEL
        output_action = "defer"
        decision = "defer_kera_selective"
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
        "candidate_known_score": float(chosen.get("known_score", 0.0)),
        "candidate_unknown_score": float(chosen.get("unknown_score", 0.0)),
        "candidate_old_anchor_score": float(chosen.get("old_anchor_score", 0.0)),
        "candidate_margin": float(chosen.get("margin", 0.0)),
        "candidate_receiver_count": int(chosen.get("receiver_count", 0)),
        "candidate_vote_fraction": float(chosen.get("vote_fraction", 0.0)),
        "seen_enrollment": bool(seen_enrollment),
        "old_guard": bool(old_guard),
        "unknown_fraction": float(unknown_fraction),
        "low_margin_fraction": float(low_margin_fraction),
        "mean_unknown_score": float(mean_unknown),
        "receiver_disagreement": float(disagreement),
        "unique_top_labels": int(len(label_counts)),
        "receiver_count": int(len(rows)),
        "bytes_per_event": float(total_bytes),
        "latency_ms": float(latency_ms),
        "resource_proxy_pass": bool(resource_proxy_pass),
    }


def evaluate_kera(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[KeraProfile],
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
                        _fuse_kera_event(
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
                "pseudo_unknown_uses_target_unknown": False,
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
        "algorithm": "KERA-CI",
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
        "unknown_query_eval_only": True,
        "target_unknown_training_count": 0,
        "profile_selection_uses_target_unknown": False,
        "reliability_uses_target_unknown": False,
        "pseudo_unknown_uses_target_unknown": False,
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
    parser.add_argument("--evidence_packet_bytes", type=float, default=192.0)
    parser.add_argument("--max_event_bytes", type=float, default=1152.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--adapter_alpha", type=float, default=0.18)
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
    profile_names = _profile_names(args.profiles)
    profiles = [_profile_by_name(name) for name in profile_names]
    alpha = float(args.adapter_alpha)
    if len(profiles) == 1:
        alpha = float(profiles[0].adapter_alpha)
    evidence_rows, metadata = build_aor_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        adapter_alpha=alpha,
    )
    result = evaluate_kera(
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
            "adapter_fit_scope": metadata["adapter_fit_scope"],
            "threshold_uses_target_unknown": metadata["threshold_uses_target_unknown"],
            "kera_metadata": {
                **metadata,
                "algorithm": "KERA-CI",
                "fusion_order": "seen_new_enrollment_then_old_anchor_then_unknown_reject",
            },
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
