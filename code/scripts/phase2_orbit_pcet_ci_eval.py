#!/usr/bin/env python
"""PCET-CI collaborative open-set evaluation for Stage2-C qknn8.

PCET-CI adds two deployable, support-only risk signals on top of qknn8
evidence: prototype-consistency risk and EVT/tail risk. Unknown query rows are
still evaluation-only; they are never used to fit thresholds or select support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    build_collaborative_evidence,
    load_feature_npz,
)


@dataclass(frozen=True)
class PcetProfile:
    name: str
    description: str
    unknown_risk_threshold: float
    candidate_set_unknown_reject_risk: float
    candidate_set_max_label_unknown_risk: float
    candidate_set_max_event_unknown_risk: float
    candidate_set_min_conformal_pvalue: float
    candidate_set_min_label_receiver_class_reliability: float
    candidate_set_max_label_shell_risk: float
    old_gate_max_effective_unknown_risk: float
    seen_new_gate_max_effective_unknown_risk: float
    accept_margin_threshold: float
    consensus_score_threshold: float
    scorer_component_vote_threshold: float
    class_set_gate_enabled: bool = True


PROFILES: tuple[PcetProfile, ...] = (
    PcetProfile(
        name="pcet_known_preserving",
        description="PCET risk telemetry with loose known acceptance, used to measure old/new retention",
        unknown_risk_threshold=0.98,
        candidate_set_unknown_reject_risk=0.98,
        candidate_set_max_label_unknown_risk=0.99,
        candidate_set_max_event_unknown_risk=0.99,
        candidate_set_min_conformal_pvalue=0.0,
        candidate_set_min_label_receiver_class_reliability=0.0,
        candidate_set_max_label_shell_risk=1.0,
        old_gate_max_effective_unknown_risk=1.0,
        seen_new_gate_max_effective_unknown_risk=1.0,
        accept_margin_threshold=0.0,
        consensus_score_threshold=0.0,
        scorer_component_vote_threshold=1.0,
        class_set_gate_enabled=False,
    ),
    PcetProfile(
        name="pcet_balanced",
        description="prototype-consistency plus tail-risk rejection with support-confirmed known rescue",
        unknown_risk_threshold=0.78,
        candidate_set_unknown_reject_risk=0.78,
        candidate_set_max_label_unknown_risk=0.66,
        candidate_set_max_event_unknown_risk=0.78,
        candidate_set_min_conformal_pvalue=0.12,
        candidate_set_min_label_receiver_class_reliability=0.24,
        candidate_set_max_label_shell_risk=0.74,
        old_gate_max_effective_unknown_risk=0.74,
        seen_new_gate_max_effective_unknown_risk=0.70,
        accept_margin_threshold=0.035,
        consensus_score_threshold=0.05,
        scorer_component_vote_threshold=0.50,
    ),
    PcetProfile(
        name="pcet_unknown_strict",
        description="strict PCET unknown safety profile for exposing retention trade-offs",
        unknown_risk_threshold=0.62,
        candidate_set_unknown_reject_risk=0.62,
        candidate_set_max_label_unknown_risk=0.54,
        candidate_set_max_event_unknown_risk=0.66,
        candidate_set_min_conformal_pvalue=0.18,
        candidate_set_min_label_receiver_class_reliability=0.30,
        candidate_set_max_label_shell_risk=0.62,
        old_gate_max_effective_unknown_risk=0.62,
        seen_new_gate_max_effective_unknown_risk=0.58,
        accept_margin_threshold=0.05,
        consensus_score_threshold=0.08,
        scorer_component_vote_threshold=0.42,
    ),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    unknown = sorted(set(names) - known)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown PCET-CI profile(s): {', '.join(unknown)}")
    return names


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


def _count_value(counts: Mapping[str, Any], key: str, *aliases: str, default: float = 0.0) -> float:
    for name in (key, *aliases):
        if name in counts:
            try:
                return float(counts[name])
            except (TypeError, ValueError):
                return float(default)
    return float(default)


def _safe_known_evidence(
    row: Mapping[str, Any],
    *,
    safe_pvalue: float,
    safe_reliability: float,
    safe_margin: float,
    tail_risk: float,
    tail_safe: float,
) -> bool:
    pvalue = _float(row, "class_evidence_top1_conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0))
    reliability = _float(
        row,
        "class_evidence_top1_receiver_class_reliability",
        _float(row, "receiver_class_reliability", _float(row, "support_density", 0.0)),
    )
    margin = _float(row, "class_evidence_top1_margin", _float(row, "known_margin", 0.0))
    support = _float(row, "class_evidence_top1_support_count", _float(row, "class_conformal_support_count", 0.0))
    return (
        pvalue >= float(safe_pvalue)
        and reliability >= float(safe_reliability)
        and margin >= float(safe_margin)
        and support >= 2.0
        and tail_risk <= float(tail_safe)
    )


def augment_pcet_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    proto_gap_scale: float = 0.08,
    support_shortfall: float = 3.0,
    base_weight: float = 0.35,
    proto_weight: float = 0.35,
    tail_weight: float = 0.30,
    safe_pvalue: float = 0.70,
    safe_reliability: float = 0.55,
    safe_margin: float = 0.05,
    tail_safe: float = 0.45,
    safe_known_risk_cap: float = 0.55,
) -> list[dict[str, Any]]:
    """Add PCET-CI risk fields without using query labels for calibration."""
    out: list[dict[str, Any]] = []
    gap_scale = max(float(proto_gap_scale), 1e-6)
    shortfall_den = max(float(support_shortfall), 1.0)
    for source in evidence:
        row = dict(source)
        top1_score = _float(row, "class_evidence_top1_score", _float(row, "known_score", 0.0))
        top2_score = _float(row, "class_evidence_top2_score", _float(row, "second_score", 0.0))
        explicit_gap = _float(row, "label_score_gap", top1_score - top2_score)
        gap = max(0.0, explicit_gap, top1_score - top2_score)
        pvalue = _float(row, "class_evidence_top1_conformal_pvalue", _float(row, "class_conformal_pvalue", 0.0))
        reliability = _float(
            row,
            "class_evidence_top1_receiver_class_reliability",
            _float(row, "receiver_class_reliability", _float(row, "support_density", 0.0)),
        )
        support_count = _float(
            row,
            "class_evidence_top1_support_count",
            _float(row, "class_conformal_support_count", 0.0),
        )
        verifier_changed = _str(row, "class_verifier_changed", "False").lower() in {"1", "true", "yes"}
        label_disagree = float(
            _str(row, "class_evidence_top1_label", _str(row, "predicted_label"))
            != _str(row, "predicted_label")
        )
        instability = _clip01(1.0 - gap / gap_scale)
        proto_consistency_risk = _clip01(
            0.42 * instability
            + 0.22 * (1.0 - _clip01(pvalue))
            + 0.20 * (1.0 - _clip01(reliability))
            + 0.10 * _clip01((shortfall_den - support_count) / shortfall_den)
            + 0.06 * max(label_disagree, float(verifier_changed))
        )
        tail_risk = max(
            _clip01(_float(row, "evt_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_evt_risk", 0.0)),
            _clip01(_float(row, "mahalanobis_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_mahalanobis_risk", 0.0)),
            _clip01(_float(row, "class_shell_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_class_shell_risk", 0.0)),
            _clip01(_float(row, "radius_risk", 0.0)),
        )
        base_risk = max(
            _clip01(_float(row, "unknown_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
        )
        combined = _clip01(
            float(base_weight) * base_risk
            + float(proto_weight) * proto_consistency_risk
            + float(tail_weight) * tail_risk
        )
        pcet_unknown_risk = max(base_risk, combined)
        safe_known = _safe_known_evidence(
            row,
            safe_pvalue=safe_pvalue,
            safe_reliability=safe_reliability,
            safe_margin=safe_margin,
            tail_risk=tail_risk,
            tail_safe=tail_safe,
        )
        if safe_known:
            pcet_unknown_risk = min(pcet_unknown_risk, float(safe_known_risk_cap))
        row["pcet_proto_consistency_risk"] = proto_consistency_risk
        row["pcet_tail_risk"] = tail_risk
        row["pcet_base_unknown_risk"] = base_risk
        row["pcet_unknown_risk"] = _clip01(pcet_unknown_risk)
        row["pcet_safe_known_cap_applied"] = bool(safe_known)
        row["pcet_gap"] = float(gap)
        row["pcet_pvalue"] = float(pvalue)
        row["pcet_reliability"] = float(reliability)
        row["pcet_support_count"] = float(support_count)
        row["unknown_risk"] = row["pcet_unknown_risk"]
        row["class_evidence_top1_unknown_risk"] = row["pcet_unknown_risk"]
        row["pcet_original_class_shell_risk"] = _clip01(_float(row, "class_shell_risk", 0.0))
        row["pcet_original_evt_risk"] = _clip01(_float(row, "evt_risk", 0.0))
        out.append(row)
    return out


def _target_pass(row: Mapping[str, Any]) -> bool:
    return bool(
        row["old_acc"] >= row["target_old_acc"]
        and row["min_old"] >= row["target_min_old"]
        and row["seen_new_acc"] >= row["target_seen_new_acc"]
        and row["min_seen"] >= row["target_min_seen"]
        and row["unknown_reject"] >= row["target_unknown_reject"]
    )


def run_pcet_ci(args: argparse.Namespace) -> dict[str, Any]:
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
        unknown_gate_mode="support_envelope_full",
        radius_quantile=float(args.radius_quantile),
        margin_quantile=float(args.margin_quantile),
        score_quantile=float(args.score_quantile),
        mahalanobis_quantile=float(args.mahalanobis_quantile),
        evt_tail_quantile=float(args.evt_tail_quantile),
        oldness_quantile=float(args.oldness_quantile),
        radius_slack=float(args.radius_slack),
        margin_slack=float(args.margin_slack),
        score_slack=float(args.score_slack),
        mahalanobis_slack=float(args.mahalanobis_slack),
        evt_tail_slack=float(args.evt_tail_slack),
        oldness_slack=float(args.oldness_slack),
        support_calibration_mode="leave_one_out",
        score_threshold_combine="max",
        class_score_threshold_enabled=True,
        class_score_threshold_quantile=0.05,
        class_score_threshold_min_support=2,
        class_conformal_enabled=True,
        class_conformal_min_support=1,
        class_evidence_top_m=int(args.class_evidence_top_m),
        virtual_unknown_risk_enabled=True,
        virtual_unknown_risk_samples_per_class=int(args.virtual_unknown_samples_per_class),
        virtual_unknown_mix_alpha=float(args.virtual_unknown_mix_alpha),
        virtual_unknown_noise_scale=float(args.virtual_unknown_noise_scale),
        virtual_unknown_neighbor_count=int(args.virtual_unknown_neighbor_count),
        virtual_unknown_risk_temperature=float(args.virtual_unknown_risk_temperature),
        virtual_unknown_risk_margin=float(args.virtual_unknown_risk_margin),
        class_negative_risk_enabled=True,
        class_negative_samples_per_class=int(args.class_negative_samples_per_class),
        class_negative_mix_alpha=float(args.class_negative_mix_alpha),
        class_negative_neighbor_count=int(args.class_negative_neighbor_count),
        class_negative_risk_temperature=float(args.class_negative_risk_temperature),
        class_negative_risk_margin=float(args.class_negative_risk_margin),
        class_negative_combine_mode="max",
        class_negative_risk_floor=float(args.class_negative_risk_floor),
        class_shell_unknown_risk_enabled=True,
        class_shell_radius_scale=float(args.class_shell_radius_scale),
        class_shell_risk_temperature=float(args.class_shell_risk_temperature),
        class_shell_risk_margin=float(args.class_shell_risk_margin),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        receiver_reliability_policy="support_density",
        receiver_class_reliability_policy="support_calibrated",
        prototype_score_blend=float(args.prototype_score_blend),
        mahalanobis_score_blend=float(args.mahalanobis_score_blend),
        mahalanobis_score_temperature=float(args.mahalanobis_score_temperature),
        source_old_prototype_shrinkage_alpha=float(args.source_old_prototype_shrinkage_alpha),
        class_verifier_policy="support_quality",
        class_verifier_top_m=int(args.class_verifier_top_m),
        class_verifier_pvalue_weight=0.35,
        class_verifier_reliability_weight=0.35,
        class_verifier_risk_weight=0.30,
    )
    evidence = augment_pcet_evidence(
        evidence,
        proto_gap_scale=float(args.pcet_proto_gap_scale),
        support_shortfall=float(args.pcet_support_shortfall),
        base_weight=float(args.pcet_base_weight),
        proto_weight=float(args.pcet_proto_weight),
        tail_weight=float(args.pcet_tail_weight),
        safe_pvalue=float(args.pcet_safe_pvalue),
        safe_reliability=float(args.pcet_safe_reliability),
        safe_margin=float(args.pcet_safe_margin),
        tail_safe=float(args.pcet_tail_safe),
        safe_known_risk_cap=float(args.pcet_safe_known_risk_cap),
    )
    metadata = dict(metadata)
    metadata["algorithm_wrapper"] = "PCET-CI"
    metadata["unknown_query_eval_only"] = True
    metadata["labeled_unknown_support_used_for_boundary_fit"] = False
    metadata["in_orbit_method"] = "qknn8"
    metadata["pcet_components"] = ["prototype_consistency", "evt_tail", "support_conformal"]

    requested_profiles = set(_profile_names(args.profiles))
    profile_map = {profile.name: profile for profile in PROFILES}
    profile_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for profile_name in [profile.name for profile in PROFILES if profile.name in requested_profiles]:
        profile = profile_map[profile_name]
        result = evaluate_collaborative_open_set_evidence(
            evidence,
            collab_counts=args.collab_counts,
            unknown_risk_threshold=profile.unknown_risk_threshold,
            accept_margin_threshold=profile.accept_margin_threshold,
            unknown_quantile=float(args.unknown_quantile),
            fusion_policy="scg_qknn_cvs",
            consensus_gap_threshold=float(args.consensus_gap_threshold),
            consensus_score_threshold=profile.consensus_score_threshold,
            scorer_component_vote_threshold=profile.scorer_component_vote_threshold,
            scorer_risk_components=["score", "radius", "margin", "evt", "class_shell"],
            label_fusion_policy="weighted_vote_margin",
            class_reliability_policy="conformal_margin_risk",
            receiver_class_reliability_policy="support_calibrated",
            collaboration_policy=str(args.collaboration_policy),
            latency_budget_ms=float(args.latency_budget_ms),
            max_event_bytes=float(args.max_event_bytes),
            max_event_latency_ms=float(args.max_event_latency_ms),
            class_set_gate_enabled=bool(profile.class_set_gate_enabled),
            old_gate_min_receivers=1,
            old_gate_max_effective_unknown_risk=profile.old_gate_max_effective_unknown_risk,
            old_gate_max_component_agreement=0.75,
            old_gate_min_support_density=0.05,
            seen_new_gate_min_receivers=1,
            seen_new_gate_max_effective_unknown_risk=profile.seen_new_gate_max_effective_unknown_risk,
            seen_new_gate_max_component_agreement=0.70,
            seen_new_gate_min_support_density=0.05,
            candidate_set_min_receivers=int(args.candidate_set_min_receivers),
            candidate_set_min_top1_receivers=1,
            candidate_set_min_conformal_pvalue=profile.candidate_set_min_conformal_pvalue,
            candidate_set_min_label_receiver_class_reliability=(
                profile.candidate_set_min_label_receiver_class_reliability
            ),
            candidate_set_max_label_unknown_risk=profile.candidate_set_max_label_unknown_risk,
            candidate_set_max_event_unknown_risk=profile.candidate_set_max_event_unknown_risk,
            candidate_set_max_label_risk_component_agreement=0.75,
            candidate_set_max_label_shell_risk=profile.candidate_set_max_label_shell_risk,
            candidate_set_shell_reject_risk=profile.candidate_set_max_label_shell_risk,
            candidate_set_event_high_unknown_risk_veto=profile.candidate_set_unknown_reject_risk,
            candidate_set_max_label_high_unknown_risk_fraction=0.50,
            candidate_set_high_unknown_risk_threshold=profile.candidate_set_unknown_reject_risk,
            candidate_set_min_score_gap=0.01,
            candidate_set_unknown_reject_risk=profile.candidate_set_unknown_reject_risk,
            candidate_set_max_receiver_pair_label_disagreement=0.50,
            candidate_set_max_receiver_pair_unknown_risk_range=0.50,
            threshold_selection_label_scope=str(metadata["threshold_scope"]),
            unknown_query_eval_only=True,
            receiver_selection_policy="support_quality_prior",
            collab_group_policy=str(args.collab_group_policy),
            partial_collab_min_receivers=int(args.partial_collab_min_receivers),
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
            include_event_results=bool(args.include_event_results),
        )
        result["pcet_profile"] = profile.__dict__
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
                "request_more": _count_value(counts, "unknown_request_more_rate", "request_more_rate"),
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
                (float(args.max_event_bytes) <= 0 or row["bytes_per_event"] <= float(args.max_event_bytes))
                and (
                    float(args.max_event_latency_ms) <= 0
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
        "algorithm": "PCET-CI",
        "feature_npz": str(args.feature_npz),
        "profiles": [profile.__dict__ for profile in PROFILES if profile.name in requested_profiles],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_summary_csv", type=Path)
    parser.add_argument("--output_evidence_csv", type=Path)
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--collab_counts", default="all")
    parser.add_argument("--collab_group_policy", default="available_up_to_k", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    parser.add_argument("--partial_collab_min_receivers", type=_positive_int, default=1)
    parser.add_argument("--k_shot", type=_positive_int, default=8)
    parser.add_argument("--query_per_class", type=_positive_int, default=20)
    parser.add_argument("--qknn_k", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=4070404)
    parser.add_argument("--event_alignment_policy", choices=["strict_event_key", "receiver_domain_ranked"], default="receiver_domain_ranked")
    parser.add_argument("--support_selection_policy", choices=["stable_first", "centroid", "scenario_diverse"], default="scenario_diverse")
    parser.add_argument("--support_quantile", type=float, default=0.05)
    parser.add_argument("--proxy_quantile", type=float, default=0.95)
    parser.add_argument("--risk_temperature", type=float, default=0.05)
    parser.add_argument("--radius_quantile", type=float, default=0.95)
    parser.add_argument("--margin_quantile", type=float, default=0.05)
    parser.add_argument("--score_quantile", type=float, default=0.05)
    parser.add_argument("--mahalanobis_quantile", type=float, default=0.95)
    parser.add_argument("--evt_tail_quantile", type=float, default=0.95)
    parser.add_argument("--oldness_quantile", type=float, default=0.05)
    parser.add_argument("--radius_slack", type=float, default=1.05)
    parser.add_argument("--margin_slack", type=float, default=0.0)
    parser.add_argument("--score_slack", type=float, default=0.0)
    parser.add_argument("--mahalanobis_slack", type=float, default=1.05)
    parser.add_argument("--evt_tail_slack", type=float, default=1.05)
    parser.add_argument("--oldness_slack", type=float, default=0.0)
    parser.add_argument("--class_evidence_top_m", type=_positive_int, default=3)
    parser.add_argument("--virtual_unknown_samples_per_class", type=_positive_int, default=4)
    parser.add_argument("--virtual_unknown_mix_alpha", type=float, default=0.50)
    parser.add_argument("--virtual_unknown_noise_scale", type=float, default=0.02)
    parser.add_argument("--virtual_unknown_neighbor_count", type=_positive_int, default=2)
    parser.add_argument("--virtual_unknown_risk_temperature", type=float, default=0.05)
    parser.add_argument("--virtual_unknown_risk_margin", type=float, default=0.0)
    parser.add_argument("--class_negative_samples_per_class", type=_positive_int, default=4)
    parser.add_argument("--class_negative_mix_alpha", type=float, default=0.50)
    parser.add_argument("--class_negative_neighbor_count", type=_positive_int, default=2)
    parser.add_argument("--class_negative_risk_temperature", type=float, default=0.05)
    parser.add_argument("--class_negative_risk_margin", type=float, default=0.0)
    parser.add_argument("--class_negative_risk_floor", type=float, default=0.0)
    parser.add_argument("--class_shell_radius_scale", type=float, default=1.50)
    parser.add_argument("--class_shell_risk_temperature", type=float, default=0.05)
    parser.add_argument("--class_shell_risk_margin", type=float, default=0.0)
    parser.add_argument("--prototype_score_blend", type=float, default=0.20)
    parser.add_argument("--mahalanobis_score_blend", type=float, default=0.10)
    parser.add_argument("--mahalanobis_score_temperature", type=float, default=0.25)
    parser.add_argument("--source_old_prototype_shrinkage_alpha", type=float, default=0.25)
    parser.add_argument("--class_verifier_top_m", type=_positive_int, default=3)
    parser.add_argument("--pcet_proto_gap_scale", type=float, default=0.08)
    parser.add_argument("--pcet_support_shortfall", type=float, default=3.0)
    parser.add_argument("--pcet_base_weight", type=float, default=0.70)
    parser.add_argument("--pcet_proto_weight", type=float, default=0.15)
    parser.add_argument("--pcet_tail_weight", type=float, default=0.15)
    parser.add_argument("--pcet_safe_pvalue", type=float, default=0.70)
    parser.add_argument("--pcet_safe_reliability", type=float, default=0.55)
    parser.add_argument("--pcet_safe_margin", type=float, default=0.05)
    parser.add_argument("--pcet_tail_safe", type=float, default=0.45)
    parser.add_argument("--pcet_safe_known_risk_cap", type=float, default=0.45)
    parser.add_argument("--unknown_quantile", type=float, default=0.75)
    parser.add_argument("--consensus_gap_threshold", type=float, default=0.02)
    parser.add_argument("--candidate_set_min_receivers", type=_positive_int, default=2)
    parser.add_argument("--collaboration_policy", default="fixed_k", choices=["fixed_k", "progressive_budget", "adaptive_gain", "support_utility", "rb_capr_utility", "dual_route_cvs"])
    parser.add_argument("--latency_budget_ms", type=float, default=20.0)
    parser.add_argument("--max_event_bytes", type=float, default=1152.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--evidence_packet_bytes", type=float, default=128.0)
    parser.add_argument("--target_old_acc", type=float, default=0.99)
    parser.add_argument("--target_min_old", type=float, default=0.95)
    parser.add_argument("--target_seen_new_acc", type=float, default=0.97)
    parser.add_argument("--target_min_seen", type=float, default=0.93)
    parser.add_argument("--target_unknown_reject", type=float, default=0.99)
    parser.add_argument("--include_event_results", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_pcet_ci(args)
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
