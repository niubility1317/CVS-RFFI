#!/usr/bin/env python
"""SOVC-CI verifier collaborative open-set evaluation for Stage2-C qknn8.

SOVC-CI adds a source/support-side open-set verifier calibration layer on top
of the existing qknn8 collaborative evidence. The wrapper is deployable on an
onboard receiver cluster because it uses only support-derived verifier fields,
prototype telemetry, and bounded per-event scalar packets. Unknown query rows
remain evaluation-only and are never used for threshold fitting.
"""

from __future__ import annotations

import argparse
import copy
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


@dataclass(frozen=True)
class SovcProfile:
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


PROFILES: tuple[SovcProfile, ...] = (
    SovcProfile(
        name="sovc_known_preserving",
        description="SOVC telemetry with loose known acceptance for old/new retention measurement",
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
    SovcProfile(
        name="sovc_balanced",
        description="verifier-calibrated rejection with support-confirmed known rescue",
        unknown_risk_threshold=0.74,
        candidate_set_unknown_reject_risk=0.74,
        candidate_set_max_label_unknown_risk=0.64,
        candidate_set_max_event_unknown_risk=0.74,
        candidate_set_min_conformal_pvalue=0.10,
        candidate_set_min_label_receiver_class_reliability=0.22,
        candidate_set_max_label_shell_risk=0.72,
        old_gate_max_effective_unknown_risk=0.72,
        seen_new_gate_max_effective_unknown_risk=0.68,
        accept_margin_threshold=0.03,
        consensus_score_threshold=0.05,
        scorer_component_vote_threshold=0.50,
    ),
    SovcProfile(
        name="sovc_old_safe",
        description="event-level verifier rejection without class-set hard gates, used to preserve old classes",
        unknown_risk_threshold=0.50,
        candidate_set_unknown_reject_risk=0.65,
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
    SovcProfile(
        name="sovc_unknown_strict",
        description="strict verifier rejection profile used to expose unknown/known trade-offs",
        unknown_risk_threshold=0.58,
        candidate_set_unknown_reject_risk=0.58,
        candidate_set_max_label_unknown_risk=0.52,
        candidate_set_max_event_unknown_risk=0.62,
        candidate_set_min_conformal_pvalue=0.16,
        candidate_set_min_label_receiver_class_reliability=0.28,
        candidate_set_max_label_shell_risk=0.60,
        old_gate_max_effective_unknown_risk=0.60,
        seen_new_gate_max_effective_unknown_risk=0.56,
        accept_margin_threshold=0.045,
        consensus_score_threshold=0.08,
        scorer_component_vote_threshold=0.42,
    ),
)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _sovc_profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    unknown = sorted(set(names) - known)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown SOVC-CI profile(s): {', '.join(unknown)}")
    return names


def _verifier_changed(row: Mapping[str, Any]) -> bool:
    return _str(row, "class_verifier_changed", "False").lower() in {"1", "true", "yes"}


def _safe_sovc_known(
    row: Mapping[str, Any],
    *,
    verifier_risk: float,
    safe_verified_score: float,
    safe_pvalue: float,
    safe_reliability: float,
    safe_margin: float,
    safe_verifier_risk: float,
) -> bool:
    verified = _float(row, "class_verifier_top1_verified_score", _float(row, "known_score", 0.0))
    pvalue = _float(row, "class_verifier_top1_pvalue", _float(row, "class_conformal_pvalue", 0.0))
    reliability = _float(
        row,
        "class_verifier_top1_receiver_class_reliability",
        _float(row, "receiver_class_reliability", 0.0),
    )
    margin = verified - _float(row, "class_verifier_second_verified_score", 0.0)
    return (
        not _verifier_changed(row)
        and verified >= float(safe_verified_score)
        and pvalue >= float(safe_pvalue)
        and reliability >= float(safe_reliability)
        and margin >= float(safe_margin)
        and verifier_risk <= float(safe_verifier_risk)
    )


def augment_sovc_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    base_weight: float = 0.40,
    verifier_weight: float = 0.60,
    drop_scale: float = 0.65,
    margin_scale: float = 0.12,
    safe_verified_score: float = 0.18,
    safe_pvalue: float = 0.70,
    safe_reliability: float = 0.55,
    safe_margin: float = 0.03,
    safe_verifier_risk: float = 0.45,
    safe_known_risk_cap: float = 0.45,
) -> list[dict[str, Any]]:
    """Add verifier-calibrated unknown risk without using query labels."""
    out: list[dict[str, Any]] = []
    drop_den = max(float(drop_scale), 1e-6)
    margin_den = max(float(margin_scale), 1e-6)
    for source in evidence:
        row = dict(source)
        raw_score = _float(row, "class_verifier_top1_raw_score", _float(row, "known_score", 0.0))
        verified_score = _float(row, "class_verifier_top1_verified_score", raw_score)
        second_verified = _float(row, "class_verifier_second_verified_score", 0.0)
        pvalue = _clip01(_float(row, "class_verifier_top1_pvalue", _float(row, "class_conformal_pvalue", 0.0)))
        reliability = _clip01(
            _float(
                row,
                "class_verifier_top1_receiver_class_reliability",
                _float(row, "receiver_class_reliability", 0.0),
            )
        )
        verifier_unknown_risk = _clip01(_float(row, "class_verifier_top1_unknown_risk", 0.0))
        class_negative_risk = _clip01(_float(row, "class_verifier_top1_class_negative_risk", 0.0))
        shell_risk = max(
            _clip01(_float(row, "class_verifier_top1_class_shell_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_class_shell_risk", 0.0)),
            _clip01(_float(row, "class_shell_risk", 0.0)),
        )
        drop = max(0.0, raw_score - verified_score) / max(abs(raw_score), 1e-6)
        verified_margin = verified_score - second_verified
        drop_risk = _clip01(drop / drop_den)
        margin_risk = _clip01(1.0 - verified_margin / margin_den)
        changed_risk = 1.0 if _verifier_changed(row) else 0.0
        verifier_risk = _clip01(
            0.23 * drop_risk
            + 0.18 * margin_risk
            + 0.16 * (1.0 - pvalue)
            + 0.16 * (1.0 - reliability)
            + 0.12 * verifier_unknown_risk
            + 0.08 * class_negative_risk
            + 0.04 * shell_risk
            + 0.03 * changed_risk
        )
        base_risk = max(
            _clip01(_float(row, "unknown_risk", 0.0)),
            _clip01(_float(row, "class_evidence_top1_unknown_risk", 0.0)),
            _clip01(_float(row, "pcet_unknown_risk", 0.0)),
        )
        sovc_unknown_risk = max(
            base_risk,
            _clip01(float(base_weight) * base_risk + float(verifier_weight) * verifier_risk),
        )
        safe_known = _safe_sovc_known(
            row,
            verifier_risk=verifier_risk,
            safe_verified_score=safe_verified_score,
            safe_pvalue=safe_pvalue,
            safe_reliability=safe_reliability,
            safe_margin=safe_margin,
            safe_verifier_risk=safe_verifier_risk,
        )
        if safe_known:
            sovc_unknown_risk = min(sovc_unknown_risk, float(safe_known_risk_cap))
        row["sovc_raw_score"] = raw_score
        row["sovc_verified_score"] = verified_score
        row["sovc_verified_margin"] = verified_margin
        row["sovc_verifier_drop"] = drop
        row["sovc_drop_risk"] = drop_risk
        row["sovc_margin_risk"] = margin_risk
        row["sovc_verifier_risk"] = verifier_risk
        row["sovc_base_unknown_risk"] = base_risk
        row["sovc_unknown_risk"] = _clip01(sovc_unknown_risk)
        row["sovc_safe_known_cap_applied"] = bool(safe_known)
        row["unknown_risk"] = row["sovc_unknown_risk"]
        row["class_evidence_top1_unknown_risk"] = row["sovc_unknown_risk"]
        out.append(row)
    return out


def run_sovc_ci(args: argparse.Namespace) -> dict[str, Any]:
    base_args = copy.copy(args)
    base_args.profiles = "pcet_known_preserving"
    base = run_pcet_ci(base_args)
    evidence = augment_sovc_evidence(
        base.pop("_evidence_rows"),
        base_weight=float(args.sovc_base_weight),
        verifier_weight=float(args.sovc_verifier_weight),
        drop_scale=float(args.sovc_drop_scale),
        margin_scale=float(args.sovc_margin_scale),
        safe_verified_score=float(args.sovc_safe_verified_score),
        safe_pvalue=float(args.sovc_safe_pvalue),
        safe_reliability=float(args.sovc_safe_reliability),
        safe_margin=float(args.sovc_safe_margin),
        safe_verifier_risk=float(args.sovc_safe_verifier_risk),
        safe_known_risk_cap=float(args.sovc_safe_known_risk_cap),
    )
    metadata = dict(base["qknn_metadata"])
    metadata["algorithm_wrapper"] = "SOVC-CI"
    metadata["sovc_base_algorithm"] = "PCET-CI evidence + support-quality verifier"
    metadata["unknown_query_eval_only"] = True
    metadata["labeled_unknown_support_used_for_boundary_fit"] = False
    metadata["in_orbit_method"] = "qknn8"
    metadata["sovc_components"] = [
        "support_quality_verifier",
        "verified_score_drop",
        "receiver_class_reliability",
        "conformal_pvalue",
        "safe_known_cap",
    ]

    requested_profiles = set(_sovc_profile_names(args.sovc_profiles))
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
            candidate_set_event_high_unknown_risk_veto=(
                profile.candidate_set_unknown_reject_risk if profile.class_set_gate_enabled else 1.0e12
            ),
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
        result["sovc_profile"] = profile.__dict__
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
        "algorithm": "SOVC-CI",
        "feature_npz": str(args.feature_npz),
        "profiles": [profile.__dict__ for profile in PROFILES if profile.name in requested_profiles],
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
    sovc_parser = argparse.ArgumentParser(add_help=False)
    sovc_parser.add_argument("--profiles", dest="sovc_profiles", default="all")
    sovc_parser.add_argument("--sovc_base_weight", type=float, default=0.40)
    sovc_parser.add_argument("--sovc_verifier_weight", type=float, default=0.60)
    sovc_parser.add_argument("--sovc_drop_scale", type=float, default=0.65)
    sovc_parser.add_argument("--sovc_margin_scale", type=float, default=0.12)
    sovc_parser.add_argument("--sovc_safe_verified_score", type=float, default=0.18)
    sovc_parser.add_argument("--sovc_safe_pvalue", type=float, default=0.70)
    sovc_parser.add_argument("--sovc_safe_reliability", type=float, default=0.55)
    sovc_parser.add_argument("--sovc_safe_margin", type=float, default=0.03)
    sovc_parser.add_argument("--sovc_safe_verifier_risk", type=float, default=0.45)
    sovc_parser.add_argument("--sovc_safe_known_risk_cap", type=float, default=0.45)
    sovc_args, remaining = sovc_parser.parse_known_args(raw)
    args = _parse_pcet_args(remaining)
    for key, value in vars(sovc_args).items():
        setattr(args, key, value)
    _sovc_profile_names(args.sovc_profiles)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_sovc_ci(args)
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
