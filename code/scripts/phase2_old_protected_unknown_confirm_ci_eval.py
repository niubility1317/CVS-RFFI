#!/usr/bin/env python
"""Evaluate old-protected unknown confirmation for satellite collaborative RFFI.

OPU-CI keeps the ADV3B02/qknn8 evidence path frozen. It combines a known route
with a support-only safety route, then requires support-confirmed old/seen-new
evidence before accepting a known label. Unknown evidence requests more
receivers when budget remains and rejects only after multi-source confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
from scripts.phase2_socapr_dual_route_veto_eval import (
    build_dual_route_evidence,
    _read_csv_rows,
    _run_route,
    _write_csv,
)


@dataclass(frozen=True)
class OpuPolicy:
    name: str
    unknown_risk_threshold: float
    candidate_set_unknown_reject_risk: float
    scorer_component_vote_threshold: float
    candidate_set_min_receivers: int
    candidate_set_min_top1_receivers: int
    candidate_set_min_conformal_pvalue: float
    candidate_set_max_label_unknown_risk: float
    candidate_set_max_event_unknown_risk: float
    candidate_set_max_label_risk_component_agreement: float
    candidate_set_max_label_shell_risk: float
    candidate_set_shell_reject_risk: float
    candidate_set_max_receiver_pair_label_disagreement: float
    candidate_set_max_receiver_pair_unknown_risk_range: float
    candidate_set_min_label_receiver_class_reliability: float
    old_gate_min_support_density: float
    seen_new_gate_min_support_density: float
    accept_margin_threshold: float
    consensus_score_threshold: float
    consensus_gap_threshold: float


POLICIES: tuple[OpuPolicy, ...] = (
    OpuPolicy(
        name="opu_old_preserve",
        unknown_risk_threshold=0.82,
        candidate_set_unknown_reject_risk=0.86,
        scorer_component_vote_threshold=0.70,
        candidate_set_min_receivers=1,
        candidate_set_min_top1_receivers=0,
        candidate_set_min_conformal_pvalue=0.00,
        candidate_set_max_label_unknown_risk=0.95,
        candidate_set_max_event_unknown_risk=0.98,
        candidate_set_max_label_risk_component_agreement=0.90,
        candidate_set_max_label_shell_risk=1.00,
        candidate_set_shell_reject_risk=1.00,
        candidate_set_max_receiver_pair_label_disagreement=0.88,
        candidate_set_max_receiver_pair_unknown_risk_range=1.00,
        candidate_set_min_label_receiver_class_reliability=0.00,
        old_gate_min_support_density=0.00,
        seen_new_gate_min_support_density=0.00,
        accept_margin_threshold=0.005,
        consensus_score_threshold=0.00,
        consensus_gap_threshold=0.00,
    ),
    OpuPolicy(
        name="opu_balanced",
        unknown_risk_threshold=0.58,
        candidate_set_unknown_reject_risk=0.62,
        scorer_component_vote_threshold=0.42,
        candidate_set_min_receivers=2,
        candidate_set_min_top1_receivers=1,
        candidate_set_min_conformal_pvalue=0.02,
        candidate_set_max_label_unknown_risk=0.72,
        candidate_set_max_event_unknown_risk=0.78,
        candidate_set_max_label_risk_component_agreement=0.62,
        candidate_set_max_label_shell_risk=0.88,
        candidate_set_shell_reject_risk=0.92,
        candidate_set_max_receiver_pair_label_disagreement=0.62,
        candidate_set_max_receiver_pair_unknown_risk_range=0.88,
        candidate_set_min_label_receiver_class_reliability=0.05,
        old_gate_min_support_density=0.10,
        seen_new_gate_min_support_density=0.10,
        accept_margin_threshold=0.015,
        consensus_score_threshold=0.05,
        consensus_gap_threshold=0.02,
    ),
    OpuPolicy(
        name="opu_old_guarded",
        unknown_risk_threshold=0.68,
        candidate_set_unknown_reject_risk=0.72,
        scorer_component_vote_threshold=0.50,
        candidate_set_min_receivers=2,
        candidate_set_min_top1_receivers=1,
        candidate_set_min_conformal_pvalue=0.00,
        candidate_set_max_label_unknown_risk=0.82,
        candidate_set_max_event_unknown_risk=0.88,
        candidate_set_max_label_risk_component_agreement=0.72,
        candidate_set_max_label_shell_risk=0.95,
        candidate_set_shell_reject_risk=0.97,
        candidate_set_max_receiver_pair_label_disagreement=0.75,
        candidate_set_max_receiver_pair_unknown_risk_range=0.95,
        candidate_set_min_label_receiver_class_reliability=0.05,
        old_gate_min_support_density=0.05,
        seen_new_gate_min_support_density=0.05,
        accept_margin_threshold=0.01,
        consensus_score_threshold=0.03,
        consensus_gap_threshold=0.01,
    ),
    OpuPolicy(
        name="opu_unknown_strict",
        unknown_risk_threshold=0.48,
        candidate_set_unknown_reject_risk=0.52,
        scorer_component_vote_threshold=0.34,
        candidate_set_min_receivers=2,
        candidate_set_min_top1_receivers=1,
        candidate_set_min_conformal_pvalue=0.04,
        candidate_set_max_label_unknown_risk=0.58,
        candidate_set_max_event_unknown_risk=0.68,
        candidate_set_max_label_risk_component_agreement=0.50,
        candidate_set_max_label_shell_risk=0.78,
        candidate_set_shell_reject_risk=0.86,
        candidate_set_max_receiver_pair_label_disagreement=0.50,
        candidate_set_max_receiver_pair_unknown_risk_range=0.78,
        candidate_set_min_label_receiver_class_reliability=0.08,
        old_gate_min_support_density=0.15,
        seen_new_gate_min_support_density=0.15,
        accept_margin_threshold=0.02,
        consensus_score_threshold=0.06,
        consensus_gap_threshold=0.03,
    ),
)


def _parse_policy_names(value: str) -> list[str]:
    if str(value).strip().lower() in {"", "all", "*"}:
        return [policy.name for policy in POLICIES]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _policy_by_name(name: str) -> OpuPolicy:
    for policy in POLICIES:
        if policy.name == name:
            return policy
    raise ValueError(f"unknown OPU policy {name!r}; expected one of {[p.name for p in POLICIES]}")


def _float_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _joint_score(metrics: Mapping[str, Any]) -> float:
    old_acc = _float_metric(metrics, "old_acc")
    seen_new_acc = _float_metric(metrics, "seen_new_acc")
    unknown_reject = _float_metric(metrics, "unknown_reject_rate")
    unknown_far = _float_metric(metrics, "unknown_FAR")
    defer = _float_metric(metrics, "defer_rate")
    return old_acc + seen_new_acc + unknown_reject - unknown_far - 0.5 * defer


def _summary_row(*, policy: OpuPolicy, count: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": policy.name,
        "fusion_policy": "old_protected_unknown_confirm_cvs",
        "collab_count": int(count),
        "joint_score": _joint_score(metrics),
        "old_acc": metrics.get("old_acc", 0.0),
        "min_old_class_acc": metrics.get("min_old_class_acc", 0.0),
        "seen_new_acc": metrics.get("seen_new_acc", 0.0),
        "min_seen_new_class_acc": metrics.get("min_seen_new_class_acc", 0.0),
        "unknown_reject_rate": metrics.get("unknown_reject_rate", 0.0),
        "unknown_FAR": metrics.get("unknown_FAR", 0.0),
        "known_coverage": metrics.get("known_coverage", 0.0),
        "defer_rate": metrics.get("defer_rate", 0.0),
        "request_more_rate": metrics.get("request_more_rate", 0.0),
        "participating_receivers_p95": metrics.get("participating_receivers_p95", 0.0),
        "bytes_per_event": metrics.get("bytes_per_event", 0.0),
        "latency_ms_p95": metrics.get("latency_ms_p95", 0.0),
        "unknown_risk_threshold": policy.unknown_risk_threshold,
        "candidate_set_unknown_reject_risk": policy.candidate_set_unknown_reject_risk,
        "scorer_component_vote_threshold": policy.scorer_component_vote_threshold,
        "accept_margin_threshold": policy.accept_margin_threshold,
    }


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _evaluate_policy(
    evidence: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    policy: OpuPolicy,
    *,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    return evaluate_collaborative_open_set_evidence(
        evidence,
        collab_counts="all",
        threshold_selection_label_scope=str(metadata["threshold_scope"]),
        unknown_query_eval_only=True,
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
        collab_group_policy="available_up_to_k",
        partial_collab_min_receivers=1,
        fusion_policy="old_protected_unknown_confirm_cvs",
        label_fusion_policy="weighted_vote_margin",
        receiver_selection_policy="support_quality_prior",
        collaboration_policy="dual_route_cvs",
        latency_budget_ms=float(max_event_latency_ms),
        max_event_bytes=float(max_event_bytes),
        max_event_latency_ms=float(max_event_latency_ms),
        unknown_risk_threshold=policy.unknown_risk_threshold,
        accept_margin_threshold=policy.accept_margin_threshold,
        unknown_quantile=0.75,
        consensus_gap_threshold=policy.consensus_gap_threshold,
        consensus_score_threshold=policy.consensus_score_threshold,
        scorer_component_vote_threshold=policy.scorer_component_vote_threshold,
        scorer_risk_components="score,radius,margin,mahalanobis,evt,class_negative,class_shell",
        class_reliability_policy="conformal_margin_risk",
        receiver_class_reliability_policy="support_calibrated",
        class_set_gate_enabled=True,
        old_gate_min_receivers=1,
        old_gate_max_effective_unknown_risk=0.90,
        old_gate_max_component_agreement=0.85,
        old_gate_min_support_density=policy.old_gate_min_support_density,
        seen_new_gate_min_receivers=1,
        seen_new_gate_max_effective_unknown_risk=0.90,
        seen_new_gate_max_component_agreement=0.85,
        seen_new_gate_min_support_density=policy.seen_new_gate_min_support_density,
        candidate_set_min_receivers=policy.candidate_set_min_receivers,
        candidate_set_min_top1_receivers=policy.candidate_set_min_top1_receivers,
        candidate_set_min_conformal_pvalue=policy.candidate_set_min_conformal_pvalue,
        candidate_set_max_label_unknown_risk=policy.candidate_set_max_label_unknown_risk,
        candidate_set_max_event_unknown_risk=policy.candidate_set_max_event_unknown_risk,
        candidate_set_max_label_risk_component_agreement=(
            policy.candidate_set_max_label_risk_component_agreement
        ),
        candidate_set_max_label_shell_risk=policy.candidate_set_max_label_shell_risk,
        candidate_set_shell_reject_risk=policy.candidate_set_shell_reject_risk,
        candidate_set_event_high_unknown_risk_veto=policy.candidate_set_unknown_reject_risk,
        candidate_set_max_label_high_unknown_risk_fraction=0.50,
        candidate_set_high_unknown_risk_threshold=policy.candidate_set_unknown_reject_risk,
        candidate_set_unknown_reject_risk=policy.candidate_set_unknown_reject_risk,
        candidate_set_max_receiver_pair_label_disagreement=(
            policy.candidate_set_max_receiver_pair_label_disagreement
        ),
        candidate_set_max_receiver_pair_unknown_risk_range=(
            policy.candidate_set_max_receiver_pair_unknown_risk_range
        ),
        candidate_set_min_label_receiver_class_reliability=(
            policy.candidate_set_min_label_receiver_class_reliability
        ),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--policies", default="all")
    parser.add_argument("--score_anchor", type=float, default=0.70)
    parser.add_argument("--margin_anchor", type=float, default=0.40)
    parser.add_argument("--safety_weight", type=float, default=0.35)
    parser.add_argument("--discount_mode", choices=["prod", "mean", "max"], default="mean")
    parser.add_argument("--max_event_bytes", type=float, default=900.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=1.0)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    _, safety_csv = _run_route(
        route="safety_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    known_result = json.loads(known_json.read_text(encoding="utf-8"))
    metadata = dict(known_result["qknn_metadata"])
    metadata["adapter_type"] = "opu_ci_old_protected_unknown_confirmation"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0

    evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    evidence_csv = args.output_dir / "opu_ci_evidence.csv"
    _write_csv(evidence_csv, evidence)

    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name in _parse_policy_names(args.policies):
        policy = _policy_by_name(name)
        result = _evaluate_policy(
            evidence,
            metadata,
            policy,
            max_event_bytes=float(args.max_event_bytes),
            max_event_latency_ms=float(args.max_event_latency_ms),
        )
        results[policy.name] = result
        for count, metrics in result["counts"].items():
            summary_rows.append(_summary_row(policy=policy, count=count, metrics=metrics))

    summary_rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    summary_csv = args.output_dir / "opu_ci_summary.csv"
    summary_json = args.output_dir / "opu_ci_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "policies": _parse_policy_names(args.policies),
                    "score_anchor": float(args.score_anchor),
                    "margin_anchor": float(args.margin_anchor),
                    "safety_weight": float(args.safety_weight),
                    "discount_mode": str(args.discount_mode),
                    "max_event_bytes": float(args.max_event_bytes),
                    "max_event_latency_ms": float(args.max_event_latency_ms),
                },
                "metadata": metadata,
                "results": results,
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "summary_csv": str(summary_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
