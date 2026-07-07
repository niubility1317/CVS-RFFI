#!/usr/bin/env python
"""Frozen-feature ManyTx unknown rejection diagnostic for CVS Stage2-C.

This wrapper is intentionally read-only. It consumes an already exported
feature NPZ, builds ADV3B02/qknn-style collaborative evidence, and evaluates
receiver counts from one to all target receivers. Real target unknown rows are
evaluation-only: they are never used for threshold fitting, calibration, model
selection, or training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    UNKNOWN_ROLE,
    build_collaborative_evidence,
    canonical_tx_id,
    load_feature_npz,
)


GOAL_OLD_ACC = 0.99
GOAL_MIN_OLD_CLASS_ACC = 0.95
GOAL_SEEN_NEW_ACC = 0.97
GOAL_MIN_SEEN_NEW_CLASS_ACC = 0.93
GOAL_UNKNOWN_REJECT = 0.99


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _float_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _int_metric(metrics: Mapping[str, Any], key: str) -> int:
    try:
        return int(metrics.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _role_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    return {str(role): int(np.sum(roles == role)) for role in sorted(set(roles.tolist()))}


def _manifest_list(manifest: Mapping[str, Any], key: str) -> set[str]:
    value = manifest.get(key, [])
    if isinstance(value, str):
        return {canonical_tx_id(part.strip()) for part in value.replace(";", ",").split(",") if part.strip()}
    try:
        return {canonical_tx_id(part) for part in value if str(part).strip()}
    except TypeError:
        return set()


def _repair_legacy_roles_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Split legacy target_new rows into target_new/target_unknown by manifest.

    Older Stage2-C feature packages sometimes recorded all non-old target query
    rows as target_new while their manifest still preserved unknown_tx_ids. This
    read-only compatibility repair makes the diagnostic runnable and records the
    exact change in protocol_safety. It does not create support rows or tune any
    threshold with unknown query data.
    """
    manifest = payload.get("manifest", {})
    if not isinstance(manifest, Mapping):
        return {"legacy_role_repair_applied": False, "legacy_role_repair_reason": "manifest_missing"}
    new_ids = _manifest_list(manifest, "new_tx_ids")
    unknown_ids = _manifest_list(manifest, "unknown_tx_ids")
    if not new_ids or not unknown_ids:
        return {"legacy_role_repair_applied": False, "legacy_role_repair_reason": "manifest_split_missing"}
    roles = np.asarray([str(value) for value in np.asarray(payload["dataset_role"]).tolist()], dtype=object)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    has_unknown_role = bool(np.any(roles == UNKNOWN_ROLE))
    if has_unknown_role:
        return {"legacy_role_repair_applied": False, "legacy_role_repair_reason": "target_unknown_already_present"}
    old_target_new = int(np.sum(roles == "target_new"))
    unknown_mask = (roles == "target_new") & np.isin(tx_ids, sorted(unknown_ids))
    new_mask = (roles == "target_new") & np.isin(tx_ids, sorted(new_ids))
    if int(np.sum(unknown_mask)) <= 0 or int(np.sum(new_mask)) <= 0:
        return {"legacy_role_repair_applied": False, "legacy_role_repair_reason": "manifest_ids_not_found_in_target_new"}
    roles[unknown_mask] = UNKNOWN_ROLE
    payload["dataset_role"] = roles
    return {
        "legacy_role_repair_applied": True,
        "legacy_role_repair_reason": "split_target_new_by_manifest_new_and_unknown_tx_ids",
        "legacy_target_new_rows_before": old_target_new,
        "legacy_target_new_rows_after": int(np.sum(roles == "target_new")),
        "legacy_target_unknown_rows_after": int(np.sum(roles == UNKNOWN_ROLE)),
        "legacy_manifest_new_tx_ids": sorted(new_ids),
        "legacy_manifest_unknown_tx_ids": sorted(unknown_ids),
    }


def _summary_row(count: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    old_acc = _float_metric(metrics, "old_acc")
    min_old = _float_metric(metrics, "min_old_class_acc")
    seen_new = _float_metric(metrics, "seen_new_acc")
    min_seen_new = _float_metric(metrics, "min_seen_new_class_acc")
    unknown_reject = _float_metric(metrics, "unknown_reject_rate")
    unknown_far = _float_metric(metrics, "unknown_FAR")
    return {
        "collab_count": int(count),
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": seen_new,
        "min_seen_new_class_acc": min_seen_new,
        "unknown_reject_rate": unknown_reject,
        "unknown_FAR": unknown_far,
        "known_coverage": _float_metric(metrics, "known_coverage"),
        "defer_rate": _float_metric(metrics, "defer_rate"),
        "request_more_rate": _float_metric(metrics, "request_more_rate"),
        "candidate_set_shell_veto_count": _int_metric(metrics, "candidate_set_shell_veto_count"),
        "candidate_set_shell_veto_rate": _float_metric(metrics, "candidate_set_shell_veto_rate"),
        "bytes_per_event": _float_metric(metrics, "bytes_per_event"),
        "latency_ms_p95": _float_metric(metrics, "latency_ms_p95"),
        "meets_old_goal": old_acc >= GOAL_OLD_ACC and min_old >= GOAL_MIN_OLD_CLASS_ACC,
        "meets_seen_new_goal": seen_new >= GOAL_SEEN_NEW_ACC and min_seen_new >= GOAL_MIN_SEEN_NEW_CLASS_ACC,
        "meets_unknown_goal": unknown_reject >= GOAL_UNKNOWN_REJECT and unknown_far <= (1.0 - GOAL_UNKNOWN_REJECT),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _safe_protocol_metadata(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    role_repair: Mapping[str, Any],
) -> dict[str, Any]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    target_unknown_count = int(np.sum(roles == UNKNOWN_ROLE))
    proxy_unknown_count = int(np.sum(roles == "proxy_unknown"))
    return {
        "diagnostic_only": True,
        "stage2_success_claim": False,
        "deployment_success_claim": False,
        "base_model": "ADV3B02_CORE90_SOFT_E200",
        "deployment_adapter": "qknn8",
        "ground_training_unknown_access": False,
        "target_unknown_training_count": 0,
        "target_unknown_calibration_count": 0,
        "target_unknown_query_count": target_unknown_count,
        "source_proxy_unknown_count": proxy_unknown_count,
        "uses_unknown_query_for_threshold": False,
        "unknown_query_eval_only": True,
        "threshold_scope": str(metadata.get("threshold_scope", "")),
        "threshold_scope_note": "known support, source proxy, or virtual negatives only; real target unknown query is eval-only",
        "source_receiver_ids": list(metadata.get("source_receiver_ids", [])),
        "target_receiver_ids": list(metadata.get("target_receiver_ids", [])),
        "old_tx_ids": list(metadata.get("old_tx_ids", [])),
        "seen_new_tx_ids": list(metadata.get("seen_new_tx_ids", [])),
        "unknown_tx_ids": list(metadata.get("unknown_tx_ids", [])),
        "target_channel_view": str(metadata.get("target_channel_view", "")),
        "k_shot": int(metadata.get("k_shot", 0)),
        "qknn_k": int(metadata.get("qknn_k", 0)),
        "event_alignment_policy": str(metadata.get("event_alignment_policy", "")),
        "receiver_count": len(list(metadata.get("target_receiver_ids", []))),
        "role_counts": _role_counts(payload),
        **dict(role_repair),
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(Path(args.feature_npz))
    role_repair: Mapping[str, Any] = {
        "legacy_role_repair_applied": False,
        "legacy_role_repair_reason": "disabled",
    }
    if bool(args.repair_legacy_roles_from_manifest):
        role_repair = _repair_legacy_roles_from_manifest(payload)
    evidence, metadata = build_collaborative_evidence(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        qknn_k=int(args.qknn_k),
        seed=int(args.seed),
        event_alignment_policy=str(args.event_alignment_policy),
        strict_event_min_receivers=int(args.strict_event_min_receivers),
        support_selection_policy=str(args.support_selection_policy),
        unknown_gate_mode=str(args.unknown_gate_mode),
        support_quantile=float(args.support_quantile),
        proxy_quantile=float(args.proxy_quantile),
        risk_temperature=float(args.risk_temperature),
        radius_quantile=float(args.radius_quantile),
        margin_quantile=float(args.margin_quantile),
        score_quantile=float(args.score_quantile),
        mahalanobis_quantile=float(args.mahalanobis_quantile),
        evt_tail_quantile=float(args.evt_tail_quantile),
        oldness_quantile=float(args.oldness_quantile),
        support_calibration_mode=str(args.support_calibration_mode),
        score_threshold_combine=str(args.score_threshold_combine),
        class_score_threshold_enabled=bool(args.class_score_threshold_enabled),
        class_conformal_enabled=bool(args.class_conformal_enabled),
        class_conformal_min_support=int(args.class_conformal_min_support),
        virtual_unknown_calibration_enabled=bool(args.virtual_unknown_calibration_enabled),
        virtual_unknown_samples_per_class=int(args.virtual_unknown_samples_per_class),
        virtual_unknown_risk_enabled=bool(args.virtual_unknown_risk_enabled),
        virtual_unknown_risk_samples_per_class=int(args.virtual_unknown_risk_samples_per_class),
        class_negative_risk_enabled=bool(args.class_negative_risk_enabled),
        class_negative_samples_per_class=int(args.class_negative_samples_per_class),
        class_shell_unknown_risk_enabled=bool(args.class_shell_unknown_risk_enabled),
        class_shell_radius_scale=float(args.class_shell_radius_scale),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        receiver_reliability_policy=str(args.receiver_reliability_policy),
        receiver_class_reliability_policy=str(args.receiver_class_reliability_policy),
        prototype_score_blend=float(args.prototype_score_blend),
        mahalanobis_score_blend=float(args.mahalanobis_score_blend),
        seen_new_old_contrast_weight=float(args.seen_new_old_contrast_weight),
        seen_new_old_contrast_margin=float(args.seen_new_old_contrast_margin),
        source_old_prototype_shrinkage_alpha=float(args.source_old_prototype_shrinkage_alpha),
        feature_adapter_policy=str(args.feature_adapter_policy),
        feature_adapter_strength=float(args.feature_adapter_strength),
        candidate_audit_unknown_risk_enabled=bool(args.candidate_audit_unknown_risk_enabled),
        class_verifier_policy=str(args.class_verifier_policy),
        class_verifier_top_m=int(args.class_verifier_top_m),
    )
    result = evaluate_collaborative_open_set_evidence(
        evidence,
        collab_counts=str(args.collab_counts),
        threshold_selection_label_scope=str(metadata["threshold_scope"]),
        unknown_query_eval_only=True,
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
        collab_group_policy=str(args.collab_group_policy),
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        receiver_selection_policy=str(args.receiver_selection_policy),
        fusion_policy=str(args.fusion_policy),
        collaboration_policy=str(args.collaboration_policy),
        label_fusion_policy=str(args.label_fusion_policy),
        unknown_risk_threshold=float(args.unknown_risk_threshold),
        accept_margin_threshold=float(args.accept_margin_threshold),
        unknown_quantile=float(args.unknown_quantile),
        consensus_gap_threshold=float(args.consensus_gap_threshold),
        consensus_score_threshold=float(args.consensus_score_threshold),
        scorer_component_vote_threshold=float(args.scorer_component_vote_threshold),
        class_reliability_policy=str(args.class_reliability_policy),
        receiver_class_reliability_policy=str(args.receiver_class_reliability_policy),
        seen_new_rescue_enabled=bool(args.seen_new_rescue_enabled),
        seen_new_rescue_risk_scale=float(args.seen_new_rescue_risk_scale),
        seen_new_rescue_min_score=float(args.seen_new_rescue_min_score),
        seen_new_rescue_min_margin=float(args.seen_new_rescue_min_margin),
        seen_new_rescue_min_agreement=float(args.seen_new_rescue_min_agreement),
        conformal_rescue_enabled=bool(args.conformal_rescue_enabled),
        conformal_rescue_min_pvalue=float(args.conformal_rescue_min_pvalue),
        conformal_rescue_risk_scale=float(args.conformal_rescue_risk_scale),
        conformal_rescue_min_agreement=float(args.conformal_rescue_min_agreement),
        rescue_unknown_veto_enabled=bool(args.rescue_unknown_veto_enabled),
        rescue_unknown_veto_event_risk=float(args.rescue_unknown_veto_event_risk),
        rescue_unknown_veto_label_risk=float(args.rescue_unknown_veto_label_risk),
        rescue_unknown_veto_shell_risk=float(args.rescue_unknown_veto_shell_risk),
        rescue_unknown_veto_component_agreement=float(
            args.rescue_unknown_veto_component_agreement
        ),
        rescue_unknown_veto_min_sources=int(args.rescue_unknown_veto_min_sources),
        rescue_unknown_veto_action=str(args.rescue_unknown_veto_action),
        class_set_gate_enabled=bool(args.class_set_gate_enabled),
        old_gate_min_receivers=int(args.old_gate_min_receivers),
        old_gate_max_effective_unknown_risk=float(args.old_gate_max_effective_unknown_risk),
        old_gate_max_component_agreement=float(args.old_gate_max_component_agreement),
        old_gate_min_support_density=float(args.old_gate_min_support_density),
        seen_new_gate_min_receivers=int(args.seen_new_gate_min_receivers),
        seen_new_gate_max_effective_unknown_risk=float(args.seen_new_gate_max_effective_unknown_risk),
        seen_new_gate_max_component_agreement=float(args.seen_new_gate_max_component_agreement),
        seen_new_gate_min_support_density=float(args.seen_new_gate_min_support_density),
        seen_new_contrast_gate_enabled=bool(args.seen_new_contrast_gate_enabled),
        seen_new_contrast_gate_min_delta=float(args.seen_new_contrast_gate_min_delta),
        seen_new_contrast_gate_min_receivers=int(args.seen_new_contrast_gate_min_receivers),
        seen_new_contrast_risk_relief_enabled=bool(args.seen_new_contrast_risk_relief_enabled),
        seen_new_contrast_risk_relief_min_delta=float(args.seen_new_contrast_risk_relief_min_delta),
        seen_new_contrast_risk_relief_min_receivers=int(args.seen_new_contrast_risk_relief_min_receivers),
        seen_new_contrast_risk_relief_min_support_count=int(
            args.seen_new_contrast_risk_relief_min_support_count
        ),
        seen_new_contrast_risk_relief_min_pvalue=float(
            args.seen_new_contrast_risk_relief_min_pvalue
        ),
        seen_new_contrast_risk_relief_min_receiver_class_reliability=float(
            args.seen_new_contrast_risk_relief_min_receiver_class_reliability
        ),
        seen_new_contrast_label_risk_scale=float(args.seen_new_contrast_label_risk_scale),
        seen_new_contrast_event_risk_scale=float(args.seen_new_contrast_event_risk_scale),
        seen_new_contrast_component_agreement_scale=float(args.seen_new_contrast_component_agreement_scale),
        candidate_set_min_receivers=int(args.candidate_set_min_receivers),
        candidate_set_min_top1_receivers=int(args.candidate_set_min_top1_receivers),
        candidate_set_min_conformal_pvalue=float(args.candidate_set_min_conformal_pvalue),
        candidate_set_max_label_unknown_risk=float(args.candidate_set_max_label_unknown_risk),
        candidate_set_max_event_unknown_risk=float(args.candidate_set_max_event_unknown_risk),
        candidate_set_max_label_shell_risk=float(args.candidate_set_max_label_shell_risk),
        candidate_set_max_label_risk_component_agreement=float(args.candidate_set_max_label_risk_component_agreement),
        candidate_set_min_label_receiver_class_reliability=float(
            args.candidate_set_min_label_receiver_class_reliability
        ),
        candidate_set_unknown_reject_risk=float(args.candidate_set_unknown_reject_risk),
        candidate_set_shell_reject_risk=float(args.candidate_set_shell_reject_risk),
        latency_budget_ms=float(args.latency_budget_ms),
        max_event_bytes=float(args.max_event_bytes),
        max_event_latency_ms=float(args.max_event_latency_ms),
        scorer_risk_components=metadata["active_risk_components"],
        include_event_results=bool(args.include_event_results),
    )
    summary_rows = [_summary_row(count, metrics) for count, metrics in sorted(result["counts"].items(), key=lambda kv: int(kv[0]))]
    all_goal_counts = [
        row["collab_count"]
        for row in summary_rows
        if row["meets_old_goal"] and row["meets_seen_new_goal"] and row["meets_unknown_goal"]
    ]
    result.update(
        {
            "diagnostic_name": "phase2_frozen_manytx_unknown_diagnostic",
            "feature_npz": str(args.feature_npz),
            "output_json": str(args.output_json),
            "output_summary_csv": str(args.output_summary_csv) if args.output_summary_csv else "",
            "output_evidence_csv": str(args.output_evidence_csv) if args.output_evidence_csv else "",
            "protocol_safety": _safe_protocol_metadata(metadata, payload, role_repair=role_repair),
            "goal_thresholds": {
                "old_acc": GOAL_OLD_ACC,
                "min_old_class_acc": GOAL_MIN_OLD_CLASS_ACC,
                "seen_new_acc": GOAL_SEEN_NEW_ACC,
                "min_seen_new_class_acc": GOAL_MIN_SEEN_NEW_CLASS_ACC,
                "unknown_reject_rate": GOAL_UNKNOWN_REJECT,
            },
            "goal_satisfied_counts": all_goal_counts,
            "summary_rows": summary_rows,
            "run_command_argv": [str(item) for item in sys.argv],
            "run_cwd": str(Path.cwd()),
            "python_executable": str(sys.executable),
            "qknn_metadata": metadata,
            "evidence_row_count": len(evidence),
        }
    )
    if args.output_summary_csv:
        _write_csv(Path(args.output_summary_csv), summary_rows)
    if args.output_evidence_csv:
        output_csv = Path(args.output_evidence_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted(set().union(*(row.keys() for row in evidence)))
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(evidence)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_summary_csv", type=Path, default=None)
    p.add_argument("--output_evidence_csv", type=Path, default=None)
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="available_up_to_k", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=_positive_int, default=1)
    p.add_argument("--k_shot", type=_positive_int, default=8)
    p.add_argument("--query_per_class", type=_positive_int, default=20)
    p.add_argument("--qknn_k", type=_positive_int, default=8)
    p.add_argument("--seed", type=int, default=4070606)
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--strict_event_min_receivers", type=int, default=0)
    p.add_argument(
        "--support_selection_policy",
        default="stable_first",
        choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"],
    )
    p.add_argument("--unknown_gate_mode", default="support_envelope")
    p.add_argument("--support_quantile", type=float, default=0.05)
    p.add_argument("--proxy_quantile", type=float, default=0.95)
    p.add_argument("--risk_temperature", type=float, default=0.035)
    p.add_argument("--radius_quantile", type=float, default=0.95)
    p.add_argument("--margin_quantile", type=float, default=0.05)
    p.add_argument("--score_quantile", type=float, default=0.05)
    p.add_argument("--mahalanobis_quantile", type=float, default=0.95)
    p.add_argument("--evt_tail_quantile", type=float, default=0.80)
    p.add_argument("--oldness_quantile", type=float, default=0.05)
    p.add_argument("--support_calibration_mode", default="self", choices=["self", "leave_one_out", "loo"])
    p.add_argument("--score_threshold_combine", default="max", choices=["max", "qknn_only", "centroid_only", "min", "mean"])
    p.add_argument("--class_score_threshold_enabled", action="store_true")
    p.add_argument("--class_conformal_enabled", action="store_true", default=True)
    p.add_argument("--class_conformal_min_support", type=_positive_int, default=2)
    p.add_argument("--virtual_unknown_calibration_enabled", action="store_true")
    p.add_argument("--virtual_unknown_samples_per_class", type=int, default=0)
    p.add_argument("--virtual_unknown_risk_enabled", action="store_true", default=True)
    p.add_argument("--virtual_unknown_risk_samples_per_class", type=int, default=2)
    p.add_argument("--class_negative_risk_enabled", action="store_true", default=True)
    p.add_argument("--class_negative_samples_per_class", type=int, default=2)
    p.add_argument("--class_shell_unknown_risk_enabled", action="store_true", default=True)
    p.add_argument("--class_shell_radius_scale", type=float, default=1.25)
    p.add_argument("--evidence_packet_bytes", type=float, default=96.0)
    p.add_argument("--receiver_reliability_policy", default="deployment_prior", choices=["deployment_prior", "support_density", "margin_density"])
    p.add_argument("--receiver_selection_policy", default="support_quality_prior", choices=["fixed_receiver_order", "reliability_prior", "support_quality_prior"])
    p.add_argument("--receiver_class_reliability_policy", default="support_calibrated", choices=["none", "support_calibrated"])
    p.add_argument("--prototype_score_blend", type=float, default=0.25)
    p.add_argument("--mahalanobis_score_blend", type=float, default=0.15)
    p.add_argument("--seen_new_old_contrast_weight", type=float, default=0.0)
    p.add_argument("--seen_new_old_contrast_margin", type=float, default=0.0)
    p.add_argument("--source_old_prototype_shrinkage_alpha", type=float, default=0.10)
    p.add_argument("--feature_adapter_policy", default="none", choices=["none", "support_center", "support_bn_affine"])
    p.add_argument("--feature_adapter_strength", type=float, default=0.0)
    p.add_argument("--candidate_audit_unknown_risk_enabled", action="store_true", default=True)
    p.add_argument("--class_verifier_policy", default="support_quality", choices=["none", "support_quality"])
    p.add_argument("--class_verifier_top_m", type=int, default=2)
    p.add_argument("--fusion_policy", default="old_protected_unknown_confirm_cvs")
    p.add_argument("--collaboration_policy", default="dual_route_cvs")
    p.add_argument("--label_fusion_policy", default="weighted_vote_margin")
    p.add_argument("--unknown_risk_threshold", type=float, default=0.68)
    p.add_argument("--accept_margin_threshold", type=float, default=0.01)
    p.add_argument("--unknown_quantile", type=float, default=0.75)
    p.add_argument("--consensus_gap_threshold", type=float, default=0.01)
    p.add_argument("--consensus_score_threshold", type=float, default=0.03)
    p.add_argument("--scorer_component_vote_threshold", type=float, default=0.50)
    p.add_argument("--class_reliability_policy", default="conformal_margin_risk")
    p.add_argument("--seen_new_rescue_enabled", action="store_true")
    p.add_argument("--seen_new_rescue_risk_scale", type=float, default=1.0)
    p.add_argument("--seen_new_rescue_min_score", type=float, default=0.0)
    p.add_argument("--seen_new_rescue_min_margin", type=float, default=0.0)
    p.add_argument("--seen_new_rescue_min_agreement", type=float, default=0.5)
    p.add_argument("--conformal_rescue_enabled", action="store_true")
    p.add_argument("--conformal_rescue_min_pvalue", type=float, default=0.05)
    p.add_argument("--conformal_rescue_risk_scale", type=float, default=0.5)
    p.add_argument("--conformal_rescue_min_agreement", type=float, default=0.5)
    p.add_argument("--rescue_unknown_veto_enabled", action="store_true")
    p.add_argument("--rescue_unknown_veto_event_risk", type=float, default=1.0)
    p.add_argument("--rescue_unknown_veto_label_risk", type=float, default=1.0)
    p.add_argument("--rescue_unknown_veto_shell_risk", type=float, default=1.0)
    p.add_argument("--rescue_unknown_veto_component_agreement", type=float, default=1.0)
    p.add_argument("--rescue_unknown_veto_min_sources", type=_positive_int, default=1)
    p.add_argument(
        "--rescue_unknown_veto_action",
        choices=["unknown_reject", "defer", "request_more"],
        default="unknown_reject",
    )
    p.add_argument("--class_set_gate_enabled", action="store_true", default=True)
    p.add_argument("--old_gate_min_receivers", type=_positive_int, default=1)
    p.add_argument("--old_gate_max_effective_unknown_risk", type=float, default=0.90)
    p.add_argument("--old_gate_max_component_agreement", type=float, default=0.85)
    p.add_argument("--old_gate_min_support_density", type=float, default=0.05)
    p.add_argument("--seen_new_gate_min_receivers", type=_positive_int, default=1)
    p.add_argument("--seen_new_gate_max_effective_unknown_risk", type=float, default=0.90)
    p.add_argument("--seen_new_gate_max_component_agreement", type=float, default=0.85)
    p.add_argument("--seen_new_gate_min_support_density", type=float, default=0.05)
    p.add_argument("--seen_new_contrast_gate_enabled", action="store_true")
    p.add_argument("--seen_new_contrast_gate_min_delta", type=float, default=0.0)
    p.add_argument("--seen_new_contrast_gate_min_receivers", type=_positive_int, default=1)
    p.add_argument("--seen_new_contrast_risk_relief_enabled", action="store_true")
    p.add_argument("--seen_new_contrast_risk_relief_min_delta", type=float, default=0.0)
    p.add_argument("--seen_new_contrast_risk_relief_min_receivers", type=_positive_int, default=1)
    p.add_argument("--seen_new_contrast_risk_relief_min_support_count", type=int, default=0)
    p.add_argument("--seen_new_contrast_risk_relief_min_pvalue", type=float, default=0.0)
    p.add_argument(
        "--seen_new_contrast_risk_relief_min_receiver_class_reliability",
        type=float,
        default=0.0,
    )
    p.add_argument("--seen_new_contrast_label_risk_scale", type=float, default=1.0)
    p.add_argument("--seen_new_contrast_event_risk_scale", type=float, default=1.0)
    p.add_argument("--seen_new_contrast_component_agreement_scale", type=float, default=1.0)
    p.add_argument("--candidate_set_min_receivers", type=_positive_int, default=2)
    p.add_argument("--candidate_set_min_top1_receivers", type=int, default=1)
    p.add_argument("--candidate_set_min_conformal_pvalue", type=float, default=0.0)
    p.add_argument("--candidate_set_max_label_unknown_risk", type=float, default=0.82)
    p.add_argument("--candidate_set_max_event_unknown_risk", type=float, default=0.88)
    p.add_argument("--candidate_set_max_label_shell_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_max_label_risk_component_agreement", type=float, default=0.72)
    p.add_argument("--candidate_set_min_label_receiver_class_reliability", type=float, default=0.0)
    p.add_argument("--candidate_set_unknown_reject_risk", type=float, default=0.72)
    p.add_argument("--candidate_set_shell_reject_risk", type=float, default=1.0e12)
    p.add_argument("--latency_budget_ms", type=float, default=0.0)
    p.add_argument("--max_event_bytes", type=float, default=0.0)
    p.add_argument("--max_event_latency_ms", type=float, default=0.0)
    p.add_argument("--include_event_results", action="store_true")
    p.add_argument(
        "--repair_legacy_roles_from_manifest",
        action="store_true",
        help=(
            "Compatibility-only diagnostic repair for older feature NPZs that stored "
            "manifest unknown_tx_ids under dataset_role=target_new. The repair is "
            "recorded in protocol_safety and remains non-deployment diagnostic."
        ),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_diagnostic(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
