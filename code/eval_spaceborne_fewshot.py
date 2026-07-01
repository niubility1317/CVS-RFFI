#!/usr/bin/env python
"""Evaluate feature-level CVS-SFE/CVS-FTRC protocols from saved z_id features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from cvsrffi.adaptation_safety import DEFAULT_SFE_ROLLBACK_RULES, evaluate_rollback_gate, rules_from_policy
from cvsrffi.spaceborne_fewshot import (
    AdaptationResult,
    LIFECYCLE_ACTIVE_LOCAL,
    LIFECYCLE_GROUND_CONFIRMED,
    LIFECYCLE_QUARANTINE,
    OrbitAdaptiveMSEHead,
    OpenSetGateConfig,
    UNKNOWN_LABEL,
    accepted_only_online_update,
    apply_class_envelope_gate,
    apply_density_shell_inlier_gate,
    apply_identity_consensus_arbitration,
    apply_old_unknown_acceptance_guard,
    apply_old_primary_acceptance_gate,
    apply_pre_reject_defer_arbitration,
    apply_three_way_decision_head,
    apply_pseudo_unknown_void_gate,
    apply_retention_rescue_gate,
    apply_seen_new_registration_override,
    apply_siamese_verifier_to_ambiguous,
    apply_support_conformal_arbitration,
    apply_support_reconstruction_arbitration,
    apply_source_looo_unknown_risk_arbitration,
    apply_two_branch_background_guard,
    build_prototype_set,
    calibrate_anchor_density_gates,
    calibrate_class_envelope_gates,
    calibrate_thresholds,
    compute_open_set_metrics,
    fit_low_compute_target_adapter,
    fit_siamese_verifier,
    generate_pseudo_unknown_features,
    apply_old80_first_head,
    predict_with_prototypes,
    predict_with_oa_mse_head,
    register_new_classes,
    register_old_classes,
    run_ftrc_calibration,
    run_sfe_enrollment,
)
from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays, parse_tx_id_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["sfe", "ftrc", "source_open_set"], required=True)
    parser.add_argument("--feature_npz", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--unknown_threshold", type=float, default=0.70)
    parser.add_argument("--gate_mode", default="cosine", choices=["none", "cosine", "margin", "mahalanobis", "openmax", "evt", "combined", "oa_mse"])
    parser.add_argument("--min_margin", type=float, default=None)
    parser.add_argument("--max_mahalanobis", type=float, default=None)
    parser.add_argument("--openmax_tail_size", type=int, default=20)
    parser.add_argument("--openmax_quantile", type=float, default=0.95)
    parser.add_argument("--openmax_min_threshold", type=float, default=0.02)
    parser.add_argument(
        "--lifecycle_initial_state",
        default=LIFECYCLE_QUARANTINE,
        choices=[LIFECYCLE_QUARANTINE, LIFECYCLE_ACTIVE_LOCAL, LIFECYCLE_GROUND_CONFIRMED],
    )
    parser.add_argument("--rollback_policy_json", type=Path, default=None)
    parser.add_argument("--baseline_metrics_json", type=Path, default=None)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--dry_run_synthetic", action="store_true")
    parser.add_argument("--features_key", default="features")
    parser.add_argument("--tx_ids_key", default="tx_ids")
    parser.add_argument("--source_tx_ids", default=None, help="Comma-separated source/old TX identities for full-feature SFE NPZ.")
    parser.add_argument("--target_old_tx_ids", default=None, help="Optional target-domain old TX identities for Stage2 target-old query rows.")
    parser.add_argument("--new_tx_ids", default=None, help="Comma-separated new TX identities for full-feature SFE NPZ.")
    parser.add_argument("--unknown_tx_ids", default=None, help="Optional comma-separated unknown TX identities for open-set query samples.")
    parser.add_argument("--source_proto_per_tx", type=int, default=20)
    parser.add_argument("--source_query_per_tx", type=int, default=20)
    parser.add_argument("--target_old_support_per_tx", type=int, default=0)
    parser.add_argument("--target_old_query_per_tx", type=int, default=None)
    parser.add_argument("--query_per_tx", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--oa_mse_adapter_rank", type=int, default=2)
    parser.add_argument("--oa_mse_adapter_kind", default="low_rank", choices=["low_rank", "residual_mlp"])
    parser.add_argument("--oa_mse_adapter_steps", type=int, default=40)
    parser.add_argument("--oa_mse_adapter_lr", type=float, default=0.05)
    parser.add_argument("--oa_mse_source_anchor_weight", type=float, default=0.05)
    parser.add_argument("--oa_mse_source_ce_weight", type=float, default=0.10)
    parser.add_argument("--oa_mse_unknown_moat_weight", type=float, default=0.10)
    parser.add_argument("--oa_mse_unknown_moat_margin", type=float, default=0.45)
    parser.add_argument("--pseudo_unknown_samples_per_pair", type=int, default=4)
    parser.add_argument("--pseudo_unknown_offset_scale", type=float, default=0.15)
    parser.add_argument("--pseudo_unknown_source_boundary_samples_per_pair", type=int, default=0)
    parser.add_argument("--pseudo_unknown_source_boundary_offset_scale", type=float, default=0.20)
    parser.add_argument("--pseudo_unknown_target_shift_samples_per_class", type=int, default=0)
    parser.add_argument("--pseudo_unknown_target_shift_offset_scale", type=float, default=0.20)
    parser.add_argument("--pseudo_unknown_target_halo_samples_per_class", type=int, default=0)
    parser.add_argument("--pseudo_unknown_target_halo_offset_scale", type=float, default=0.35)
    parser.add_argument("--pseudo_unknown_target_ring_samples_per_class", type=int, default=0)
    parser.add_argument("--pseudo_unknown_target_ring_offset_scale", type=float, default=0.45)
    parser.add_argument("--oa_mse_old_bridge_weight", type=float, default=0.10)
    parser.add_argument("--old_bridge_samples_per_class", type=int, default=2)
    parser.add_argument("--old_bridge_max_mix", type=float, default=0.85)
    parser.add_argument("--oa_mse_support_contrast_weight", type=float, default=0.0)
    parser.add_argument("--old_support_contrast_negative_margin", type=float, default=0.78)
    parser.add_argument("--old_support_contrast_positive_margin", type=float, default=0.88)
    parser.add_argument("--oa_mse_support_center_ce_weight", type=float, default=0.0)
    parser.add_argument("--support_center_temperature", type=float, default=0.10)
    parser.add_argument("--support_center_margin", type=float, default=0.10)
    parser.add_argument("--oa_mse_soft_proto_weight", type=float, default=0.0)
    parser.add_argument("--soft_proto_topk", type=int, default=2)
    parser.add_argument("--soft_proto_temperature", type=float, default=0.10)
    parser.add_argument("--oa_mse_soft_proto_boundary_weight", type=float, default=0.0)
    parser.add_argument("--soft_proto_boundary_margin", type=float, default=0.15)
    parser.add_argument("--oa_mse_multiproto_score", action="store_true")
    parser.add_argument("--multiproto_topk", type=int, default=2)
    parser.add_argument("--multiproto_temperature", type=float, default=0.10)
    parser.add_argument("--multiproto_score_weight", type=float, default=1.0)
    parser.add_argument("--oa_mse_mixture_consistency_gate", action="store_true")
    parser.add_argument("--mixture_consistency_min_cos", type=float, default=-1.0)
    parser.add_argument("--mixture_consistency_max_residual", type=float, default=1.0e6)
    parser.add_argument("--mixture_consistency_min_margin", type=float, default=-1.0e6)
    parser.add_argument("--mixture_consistency_action", default="uncertain", choices=["uncertain", "defer", "reject"])
    parser.add_argument("--oa_mse_anchor_density_gate", action="store_true")
    parser.add_argument("--anchor_density_topk", type=int, default=3)
    parser.add_argument("--anchor_density_temperature", type=float, default=0.08)
    parser.add_argument("--anchor_density_min_quantile", type=float, default=0.05)
    parser.add_argument("--anchor_density_margin_quantile", type=float, default=0.05)
    parser.add_argument("--anchor_density_gate_action", default="uncertain", choices=["uncertain", "reject"])
    parser.add_argument("--oa_mse_class_envelope_gate", action="store_true")
    parser.add_argument("--class_envelope_evidence_quantile", type=float, default=0.05)
    parser.add_argument("--class_envelope_residual_quantile", type=float, default=0.95)
    parser.add_argument("--class_envelope_score_quantile", type=float, default=0.05)
    parser.add_argument("--class_envelope_margin_quantile", type=float, default=0.05)
    parser.add_argument("--class_envelope_evidence_slack", type=float, default=0.02)
    parser.add_argument("--class_envelope_residual_slack", type=float, default=0.02)
    parser.add_argument("--class_envelope_score_slack", type=float, default=0.05)
    parser.add_argument("--class_envelope_margin_slack", type=float, default=0.02)
    parser.add_argument("--class_envelope_min_failures", type=int, default=1)
    parser.add_argument("--class_envelope_gate_action", default="reject", choices=["uncertain", "reject"])
    parser.add_argument("--oa_mse_old_primary_gate", action="store_true")
    parser.add_argument("--old_primary_min_old_support_evidence_delta", type=float, default=0.0)
    parser.add_argument("--old_primary_min_old_support_anchor_delta", type=float, default=-0.02)
    parser.add_argument("--old_primary_min_old_support_anchor_margin", type=float, default=0.0)
    parser.add_argument("--old_primary_min_score_margin", type=float, default=0.0)
    parser.add_argument("--old_primary_require_soft_mixture", action="store_true")
    parser.add_argument("--old_primary_min_soft_mixture_margin", type=float, default=-1.0e6)
    parser.add_argument("--old_primary_min_soft_mixture_cos", type=float, default=-1.0)
    parser.add_argument("--old_primary_max_soft_mixture_residual", type=float, default=1.0e6)
    parser.add_argument("--old_primary_require_support_knn", action="store_true")
    parser.add_argument("--old_primary_no_support_knn_label_match", action="store_true")
    parser.add_argument("--old_primary_min_support_knn_margin", type=float, default=0.0)
    parser.add_argument("--old_primary_max_support_knn_seen_new_minus_old", type=float, default=None)
    parser.add_argument("--old_primary_min_old_drift_cos", type=float, default=-1.0)
    parser.add_argument("--old_primary_max_old_drift_dist", type=float, default=1.0e6)
    parser.add_argument("--old_primary_require_class_envelope", action="store_true")
    parser.add_argument("--old_primary_unknown_veto_background_score", type=float, default=0.86)
    parser.add_argument("--old_primary_unknown_veto_background_margin", type=float, default=0.10)
    parser.add_argument("--old_primary_unknown_veto_min_sources", type=int, default=1)
    parser.add_argument("--old_primary_fail_action", default="defer", choices=["reject", "defer", "uncertain"])
    parser.add_argument("--old_primary_unknown_veto_action", default="reject", choices=["reject", "defer", "uncertain"])
    parser.add_argument("--old_primary_promote_rescue_candidates", action="store_true")
    parser.add_argument("--oa_mse_density_shell_gate", action="store_true")
    parser.add_argument("--density_shell_old_min_evidence_delta", type=float, default=-0.04)
    parser.add_argument("--density_shell_old_min_anchor_delta", type=float, default=-0.08)
    parser.add_argument("--density_shell_old_min_density_delta", type=float, default=-0.06)
    parser.add_argument("--density_shell_seen_new_min_evidence_delta", type=float, default=-0.04)
    parser.add_argument("--density_shell_seen_new_min_anchor_delta", type=float, default=-0.08)
    parser.add_argument("--density_shell_seen_new_min_density_delta", type=float, default=-0.06)
    parser.add_argument("--density_shell_accept_background_margin", type=float, default=0.18)
    parser.add_argument("--density_shell_reject_background_score", type=float, default=0.86)
    parser.add_argument("--density_shell_reject_background_margin", type=float, default=0.14)
    parser.add_argument("--density_shell_reject_min_failed_shells", type=int, default=2)
    parser.add_argument("--oa_mse_identity_consensus_arbitration", action="store_true")
    parser.add_argument("--identity_consensus_old_min_evidence_delta", type=float, default=-0.06)
    parser.add_argument("--identity_consensus_old_min_anchor_delta", type=float, default=-0.10)
    parser.add_argument("--identity_consensus_old_min_density_delta", type=float, default=-0.08)
    parser.add_argument("--identity_consensus_seen_new_min_evidence_delta", type=float, default=-0.04)
    parser.add_argument("--identity_consensus_seen_new_min_anchor_delta", type=float, default=-0.08)
    parser.add_argument("--identity_consensus_seen_new_min_density_delta", type=float, default=-0.06)
    parser.add_argument("--identity_consensus_min_identity_margin", type=float, default=-0.05)
    parser.add_argument("--identity_consensus_background_accept_margin", type=float, default=0.22)
    parser.add_argument("--identity_consensus_reject_background_score", type=float, default=0.90)
    parser.add_argument("--identity_consensus_reject_background_margin", type=float, default=0.18)
    parser.add_argument("--identity_consensus_reject_min_identity_failures", type=int, default=4)
    parser.add_argument("--identity_consensus_support_background_cap", action="store_true")
    parser.add_argument("--identity_consensus_support_background_cap_quantile", type=float, default=0.90)
    parser.add_argument("--identity_consensus_support_background_cap_slack", type=float, default=0.05)
    parser.add_argument("--identity_consensus_support_background_cap_min_anchors", type=int, default=2)
    parser.add_argument("--oa_mse_support_conformal_arbitration", action="store_true")
    parser.add_argument("--support_conformal_calibration_quantile", type=float, default=0.05)
    parser.add_argument("--support_conformal_conformity_slack", type=float, default=0.12)
    parser.add_argument("--support_conformal_anchor_margin_slack", type=float, default=0.06)
    parser.add_argument("--support_conformal_background_score", type=float, default=0.82)
    parser.add_argument("--support_conformal_background_margin", type=float, default=0.08)
    parser.add_argument("--support_conformal_hard_reject_margin", type=float, default=0.18)
    parser.add_argument("--support_conformal_reject_min_failures", type=int, default=2)
    parser.add_argument("--support_conformal_reject_action", default="reject", choices=["reject", "defer"])
    parser.add_argument("--oa_mse_support_reconstruction_arbitration", action="store_true")
    parser.add_argument("--support_reconstruction_rank", type=int, default=2)
    parser.add_argument("--support_reconstruction_residual_quantile", type=float, default=0.95)
    parser.add_argument("--support_reconstruction_residual_slack", type=float, default=0.04)
    parser.add_argument("--support_reconstruction_min_residual_floor", type=float, default=0.03)
    parser.add_argument("--support_reconstruction_negative_scale", type=float, default=0.55)
    parser.add_argument("--support_reconstruction_negative_margin", type=float, default=-0.02)
    parser.add_argument("--support_reconstruction_hard_residual_margin", type=float, default=0.08)
    parser.add_argument("--support_reconstruction_background_score", type=float, default=0.86)
    parser.add_argument("--support_reconstruction_background_margin", type=float, default=0.12)
    parser.add_argument("--support_reconstruction_reject_min_failures", type=int, default=2)
    parser.add_argument("--support_reconstruction_reject_action", default="reject", choices=["reject", "defer"])
    parser.add_argument("--oa_mse_pre_reject_defer_arbitration", action="store_true")
    parser.add_argument("--oa_mse_three_way_decision_head", action="store_true")
    parser.add_argument("--oa_mse_three_way_head_weight", type=float, default=0.0)
    parser.add_argument("--three_way_head_temperature", type=float, default=0.10)
    parser.add_argument("--three_way_head_known_margin", type=float, default=0.08)
    parser.add_argument("--three_way_head_background_margin", type=float, default=0.08)
    parser.add_argument("--three_way_head_support_ce_weight", type=float, default=1.0)
    parser.add_argument("--three_way_head_pseudo_ce_weight", type=float, default=0.35)
    parser.add_argument("--three_way_head_support_background_margin_weight", type=float, default=1.0)
    parser.add_argument("--three_way_head_pseudo_margin_weight", type=float, default=0.50)
    parser.add_argument("--three_way_accept_prob", type=float, default=0.50)
    parser.add_argument("--three_way_reject_prob", type=float, default=0.55)
    parser.add_argument("--three_way_defer_prob", type=float, default=0.45)
    parser.add_argument("--three_way_known_background_margin", type=float, default=0.02)
    parser.add_argument("--three_way_reject_margin", type=float, default=0.04)
    parser.add_argument("--three_way_old_seen_ambiguity_margin", type=float, default=0.04)
    parser.add_argument("--three_way_defer_action", default="uncertain", choices=["uncertain", "defer"])
    parser.add_argument("--three_way_decision_policy", default="background_competition", choices=["background_competition", "class_first", "evidence_balanced"])
    parser.add_argument("--three_way_known_floor", action="store_true")
    parser.add_argument("--three_way_known_floor_action", default="defer", choices=["accept", "defer", "uncertain"])
    parser.add_argument("--three_way_known_floor_old_min_evidence_delta", type=float, default=-0.04)
    parser.add_argument("--three_way_known_floor_old_min_anchor_delta", type=float, default=-0.08)
    parser.add_argument("--three_way_known_floor_old_min_anchor_margin", type=float, default=-0.04)
    parser.add_argument("--three_way_known_floor_old_min_score_margin", type=float, default=-0.12)
    parser.add_argument("--three_way_known_floor_seen_new_min_evidence_delta", type=float, default=-0.04)
    parser.add_argument("--three_way_known_floor_seen_new_min_anchor_delta", type=float, default=-0.08)
    parser.add_argument("--three_way_known_floor_seen_new_min_score_margin", type=float, default=-0.12)
    parser.add_argument("--three_way_known_floor_background_override_prob", type=float, default=0.995)
    parser.add_argument("--three_way_known_floor_background_override_margin", type=float, default=1.0)
    parser.add_argument("--pre_reject_old_min_evidence_delta", type=float, default=0.0)
    parser.add_argument("--pre_reject_old_min_anchor_delta", type=float, default=-0.02)
    parser.add_argument("--pre_reject_old_min_anchor_margin", type=float, default=0.0)
    parser.add_argument("--pre_reject_old_min_score_margin", type=float, default=-0.02)
    parser.add_argument("--pre_reject_seen_new_min_evidence_delta", type=float, default=0.0)
    parser.add_argument("--pre_reject_seen_new_min_anchor_delta", type=float, default=0.0)
    parser.add_argument("--pre_reject_seen_new_min_score_margin", type=float, default=-0.05)
    parser.add_argument("--pre_reject_max_background_score", type=float, default=0.74)
    parser.add_argument("--pre_reject_max_background_margin", type=float, default=0.10)
    parser.add_argument("--pre_reject_defer_background_score", type=float, default=0.70)
    parser.add_argument("--pre_reject_defer_background_margin", type=float, default=0.04)
    parser.add_argument("--pre_reject_reject_background_score", type=float, default=0.82)
    parser.add_argument("--pre_reject_reject_background_margin", type=float, default=0.12)
    parser.add_argument("--pre_reject_defer_action", default="uncertain", choices=["uncertain", "defer"])
    parser.add_argument("--pre_reject_support_neighborhood_retention", action="store_true")
    parser.add_argument("--pre_reject_support_retention_old_min_evidence_delta", type=float, default=0.02)
    parser.add_argument("--pre_reject_support_retention_old_min_anchor_delta", type=float, default=-0.04)
    parser.add_argument("--pre_reject_support_retention_old_min_anchor_margin", type=float, default=-0.02)
    parser.add_argument("--pre_reject_support_retention_old_min_score_margin", type=float, default=-0.04)
    parser.add_argument("--pre_reject_support_retention_seen_new_min_evidence_delta", type=float, default=0.02)
    parser.add_argument("--pre_reject_support_retention_seen_new_min_anchor_delta", type=float, default=-0.04)
    parser.add_argument("--pre_reject_support_retention_seen_new_min_score_margin", type=float, default=-0.08)
    parser.add_argument("--pre_reject_support_retention_max_background_score", type=float, default=0.96)
    parser.add_argument("--pre_reject_support_retention_max_background_margin", type=float, default=0.30)
    parser.add_argument("--pre_reject_support_retention_require_source_looo_pass", action="store_true")
    parser.add_argument("--pre_reject_support_retention_source_looo_max_failures", type=int, default=0)
    parser.add_argument("--oa_mse_retention_rescue_gate", action="store_true")
    parser.add_argument("--retention_rescue_old_min_evidence_delta", type=float, default=0.02)
    parser.add_argument("--retention_rescue_old_min_anchor_delta", type=float, default=-0.01)
    parser.add_argument("--retention_rescue_old_min_anchor_margin", type=float, default=0.0)
    parser.add_argument("--retention_rescue_old_min_score_margin", type=float, default=0.0)
    parser.add_argument("--retention_rescue_seen_new_min_evidence_delta", type=float, default=0.02)
    parser.add_argument("--retention_rescue_seen_new_min_anchor_delta", type=float, default=0.0)
    parser.add_argument("--retention_rescue_seen_new_min_score_margin", type=float, default=-0.02)
    parser.add_argument("--retention_rescue_max_background_score", type=float, default=0.70)
    parser.add_argument("--retention_rescue_max_background_margin", type=float, default=0.06)
    parser.add_argument("--retention_rescue_candidate_only", action="store_true")
    parser.add_argument("--oa_mse_void_background_weight", type=float, default=0.0)
    parser.add_argument("--oa_mse_negative_anchor_weight", type=float, default=0.0)
    parser.add_argument("--negative_anchor_margin", type=float, default=0.12)
    parser.add_argument("--negative_anchor_temperature", type=float, default=0.10)
    parser.add_argument("--negative_anchor_max_anchors", type=int, default=256)
    parser.add_argument("--oa_mse_void_gate", action="store_true")
    parser.add_argument("--oa_mse_void_gate_min_score", type=float, default=0.55)
    parser.add_argument("--oa_mse_void_gate_min_margin", type=float, default=0.05)
    parser.add_argument("--oa_mse_old_neighborhood_weight", type=float, default=0.10)
    parser.add_argument("--old_neighborhood_samples_per_class", type=int, default=2)
    parser.add_argument("--old_neighborhood_radius", type=float, default=0.06)
    parser.add_argument("--oa_mse_old_surrogate_margin_weight", type=float, default=0.05)
    parser.add_argument("--old_surrogate_margin", type=float, default=0.10)
    parser.add_argument("--oa_mse_source_looo_unknown_weight", type=float, default=0.0)
    parser.add_argument("--source_looo_unknown_margin", type=float, default=0.35)
    parser.add_argument("--source_looo_interclass_margin", type=float, default=0.08)
    parser.add_argument("--source_looo_max_samples_per_class", type=int, default=24)
    parser.add_argument("--oa_mse_source_looo_risk_arbitration", action="store_true")
    parser.add_argument("--source_looo_risk_quantile", type=float, default=0.85)
    parser.add_argument("--source_looo_risk_slack", type=float, default=0.0)
    parser.add_argument("--source_looo_risk_min_score_margin", type=float, default=0.02)
    parser.add_argument("--source_looo_risk_min_known_evidence_delta", type=float, default=-0.08)
    parser.add_argument("--source_looo_risk_background_score", type=float, default=0.86)
    parser.add_argument("--source_looo_risk_background_margin", type=float, default=0.10)
    parser.add_argument("--source_looo_risk_reject_min_failures", type=int, default=2)
    parser.add_argument("--source_looo_risk_reject_action", default="reject", choices=["reject", "defer"])
    parser.add_argument("--oa_mse_known_coverage_weight", type=float, default=0.0)
    parser.add_argument("--known_coverage_margin", type=float, default=0.12)
    parser.add_argument("--known_coverage_min_affinity", type=float, default=0.35)
    parser.add_argument("--known_coverage_max_samples", type=int, default=256)
    parser.add_argument("--old_surrogate_evidence_margin", type=float, default=0.0)
    parser.add_argument("--old_surrogate_reject_relax", type=float, default=0.0)
    parser.add_argument("--oa_mse_siamese_quantile", type=float, default=0.10)
    parser.add_argument("--oa_mse_siamese_accept_threshold", type=float, default=0.50)
    parser.add_argument("--oa_mse_siamese_unknown_veto", action="store_true")
    parser.add_argument("--oa_mse_siamese_unknown_veto_mode", default="any", choices=["any", "coupled"])
    parser.add_argument("--oa_mse_siamese_min_old_support_evidence_delta", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_old_surrogate_reject_delta", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_energy_delta", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_mahalanobis_delta", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_accept_delta", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_old_support_anchor_margin", type=float, default=None)
    parser.add_argument("--oa_mse_siamese_min_veto_failures", type=int, default=1)
    parser.add_argument("--oa_mse_old_unknown_acceptance_guard", action="store_true")
    parser.add_argument("--oa_mse_old_unknown_guard_min_old_support_evidence_delta", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_old_surrogate_reject_delta", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_energy_delta", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_mahalanobis_delta", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_accept_delta", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_old_support_anchor_margin", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_best_old_score", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_margin", type=float, default=None)
    parser.add_argument("--oa_mse_old_unknown_guard_min_failures", type=int, default=1)
    parser.add_argument(
        "--oa_mse_old80_head_mode",
        default="disabled",
        choices=["disabled", "fused_centroid", "support_centroid", "support_knn1", "support_knn3", "support_cv_select"],
    )
    parser.add_argument(
        "--old80_head_apply_policy",
        default="replace_all",
        choices=["replace_all", "rescue_rejected", "replace_unknown"],
    )
    parser.add_argument("--old80_head_fusion_rho", type=float, default=0.75)
    parser.add_argument("--old80_head_knn_k", type=int, default=3)
    parser.add_argument("--old_anchor_override_min_quality", type=float, default=0.55)
    parser.add_argument("--old_retention_quantile", type=float, default=0.95)
    parser.add_argument("--oa_mse_support_retention_guard", action="store_true")
    parser.add_argument("--support_retention_guard_quantile", type=float, default=0.05)
    parser.add_argument("--support_retention_guard_slack", type=float, default=0.02)
    parser.add_argument("--oa_mse_two_branch_background_guard", action="store_true")
    parser.add_argument("--two_branch_bg_min_score", type=float, default=0.62)
    parser.add_argument("--two_branch_bg_min_margin", type=float, default=-0.02)
    parser.add_argument("--two_branch_old_support_evidence_delta", type=float, default=0.0)
    parser.add_argument("--two_branch_old_anchor_delta", type=float, default=-0.02)
    parser.add_argument("--two_branch_old_anchor_margin", type=float, default=0.0)
    parser.add_argument("--two_branch_seen_new_evidence_delta", type=float, default=0.0)
    parser.add_argument("--two_branch_seen_new_anchor_delta", type=float, default=0.0)
    parser.add_argument("--oa_mse_seen_new_registration_override", action="store_true")
    parser.add_argument("--seen_new_override_min_evidence_delta", type=float, default=0.0)
    parser.add_argument("--seen_new_override_min_anchor_delta", type=float, default=0.0)
    parser.add_argument("--seen_new_override_min_affinity_delta", type=float, default=-0.02)
    parser.add_argument("--seen_new_override_min_residual_delta", type=float, default=-0.02)
    parser.add_argument("--seen_new_override_min_score_margin", type=float, default=-0.10)
    parser.add_argument("--seen_new_override_min_seen_vs_old_evidence_margin", type=float, default=0.02)
    parser.add_argument("--seen_new_override_max_background_score", type=float, default=0.72)
    parser.add_argument("--seen_new_override_max_background_margin", type=float, default=0.08)
    parser.add_argument("--seen_new_override_min_support_knn_seen_new_minus_old", type=float, default=None)
    parser.add_argument("--seen_new_override_min_support_knn_margin", type=float, default=None)
    parser.add_argument(
        "--oa_mse_adapter_selection_policy",
        default="final",
        choices=[
            "final",
            "proxy_line_search",
            "target_boundary_guard",
            "retention_risk_balanced",
            "constrained_retention_risk",
            "identity_preserving",
            "identity_preserving_risk",
            "support_cv_constrained",
            "support_cv_risk_balanced",
            "identity_preserving_cv",
            "identity_preserving_risk_cv",
        ],
    )
    parser.add_argument(
        "--oa_mse_adapter_alpha_eval_sweep",
        action="store_true",
        help="Eval-only diagnostic: report OA-MSE metrics for fixed adapter alpha candidates without using query labels for training or calibration.",
    )
    parser.add_argument("--old_acc_target", type=float, default=0.90)
    parser.add_argument("--seen_new_acc_target", type=float, default=0.75)
    parser.add_argument("--manifest_json", type=Path, default=None)
    parser.add_argument("--score_table_csv", type=Path, default=None)
    return parser.parse_args()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json_mapping(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _standard_loss_trace_payload(result: AdaptationResult) -> dict:
    telemetry = result.telemetry if isinstance(result.telemetry, dict) else {}
    oa_mse = telemetry.get("oa_mse_onboard_adaptation", {})
    target_adapter = oa_mse.get("target_adapter", {}) if isinstance(oa_mse, dict) else {}
    if not isinstance(target_adapter, dict) or not bool(target_adapter.get("enabled", False)):
        return {
            "loss_trace_status": "EVAL_ONLY_NO_TRAINING_LOSS",
            "loss_trace_source": "no_target_adapter_training",
            "loss_trace_schema": None,
            "loss_trace": [],
            "loss_initial": None,
            "loss_final": None,
            "loss_terms": {},
        }
    trace = target_adapter.get("loss_trace", [])
    if not isinstance(trace, list):
        trace = []
    steps = int(target_adapter.get("steps", 0) or 0)
    status = "PRESENT" if trace else ("EVAL_ONLY_NO_TRAINING_LOSS" if steps <= 0 else "MISSING_LOSS_TELEMETRY")
    return {
        "loss_trace_status": status,
        "loss_trace_source": "telemetry.oa_mse_onboard_adaptation.target_adapter",
        "loss_trace_schema": target_adapter.get("loss_trace_schema"),
        "loss_trace": trace,
        "loss_initial": target_adapter.get("loss_initial"),
        "loss_final": target_adapter.get("loss_final"),
        "loss_terms": target_adapter.get("loss_terms", {}),
    }


def _promote_target_view_manifest(manifest: dict) -> dict:
    embedded = manifest.get("embedded_manifest")
    extra_metadata = manifest.get("extra_metadata")
    if not isinstance(embedded, dict) and isinstance(extra_metadata, dict):
        embedded = extra_metadata.get("embedded_manifest")
    channel_profile = manifest.get("channel_profile")
    if not isinstance(channel_profile, dict) and isinstance(embedded, dict):
        channel_profile = embedded.get("channel_profile")
    target_old = channel_profile.get("target_old") if isinstance(channel_profile, dict) else None
    target_new = channel_profile.get("target_new") if isinstance(channel_profile, dict) else None
    target_unknown = channel_profile.get("target_unknown") if isinstance(channel_profile, dict) else None
    target_view = ""
    for candidate_view in (
        manifest.get("target_new_channel_view"),
        manifest.get("target_unknown_channel_view"),
        manifest.get("target_channel_view"),
        (target_old or {}).get("view"),
        (target_unknown or {}).get("view"),
        (target_new or {}).get("view"),
    ):
        normalized_view = str(candidate_view or "").strip().lower()
        if normalized_view and normalized_view not in {"disabled", "none", "old_unknown_only"}:
            target_view = normalized_view
            break
    if target_view in {"satellite/leo", "leo", "satellite"}:
        manifest["target_new_channel_view"] = "satellite"
        manifest["target_channel_view"] = "satellite/LEO"
        manifest.setdefault("deployment_primary_view", "satellite/LEO target view")
    elif target_view:
        manifest["target_new_channel_view"] = target_view
        manifest["target_channel_view"] = target_view
        manifest.setdefault("deployment_primary_view", "clean control/source reference")
    if "target_channel_scenarios" not in manifest and (isinstance(target_old, dict) or isinstance(target_new, dict)):
        profile = target_old if isinstance(target_old, dict) else target_new
        scenarios = profile.get("scenarios") if isinstance(profile, dict) else None
        if isinstance(scenarios, list):
            manifest["target_channel_scenarios"] = [str(item) for item in scenarios]
    return manifest


def _tensor_to_float_array(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return np.asarray(value.detach().cpu().reshape(-1).tolist(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _auroc(labels_unknown: np.ndarray, scores: np.ndarray) -> float:
    pos = labels_unknown.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores.astype(np.float64))
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _fpr_at_tpr(labels_unknown: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    pos = labels_unknown.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    best = float("inf")
    for threshold in np.unique(scores.astype(np.float64))[::-1]:
        pred_unknown = scores >= threshold
        tpr = float((pred_unknown & pos).sum()) / float(n_pos)
        fpr = float((pred_unknown & ~pos).sum()) / float(n_neg)
        if tpr >= float(target_tpr):
            best = min(best, fpr)
    if math.isinf(best):
        return float("nan")
    return float(best)


def _unknown_score_columns(result, gate_config: OpenSetGateConfig) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    scores = _tensor_to_float_array(result.scores)
    if scores is None:
        raise ValueError("result.scores is required for score table output")
    columns: dict[str, np.ndarray] = {"cosine_unknown_score": 1.0 - scores}
    margins = _tensor_to_float_array(getattr(result, "margins", None))
    if margins is not None:
        columns["negative_margin"] = -margins
    mahal = _tensor_to_float_array(getattr(result, "mahalanobis", None))
    if mahal is not None:
        columns["mahalanobis"] = mahal
    openmax_distance = _tensor_to_float_array(getattr(result, "openmax_distance", None))
    if openmax_distance is not None:
        columns["openmax_distance"] = openmax_distance
    energy = _tensor_to_float_array(getattr(result, "energy", None))
    if energy is not None:
        columns["energy"] = energy
    subspace_residual = _tensor_to_float_array(getattr(result, "subspace_residual", None))
    if subspace_residual is not None:
        columns["subspace_residual"] = subspace_residual
    seen_new_evidence = _tensor_to_float_array(getattr(result, "seen_new_evidence", None))
    if seen_new_evidence is not None:
        columns["seen_new_evidence"] = seen_new_evidence
        columns["negative_seen_new_evidence"] = -seen_new_evidence
    seen_new_support_affinity = _tensor_to_float_array(getattr(result, "seen_new_support_affinity", None))
    if seen_new_support_affinity is not None:
        columns["seen_new_support_affinity"] = seen_new_support_affinity
        columns["negative_seen_new_support_affinity"] = -seen_new_support_affinity
    seen_new_support_residual = _tensor_to_float_array(getattr(result, "seen_new_support_residual", None))
    if seen_new_support_residual is not None:
        columns["seen_new_support_residual"] = seen_new_support_residual
    seen_new_anchor_similarity = _tensor_to_float_array(getattr(result, "seen_new_anchor_similarity", None))
    if seen_new_anchor_similarity is not None:
        columns["seen_new_anchor_similarity"] = seen_new_anchor_similarity
        columns["negative_seen_new_anchor_similarity"] = -seen_new_anchor_similarity
    seen_new_anchor_delta = _tensor_to_float_array(getattr(result, "seen_new_anchor_delta", None))
    if seen_new_anchor_delta is not None:
        columns["seen_new_anchor_delta"] = seen_new_anchor_delta

    mode = str(gate_config.mode).lower()
    if mode == "oa_mse" and "negative_seen_new_evidence" in columns:
        return "negative_seen_new_evidence", columns["negative_seen_new_evidence"], columns
    if mode == "oa_mse" and "energy" in columns:
        return "energy", columns["energy"], columns
    if mode == "mahalanobis" and "mahalanobis" in columns:
        return "mahalanobis", columns["mahalanobis"], columns
    if mode in {"openmax", "evt", "combined"} and "openmax_distance" in columns:
        return "openmax_distance", columns["openmax_distance"], columns
    return "cosine_unknown_score", columns["cosine_unknown_score"], columns


def _add_open_set_curve_metrics(metrics: dict, query_labels: torch.Tensor, result, gate_config: OpenSetGateConfig) -> None:
    labels = np.asarray(query_labels.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    labels_unknown = labels == -1
    score_kind, unknown_scores, all_scores = _unknown_score_columns(result, gate_config)
    metrics["auroc"] = _auroc(labels_unknown, unknown_scores)
    metrics["fpr95"] = _fpr_at_tpr(labels_unknown, unknown_scores, target_tpr=0.95)
    metrics["unknown_score_kind"] = score_kind
    for name, values in all_scores.items():
        if name == score_kind:
            continue
        metrics[f"auroc_{name}"] = _auroc(labels_unknown, values)
        metrics[f"fpr95_{name}"] = _fpr_at_tpr(labels_unknown, values, target_tpr=0.95)


def _add_split_confusion_metrics(metrics: dict, query_labels: torch.Tensor, result, old_labels: set[int], new_labels: set[int] | None) -> None:
    labels = np.asarray(query_labels.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    predicted = np.asarray(result.predicted_labels.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    old = set(int(v) for v in old_labels)
    new = set(int(v) for v in (new_labels or set()))
    old_acc = metrics.get("old_class_accuracy")
    new_acc = metrics.get("new_class_accuracy")
    if old_acc is not None:
        metrics["old_acc"] = float(old_acc)
    if new_acc is not None:
        metrics["seen_new_acc"] = float(new_acc)
    if "unknown_false_accept_rate" in metrics:
        metrics["unknown_FAR"] = float(metrics["unknown_false_accept_rate"])
    if old_acc is not None and new_acc is not None and math.isfinite(float(old_acc)) and math.isfinite(float(new_acc)):
        denom = float(old_acc) + float(new_acc)
        metrics["H_old_new"] = float(0.0 if denom <= 0.0 else 2.0 * float(old_acc) * float(new_acc) / denom)
    unknown_mask = labels == -1
    new_mask = np.asarray([int(v) in new for v in labels], dtype=bool)
    old_mask = np.asarray([int(v) in old for v in labels], dtype=bool)
    pred_new = np.asarray([int(v) in new for v in predicted], dtype=bool)
    pred_old = np.asarray([int(v) in old for v in predicted], dtype=bool)
    pred_reject = predicted == -1
    if int(unknown_mask.sum()) > 0:
        denom = float(int(unknown_mask.sum()))
        metrics["unknown_to_seen_new_rate"] = float((unknown_mask & pred_new).sum()) / denom
        metrics["unknown_to_old_rate"] = float((unknown_mask & pred_old).sum()) / denom
    if int(new_mask.sum()) > 0:
        denom = float(int(new_mask.sum()))
        metrics["seen_new_reject_rate"] = float((new_mask & pred_reject).sum()) / denom
        metrics["seen_new_to_old_rate"] = float((new_mask & pred_old).sum()) / denom
    if int(old_mask.sum()) > 0:
        denom = float(int(old_mask.sum()))
        metrics["old_reject_rate"] = float((old_mask & pred_reject).sum()) / denom
        metrics["old_to_seen_new_rate"] = float((old_mask & pred_new).sum()) / denom


def _write_score_table(path: Path, payload: dict[str, np.ndarray], query_labels: torch.Tensor, result, gate_config: OpenSetGateConfig) -> None:
    labels = np.asarray(query_labels.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    predicted = np.asarray(result.predicted_labels.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    accepted = np.asarray(result.accepted.detach().cpu().reshape(-1).tolist(), dtype=bool)
    candidate_labels_tensor = getattr(result, "candidate_labels", None)
    if candidate_labels_tensor is None:
        candidate_labels = predicted
    else:
        candidate_labels = np.asarray(candidate_labels_tensor.detach().cpu().reshape(-1).tolist(), dtype=np.int64)
    scores = _tensor_to_float_array(result.scores)
    margins = _tensor_to_float_array(getattr(result, "margins", None))
    mahal = _tensor_to_float_array(getattr(result, "mahalanobis", None))
    openmax_distance = _tensor_to_float_array(getattr(result, "openmax_distance", None))
    energy = _tensor_to_float_array(getattr(result, "energy", None))
    subspace_residual = _tensor_to_float_array(getattr(result, "subspace_residual", None))
    seen_new_evidence = _tensor_to_float_array(getattr(result, "seen_new_evidence", None))
    seen_new_support_affinity = _tensor_to_float_array(getattr(result, "seen_new_support_affinity", None))
    seen_new_support_residual = _tensor_to_float_array(getattr(result, "seen_new_support_residual", None))
    seen_new_anchor_similarity = _tensor_to_float_array(getattr(result, "seen_new_anchor_similarity", None))
    seen_new_anchor_delta = _tensor_to_float_array(getattr(result, "seen_new_anchor_delta", None))
    diagnostics = {
        str(name): _tensor_to_float_array(value)
        for name, value in (getattr(result, "diagnostics", {}) or {}).items()
    }
    decisions = list(getattr(result, "decisions", []) or [])
    if len(decisions) != labels.shape[0]:
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    score_kind, unknown_scores, all_scores = _unknown_score_columns(result, gate_config)
    query_tx_ids = np.asarray(payload.get("query_tx_ids", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_roles = np.asarray(payload.get("query_roles", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_dataset_roles = np.asarray(payload.get("query_dataset_roles", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_rx_ids = np.asarray(payload.get("query_rx_ids", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_day_ids = np.asarray(payload.get("query_day_ids", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_channel_views = np.asarray(payload.get("query_channel_views", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_sat_scenarios = np.asarray(payload.get("query_sat_scenarios", np.asarray([""] * labels.shape[0])), dtype=str).reshape(-1)
    query_sample_indices = np.asarray(payload.get("query_sample_indices", np.arange(labels.shape[0])), dtype=np.int64).reshape(-1)
    source_label_set = set(int(v) for v in payload.get("source_labels", np.asarray([], dtype=np.int64)).reshape(-1).tolist())
    support_label_set = set(int(v) for v in payload.get("support_labels", np.asarray([], dtype=np.int64)).reshape(-1).tolist())
    gate_reasons = list(getattr(result, "gate_reasons", []) or [])
    if len(gate_reasons) != labels.shape[0]:
        gate_reasons = list(result.telemetry.get("gate_reasons", [])) if isinstance(result.telemetry, dict) else []
    if len(gate_reasons) != labels.shape[0]:
        gate_reasons = [""] * labels.shape[0]

    def group_for(label: int) -> str:
        if int(label) == -1:
            return "unknown"
        if int(label) in source_label_set:
            return "old"
        if int(label) in support_label_set:
            return "new"
        return "other_known"

    def outcome_for(true_label: int, predicted_label: int, accepted_value: bool) -> str:
        true_group = group_for(int(true_label))
        pred_group = group_for(int(predicted_label))
        if pred_group == "unknown" or not bool(accepted_value):
            return f"{true_group}_rejected"
        if true_group == pred_group and int(true_label) == int(predicted_label):
            return f"{true_group}_correct"
        return f"{true_group}_to_{pred_group}"

    def value_at(values: np.ndarray | None, i: int):
        if values is None or i >= values.shape[0]:
            return None
        value = float(values[i])
        if not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "true_label",
        "true_group",
        "query_tx_id",
        "query_role",
        "query_dataset_role",
        "query_rx_id",
        "query_day_id",
        "query_channel_view",
        "query_sat_scenario",
        "query_sample_index",
        "candidate_label",
        "candidate_group",
        "predicted_label",
        "predicted_group",
        "accepted",
        "outcome_code",
        "score",
        "unknown_score_kind",
        "unknown_score",
        "cosine_unknown_score",
        "margin",
        "mahalanobis",
        "openmax_distance",
        "energy",
        "subspace_residual",
        "best_old_label",
        "best_seen_new_label",
        "best_old_score",
        "best_seen_new_score",
        "seen_new_minus_old_score",
        "old_drift_cos",
        "old_drift_dist",
        "old_effective_rho",
        "old_support_count",
        "old_support_compactness",
        "old_support_anchor_similarity",
        "old_support_anchor_margin",
        "old_support_anchor_delta",
        "old_support_evidence",
        "old_support_evidence_delta",
        "old_surrogate_evidence_delta",
        "old_surrogate_reject_evidence_delta",
        "old_support_quality",
        "old_support_quality_delta",
        "anchor_density",
        "anchor_density_margin",
        "anchor_density_delta",
        "anchor_density_margin_delta",
        "support_knn_label",
        "support_knn_score",
        "support_knn_margin",
        "support_knn_old_score",
        "support_knn_seen_new_score",
        "support_knn_seen_new_minus_old",
        "support_knn_topk",
        "soft_mixture_score",
        "soft_mixture_cos",
        "soft_mixture_residual",
        "soft_mixture_maha",
        "soft_mixture_score_margin",
        "soft_mixture_consistency_pass",
        "class_envelope_label",
        "class_envelope_evidence",
        "class_envelope_residual",
        "class_envelope_score",
        "class_envelope_margin",
        "class_envelope_failure_count",
        "class_envelope_reject",
        "old_primary_label",
        "old_primary_candidate",
        "old_primary_evidence_delta",
        "old_primary_anchor_delta",
        "old_primary_anchor_margin",
        "old_primary_score_margin",
        "old_primary_soft_mixture_margin",
        "old_primary_soft_mixture_cos",
        "old_primary_soft_mixture_residual",
        "old_primary_support_knn_label",
        "old_primary_support_knn_margin",
        "old_primary_support_knn_seen_new_minus_old",
        "old_primary_drift_cos",
        "old_primary_drift_dist",
        "old_primary_background_score",
        "old_primary_background_margin",
        "old_primary_prior_veto_count",
        "old_primary_evidence_pass",
        "old_primary_anchor_delta_pass",
        "old_primary_anchor_margin_pass",
        "old_primary_score_margin_pass",
        "old_primary_soft_mixture_pass",
        "old_primary_support_knn_pass",
        "old_primary_drift_pass",
        "old_primary_class_envelope_pass",
        "old_primary_consistency_pass",
        "old_primary_unknown_veto",
        "old_primary_unknown_veto_applied",
        "old_primary_blocked_accept",
        "old_primary_rescue_promoted",
        "old_primary_rescue_blocked",
        "old80_first_label",
        "old80_first_score",
        "old80_first_margin",
        "old80_first_applied",
        "old80_first_support_cv_acc",
        "old80_first_support_cv_count",
        "old80_first_mode_code",
        "density_shell_old_label",
        "density_shell_seen_new_label",
        "density_shell_chosen_label",
        "density_shell_old_evidence_delta",
        "density_shell_seen_new_evidence_delta",
        "density_shell_old_density_delta",
        "density_shell_seen_new_density_delta",
        "density_shell_background_score",
        "density_shell_background_margin",
        "density_shell_old_pass",
        "density_shell_seen_new_pass",
        "density_shell_accept",
        "density_shell_reject",
        "density_shell_failed_shell_count",
        "identity_consensus_old_label",
        "identity_consensus_seen_new_label",
        "identity_consensus_chosen_label",
        "identity_consensus_old_score",
        "identity_consensus_seen_new_score",
        "identity_consensus_chosen_score",
        "identity_consensus_margin",
        "identity_consensus_old_evidence_delta",
        "identity_consensus_seen_new_evidence_delta",
        "identity_consensus_old_anchor_delta",
        "identity_consensus_seen_new_anchor_delta",
        "identity_consensus_old_density_delta",
        "identity_consensus_seen_new_density_delta",
        "identity_consensus_background_score",
        "identity_consensus_background_margin",
        "identity_consensus_support_background_cap",
        "identity_consensus_support_background_cap_pass",
        "identity_consensus_old_pass",
        "identity_consensus_seen_new_pass",
        "identity_consensus_accept",
        "identity_consensus_reject",
        "identity_consensus_failure_count",
        "support_conformal_label",
        "support_conformal_score",
        "support_conformal_floor",
        "support_conformal_margin",
        "support_conformal_anchor_margin",
        "support_conformal_anchor_margin_floor",
        "support_conformal_background_score",
        "support_conformal_background_margin",
        "support_conformal_pass",
        "support_conformal_reject",
        "support_conformal_failure_count",
        "support_reconstruction_label",
        "support_reconstruction_residual",
        "support_reconstruction_residual_ceiling",
        "support_reconstruction_residual_margin",
        "support_reconstruction_center_cosine",
        "support_reconstruction_negative_score",
        "support_reconstruction_negative_margin",
        "support_reconstruction_background_score",
        "support_reconstruction_background_margin",
        "support_reconstruction_pass",
        "support_reconstruction_reject",
        "support_reconstruction_failure_count",
        "source_looo_risk_label",
        "source_looo_risk_score",
        "source_looo_risk_floor",
        "source_looo_risk_margin",
        "source_looo_second_score",
        "source_looo_score_margin",
        "source_looo_known_evidence_delta",
        "source_looo_background_score",
        "source_looo_background_margin",
        "source_looo_pass",
        "source_looo_reject",
        "source_looo_failure_count",
        "pair_verifier_label",
        "pair_verifier_prob",
        "pair_verifier_threshold",
        "pair_verifier_called",
        "pair_verifier_veto",
        "three_way_label",
        "three_way_old_score",
        "three_way_seen_new_score",
        "three_way_background_score",
        "three_way_old_prob",
        "three_way_seen_new_prob",
        "three_way_background_prob",
        "three_way_old_known_prob",
        "three_way_seen_new_known_prob",
        "three_way_known_prob_class_first",
        "three_way_known_background_gap",
        "three_way_background_margin",
        "three_way_old_seen_gap",
        "three_way_background_available",
        "three_way_base_reject",
        "three_way_known_floor",
        "three_way_old_floor",
        "three_way_seen_new_floor",
        "three_way_extreme_background",
        "three_way_floor_accept",
        "three_way_floor_defer",
        "three_way_class_first_known_evidence",
        "three_way_evidence_balanced_known_evidence",
        "three_way_reject_suppressed_by_floor",
        "three_way_accept",
        "three_way_reject",
        "three_way_defer",
        "pre_reject_arbitration_label",
        "pre_reject_arbitration_score",
        "pre_reject_arbitration_margin",
        "pre_reject_arbitration_evidence_delta",
        "pre_reject_arbitration_anchor_delta",
        "pre_reject_arbitration_anchor_margin",
        "pre_reject_arbitration_background_score",
        "pre_reject_arbitration_background_margin",
        "pre_reject_arbitration_background_available",
        "pre_reject_arbitration_background_accept_ok",
        "pre_reject_arbitration_background_defer_risk",
        "pre_reject_arbitration_background_reject_risk",
        "pre_reject_arbitration_evidence_ok",
        "pre_reject_arbitration_support_retention",
        "pre_reject_arbitration_support_retention_source_looo_block",
        "pre_reject_arbitration_extreme_background",
        "pre_reject_arbitration_accept",
        "pre_reject_arbitration_reject",
        "pre_reject_arbitration_defer",
        "pre_reject_arbitration_uncertain",
        "retention_rescue_label",
        "retention_rescue_score",
        "retention_rescue_margin",
        "retention_rescue_evidence_delta",
        "retention_rescue_anchor_delta",
        "retention_rescue_anchor_margin",
        "retention_rescue_background_score",
        "retention_rescue_background_margin",
        "retention_rescue_eligible",
        "retention_rescue_accept",
        "two_branch_background_score",
        "two_branch_known_score",
        "two_branch_background_margin",
        "two_branch_background_risk",
        "two_branch_support_override",
        "two_branch_background_reject",
        "seen_new_override_label",
        "seen_new_override_evidence_delta",
        "seen_new_override_anchor_delta",
        "seen_new_override_affinity_delta",
        "seen_new_override_residual_delta",
        "seen_new_override_seen_minus_old_evidence",
        "seen_new_override_seen_minus_old_score",
        "seen_new_override_background_score",
        "seen_new_override_background_margin",
        "seen_new_override_background_risk",
        "seen_new_override_support_knn_seen_new_minus_old",
        "seen_new_override_support_knn_margin",
        "seen_new_override_support_knn_pass",
        "seen_new_registration_override",
        "residual_delta",
        "mahalanobis_delta",
        "margin_delta",
        "energy_delta",
        "evt_delta",
        "min_accept_delta",
        "seen_new_evidence",
        "seen_new_support_affinity",
        "seen_new_support_residual",
        "seen_new_evidence_delta",
        "seen_new_affinity_delta",
        "seen_new_residual_delta",
        "seen_new_anchor_similarity",
        "seen_new_anchor_delta",
        "decision",
        "gate_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, label in enumerate(labels.tolist()):
            writer.writerow(
                {
                    "row": int(i),
                    "true_label": int(label),
                    "true_group": group_for(int(label)),
                    "query_tx_id": str(query_tx_ids[i]) if i < query_tx_ids.shape[0] else "",
                    "query_role": str(query_roles[i]) if i < query_roles.shape[0] else "",
                    "query_dataset_role": str(query_dataset_roles[i]) if i < query_dataset_roles.shape[0] else "",
                    "query_rx_id": str(query_rx_ids[i]) if i < query_rx_ids.shape[0] else "",
                    "query_day_id": str(query_day_ids[i]) if i < query_day_ids.shape[0] else "",
                    "query_channel_view": str(query_channel_views[i]) if i < query_channel_views.shape[0] else "",
                    "query_sat_scenario": str(query_sat_scenarios[i]) if i < query_sat_scenarios.shape[0] else "",
                    "query_sample_index": int(query_sample_indices[i]) if i < query_sample_indices.shape[0] else int(i),
                    "candidate_label": int(candidate_labels[i]) if i < candidate_labels.shape[0] else int(predicted[i]),
                    "candidate_group": group_for(int(candidate_labels[i])) if i < candidate_labels.shape[0] else group_for(int(predicted[i])),
                    "predicted_label": int(predicted[i]),
                    "predicted_group": group_for(int(predicted[i])),
                    "accepted": bool(accepted[i]),
                    "outcome_code": outcome_for(int(label), int(predicted[i]), bool(accepted[i])),
                    "score": value_at(scores, i),
                    "unknown_score_kind": score_kind,
                    "unknown_score": value_at(unknown_scores, i),
                    "cosine_unknown_score": value_at(all_scores["cosine_unknown_score"], i),
                    "margin": value_at(margins, i),
                    "mahalanobis": value_at(mahal, i),
                    "openmax_distance": value_at(openmax_distance, i),
                    "energy": value_at(energy, i),
                    "subspace_residual": value_at(subspace_residual, i),
                    "best_old_label": value_at(diagnostics.get("best_old_label"), i),
                    "best_seen_new_label": value_at(diagnostics.get("best_seen_new_label"), i),
                    "best_old_score": value_at(diagnostics.get("best_old_score"), i),
                    "best_seen_new_score": value_at(diagnostics.get("best_seen_new_score"), i),
                    "seen_new_minus_old_score": value_at(diagnostics.get("seen_new_minus_old_score"), i),
                    "old_drift_cos": value_at(diagnostics.get("old_drift_cos"), i),
                    "old_drift_dist": value_at(diagnostics.get("old_drift_dist"), i),
                    "old_effective_rho": value_at(diagnostics.get("old_effective_rho"), i),
                    "old_support_count": value_at(diagnostics.get("old_support_count"), i),
                    "old_support_compactness": value_at(diagnostics.get("old_support_compactness"), i),
                    "old_support_anchor_similarity": value_at(diagnostics.get("old_support_anchor_similarity"), i),
                    "old_support_anchor_margin": value_at(diagnostics.get("old_support_anchor_margin"), i),
                    "old_support_anchor_delta": value_at(diagnostics.get("old_support_anchor_delta"), i),
                    "old_support_evidence": value_at(diagnostics.get("old_support_evidence"), i),
                    "old_support_evidence_delta": value_at(diagnostics.get("old_support_evidence_delta"), i),
                    "old_surrogate_evidence_delta": value_at(diagnostics.get("old_surrogate_evidence_delta"), i),
                    "old_surrogate_reject_evidence_delta": value_at(diagnostics.get("old_surrogate_reject_evidence_delta"), i),
                    "old_support_quality": value_at(diagnostics.get("old_support_quality"), i),
                    "old_support_quality_delta": value_at(diagnostics.get("old_support_quality_delta"), i),
                    "anchor_density": value_at(diagnostics.get("anchor_density"), i),
                    "anchor_density_margin": value_at(diagnostics.get("anchor_density_margin"), i),
                    "anchor_density_delta": value_at(diagnostics.get("anchor_density_delta"), i),
                    "anchor_density_margin_delta": value_at(diagnostics.get("anchor_density_margin_delta"), i),
                    "support_knn_label": value_at(diagnostics.get("support_knn_label"), i),
                    "support_knn_score": value_at(diagnostics.get("support_knn_score"), i),
                    "support_knn_margin": value_at(diagnostics.get("support_knn_margin"), i),
                    "support_knn_old_score": value_at(diagnostics.get("support_knn_old_score"), i),
                    "support_knn_seen_new_score": value_at(diagnostics.get("support_knn_seen_new_score"), i),
                    "support_knn_seen_new_minus_old": value_at(diagnostics.get("support_knn_seen_new_minus_old"), i),
                    "support_knn_topk": value_at(diagnostics.get("support_knn_topk"), i),
                    "soft_mixture_score": value_at(diagnostics.get("soft_mixture_score"), i),
                    "soft_mixture_cos": value_at(diagnostics.get("soft_mixture_cos"), i),
                    "soft_mixture_residual": value_at(diagnostics.get("soft_mixture_residual"), i),
                    "soft_mixture_maha": value_at(diagnostics.get("soft_mixture_maha"), i),
                    "soft_mixture_score_margin": value_at(diagnostics.get("soft_mixture_score_margin"), i),
                    "soft_mixture_consistency_pass": value_at(diagnostics.get("soft_mixture_consistency_pass_mask"), i),
                    "class_envelope_label": value_at(diagnostics.get("class_envelope_label"), i),
                    "class_envelope_evidence": value_at(diagnostics.get("class_envelope_evidence"), i),
                    "class_envelope_residual": value_at(diagnostics.get("class_envelope_residual"), i),
                    "class_envelope_score": value_at(diagnostics.get("class_envelope_score"), i),
                    "class_envelope_margin": value_at(diagnostics.get("class_envelope_margin"), i),
                    "class_envelope_failure_count": value_at(diagnostics.get("class_envelope_failure_count"), i),
                    "class_envelope_reject": value_at(diagnostics.get("class_envelope_reject_mask"), i),
                    "old_primary_label": value_at(diagnostics.get("old_primary_label"), i),
                    "old_primary_candidate": value_at(diagnostics.get("old_primary_candidate_mask"), i),
                    "old_primary_evidence_delta": value_at(diagnostics.get("old_primary_evidence_delta"), i),
                    "old_primary_anchor_delta": value_at(diagnostics.get("old_primary_anchor_delta"), i),
                    "old_primary_anchor_margin": value_at(diagnostics.get("old_primary_anchor_margin"), i),
                    "old_primary_score_margin": value_at(diagnostics.get("old_primary_score_margin"), i),
                    "old_primary_soft_mixture_margin": value_at(diagnostics.get("old_primary_soft_mixture_margin"), i),
                    "old_primary_soft_mixture_cos": value_at(diagnostics.get("old_primary_soft_mixture_cos"), i),
                    "old_primary_soft_mixture_residual": value_at(diagnostics.get("old_primary_soft_mixture_residual"), i),
                    "old_primary_support_knn_label": value_at(diagnostics.get("old_primary_support_knn_label"), i),
                    "old_primary_support_knn_margin": value_at(diagnostics.get("old_primary_support_knn_margin"), i),
                    "old_primary_support_knn_seen_new_minus_old": value_at(
                        diagnostics.get("old_primary_support_knn_seen_new_minus_old"), i
                    ),
                    "old_primary_drift_cos": value_at(diagnostics.get("old_primary_drift_cos"), i),
                    "old_primary_drift_dist": value_at(diagnostics.get("old_primary_drift_dist"), i),
                    "old_primary_background_score": value_at(diagnostics.get("old_primary_background_score"), i),
                    "old_primary_background_margin": value_at(diagnostics.get("old_primary_background_margin"), i),
                    "old_primary_prior_veto_count": value_at(diagnostics.get("old_primary_prior_veto_count"), i),
                    "old_primary_evidence_pass": value_at(diagnostics.get("old_primary_evidence_pass_mask"), i),
                    "old_primary_anchor_delta_pass": value_at(diagnostics.get("old_primary_anchor_delta_pass_mask"), i),
                    "old_primary_anchor_margin_pass": value_at(diagnostics.get("old_primary_anchor_margin_pass_mask"), i),
                    "old_primary_score_margin_pass": value_at(diagnostics.get("old_primary_score_margin_pass_mask"), i),
                    "old_primary_soft_mixture_pass": value_at(diagnostics.get("old_primary_soft_mixture_pass_mask"), i),
                    "old_primary_support_knn_pass": value_at(diagnostics.get("old_primary_support_knn_pass_mask"), i),
                    "old_primary_drift_pass": value_at(diagnostics.get("old_primary_drift_pass_mask"), i),
                    "old_primary_class_envelope_pass": value_at(
                        diagnostics.get("old_primary_class_envelope_pass_mask"), i
                    ),
                    "old_primary_consistency_pass": value_at(diagnostics.get("old_primary_consistency_pass_mask"), i),
                    "old_primary_unknown_veto": value_at(diagnostics.get("old_primary_unknown_veto_mask"), i),
                    "old_primary_unknown_veto_applied": value_at(
                        diagnostics.get("old_primary_unknown_veto_applied_mask"), i
                    ),
                    "old_primary_blocked_accept": value_at(diagnostics.get("old_primary_blocked_accept_mask"), i),
                    "old_primary_rescue_promoted": value_at(
                        diagnostics.get("old_primary_rescue_promoted_mask"), i
                    ),
                    "old_primary_rescue_blocked": value_at(
                        diagnostics.get("old_primary_rescue_blocked_mask"), i
                    ),
                    "old80_first_label": value_at(diagnostics.get("old80_first_label"), i),
                    "old80_first_score": value_at(diagnostics.get("old80_first_score"), i),
                    "old80_first_margin": value_at(diagnostics.get("old80_first_margin"), i),
                    "old80_first_applied": value_at(diagnostics.get("old80_first_applied_mask"), i),
                    "old80_first_support_cv_acc": value_at(diagnostics.get("old80_first_support_cv_acc"), i),
                    "old80_first_support_cv_count": value_at(diagnostics.get("old80_first_support_cv_count"), i),
                    "old80_first_mode_code": value_at(diagnostics.get("old80_first_mode_code"), i),
                    "density_shell_old_label": value_at(diagnostics.get("density_shell_old_label"), i),
                    "density_shell_seen_new_label": value_at(diagnostics.get("density_shell_seen_new_label"), i),
                    "density_shell_chosen_label": value_at(diagnostics.get("density_shell_chosen_label"), i),
                    "density_shell_old_evidence_delta": value_at(diagnostics.get("density_shell_old_evidence_delta"), i),
                    "density_shell_seen_new_evidence_delta": value_at(diagnostics.get("density_shell_seen_new_evidence_delta"), i),
                    "density_shell_old_density_delta": value_at(diagnostics.get("density_shell_old_density_delta"), i),
                    "density_shell_seen_new_density_delta": value_at(diagnostics.get("density_shell_seen_new_density_delta"), i),
                    "density_shell_background_score": value_at(diagnostics.get("density_shell_background_score"), i),
                    "density_shell_background_margin": value_at(diagnostics.get("density_shell_background_margin"), i),
                    "density_shell_old_pass": value_at(diagnostics.get("density_shell_old_pass_mask"), i),
                    "density_shell_seen_new_pass": value_at(diagnostics.get("density_shell_seen_new_pass_mask"), i),
                    "density_shell_accept": value_at(diagnostics.get("density_shell_accept_mask"), i),
                    "density_shell_reject": value_at(diagnostics.get("density_shell_reject_mask"), i),
                    "density_shell_failed_shell_count": value_at(diagnostics.get("density_shell_failed_shell_count"), i),
                    "identity_consensus_old_label": value_at(diagnostics.get("identity_consensus_old_label"), i),
                    "identity_consensus_seen_new_label": value_at(diagnostics.get("identity_consensus_seen_new_label"), i),
                    "identity_consensus_chosen_label": value_at(diagnostics.get("identity_consensus_chosen_label"), i),
                    "identity_consensus_old_score": value_at(diagnostics.get("identity_consensus_old_score"), i),
                    "identity_consensus_seen_new_score": value_at(diagnostics.get("identity_consensus_seen_new_score"), i),
                    "identity_consensus_chosen_score": value_at(diagnostics.get("identity_consensus_chosen_score"), i),
                    "identity_consensus_margin": value_at(diagnostics.get("identity_consensus_margin"), i),
                    "identity_consensus_old_evidence_delta": value_at(diagnostics.get("identity_consensus_old_evidence_delta"), i),
                    "identity_consensus_seen_new_evidence_delta": value_at(diagnostics.get("identity_consensus_seen_new_evidence_delta"), i),
                    "identity_consensus_old_anchor_delta": value_at(diagnostics.get("identity_consensus_old_anchor_delta"), i),
                    "identity_consensus_seen_new_anchor_delta": value_at(diagnostics.get("identity_consensus_seen_new_anchor_delta"), i),
                    "identity_consensus_old_density_delta": value_at(diagnostics.get("identity_consensus_old_density_delta"), i),
                    "identity_consensus_seen_new_density_delta": value_at(diagnostics.get("identity_consensus_seen_new_density_delta"), i),
                    "identity_consensus_background_score": value_at(diagnostics.get("identity_consensus_background_score"), i),
                    "identity_consensus_background_margin": value_at(diagnostics.get("identity_consensus_background_margin"), i),
                    "identity_consensus_support_background_cap": value_at(diagnostics.get("identity_consensus_support_background_cap"), i),
                    "identity_consensus_support_background_cap_pass": value_at(diagnostics.get("identity_consensus_support_background_cap_pass_mask"), i),
                    "identity_consensus_old_pass": value_at(diagnostics.get("identity_consensus_old_pass_mask"), i),
                    "identity_consensus_seen_new_pass": value_at(diagnostics.get("identity_consensus_seen_new_pass_mask"), i),
                    "identity_consensus_accept": value_at(diagnostics.get("identity_consensus_accept_mask"), i),
                    "identity_consensus_reject": value_at(diagnostics.get("identity_consensus_reject_mask"), i),
                    "identity_consensus_failure_count": value_at(diagnostics.get("identity_consensus_failure_count"), i),
                    "support_conformal_label": value_at(diagnostics.get("support_conformal_label"), i),
                    "support_conformal_score": value_at(diagnostics.get("support_conformal_score"), i),
                    "support_conformal_floor": value_at(diagnostics.get("support_conformal_floor"), i),
                    "support_conformal_margin": value_at(diagnostics.get("support_conformal_margin"), i),
                    "support_conformal_anchor_margin": value_at(diagnostics.get("support_conformal_anchor_margin"), i),
                    "support_conformal_anchor_margin_floor": value_at(diagnostics.get("support_conformal_anchor_margin_floor"), i),
                    "support_conformal_background_score": value_at(diagnostics.get("support_conformal_background_score"), i),
                    "support_conformal_background_margin": value_at(diagnostics.get("support_conformal_background_margin"), i),
                    "support_conformal_pass": value_at(diagnostics.get("support_conformal_pass_mask"), i),
                    "support_conformal_reject": value_at(diagnostics.get("support_conformal_reject_mask"), i),
                    "support_conformal_failure_count": value_at(diagnostics.get("support_conformal_failure_count"), i),
                    "support_reconstruction_label": value_at(diagnostics.get("support_reconstruction_label"), i),
                    "support_reconstruction_residual": value_at(diagnostics.get("support_reconstruction_residual"), i),
                    "support_reconstruction_residual_ceiling": value_at(diagnostics.get("support_reconstruction_residual_ceiling"), i),
                    "support_reconstruction_residual_margin": value_at(diagnostics.get("support_reconstruction_residual_margin"), i),
                    "support_reconstruction_center_cosine": value_at(diagnostics.get("support_reconstruction_center_cosine"), i),
                    "support_reconstruction_negative_score": value_at(diagnostics.get("support_reconstruction_negative_score"), i),
                    "support_reconstruction_negative_margin": value_at(diagnostics.get("support_reconstruction_negative_margin"), i),
                    "support_reconstruction_background_score": value_at(diagnostics.get("support_reconstruction_background_score"), i),
                    "support_reconstruction_background_margin": value_at(diagnostics.get("support_reconstruction_background_margin"), i),
                    "support_reconstruction_pass": value_at(diagnostics.get("support_reconstruction_pass_mask"), i),
                    "support_reconstruction_reject": value_at(diagnostics.get("support_reconstruction_reject_mask"), i),
                    "support_reconstruction_failure_count": value_at(diagnostics.get("support_reconstruction_failure_count"), i),
                    "source_looo_risk_label": value_at(diagnostics.get("source_looo_risk_label"), i),
                    "source_looo_risk_score": value_at(diagnostics.get("source_looo_risk_score"), i),
                    "source_looo_risk_floor": value_at(diagnostics.get("source_looo_risk_floor"), i),
                    "source_looo_risk_margin": value_at(diagnostics.get("source_looo_risk_margin"), i),
                    "source_looo_second_score": value_at(diagnostics.get("source_looo_second_score"), i),
                    "source_looo_score_margin": value_at(diagnostics.get("source_looo_score_margin"), i),
                    "source_looo_known_evidence_delta": value_at(diagnostics.get("source_looo_known_evidence_delta"), i),
                    "source_looo_background_score": value_at(diagnostics.get("source_looo_background_score"), i),
                    "source_looo_background_margin": value_at(diagnostics.get("source_looo_background_margin"), i),
                    "source_looo_pass": value_at(diagnostics.get("source_looo_pass_mask"), i),
                    "source_looo_reject": value_at(diagnostics.get("source_looo_reject_mask"), i),
                    "source_looo_failure_count": value_at(diagnostics.get("source_looo_failure_count"), i),
                    "pair_verifier_label": value_at(diagnostics.get("pair_verifier_label"), i),
                    "pair_verifier_prob": value_at(diagnostics.get("pair_verifier_prob"), i),
                    "pair_verifier_threshold": value_at(diagnostics.get("pair_verifier_threshold"), i),
                    "pair_verifier_called": value_at(diagnostics.get("pair_verifier_called_mask"), i),
                    "pair_verifier_veto": value_at(diagnostics.get("pair_verifier_veto_mask"), i),
                    "three_way_label": value_at(diagnostics.get("three_way_label"), i),
                    "three_way_old_score": value_at(diagnostics.get("three_way_old_score"), i),
                    "three_way_seen_new_score": value_at(diagnostics.get("three_way_seen_new_score"), i),
                    "three_way_background_score": value_at(diagnostics.get("three_way_background_score"), i),
                    "three_way_old_prob": value_at(diagnostics.get("three_way_old_prob"), i),
                    "three_way_seen_new_prob": value_at(diagnostics.get("three_way_seen_new_prob"), i),
                    "three_way_background_prob": value_at(diagnostics.get("three_way_background_prob"), i),
                    "three_way_old_known_prob": value_at(diagnostics.get("three_way_old_known_prob"), i),
                    "three_way_seen_new_known_prob": value_at(diagnostics.get("three_way_seen_new_known_prob"), i),
                    "three_way_known_prob_class_first": value_at(diagnostics.get("three_way_known_prob_class_first"), i),
                    "three_way_known_background_gap": value_at(diagnostics.get("three_way_known_background_gap"), i),
                    "three_way_background_margin": value_at(diagnostics.get("three_way_background_margin"), i),
                    "three_way_old_seen_gap": value_at(diagnostics.get("three_way_old_seen_gap"), i),
                    "three_way_background_available": value_at(diagnostics.get("three_way_background_available_mask"), i),
                    "three_way_base_reject": value_at(diagnostics.get("three_way_base_reject_mask"), i),
                    "three_way_known_floor": value_at(diagnostics.get("three_way_known_floor_mask"), i),
                    "three_way_old_floor": value_at(diagnostics.get("three_way_old_floor_mask"), i),
                    "three_way_seen_new_floor": value_at(diagnostics.get("three_way_seen_new_floor_mask"), i),
                    "three_way_extreme_background": value_at(diagnostics.get("three_way_extreme_background_mask"), i),
                    "three_way_floor_accept": value_at(diagnostics.get("three_way_floor_accept_mask"), i),
                    "three_way_floor_defer": value_at(diagnostics.get("three_way_floor_defer_mask"), i),
                    "three_way_class_first_known_evidence": value_at(diagnostics.get("three_way_class_first_known_evidence_mask"), i),
                    "three_way_evidence_balanced_known_evidence": value_at(diagnostics.get("three_way_evidence_balanced_known_evidence_mask"), i),
                    "three_way_reject_suppressed_by_floor": value_at(diagnostics.get("three_way_reject_suppressed_by_floor_mask"), i),
                    "three_way_accept": value_at(diagnostics.get("three_way_accept_mask"), i),
                    "three_way_reject": value_at(diagnostics.get("three_way_reject_mask"), i),
                    "three_way_defer": value_at(diagnostics.get("three_way_defer_mask"), i),
                    "pre_reject_arbitration_label": value_at(diagnostics.get("pre_reject_arbitration_label"), i),
                    "pre_reject_arbitration_score": value_at(diagnostics.get("pre_reject_arbitration_score"), i),
                    "pre_reject_arbitration_margin": value_at(diagnostics.get("pre_reject_arbitration_margin"), i),
                    "pre_reject_arbitration_evidence_delta": value_at(diagnostics.get("pre_reject_arbitration_evidence_delta"), i),
                    "pre_reject_arbitration_anchor_delta": value_at(diagnostics.get("pre_reject_arbitration_anchor_delta"), i),
                    "pre_reject_arbitration_anchor_margin": value_at(diagnostics.get("pre_reject_arbitration_anchor_margin"), i),
                    "pre_reject_arbitration_background_score": value_at(diagnostics.get("pre_reject_arbitration_background_score"), i),
                    "pre_reject_arbitration_background_margin": value_at(diagnostics.get("pre_reject_arbitration_background_margin"), i),
                    "pre_reject_arbitration_background_available": value_at(diagnostics.get("pre_reject_arbitration_background_available_mask"), i),
                    "pre_reject_arbitration_background_accept_ok": value_at(diagnostics.get("pre_reject_arbitration_background_accept_ok_mask"), i),
                    "pre_reject_arbitration_background_defer_risk": value_at(diagnostics.get("pre_reject_arbitration_background_defer_risk_mask"), i),
                    "pre_reject_arbitration_background_reject_risk": value_at(diagnostics.get("pre_reject_arbitration_background_reject_risk_mask"), i),
                    "pre_reject_arbitration_evidence_ok": value_at(diagnostics.get("pre_reject_arbitration_evidence_ok_mask"), i),
                    "pre_reject_arbitration_support_retention": value_at(diagnostics.get("pre_reject_arbitration_support_retention_mask"), i),
                    "pre_reject_arbitration_support_retention_source_looo_block": value_at(diagnostics.get("pre_reject_arbitration_support_retention_source_looo_block_mask"), i),
                    "pre_reject_arbitration_extreme_background": value_at(diagnostics.get("pre_reject_arbitration_extreme_background_mask"), i),
                    "pre_reject_arbitration_accept": value_at(diagnostics.get("pre_reject_arbitration_accept_mask"), i),
                    "pre_reject_arbitration_reject": value_at(diagnostics.get("pre_reject_arbitration_reject_mask"), i),
                    "pre_reject_arbitration_defer": value_at(diagnostics.get("pre_reject_arbitration_defer_mask"), i),
                    "pre_reject_arbitration_uncertain": value_at(diagnostics.get("pre_reject_arbitration_uncertain_mask"), i),
                    "retention_rescue_label": value_at(diagnostics.get("retention_rescue_label"), i),
                    "retention_rescue_score": value_at(diagnostics.get("retention_rescue_score"), i),
                    "retention_rescue_margin": value_at(diagnostics.get("retention_rescue_margin"), i),
                    "retention_rescue_evidence_delta": value_at(diagnostics.get("retention_rescue_evidence_delta"), i),
                    "retention_rescue_anchor_delta": value_at(diagnostics.get("retention_rescue_anchor_delta"), i),
                    "retention_rescue_anchor_margin": value_at(diagnostics.get("retention_rescue_anchor_margin"), i),
                    "retention_rescue_background_score": value_at(diagnostics.get("retention_rescue_background_score"), i),
                    "retention_rescue_background_margin": value_at(diagnostics.get("retention_rescue_background_margin"), i),
                    "retention_rescue_eligible": value_at(diagnostics.get("retention_rescue_eligible_mask"), i),
                    "retention_rescue_accept": value_at(diagnostics.get("retention_rescue_accept_mask"), i),
                    "two_branch_background_score": value_at(diagnostics.get("two_branch_background_score"), i),
                    "two_branch_known_score": value_at(diagnostics.get("two_branch_known_score"), i),
                    "two_branch_background_margin": value_at(diagnostics.get("two_branch_background_margin"), i),
                    "two_branch_background_risk": value_at(diagnostics.get("two_branch_background_risk_mask"), i),
                    "two_branch_support_override": value_at(diagnostics.get("two_branch_support_override_mask"), i),
                    "two_branch_background_reject": value_at(diagnostics.get("two_branch_background_reject_mask"), i),
                    "seen_new_override_label": value_at(diagnostics.get("seen_new_override_label"), i),
                    "seen_new_override_evidence_delta": value_at(diagnostics.get("seen_new_override_evidence_delta"), i),
                    "seen_new_override_anchor_delta": value_at(diagnostics.get("seen_new_override_anchor_delta"), i),
                    "seen_new_override_affinity_delta": value_at(diagnostics.get("seen_new_override_affinity_delta"), i),
                    "seen_new_override_residual_delta": value_at(diagnostics.get("seen_new_override_residual_delta"), i),
                    "seen_new_override_seen_minus_old_evidence": value_at(diagnostics.get("seen_new_override_seen_minus_old_evidence"), i),
                    "seen_new_override_seen_minus_old_score": value_at(diagnostics.get("seen_new_override_seen_minus_old_score"), i),
                    "seen_new_override_background_score": value_at(diagnostics.get("seen_new_override_background_score"), i),
                    "seen_new_override_background_margin": value_at(diagnostics.get("seen_new_override_background_margin"), i),
                    "seen_new_override_background_risk": value_at(diagnostics.get("seen_new_override_background_risk_mask"), i),
                    "seen_new_override_support_knn_seen_new_minus_old": value_at(
                        diagnostics.get("seen_new_override_support_knn_seen_new_minus_old"), i
                    ),
                    "seen_new_override_support_knn_margin": value_at(
                        diagnostics.get("seen_new_override_support_knn_margin"), i
                    ),
                    "seen_new_override_support_knn_pass": value_at(
                        diagnostics.get("seen_new_override_support_knn_pass_mask"), i
                    ),
                    "seen_new_registration_override": value_at(diagnostics.get("seen_new_registration_override_mask"), i),
                    "residual_delta": value_at(diagnostics.get("residual_delta"), i),
                    "mahalanobis_delta": value_at(diagnostics.get("mahalanobis_delta"), i),
                    "margin_delta": value_at(diagnostics.get("margin_delta"), i),
                    "energy_delta": value_at(diagnostics.get("energy_delta"), i),
                    "evt_delta": value_at(diagnostics.get("evt_delta"), i),
                    "min_accept_delta": value_at(diagnostics.get("min_accept_delta"), i),
                    "seen_new_evidence": value_at(seen_new_evidence, i),
                    "seen_new_support_affinity": value_at(seen_new_support_affinity, i),
                    "seen_new_support_residual": value_at(seen_new_support_residual, i),
                    "seen_new_evidence_delta": value_at(diagnostics.get("seen_new_evidence_delta"), i),
                    "seen_new_affinity_delta": value_at(diagnostics.get("seen_new_affinity_delta"), i),
                    "seen_new_residual_delta": value_at(diagnostics.get("seen_new_residual_delta"), i),
                    "seen_new_anchor_similarity": value_at(seen_new_anchor_similarity, i),
                    "seen_new_anchor_delta": value_at(seen_new_anchor_delta, i),
                    "decision": str(decisions[i]),
                    "gate_reason": str(gate_reasons[i]),
                }
            )


def _make_gate_config(args: argparse.Namespace) -> OpenSetGateConfig:
    return OpenSetGateConfig(
        mode=str(args.gate_mode),
        min_cosine=None if str(args.gate_mode) == "none" else float(args.unknown_threshold),
        min_margin=args.min_margin,
        max_mahalanobis=args.max_mahalanobis,
        openmax_tail_size=int(args.openmax_tail_size),
        openmax_quantile=float(args.openmax_quantile),
        openmax_min_threshold=float(args.openmax_min_threshold),
    )


def _synthetic_payload(shots: int) -> dict[str, np.ndarray]:
    shots = max(1, int(shots))
    source_features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    source_labels = np.array([0, 1], dtype=np.int64)
    support_features = np.repeat(np.array([[-1.0, 0.0]], dtype=np.float32), shots, axis=0)
    support_labels = np.repeat(np.array([2], dtype=np.int64), shots, axis=0)
    query_features = np.array([[0.95, 0.05], [-0.95, 0.02], [0.05, -0.95]], dtype=np.float32)
    query_labels = np.array([0, 2, -1], dtype=np.int64)
    return {
        "source_features": source_features,
        "source_labels": source_labels,
        "support_features": support_features,
        "support_labels": support_labels,
        "query_features": query_features,
        "query_labels": query_labels,
    }


def _load_payload(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    if args.dry_run_synthetic:
        return _synthetic_payload(args.shots), {
            "protocol": f"CVS-{str(args.protocol).upper()}",
            "payload_source": "dry_run_synthetic",
            "target_visibility": "synthetic_feature_smoke",
        }
    if args.feature_npz is None:
        raise ValueError("--feature_npz is required unless --dry_run_synthetic is set")
    with np.load(args.feature_npz, allow_pickle=True) as data:
        direct_required = ["source_features", "source_labels", "support_features", "support_labels", "query_features", "query_labels"]
        if all(name in data for name in direct_required):
            manifest = {
                "protocol": f"CVS-{str(args.protocol).upper()}",
                "payload_source": str(args.feature_npz),
                "payload_kind": "direct_protocol_arrays",
            }
            if "manifest_json" in data:
                try:
                    manifest.update(json.loads(str(data["manifest_json"].item())))
                except Exception:
                    manifest["embedded_manifest_decode_error"] = True
            arrays = {name: data[name] for name in direct_required}
            for name in (
                "source_rx_ids",
                "support_rx_ids",
                "query_rx_ids",
                "source_day_ids",
                "support_day_ids",
                "query_day_ids",
                "source_tx_ids",
                "support_tx_ids",
                "query_tx_ids",
            ):
                if name in data:
                    arrays[name] = data[name]
            return arrays, _promote_target_view_manifest(manifest)

        if args.protocol not in {"sfe", "source_open_set", "ftrc"}:
            missing = [name for name in direct_required if name not in data]
            raise KeyError(f"feature npz missing arrays for FTRC/direct mode: {missing}")

        features_key = str(args.features_key)
        tx_ids_key = str(args.tx_ids_key)
        if features_key not in data or tx_ids_key not in data:
            raise KeyError(
                f"SFE full-feature mode requires arrays {features_key!r} and {tx_ids_key!r}; "
                f"available={list(data.keys())}"
            )
        source_id_list = parse_tx_id_list(args.source_tx_ids)
        new_id_list = parse_tx_id_list(args.new_tx_ids)
        target_old_id_list = parse_tx_id_list(args.target_old_tx_ids)
        old_unknown_only = (not new_id_list) and bool(target_old_id_list) and int(args.shots) == 0
        if not source_id_list or (not new_id_list and not old_unknown_only):
            raise ValueError(
                "--source_tx_ids and --new_tx_ids are required for full-feature SFE NPZ; "
                "omit --new_tx_ids only for shots=0 target-old/unknown-only evaluation"
            )
        embedded_manifest = {}
        if "manifest_json" in data:
            try:
                embedded_manifest = json.loads(str(data["manifest_json"].item()))
            except Exception:
                embedded_manifest = {"embedded_manifest_decode_error": True}
        tx_id_values = {str(v) for v in np.asarray(data[tx_ids_key]).reshape(-1).astype(str).tolist()}
        source_tx_ids = args.source_tx_ids
        target_old_tx_ids = args.target_old_tx_ids
        new_tx_ids = args.new_tx_ids
        unknown_tx_ids = args.unknown_tx_ids
        if embedded_manifest:
            if any(tx not in tx_id_values for tx in parse_tx_id_list(source_tx_ids)):
                source_tx_ids = ",".join(parse_tx_id_list(embedded_manifest.get("source_tx_ids", [])))
            if parse_tx_id_list(target_old_tx_ids) and any(
                tx not in tx_id_values for tx in parse_tx_id_list(target_old_tx_ids)
            ):
                target_old_tx_ids = ",".join(parse_tx_id_list(embedded_manifest.get("target_old_tx_ids", [])))
            if any(tx not in tx_id_values for tx in parse_tx_id_list(new_tx_ids)):
                new_tx_ids = ",".join(parse_tx_id_list(embedded_manifest.get("new_tx_ids", [])))
            if parse_tx_id_list(unknown_tx_ids) and any(tx not in tx_id_values for tx in parse_tx_id_list(unknown_tx_ids)):
                unknown_tx_ids = ",".join(parse_tx_id_list(embedded_manifest.get("unknown_tx_ids", [])))
        payload = build_sfe_payload_from_feature_arrays(
            features=data[features_key],
            tx_ids=data[tx_ids_key],
            dataset_roles=data["dataset_role"] if "dataset_role" in data else None,
            sample_metadata={
                key: data[key]
                for key in ("rx_ids", "day_ids", "eq_ids", "sig_ids", "channel_views", "sat_scenarios")
                if key in data
            },
            source_tx_ids=source_tx_ids,
            target_old_tx_ids=target_old_tx_ids,
            new_tx_ids=new_tx_ids,
            unknown_tx_ids=unknown_tx_ids,
            shots=int(args.shots),
            source_proto_per_tx=int(args.source_proto_per_tx),
            source_query_per_tx=int(args.source_query_per_tx),
            target_old_support_per_tx=int(args.target_old_support_per_tx),
            target_old_query_per_tx=args.target_old_query_per_tx,
            query_per_tx=int(args.query_per_tx),
            seed=int(args.seed),
            extra_metadata={
                "payload_source": str(args.feature_npz),
                "payload_kind": "full_feature_npz",
                "features_key": features_key,
                "tx_ids_key": tx_ids_key,
                "embedded_manifest": embedded_manifest,
            },
        )
        return payload.arrays, _promote_target_view_manifest(payload.manifest)


def tensor_from_numpy_compatible(array: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    """Convert NumPy arrays without relying on torch.from_numpy.

    N607 currently has a torch/numpy pair where `torch.from_numpy` can fail with
    `expected np.ndarray (got numpy.ndarray)`. The list conversion is slower but
    robust for the small feature-level smoke payloads used by this CLI.
    """

    return torch.tensor(array.tolist(), dtype=dtype)


def _string_array(payload: dict[str, np.ndarray], key: str, expected: int | None = None) -> list[str]:
    if key not in payload:
        return []
    values = [str(v) for v in np.asarray(payload[key]).reshape(-1).tolist()]
    if expected is not None and len(values) != int(expected):
        raise ValueError(f"{key} must have {expected} values, got {len(values)}")
    return values


def _validate_stage2_receiver_domain(
    payload: dict[str, np.ndarray],
    support_labels: torch.Tensor,
    *,
    old_labels: set[int],
) -> dict:
    source_rx = set(_string_array(payload, "source_rx_ids"))
    support_rx = _string_array(payload, "support_rx_ids", expected=int(support_labels.numel()))
    query_rx = set(_string_array(payload, "query_rx_ids"))
    if not source_rx and not support_rx and not query_rx:
        return {"checked": False, "reason": "receiver_metadata_missing"}
    if not source_rx:
        return {"checked": False, "reason": "source_receiver_metadata_missing"}
    support_values = [int(v) for v in support_labels.detach().cpu().tolist()]
    old_support_rx = {rx for rx, label in zip(support_rx, support_values) if int(label) in old_labels}
    new_support_rx = {rx for rx, label in zip(support_rx, support_values) if int(label) != UNKNOWN_LABEL and int(label) not in old_labels}
    bad_old = old_support_rx & source_rx
    bad_new = new_support_rx & source_rx
    bad_query = query_rx & source_rx
    if bad_old or bad_new or bad_query:
        raise ValueError(
            "target receiver domain must be disjoint from CEN51/train receivers for old support, "
            f"new support, and query samples; overlaps old={sorted(bad_old)} new={sorted(bad_new)} query={sorted(bad_query)}"
        )
    return {
        "checked": True,
        "train_receivers": sorted(source_rx),
        "target_old_support_receivers": sorted(old_support_rx),
        "target_new_support_receivers": sorted(new_support_rx),
        "target_query_receivers": sorted(query_rx),
    }


def _enable_multiproto_score_head(
    class_states: dict[int, object],
    *,
    enabled: bool,
    topk: int,
    temperature: float,
    score_weight: float,
    consistency_gate: bool = False,
    consistency_min_cos: float = -1.0,
    consistency_max_residual: float = 1.0e6,
    consistency_min_margin: float = -1.0e6,
    consistency_action: str = "uncertain",
) -> dict:
    class_count = 0
    anchor_classes = 0
    if bool(enabled):
        for state in class_states.values():
            anchors = getattr(state, "support_anchors", None)
            has_anchors = anchors is not None and hasattr(anchors, "numel") and int(anchors.numel()) > 0
            if not has_anchors:
                continue
            state.thresholds["soft_mixture_score_enabled"] = True
            state.thresholds["soft_mixture_topk"] = int(topk)
            state.thresholds["soft_mixture_temperature"] = float(temperature)
            state.thresholds["soft_mixture_score_weight"] = float(score_weight)
            if bool(consistency_gate):
                state.thresholds["soft_mixture_consistency_gate_enabled"] = True
                state.thresholds["soft_mixture_min_cos"] = float(consistency_min_cos)
                state.thresholds["soft_mixture_max_residual"] = float(consistency_max_residual)
                state.thresholds["soft_mixture_min_margin"] = float(consistency_min_margin)
                state.thresholds["soft_mixture_consistency_action"] = str(consistency_action)
            class_count += 1
            anchor_classes += 1
    return {
        "enabled": bool(enabled),
        "class_count": int(class_count),
        "anchor_class_count": int(anchor_classes),
        "topk": int(topk),
        "temperature": float(temperature),
        "score_weight": float(score_weight),
        "consistency_gate": bool(consistency_gate),
        "consistency_min_cos": float(consistency_min_cos),
        "consistency_max_residual": float(consistency_max_residual),
        "consistency_min_margin": float(consistency_min_margin),
        "consistency_action": str(consistency_action),
        "source": "same_class_source_or_target_prototype_plus_allowed_support_anchors",
        "unknown_query_threshold_calibration": False,
    }


def _run_oa_mse_protocol(
    *,
    protocol: str,
    source: object,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gate_config: OpenSetGateConfig,
    source_adapter_features: torch.Tensor | None = None,
    source_adapter_labels: torch.Tensor | None = None,
    old_acc_target: float = 0.90,
    seen_new_acc_target: float = 0.75,
    adapter_rank: int = 2,
    adapter_kind: str = "low_rank",
    adapter_steps: int = 40,
    adapter_lr: float = 0.05,
    source_anchor_weight: float = 0.05,
    source_ce_weight: float = 0.10,
    unknown_moat_weight: float = 0.10,
    unknown_moat_margin: float = 0.45,
    pseudo_unknown_samples_per_pair: int = 2,
    pseudo_unknown_offset_scale: float = 0.15,
    pseudo_unknown_source_boundary_samples_per_pair: int = 0,
    pseudo_unknown_source_boundary_offset_scale: float = 0.20,
    pseudo_unknown_target_shift_samples_per_class: int = 0,
    pseudo_unknown_target_shift_offset_scale: float = 0.20,
    pseudo_unknown_target_halo_samples_per_class: int = 0,
    pseudo_unknown_target_halo_offset_scale: float = 0.35,
    pseudo_unknown_target_ring_samples_per_class: int = 0,
    pseudo_unknown_target_ring_offset_scale: float = 0.45,
    old_bridge_weight: float = 0.10,
    old_bridge_samples_per_class: int = 2,
    old_bridge_max_mix: float = 0.85,
    support_contrast_weight: float = 0.0,
    support_contrast_negative_margin: float = 0.78,
    support_contrast_positive_margin: float = 0.88,
    support_center_ce_weight: float = 0.0,
    support_center_temperature: float = 0.10,
    support_center_margin: float = 0.10,
    soft_proto_weight: float = 0.0,
    soft_proto_topk: int = 2,
    soft_proto_temperature: float = 0.10,
    soft_proto_boundary_weight: float = 0.0,
    soft_proto_boundary_margin: float = 0.15,
    multiproto_score: bool = False,
    multiproto_topk: int = 2,
    multiproto_temperature: float = 0.10,
    multiproto_score_weight: float = 1.0,
    mixture_consistency_gate: bool = False,
    mixture_consistency_min_cos: float = -1.0,
    mixture_consistency_max_residual: float = 1.0e6,
    mixture_consistency_min_margin: float = -1.0e6,
    mixture_consistency_action: str = "uncertain",
    anchor_density_gate: bool = False,
    anchor_density_topk: int = 3,
    anchor_density_temperature: float = 0.08,
    anchor_density_min_quantile: float = 0.05,
    anchor_density_margin_quantile: float = 0.05,
    anchor_density_gate_action: str = "uncertain",
    class_envelope_gate: bool = False,
    class_envelope_evidence_quantile: float = 0.05,
    class_envelope_residual_quantile: float = 0.95,
    class_envelope_score_quantile: float = 0.05,
    class_envelope_margin_quantile: float = 0.05,
    class_envelope_evidence_slack: float = 0.02,
    class_envelope_residual_slack: float = 0.02,
    class_envelope_score_slack: float = 0.05,
    class_envelope_margin_slack: float = 0.02,
    class_envelope_min_failures: int = 1,
    class_envelope_gate_action: str = "reject",
    old_primary_gate: bool = False,
    old_primary_min_old_support_evidence_delta: float = 0.0,
    old_primary_min_old_support_anchor_delta: float = -0.02,
    old_primary_min_old_support_anchor_margin: float = 0.0,
    old_primary_min_score_margin: float = 0.0,
    old_primary_require_soft_mixture: bool = False,
    old_primary_min_soft_mixture_margin: float = -1.0e6,
    old_primary_min_soft_mixture_cos: float = -1.0,
    old_primary_max_soft_mixture_residual: float = 1.0e6,
    old_primary_require_support_knn: bool = False,
    old_primary_require_support_knn_label_match: bool = True,
    old_primary_min_support_knn_margin: float = 0.0,
    old_primary_max_support_knn_seen_new_minus_old: float | None = None,
    old_primary_min_old_drift_cos: float = -1.0,
    old_primary_max_old_drift_dist: float = 1.0e6,
    old_primary_require_class_envelope: bool = False,
    old_primary_unknown_veto_background_score: float = 0.86,
    old_primary_unknown_veto_background_margin: float = 0.10,
    old_primary_unknown_veto_min_sources: int = 1,
    old_primary_fail_action: str = "defer",
    old_primary_unknown_veto_action: str = "reject",
    old_primary_promote_rescue_candidates: bool = False,
    density_shell_gate: bool = False,
    density_shell_old_min_evidence_delta: float = -0.04,
    density_shell_old_min_anchor_delta: float = -0.08,
    density_shell_old_min_density_delta: float = -0.06,
    density_shell_seen_new_min_evidence_delta: float = -0.04,
    density_shell_seen_new_min_anchor_delta: float = -0.08,
    density_shell_seen_new_min_density_delta: float = -0.06,
    density_shell_accept_background_margin: float = 0.18,
    density_shell_reject_background_score: float = 0.86,
    density_shell_reject_background_margin: float = 0.14,
    density_shell_reject_min_failed_shells: int = 2,
    identity_consensus_arbitration: bool = False,
    identity_consensus_old_min_evidence_delta: float = -0.06,
    identity_consensus_old_min_anchor_delta: float = -0.10,
    identity_consensus_old_min_density_delta: float = -0.08,
    identity_consensus_seen_new_min_evidence_delta: float = -0.04,
    identity_consensus_seen_new_min_anchor_delta: float = -0.08,
    identity_consensus_seen_new_min_density_delta: float = -0.06,
    identity_consensus_min_identity_margin: float = -0.05,
    identity_consensus_background_accept_margin: float = 0.22,
    identity_consensus_reject_background_score: float = 0.90,
    identity_consensus_reject_background_margin: float = 0.18,
    identity_consensus_reject_min_identity_failures: int = 4,
    identity_consensus_support_background_cap: bool = False,
    identity_consensus_support_background_cap_quantile: float = 0.90,
    identity_consensus_support_background_cap_slack: float = 0.05,
    identity_consensus_support_background_cap_min_anchors: int = 2,
    support_conformal_arbitration: bool = False,
    support_conformal_calibration_quantile: float = 0.05,
    support_conformal_conformity_slack: float = 0.12,
    support_conformal_anchor_margin_slack: float = 0.06,
    support_conformal_background_score: float = 0.82,
    support_conformal_background_margin: float = 0.08,
    support_conformal_hard_reject_margin: float = 0.18,
    support_conformal_reject_min_failures: int = 2,
    support_conformal_reject_action: str = "reject",
    support_reconstruction_arbitration: bool = False,
    support_reconstruction_rank: int = 2,
    support_reconstruction_residual_quantile: float = 0.95,
    support_reconstruction_residual_slack: float = 0.04,
    support_reconstruction_min_residual_floor: float = 0.03,
    support_reconstruction_negative_scale: float = 0.55,
    support_reconstruction_negative_margin: float = -0.02,
    support_reconstruction_hard_residual_margin: float = 0.08,
    support_reconstruction_background_score: float = 0.86,
    support_reconstruction_background_margin: float = 0.12,
    support_reconstruction_reject_min_failures: int = 2,
    support_reconstruction_reject_action: str = "reject",
    pre_reject_defer_arbitration: bool = False,
    three_way_decision_head: bool = False,
    three_way_head_weight: float = 0.0,
    three_way_head_temperature: float = 0.10,
    three_way_head_known_margin: float = 0.08,
    three_way_head_background_margin: float = 0.08,
    three_way_head_support_ce_weight: float = 1.0,
    three_way_head_pseudo_ce_weight: float = 0.35,
    three_way_head_support_background_margin_weight: float = 1.0,
    three_way_head_pseudo_margin_weight: float = 0.50,
    three_way_accept_prob: float = 0.50,
    three_way_reject_prob: float = 0.55,
    three_way_defer_prob: float = 0.45,
    three_way_known_background_margin: float = 0.02,
    three_way_reject_margin: float = 0.04,
    three_way_old_seen_ambiguity_margin: float = 0.04,
    three_way_defer_action: str = "uncertain",
    three_way_decision_policy: str = "background_competition",
    three_way_known_floor: bool = False,
    three_way_known_floor_action: str = "defer",
    three_way_known_floor_old_min_evidence_delta: float = -0.04,
    three_way_known_floor_old_min_anchor_delta: float = -0.08,
    three_way_known_floor_old_min_anchor_margin: float = -0.04,
    three_way_known_floor_old_min_score_margin: float = -0.12,
    three_way_known_floor_seen_new_min_evidence_delta: float = -0.04,
    three_way_known_floor_seen_new_min_anchor_delta: float = -0.08,
    three_way_known_floor_seen_new_min_score_margin: float = -0.12,
    three_way_known_floor_background_override_prob: float = 0.995,
    three_way_known_floor_background_override_margin: float = 1.0,
    pre_reject_old_min_evidence_delta: float = 0.0,
    pre_reject_old_min_anchor_delta: float = -0.02,
    pre_reject_old_min_anchor_margin: float = 0.0,
    pre_reject_old_min_score_margin: float = -0.02,
    pre_reject_seen_new_min_evidence_delta: float = 0.0,
    pre_reject_seen_new_min_anchor_delta: float = 0.0,
    pre_reject_seen_new_min_score_margin: float = -0.05,
    pre_reject_max_background_score: float = 0.74,
    pre_reject_max_background_margin: float = 0.10,
    pre_reject_defer_background_score: float = 0.70,
    pre_reject_defer_background_margin: float = 0.04,
    pre_reject_reject_background_score: float = 0.82,
    pre_reject_reject_background_margin: float = 0.12,
    pre_reject_defer_action: str = "uncertain",
    pre_reject_support_neighborhood_retention: bool = False,
    pre_reject_support_retention_old_min_evidence_delta: float = 0.02,
    pre_reject_support_retention_old_min_anchor_delta: float = -0.04,
    pre_reject_support_retention_old_min_anchor_margin: float = -0.02,
    pre_reject_support_retention_old_min_score_margin: float = -0.04,
    pre_reject_support_retention_seen_new_min_evidence_delta: float = 0.02,
    pre_reject_support_retention_seen_new_min_anchor_delta: float = -0.04,
    pre_reject_support_retention_seen_new_min_score_margin: float = -0.08,
    pre_reject_support_retention_max_background_score: float = 0.96,
    pre_reject_support_retention_max_background_margin: float = 0.30,
    pre_reject_support_retention_require_source_looo_pass: bool = False,
    pre_reject_support_retention_source_looo_max_failures: int = 0,
    retention_rescue_gate: bool = False,
    retention_rescue_old_min_evidence_delta: float = 0.02,
    retention_rescue_old_min_anchor_delta: float = -0.01,
    retention_rescue_old_min_anchor_margin: float = 0.0,
    retention_rescue_old_min_score_margin: float = 0.0,
    retention_rescue_seen_new_min_evidence_delta: float = 0.02,
    retention_rescue_seen_new_min_anchor_delta: float = 0.0,
    retention_rescue_seen_new_min_score_margin: float = -0.02,
    retention_rescue_max_background_score: float = 0.70,
    retention_rescue_max_background_margin: float = 0.06,
    retention_rescue_candidate_only: bool = False,
    void_background_weight: float = 0.0,
    negative_anchor_weight: float = 0.0,
    negative_anchor_margin: float = 0.12,
    negative_anchor_temperature: float = 0.10,
    negative_anchor_max_anchors: int = 256,
    void_gate: bool = False,
    void_gate_min_score: float = 0.55,
    void_gate_min_margin: float = 0.05,
    old_neighborhood_weight: float = 0.10,
    old_neighborhood_samples_per_class: int = 2,
    old_neighborhood_radius: float = 0.06,
    old_surrogate_margin_weight: float = 0.05,
    old_surrogate_margin: float = 0.10,
    source_looo_unknown_weight: float = 0.0,
    source_looo_unknown_margin: float = 0.35,
    source_looo_interclass_margin: float = 0.08,
    source_looo_max_samples_per_class: int = 24,
    source_looo_risk_arbitration: bool = False,
    source_looo_risk_quantile: float = 0.85,
    source_looo_risk_slack: float = 0.0,
    source_looo_risk_min_score_margin: float = 0.02,
    source_looo_risk_min_known_evidence_delta: float = -0.08,
    source_looo_risk_background_score: float = 0.86,
    source_looo_risk_background_margin: float = 0.10,
    source_looo_risk_reject_min_failures: int = 2,
    source_looo_risk_reject_action: str = "reject",
    known_coverage_weight: float = 0.0,
    known_coverage_margin: float = 0.12,
    known_coverage_min_affinity: float = 0.35,
    known_coverage_max_samples: int = 256,
    old_surrogate_evidence_margin: float = 0.0,
    old_surrogate_reject_relax: float = 0.0,
    siamese_quantile: float = 0.10,
    siamese_accept_threshold: float = 0.50,
    siamese_unknown_veto: bool = False,
    siamese_unknown_veto_mode: str = "any",
    siamese_min_old_support_evidence_delta: float | None = None,
    siamese_min_old_surrogate_reject_delta: float | None = None,
    siamese_min_energy_delta: float | None = None,
    siamese_min_mahalanobis_delta: float | None = None,
    siamese_min_accept_delta: float | None = None,
    siamese_min_old_support_anchor_margin: float | None = None,
    siamese_min_veto_failures: int = 1,
    old_unknown_acceptance_guard: bool = False,
    old_unknown_guard_min_old_support_evidence_delta: float | None = None,
    old_unknown_guard_min_old_surrogate_reject_delta: float | None = None,
    old_unknown_guard_min_energy_delta: float | None = None,
    old_unknown_guard_min_mahalanobis_delta: float | None = None,
    old_unknown_guard_min_accept_delta: float | None = None,
    old_unknown_guard_min_old_support_anchor_margin: float | None = None,
    old_unknown_guard_min_best_old_score: float | None = None,
    old_unknown_guard_min_margin: float | None = None,
    old_unknown_guard_min_failures: int = 1,
    old80_head_mode: str = "disabled",
    old80_head_apply_policy: str = "replace_all",
    old80_head_fusion_rho: float = 0.75,
    old80_head_knn_k: int = 3,
    old_anchor_override_min_quality: float = 0.55,
    old_retention_quantile: float = 0.95,
    support_retention_guard: bool = False,
    support_retention_guard_quantile: float = 0.05,
    support_retention_guard_slack: float = 0.02,
    two_branch_background_guard: bool = False,
    two_branch_bg_min_score: float = 0.62,
    two_branch_bg_min_margin: float = -0.02,
    two_branch_old_support_evidence_delta: float = 0.0,
    two_branch_old_anchor_delta: float = -0.02,
    two_branch_old_anchor_margin: float = 0.0,
    two_branch_seen_new_evidence_delta: float = 0.0,
    two_branch_seen_new_anchor_delta: float = 0.0,
    seen_new_registration_override: bool = False,
    seen_new_override_min_evidence_delta: float = 0.0,
    seen_new_override_min_anchor_delta: float = 0.0,
    seen_new_override_min_affinity_delta: float = -0.02,
    seen_new_override_min_residual_delta: float = -0.02,
    seen_new_override_min_score_margin: float = -0.10,
    seen_new_override_min_seen_vs_old_evidence_margin: float = 0.02,
    seen_new_override_max_background_score: float = 0.72,
    seen_new_override_max_background_margin: float = 0.08,
    seen_new_override_min_support_knn_seen_new_minus_old: float | None = None,
    seen_new_override_min_support_knn_margin: float | None = None,
    adapter_selection_policy: str = "final",
    adapter_alpha_eval_sweep: bool = False,
) -> AdaptationResult:
    old_labels = source.label_values()
    if int(support_features.shape[0]) != int(support_labels.numel()):
        raise ValueError("OA-MSE support features and labels must have the same sample count")
    support_label_values = [int(v) for v in support_labels.detach().cpu().tolist()]
    old_support_mask = torch.tensor([v in old_labels for v in support_label_values], dtype=torch.bool, device=support_labels.device)
    unknown_support_mask = torch.tensor([v == -1 for v in support_label_values], dtype=torch.bool, device=support_labels.device)
    new_support_mask = torch.tensor(
        [v != -1 and v not in old_labels for v in support_label_values],
        dtype=torch.bool,
        device=support_labels.device,
    )
    if bool(unknown_support_mask.any().item()):
        raise ValueError("OA-MSE support must not include unknown query labels; unknown query is eval-only")

    support_for_head = support_features
    query_for_head = query_features
    source_for_head = source
    source_risk_features = source.vectors
    source_risk_labels = source.labels
    trained_adapter = None
    onboard_telemetry: dict = {
        "compute_profile": (
            "feature_level_residual_mlp_adapter_no_backbone_update"
            if str(adapter_kind).lower() == "residual_mlp"
            else "feature_level_low_rank_adapter_no_backbone_update"
        ),
        "adapter_kind": str(adapter_kind),
        "old_acc_target": float(old_acc_target),
        "seen_new_acc_target": float(seen_new_acc_target),
        "weibull_evt_required": True,
    }
    if protocol in {"ftrc", "sfe"} and support_features.numel() > 0:
        adapter, adapter_telemetry = fit_low_compute_target_adapter(
            source,
            support_features,
            support_labels,
            source_adapter_features=source_adapter_features,
            source_adapter_labels=source_adapter_labels,
            source_boundary_pseudo_unknown_samples_per_pair=int(pseudo_unknown_source_boundary_samples_per_pair),
            source_boundary_pseudo_unknown_offset_scale=float(pseudo_unknown_source_boundary_offset_scale),
            rank=int(adapter_rank),
            adapter_kind=str(adapter_kind),
            steps=int(adapter_steps),
            lr=float(adapter_lr),
            source_anchor_weight=float(source_anchor_weight),
            source_ce_weight=float(source_ce_weight),
            unknown_moat_weight=float(unknown_moat_weight),
            unknown_moat_margin=float(unknown_moat_margin),
            pseudo_unknown_samples_per_pair=int(pseudo_unknown_samples_per_pair),
            pseudo_unknown_offset_scale=float(pseudo_unknown_offset_scale),
            pseudo_unknown_target_shift_samples_per_class=int(pseudo_unknown_target_shift_samples_per_class),
            pseudo_unknown_target_shift_offset_scale=float(pseudo_unknown_target_shift_offset_scale),
            pseudo_unknown_target_halo_samples_per_class=int(pseudo_unknown_target_halo_samples_per_class),
            pseudo_unknown_target_halo_offset_scale=float(pseudo_unknown_target_halo_offset_scale),
            pseudo_unknown_target_ring_samples_per_class=int(pseudo_unknown_target_ring_samples_per_class),
            pseudo_unknown_target_ring_offset_scale=float(pseudo_unknown_target_ring_offset_scale),
            old_bridge_weight=float(old_bridge_weight),
            old_bridge_samples_per_class=int(old_bridge_samples_per_class),
            old_bridge_max_mix=float(old_bridge_max_mix),
            support_contrast_weight=float(support_contrast_weight),
            support_contrast_negative_margin=float(support_contrast_negative_margin),
            support_contrast_positive_margin=float(support_contrast_positive_margin),
            support_center_ce_weight=float(support_center_ce_weight),
            support_center_temperature=float(support_center_temperature),
            support_center_margin=float(support_center_margin),
            soft_proto_weight=float(soft_proto_weight),
            soft_proto_topk=int(soft_proto_topk),
            soft_proto_temperature=float(soft_proto_temperature),
            soft_proto_boundary_weight=float(soft_proto_boundary_weight),
            soft_proto_boundary_margin=float(soft_proto_boundary_margin),
            void_background_weight=float(void_background_weight),
            negative_anchor_weight=float(negative_anchor_weight),
            negative_anchor_margin=float(negative_anchor_margin),
            negative_anchor_temperature=float(negative_anchor_temperature),
            negative_anchor_max_anchors=int(negative_anchor_max_anchors),
            three_way_head_weight=float(three_way_head_weight),
            three_way_head_temperature=float(three_way_head_temperature),
            three_way_head_known_margin=float(three_way_head_known_margin),
            three_way_head_background_margin=float(three_way_head_background_margin),
            three_way_head_support_ce_weight=float(three_way_head_support_ce_weight),
            three_way_head_pseudo_ce_weight=float(three_way_head_pseudo_ce_weight),
            three_way_head_support_background_margin_weight=float(three_way_head_support_background_margin_weight),
            three_way_head_pseudo_margin_weight=float(three_way_head_pseudo_margin_weight),
            old_neighborhood_weight=float(old_neighborhood_weight),
            old_neighborhood_samples_per_class=int(old_neighborhood_samples_per_class),
            old_neighborhood_radius=float(old_neighborhood_radius),
            old_surrogate_margin_weight=float(old_surrogate_margin_weight),
            old_surrogate_margin=float(old_surrogate_margin),
            source_looo_unknown_weight=float(source_looo_unknown_weight),
            source_looo_unknown_margin=float(source_looo_unknown_margin),
            source_looo_interclass_margin=float(source_looo_interclass_margin),
            source_looo_max_samples_per_class=int(source_looo_max_samples_per_class),
            known_coverage_weight=float(known_coverage_weight),
            known_coverage_margin=float(known_coverage_margin),
            known_coverage_min_affinity=float(known_coverage_min_affinity),
            known_coverage_max_samples=int(known_coverage_max_samples),
            adapter_selection_policy=str(adapter_selection_policy),
            old_acc_target=float(old_acc_target),
            seen_new_acc_target=float(seen_new_acc_target),
        )
        trained_adapter = adapter
        support_for_head = adapter(support_features)
        query_for_head = adapter(query_features)
        if source_adapter_features is not None and source_adapter_labels is not None and source_adapter_features.numel() > 0:
            source_risk_features = adapter(source_adapter_features)
            source_risk_labels = source_adapter_labels.detach().clone()
        else:
            source_risk_features = adapter(source.vectors)
            source_risk_labels = source.labels
        source_for_head = type(source)(
            labels=source.labels.clone(),
            vectors=adapter(source.vectors),
            counts=source.counts.clone(),
            metadata=dict(source.metadata),
        )
        onboard_telemetry["target_adapter"] = {
            "enabled": True,
            "adapter_kind": str(adapter_telemetry.get("adapter_kind", adapter_kind)),
            "rank": int(adapter_telemetry["rank"]),
            "steps": int(adapter_telemetry["steps"]),
            "trainable_parameters": int(adapter_telemetry["trainable_parameters"]),
            "selected_alpha": adapter_telemetry.get("selected_alpha"),
            "adapter_selection_policy": adapter_telemetry.get("adapter_selection_policy"),
            "adapter_selection": adapter_telemetry.get("adapter_selection"),
            "training_scope": str(adapter_telemetry["training_scope"]),
            "support_old_acc": adapter_telemetry["support_old_acc"],
            "support_seen_new_acc": adapter_telemetry["support_seen_new_acc"],
            "loss_profile": adapter_telemetry.get("loss_profile"),
            "source_anchor_weight": adapter_telemetry.get("source_anchor_weight"),
            "source_ce_weight": adapter_telemetry.get("source_ce_weight"),
            "unknown_moat_weight": adapter_telemetry.get("unknown_moat_weight"),
            "unknown_moat_margin": adapter_telemetry.get("unknown_moat_margin"),
            "pseudo_unknown_count": adapter_telemetry.get("pseudo_unknown_count"),
            "pseudo_unknown_geometry_count": adapter_telemetry.get("pseudo_unknown_geometry_count"),
            "source_adapter_feature_count": adapter_telemetry.get("source_adapter_feature_count"),
            "source_adapter_label_count": adapter_telemetry.get("source_adapter_label_count"),
            "source_boundary_pseudo_unknown_samples_per_pair": adapter_telemetry.get("source_boundary_pseudo_unknown_samples_per_pair"),
            "source_boundary_pseudo_unknown_offset_scale": adapter_telemetry.get("source_boundary_pseudo_unknown_offset_scale"),
            "pseudo_unknown_source_boundary_count": adapter_telemetry.get("pseudo_unknown_source_boundary_count"),
            "pseudo_unknown_target_shift_count": adapter_telemetry.get("pseudo_unknown_target_shift_count"),
            "pseudo_unknown_target_shift_samples_per_class": adapter_telemetry.get("pseudo_unknown_target_shift_samples_per_class"),
            "pseudo_unknown_target_shift_offset_scale": adapter_telemetry.get("pseudo_unknown_target_shift_offset_scale"),
            "pseudo_unknown_target_halo_count": adapter_telemetry.get("pseudo_unknown_target_halo_count"),
            "pseudo_unknown_target_halo_samples_per_class": adapter_telemetry.get("pseudo_unknown_target_halo_samples_per_class"),
            "pseudo_unknown_target_halo_offset_scale": adapter_telemetry.get("pseudo_unknown_target_halo_offset_scale"),
            "pseudo_unknown_target_ring_count": adapter_telemetry.get("pseudo_unknown_target_ring_count"),
            "pseudo_unknown_target_ring_samples_per_class": adapter_telemetry.get("pseudo_unknown_target_ring_samples_per_class"),
            "pseudo_unknown_target_ring_offset_scale": adapter_telemetry.get("pseudo_unknown_target_ring_offset_scale"),
            "old_bridge_weight": adapter_telemetry.get("old_bridge_weight"),
            "old_bridge_count": adapter_telemetry.get("old_bridge_count"),
            "old_bridge_samples_per_class": adapter_telemetry.get("old_bridge_samples_per_class"),
            "old_bridge_max_mix": adapter_telemetry.get("old_bridge_max_mix"),
            "support_contrast_weight": adapter_telemetry.get("support_contrast_weight"),
            "support_contrast_negative_margin": adapter_telemetry.get("support_contrast_negative_margin"),
            "support_contrast_positive_margin": adapter_telemetry.get("support_contrast_positive_margin"),
            "support_contrast_anchor_count": adapter_telemetry.get("support_contrast_anchor_count"),
            "support_center_ce_weight": adapter_telemetry.get("support_center_ce_weight"),
            "support_center_temperature": adapter_telemetry.get("support_center_temperature"),
            "support_center_margin": adapter_telemetry.get("support_center_margin"),
            "support_center_class_count": adapter_telemetry.get("support_center_class_count"),
            "soft_proto_weight": adapter_telemetry.get("soft_proto_weight"),
            "soft_proto_topk": adapter_telemetry.get("soft_proto_topk"),
            "soft_proto_temperature": adapter_telemetry.get("soft_proto_temperature"),
            "soft_proto_boundary_weight": adapter_telemetry.get("soft_proto_boundary_weight"),
            "soft_proto_boundary_margin": adapter_telemetry.get("soft_proto_boundary_margin"),
            "soft_proto_anchor_count": adapter_telemetry.get("soft_proto_anchor_count"),
            "soft_proto_train_count": adapter_telemetry.get("soft_proto_train_count"),
            "void_background_weight": adapter_telemetry.get("void_background_weight"),
            "negative_anchor_weight": adapter_telemetry.get("negative_anchor_weight"),
            "negative_anchor_margin": adapter_telemetry.get("negative_anchor_margin"),
            "negative_anchor_temperature": adapter_telemetry.get("negative_anchor_temperature"),
            "negative_anchor_count": adapter_telemetry.get("negative_anchor_count"),
            "three_way_head_weight": adapter_telemetry.get("three_way_head_weight"),
            "three_way_head_temperature": adapter_telemetry.get("three_way_head_temperature"),
            "three_way_head_known_margin": adapter_telemetry.get("three_way_head_known_margin"),
            "three_way_head_background_margin": adapter_telemetry.get("three_way_head_background_margin"),
            "three_way_head_support_ce_weight": adapter_telemetry.get("three_way_head_support_ce_weight"),
            "three_way_head_pseudo_ce_weight": adapter_telemetry.get("three_way_head_pseudo_ce_weight"),
            "three_way_head_support_background_margin_weight": adapter_telemetry.get("three_way_head_support_background_margin_weight"),
            "three_way_head_pseudo_margin_weight": adapter_telemetry.get("three_way_head_pseudo_margin_weight"),
            "three_way_head_count": adapter_telemetry.get("three_way_head_count"),
            "old_neighborhood_weight": adapter_telemetry.get("old_neighborhood_weight"),
            "old_neighborhood_count": adapter_telemetry.get("old_neighborhood_count"),
            "old_neighborhood_radius": adapter_telemetry.get("old_neighborhood_radius"),
            "old_surrogate_margin_weight": adapter_telemetry.get("old_surrogate_margin_weight"),
            "old_surrogate_margin": adapter_telemetry.get("old_surrogate_margin"),
            "old_surrogate_margin_count": adapter_telemetry.get("old_surrogate_margin_count"),
            "source_looo_unknown_weight": adapter_telemetry.get("source_looo_unknown_weight"),
            "source_looo_unknown_margin": adapter_telemetry.get("source_looo_unknown_margin"),
            "source_looo_interclass_margin": adapter_telemetry.get("source_looo_interclass_margin"),
            "source_looo_max_samples_per_class": adapter_telemetry.get("source_looo_max_samples_per_class"),
            "source_looo_sample_count": adapter_telemetry.get("source_looo_sample_count"),
            "known_coverage_weight": adapter_telemetry.get("known_coverage_weight"),
            "known_coverage_margin": adapter_telemetry.get("known_coverage_margin"),
            "known_coverage_min_affinity": adapter_telemetry.get("known_coverage_min_affinity"),
            "known_coverage_max_samples": adapter_telemetry.get("known_coverage_max_samples"),
            "known_coverage_count": adapter_telemetry.get("known_coverage_count"),
            "old_surrogate_reject_relax": float(old_surrogate_reject_relax),
            "loss_terms": adapter_telemetry.get("loss_terms", {}),
            "loss_trace_schema": adapter_telemetry.get("loss_trace_schema"),
            "loss_initial": adapter_telemetry.get("loss_initial"),
            "loss_final": adapter_telemetry.get("loss_final"),
            "loss_trace": adapter_telemetry.get("loss_trace", []),
        }
    else:
        onboard_telemetry["target_adapter"] = {"enabled": False, "reason": "no_target_support_for_stage"}

    if protocol == "source_open_set":
        if support_features.numel() > 0 or support_labels.numel() > 0:
            raise ValueError("Stage2-A OA-MSE source_open_set must not receive support samples")
        class_states, u_orbit = register_old_classes(
            source_for_head,
            torch.empty((0, source_for_head.vectors.shape[1]), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            stage="Stage2-A",
            old_anchor_override_min_quality=float(old_anchor_override_min_quality),
            gate_config=gate_config,
        )
        prototype_set = source_for_head
        new_labels = set()
    elif protocol == "ftrc":
        if bool((~old_support_mask).any().item()):
            raise ValueError("Stage2-B OA-MSE support must contain old classes only; target-new support is forbidden")
        class_states, u_orbit = register_old_classes(
            source_for_head,
            support_for_head,
            support_labels,
            stage="Stage2-B",
            old_anchor_override_min_quality=float(old_anchor_override_min_quality),
            gate_config=gate_config,
        )
        prototype_set = source_for_head
        new_labels = set()
    else:
        class_states, u_orbit = register_old_classes(
            source_for_head,
            support_for_head[old_support_mask] if bool(old_support_mask.any().item()) else torch.empty((0, source_for_head.vectors.shape[1])),
            support_labels[old_support_mask] if bool(old_support_mask.any().item()) else torch.empty((0,), dtype=torch.long),
            stage="Stage2-C",
            old_anchor_override_min_quality=float(old_anchor_override_min_quality),
            gate_config=gate_config,
        )
        new_states = {}
        if bool(new_support_mask.any().item()):
            new_states = register_new_classes(
                support_for_head[new_support_mask],
                support_labels[new_support_mask],
                class_states,
                u_orbit,
                stage="Stage2-C",
                gate_config=gate_config,
            )
            class_states.update(new_states)
        prototype_set = source_for_head if not new_states else _prototype_set_from_states_for_eval(class_states)
        new_labels = {int(v) for v in new_states}

    pseudo_unknown = (
        generate_pseudo_unknown_features(
            class_states,
            samples_per_pair=int(pseudo_unknown_samples_per_pair),
            offset_scale=float(pseudo_unknown_offset_scale),
            target_shift_samples_per_class=int(pseudo_unknown_target_shift_samples_per_class),
            target_shift_offset_scale=float(pseudo_unknown_target_shift_offset_scale),
            target_halo_samples_per_class=int(pseudo_unknown_target_halo_samples_per_class),
            target_halo_offset_scale=float(pseudo_unknown_target_halo_offset_scale),
            target_ring_samples_per_class=int(pseudo_unknown_target_ring_samples_per_class),
            target_ring_offset_scale=float(pseudo_unknown_target_ring_offset_scale),
        )
        if len(class_states) >= 2
        else None
    )
    pseudo_unknown_geometry_count = 0
    pseudo_unknown_target_shift_count = 0
    pseudo_unknown_target_halo_count = 0
    pseudo_unknown_target_ring_count = 0
    if pseudo_unknown is not None:
        pseudo_unknown_geometry_count = (len(class_states) * (len(class_states) - 1) // 2) * max(1, int(pseudo_unknown_samples_per_pair))
        old_support_class_count = sum(
            1
            for state in class_states.values()
            if str(state.group) == "old"
            and state.support_anchors is not None
            and hasattr(state.support_anchors, "numel")
            and int(state.support_anchors.numel()) > 0
        )
        pseudo_unknown_target_shift_count = int(old_support_class_count) * max(0, int(pseudo_unknown_target_shift_samples_per_class))
        pseudo_unknown_target_halo_count = int(old_support_class_count) * max(0, int(pseudo_unknown_target_halo_samples_per_class))
        pseudo_unknown_target_ring_count = max(
            0,
            int(pseudo_unknown.shape[0])
            - int(pseudo_unknown_geometry_count)
            - int(pseudo_unknown_target_shift_count)
            - int(pseudo_unknown_target_halo_count),
        )
    calibration_features = support_for_head
    calibration_labels = support_labels
    calibration_source = "target_support_known"
    if calibration_features.numel() == 0 and protocol == "source_open_set":
        calibration_features = source_for_head.vectors
        calibration_labels = source_for_head.labels
        calibration_source = "source_old_prototypes_no_target_labels"
    if calibration_features.numel() > 0:
        class_states = calibrate_thresholds(
            class_states,
            calibration_features,
            calibration_labels,
            surrogate_unknown=pseudo_unknown,
            target_far=0.05,
            evt_mode="weibull",
            unknown_source="surrogate",
            old_retention_quantile=float(old_retention_quantile),
            old_surrogate_evidence_margin=float(old_surrogate_evidence_margin),
            old_surrogate_reject_relax=float(old_surrogate_reject_relax),
            support_retention_guard=bool(support_retention_guard),
            support_retention_guard_quantile=float(support_retention_guard_quantile),
            support_retention_guard_slack=float(support_retention_guard_slack),
        )
        onboard_telemetry["anchor_density_gate"] = calibrate_anchor_density_gates(
            class_states,
            calibration_features,
            calibration_labels,
            enabled=bool(anchor_density_gate),
            topk=int(anchor_density_topk),
            temperature=float(anchor_density_temperature),
            min_quantile=float(anchor_density_min_quantile),
            margin_quantile=float(anchor_density_margin_quantile),
            action=str(anchor_density_gate_action),
        )
        envelope_features = torch.cat([source_for_head.vectors.detach(), calibration_features.detach()], dim=0)
        envelope_labels = torch.cat([source_for_head.labels.detach(), calibration_labels.detach().cpu().to(source_for_head.labels.device)], dim=0)
        onboard_telemetry["class_envelope_gate"] = calibrate_class_envelope_gates(
            class_states,
            envelope_features,
            envelope_labels,
            enabled=bool(class_envelope_gate),
            evidence_quantile=float(class_envelope_evidence_quantile),
            residual_quantile=float(class_envelope_residual_quantile),
            score_quantile=float(class_envelope_score_quantile),
            margin_quantile=float(class_envelope_margin_quantile),
            evidence_slack=float(class_envelope_evidence_slack),
            residual_slack=float(class_envelope_residual_slack),
            score_slack=float(class_envelope_score_slack),
            margin_slack=float(class_envelope_margin_slack),
            min_failures=int(class_envelope_min_failures),
            action=str(class_envelope_gate_action),
        )
    else:
        onboard_telemetry["anchor_density_gate"] = {
            "enabled": bool(anchor_density_gate),
            "class_count": 0,
            "reason": "empty_known_calibration",
            "unknown_query_threshold_calibration": False,
        }
        onboard_telemetry["class_envelope_gate"] = {
            "enabled": bool(class_envelope_gate),
            "class_count": 0,
            "reason": "empty_known_calibration",
            "unknown_query_threshold_calibration": False,
        }
    onboard_telemetry["pseudo_unknown_energy"] = {
        "enabled": pseudo_unknown is not None,
        "sample_count": 0 if pseudo_unknown is None else int(pseudo_unknown.shape[0]),
        "geometry_sample_count": int(pseudo_unknown_geometry_count),
        "target_shift_sample_count": int(pseudo_unknown_target_shift_count),
        "target_halo_sample_count": int(pseudo_unknown_target_halo_count),
        "target_ring_sample_count": int(pseudo_unknown_target_ring_count),
        "source": "support_prototype_geometry+allowed_target_old_support_shift_halo_ring",
        "samples_per_pair": int(pseudo_unknown_samples_per_pair),
        "offset_scale": float(pseudo_unknown_offset_scale),
        "target_shift_samples_per_class": int(pseudo_unknown_target_shift_samples_per_class),
        "target_shift_offset_scale": float(pseudo_unknown_target_shift_offset_scale),
        "target_halo_samples_per_class": int(pseudo_unknown_target_halo_samples_per_class),
        "target_halo_offset_scale": float(pseudo_unknown_target_halo_offset_scale),
        "target_ring_samples_per_class": int(pseudo_unknown_target_ring_samples_per_class),
        "target_ring_offset_scale": float(pseudo_unknown_target_ring_offset_scale),
        "target_shift_source": "source_prototypes_and_allowed_target_old_support_only",
        "target_halo_source": "allowed_target_old_support_toward_nearest_old_competitor_only",
        "target_ring_source": "allowed_target_old_support_near_support_boundary_only",
        "accept_gate": "old_retention_constrained_energy_plus_surrogate_reject_energy",
        "old_evidence_gate": "support_derived_old_vs_surrogate_unknown_evidence_hard_reject",
        "support_contrast_gate": "adapter_penalizes_surrogate_anchor_affinity_and_preserves_old_bridge_anchor_affinity",
        "known_calibration_source": calibration_source,
        "old_retention_quantile": float(old_retention_quantile),
        "old_surrogate_evidence_margin": float(old_surrogate_evidence_margin),
        "old_surrogate_reject_relax": float(old_surrogate_reject_relax),
        "support_retention_guard": bool(support_retention_guard),
        "support_retention_guard_quantile": float(support_retention_guard_quantile),
        "support_retention_guard_slack": float(support_retention_guard_slack),
        "old_anchor_override_min_quality": float(old_anchor_override_min_quality),
    }
    onboard_telemetry["weibull_evt"] = {
        "enabled": True,
        "class_count": int(len(class_states)),
        "fit": "weibull_moments",
    }
    onboard_telemetry["multiproto_score_head"] = _enable_multiproto_score_head(
        class_states,
        enabled=bool(multiproto_score),
        topk=int(multiproto_topk),
        temperature=float(multiproto_temperature),
        score_weight=float(multiproto_score_weight),
        consistency_gate=bool(mixture_consistency_gate),
        consistency_min_cos=float(mixture_consistency_min_cos),
        consistency_max_residual=float(mixture_consistency_max_residual),
        consistency_min_margin=float(mixture_consistency_min_margin),
        consistency_action=str(mixture_consistency_action),
    )
    seen_new_gate_classes = [
        int(label)
        for label, state in class_states.items()
        if str(state.group) == "seen_new" and "min_seen_new_evidence" in state.thresholds
    ]
    seen_new_anchor_gate_classes = [
        int(label)
        for label, state in class_states.items()
        if str(state.group) == "seen_new" and "min_seen_new_anchor_similarity" in state.thresholds
    ]
    onboard_telemetry["seen_new_evidence_gate"] = {
        "enabled": bool(seen_new_gate_classes),
        "class_count": int(len(seen_new_gate_classes)),
        "anchor_gate_enabled": bool(seen_new_anchor_gate_classes),
        "anchor_gate_class_count": int(len(seen_new_anchor_gate_classes)),
        "source": "seen_new_support_geometry_only",
        "unknown_query_threshold_calibration": False,
        "columns": [
            "seen_new_evidence",
            "seen_new_support_affinity",
            "seen_new_support_residual",
            "seen_new_anchor_similarity",
            "seen_new_anchor_delta",
        ],
    }

    head = OrbitAdaptiveMSEHead(dim=int(source_for_head.vectors.shape[1]), class_states=class_states, beta_residual=0.5, eta_mahalanobis=0.05)
    pred = predict_with_oa_mse_head(query_for_head, head)
    before_seen_override_accepts = int(pred.accepted.sum().item())
    pred = apply_seen_new_registration_override(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(seen_new_registration_override),
        min_evidence_delta=float(seen_new_override_min_evidence_delta),
        min_anchor_delta=float(seen_new_override_min_anchor_delta),
        min_affinity_delta=float(seen_new_override_min_affinity_delta),
        min_residual_delta=float(seen_new_override_min_residual_delta),
        min_score_margin=float(seen_new_override_min_score_margin),
        min_seen_vs_old_evidence_margin=float(seen_new_override_min_seen_vs_old_evidence_margin),
        max_background_score=float(seen_new_override_max_background_score),
        max_background_margin=float(seen_new_override_max_background_margin),
        min_support_knn_seen_new_minus_old=seen_new_override_min_support_knn_seen_new_minus_old,
        min_support_knn_margin=seen_new_override_min_support_knn_margin,
    )
    override_mask = pred.diagnostics.get("seen_new_registration_override_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["seen_new_registration_override"] = {
        "enabled": bool(seen_new_registration_override),
        "scope": "post_oa_mse_head_pre_unknown_guard_seen_new_support_evidence_override",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_seen_override_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "override_count": int(override_mask.sum().item()) if hasattr(override_mask, "sum") else 0,
        "thresholds": {
            "min_evidence_delta": float(seen_new_override_min_evidence_delta),
            "min_anchor_delta": float(seen_new_override_min_anchor_delta),
            "min_affinity_delta": float(seen_new_override_min_affinity_delta),
            "min_residual_delta": float(seen_new_override_min_residual_delta),
            "min_score_margin": float(seen_new_override_min_score_margin),
            "min_seen_vs_old_evidence_margin": float(seen_new_override_min_seen_vs_old_evidence_margin),
            "max_background_score": float(seen_new_override_max_background_score),
            "max_background_margin": float(seen_new_override_max_background_margin),
            "min_support_knn_seen_new_minus_old": seen_new_override_min_support_knn_seen_new_minus_old,
            "min_support_knn_margin": seen_new_override_min_support_knn_margin,
        },
    }
    before_class_envelope_accepts = int(pred.accepted.sum().item())
    pred = apply_class_envelope_gate(
        query_for_head,
        pred,
        head,
        enabled=bool(class_envelope_gate),
    )
    envelope_reject_mask = pred.diagnostics.get("class_envelope_reject_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["class_envelope_gate"].update(
        {
            "scope": "post_seen_new_override_pre_unknown_guard_source_support_class_envelope",
            "accept_count_before": int(before_class_envelope_accepts),
            "accept_count_after": int(pred.accepted.sum().item()),
            "rejected_count": int(envelope_reject_mask.sum().item()) if hasattr(envelope_reject_mask, "sum") else 0,
        }
    )
    before_three_way_accepts = int(pred.accepted.sum().item())
    pred = apply_three_way_decision_head(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(three_way_decision_head),
        temperature=float(three_way_head_temperature),
        accept_prob=float(three_way_accept_prob),
        reject_prob=float(three_way_reject_prob),
        defer_prob=float(three_way_defer_prob),
        known_background_margin=float(three_way_known_background_margin),
        reject_margin=float(three_way_reject_margin),
        old_seen_margin=float(three_way_old_seen_ambiguity_margin),
        defer_action=str(three_way_defer_action),
        decision_policy=str(three_way_decision_policy),
        known_floor_enabled=bool(three_way_known_floor),
        known_floor_action=str(three_way_known_floor_action),
        known_floor_old_min_evidence_delta=float(three_way_known_floor_old_min_evidence_delta),
        known_floor_old_min_anchor_delta=float(three_way_known_floor_old_min_anchor_delta),
        known_floor_old_min_anchor_margin=float(three_way_known_floor_old_min_anchor_margin),
        known_floor_old_min_score_margin=float(three_way_known_floor_old_min_score_margin),
        known_floor_seen_new_min_evidence_delta=float(three_way_known_floor_seen_new_min_evidence_delta),
        known_floor_seen_new_min_anchor_delta=float(three_way_known_floor_seen_new_min_anchor_delta),
        known_floor_seen_new_min_score_margin=float(three_way_known_floor_seen_new_min_score_margin),
        known_floor_background_override_prob=float(three_way_known_floor_background_override_prob),
        known_floor_background_override_margin=float(three_way_known_floor_background_override_margin),
    )
    three_way_accept_mask = pred.diagnostics.get("three_way_accept_mask") if isinstance(pred.diagnostics, dict) else None
    three_way_reject_mask = pred.diagnostics.get("three_way_reject_mask") if isinstance(pred.diagnostics, dict) else None
    three_way_defer_mask = pred.diagnostics.get("three_way_defer_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["three_way_decision_head"] = {
        "enabled": bool(three_way_decision_head),
        "scope": (
            "post_class_envelope_pre_pre_reject_old_seen_new_background_competition"
            if str(three_way_decision_policy) == "background_competition"
            else "post_class_envelope_support_evidence_balanced_known_background_competition"
            if str(three_way_decision_policy) == "evidence_balanced"
            else "post_class_envelope_class_first_known_assignment_then_background_veto"
        ),
        "decision_policy": str(three_way_decision_policy),
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_three_way_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "accepted_count": int(three_way_accept_mask.sum().item()) if hasattr(three_way_accept_mask, "sum") else 0,
        "rejected_count": int(three_way_reject_mask.sum().item()) if hasattr(three_way_reject_mask, "sum") else 0,
        "deferred_count": int(three_way_defer_mask.sum().item()) if hasattr(three_way_defer_mask, "sum") else 0,
        "training_loss_weight": float(three_way_head_weight),
        "thresholds": {
            "temperature": float(three_way_head_temperature),
            "accept_prob": float(three_way_accept_prob),
            "reject_prob": float(three_way_reject_prob),
            "defer_prob": float(three_way_defer_prob),
            "known_background_margin": float(three_way_known_background_margin),
            "reject_margin": float(three_way_reject_margin),
            "old_seen_ambiguity_margin": float(three_way_old_seen_ambiguity_margin),
            "defer_action": str(three_way_defer_action),
            "decision_policy": str(three_way_decision_policy),
            "known_floor": bool(three_way_known_floor),
            "known_floor_action": str(three_way_known_floor_action),
            "known_floor_old_min_evidence_delta": float(three_way_known_floor_old_min_evidence_delta),
            "known_floor_old_min_anchor_delta": float(three_way_known_floor_old_min_anchor_delta),
            "known_floor_old_min_anchor_margin": float(three_way_known_floor_old_min_anchor_margin),
            "known_floor_old_min_score_margin": float(three_way_known_floor_old_min_score_margin),
            "known_floor_seen_new_min_evidence_delta": float(three_way_known_floor_seen_new_min_evidence_delta),
            "known_floor_seen_new_min_anchor_delta": float(three_way_known_floor_seen_new_min_anchor_delta),
            "known_floor_seen_new_min_score_margin": float(three_way_known_floor_seen_new_min_score_margin),
            "known_floor_background_override_prob": float(three_way_known_floor_background_override_prob),
            "known_floor_background_override_margin": float(three_way_known_floor_background_override_margin),
        },
    }
    before_density_shell_accepts = int(pred.accepted.sum().item())
    pred = apply_density_shell_inlier_gate(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(density_shell_gate),
        old_min_evidence_delta=float(density_shell_old_min_evidence_delta),
        old_min_anchor_delta=float(density_shell_old_min_anchor_delta),
        old_min_density_delta=float(density_shell_old_min_density_delta),
        seen_new_min_evidence_delta=float(density_shell_seen_new_min_evidence_delta),
        seen_new_min_anchor_delta=float(density_shell_seen_new_min_anchor_delta),
        seen_new_min_density_delta=float(density_shell_seen_new_min_density_delta),
        accept_background_margin=float(density_shell_accept_background_margin),
        reject_background_score=float(density_shell_reject_background_score),
        reject_background_margin=float(density_shell_reject_background_margin),
        reject_min_failed_shells=int(density_shell_reject_min_failed_shells),
    )
    density_shell_accept_mask = pred.diagnostics.get("density_shell_accept_mask") if isinstance(pred.diagnostics, dict) else None
    density_shell_reject_mask = pred.diagnostics.get("density_shell_reject_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["density_shell_inlier_gate"] = {
        "enabled": bool(density_shell_gate),
        "scope": "post_three_way_pre_pre_reject_class_conditional_inlier_first_open_space_arbitration",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_density_shell_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "accepted_count": int(density_shell_accept_mask.sum().item()) if hasattr(density_shell_accept_mask, "sum") else 0,
        "rejected_count": int(density_shell_reject_mask.sum().item()) if hasattr(density_shell_reject_mask, "sum") else 0,
        "thresholds": {
            "old_min_evidence_delta": float(density_shell_old_min_evidence_delta),
            "old_min_anchor_delta": float(density_shell_old_min_anchor_delta),
            "old_min_density_delta": float(density_shell_old_min_density_delta),
            "seen_new_min_evidence_delta": float(density_shell_seen_new_min_evidence_delta),
            "seen_new_min_anchor_delta": float(density_shell_seen_new_min_anchor_delta),
            "seen_new_min_density_delta": float(density_shell_seen_new_min_density_delta),
            "accept_background_margin": float(density_shell_accept_background_margin),
            "reject_background_score": float(density_shell_reject_background_score),
            "reject_background_margin": float(density_shell_reject_background_margin),
            "reject_min_failed_shells": int(density_shell_reject_min_failed_shells),
        },
    }
    before_identity_accepts = int(pred.accepted.sum().item())
    pred = apply_identity_consensus_arbitration(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(identity_consensus_arbitration),
        old_min_evidence_delta=float(identity_consensus_old_min_evidence_delta),
        old_min_anchor_delta=float(identity_consensus_old_min_anchor_delta),
        old_min_density_delta=float(identity_consensus_old_min_density_delta),
        seen_new_min_evidence_delta=float(identity_consensus_seen_new_min_evidence_delta),
        seen_new_min_anchor_delta=float(identity_consensus_seen_new_min_anchor_delta),
        seen_new_min_density_delta=float(identity_consensus_seen_new_min_density_delta),
        min_identity_margin=float(identity_consensus_min_identity_margin),
        background_accept_margin=float(identity_consensus_background_accept_margin),
        reject_background_score=float(identity_consensus_reject_background_score),
        reject_background_margin=float(identity_consensus_reject_background_margin),
        reject_min_identity_failures=int(identity_consensus_reject_min_identity_failures),
        support_background_cap_enabled=bool(identity_consensus_support_background_cap),
        support_background_cap_quantile=float(identity_consensus_support_background_cap_quantile),
        support_background_cap_slack=float(identity_consensus_support_background_cap_slack),
        support_background_cap_min_anchors=int(identity_consensus_support_background_cap_min_anchors),
    )
    identity_accept_mask = pred.diagnostics.get("identity_consensus_accept_mask") if isinstance(pred.diagnostics, dict) else None
    identity_reject_mask = pred.diagnostics.get("identity_consensus_reject_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["identity_consensus_arbitration"] = {
        "enabled": bool(identity_consensus_arbitration),
        "scope": "post_density_shell_pre_pre_reject_identity_first_old_seen_new_background_arbitration",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_identity_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "accepted_count": int(identity_accept_mask.sum().item()) if hasattr(identity_accept_mask, "sum") else 0,
        "rejected_count": int(identity_reject_mask.sum().item()) if hasattr(identity_reject_mask, "sum") else 0,
        "thresholds": {
            "old_min_evidence_delta": float(identity_consensus_old_min_evidence_delta),
            "old_min_anchor_delta": float(identity_consensus_old_min_anchor_delta),
            "old_min_density_delta": float(identity_consensus_old_min_density_delta),
            "seen_new_min_evidence_delta": float(identity_consensus_seen_new_min_evidence_delta),
            "seen_new_min_anchor_delta": float(identity_consensus_seen_new_min_anchor_delta),
            "seen_new_min_density_delta": float(identity_consensus_seen_new_min_density_delta),
            "min_identity_margin": float(identity_consensus_min_identity_margin),
            "background_accept_margin": float(identity_consensus_background_accept_margin),
            "reject_background_score": float(identity_consensus_reject_background_score),
            "reject_background_margin": float(identity_consensus_reject_background_margin),
            "reject_min_identity_failures": int(identity_consensus_reject_min_identity_failures),
            "support_background_cap": bool(identity_consensus_support_background_cap),
            "support_background_cap_quantile": float(identity_consensus_support_background_cap_quantile),
            "support_background_cap_slack": float(identity_consensus_support_background_cap_slack),
            "support_background_cap_min_anchors": int(identity_consensus_support_background_cap_min_anchors),
        },
    }
    before_support_conformal_accepts = int(pred.accepted.sum().item())
    pred = apply_support_conformal_arbitration(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(support_conformal_arbitration),
        calibration_quantile=float(support_conformal_calibration_quantile),
        conformity_slack=float(support_conformal_conformity_slack),
        anchor_margin_slack=float(support_conformal_anchor_margin_slack),
        background_score=float(support_conformal_background_score),
        background_margin=float(support_conformal_background_margin),
        hard_reject_margin=float(support_conformal_hard_reject_margin),
        reject_min_failures=int(support_conformal_reject_min_failures),
        reject_action=str(support_conformal_reject_action),
    )
    support_conformal_reject_mask = pred.diagnostics.get("support_conformal_reject_mask") if isinstance(pred.diagnostics, dict) else None
    support_conformal_pass_mask = pred.diagnostics.get("support_conformal_pass_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["support_conformal_arbitration"] = {
        "enabled": bool(support_conformal_arbitration),
        "scope": "post_identity_consensus_pre_pre_reject_class_conditional_support_conformal_veto",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_support_conformal_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "passed_count": int(support_conformal_pass_mask.sum().item()) if hasattr(support_conformal_pass_mask, "sum") else 0,
        "rejected_count": int(support_conformal_reject_mask.sum().item()) if hasattr(support_conformal_reject_mask, "sum") else 0,
        "thresholds": {
            "calibration_quantile": float(support_conformal_calibration_quantile),
            "conformity_slack": float(support_conformal_conformity_slack),
            "anchor_margin_slack": float(support_conformal_anchor_margin_slack),
            "background_score": float(support_conformal_background_score),
            "background_margin": float(support_conformal_background_margin),
            "hard_reject_margin": float(support_conformal_hard_reject_margin),
            "reject_min_failures": int(support_conformal_reject_min_failures),
            "reject_action": str(support_conformal_reject_action),
        },
    }
    before_support_reconstruction_accepts = int(pred.accepted.sum().item())
    pred = apply_support_reconstruction_arbitration(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(support_reconstruction_arbitration),
        rank=int(support_reconstruction_rank),
        residual_quantile=float(support_reconstruction_residual_quantile),
        residual_slack=float(support_reconstruction_residual_slack),
        min_residual_floor=float(support_reconstruction_min_residual_floor),
        negative_scale=float(support_reconstruction_negative_scale),
        negative_margin=float(support_reconstruction_negative_margin),
        hard_residual_margin=float(support_reconstruction_hard_residual_margin),
        background_score=float(support_reconstruction_background_score),
        background_margin=float(support_reconstruction_background_margin),
        reject_min_failures=int(support_reconstruction_reject_min_failures),
        reject_action=str(support_reconstruction_reject_action),
    )
    support_reconstruction_reject_mask = (
        pred.diagnostics.get("support_reconstruction_reject_mask") if isinstance(pred.diagnostics, dict) else None
    )
    support_reconstruction_pass_mask = (
        pred.diagnostics.get("support_reconstruction_pass_mask") if isinstance(pred.diagnostics, dict) else None
    )
    onboard_telemetry["support_reconstruction_arbitration"] = {
        "enabled": bool(support_reconstruction_arbitration),
        "scope": "post_support_conformal_pre_pre_reject_class_local_reconstruction_boundary",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_support_reconstruction_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "passed_count": int(support_reconstruction_pass_mask.sum().item()) if hasattr(support_reconstruction_pass_mask, "sum") else 0,
        "rejected_count": int(support_reconstruction_reject_mask.sum().item()) if hasattr(support_reconstruction_reject_mask, "sum") else 0,
        "thresholds": {
            "rank": int(support_reconstruction_rank),
            "residual_quantile": float(support_reconstruction_residual_quantile),
            "residual_slack": float(support_reconstruction_residual_slack),
            "min_residual_floor": float(support_reconstruction_min_residual_floor),
            "negative_scale": float(support_reconstruction_negative_scale),
            "negative_margin": float(support_reconstruction_negative_margin),
            "hard_residual_margin": float(support_reconstruction_hard_residual_margin),
            "background_score": float(support_reconstruction_background_score),
            "background_margin": float(support_reconstruction_background_margin),
            "reject_min_failures": int(support_reconstruction_reject_min_failures),
            "reject_action": str(support_reconstruction_reject_action),
        },
    }
    before_source_looo_risk_accepts = int(pred.accepted.sum().item())
    pred = apply_source_looo_unknown_risk_arbitration(
        query_for_head,
        pred,
        head,
        source_risk_features,
        source_risk_labels,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(source_looo_risk_arbitration),
        risk_quantile=float(source_looo_risk_quantile),
        risk_slack=float(source_looo_risk_slack),
        min_score_margin=float(source_looo_risk_min_score_margin),
        min_known_evidence_delta=float(source_looo_risk_min_known_evidence_delta),
        background_score=float(source_looo_risk_background_score),
        background_margin=float(source_looo_risk_background_margin),
        reject_min_failures=int(source_looo_risk_reject_min_failures),
        reject_action=str(source_looo_risk_reject_action),
    )
    source_looo_risk_reject_mask = pred.diagnostics.get("source_looo_reject_mask") if isinstance(pred.diagnostics, dict) else None
    source_looo_risk_pass_mask = pred.diagnostics.get("source_looo_pass_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["source_looo_unknown_risk_arbitration"] = {
        "enabled": bool(source_looo_risk_arbitration),
        "scope": "post_support_reconstruction_pre_pre_reject_source_leave_one_old_out_impostor_risk",
        "unknown_query_threshold_calibration": False,
        "source_risk_feature_count": int(source_risk_features.shape[0]) if hasattr(source_risk_features, "shape") else 0,
        "accept_count_before": int(before_source_looo_risk_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "passed_count": int(source_looo_risk_pass_mask.sum().item()) if hasattr(source_looo_risk_pass_mask, "sum") else 0,
        "rejected_count": int(source_looo_risk_reject_mask.sum().item()) if hasattr(source_looo_risk_reject_mask, "sum") else 0,
        "thresholds": {
            "risk_quantile": float(source_looo_risk_quantile),
            "risk_slack": float(source_looo_risk_slack),
            "min_score_margin": float(source_looo_risk_min_score_margin),
            "min_known_evidence_delta": float(source_looo_risk_min_known_evidence_delta),
            "background_score": float(source_looo_risk_background_score),
            "background_margin": float(source_looo_risk_background_margin),
            "reject_min_failures": int(source_looo_risk_reject_min_failures),
            "reject_action": str(source_looo_risk_reject_action),
        },
    }
    before_pre_reject_accepts = int(pred.accepted.sum().item())
    pred = apply_pre_reject_defer_arbitration(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(pre_reject_defer_arbitration),
        old_min_evidence_delta=float(pre_reject_old_min_evidence_delta),
        old_min_anchor_delta=float(pre_reject_old_min_anchor_delta),
        old_min_anchor_margin=float(pre_reject_old_min_anchor_margin),
        old_min_score_margin=float(pre_reject_old_min_score_margin),
        seen_new_min_evidence_delta=float(pre_reject_seen_new_min_evidence_delta),
        seen_new_min_anchor_delta=float(pre_reject_seen_new_min_anchor_delta),
        seen_new_min_score_margin=float(pre_reject_seen_new_min_score_margin),
        max_background_score=float(pre_reject_max_background_score),
        max_background_margin=float(pre_reject_max_background_margin),
        defer_background_score=float(pre_reject_defer_background_score),
        defer_background_margin=float(pre_reject_defer_background_margin),
        reject_background_score=float(pre_reject_reject_background_score),
        reject_background_margin=float(pre_reject_reject_background_margin),
        defer_action=str(pre_reject_defer_action),
        support_neighborhood_retention=bool(pre_reject_support_neighborhood_retention),
        support_neighborhood_old_min_evidence_delta=float(pre_reject_support_retention_old_min_evidence_delta),
        support_neighborhood_old_min_anchor_delta=float(pre_reject_support_retention_old_min_anchor_delta),
        support_neighborhood_old_min_anchor_margin=float(pre_reject_support_retention_old_min_anchor_margin),
        support_neighborhood_old_min_score_margin=float(pre_reject_support_retention_old_min_score_margin),
        support_neighborhood_seen_new_min_evidence_delta=float(pre_reject_support_retention_seen_new_min_evidence_delta),
        support_neighborhood_seen_new_min_anchor_delta=float(pre_reject_support_retention_seen_new_min_anchor_delta),
        support_neighborhood_seen_new_min_score_margin=float(pre_reject_support_retention_seen_new_min_score_margin),
        support_neighborhood_max_background_score=float(pre_reject_support_retention_max_background_score),
        support_neighborhood_max_background_margin=float(pre_reject_support_retention_max_background_margin),
        support_neighborhood_require_source_looo_pass=bool(pre_reject_support_retention_require_source_looo_pass),
        support_neighborhood_source_looo_max_failures=int(pre_reject_support_retention_source_looo_max_failures),
    )
    pre_reject_accept_mask = pred.diagnostics.get("pre_reject_arbitration_accept_mask") if isinstance(pred.diagnostics, dict) else None
    pre_reject_reject_mask = pred.diagnostics.get("pre_reject_arbitration_reject_mask") if isinstance(pred.diagnostics, dict) else None
    pre_reject_defer_mask = pred.diagnostics.get("pre_reject_arbitration_defer_mask") if isinstance(pred.diagnostics, dict) else None
    pre_reject_uncertain_mask = pred.diagnostics.get("pre_reject_arbitration_uncertain_mask") if isinstance(pred.diagnostics, dict) else None
    pre_reject_support_retention_mask = (
        pred.diagnostics.get("pre_reject_arbitration_support_retention_mask")
        if isinstance(pred.diagnostics, dict)
        else None
    )
    pre_reject_extreme_background_mask = (
        pred.diagnostics.get("pre_reject_arbitration_extreme_background_mask")
        if isinstance(pred.diagnostics, dict)
        else None
    )
    pre_reject_support_retention_source_looo_block_mask = (
        pred.diagnostics.get("pre_reject_arbitration_support_retention_source_looo_block_mask")
        if isinstance(pred.diagnostics, dict)
        else None
    )
    onboard_telemetry["pre_reject_defer_arbitration"] = {
        "enabled": bool(pre_reject_defer_arbitration),
        "scope": "post_class_envelope_pre_siamese_known_vs_pseudo_background_arbitration",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_pre_reject_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "accepted_count": int(pre_reject_accept_mask.sum().item()) if hasattr(pre_reject_accept_mask, "sum") else 0,
        "rejected_count": int(pre_reject_reject_mask.sum().item()) if hasattr(pre_reject_reject_mask, "sum") else 0,
        "deferred_count": int(pre_reject_defer_mask.sum().item()) if hasattr(pre_reject_defer_mask, "sum") else 0,
        "uncertain_count": int(pre_reject_uncertain_mask.sum().item()) if hasattr(pre_reject_uncertain_mask, "sum") else 0,
        "support_retention_count": int(pre_reject_support_retention_mask.sum().item()) if hasattr(pre_reject_support_retention_mask, "sum") else 0,
        "extreme_background_count": int(pre_reject_extreme_background_mask.sum().item()) if hasattr(pre_reject_extreme_background_mask, "sum") else 0,
        "support_retention_source_looo_block_count": (
            int(pre_reject_support_retention_source_looo_block_mask.sum().item())
            if hasattr(pre_reject_support_retention_source_looo_block_mask, "sum")
            else 0
        ),
        "thresholds": {
            "old_min_evidence_delta": float(pre_reject_old_min_evidence_delta),
            "old_min_anchor_delta": float(pre_reject_old_min_anchor_delta),
            "old_min_anchor_margin": float(pre_reject_old_min_anchor_margin),
            "old_min_score_margin": float(pre_reject_old_min_score_margin),
            "seen_new_min_evidence_delta": float(pre_reject_seen_new_min_evidence_delta),
            "seen_new_min_anchor_delta": float(pre_reject_seen_new_min_anchor_delta),
            "seen_new_min_score_margin": float(pre_reject_seen_new_min_score_margin),
            "max_background_score": float(pre_reject_max_background_score),
            "max_background_margin": float(pre_reject_max_background_margin),
            "defer_background_score": float(pre_reject_defer_background_score),
            "defer_background_margin": float(pre_reject_defer_background_margin),
            "reject_background_score": float(pre_reject_reject_background_score),
            "reject_background_margin": float(pre_reject_reject_background_margin),
            "defer_action": str(pre_reject_defer_action),
            "support_neighborhood_retention": bool(pre_reject_support_neighborhood_retention),
            "support_retention_old_min_evidence_delta": float(pre_reject_support_retention_old_min_evidence_delta),
            "support_retention_old_min_anchor_delta": float(pre_reject_support_retention_old_min_anchor_delta),
            "support_retention_old_min_anchor_margin": float(pre_reject_support_retention_old_min_anchor_margin),
            "support_retention_old_min_score_margin": float(pre_reject_support_retention_old_min_score_margin),
            "support_retention_seen_new_min_evidence_delta": float(pre_reject_support_retention_seen_new_min_evidence_delta),
            "support_retention_seen_new_min_anchor_delta": float(pre_reject_support_retention_seen_new_min_anchor_delta),
            "support_retention_seen_new_min_score_margin": float(pre_reject_support_retention_seen_new_min_score_margin),
            "support_retention_max_background_score": float(pre_reject_support_retention_max_background_score),
            "support_retention_max_background_margin": float(pre_reject_support_retention_max_background_margin),
            "support_retention_require_source_looo_pass": bool(pre_reject_support_retention_require_source_looo_pass),
            "support_retention_source_looo_max_failures": int(pre_reject_support_retention_source_looo_max_failures),
        },
    }
    if support_for_head.numel() > 0 and len(class_states) >= 2:
        verifier = fit_siamese_verifier(support_for_head, support_labels, quantile=float(siamese_quantile))
        before_uncertain = sum(1 for item in pred.decisions if str(item) == "uncertain")
        pred = apply_siamese_verifier_to_ambiguous(
            query_for_head,
            pred,
            class_states,
            verifier,
            threshold=float(siamese_accept_threshold),
            unknown_risk_veto=bool(siamese_unknown_veto),
            unknown_risk_veto_mode=str(siamese_unknown_veto_mode),
            min_old_support_evidence_delta=siamese_min_old_support_evidence_delta,
            min_old_surrogate_reject_delta=siamese_min_old_surrogate_reject_delta,
            min_energy_delta=siamese_min_energy_delta,
            min_mahalanobis_delta=siamese_min_mahalanobis_delta,
            min_accept_delta=siamese_min_accept_delta,
            min_old_support_anchor_margin=siamese_min_old_support_anchor_margin,
            min_veto_failures=int(siamese_min_veto_failures),
        )
        onboard_telemetry["siamese_verifier"] = {
            "enabled": True,
            "ambiguous_only": True,
            "called_rows": int(before_uncertain),
            "threshold": float(verifier.threshold),
            "fit_quantile": float(siamese_quantile),
            "accept_threshold": float(siamese_accept_threshold),
            "unknown_risk_veto": bool(siamese_unknown_veto),
            "unknown_risk_veto_mode": str(siamese_unknown_veto_mode),
            "unknown_risk_veto_thresholds": {
                "old_support_evidence_delta": siamese_min_old_support_evidence_delta,
                "old_surrogate_reject_evidence_delta": siamese_min_old_surrogate_reject_delta,
                "energy_delta": siamese_min_energy_delta,
                "mahalanobis_delta": siamese_min_mahalanobis_delta,
                "min_accept_delta": siamese_min_accept_delta,
                "old_support_anchor_margin": siamese_min_old_support_anchor_margin,
                "min_veto_failures": int(siamese_min_veto_failures),
            },
        }
    else:
        onboard_telemetry["siamese_verifier"] = {"enabled": False, "ambiguous_only": True, "called_rows": 0}
    before_old_unknown_guard_accepts = int(pred.accepted.sum().item())
    pred = apply_old_unknown_acceptance_guard(
        pred,
        class_states,
        enabled=bool(old_unknown_acceptance_guard),
        min_old_support_evidence_delta=old_unknown_guard_min_old_support_evidence_delta,
        min_old_surrogate_reject_delta=old_unknown_guard_min_old_surrogate_reject_delta,
        min_energy_delta=old_unknown_guard_min_energy_delta,
        min_mahalanobis_delta=old_unknown_guard_min_mahalanobis_delta,
        min_accept_delta=old_unknown_guard_min_accept_delta,
        min_old_support_anchor_margin=old_unknown_guard_min_old_support_anchor_margin,
        min_best_old_score=old_unknown_guard_min_best_old_score,
        min_margin=old_unknown_guard_min_margin,
        min_guard_failures=int(old_unknown_guard_min_failures),
    )
    onboard_telemetry["old_unknown_acceptance_guard"] = {
        "enabled": bool(old_unknown_acceptance_guard),
        "scope": "post_accept_old_like_outputs",
        "unknown_query_threshold_calibration": False,
        "accept_count_before": int(before_old_unknown_guard_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "rejected_count": int(before_old_unknown_guard_accepts - int(pred.accepted.sum().item())),
        "thresholds": {
            "old_support_evidence_delta": old_unknown_guard_min_old_support_evidence_delta,
            "old_surrogate_reject_evidence_delta": old_unknown_guard_min_old_surrogate_reject_delta,
            "energy_delta": old_unknown_guard_min_energy_delta,
            "mahalanobis_delta": old_unknown_guard_min_mahalanobis_delta,
            "min_accept_delta": old_unknown_guard_min_accept_delta,
            "old_support_anchor_margin": old_unknown_guard_min_old_support_anchor_margin,
            "best_old_score": old_unknown_guard_min_best_old_score,
            "margin": old_unknown_guard_min_margin,
            "min_failures": int(old_unknown_guard_min_failures),
        },
    }
    before_two_branch_accepts = int(pred.accepted.sum().item())
    pred = apply_two_branch_background_guard(
        query_for_head,
        pred,
        class_states,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(two_branch_background_guard),
        min_background_score=float(two_branch_bg_min_score),
        min_background_margin=float(two_branch_bg_min_margin),
        old_support_evidence_delta=float(two_branch_old_support_evidence_delta),
        old_support_anchor_delta=float(two_branch_old_anchor_delta),
        old_support_anchor_margin=float(two_branch_old_anchor_margin),
        seen_new_evidence_delta=float(two_branch_seen_new_evidence_delta),
        seen_new_anchor_delta=float(two_branch_seen_new_anchor_delta),
    )
    onboard_telemetry["two_branch_background_guard"] = {
        "enabled": bool(two_branch_background_guard),
        "scope": "post_accept_known_outputs_background_risk_veto_with_support_retention_override",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_two_branch_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "rejected_count": int(before_two_branch_accepts - int(pred.accepted.sum().item())),
        "thresholds": {
            "background_score": float(two_branch_bg_min_score),
            "background_margin": float(two_branch_bg_min_margin),
            "old_support_evidence_delta": float(two_branch_old_support_evidence_delta),
            "old_support_anchor_delta": float(two_branch_old_anchor_delta),
            "old_support_anchor_margin": float(two_branch_old_anchor_margin),
            "seen_new_evidence_delta": float(two_branch_seen_new_evidence_delta),
            "seen_new_anchor_delta": float(two_branch_seen_new_anchor_delta),
        },
    }
    before_void_gate_accepts = int(pred.accepted.sum().item())
    pred = apply_pseudo_unknown_void_gate(
        query_for_head,
        pred,
        class_states,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(void_gate),
        min_void_score=float(void_gate_min_score),
        min_void_margin=float(void_gate_min_margin),
        old_support_evidence_delta=float(two_branch_old_support_evidence_delta),
        old_support_anchor_delta=float(two_branch_old_anchor_delta),
        old_support_anchor_margin=float(two_branch_old_anchor_margin),
        seen_new_evidence_delta=float(two_branch_seen_new_evidence_delta),
        seen_new_anchor_delta=float(two_branch_seen_new_anchor_delta),
    )
    onboard_telemetry["void_background_gate"] = {
        "enabled": bool(void_gate),
        "scope": "post_accept_known_outputs",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_void_gate_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "rejected_count": int(before_void_gate_accepts - int(pred.accepted.sum().item())),
        "thresholds": {
            "min_void_score": float(void_gate_min_score),
            "min_void_margin": float(void_gate_min_margin),
            "old_support_evidence_delta": float(two_branch_old_support_evidence_delta),
            "old_support_anchor_delta": float(two_branch_old_anchor_delta),
            "old_support_anchor_margin": float(two_branch_old_anchor_margin),
            "seen_new_evidence_delta": float(two_branch_seen_new_evidence_delta),
            "seen_new_anchor_delta": float(two_branch_seen_new_anchor_delta),
        },
    }
    before_retention_rescue_accepts = int(pred.accepted.sum().item())
    pred = apply_retention_rescue_gate(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(retention_rescue_gate),
        old_min_evidence_delta=float(retention_rescue_old_min_evidence_delta),
        old_min_anchor_delta=float(retention_rescue_old_min_anchor_delta),
        old_min_anchor_margin=float(retention_rescue_old_min_anchor_margin),
        old_min_score_margin=float(retention_rescue_old_min_score_margin),
        seen_new_min_evidence_delta=float(retention_rescue_seen_new_min_evidence_delta),
        seen_new_min_anchor_delta=float(retention_rescue_seen_new_min_anchor_delta),
        seen_new_min_score_margin=float(retention_rescue_seen_new_min_score_margin),
        max_background_score=float(retention_rescue_max_background_score),
        max_background_margin=float(retention_rescue_max_background_margin),
        direct_accept=not bool(retention_rescue_candidate_only),
    )
    retention_rescue_mask = (
        pred.diagnostics.get("retention_rescue_accept_mask")
        if isinstance(pred.diagnostics, dict)
        else None
    )
    onboard_telemetry["retention_rescue_gate"] = {
        "enabled": bool(retention_rescue_gate),
        "scope": "post_reject_gates_pre_accepted_only_online_update",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_retention_rescue_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "rescued_count": int(retention_rescue_mask.sum().item()) if hasattr(retention_rescue_mask, "sum") else 0,
        "thresholds": {
            "old_min_evidence_delta": float(retention_rescue_old_min_evidence_delta),
            "old_min_anchor_delta": float(retention_rescue_old_min_anchor_delta),
            "old_min_anchor_margin": float(retention_rescue_old_min_anchor_margin),
            "old_min_score_margin": float(retention_rescue_old_min_score_margin),
            "seen_new_min_evidence_delta": float(retention_rescue_seen_new_min_evidence_delta),
            "seen_new_min_anchor_delta": float(retention_rescue_seen_new_min_anchor_delta),
            "seen_new_min_score_margin": float(retention_rescue_seen_new_min_score_margin),
            "max_background_score": float(retention_rescue_max_background_score),
            "max_background_margin": float(retention_rescue_max_background_margin),
            "candidate_only": bool(retention_rescue_candidate_only),
        },
    }
    before_old_primary_accepts = int(pred.accepted.sum().item())
    pred = apply_old_primary_acceptance_gate(
        query_for_head,
        pred,
        head,
        pseudo_unknown if pseudo_unknown is not None else torch.empty((0, query_for_head.shape[1]), dtype=torch.float32),
        enabled=bool(old_primary_gate),
        min_old_support_evidence_delta=float(old_primary_min_old_support_evidence_delta),
        min_old_support_anchor_delta=float(old_primary_min_old_support_anchor_delta),
        min_old_support_anchor_margin=float(old_primary_min_old_support_anchor_margin),
        min_score_margin=float(old_primary_min_score_margin),
        require_soft_mixture=bool(old_primary_require_soft_mixture),
        min_soft_mixture_margin=float(old_primary_min_soft_mixture_margin),
        min_soft_mixture_cos=float(old_primary_min_soft_mixture_cos),
        max_soft_mixture_residual=float(old_primary_max_soft_mixture_residual),
        require_support_knn=bool(old_primary_require_support_knn),
        require_support_knn_label_match=bool(old_primary_require_support_knn_label_match),
        min_support_knn_margin=float(old_primary_min_support_knn_margin),
        max_support_knn_seen_new_minus_old=old_primary_max_support_knn_seen_new_minus_old,
        min_old_drift_cos=float(old_primary_min_old_drift_cos),
        max_old_drift_dist=float(old_primary_max_old_drift_dist),
        require_class_envelope=bool(old_primary_require_class_envelope),
        unknown_veto_background_score=float(old_primary_unknown_veto_background_score),
        unknown_veto_background_margin=float(old_primary_unknown_veto_background_margin),
        unknown_veto_min_sources=int(old_primary_unknown_veto_min_sources),
        fail_action=str(old_primary_fail_action),
        unknown_veto_action=str(old_primary_unknown_veto_action),
        promote_rescue_candidates=bool(old_primary_promote_rescue_candidates),
    )
    old_primary_pass_mask = pred.diagnostics.get("old_primary_consistency_pass_mask") if isinstance(pred.diagnostics, dict) else None
    old_primary_veto_mask = pred.diagnostics.get("old_primary_unknown_veto_applied_mask") if isinstance(pred.diagnostics, dict) else None
    old_primary_blocked_mask = pred.diagnostics.get("old_primary_blocked_accept_mask") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["old_primary_acceptance_gate"] = {
        "enabled": bool(old_primary_gate),
        "scope": "terminal_post_rescue_pre_online_update_old_accept_consistency_veto",
        "unknown_query_threshold_calibration": False,
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]) if pseudo_unknown is not None else 0,
        "accept_count_before": int(before_old_primary_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "pass_count": int(old_primary_pass_mask.sum().item()) if hasattr(old_primary_pass_mask, "sum") else 0,
        "unknown_veto_count": int(old_primary_veto_mask.sum().item()) if hasattr(old_primary_veto_mask, "sum") else 0,
        "blocked_accept_count": int(old_primary_blocked_mask.sum().item()) if hasattr(old_primary_blocked_mask, "sum") else 0,
        "thresholds": {
            "min_old_support_evidence_delta": float(old_primary_min_old_support_evidence_delta),
            "min_old_support_anchor_delta": float(old_primary_min_old_support_anchor_delta),
            "min_old_support_anchor_margin": float(old_primary_min_old_support_anchor_margin),
            "min_score_margin": float(old_primary_min_score_margin),
            "require_soft_mixture": bool(old_primary_require_soft_mixture),
            "min_soft_mixture_margin": float(old_primary_min_soft_mixture_margin),
            "min_soft_mixture_cos": float(old_primary_min_soft_mixture_cos),
            "max_soft_mixture_residual": float(old_primary_max_soft_mixture_residual),
            "require_support_knn": bool(old_primary_require_support_knn),
            "require_support_knn_label_match": bool(old_primary_require_support_knn_label_match),
            "min_support_knn_margin": float(old_primary_min_support_knn_margin),
            "max_support_knn_seen_new_minus_old": old_primary_max_support_knn_seen_new_minus_old,
            "min_old_drift_cos": float(old_primary_min_old_drift_cos),
            "max_old_drift_dist": float(old_primary_max_old_drift_dist),
            "require_class_envelope": bool(old_primary_require_class_envelope),
            "unknown_veto_background_score": float(old_primary_unknown_veto_background_score),
            "unknown_veto_background_margin": float(old_primary_unknown_veto_background_margin),
            "unknown_veto_min_sources": int(old_primary_unknown_veto_min_sources),
            "fail_action": str(old_primary_fail_action),
            "unknown_veto_action": str(old_primary_unknown_veto_action),
            "promote_rescue_candidates": bool(old_primary_promote_rescue_candidates),
        },
    }
    before_old80_accepts = int(pred.accepted.sum().item())
    pred = apply_old80_first_head(
        query_for_head,
        pred,
        class_states,
        mode=str(old80_head_mode),
        apply_policy=str(old80_head_apply_policy),
        fusion_rho=float(old80_head_fusion_rho),
        knn_k=int(old80_head_knn_k),
    )
    old80_applied_mask = pred.diagnostics.get("old80_first_applied_mask") if isinstance(pred.diagnostics, dict) else None
    old80_support_cv_acc = pred.diagnostics.get("old80_first_support_cv_acc") if isinstance(pred.diagnostics, dict) else None
    old80_mode_code = pred.diagnostics.get("old80_first_mode_code") if isinstance(pred.diagnostics, dict) else None
    onboard_telemetry["old80_first_head"] = {
        "enabled": str(old80_head_mode).lower() not in {"", "disabled", "none"},
        "scope": "terminal_old_class_recovery_before_online_update_and_before_restoring_open_set_gate",
        "unknown_query_threshold_calibration": False,
        "fit_source": "source_old_prototypes_and_target_old_support_only",
        "mode": str(old80_head_mode),
        "apply_policy": str(old80_head_apply_policy),
        "fusion_rho": float(old80_head_fusion_rho),
        "knn_k": int(old80_head_knn_k),
        "accept_count_before": int(before_old80_accepts),
        "accept_count_after": int(pred.accepted.sum().item()),
        "applied_count": int(old80_applied_mask.sum().item()) if hasattr(old80_applied_mask, "sum") else 0,
        "support_cv_acc": float(old80_support_cv_acc[0].item()) if hasattr(old80_support_cv_acc, "numel") and int(old80_support_cv_acc.numel()) else None,
        "mode_code": float(old80_mode_code[0].item()) if hasattr(old80_mode_code, "numel") and int(old80_mode_code.numel()) else None,
        "phase_gate": "OLD80_FIRST",
        "secondary_objective_policy": "restore_unknown_gate_after_target_old_accuracy_reaches_gate",
    }
    _, online_update = accepted_only_online_update(class_states, query_for_head, pred, momentum=0.05)
    onboard_telemetry["online_update"] = online_update
    metrics = compute_open_set_metrics(
        true_labels=query_labels,
        predicted_labels=pred.predicted_labels,
        accepted=pred.accepted,
        old_labels=old_labels,
        new_labels=new_labels if new_labels else None,
    )
    _add_split_confusion_metrics(metrics, query_labels, pred, old_labels, new_labels if new_labels else None)
    if adapter_alpha_eval_sweep and trained_adapter is not None:
        original_alpha = float(getattr(trained_adapter, "alpha", 1.0))
        alpha_rows = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            try:
                trained_adapter.alpha = float(alpha)
                sweep_support = trained_adapter(support_features)
                sweep_query = trained_adapter(query_features)
                sweep_source = type(source)(
                    labels=source.labels.clone(),
                    vectors=trained_adapter(source.vectors),
                    counts=source.counts.clone(),
                    metadata=dict(source.metadata),
                )
                if protocol == "ftrc":
                    if bool((~old_support_mask).any().item()):
                        raise ValueError("Stage2-B OA-MSE support must contain old classes only")
                    sweep_states, sweep_u_orbit = register_old_classes(
                        sweep_source,
                        sweep_support,
                        support_labels,
                        stage="Stage2-B",
                        old_anchor_override_min_quality=float(old_anchor_override_min_quality),
                        gate_config=gate_config,
                    )
                    sweep_new_labels = set()
                elif protocol == "sfe":
                    sweep_states, sweep_u_orbit = register_old_classes(
                        sweep_source,
                        sweep_support[old_support_mask] if bool(old_support_mask.any().item()) else torch.empty((0, sweep_source.vectors.shape[1])),
                        support_labels[old_support_mask] if bool(old_support_mask.any().item()) else torch.empty((0,), dtype=torch.long),
                        stage="Stage2-C",
                        old_anchor_override_min_quality=float(old_anchor_override_min_quality),
                        gate_config=gate_config,
                    )
                    sweep_new_states = {}
                    if bool(new_support_mask.any().item()):
                        sweep_new_states = register_new_classes(
                            sweep_support[new_support_mask],
                            support_labels[new_support_mask],
                            sweep_states,
                            sweep_u_orbit,
                            stage="Stage2-C",
                            gate_config=gate_config,
                        )
                        sweep_states.update(sweep_new_states)
                    sweep_new_labels = {int(v) for v in sweep_new_states}
                else:
                    sweep_states, sweep_u_orbit = register_old_classes(
                        sweep_source,
                        torch.empty((0, sweep_source.vectors.shape[1]), dtype=torch.float32),
                        torch.empty((0,), dtype=torch.long),
                        stage="Stage2-A",
                        old_anchor_override_min_quality=float(old_anchor_override_min_quality),
                        gate_config=gate_config,
                    )
                    sweep_new_labels = set()
                    _enable_multiproto_score_head(
                        sweep_states,
                        enabled=bool(multiproto_score),
                        topk=int(multiproto_topk),
                        temperature=float(multiproto_temperature),
                        score_weight=float(multiproto_score_weight),
                        consistency_gate=bool(mixture_consistency_gate),
                        consistency_min_cos=float(mixture_consistency_min_cos),
                        consistency_max_residual=float(mixture_consistency_max_residual),
                        consistency_min_margin=float(mixture_consistency_min_margin),
                        consistency_action=str(mixture_consistency_action),
                    )
                sweep_head = OrbitAdaptiveMSEHead(
                    dim=int(sweep_source.vectors.shape[1]),
                    class_states=sweep_states,
                    beta_residual=0.5,
                    eta_mahalanobis=0.05,
                )
                sweep_pred = predict_with_oa_mse_head(sweep_query, sweep_head)
                if sweep_support.numel() > 0 and len(sweep_states) >= 2:
                    sweep_verifier = fit_siamese_verifier(sweep_support, support_labels)
                    sweep_pred = apply_siamese_verifier_to_ambiguous(
                        sweep_query,
                        sweep_pred,
                        sweep_states,
                        sweep_verifier,
                        threshold=float(siamese_accept_threshold),
                        unknown_risk_veto=bool(siamese_unknown_veto),
                        unknown_risk_veto_mode=str(siamese_unknown_veto_mode),
                        min_old_support_evidence_delta=siamese_min_old_support_evidence_delta,
                        min_old_surrogate_reject_delta=siamese_min_old_surrogate_reject_delta,
                        min_energy_delta=siamese_min_energy_delta,
                        min_mahalanobis_delta=siamese_min_mahalanobis_delta,
                        min_accept_delta=siamese_min_accept_delta,
                        min_old_support_anchor_margin=siamese_min_old_support_anchor_margin,
                        min_veto_failures=int(siamese_min_veto_failures),
                    )
                sweep_pseudo_unknown = (
                    generate_pseudo_unknown_features(
                        sweep_states,
                        samples_per_pair=int(pseudo_unknown_samples_per_pair),
                        offset_scale=float(pseudo_unknown_offset_scale),
                        target_shift_samples_per_class=int(pseudo_unknown_target_shift_samples_per_class),
                        target_shift_offset_scale=float(pseudo_unknown_target_shift_offset_scale),
                        target_halo_samples_per_class=int(pseudo_unknown_target_halo_samples_per_class),
                        target_halo_offset_scale=float(pseudo_unknown_target_halo_offset_scale),
                        target_ring_samples_per_class=int(pseudo_unknown_target_ring_samples_per_class),
                        target_ring_offset_scale=float(pseudo_unknown_target_ring_offset_scale),
                    )
                    if len(sweep_states) >= 2
                    else None
                )
                sweep_pred = apply_pseudo_unknown_void_gate(
                    sweep_query,
                    sweep_pred,
                    sweep_states,
                    sweep_pseudo_unknown
                    if sweep_pseudo_unknown is not None
                    else torch.empty((0, sweep_query.shape[1]), dtype=torch.float32),
                    enabled=bool(void_gate),
                    min_void_score=float(void_gate_min_score),
                    min_void_margin=float(void_gate_min_margin),
                    old_support_evidence_delta=float(two_branch_old_support_evidence_delta),
                    old_support_anchor_delta=float(two_branch_old_anchor_delta),
                    old_support_anchor_margin=float(two_branch_old_anchor_margin),
                    seen_new_evidence_delta=float(two_branch_seen_new_evidence_delta),
                    seen_new_anchor_delta=float(two_branch_seen_new_anchor_delta),
                )
                sweep_metrics = compute_open_set_metrics(
                    true_labels=query_labels,
                    predicted_labels=sweep_pred.predicted_labels,
                    accepted=sweep_pred.accepted,
                    old_labels=old_labels,
                    new_labels=sweep_new_labels if sweep_new_labels else None,
                )
                _add_split_confusion_metrics(sweep_metrics, query_labels, sweep_pred, old_labels, sweep_new_labels if sweep_new_labels else None)
                alpha_rows.append(
                    {
                        "alpha": float(alpha),
                        "metrics": sweep_metrics,
                        "class_count": int(len(sweep_states)),
                        "u_orbit_rank": int(sweep_u_orbit.shape[1]),
                        "note": "eval_only_query_labels_not_used_for_training_or_threshold_selection",
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic should not break primary metrics
                alpha_rows.append({"alpha": float(alpha), "error": repr(exc)})
        trained_adapter.alpha = original_alpha
        if isinstance(onboard_telemetry.get("target_adapter"), dict):
            onboard_telemetry["target_adapter"]["alpha_eval_sweep"] = alpha_rows
            onboard_telemetry["target_adapter"]["alpha_eval_sweep_policy"] = "eval_only_no_query_training_or_threshold_fit"
    telemetry = {
        "gate": gate_config.__dict__,
        "gate_reasons": pred.gate_reasons,
        "oa_mse_head": {
            "stage": {"source_open_set": "Stage2-A", "ftrc": "Stage2-B", "sfe": "Stage2-C"}[protocol],
            "class_count": int(len(class_states)),
            "old_class_count": int(len(old_labels)),
            "seen_new_class_count": int(len(new_labels)),
            "u_orbit_rank": int(u_orbit.shape[1]),
            "output_semantics": ["old label", "seen-new label", "reject", "uncertain", "defer"],
            "unknown_query_threshold_calibration": False,
        },
        "oa_mse_onboard_adaptation": onboard_telemetry,
    }
    return AdaptationResult(
        prototype_set=prototype_set,
        predicted_labels=pred.predicted_labels,
        scores=pred.scores,
        accepted=pred.accepted,
        metrics=metrics,
        telemetry=telemetry,
        candidate_labels=pred.candidate_labels,
        diagnostics=dict(pred.diagnostics),
        margins=pred.margins,
        mahalanobis=pred.mahalanobis,
        openmax_distance=pred.openmax_distance,
        gate_reasons=pred.gate_reasons,
        decisions=pred.decisions,
        energy=pred.energy,
        subspace_residual=pred.subspace_residual,
        seen_new_evidence=pred.seen_new_evidence,
        seen_new_support_affinity=pred.seen_new_support_affinity,
        seen_new_support_residual=pred.seen_new_support_residual,
        seen_new_anchor_similarity=pred.seen_new_anchor_similarity,
        seen_new_anchor_delta=pred.seen_new_anchor_delta,
    )


def _prototype_set_from_states_for_eval(class_states: dict[int, object]):
    from cvsrffi.spaceborne_fewshot import _prototype_set_from_class_states

    return _prototype_set_from_class_states(class_states)


def main() -> int:
    args = parse_args()
    payload, manifest = _load_payload(args)
    gate_config = _make_gate_config(args)
    source = build_prototype_set(
        tensor_from_numpy_compatible(payload["source_features"], dtype=torch.float32),
        tensor_from_numpy_compatible(payload["source_labels"], dtype=torch.long),
        gate_config=gate_config,
    )
    support_features = tensor_from_numpy_compatible(payload["support_features"], dtype=torch.float32)
    support_labels = tensor_from_numpy_compatible(payload["support_labels"], dtype=torch.long)
    query_features = tensor_from_numpy_compatible(payload["query_features"], dtype=torch.float32)
    query_labels = tensor_from_numpy_compatible(payload["query_labels"], dtype=torch.long)
    baseline_pred = predict_with_prototypes(query_features, source, gate_config=gate_config)
    baseline_metrics = compute_open_set_metrics(
        true_labels=query_labels,
        predicted_labels=baseline_pred.predicted_labels,
        accepted=baseline_pred.accepted,
        old_labels=source.label_values(),
    )
    external_baseline = _load_json_mapping(args.baseline_metrics_json)
    rollback_baseline_metrics = external_baseline or baseline_metrics
    receiver_domain = None
    if str(gate_config.mode).lower() == "oa_mse":
        receiver_domain = _validate_stage2_receiver_domain(payload, support_labels, old_labels=source.label_values())
        result = _run_oa_mse_protocol(
            protocol=str(args.protocol),
            source=source,
            support_features=support_features,
            support_labels=support_labels,
            query_features=query_features,
            query_labels=query_labels,
            source_adapter_features=tensor_from_numpy_compatible(payload["source_features"], dtype=torch.float32),
            source_adapter_labels=tensor_from_numpy_compatible(payload["source_labels"], dtype=torch.long),
            gate_config=gate_config,
            old_acc_target=float(args.old_acc_target),
            seen_new_acc_target=float(args.seen_new_acc_target),
            adapter_rank=int(args.oa_mse_adapter_rank),
            adapter_kind=str(args.oa_mse_adapter_kind),
            adapter_steps=int(args.oa_mse_adapter_steps),
            adapter_lr=float(args.oa_mse_adapter_lr),
            source_anchor_weight=float(args.oa_mse_source_anchor_weight),
            source_ce_weight=float(args.oa_mse_source_ce_weight),
            unknown_moat_weight=float(args.oa_mse_unknown_moat_weight),
            unknown_moat_margin=float(args.oa_mse_unknown_moat_margin),
            pseudo_unknown_samples_per_pair=int(args.pseudo_unknown_samples_per_pair),
            pseudo_unknown_offset_scale=float(args.pseudo_unknown_offset_scale),
            pseudo_unknown_source_boundary_samples_per_pair=int(args.pseudo_unknown_source_boundary_samples_per_pair),
            pseudo_unknown_source_boundary_offset_scale=float(args.pseudo_unknown_source_boundary_offset_scale),
            pseudo_unknown_target_shift_samples_per_class=int(args.pseudo_unknown_target_shift_samples_per_class),
            pseudo_unknown_target_shift_offset_scale=float(args.pseudo_unknown_target_shift_offset_scale),
            pseudo_unknown_target_halo_samples_per_class=int(args.pseudo_unknown_target_halo_samples_per_class),
            pseudo_unknown_target_halo_offset_scale=float(args.pseudo_unknown_target_halo_offset_scale),
            pseudo_unknown_target_ring_samples_per_class=int(args.pseudo_unknown_target_ring_samples_per_class),
            pseudo_unknown_target_ring_offset_scale=float(args.pseudo_unknown_target_ring_offset_scale),
            old_bridge_weight=float(args.oa_mse_old_bridge_weight),
            old_bridge_samples_per_class=int(args.old_bridge_samples_per_class),
            old_bridge_max_mix=float(args.old_bridge_max_mix),
            support_contrast_weight=float(args.oa_mse_support_contrast_weight),
            support_contrast_negative_margin=float(args.old_support_contrast_negative_margin),
            support_contrast_positive_margin=float(args.old_support_contrast_positive_margin),
            support_center_ce_weight=float(args.oa_mse_support_center_ce_weight),
            support_center_temperature=float(args.support_center_temperature),
            support_center_margin=float(args.support_center_margin),
            soft_proto_weight=float(args.oa_mse_soft_proto_weight),
            soft_proto_topk=int(args.soft_proto_topk),
            soft_proto_temperature=float(args.soft_proto_temperature),
            soft_proto_boundary_weight=float(args.oa_mse_soft_proto_boundary_weight),
            soft_proto_boundary_margin=float(args.soft_proto_boundary_margin),
            multiproto_score=bool(args.oa_mse_multiproto_score),
            multiproto_topk=int(args.multiproto_topk),
            multiproto_temperature=float(args.multiproto_temperature),
            multiproto_score_weight=float(args.multiproto_score_weight),
            mixture_consistency_gate=bool(args.oa_mse_mixture_consistency_gate),
            mixture_consistency_min_cos=float(args.mixture_consistency_min_cos),
            mixture_consistency_max_residual=float(args.mixture_consistency_max_residual),
            mixture_consistency_min_margin=float(args.mixture_consistency_min_margin),
            mixture_consistency_action=str(args.mixture_consistency_action),
            anchor_density_gate=bool(args.oa_mse_anchor_density_gate),
            anchor_density_topk=int(args.anchor_density_topk),
            anchor_density_temperature=float(args.anchor_density_temperature),
            anchor_density_min_quantile=float(args.anchor_density_min_quantile),
            anchor_density_margin_quantile=float(args.anchor_density_margin_quantile),
            anchor_density_gate_action=str(args.anchor_density_gate_action),
            class_envelope_gate=bool(args.oa_mse_class_envelope_gate),
            class_envelope_evidence_quantile=float(args.class_envelope_evidence_quantile),
            class_envelope_residual_quantile=float(args.class_envelope_residual_quantile),
            class_envelope_score_quantile=float(args.class_envelope_score_quantile),
            class_envelope_margin_quantile=float(args.class_envelope_margin_quantile),
            class_envelope_evidence_slack=float(args.class_envelope_evidence_slack),
            class_envelope_residual_slack=float(args.class_envelope_residual_slack),
            class_envelope_score_slack=float(args.class_envelope_score_slack),
            class_envelope_margin_slack=float(args.class_envelope_margin_slack),
            class_envelope_min_failures=int(args.class_envelope_min_failures),
            class_envelope_gate_action=str(args.class_envelope_gate_action),
            old_primary_gate=bool(args.oa_mse_old_primary_gate),
            old_primary_min_old_support_evidence_delta=float(args.old_primary_min_old_support_evidence_delta),
            old_primary_min_old_support_anchor_delta=float(args.old_primary_min_old_support_anchor_delta),
            old_primary_min_old_support_anchor_margin=float(args.old_primary_min_old_support_anchor_margin),
            old_primary_min_score_margin=float(args.old_primary_min_score_margin),
            old_primary_require_soft_mixture=bool(args.old_primary_require_soft_mixture),
            old_primary_min_soft_mixture_margin=float(args.old_primary_min_soft_mixture_margin),
            old_primary_min_soft_mixture_cos=float(args.old_primary_min_soft_mixture_cos),
            old_primary_max_soft_mixture_residual=float(args.old_primary_max_soft_mixture_residual),
            old_primary_require_support_knn=bool(args.old_primary_require_support_knn),
            old_primary_require_support_knn_label_match=not bool(args.old_primary_no_support_knn_label_match),
            old_primary_min_support_knn_margin=float(args.old_primary_min_support_knn_margin),
            old_primary_max_support_knn_seen_new_minus_old=args.old_primary_max_support_knn_seen_new_minus_old,
            old_primary_min_old_drift_cos=float(args.old_primary_min_old_drift_cos),
            old_primary_max_old_drift_dist=float(args.old_primary_max_old_drift_dist),
            old_primary_require_class_envelope=bool(args.old_primary_require_class_envelope),
            old_primary_unknown_veto_background_score=float(args.old_primary_unknown_veto_background_score),
            old_primary_unknown_veto_background_margin=float(args.old_primary_unknown_veto_background_margin),
            old_primary_unknown_veto_min_sources=int(args.old_primary_unknown_veto_min_sources),
            old_primary_fail_action=str(args.old_primary_fail_action),
            old_primary_unknown_veto_action=str(args.old_primary_unknown_veto_action),
            old_primary_promote_rescue_candidates=bool(args.old_primary_promote_rescue_candidates),
            density_shell_gate=bool(args.oa_mse_density_shell_gate),
            density_shell_old_min_evidence_delta=float(args.density_shell_old_min_evidence_delta),
            density_shell_old_min_anchor_delta=float(args.density_shell_old_min_anchor_delta),
            density_shell_old_min_density_delta=float(args.density_shell_old_min_density_delta),
            density_shell_seen_new_min_evidence_delta=float(args.density_shell_seen_new_min_evidence_delta),
            density_shell_seen_new_min_anchor_delta=float(args.density_shell_seen_new_min_anchor_delta),
            density_shell_seen_new_min_density_delta=float(args.density_shell_seen_new_min_density_delta),
            density_shell_accept_background_margin=float(args.density_shell_accept_background_margin),
            density_shell_reject_background_score=float(args.density_shell_reject_background_score),
            density_shell_reject_background_margin=float(args.density_shell_reject_background_margin),
            density_shell_reject_min_failed_shells=int(args.density_shell_reject_min_failed_shells),
            identity_consensus_arbitration=bool(args.oa_mse_identity_consensus_arbitration),
            identity_consensus_old_min_evidence_delta=float(args.identity_consensus_old_min_evidence_delta),
            identity_consensus_old_min_anchor_delta=float(args.identity_consensus_old_min_anchor_delta),
            identity_consensus_old_min_density_delta=float(args.identity_consensus_old_min_density_delta),
            identity_consensus_seen_new_min_evidence_delta=float(args.identity_consensus_seen_new_min_evidence_delta),
            identity_consensus_seen_new_min_anchor_delta=float(args.identity_consensus_seen_new_min_anchor_delta),
            identity_consensus_seen_new_min_density_delta=float(args.identity_consensus_seen_new_min_density_delta),
            identity_consensus_min_identity_margin=float(args.identity_consensus_min_identity_margin),
            identity_consensus_background_accept_margin=float(args.identity_consensus_background_accept_margin),
            identity_consensus_reject_background_score=float(args.identity_consensus_reject_background_score),
            identity_consensus_reject_background_margin=float(args.identity_consensus_reject_background_margin),
            identity_consensus_reject_min_identity_failures=int(args.identity_consensus_reject_min_identity_failures),
            identity_consensus_support_background_cap=bool(args.identity_consensus_support_background_cap),
            identity_consensus_support_background_cap_quantile=float(args.identity_consensus_support_background_cap_quantile),
            identity_consensus_support_background_cap_slack=float(args.identity_consensus_support_background_cap_slack),
            identity_consensus_support_background_cap_min_anchors=int(args.identity_consensus_support_background_cap_min_anchors),
            support_conformal_arbitration=bool(args.oa_mse_support_conformal_arbitration),
            support_conformal_calibration_quantile=float(args.support_conformal_calibration_quantile),
            support_conformal_conformity_slack=float(args.support_conformal_conformity_slack),
            support_conformal_anchor_margin_slack=float(args.support_conformal_anchor_margin_slack),
            support_conformal_background_score=float(args.support_conformal_background_score),
            support_conformal_background_margin=float(args.support_conformal_background_margin),
            support_conformal_hard_reject_margin=float(args.support_conformal_hard_reject_margin),
            support_conformal_reject_min_failures=int(args.support_conformal_reject_min_failures),
            support_conformal_reject_action=str(args.support_conformal_reject_action),
            support_reconstruction_arbitration=bool(args.oa_mse_support_reconstruction_arbitration),
            support_reconstruction_rank=int(args.support_reconstruction_rank),
            support_reconstruction_residual_quantile=float(args.support_reconstruction_residual_quantile),
            support_reconstruction_residual_slack=float(args.support_reconstruction_residual_slack),
            support_reconstruction_min_residual_floor=float(args.support_reconstruction_min_residual_floor),
            support_reconstruction_negative_scale=float(args.support_reconstruction_negative_scale),
            support_reconstruction_negative_margin=float(args.support_reconstruction_negative_margin),
            support_reconstruction_hard_residual_margin=float(args.support_reconstruction_hard_residual_margin),
            support_reconstruction_background_score=float(args.support_reconstruction_background_score),
            support_reconstruction_background_margin=float(args.support_reconstruction_background_margin),
            support_reconstruction_reject_min_failures=int(args.support_reconstruction_reject_min_failures),
            support_reconstruction_reject_action=str(args.support_reconstruction_reject_action),
            three_way_decision_head=bool(args.oa_mse_three_way_decision_head),
            three_way_head_weight=float(args.oa_mse_three_way_head_weight),
            three_way_head_temperature=float(args.three_way_head_temperature),
            three_way_head_known_margin=float(args.three_way_head_known_margin),
            three_way_head_background_margin=float(args.three_way_head_background_margin),
            three_way_head_support_ce_weight=float(args.three_way_head_support_ce_weight),
            three_way_head_pseudo_ce_weight=float(args.three_way_head_pseudo_ce_weight),
            three_way_head_support_background_margin_weight=float(args.three_way_head_support_background_margin_weight),
            three_way_head_pseudo_margin_weight=float(args.three_way_head_pseudo_margin_weight),
            three_way_accept_prob=float(args.three_way_accept_prob),
            three_way_reject_prob=float(args.three_way_reject_prob),
            three_way_defer_prob=float(args.three_way_defer_prob),
            three_way_known_background_margin=float(args.three_way_known_background_margin),
            three_way_reject_margin=float(args.three_way_reject_margin),
            three_way_old_seen_ambiguity_margin=float(args.three_way_old_seen_ambiguity_margin),
            three_way_defer_action=str(args.three_way_defer_action),
            three_way_decision_policy=str(args.three_way_decision_policy),
            three_way_known_floor=bool(args.three_way_known_floor),
            three_way_known_floor_action=str(args.three_way_known_floor_action),
            three_way_known_floor_old_min_evidence_delta=float(args.three_way_known_floor_old_min_evidence_delta),
            three_way_known_floor_old_min_anchor_delta=float(args.three_way_known_floor_old_min_anchor_delta),
            three_way_known_floor_old_min_anchor_margin=float(args.three_way_known_floor_old_min_anchor_margin),
            three_way_known_floor_old_min_score_margin=float(args.three_way_known_floor_old_min_score_margin),
            three_way_known_floor_seen_new_min_evidence_delta=float(args.three_way_known_floor_seen_new_min_evidence_delta),
            three_way_known_floor_seen_new_min_anchor_delta=float(args.three_way_known_floor_seen_new_min_anchor_delta),
            three_way_known_floor_seen_new_min_score_margin=float(args.three_way_known_floor_seen_new_min_score_margin),
            three_way_known_floor_background_override_prob=float(args.three_way_known_floor_background_override_prob),
            three_way_known_floor_background_override_margin=float(args.three_way_known_floor_background_override_margin),
            pre_reject_defer_arbitration=bool(args.oa_mse_pre_reject_defer_arbitration),
            pre_reject_old_min_evidence_delta=float(args.pre_reject_old_min_evidence_delta),
            pre_reject_old_min_anchor_delta=float(args.pre_reject_old_min_anchor_delta),
            pre_reject_old_min_anchor_margin=float(args.pre_reject_old_min_anchor_margin),
            pre_reject_old_min_score_margin=float(args.pre_reject_old_min_score_margin),
            pre_reject_seen_new_min_evidence_delta=float(args.pre_reject_seen_new_min_evidence_delta),
            pre_reject_seen_new_min_anchor_delta=float(args.pre_reject_seen_new_min_anchor_delta),
            pre_reject_seen_new_min_score_margin=float(args.pre_reject_seen_new_min_score_margin),
            pre_reject_max_background_score=float(args.pre_reject_max_background_score),
            pre_reject_max_background_margin=float(args.pre_reject_max_background_margin),
            pre_reject_defer_background_score=float(args.pre_reject_defer_background_score),
            pre_reject_defer_background_margin=float(args.pre_reject_defer_background_margin),
            pre_reject_reject_background_score=float(args.pre_reject_reject_background_score),
            pre_reject_reject_background_margin=float(args.pre_reject_reject_background_margin),
            pre_reject_defer_action=str(args.pre_reject_defer_action),
            pre_reject_support_neighborhood_retention=bool(args.pre_reject_support_neighborhood_retention),
            pre_reject_support_retention_old_min_evidence_delta=float(args.pre_reject_support_retention_old_min_evidence_delta),
            pre_reject_support_retention_old_min_anchor_delta=float(args.pre_reject_support_retention_old_min_anchor_delta),
            pre_reject_support_retention_old_min_anchor_margin=float(args.pre_reject_support_retention_old_min_anchor_margin),
            pre_reject_support_retention_old_min_score_margin=float(args.pre_reject_support_retention_old_min_score_margin),
            pre_reject_support_retention_seen_new_min_evidence_delta=float(args.pre_reject_support_retention_seen_new_min_evidence_delta),
            pre_reject_support_retention_seen_new_min_anchor_delta=float(args.pre_reject_support_retention_seen_new_min_anchor_delta),
            pre_reject_support_retention_seen_new_min_score_margin=float(args.pre_reject_support_retention_seen_new_min_score_margin),
            pre_reject_support_retention_max_background_score=float(args.pre_reject_support_retention_max_background_score),
            pre_reject_support_retention_max_background_margin=float(args.pre_reject_support_retention_max_background_margin),
            pre_reject_support_retention_require_source_looo_pass=bool(args.pre_reject_support_retention_require_source_looo_pass),
            pre_reject_support_retention_source_looo_max_failures=int(args.pre_reject_support_retention_source_looo_max_failures),
            retention_rescue_gate=bool(args.oa_mse_retention_rescue_gate),
            retention_rescue_old_min_evidence_delta=float(args.retention_rescue_old_min_evidence_delta),
            retention_rescue_old_min_anchor_delta=float(args.retention_rescue_old_min_anchor_delta),
            retention_rescue_old_min_anchor_margin=float(args.retention_rescue_old_min_anchor_margin),
            retention_rescue_old_min_score_margin=float(args.retention_rescue_old_min_score_margin),
            retention_rescue_seen_new_min_evidence_delta=float(args.retention_rescue_seen_new_min_evidence_delta),
            retention_rescue_seen_new_min_anchor_delta=float(args.retention_rescue_seen_new_min_anchor_delta),
            retention_rescue_seen_new_min_score_margin=float(args.retention_rescue_seen_new_min_score_margin),
            retention_rescue_max_background_score=float(args.retention_rescue_max_background_score),
            retention_rescue_max_background_margin=float(args.retention_rescue_max_background_margin),
            retention_rescue_candidate_only=bool(args.retention_rescue_candidate_only),
            void_background_weight=float(args.oa_mse_void_background_weight),
            negative_anchor_weight=float(args.oa_mse_negative_anchor_weight),
            negative_anchor_margin=float(args.negative_anchor_margin),
            negative_anchor_temperature=float(args.negative_anchor_temperature),
            negative_anchor_max_anchors=int(args.negative_anchor_max_anchors),
            void_gate=bool(args.oa_mse_void_gate),
            void_gate_min_score=float(args.oa_mse_void_gate_min_score),
            void_gate_min_margin=float(args.oa_mse_void_gate_min_margin),
            old_neighborhood_weight=float(args.oa_mse_old_neighborhood_weight),
            old_neighborhood_samples_per_class=int(args.old_neighborhood_samples_per_class),
            old_neighborhood_radius=float(args.old_neighborhood_radius),
            old_surrogate_margin_weight=float(args.oa_mse_old_surrogate_margin_weight),
            old_surrogate_margin=float(args.old_surrogate_margin),
            source_looo_unknown_weight=float(args.oa_mse_source_looo_unknown_weight),
            source_looo_unknown_margin=float(args.source_looo_unknown_margin),
            source_looo_interclass_margin=float(args.source_looo_interclass_margin),
            source_looo_max_samples_per_class=int(args.source_looo_max_samples_per_class),
            source_looo_risk_arbitration=bool(args.oa_mse_source_looo_risk_arbitration),
            source_looo_risk_quantile=float(args.source_looo_risk_quantile),
            source_looo_risk_slack=float(args.source_looo_risk_slack),
            source_looo_risk_min_score_margin=float(args.source_looo_risk_min_score_margin),
            source_looo_risk_min_known_evidence_delta=float(args.source_looo_risk_min_known_evidence_delta),
            source_looo_risk_background_score=float(args.source_looo_risk_background_score),
            source_looo_risk_background_margin=float(args.source_looo_risk_background_margin),
            source_looo_risk_reject_min_failures=int(args.source_looo_risk_reject_min_failures),
            source_looo_risk_reject_action=str(args.source_looo_risk_reject_action),
            known_coverage_weight=float(args.oa_mse_known_coverage_weight),
            known_coverage_margin=float(args.known_coverage_margin),
            known_coverage_min_affinity=float(args.known_coverage_min_affinity),
            known_coverage_max_samples=int(args.known_coverage_max_samples),
            old_surrogate_evidence_margin=float(args.old_surrogate_evidence_margin),
            old_surrogate_reject_relax=float(args.old_surrogate_reject_relax),
            siamese_quantile=float(args.oa_mse_siamese_quantile),
            siamese_accept_threshold=float(args.oa_mse_siamese_accept_threshold),
            siamese_unknown_veto=bool(args.oa_mse_siamese_unknown_veto),
            siamese_unknown_veto_mode=str(args.oa_mse_siamese_unknown_veto_mode),
            siamese_min_old_support_evidence_delta=args.oa_mse_siamese_min_old_support_evidence_delta,
            siamese_min_old_surrogate_reject_delta=args.oa_mse_siamese_min_old_surrogate_reject_delta,
            siamese_min_energy_delta=args.oa_mse_siamese_min_energy_delta,
            siamese_min_mahalanobis_delta=args.oa_mse_siamese_min_mahalanobis_delta,
            siamese_min_accept_delta=args.oa_mse_siamese_min_accept_delta,
            siamese_min_old_support_anchor_margin=args.oa_mse_siamese_min_old_support_anchor_margin,
            siamese_min_veto_failures=int(args.oa_mse_siamese_min_veto_failures),
            old_unknown_acceptance_guard=bool(args.oa_mse_old_unknown_acceptance_guard),
            old_unknown_guard_min_old_support_evidence_delta=args.oa_mse_old_unknown_guard_min_old_support_evidence_delta,
            old_unknown_guard_min_old_surrogate_reject_delta=args.oa_mse_old_unknown_guard_min_old_surrogate_reject_delta,
            old_unknown_guard_min_energy_delta=args.oa_mse_old_unknown_guard_min_energy_delta,
            old_unknown_guard_min_mahalanobis_delta=args.oa_mse_old_unknown_guard_min_mahalanobis_delta,
            old_unknown_guard_min_accept_delta=args.oa_mse_old_unknown_guard_min_accept_delta,
            old_unknown_guard_min_old_support_anchor_margin=args.oa_mse_old_unknown_guard_min_old_support_anchor_margin,
            old_unknown_guard_min_best_old_score=args.oa_mse_old_unknown_guard_min_best_old_score,
            old_unknown_guard_min_margin=args.oa_mse_old_unknown_guard_min_margin,
            old_unknown_guard_min_failures=int(args.oa_mse_old_unknown_guard_min_failures),
            old80_head_mode=str(args.oa_mse_old80_head_mode),
            old80_head_apply_policy=str(args.old80_head_apply_policy),
            old80_head_fusion_rho=float(args.old80_head_fusion_rho),
            old80_head_knn_k=int(args.old80_head_knn_k),
            old_anchor_override_min_quality=float(args.old_anchor_override_min_quality),
            old_retention_quantile=float(args.old_retention_quantile),
            support_retention_guard=bool(args.oa_mse_support_retention_guard),
            support_retention_guard_quantile=float(args.support_retention_guard_quantile),
            support_retention_guard_slack=float(args.support_retention_guard_slack),
            two_branch_background_guard=bool(args.oa_mse_two_branch_background_guard),
            two_branch_bg_min_score=float(args.two_branch_bg_min_score),
            two_branch_bg_min_margin=float(args.two_branch_bg_min_margin),
            two_branch_old_support_evidence_delta=float(args.two_branch_old_support_evidence_delta),
            two_branch_old_anchor_delta=float(args.two_branch_old_anchor_delta),
            two_branch_old_anchor_margin=float(args.two_branch_old_anchor_margin),
            two_branch_seen_new_evidence_delta=float(args.two_branch_seen_new_evidence_delta),
            two_branch_seen_new_anchor_delta=float(args.two_branch_seen_new_anchor_delta),
            seen_new_registration_override=bool(args.oa_mse_seen_new_registration_override),
            seen_new_override_min_evidence_delta=float(args.seen_new_override_min_evidence_delta),
            seen_new_override_min_anchor_delta=float(args.seen_new_override_min_anchor_delta),
            seen_new_override_min_affinity_delta=float(args.seen_new_override_min_affinity_delta),
            seen_new_override_min_residual_delta=float(args.seen_new_override_min_residual_delta),
            seen_new_override_min_score_margin=float(args.seen_new_override_min_score_margin),
            seen_new_override_min_seen_vs_old_evidence_margin=float(
                args.seen_new_override_min_seen_vs_old_evidence_margin
            ),
            seen_new_override_max_background_score=float(args.seen_new_override_max_background_score),
            seen_new_override_max_background_margin=float(args.seen_new_override_max_background_margin),
            seen_new_override_min_support_knn_seen_new_minus_old=args.seen_new_override_min_support_knn_seen_new_minus_old,
            seen_new_override_min_support_knn_margin=args.seen_new_override_min_support_knn_margin,
            adapter_selection_policy=str(args.oa_mse_adapter_selection_policy),
            adapter_alpha_eval_sweep=bool(args.oa_mse_adapter_alpha_eval_sweep),
        )
        result.telemetry["stage2_receiver_domain"] = receiver_domain
    elif args.protocol == "source_open_set":
        result = AdaptationResult(
            prototype_set=source,
            predicted_labels=baseline_pred.predicted_labels,
            scores=baseline_pred.scores,
            accepted=baseline_pred.accepted,
            metrics=dict(baseline_metrics),
            telemetry={"source_open_set": True, "new_identity_claim": False},
            margins=baseline_pred.margins,
            mahalanobis=baseline_pred.mahalanobis,
            openmax_distance=baseline_pred.openmax_distance,
            gate_reasons=list(baseline_pred.gate_reasons),
            decisions=list(baseline_pred.decisions),
        )
    elif args.protocol == "sfe":
        result = run_sfe_enrollment(
            source,
            support_features,
            support_labels,
            query_features,
            query_labels,
            gate_config=gate_config,
            lifecycle_initial_state=str(args.lifecycle_initial_state),
        )
    else:
        result = run_ftrc_calibration(
            source,
            support_features,
            support_labels,
            query_features,
            query_labels,
            kappa=float(args.kappa),
            gate_config=gate_config,
        )
    _add_open_set_curve_metrics(result.metrics, query_labels, result, gate_config)
    result_new_labels = set(result.prototype_set.metadata.get("new_labels", [])) if isinstance(result.prototype_set.metadata, dict) else set()
    _add_split_confusion_metrics(result.metrics, query_labels, result, source.label_values(), result_new_labels or None)
    rollback_policy = _load_json_mapping(args.rollback_policy_json)
    rollback_rules = rules_from_policy(rollback_policy, default=DEFAULT_SFE_ROLLBACK_RULES)
    rollback = evaluate_rollback_gate(
        before_metrics=rollback_baseline_metrics,
        after_metrics=result.metrics,
        rules=rollback_rules,
    )
    deployed_metrics = rollback_baseline_metrics if rollback.rollback_triggered else result.metrics
    score_table_csv = args.score_table_csv or args.output_json.with_suffix(".score_table.csv")
    _write_score_table(score_table_csv, payload, query_labels, result, gate_config)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol": args.protocol,
        "shots": int(args.shots),
        "manifest": manifest,
        "gate": gate_config.__dict__,
        "baseline_metrics": rollback_baseline_metrics,
        "metrics": result.metrics,
        "deployed_metrics": deployed_metrics,
        "rollback": rollback.to_dict(),
        "telemetry": result.telemetry,
        "score_table_csv": str(score_table_csv),
        "query_labels": [int(v) for v in query_labels.tolist()],
        "query_tx_ids": [str(v) for v in np.asarray(payload.get("query_tx_ids", np.asarray([], dtype=str))).reshape(-1).tolist()],
        "predicted_labels": [int(v) for v in result.predicted_labels.tolist()],
        "candidate_labels": None if result.candidate_labels is None else [int(v) for v in result.candidate_labels.tolist()],
        "accepted": [bool(v) for v in result.accepted.tolist()],
        "scores": [float(v) for v in result.scores.tolist()],
        "diagnostics": dict(result.diagnostics),
        "margins": None if result.margins is None else [float(v) for v in result.margins.tolist()],
        "mahalanobis": None if result.mahalanobis is None else [float(v) for v in result.mahalanobis.tolist()],
        "openmax_distance": None if result.openmax_distance is None else [float(v) for v in result.openmax_distance.tolist()],
        "energy": None if result.energy is None else [float(v) for v in result.energy.tolist()],
        "subspace_residual": None if result.subspace_residual is None else [float(v) for v in result.subspace_residual.tolist()],
        "seen_new_evidence": None if result.seen_new_evidence is None else [float(v) for v in result.seen_new_evidence.tolist()],
        "seen_new_support_affinity": None if result.seen_new_support_affinity is None else [float(v) for v in result.seen_new_support_affinity.tolist()],
        "seen_new_support_residual": None if result.seen_new_support_residual is None else [float(v) for v in result.seen_new_support_residual.tolist()],
        "seen_new_anchor_similarity": None if result.seen_new_anchor_similarity is None else [float(v) for v in result.seen_new_anchor_similarity.tolist()],
        "seen_new_anchor_delta": None if result.seen_new_anchor_delta is None else [float(v) for v in result.seen_new_anchor_delta.tolist()],
        "decisions": list(result.decisions),
        "gate_reasons": list(result.gate_reasons),
    }
    out.update(_standard_loss_trace_payload(result))
    args.output_json.write_text(json.dumps(_json_safe(out), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
