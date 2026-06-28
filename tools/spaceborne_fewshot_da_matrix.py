#!/usr/bin/env python
"""Generate launch artifacts for spaceborne CVS-RFFI few-shot DA validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ONBOARD_ADAPTATION_BUNDLE = (
    "weibull_evt+target_adapter+pseudo_unknown_energy+"
    "seen_new_evidence_gate+seen_new_anchor_gate+siamese_verifier+"
    "accepted_only_online_update+stage2_receiver_domain"
)
PHASE2_TARGET_RECEIVER_ALIAS = "rx7"
PHASE2_TARGET_RECEIVER_LABEL = "20-1"
PHASE2_SOURCE_RECEIVER_LABELS = "1-1,1-19,14-7,18-2,19-2,2-1,2-19"
PHASE2_MANYTX_TARGET_RX_INDEX = "10"
PHASE2_MANYSIG_TARGET_RX_INDEX = "7"
PHASE2_TARGET_RECEIVER_POOL = (
    {"label": "20-1", "manysig_rx_index": "7", "manytx_rx_index": "10", "samples": 27638},
    {"label": "3-19", "manysig_rx_index": "8", "manytx_rx_index": "12", "samples": 26887},
    {"label": "7-14", "manysig_rx_index": "9", "manytx_rx_index": "13", "samples": 26445},
    {"label": "7-7", "manysig_rx_index": "10", "manytx_rx_index": "14", "samples": 26868},
    {"label": "8-8", "manysig_rx_index": "11", "manytx_rx_index": "17", "samples": 26474},
)
PHASE2_TARGET_NEW_TX_INDICES = "6,7"
PHASE2_TARGET_NEW_TX_LABELS = "1-16,1-18"
PHASE2_UNKNOWN_TX_INDICES = "11,12"
PHASE2_UNKNOWN_TX_LABELS = "10-1,10-10"
STAR_GROUND_CHANNEL_IMPL = "simplified_leo_residual"
SIMPLIFIED_LEO_SCENARIOS = "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
LEGACY_LEO_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
PHASE1_PAIC_SAT_VIEW_SCHEDULE = (
    "1@0.30:leo_clear_weak;"
    "41@0.60:leo_low_elev_weak,leo_rain_weak;"
    "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
)
PHASE1_GROUND_PROTO_MASK_MODULES = (
    "phase2_prototypes,feature_masks,tx_rx_geometry,"
    "balanced_tx_rx_sampler,open_world_head"
)
PHASE1_CEN51_NON_REGRESSION_FLOORS = (
    "overall>=88.57; strict_udu>=84.87; receiver_floor>=79.53; "
    "sat_mean_5>=46.564; sat_floor_5>=41.52"
)
PHASE1_GPU0_JOINTSAFE_VARIANTS = {
    "softpseudo_190x10": {
        "label_epochs": 190,
        "pseudo_epochs": 10,
        "lambda_u": 0.20,
        "tau_min": 0.92,
        "tau_max": 0.97,
        "pseudo_quantile": 0.86,
        "use_ema_teacher": True,
        "lambda_sat_cls": 0.60,
        "lambda_sat_cons": 0.0,
        "lambda_group_ce": 0.16,
        "lambda_fishr": 0.04,
        "lambda_domain": 1.0,
        "description": "GPU0_A soft pseudo 190+10 with EMA and lower PAIC pressure",
    },
    "ema_keep15": {
        "label_epochs": 185,
        "pseudo_epochs": 15,
        "lambda_u": 0.25,
        "tau_min": 0.91,
        "tau_max": 0.97,
        "pseudo_quantile": 0.85,
        "use_ema_teacher": True,
        "lambda_sat_cls": 0.65,
        "lambda_sat_cons": 0.005,
        "lambda_group_ce": 0.16,
        "lambda_fishr": 0.04,
        "lambda_domain": 1.15,
        "description": "GPU0_A 185+15 with EMA teacher and guarded pseudo stage",
    },
    "satsoft_no_cons": {
        "label_epochs": 185,
        "pseudo_epochs": 15,
        "lambda_u": 0.25,
        "tau_min": 0.90,
        "tau_max": 0.97,
        "pseudo_quantile": 0.84,
        "use_ema_teacher": False,
        "lambda_sat_cls": 0.55,
        "lambda_sat_cons": 0.0,
        "lambda_group_ce": 0.18,
        "lambda_fishr": 0.04,
        "lambda_domain": 1.0,
        "description": "GPU0_A satellite objective softened with consistency disabled",
    },
    "groupsoft_190x10": {
        "label_epochs": 190,
        "pseudo_epochs": 10,
        "lambda_u": 0.25,
        "tau_min": 0.92,
        "tau_max": 0.985,
        "pseudo_quantile": 0.86,
        "use_ema_teacher": False,
        "lambda_sat_cls": 0.60,
        "lambda_sat_cons": 0.0,
        "lambda_group_ce": 0.14,
        "lambda_fishr": 0.035,
        "lambda_domain": 1.0,
        "description": "GPU0_A group/Fishr pressure softened with 190+10 schedule",
    },
}
DEFAULT_BEX02_TEACHER_CKPT = "${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth"
H06_LOW_PROB_HYBRID_LATEST_CKPT = (
    "${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/"
    "CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth"
)
PHASE2_MANYTX_SAMPLE_AUDIT = {
    "source": "N607 read-only ManyTx.pkl probe 2026-06-22T11:44+08:00",
    "pkl": "/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl",
    "receiver_label": PHASE2_TARGET_RECEIVER_LABEL,
    "manytx_rx_index": PHASE2_MANYTX_TARGET_RX_INDEX,
    "equalized": 1,
    "target_new": {
        "1-16": {"tx_index": 6, "eq1_samples": 200, "min_required": 55},
        "1-18": {"tx_index": 7, "eq1_samples": 200, "min_required": 55},
    },
    "unknown": {
        "10-1": {"tx_index": 11, "eq1_samples": 166, "min_required": 50},
        "10-10": {"tx_index": 12, "eq1_samples": 155, "min_required": 50},
    },
}


@dataclass(frozen=True)
class Candidate:
    cid: str
    protocol: str
    k: int
    target_visibility: str
    label_set_relation: str
    update_module: str
    metrics: str
    command_kind: str
    gpu: int
    description: str
    slot: str = ""
    epochs: int = 0
    adapt_steps_per_epoch: int = 0
    eval_max_batches: int = -1
    sat_eval_max_batches: int = -1
    loss_profile: str = "feature_only"
    seed: int = 1337
    unknown_threshold: float = 0.70
    gate_mode: str = "cosine"
    min_margin: float | None = None
    max_mahalanobis: float | None = None
    openmax_tail_size: int = 20
    openmax_quantile: float = 0.95
    openmax_min_threshold: float = 0.02
    target_adapter_type: str = "logit_calibration"
    adapter_rank: int = 4
    adapter_bottleneck: int = 16
    adapter_alpha: float = 1.0
    adapter_dropout: float = 0.0
    freeze_base_stats: bool = False
    source_tx_ids: str = ""
    target_old_tx_ids: str = ""
    new_tx_ids: str = ""
    unknown_tx_ids: str = ""
    source_rxs: str = ""
    target_receiver_ids: str = ""
    target_receiver_label: str = PHASE2_TARGET_RECEIVER_LABEL
    manysig_target_rx_index: str = PHASE2_MANYSIG_TARGET_RX_INDEX
    manytx_target_rx_index: str = PHASE2_MANYTX_TARGET_RX_INDEX
    source_proto_per_tx: int = 20
    source_query_per_tx: int = 20
    target_old_support_per_tx: int = 0
    target_new_support_per_tx: int = 0
    target_old_query_per_tx: int = 50
    query_per_tx: int = 50
    sfe_max_samples_per_tx: int = 200
    sfe_max_samples_per_combo: int = 0
    export_batch_size: int = 512
    ground_model_label: str = "BEX02_fishr002_mixed_e170"
    ground_model_default_ckpt: str = DEFAULT_BEX02_TEACHER_CKPT
    route_family: str = ""
    oa_mse_stage: str = ""
    eval_protocol: str = ""
    source_target_fusion_policy: str = ""
    fusion_inputs: str = ""
    threshold_selection_label_scope: str = "source_old_and_allowed_support_only; unknown_query_eval_only"
    unknown_query_role: str = "eval_only"
    unknown_query_eval_only: bool = True
    target_new_query_not_threshold_fit: bool = True
    unknown_FAR_target: float = 0.05
    FPR95_target: float = 0.95
    target_old_leo_support: bool = False
    target_new_leo_support: bool = False
    unknown_leo_query: bool = False
    model_output_semantics: str = "old_label,seen_new_label,reject,uncertain,defer"
    uncertain_policy: str = "mark_uncertain_then_defer_if_support_quality_low"
    onboard_low_compute_training: bool = True
    compute_budget_profile: str = "feature_level_low_rank_adapter_rank2_max40steps_no_backbone_update"
    adapter_trainable_params_cap: int = 4096
    max_adapt_steps: int = 40
    oa_mse_source_anchor_weight: float = 0.05
    oa_mse_source_ce_weight: float = 0.10
    oa_mse_unknown_moat_weight: float = 0.10
    oa_mse_unknown_moat_margin: float = 0.45
    pseudo_unknown_samples_per_pair: int = 4
    pseudo_unknown_offset_scale: float = 0.15
    pseudo_unknown_source_boundary_samples_per_pair: int = 0
    pseudo_unknown_source_boundary_offset_scale: float = 0.20
    pseudo_unknown_target_shift_samples_per_class: int = 0
    pseudo_unknown_target_shift_offset_scale: float = 0.20
    pseudo_unknown_target_halo_samples_per_class: int = 0
    pseudo_unknown_target_halo_offset_scale: float = 0.35
    pseudo_unknown_target_ring_samples_per_class: int = 0
    pseudo_unknown_target_ring_offset_scale: float = 0.45
    oa_mse_old_bridge_weight: float = 0.10
    old_bridge_samples_per_class: int = 2
    old_bridge_max_mix: float = 0.85
    oa_mse_support_contrast_weight: float = 0.0
    old_support_contrast_negative_margin: float = 0.78
    old_support_contrast_positive_margin: float = 0.88
    oa_mse_support_center_ce_weight: float = 0.0
    support_center_temperature: float = 0.10
    support_center_margin: float = 0.10
    oa_mse_soft_proto_weight: float = 0.0
    soft_proto_topk: int = 2
    soft_proto_temperature: float = 0.10
    oa_mse_soft_proto_boundary_weight: float = 0.0
    soft_proto_boundary_margin: float = 0.15
    oa_mse_multiproto_score: bool = False
    multiproto_topk: int = 2
    multiproto_temperature: float = 0.10
    multiproto_score_weight: float = 1.0
    oa_mse_mixture_consistency_gate: bool = False
    mixture_consistency_min_cos: float = -1.0
    mixture_consistency_max_residual: float = 1.0e6
    mixture_consistency_min_margin: float = -1.0e6
    mixture_consistency_action: str = "uncertain"
    oa_mse_void_background_weight: float = 0.0
    oa_mse_negative_anchor_weight: float = 0.0
    negative_anchor_margin: float = 0.12
    negative_anchor_temperature: float = 0.10
    negative_anchor_max_anchors: int = 256
    oa_mse_void_gate: bool = False
    oa_mse_void_gate_min_score: float = 0.55
    oa_mse_void_gate_min_margin: float = 0.05
    oa_mse_old_neighborhood_weight: float = 0.10
    old_neighborhood_samples_per_class: int = 2
    old_neighborhood_radius: float = 0.06
    oa_mse_old_surrogate_margin_weight: float = 0.05
    old_surrogate_margin: float = 0.10
    oa_mse_source_looo_unknown_weight: float = 0.0
    source_looo_unknown_margin: float = 0.35
    source_looo_interclass_margin: float = 0.08
    source_looo_max_samples_per_class: int = 24
    oa_mse_source_looo_risk_arbitration: bool = False
    source_looo_risk_quantile: float = 0.85
    source_looo_risk_slack: float = 0.0
    source_looo_risk_min_score_margin: float = 0.02
    source_looo_risk_min_known_evidence_delta: float = -0.08
    source_looo_risk_background_score: float = 0.86
    source_looo_risk_background_margin: float = 0.10
    source_looo_risk_reject_min_failures: int = 2
    source_looo_risk_reject_action: str = "reject"
    oa_mse_known_coverage_weight: float = 0.0
    known_coverage_margin: float = 0.12
    known_coverage_min_affinity: float = 0.35
    known_coverage_max_samples: int = 256
    old_surrogate_evidence_margin: float = 0.0
    old_surrogate_reject_relax: float = 0.0
    oa_mse_siamese_quantile: float = 0.10
    oa_mse_siamese_accept_threshold: float = 0.50
    oa_mse_siamese_unknown_veto: bool = False
    oa_mse_siamese_unknown_veto_mode: str = "any"
    oa_mse_siamese_min_old_support_evidence_delta: float | None = None
    oa_mse_siamese_min_old_surrogate_reject_delta: float | None = None
    oa_mse_siamese_min_energy_delta: float | None = None
    oa_mse_siamese_min_mahalanobis_delta: float | None = None
    oa_mse_siamese_min_accept_delta: float | None = None
    oa_mse_siamese_min_old_support_anchor_margin: float | None = None
    oa_mse_siamese_min_veto_failures: int = 1
    oa_mse_old_unknown_acceptance_guard: bool = False
    oa_mse_old_unknown_guard_min_old_support_evidence_delta: float | None = None
    oa_mse_old_unknown_guard_min_old_surrogate_reject_delta: float | None = None
    oa_mse_old_unknown_guard_min_energy_delta: float | None = None
    oa_mse_old_unknown_guard_min_mahalanobis_delta: float | None = None
    oa_mse_old_unknown_guard_min_accept_delta: float | None = None
    oa_mse_old_unknown_guard_min_old_support_anchor_margin: float | None = None
    oa_mse_old_unknown_guard_min_best_old_score: float | None = None
    oa_mse_old_unknown_guard_min_margin: float | None = None
    oa_mse_old_unknown_guard_min_failures: int = 1
    stage2_max_active_per_gpu: int | None = None
    oa_mse_adapter_selection_policy: str = "final"
    oa_mse_adapter_alpha_eval_sweep: bool = False
    old_anchor_override_min_quality: float = 0.55
    old_retention_quantile: float = 0.95
    oa_mse_support_retention_guard: bool = False
    support_retention_guard_quantile: float = 0.05
    support_retention_guard_slack: float = 0.02
    oa_mse_two_branch_background_guard: bool = False
    two_branch_bg_min_score: float = 0.62
    two_branch_bg_min_margin: float = -0.02
    two_branch_old_support_evidence_delta: float = 0.0
    two_branch_old_anchor_delta: float = -0.02
    two_branch_old_anchor_margin: float = 0.0
    two_branch_seen_new_evidence_delta: float = 0.0
    two_branch_seen_new_anchor_delta: float = 0.0
    oa_mse_seen_new_registration_override: bool = False
    seen_new_override_min_evidence_delta: float = 0.0
    seen_new_override_min_anchor_delta: float = 0.0
    seen_new_override_min_affinity_delta: float = -0.02
    seen_new_override_min_residual_delta: float = -0.02
    seen_new_override_min_score_margin: float = -0.10
    seen_new_override_min_seen_vs_old_evidence_margin: float = 0.02
    seen_new_override_max_background_score: float = 0.72
    seen_new_override_max_background_margin: float = 0.08
    seen_new_override_min_support_knn_seen_new_minus_old: float | None = None
    seen_new_override_min_support_knn_margin: float | None = None
    old_acc_target: float = 0.90
    seen_new_acc_target: float = 0.75
    stage2_priority_phase: str = ""
    old_acc_phase_gate: float = 0.0
    secondary_objectives_after_old_gate: str = ""
    optimization_category: str = "conservative"
    oa_mse_adapter_kind: str = "low_rank"
    oa_mse_anchor_density_gate: bool = False
    anchor_density_topk: int = 3
    anchor_density_temperature: float = 0.08
    anchor_density_min_quantile: float = 0.05
    anchor_density_margin_quantile: float = 0.05
    anchor_density_gate_action: str = "uncertain"
    oa_mse_class_envelope_gate: bool = False
    class_envelope_evidence_quantile: float = 0.05
    class_envelope_residual_quantile: float = 0.95
    class_envelope_score_quantile: float = 0.05
    class_envelope_margin_quantile: float = 0.05
    class_envelope_evidence_slack: float = 0.02
    class_envelope_residual_slack: float = 0.02
    class_envelope_score_slack: float = 0.05
    class_envelope_margin_slack: float = 0.02
    class_envelope_min_failures: int = 1
    class_envelope_gate_action: str = "reject"
    oa_mse_old_primary_gate: bool = False
    old_primary_min_old_support_evidence_delta: float = 0.0
    old_primary_min_old_support_anchor_delta: float = -0.02
    old_primary_min_old_support_anchor_margin: float = 0.0
    old_primary_min_score_margin: float = 0.0
    old_primary_require_soft_mixture: bool = False
    old_primary_min_soft_mixture_margin: float = -1.0e6
    old_primary_min_soft_mixture_cos: float = -1.0
    old_primary_max_soft_mixture_residual: float = 1.0e6
    old_primary_require_support_knn: bool = False
    old_primary_require_support_knn_label_match: bool = True
    old_primary_min_support_knn_margin: float = 0.0
    old_primary_max_support_knn_seen_new_minus_old: float | None = None
    old_primary_min_old_drift_cos: float = -1.0
    old_primary_max_old_drift_dist: float = 1.0e6
    old_primary_require_class_envelope: bool = False
    old_primary_unknown_veto_background_score: float = 0.86
    old_primary_unknown_veto_background_margin: float = 0.10
    old_primary_unknown_veto_min_sources: int = 1
    old_primary_fail_action: str = "defer"
    old_primary_unknown_veto_action: str = "reject"
    old_primary_promote_rescue_candidates: bool = False
    oa_mse_density_shell_gate: bool = False
    density_shell_old_min_evidence_delta: float = -0.04
    density_shell_old_min_anchor_delta: float = -0.08
    density_shell_old_min_density_delta: float = -0.06
    density_shell_seen_new_min_evidence_delta: float = -0.04
    density_shell_seen_new_min_anchor_delta: float = -0.08
    density_shell_seen_new_min_density_delta: float = -0.06
    density_shell_accept_background_margin: float = 0.18
    density_shell_reject_background_score: float = 0.86
    density_shell_reject_background_margin: float = 0.14
    density_shell_reject_min_failed_shells: int = 2
    oa_mse_identity_consensus_arbitration: bool = False
    identity_consensus_old_min_evidence_delta: float = -0.06
    identity_consensus_old_min_anchor_delta: float = -0.10
    identity_consensus_old_min_density_delta: float = -0.08
    identity_consensus_seen_new_min_evidence_delta: float = -0.04
    identity_consensus_seen_new_min_anchor_delta: float = -0.08
    identity_consensus_seen_new_min_density_delta: float = -0.06
    identity_consensus_min_identity_margin: float = -0.05
    identity_consensus_background_accept_margin: float = 0.22
    identity_consensus_reject_background_score: float = 0.90
    identity_consensus_reject_background_margin: float = 0.18
    identity_consensus_reject_min_identity_failures: int = 4
    identity_consensus_support_background_cap: bool = False
    identity_consensus_support_background_cap_quantile: float = 0.90
    identity_consensus_support_background_cap_slack: float = 0.05
    identity_consensus_support_background_cap_min_anchors: int = 2
    oa_mse_support_conformal_arbitration: bool = False
    support_conformal_calibration_quantile: float = 0.05
    support_conformal_conformity_slack: float = 0.12
    support_conformal_anchor_margin_slack: float = 0.06
    support_conformal_background_score: float = 0.82
    support_conformal_background_margin: float = 0.08
    support_conformal_hard_reject_margin: float = 0.18
    support_conformal_reject_min_failures: int = 2
    support_conformal_reject_action: str = "reject"
    oa_mse_support_reconstruction_arbitration: bool = False
    support_reconstruction_rank: int = 2
    support_reconstruction_residual_quantile: float = 0.95
    support_reconstruction_residual_slack: float = 0.04
    support_reconstruction_min_residual_floor: float = 0.03
    support_reconstruction_negative_scale: float = 0.55
    support_reconstruction_negative_margin: float = -0.02
    support_reconstruction_hard_residual_margin: float = 0.08
    support_reconstruction_background_score: float = 0.86
    support_reconstruction_background_margin: float = 0.12
    support_reconstruction_reject_min_failures: int = 2
    support_reconstruction_reject_action: str = "reject"
    oa_mse_pre_reject_defer_arbitration: bool = False
    oa_mse_three_way_decision_head: bool = False
    oa_mse_three_way_head_weight: float = 0.0
    three_way_head_temperature: float = 0.10
    three_way_head_known_margin: float = 0.08
    three_way_head_background_margin: float = 0.08
    three_way_head_support_ce_weight: float = 1.0
    three_way_head_pseudo_ce_weight: float = 0.35
    three_way_head_support_background_margin_weight: float = 1.0
    three_way_head_pseudo_margin_weight: float = 0.50
    three_way_accept_prob: float = 0.50
    three_way_reject_prob: float = 0.55
    three_way_defer_prob: float = 0.45
    three_way_known_background_margin: float = 0.02
    three_way_reject_margin: float = 0.04
    three_way_old_seen_ambiguity_margin: float = 0.04
    three_way_defer_action: str = "uncertain"
    three_way_decision_policy: str = "background_competition"
    three_way_known_floor: bool = False
    three_way_known_floor_action: str = "defer"
    three_way_known_floor_old_min_evidence_delta: float = -0.04
    three_way_known_floor_old_min_anchor_delta: float = -0.08
    three_way_known_floor_old_min_anchor_margin: float = -0.04
    three_way_known_floor_old_min_score_margin: float = -0.12
    three_way_known_floor_seen_new_min_evidence_delta: float = -0.04
    three_way_known_floor_seen_new_min_anchor_delta: float = -0.08
    three_way_known_floor_seen_new_min_score_margin: float = -0.12
    three_way_known_floor_background_override_prob: float = 0.995
    three_way_known_floor_background_override_margin: float = 1.0
    pre_reject_old_min_evidence_delta: float = 0.0
    pre_reject_old_min_anchor_delta: float = -0.02
    pre_reject_old_min_anchor_margin: float = 0.0
    pre_reject_old_min_score_margin: float = -0.02
    pre_reject_seen_new_min_evidence_delta: float = 0.0
    pre_reject_seen_new_min_anchor_delta: float = 0.0
    pre_reject_seen_new_min_score_margin: float = -0.05
    pre_reject_max_background_score: float = 0.74
    pre_reject_max_background_margin: float = 0.10
    pre_reject_defer_background_score: float = 0.70
    pre_reject_defer_background_margin: float = 0.04
    pre_reject_reject_background_score: float = 0.82
    pre_reject_reject_background_margin: float = 0.12
    pre_reject_defer_action: str = "uncertain"
    pre_reject_support_neighborhood_retention: bool = False
    pre_reject_support_retention_old_min_evidence_delta: float = 0.02
    pre_reject_support_retention_old_min_anchor_delta: float = -0.04
    pre_reject_support_retention_old_min_anchor_margin: float = -0.02
    pre_reject_support_retention_old_min_score_margin: float = -0.04
    pre_reject_support_retention_seen_new_min_evidence_delta: float = 0.02
    pre_reject_support_retention_seen_new_min_anchor_delta: float = -0.04
    pre_reject_support_retention_seen_new_min_score_margin: float = -0.08
    pre_reject_support_retention_max_background_score: float = 0.96
    pre_reject_support_retention_max_background_margin: float = 0.30
    pre_reject_support_retention_require_source_looo_pass: bool = False
    pre_reject_support_retention_source_looo_max_failures: int = 0
    oa_mse_retention_rescue_gate: bool = False
    retention_rescue_old_min_evidence_delta: float = 0.02
    retention_rescue_old_min_anchor_delta: float = -0.01
    retention_rescue_old_min_anchor_margin: float = 0.0
    retention_rescue_old_min_score_margin: float = 0.0
    retention_rescue_seen_new_min_evidence_delta: float = 0.02
    retention_rescue_seen_new_min_anchor_delta: float = 0.0
    retention_rescue_seen_new_min_score_margin: float = -0.02
    retention_rescue_max_background_score: float = 0.70
    retention_rescue_max_background_margin: float = 0.06
    retention_rescue_candidate_only: bool = False
    star_ground_channel_impl: str = STAR_GROUND_CHANNEL_IMPL
    target_channel_scenarios: str = SIMPLIFIED_LEO_SCENARIOS
    weibull_evt_required: bool = True
    target_adapter_required: bool = True
    pseudo_unknown_energy_required: bool = True
    seen_new_evidence_gate_required: bool = True
    seen_new_anchor_gate_required: bool = True
    siamese_verifier_required: bool = True
    accepted_only_online_update_required: bool = True
    oa_mse_onboard_adaptation_bundle: str = ONBOARD_ADAPTATION_BUNDLE
    lane: str = "phase2_spaceborne_fsl"
    phase_axis: str = "Phase2-Spaceborne-FSL"
    phase1_variant: str = ""
    phase1_design_report_ref: str = ""
    phase1_enable_ground_prototype_stats: bool = False
    phase1_enable_feature_distribution_audit: bool = False
    phase1_enable_feature_masks_aux: bool = False
    phase1_enable_txrx_geometry_audit: bool = False


def _oa_mse_struct48_stage_specs() -> list[dict]:
    """Half conservative, half aggressive structural OA-MSE candidates."""

    return [
        {
            "slot": "A",
            "category": "conservative",
            "stage": "mse_subspace",
            "eval_protocol": "ftrc",
            "k_old": 5,
            "k_new": 0,
            "steps": 36,
            "unknown_threshold": 0.96,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.10,
            "source_ce": 0.28,
            "unknown_moat": 0.22,
            "unknown_margin": 0.52,
            "boundary_samples": 4,
            "boundary_offset": 0.20,
            "soft_proto": 0.18,
            "soft_proto_boundary": 0.12,
            "support_contrast": 0.16,
            "old_bridge": 0.22,
            "old_neighborhood": 0.24,
            "multiproto_score": True,
            "multiproto_topk": 2,
            "multiproto_temperature": 0.08,
            "multiproto_score_weight": 0.85,
            "anchor_density_gate": True,
            "anchor_density_action": "uncertain",
            "anchor_density_quantile": 0.03,
            "anchor_density_margin_quantile": 0.03,
            "adapter_kind": "low_rank",
            "old_retention_quantile": 0.80,
            "description": "Conservative structure: keep best-old mixhead behavior and add support-anchor density uncertainty instead of hard void rejection.",
        },
        {
            "slot": "B",
            "category": "conservative",
            "stage": "oa_mse_head",
            "eval_protocol": "sfe",
            "k_old": 5,
            "k_new": 5,
            "steps": 40,
            "unknown_threshold": 0.97,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.10,
            "source_ce": 0.30,
            "unknown_moat": 0.22,
            "unknown_margin": 0.50,
            "boundary_samples": 4,
            "boundary_offset": 0.20,
            "soft_proto": 0.20,
            "soft_proto_boundary": 0.14,
            "support_contrast": 0.17,
            "old_bridge": 0.22,
            "old_neighborhood": 0.24,
            "multiproto_score": True,
            "multiproto_topk": 2,
            "multiproto_temperature": 0.08,
            "multiproto_score_weight": 1.0,
            "anchor_density_gate": True,
            "anchor_density_action": "uncertain",
            "anchor_density_quantile": 0.03,
            "anchor_density_margin_quantile": 0.03,
            "adapter_kind": "low_rank",
            "old_retention_quantile": 0.78,
            "description": "Conservative Stage2-C: preserve best seen-new mixhead route while density-gating low-support local neighborhoods.",
        },
        {
            "slot": "C",
            "category": "conservative",
            "stage": "oa_mse_head",
            "eval_protocol": "sfe",
            "k_old": 10,
            "k_new": 10,
            "steps": 44,
            "unknown_threshold": 0.98,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.12,
            "source_ce": 0.32,
            "unknown_moat": 0.24,
            "unknown_margin": 0.54,
            "boundary_samples": 5,
            "boundary_offset": 0.20,
            "soft_proto": 0.21,
            "soft_proto_boundary": 0.14,
            "support_contrast": 0.18,
            "old_bridge": 0.24,
            "old_neighborhood": 0.25,
            "multiproto_score": True,
            "multiproto_topk": 3,
            "multiproto_temperature": 0.09,
            "multiproto_score_weight": 0.85,
            "anchor_density_gate": True,
            "anchor_density_action": "uncertain",
            "anchor_density_quantile": 0.05,
            "anchor_density_margin_quantile": 0.04,
            "adapter_kind": "low_rank",
            "old_retention_quantile": 0.76,
            "description": "Conservative higher-support check: test whether density gate improves FAR without repeating hard void collapse.",
        },
        {
            "slot": "D",
            "category": "aggressive",
            "stage": "mse_subspace",
            "eval_protocol": "ftrc",
            "k_old": 5,
            "k_new": 0,
            "steps": 52,
            "unknown_threshold": 0.98,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.12,
            "source_ce": 0.38,
            "unknown_moat": 0.32,
            "unknown_margin": 0.62,
            "boundary_samples": 6,
            "boundary_offset": 0.22,
            "soft_proto": 0.22,
            "soft_proto_boundary": 0.18,
            "support_contrast": 0.24,
            "old_bridge": 0.28,
            "old_neighborhood": 0.30,
            "multiproto_score": True,
            "multiproto_topk": 3,
            "multiproto_temperature": 0.07,
            "multiproto_score_weight": 1.0,
            "anchor_density_gate": True,
            "anchor_density_action": "reject",
            "anchor_density_quantile": 0.08,
            "anchor_density_margin_quantile": 0.06,
            "adapter_kind": "residual_mlp",
            "old_unknown_acceptance_guard": True,
            "guard_min_old_support_evidence_delta": -0.02,
            "guard_min_old_surrogate_reject_delta": 0.02,
            "guard_min_energy_delta": 0.0,
            "guard_min_margin": 0.10,
            "guard_min_failures": 3,
            "old_retention_quantile": 0.76,
            "description": "Aggressive representation repair: residual-MLP adapter plus density rejection to force a cleaner old/unknown boundary.",
        },
        {
            "slot": "E",
            "category": "aggressive",
            "stage": "oa_mse_head",
            "eval_protocol": "sfe",
            "k_old": 5,
            "k_new": 5,
            "steps": 56,
            "unknown_threshold": 0.985,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.12,
            "source_ce": 0.40,
            "unknown_moat": 0.34,
            "unknown_margin": 0.64,
            "boundary_samples": 6,
            "boundary_offset": 0.22,
            "soft_proto": 0.24,
            "soft_proto_boundary": 0.20,
            "soft_proto_boundary_margin": 0.20,
            "support_contrast": 0.26,
            "old_bridge": 0.30,
            "old_neighborhood": 0.30,
            "multiproto_score": True,
            "multiproto_topk": 3,
            "multiproto_temperature": 0.07,
            "multiproto_score_weight": 1.0,
            "anchor_density_gate": True,
            "anchor_density_action": "reject",
            "anchor_density_quantile": 0.08,
            "anchor_density_margin_quantile": 0.08,
            "adapter_kind": "residual_mlp",
            "siamese_unknown_veto": True,
            "siamese_unknown_veto_mode": "coupled",
            "siamese_threshold": 0.72,
            "min_old_support_evidence_delta": 0.00,
            "min_old_surrogate_reject_delta": 0.02,
            "min_energy_delta": 0.0,
            "min_accept_delta": -6.0,
            "min_old_support_anchor_margin": 0.025,
            "min_veto_failures": 3,
            "old_retention_quantile": 0.74,
            "description": "Aggressive Stage2-C: nonlinear feature repair, supervised contrast, soft boundary, density rejection and coupled Siamese veto.",
        },
        {
            "slot": "F",
            "category": "aggressive",
            "stage": "oa_mse_head",
            "eval_protocol": "sfe",
            "k_old": 10,
            "k_new": 10,
            "steps": 56,
            "unknown_threshold": 0.99,
            "openmax_quantile": 1.0,
            "openmax_min_threshold": 0.14,
            "source_ce": 0.42,
            "unknown_moat": 0.36,
            "unknown_margin": 0.66,
            "boundary_samples": 7,
            "boundary_offset": 0.24,
            "soft_proto": 0.24,
            "soft_proto_boundary": 0.22,
            "soft_proto_boundary_margin": 0.22,
            "support_contrast": 0.28,
            "old_bridge": 0.30,
            "old_neighborhood": 0.32,
            "multiproto_score": True,
            "multiproto_topk": 3,
            "multiproto_temperature": 0.07,
            "multiproto_score_weight": 0.85,
            "anchor_density_gate": True,
            "anchor_density_action": "reject",
            "anchor_density_quantile": 0.10,
            "anchor_density_margin_quantile": 0.08,
            "adapter_kind": "residual_mlp",
            "void_background": 0.05,
            "old_unknown_acceptance_guard": True,
            "guard_min_old_support_evidence_delta": 0.00,
            "guard_min_old_surrogate_reject_delta": 0.04,
            "guard_min_energy_delta": 2.0,
            "guard_min_margin": 0.12,
            "guard_min_failures": 3,
            "old_retention_quantile": 0.72,
            "description": "Aggressive saturation: residual-MLP plus one-class density and background pressure to test whether higher K creates open-set margin.",
        },
    ]


def _oa_mse_simplified48_stage_specs() -> list[dict]:
    """Next 48-row OA-MSE matrix for the simplified LEO residual channel."""

    specs = [dict(spec) for spec in _oa_mse_struct48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        spec["description"] = (
            "Simplified-LEO conservative rescue: prioritize old-query retention under leo_clear_weak/"
            "leo_low_elev_weak/leo_rain_weak while testing soft multi-prototype class boundaries."
            if category == "conservative"
            else "Simplified-LEO aggressive repair: add representation/boundary modules beyond tuning, "
            "using density, guarded rejection, void pressure, and residual adaptation to open old/unknown space."
        )
        spec["boundary_samples"] = max(4, int(spec["boundary_samples"]) - (1 if category == "conservative" else 0))
        spec["boundary_offset"] = 0.18 if category == "conservative" else float(spec["boundary_offset"])
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if int(spec["k_old"]) <= 5 else 3
        spec["multiproto_temperature"] = 0.09 if category == "conservative" else 0.075
        spec["anchor_density_gate"] = True
        if category == "conservative":
            spec["steps"] = max(24, min(36, int(spec["steps"]) - 8))
            spec["source_ce"] = min(0.46, float(spec["source_ce"]) + 0.10)
            spec["unknown_moat"] = max(0.18, float(spec["unknown_moat"]) - 0.04)
            spec["unknown_margin"] = max(0.46, float(spec["unknown_margin"]) - 0.04)
            spec["soft_proto"] = min(0.22, float(spec["soft_proto"]) + 0.02)
            spec["soft_proto_boundary"] = min(0.16, float(spec.get("soft_proto_boundary", 0.0)) + 0.02)
            spec["support_contrast"] = min(0.20, float(spec["support_contrast"]) + 0.02)
            spec["old_bridge"] = min(0.32, float(spec["old_bridge"]) + 0.06)
            spec["old_neighborhood"] = min(0.34, float(spec["old_neighborhood"]) + 0.06)
            spec["anchor_density_action"] = "uncertain"
            spec["anchor_density_quantile"] = max(0.03, float(spec.get("anchor_density_quantile", 0.03)))
            spec["anchor_density_margin_quantile"] = max(
                0.03, float(spec.get("anchor_density_margin_quantile", 0.03))
            )
            spec["adapter_kind"] = "low_rank"
            spec["old_unknown_acceptance_guard"] = True
            spec["guard_min_failures"] = 4
            spec["guard_min_margin"] = 0.04 + 0.02 * idx
            spec["old_retention_quantile"] = max(0.78, float(spec.get("old_retention_quantile", 0.78)))
        else:
            spec["steps"] = max(40, min(52, int(spec["steps"]) - 4))
            spec["source_ce"] = min(0.50, float(spec["source_ce"]) + 0.08)
            spec["unknown_moat"] = min(0.44, float(spec["unknown_moat"]) + 0.04)
            spec["unknown_margin"] = min(0.72, float(spec["unknown_margin"]) + 0.04)
            spec["soft_proto"] = min(0.28, float(spec["soft_proto"]) + 0.04)
            spec["soft_proto_boundary"] = min(0.24, float(spec.get("soft_proto_boundary", 0.0)) + 0.04)
            spec["support_contrast"] = min(0.30, float(spec["support_contrast"]) + 0.02)
            spec["old_bridge"] = min(0.34, float(spec["old_bridge"]) + 0.04)
            spec["old_neighborhood"] = min(0.36, float(spec["old_neighborhood"]) + 0.04)
            spec["anchor_density_action"] = "reject"
            spec["anchor_density_quantile"] = min(0.12, float(spec.get("anchor_density_quantile", 0.08)) + 0.02)
            spec["anchor_density_margin_quantile"] = min(
                0.10, float(spec.get("anchor_density_margin_quantile", 0.06)) + 0.02
            )
            spec["adapter_kind"] = "low_rank" if str(spec.get("slot")) == "D" else "residual_mlp"
            spec["void_background"] = 0.04 + 0.02 * (idx - 3)
            spec["void_gate"] = True
            spec["void_gate_min_score"] = 0.58 + 0.04 * (idx - 3)
            spec["void_gate_min_margin"] = -0.04 + 0.03 * (idx - 3)
            spec["old_unknown_acceptance_guard"] = True
            spec["guard_min_failures"] = 4
            spec["old_retention_quantile"] = max(0.74, float(spec.get("old_retention_quantile", 0.74)))
    return specs


def _oa_mse_retention48_stage_specs() -> list[dict]:
    """48-row simplified-channel matrix with retention-risk-balanced adapter selection."""

    specs = [dict(spec) for spec in _oa_mse_simplified48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        spec["description"] = (
            "Retention-risk-balanced OA-MSE conservative arm: use simplified LEO residual samples, "
            "soft multi-prototype boundaries, and alpha selection that penalizes support-only overfit."
            if category == "conservative"
            else "Retention-risk-balanced OA-MSE aggressive arm: keep simplified LEO residual and void/density "
            "modules, but choose adapter strength by old-retention floor plus pseudo-unknown risk."
        )
        spec["adapter_selection_policy"] = "retention_risk_balanced"
        spec["stage2_max_active_per_gpu"] = 2
        spec["steps"] = min(int(spec["steps"]), 40 if category == "conservative" else 48)
        spec["source_ce"] = min(0.52, float(spec["source_ce"]) + (0.04 if category == "conservative" else 0.02))
        spec["old_bridge"] = min(0.38, float(spec["old_bridge"]) + (0.04 if category == "conservative" else 0.02))
        spec["old_neighborhood"] = min(0.38, float(spec["old_neighborhood"]) + (0.04 if category == "conservative" else 0.02))
        spec["support_contrast"] = min(0.28, float(spec["support_contrast"]) + 0.02)
        spec["soft_proto"] = min(0.30, float(spec["soft_proto"]) + 0.02)
        spec["old_retention_quantile"] = max(0.82 if category == "conservative" else 0.78, float(spec.get("old_retention_quantile", 0.78)))
        if category == "conservative":
            spec["unknown_moat"] = max(0.16, float(spec["unknown_moat"]) - 0.02)
            spec["unknown_margin"] = max(0.44, float(spec["unknown_margin"]) - 0.02)
            spec["void_background"] = 0.0
            spec["void_gate"] = False
            spec["anchor_density_action"] = "uncertain"
            spec["guard_min_failures"] = max(4, int(spec.get("guard_min_failures", 4)))
        else:
            spec["unknown_moat"] = min(0.42, float(spec["unknown_moat"]) + 0.02)
            spec["unknown_margin"] = min(0.70, float(spec["unknown_margin"]) + 0.02)
            spec["void_background"] = min(0.10, float(spec.get("void_background", 0.04)))
            spec["void_gate"] = True
            spec["guard_min_failures"] = max(4, int(spec.get("guard_min_failures", 4)))
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 300
        spec["route_suffix"] = "retention_risk_balanced_selector_v1_after_next48ee"
        spec["evidence_ref"] = "next48ee_loss_down_support_fit_but_old_query_reject_vs_unknown_far_conflict"
        spec["score_table_required_groups"] = "old,new,unknown"
    return specs


def _oa_mse_supportret48_stage_specs() -> list[dict]:
    """48-row matrix using target-old support retention to constrain final gates."""

    specs = [dict(spec) for spec in _oa_mse_retention48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        spec["description"] = (
            "Support-retention guarded OA-MSE conservative arm: retain simplified LEO residual samples and "
            "retention-risk alpha selection, then constrain old surrogate-reject thresholds by target-old "
            "support evidence quantiles before final gate evaluation."
            if category == "conservative"
            else "Support-retention guarded OA-MSE aggressive arm: keep void/density pressure for unknown "
            "rejection, but prevent pseudo-unknown evidence thresholds from rejecting the allowed target-old "
            "support manifold."
        )
        spec["adapter_selection_policy"] = "retention_risk_balanced"
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = 0.08 if category == "conservative" else 0.16
        spec["support_retention_guard_slack"] = 0.04 if category == "conservative" else 0.07
        spec["old_surrogate_reject_relax"] = max(
            float(spec.get("old_surrogate_reject_relax", 0.0)),
            0.08 if category == "conservative" else 0.12,
        )
        spec["old_surrogate_margin"] = max(0.06, float(spec.get("old_surrogate_margin", 0.10)) - 0.02)
        spec["old_retention_quantile"] = max(0.88 if category == "conservative" else 0.82, float(spec.get("old_retention_quantile", 0.78)))
        spec["stage2_max_active_per_gpu"] = 2
        spec["steps"] = min(int(spec["steps"]), 40 if category == "conservative" else 48)
        if category == "conservative":
            spec["anchor_density_action"] = "uncertain"
            spec["void_gate"] = False
            spec["void_background"] = 0.0
            spec["unknown_moat"] = max(0.14, float(spec["unknown_moat"]) - 0.04)
            spec["unknown_margin"] = max(0.42, float(spec["unknown_margin"]) - 0.04)
        else:
            spec["anchor_density_action"] = "reject"
            spec["void_gate"] = True
            spec["void_background"] = min(0.12, max(0.06, float(spec.get("void_background", 0.06))))
            spec["unknown_moat"] = min(0.42, float(spec["unknown_moat"]) + 0.02)
            spec["unknown_margin"] = min(0.70, float(spec["unknown_margin"]) + 0.02)
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 400
        spec["route_suffix"] = "support_retention_guarded_final_gate_v1_after_next48ef"
        spec["evidence_ref"] = "next48ef_full_evidence_loss_pass_but_old_surrogate_evidence_reject_and_unknown_far_conflict"
        spec["score_table_required_groups"] = "old,new,unknown"
    return specs


def _oa_mse_twobranch48_stage_specs() -> list[dict]:
    """48-row matrix decoupling support retention from pseudo-background risk."""

    specs = [dict(spec) for spec in _oa_mse_supportret48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        strictness = idx % 3
        if str(spec.get("stage")) == "mse_subspace":
            spec["description"] = (
                "Two-branch OA-MSE target-old arm: keep support-retention old recovery, then apply a separate "
                "query-free pseudo-background risk veto only when target-old support evidence does not override it."
                if category == "conservative"
                else "Two-branch OA-MSE aggressive target-old arm: strengthen pseudo-background/ring negatives and "
                "reject accepted old rows through an independent background-risk branch while preserving strong "
                "target-old support evidence."
            )
        else:
            spec["description"] = (
                "Two-branch OA-MSE Stage2-C arm: keep support-retention old and seen-new recovery, then apply a "
                "separate query-free pseudo-background risk veto only when class support evidence does not override it."
                if category == "conservative"
                else "Two-branch OA-MSE aggressive Stage2-C arm: strengthen pseudo-background/ring negatives and reject "
                "accepted rows through an independent background-risk branch while preserving strong class support evidence."
            )
        spec["adapter_selection_policy"] = "retention_risk_balanced"
        spec["support_retention_guard"] = True
        spec["two_branch_background_guard"] = True
        spec["stage2_max_active_per_gpu"] = 2
        spec["route_suffix"] = "two_branch_background_guard_v1_after_next48eg"
        spec["evidence_ref"] = "next48eg_support_retention_improved_old_seen_but_unknown_far_worsened"
        spec["score_table_required_groups"] = "old,new,unknown"
        spec["old_surrogate_reject_relax"] = max(
            float(spec.get("old_surrogate_reject_relax", 0.0)),
            0.10 if category == "conservative" else 0.14,
        )
        spec["old_retention_quantile"] = max(
            0.88 if category == "conservative" else 0.84,
            float(spec.get("old_retention_quantile", 0.80)),
        )
        spec["support_retention_guard_quantile"] = 0.10 if category == "conservative" else 0.18
        spec["support_retention_guard_slack"] = 0.04 if category == "conservative" else 0.06
        spec["two_branch_bg_min_score"] = (0.66, 0.63, 0.60)[strictness] if category == "conservative" else (0.62, 0.58, 0.54)[strictness]
        spec["two_branch_bg_min_margin"] = (-0.01, -0.03, -0.05)[strictness] if category == "conservative" else (-0.04, -0.07, -0.10)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.02, 0.00, 0.02)[strictness] if category == "conservative" else (-0.04, -0.02, 0.00)[strictness]
        spec["two_branch_old_anchor_delta"] = (-0.03, -0.02, -0.01)[strictness]
        spec["two_branch_old_anchor_margin"] = (0.0, 0.01, 0.02)[strictness]
        spec["two_branch_seen_new_evidence_delta"] = (-0.02, 0.00, 0.02)[strictness]
        spec["two_branch_seen_new_anchor_delta"] = (-0.02, 0.00, 0.02)[strictness]
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 500
        if category == "conservative":
            spec["steps"] = min(int(spec["steps"]), 40)
            spec["void_gate"] = False
            spec["void_background"] = 0.0
            spec["anchor_density_action"] = "uncertain"
            spec["unknown_moat"] = max(0.14, float(spec["unknown_moat"]) - 0.02)
            spec["unknown_margin"] = max(0.42, float(spec["unknown_margin"]) - 0.02)
            spec["source_ce"] = min(0.54, float(spec["source_ce"]) + 0.02)
            spec["old_bridge"] = min(0.40, float(spec["old_bridge"]) + 0.02)
            spec["old_neighborhood"] = min(0.40, float(spec["old_neighborhood"]) + 0.02)
        else:
            spec["steps"] = min(int(spec["steps"]), 48)
            spec["void_gate"] = False
            spec["void_background"] = min(0.14, max(0.08, float(spec.get("void_background", 0.08))))
            spec["anchor_density_action"] = "reject"
            spec["unknown_moat"] = min(0.46, float(spec["unknown_moat"]) + 0.04)
            spec["unknown_margin"] = min(0.74, float(spec["unknown_margin"]) + 0.04)
            spec["boundary_samples"] = max(6, int(spec["boundary_samples"]) + 1)
            spec["pseudo_ring_override"] = 8
            spec["pseudo_halo_override"] = 8
            spec["adapter_kind"] = "residual_mlp" if str(spec.get("slot")) in {"E", "F"} else str(spec.get("adapter_kind", "low_rank"))
    return specs


def _oa_mse_reghead48_stage_specs() -> list[dict]:
    """48-row matrix adding explicit seen-new registration to the two-branch baseline."""

    specs = [dict(spec) for spec in _oa_mse_twobranch48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        strictness = idx % 3
        is_seen_new = str(spec.get("stage")) == "oa_mse_head"
        spec["support_retention_guard"] = True
        spec["two_branch_background_guard"] = True
        spec["seen_new_registration_override"] = True
        spec["stage2_max_active_per_gpu"] = 2
        spec["route_suffix"] = "seen_new_registration_override_v1_after_next48eh"
        spec["evidence_ref"] = "next48eh_loss_pass_but_seen_new_zero_explicit_registration_required"
        spec["score_table_required_groups"] = "old,new,unknown"
        if is_seen_new:
            spec["description"] = (
                "Conservative OA-MSE registration-head route: keep old support retention and query-free background veto, "
                "then explicitly accept seen-new support-registered classes when their support evidence beats old support."
                if category == "conservative"
                else "Aggressive OA-MSE registration-head route: combine residual/soft multi-prototype scoring with explicit "
                "seen-new support registration while keeping a separate query-free pseudo-background branch."
            )
        else:
            spec["description"] = (
                "Conservative OA-MSE registration-head old-retention route: keep target-old support retention and "
                "query-free background veto while leaving new-class enrollment inactive for this Stage2-B row."
                if category == "conservative"
                else "Aggressive OA-MSE registration-head old-retention route: strengthen target-old residual and "
                "pseudo-background separation while leaving new-class enrollment inactive for this Stage2-B row."
            )
        spec["seen_new_override_min_evidence_delta"] = (0.04, 0.02, 0.00)[strictness] if category == "conservative" else (0.00, -0.02, -0.04)[strictness]
        spec["seen_new_override_min_anchor_delta"] = (0.02, 0.00, -0.01)[strictness] if category == "conservative" else (0.00, -0.02, -0.04)[strictness]
        spec["seen_new_override_min_affinity_delta"] = (0.00, -0.01, -0.02)[strictness] if category == "conservative" else (-0.02, -0.04, -0.06)[strictness]
        spec["seen_new_override_min_residual_delta"] = (0.00, -0.01, -0.02)[strictness] if category == "conservative" else (-0.02, -0.04, -0.06)[strictness]
        spec["seen_new_override_min_score_margin"] = (-0.04, -0.08, -0.12)[strictness] if category == "conservative" else (-0.12, -0.18, -0.24)[strictness]
        spec["seen_new_override_min_seen_vs_old_evidence_margin"] = (0.04, 0.02, 0.00)[strictness] if category == "conservative" else (0.00, -0.02, -0.04)[strictness]
        spec["seen_new_override_max_background_score"] = (0.68, 0.70, 0.72)[strictness] if category == "conservative" else (0.70, 0.74, 0.78)[strictness]
        spec["seen_new_override_max_background_margin"] = (0.03, 0.05, 0.07)[strictness] if category == "conservative" else (0.05, 0.08, 0.11)[strictness]
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 600
        if is_seen_new:
            spec["k_new"] = max(5, int(spec.get("k_new", 0)))
            spec["soft_proto"] = min(0.34 if category == "conservative" else 0.42, float(spec.get("soft_proto", 0.0)) + (0.06 if category == "conservative" else 0.12))
            spec["soft_proto_boundary"] = min(0.30 if category == "conservative" else 0.40, float(spec.get("soft_proto_boundary", 0.0)) + (0.04 if category == "conservative" else 0.10))
            spec["multiproto_score"] = True
            spec["multiproto_topk"] = 2 if category == "conservative" else 3
            spec["multiproto_temperature"] = (0.08, 0.10, 0.12)[strictness] if category == "conservative" else (0.06, 0.08, 0.10)[strictness]
            spec["multiproto_score_weight"] = (0.85, 1.00, 1.15)[strictness] if category == "conservative" else (1.10, 1.30, 1.50)[strictness]
            spec["support_contrast"] = min(0.30 if category == "conservative" else 0.40, float(spec.get("support_contrast", 0.0)) + (0.04 if category == "conservative" else 0.10))
            spec["steps"] = min(44 if category == "conservative" else 56, int(spec.get("steps", 40)) + (2 if category == "conservative" else 8))
        if category == "aggressive":
            spec["adapter_kind"] = "residual_mlp" if str(spec.get("slot")) in {"E", "F"} else str(spec.get("adapter_kind", "low_rank"))
            spec["pseudo_ring_override"] = max(10, int(spec.get("pseudo_ring_override", 0) or 0))
            spec["pseudo_halo_override"] = max(10, int(spec.get("pseudo_halo_override", 0) or 0))
            spec["boundary_samples"] = max(8, int(spec.get("boundary_samples", 0)) + 2)
            spec["unknown_moat"] = min(0.52, float(spec.get("unknown_moat", 0.0)) + 0.04)
            spec["unknown_margin"] = min(0.78, float(spec.get("unknown_margin", 0.0)) + 0.04)
        else:
            spec["unknown_moat"] = max(0.12, float(spec.get("unknown_moat", 0.0)) - 0.02)
            spec["unknown_margin"] = max(0.40, float(spec.get("unknown_margin", 0.0)) - 0.02)
            spec["anchor_density_action"] = "uncertain"
    return specs


def _oa_mse_geom48_stage_specs() -> list[dict]:
    """48-row geometry-registration matrix after next48ei failed seen-new correctness."""

    specs = [dict(spec) for spec in _oa_mse_reghead48_stage_specs()]
    for idx, spec in enumerate(specs):
        category = str(spec.get("category", "conservative"))
        strictness = idx % 3
        is_seen_new = str(spec.get("stage")) == "oa_mse_head"
        spec["stage2_max_active_per_gpu"] = 2
        spec["route_suffix"] = "support_center_geometry_registration_v1_after_next48ei"
        spec["evidence_ref"] = "next48ei_override_fired_1990_but_seen_new_correct_2_of_3200"
        spec["score_table_required_groups"] = "old,new,unknown"
        spec["support_center_ce"] = (0.18, 0.22, 0.26)[strictness] if category == "conservative" else (0.34, 0.42, 0.50)[strictness]
        spec["support_center_temperature"] = (0.12, 0.10, 0.08)[strictness] if category == "conservative" else (0.08, 0.06, 0.05)[strictness]
        spec["support_center_margin"] = (0.08, 0.10, 0.12)[strictness] if category == "conservative" else (0.14, 0.18, 0.22)[strictness]
        spec["description"] = (
            "Geometry-registration OA-MSE conservative arm: add leave-one-out support-center CE and margin loss "
            "so target old/seen-new support forms compact class centers before the background branch is applied."
            if category == "conservative"
            else "Geometry-registration OA-MSE aggressive arm: use stronger support-center margins, residual feature "
            "repair, halo/ring pseudo-background, and explicit seen-new registration to force class boundary formation."
        )
        spec["adapter_selection_policy"] = "retention_risk_balanced"
        spec["support_retention_guard"] = True
        spec["two_branch_background_guard"] = True
        spec["seen_new_registration_override"] = True
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if category == "conservative" else 3
        spec["soft_proto"] = min(0.36 if category == "conservative" else 0.46, float(spec.get("soft_proto", 0.0)) + (0.04 if category == "conservative" else 0.08))
        spec["soft_proto_boundary"] = min(0.32 if category == "conservative" else 0.44, float(spec.get("soft_proto_boundary", 0.0)) + (0.04 if category == "conservative" else 0.08))
        spec["support_contrast"] = min(0.32 if category == "conservative" else 0.46, float(spec.get("support_contrast", 0.0)) + (0.02 if category == "conservative" else 0.06))
        spec["source_ce"] = min(0.58 if category == "conservative" else 0.62, float(spec.get("source_ce", 0.0)) + 0.04)
        spec["old_bridge"] = min(0.44 if category == "conservative" else 0.46, float(spec.get("old_bridge", 0.0)) + 0.04)
        spec["old_neighborhood"] = min(0.44 if category == "conservative" else 0.48, float(spec.get("old_neighborhood", 0.0)) + 0.04)
        spec["old_retention_quantile"] = max(0.90 if category == "conservative" else 0.84, float(spec.get("old_retention_quantile", 0.82)))
        spec["support_retention_guard_quantile"] = 0.08 if category == "conservative" else 0.16
        spec["support_retention_guard_slack"] = 0.03 if category == "conservative" else 0.06
        spec["two_branch_bg_min_score"] = (0.68, 0.66, 0.64)[strictness] if category == "conservative" else (0.62, 0.58, 0.54)[strictness]
        spec["two_branch_bg_min_margin"] = (-0.01, -0.02, -0.04)[strictness] if category == "conservative" else (-0.05, -0.08, -0.12)[strictness]
        spec["adapter_kind"] = "low_rank" if category == "conservative" else "residual_mlp"
        spec["steps"] = min(48 if category == "conservative" else 64, int(spec.get("steps", 40)) + (4 if category == "conservative" else 10))
        if is_seen_new:
            spec["k_new"] = max(5, int(spec.get("k_new", 0)))
            spec["seen_new_override_min_evidence_delta"] = (0.04, 0.02, 0.00)[strictness] if category == "conservative" else (0.00, -0.04, -0.08)[strictness]
            spec["seen_new_override_min_seen_vs_old_evidence_margin"] = (0.04, 0.02, 0.00)[strictness] if category == "conservative" else (0.00, -0.03, -0.06)[strictness]
            spec["seen_new_override_max_background_score"] = (0.66, 0.68, 0.70)[strictness] if category == "conservative" else (0.72, 0.76, 0.80)[strictness]
            spec["seen_new_override_max_background_margin"] = (0.03, 0.05, 0.07)[strictness] if category == "conservative" else (0.06, 0.10, 0.14)[strictness]
        if category == "aggressive":
            spec["unknown_moat"] = min(0.56, float(spec.get("unknown_moat", 0.0)) + 0.04)
            spec["unknown_margin"] = min(0.82, float(spec.get("unknown_margin", 0.0)) + 0.04)
            spec["pseudo_ring_override"] = max(12, int(spec.get("pseudo_ring_override", 0) or 0))
            spec["pseudo_halo_override"] = max(12, int(spec.get("pseudo_halo_override", 0) or 0))
            spec["boundary_samples"] = max(10, int(spec.get("boundary_samples", 0)))
            spec["anchor_density_action"] = "reject"
        else:
            spec["unknown_moat"] = max(0.10, float(spec.get("unknown_moat", 0.0)) - 0.02)
            spec["unknown_margin"] = max(0.38, float(spec.get("unknown_margin", 0.0)) - 0.02)
            spec["anchor_density_action"] = "uncertain"
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["query_cap_source"] = "next48ej_receiver_specific_manytx_unknown_query_availability_min30"
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 700
    return specs


def _oa_mse_triage48_stage_specs() -> list[dict]:
    """48-row split-objective triage matrix after next48ek exposed three failures."""

    specs = [dict(spec) for spec in _oa_mse_geom48_stage_specs()]
    for idx, spec in enumerate(specs):
        slot = str(spec.get("slot"))
        strictness = idx % 3
        is_seen_new = str(spec.get("stage")) == "oa_mse_head"
        if slot in {"A", "B"}:
            spec["category"] = "old_retention"
            spec["route_suffix"] = "old_retention_triage_after_next48ek"
            spec["evidence_ref"] = "next48ek_old_max_0p672_but_old_reject_rate_mean_0p429"
            spec["description"] = (
                "Old-retention triage arm: loosen query-free background rejection and strengthen old support "
                "anchoring to test whether target-old accuracy can recover before unknown gates are tightened."
            )
            spec["unknown_moat"] = (0.06, 0.08, 0.10)[strictness]
            spec["unknown_margin"] = (0.30, 0.34, 0.38)[strictness]
            spec["source_ce"] = max(0.58, float(spec.get("source_ce", 0.0)))
            spec["old_bridge"] = max(0.46, float(spec.get("old_bridge", 0.0)))
            spec["old_neighborhood"] = max(0.46, float(spec.get("old_neighborhood", 0.0)))
            spec["support_center_ce"] = max(0.22, float(spec.get("support_center_ce", 0.0)))
            spec["support_center_margin"] = (0.06, 0.08, 0.10)[strictness]
            spec["support_retention_guard_quantile"] = (0.04, 0.06, 0.08)[strictness]
            spec["support_retention_guard_slack"] = (0.06, 0.08, 0.10)[strictness]
            spec["anchor_density_action"] = "uncertain"
            spec["anchor_density_quantile"] = (0.02, 0.03, 0.04)[strictness]
            spec["anchor_density_margin_quantile"] = (0.02, 0.03, 0.04)[strictness]
            spec["void_background"] = 0.0
            spec["void_gate"] = False
            spec["two_branch_bg_min_score"] = (0.72, 0.70, 0.68)[strictness]
            spec["two_branch_bg_min_margin"] = (0.04, 0.02, 0.00)[strictness]
            spec["two_branch_old_support_evidence_delta"] = (-0.08, -0.06, -0.04)[strictness]
            spec["two_branch_old_anchor_delta"] = (-0.06, -0.05, -0.04)[strictness]
            spec["old_retention_quantile"] = (0.94, 0.92, 0.90)[strictness]
            spec["old_anchor_override_min_quality"] = (0.50, 0.54, 0.58)[strictness]
            spec["adapter_kind"] = "low_rank"
            spec["steps"] = min(56, max(36, int(spec.get("steps", 40))))
        elif slot in {"C", "D"}:
            spec["category"] = "unknown_boundary"
            spec["route_suffix"] = "unknown_boundary_triage_after_next48ek"
            spec["evidence_ref"] = "next48ek_unknown_far_mean_0p490_and_unknown_to_old_1410"
            spec["description"] = (
                "Unknown-boundary triage arm: keep a strong pseudo-background and density gate, but require old "
                "support evidence before accepting old-like unknowns so unknown-to-old false accepts are isolated."
            )
            spec["unknown_moat"] = (0.42, 0.50, 0.58)[strictness]
            spec["unknown_margin"] = (0.70, 0.82, 0.92)[strictness]
            spec["boundary_samples"] = max(12, int(spec.get("boundary_samples", 0)))
            spec["pseudo_halo_override"] = max(14, int(spec.get("pseudo_halo_override", 0) or 0))
            spec["pseudo_ring_override"] = max(14, int(spec.get("pseudo_ring_override", 0) or 0))
            spec["anchor_density_action"] = "reject"
            spec["anchor_density_quantile"] = (0.10, 0.12, 0.14)[strictness]
            spec["anchor_density_margin_quantile"] = (0.08, 0.10, 0.12)[strictness]
            spec["void_background"] = (0.08, 0.10, 0.12)[strictness]
            spec["void_gate"] = True
            spec["void_gate_min_score"] = (0.58, 0.62, 0.66)[strictness]
            spec["void_gate_min_margin"] = (-0.04, 0.00, 0.04)[strictness]
            spec["old_unknown_acceptance_guard"] = True
            spec["guard_min_old_support_evidence_delta"] = (-0.02, 0.00, 0.02)[strictness]
            spec["guard_min_old_surrogate_reject_delta"] = (0.02, 0.04, 0.06)[strictness]
            spec["guard_min_best_old_score"] = (-2.0, -1.5, -1.0)[strictness]
            spec["guard_min_margin"] = (0.08, 0.12, 0.16)[strictness]
            spec["guard_min_failures"] = 4
            spec["adapter_kind"] = "residual_mlp" if slot == "D" else "low_rank"
            spec["steps"] = min(72, max(48, int(spec.get("steps", 40)) + 4))
        else:
            spec["category"] = "seen_new_rescue"
            spec["route_suffix"] = "seen_new_rescue_triage_after_next48ek"
            spec["evidence_ref"] = "next48ek_seen_new_acc_zero_and_new_rejected_1426"
            spec["description"] = (
                "Seen-new rescue triage arm: make registered seen-new support an explicit competing prototype "
                "assignment path before the background branch can reject it, while retaining unknown query eval-only."
            )
            spec["stage"] = "oa_mse_head"
            spec["eval_protocol"] = "sfe"
            spec["k_old"] = max(10, int(spec.get("k_old", 0)))
            spec["k_new"] = max(10, int(spec.get("k_new", 0)))
            spec["soft_proto"] = (0.42, 0.50, 0.58)[strictness]
            spec["soft_proto_boundary"] = (0.30, 0.38, 0.46)[strictness]
            spec["soft_proto_boundary_margin"] = (0.12, 0.16, 0.20)[strictness]
            spec["multiproto_score"] = True
            spec["multiproto_topk"] = 3
            spec["multiproto_temperature"] = (0.06, 0.08, 0.10)[strictness]
            spec["multiproto_score_weight"] = (1.6, 1.9, 2.2)[strictness]
            spec["seen_new_registration_override"] = True
            spec["seen_new_override_min_evidence_delta"] = (-0.16, -0.22, -0.28)[strictness]
            spec["seen_new_override_min_anchor_delta"] = (-0.08, -0.12, -0.16)[strictness]
            spec["seen_new_override_min_affinity_delta"] = (-0.08, -0.12, -0.16)[strictness]
            spec["seen_new_override_min_residual_delta"] = (-0.08, -0.12, -0.16)[strictness]
            spec["seen_new_override_min_score_margin"] = (-0.30, -0.38, -0.46)[strictness]
            spec["seen_new_override_min_seen_vs_old_evidence_margin"] = (-0.10, -0.16, -0.22)[strictness]
            spec["seen_new_override_max_background_score"] = (0.84, 0.90, 0.96)[strictness]
            spec["seen_new_override_max_background_margin"] = (0.14, 0.20, 0.26)[strictness]
            spec["two_branch_seen_new_evidence_delta"] = (-0.08, -0.12, -0.16)[strictness]
            spec["two_branch_seen_new_anchor_delta"] = (-0.06, -0.10, -0.14)[strictness]
            spec["anchor_density_action"] = "uncertain"
            spec["unknown_moat"] = (0.18, 0.24, 0.30)[strictness]
            spec["unknown_margin"] = (0.44, 0.52, 0.60)[strictness]
            spec["adapter_kind"] = "residual_mlp"
            spec["steps"] = min(72, max(56, int(spec.get("steps", 40)) + 8))
        if is_seen_new or spec["category"] == "seen_new_rescue":
            spec["k_new"] = max(5, int(spec.get("k_new", 0)))
        spec["stage2_max_active_per_gpu"] = 2
        spec["support_retention_guard"] = True
        spec["two_branch_background_guard"] = True
        spec["multiproto_score"] = True
        spec["score_table_required_groups"] = "old,new,unknown"
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["query_cap_source"] = "next48ej_receiver_specific_manytx_unknown_query_availability_min30"
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 800
    return specs


def _oa_mse_looo48_stage_specs() -> list[dict]:
    """48-row source leave-one-old-out meta-unknown boundary after next48el."""

    specs = [dict(spec) for spec in _oa_mse_triage48_stage_specs()]
    for idx, spec in enumerate(specs):
        slot = str(spec.get("slot"))
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "source_leave_one_old_out_meta_unknown_after_next48el"
        spec["evidence_ref"] = (
            "next48el_loss_pass_but_old_unknown_not_separable;"
            "unknown_boundary_arm_all_reject_and_old_retention_arm_unknown_to_old_high"
        )
        spec["description"] = (
            "Source leave-one-old-out meta-unknown boundary: each source-old class is trained as known for its "
            "own prototype but as unknown relative to all other class prototypes, so OA-MSE learns inter-class "
            "space for unknown rejection without using target unknown query labels."
        )
        spec["source_looo_unknown_weight"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.22, 0.28, 0.34)[strictness]
        spec["source_looo_unknown_margin"] = (0.28, 0.34, 0.40)[strictness] if not aggressive else (0.42, 0.50, 0.58)[strictness]
        spec["source_looo_interclass_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["source_looo_max_samples_per_class"] = 18 if not aggressive else 32
        spec["source_ce"] = max(0.58 if not aggressive else 0.70, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.44 if not aggressive else 0.52, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.42 if not aggressive else 0.50, float(spec.get("old_neighborhood", 0.0)))
        spec["old_surrogate_margin"] = max(0.12 if not aggressive else 0.18, float(spec.get("old_surrogate_margin", 0.0)))
        spec["old_surrogate_margin_weight"] = max(0.10 if not aggressive else 0.16, float(spec.get("old_surrogate_margin_weight", 0.0)))
        spec["support_center_ce"] = max(0.20 if not aggressive else 0.28, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.30 if slot in {"E", "F"} else 0.18, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.18 if slot in {"E", "F"} else 0.10, float(spec.get("soft_proto_boundary", 0.0)))
        spec["unknown_moat"] = max((0.16, 0.20, 0.24)[strictness] if not aggressive else (0.30, 0.38, 0.46)[strictness], float(spec.get("unknown_moat", 0.0)))
        spec["unknown_margin"] = max((0.42, 0.48, 0.54)[strictness] if not aggressive else (0.58, 0.68, 0.78)[strictness], float(spec.get("unknown_margin", 0.0)))
        spec["boundary_samples"] = max(8 if not aggressive else 12, int(spec.get("boundary_samples", 0)))
        spec["pseudo_halo_override"] = max(10 if not aggressive else 16, int(spec.get("pseudo_halo_override", 0) or 0))
        spec["pseudo_ring_override"] = max(10 if not aggressive else 16, int(spec.get("pseudo_ring_override", 0) or 0))
        spec["void_background"] = max(0.04 if not aggressive else 0.10, float(spec.get("void_background", 0.0)))
        spec["void_gate"] = bool(aggressive or spec.get("void_gate", False))
        spec["two_branch_background_guard"] = True
        spec["support_retention_guard"] = True
        spec["multiproto_score"] = True
        spec["anchor_density_action"] = "uncertain" if not aggressive else "reject"
        spec["anchor_density_quantile"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.07, 0.09, 0.11)[strictness]
        spec["anchor_density_margin_quantile"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.06, 0.08, 0.10)[strictness]
        spec["guard_min_failures"] = 4 if not aggressive else 5
        spec["old_unknown_acceptance_guard"] = True
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"D", "E", "F"} else "low_rank")
        spec["steps"] = min(80, max(56 if not aggressive else 64, int(spec.get("steps", 40)) + (4 if not aggressive else 8)))
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1200
    return specs


def _oa_mse_constrain48_stage_specs() -> list[dict]:
    """48-row constrained OA-MSE after next48em exposed all-reject pressure."""

    specs = [dict(spec) for spec in _oa_mse_looo48_stage_specs()]
    for idx, spec in enumerate(specs):
        slot = str(spec.get("slot"))
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "known_coverage_constrained_oamse_after_next48em"
        spec["evidence_ref"] = (
            "next48em_loss_pass_unknown_rejection_up_but_old_mean_0p183_seen_new_0p032;"
            "gate_reasons_two_branch_void_density_rejected_old"
        )
        spec["description"] = (
            "Constrained OA-MSE conservative arm: first satisfy source/old/support known coverage with a hard "
            "adapter-selection feasibility gate, then apply low-pressure query-free unknown separation."
            if not aggressive
            else "Constrained OA-MSE aggressive arm: keep LOOO and pseudo-background pressure, but require known "
            "coverage feasibility and support-evidence overrides before any background gate can reject known rows."
        )
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["known_coverage_weight"] = (0.46, 0.54, 0.62)[strictness] if not aggressive else (0.34, 0.42, 0.50)[strictness]
        spec["known_coverage_margin"] = (0.10, 0.12, 0.14)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["known_coverage_min_affinity"] = (0.30, 0.34, 0.38)[strictness] if not aggressive else (0.26, 0.30, 0.34)[strictness]
        spec["known_coverage_max_samples"] = 320 if not aggressive else 256
        spec["source_looo_unknown_weight"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.08, 0.11, 0.14)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.28, 0.32)[strictness] if not aggressive else (0.30, 0.36, 0.42)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.06, 0.08, 0.10)[strictness]
        spec["source_looo_max_samples_per_class"] = 14 if not aggressive else 20
        spec["source_ce"] = max(0.68 if not aggressive else 0.62, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.58 if not aggressive else 0.50, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.56 if not aggressive else 0.48, float(spec.get("old_neighborhood", 0.0)))
        spec["old_surrogate_margin_weight"] = max(0.08 if not aggressive else 0.10, float(spec.get("old_surrogate_margin_weight", 0.0)))
        spec["old_surrogate_margin"] = max(0.08 if not aggressive else 0.12, float(spec.get("old_surrogate_margin", 0.0)))
        spec["old_retention_quantile"] = (0.96, 0.94, 0.92)[strictness] if not aggressive else (0.92, 0.90, 0.88)[strictness]
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.03, 0.04, 0.05)[strictness] if not aggressive else (0.06, 0.08, 0.10)[strictness]
        spec["support_retention_guard_slack"] = (0.08, 0.09, 0.10)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.76, 0.72, 0.68)[strictness] if not aggressive else (0.70, 0.66, 0.62)[strictness]
        spec["two_branch_bg_min_margin"] = (0.08, 0.04, 0.00)[strictness] if not aggressive else (0.02, -0.02, -0.06)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.14, -0.12, -0.10)[strictness] if not aggressive else (-0.10, -0.08, -0.06)[strictness]
        spec["two_branch_old_anchor_delta"] = (-0.10, -0.08, -0.06)[strictness]
        spec["two_branch_old_anchor_margin"] = (-0.01, 0.00, 0.01)[strictness]
        spec["two_branch_seen_new_evidence_delta"] = (-0.12, -0.10, -0.08)[strictness] if not aggressive else (-0.10, -0.08, -0.06)[strictness]
        spec["two_branch_seen_new_anchor_delta"] = (-0.10, -0.08, -0.06)[strictness]
        spec["anchor_density_action"] = "uncertain"
        spec["anchor_density_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.03, 0.04, 0.05)[strictness]
        spec["anchor_density_margin_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.03, 0.04, 0.05)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["void_gate"] = bool(aggressive)
        spec["void_gate_min_score"] = (0.66, 0.70, 0.74)[strictness] if aggressive else 0.80
        spec["void_gate_min_margin"] = (0.02, 0.04, 0.06)[strictness] if aggressive else 0.20
        spec["unknown_moat"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.20, 0.26, 0.32)[strictness]
        spec["unknown_margin"] = (0.34, 0.40, 0.46)[strictness] if not aggressive else (0.46, 0.54, 0.62)[strictness]
        spec["boundary_samples"] = 6 if not aggressive else 10
        spec["pseudo_halo_override"] = 6 if not aggressive else 10
        spec["pseudo_ring_override"] = 6 if not aggressive else 10
        spec["support_center_ce"] = max(0.24 if not aggressive else 0.30, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.34 if slot in {"E", "F"} else 0.24, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.20 if slot in {"E", "F"} else 0.14, float(spec.get("soft_proto_boundary", 0.0)))
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if not aggressive else 3
        spec["multiproto_score_weight"] = (0.90, 1.05, 1.20)[strictness] if not aggressive else (1.10, 1.30, 1.50)[strictness]
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 5 if not aggressive else 4
        spec["guard_min_old_support_evidence_delta"] = (-0.12, -0.08, -0.04)[strictness]
        spec["guard_min_old_surrogate_reject_delta"] = (-0.04, 0.00, 0.04)[strictness]
        spec["guard_min_best_old_score"] = (-2.5, -2.0, -1.5)[strictness]
        spec["guard_min_margin"] = (0.02, 0.06, 0.10)[strictness]
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["steps"] = min(72, max(52 if not aggressive else 60, int(spec.get("steps", 40))))
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1400
    return specs


def _oa_mse_envelope48_stage_specs() -> list[dict]:
    """48-row source/support class-envelope OA-MSE after next48en exposed unknown leakage."""

    specs = [dict(spec) for spec in _oa_mse_constrain48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "source_support_class_envelope_after_next48en"
        spec["evidence_ref"] = (
            "next48en_loss_pass_old_mean_0p435_old_max_0p678_but_unknown_rejection_mean_0p244;"
            "training_proxy_converged_but_accepted_unknown_leakage_remained_high"
        )
        spec["description"] = (
            "Envelope OA-MSE conservative arm: preserve next48en known-coverage gains, then reject accepted rows "
            "outside a source+support class envelope fitted without unknown-query thresholds."
            if not aggressive
            else "Envelope OA-MSE aggressive arm: combine class envelope rejection with stronger source leave-one-old-out "
            "and pseudo-background pressure to attack unknown false accepts while tracking old retention loss."
        )
        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_residual_quantile"] = (0.99, 0.98, 0.97)[strictness] if not aggressive else (0.97, 0.95, 0.93)[strictness]
        spec["class_envelope_score_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_margin_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_evidence_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_residual_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_score_slack"] = (0.18, 0.14, 0.10)[strictness] if not aggressive else (0.10, 0.06, 0.03)[strictness]
        spec["class_envelope_margin_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_min_failures"] = 2 if not aggressive else 1
        spec["class_envelope_gate_action"] = "reject"
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["known_coverage_weight"] = (0.50, 0.58, 0.66)[strictness] if not aggressive else (0.38, 0.46, 0.54)[strictness]
        spec["source_ce"] = max(0.72 if not aggressive else 0.64, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.62 if not aggressive else 0.54, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.60 if not aggressive else 0.50, float(spec.get("old_neighborhood", 0.0)))
        spec["source_looo_unknown_weight"] = (0.03, 0.05, 0.07)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["source_looo_unknown_margin"] = (0.22, 0.26, 0.30)[strictness] if not aggressive else (0.36, 0.44, 0.52)[strictness]
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]
        spec["unknown_margin"] = (0.30, 0.36, 0.42)[strictness] if not aggressive else (0.54, 0.64, 0.74)[strictness]
        spec["two_branch_bg_min_score"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.68, 0.64, 0.60)[strictness]
        spec["two_branch_bg_min_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.06, 0.09, 0.12)[strictness]
        spec["void_gate"] = bool(aggressive)
        spec["void_gate_min_score"] = (0.78, 0.80, 0.82)[strictness] if not aggressive else (0.64, 0.68, 0.72)[strictness]
        spec["void_gate_min_margin"] = (0.16, 0.18, 0.20)[strictness] if not aggressive else (0.00, 0.03, 0.06)[strictness]
        spec["anchor_density_action"] = "uncertain" if not aggressive else "reject"
        spec["anchor_density_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["anchor_density_margin_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["guard_min_failures"] = 5 if not aggressive else 4
        spec["support_center_ce"] = max(0.22 if not aggressive else 0.32, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.30 if not aggressive else 0.40, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.16 if not aggressive else 0.24, float(spec.get("soft_proto_boundary", 0.0)))
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if not aggressive else 3
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1600
    return specs


def _oa_mse_rescue48_stage_specs() -> list[dict]:
    """48-row known-retention rescue matrix after envelope gates over-rejected known rows."""

    specs = [dict(spec) for spec in _oa_mse_envelope48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        if slot in {"A", "D"}:
            spec["stage"] = "oa_mse_head"
            spec["eval_protocol"] = "sfe"
            spec["k_new"] = max(5, int(spec.get("k_old", 5)))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "post_reject_retention_rescue_after_next48eo"
        spec["evidence_ref"] = (
            "next48eo_loss_pass_unknown_rejection_mean_0p431_but_old_mean_0p316_seen_new_mean_0p104;"
            "class_envelope_rejected_old_1442_new_464_unknown_408"
        )
        if str(spec.get("stage")) == "mse_subspace":
            spec["description"] = (
                "Rescue OA-MSE conservative Stage2-B arm: keep simplified LEO residual and source/support envelope, "
                "then recover only high-evidence target-old rows after reject gates when pseudo-background risk is low."
                if not aggressive
                else "Rescue OA-MSE aggressive Stage2-B arm: bind source leave-one-old-out unknown boundary pressure "
                "to a post-reject target-old retention rescue so non-old rejection cannot be achieved only by killing old rows."
            )
        else:
            spec["description"] = (
                "Rescue OA-MSE conservative Stage2-C arm: keep simplified LEO residual and source/support envelope, "
                "then recover only high-evidence target-old or seen-new rows after reject gates when pseudo-background "
                "risk is low."
                if not aggressive
                else "Rescue OA-MSE aggressive Stage2-C arm: bind source leave-one-old-out unknown boundary pressure to a "
                "post-reject known-retention rescue so unknown rejection cannot be achieved only by killing old/seen-new."
            )
        spec["retention_rescue_gate"] = True
        spec["class_envelope_gate"] = True
        spec["class_envelope_gate_action"] = "reject"
        spec["class_envelope_min_failures"] = 2 if not aggressive else 1
        spec["anchor_density_action"] = "uncertain" if not aggressive else "reject"
        spec["two_branch_background_guard"] = True
        spec["void_gate"] = bool(aggressive)
        spec["support_retention_guard"] = True
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["known_coverage_weight"] = max(0.62 if not aggressive else 0.44, float(spec.get("known_coverage_weight", 0.0)))
        spec["known_coverage_margin"] = (0.10, 0.12, 0.14)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["source_ce"] = max(0.76 if not aggressive else 0.66, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.66 if not aggressive else 0.56, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.62 if not aggressive else 0.52, float(spec.get("old_neighborhood", 0.0)))
        spec["support_center_ce"] = max(0.28 if not aggressive else 0.36, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.34 if not aggressive else 0.44, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.18 if not aggressive else 0.28, float(spec.get("soft_proto_boundary", 0.0)))
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if not aggressive else 3
        spec["multiproto_score_weight"] = (0.95, 1.05, 1.15)[strictness] if not aggressive else (1.15, 1.35, 1.55)[strictness]
        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.12, 0.16, 0.20)[strictness]
        spec["source_looo_unknown_margin"] = (0.20, 0.24, 0.28)[strictness] if not aggressive else (0.38, 0.48, 0.58)[strictness]
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.28, 0.36, 0.44)[strictness]
        spec["unknown_margin"] = (0.30, 0.36, 0.42)[strictness] if not aggressive else (0.58, 0.68, 0.78)[strictness]
        spec["two_branch_bg_min_score"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.66, 0.62, 0.58)[strictness]
        spec["two_branch_bg_min_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (-0.02, -0.06, -0.10)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.08, 0.11, 0.14)[strictness]
        spec["void_gate_min_score"] = 0.82 if not aggressive else (0.62, 0.66, 0.70)[strictness]
        spec["void_gate_min_margin"] = 0.20 if not aggressive else (-0.02, 0.02, 0.06)[strictness]
        spec["retention_rescue_old_min_evidence_delta"] = (0.06, 0.04, 0.02)[strictness] if not aggressive else (0.02, 0.00, -0.02)[strictness]
        spec["retention_rescue_old_min_anchor_delta"] = (0.00, -0.01, -0.02)[strictness] if not aggressive else (-0.02, -0.04, -0.06)[strictness]
        spec["retention_rescue_old_min_anchor_margin"] = (0.02, 0.01, 0.00)[strictness] if not aggressive else (0.00, -0.01, -0.02)[strictness]
        spec["retention_rescue_old_min_score_margin"] = (0.08, 0.04, 0.00)[strictness] if not aggressive else (0.02, -0.02, -0.06)[strictness]
        spec["retention_rescue_seen_new_min_evidence_delta"] = (0.06, 0.04, 0.02)[strictness] if not aggressive else (0.02, 0.00, -0.02)[strictness]
        spec["retention_rescue_seen_new_min_anchor_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.02, -0.04)[strictness]
        spec["retention_rescue_seen_new_min_score_margin"] = (0.04, 0.00, -0.04)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["retention_rescue_max_background_score"] = (0.64, 0.68, 0.72)[strictness] if not aggressive else (0.60, 0.64, 0.68)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.00, 0.04, 0.08)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1800
    return specs


def _oa_mse_prereject48_stage_specs() -> list[dict]:
    """48-row pre-reject defer/arbitration matrix after post-reject rescue did not fire."""

    specs = [dict(spec) for spec in _oa_mse_rescue48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "pre_reject_defer_arbitration_after_next48ep"
        spec["evidence_ref"] = (
            "next48ep_loss_pass_but_retention_rescue_eligible_3_accept_0;"
            "old_mean_0p295_seen_new_mean_0p096_unknown_far_mean_0p572"
        )
        spec["description"] = (
            "Pre-reject arbitration conservative arm: convert source/support envelope failures into uncertain rows, "
            "then accept target-old or seen-new only when support evidence beats pseudo-background risk before hard rejection."
            if not aggressive
            else "Pre-reject arbitration aggressive arm: keep stronger pseudo-background pressure and residual-MLP repair, "
            "but force known-vs-background arbitration before irreversible reject so unknown rejection cannot be won by "
            "discarding old/seen-new rows."
        )
        spec["pre_reject_defer_arbitration"] = True
        spec["class_envelope_gate"] = True
        spec["class_envelope_gate_action"] = "uncertain"
        spec["anchor_density_action"] = "uncertain"
        spec["retention_rescue_gate"] = False
        spec["two_branch_background_guard"] = True
        spec["support_retention_guard"] = True
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["known_coverage_weight"] = max(0.68 if not aggressive else 0.48, float(spec.get("known_coverage_weight", 0.0)))
        spec["known_coverage_margin"] = (0.12, 0.14, 0.16)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["known_coverage_min_affinity"] = (0.34, 0.38, 0.42)[strictness] if not aggressive else (0.28, 0.32, 0.36)[strictness]
        spec["source_ce"] = max(0.80 if not aggressive else 0.70, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.70 if not aggressive else 0.60, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.66 if not aggressive else 0.56, float(spec.get("old_neighborhood", 0.0)))
        spec["support_center_ce"] = max(0.32 if not aggressive else 0.40, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.38 if not aggressive else 0.50, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.22 if not aggressive else 0.32, float(spec.get("soft_proto_boundary", 0.0)))
        spec["soft_proto_boundary_margin"] = (0.12, 0.15, 0.18)[strictness] if not aggressive else (0.18, 0.22, 0.26)[strictness]
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if not aggressive else 3
        spec["multiproto_temperature"] = (0.10, 0.08, 0.06)[strictness] if not aggressive else (0.08, 0.06, 0.05)[strictness]
        spec["multiproto_score_weight"] = (1.05, 1.20, 1.35)[strictness] if not aggressive else (1.35, 1.60, 1.85)[strictness]
        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["source_looo_unknown_margin"] = (0.22, 0.26, 0.30)[strictness] if not aggressive else (0.38, 0.46, 0.54)[strictness]
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]
        spec["unknown_margin"] = (0.32, 0.38, 0.44)[strictness] if not aggressive else (0.56, 0.66, 0.76)[strictness]
        spec["two_branch_bg_min_score"] = (0.82, 0.78, 0.74)[strictness] if not aggressive else (0.72, 0.68, 0.64)[strictness]
        spec["two_branch_bg_min_margin"] = (0.12, 0.08, 0.04)[strictness] if not aggressive else (0.04, 0.00, -0.04)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.16, -0.12, -0.08)[strictness]
        spec["two_branch_old_anchor_delta"] = (-0.12, -0.08, -0.04)[strictness]
        spec["two_branch_old_anchor_margin"] = (-0.02, 0.00, 0.02)[strictness]
        spec["two_branch_seen_new_evidence_delta"] = (-0.14, -0.10, -0.06)[strictness]
        spec["two_branch_seen_new_anchor_delta"] = (-0.12, -0.08, -0.04)[strictness]
        spec["pre_reject_old_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["pre_reject_old_min_anchor_delta"] = (-0.02, -0.04, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["pre_reject_old_min_anchor_margin"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.02, -0.04)[strictness]
        spec["pre_reject_old_min_score_margin"] = (0.04, 0.00, -0.04)[strictness] if not aggressive else (0.00, -0.06, -0.12)[strictness]
        spec["pre_reject_seen_new_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["pre_reject_seen_new_min_anchor_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["pre_reject_seen_new_min_score_margin"] = (0.00, -0.04, -0.08)[strictness] if not aggressive else (-0.06, -0.12, -0.18)[strictness]
        spec["pre_reject_max_background_score"] = (0.72, 0.76, 0.80)[strictness] if not aggressive else (0.68, 0.72, 0.76)[strictness]
        spec["pre_reject_max_background_margin"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.02, 0.06, 0.10)[strictness]
        spec["pre_reject_defer_background_score"] = (0.70, 0.74, 0.78)[strictness] if not aggressive else (0.64, 0.68, 0.72)[strictness]
        spec["pre_reject_defer_background_margin"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.00, 0.04, 0.08)[strictness]
        spec["pre_reject_reject_background_score"] = (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.78, 0.74, 0.70)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.18, 0.14, 0.10)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["pre_reject_defer_action"] = "uncertain" if not aggressive else ("uncertain" if slot in {"A", "B", "C"} else "defer")
        spec["void_gate"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["void_background"] = 0.0 if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2000
    return specs


def _oa_mse_threeway48_stage_specs() -> list[dict]:
    """48-row structural old/seen-new/background head after next48eq."""

    specs = [dict(spec) for spec in _oa_mse_prereject48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "structural_three_way_old_seen_background_head_after_next48eq"
        spec["evidence_ref"] = (
            "next48eq_loss_pass_target_hit_0_old_mean_0p285_seen_new_mean_0p134_unknown_far_mean_0p486;"
            "high_H_rows_accept_unknown_and_low_FAR_rows_reject_known"
        )
        spec["description"] = (
            "Three-way conservative arm: train an explicit old/seen-new/pseudo-background head on allowed support "
            "and query-free pseudo-background, then defer ambiguous background-vs-known rows instead of hard reject."
            if not aggressive
            else "Three-way aggressive arm: stronger pseudo-background branch and sharper old/seen-new competition, "
            "testing whether rejection can be learned without collapsing old and seen-new query accuracy."
        )
        spec["three_way_decision_head"] = True
        spec["three_way_head_weight"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.36, 0.46, 0.56)[strictness]
        spec["three_way_head_temperature"] = (0.14, 0.12, 0.10)[strictness] if not aggressive else (0.10, 0.08, 0.06)[strictness]
        spec["three_way_head_known_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.06, 0.08, 0.10)[strictness]
        spec["three_way_head_background_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["three_way_accept_prob"] = (0.44, 0.48, 0.52)[strictness] if not aggressive else (0.48, 0.52, 0.56)[strictness]
        spec["three_way_reject_prob"] = (0.70, 0.66, 0.62)[strictness] if not aggressive else (0.62, 0.58, 0.54)[strictness]
        spec["three_way_defer_prob"] = (0.48, 0.44, 0.40)[strictness] if not aggressive else (0.44, 0.40, 0.36)[strictness]
        spec["three_way_known_background_margin"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["three_way_reject_margin"] = (0.10, 0.08, 0.06)[strictness] if not aggressive else (0.06, 0.04, 0.02)[strictness]
        spec["three_way_old_seen_ambiguity_margin"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["three_way_defer_action"] = "uncertain" if not aggressive or slot in {"A", "B", "C"} else "defer"
        spec["pre_reject_defer_arbitration"] = False if not aggressive else bool(slot in {"D", "E", "F"})
        spec["class_envelope_gate"] = False if not aggressive else True
        spec["class_envelope_gate_action"] = "uncertain"
        spec["retention_rescue_gate"] = False
        spec["two_branch_background_guard"] = False if not aggressive else True
        spec["void_gate"] = False if not aggressive else bool(slot in {"E", "F"})
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["known_coverage_weight"] = max(0.76 if not aggressive else 0.58, float(spec.get("known_coverage_weight", 0.0)))
        spec["source_ce"] = max(0.90 if not aggressive else 0.78, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.78 if not aggressive else 0.66, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.72 if not aggressive else 0.60, float(spec.get("old_neighborhood", 0.0)))
        spec["soft_proto"] = max(0.42 if not aggressive else 0.54, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.24 if not aggressive else 0.36, float(spec.get("soft_proto_boundary", 0.0)))
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["unknown_margin"] = (0.34, 0.40, 0.46)[strictness] if not aggressive else (0.54, 0.62, 0.70)[strictness]
        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["boundary_samples"] = max(6 if not aggressive else 8, int(spec.get("boundary_samples", 0)))
        spec["target_shift_samples"] = max(2 if not aggressive else 3, int(spec.get("target_shift_samples", 0)))
        spec["target_halo_samples"] = max(1 if not aggressive else 2, int(spec.get("target_halo_samples", 0)))
        spec["steps"] = min(64 if not aggressive else 72, max(48 if not aggressive else 56, int(spec.get("steps", 40)) + 4))
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2600
    return specs


def _oa_mse_covfloor48_stage_specs() -> list[dict]:
    """48-row coverage-preserving three-way head after next48er coverage collapse."""

    specs = [dict(spec) for spec in _oa_mse_threeway48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "coverage_preserving_three_way_known_floor_after_next48er"
        spec["evidence_ref"] = (
            "next48er_complete_negative_old_mean_0p052_seen_new_mean_0p009_unknown_far_0p046;"
            "background_prob_median_0p999928_known_rejected_by_three_way"
        )
        spec["description"] = (
            "Conservative coverage-floor arm: keep the old/seen-new/background head observable, reduce pseudo-background "
            "loss dominance, and defer rows with allowed known evidence instead of letting background reject first."
            if not aggressive
            else "Aggressive coverage-floor arm: allow known-evidence floor accept before background rejection unless "
            "background probability and margin are extreme, testing recovery of old and seen-new coverage."
        )
        spec["three_way_decision_head"] = True
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "defer" if not aggressive else ("accept" if slot in {"A", "B", "C", "D"} else "uncertain")
        spec["three_way_head_weight"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.18, 0.24, 0.30)[strictness]
        spec["three_way_head_temperature"] = (0.22, 0.18, 0.14)[strictness] if not aggressive else (0.18, 0.14, 0.10)[strictness]
        spec["three_way_head_known_margin"] = (0.08, 0.10, 0.12)[strictness] if not aggressive else (0.10, 0.12, 0.14)[strictness]
        spec["three_way_head_background_margin"] = (0.02, 0.03, 0.04)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.25 if not aggressive else 1.10
        spec["three_way_head_pseudo_ce_weight"] = (0.10, 0.16, 0.22)[strictness] if not aggressive else (0.18, 0.26, 0.34)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.70, 0.85, 1.00)[strictness] if not aggressive else (0.85, 1.00, 1.15)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.15, 0.25, 0.35)[strictness] if not aggressive else (0.25, 0.35, 0.45)[strictness]
        spec["three_way_accept_prob"] = (0.40, 0.44, 0.48)[strictness] if not aggressive else (0.36, 0.40, 0.44)[strictness]
        spec["three_way_reject_prob"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["three_way_defer_prob"] = (0.58, 0.54, 0.50)[strictness] if not aggressive else (0.52, 0.48, 0.44)[strictness]
        spec["three_way_known_background_margin"] = (-0.04, -0.02, 0.00)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["three_way_reject_margin"] = (0.26, 0.20, 0.14)[strictness] if not aggressive else (0.16, 0.10, 0.06)[strictness]
        spec["three_way_old_seen_ambiguity_margin"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["three_way_known_floor_old_min_evidence_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (-0.04, -0.07, -0.10)[strictness] if not aggressive else (-0.08, -0.12, -0.16)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (-0.04, -0.08, -0.12)[strictness] if not aggressive else (-0.10, -0.16, -0.22)[strictness]
        spec["three_way_known_floor_seen_new_min_evidence_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_seen_new_min_anchor_delta"] = (-0.02, -0.05, -0.08)[strictness] if not aggressive else (-0.06, -0.10, -0.14)[strictness]
        spec["three_way_known_floor_seen_new_min_score_margin"] = (-0.04, -0.08, -0.12)[strictness] if not aggressive else (-0.12, -0.18, -0.24)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.998, 0.996, 0.994)[strictness] if not aggressive else (0.996, 0.992, 0.988)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.20, 1.00, 0.80)[strictness] if not aggressive else (1.00, 0.75, 0.55)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_reject_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.22, 0.18, 0.14)[strictness] if not aggressive else (0.16, 0.10, 0.06)[strictness]
        spec["class_envelope_gate"] = False if not aggressive else bool(slot in {"D", "E", "F"})
        spec["class_envelope_gate_action"] = "uncertain"
        spec["retention_rescue_gate"] = bool(aggressive and slot in {"A", "B", "C"})
        spec["two_branch_background_guard"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["void_gate"] = False
        spec["known_coverage_weight"] = max(0.90 if not aggressive else 0.78, float(spec.get("known_coverage_weight", 0.0)))
        spec["source_ce"] = max(1.00 if not aggressive else 0.88, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(0.88 if not aggressive else 0.76, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.82 if not aggressive else 0.70, float(spec.get("old_neighborhood", 0.0)))
        spec["soft_proto"] = max(0.52 if not aggressive else 0.62, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.30 if not aggressive else 0.42, float(spec.get("soft_proto_boundary", 0.0)))
        spec["unknown_moat"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.12, 0.20, 0.28)[strictness]
        spec["unknown_margin"] = (0.28, 0.34, 0.40)[strictness] if not aggressive else (0.44, 0.54, 0.64)[strictness]
        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.04, 0.08, 0.12)[strictness]
        spec["steps"] = min(80 if not aggressive else 88, max(56 if not aggressive else 64, int(spec.get("steps", 40)) + 8))
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 3200
    return specs


def _oa_mse_classfirst48_stage_specs() -> list[dict]:
    """48-row class-first known assignment route after next48es background default."""

    specs = [dict(spec) for spec in _oa_mse_covfloor48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "class_first_known_assignment_then_unknown_veto_after_next48es"
        spec["evidence_ref"] = (
            "next48es_complete_negative_raw_old_mean_0p173_raw_seen_new_mean_0p138_unknown_far_mean_0p418;"
            "three_way_background_default_old_base_reject_0p815_seen_new_base_reject_0p878_unknown_base_reject_0p882"
        )
        spec["description"] = (
            "Conservative class-first arm: restore old/seen-new assignment with strict known evidence, then allow "
            "background only as a strong veto instead of the first winner."
            if not aggressive
            else "Aggressive class-first arm: partially unfreeze the low-compute adapter and strengthen soft "
            "prototype geometry so seen-new/old separation is learned before unknown rejection is applied."
        )
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept"
        spec["three_way_head_weight"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.16, 0.24, 0.32)[strictness]
        spec["three_way_head_temperature"] = (0.24, 0.20, 0.16)[strictness] if not aggressive else (0.18, 0.14, 0.10)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.35 if not aggressive else 1.20
        spec["three_way_head_pseudo_ce_weight"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.55, 0.70, 0.85)[strictness] if not aggressive else (0.75, 0.90, 1.05)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.08, 0.14, 0.20)[strictness] if not aggressive else (0.18, 0.28, 0.38)[strictness]
        spec["three_way_accept_prob"] = (0.62, 0.66, 0.70)[strictness] if not aggressive else (0.56, 0.60, 0.64)[strictness]
        spec["three_way_reject_prob"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["three_way_defer_prob"] = (0.70, 0.66, 0.62)[strictness] if not aggressive else (0.62, 0.58, 0.54)[strictness]
        spec["three_way_known_background_margin"] = (-0.20, -0.16, -0.12)[strictness] if not aggressive else (-0.28, -0.22, -0.16)[strictness]
        spec["three_way_reject_margin"] = (0.36, 0.30, 0.24)[strictness] if not aggressive else (0.28, 0.20, 0.12)[strictness]
        spec["three_way_old_seen_ambiguity_margin"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.999, 0.998, 0.996)[strictness] if not aggressive else (0.998, 0.995, 0.992)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.60, 1.30, 1.00)[strictness] if not aggressive else (1.20, 0.90, 0.65)[strictness]
        spec["three_way_known_floor_old_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (-0.02, -0.04, -0.06)[strictness] if not aggressive else (-0.06, -0.10, -0.14)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (0.00, -0.04, -0.08)[strictness] if not aggressive else (-0.08, -0.14, -0.20)[strictness]
        spec["three_way_known_floor_seen_new_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_seen_new_min_anchor_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_seen_new_min_score_margin"] = (0.00, -0.04, -0.08)[strictness] if not aggressive else (-0.10, -0.16, -0.22)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_max_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["pre_reject_reject_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.90, 0.84, 0.78)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.30, 0.24, 0.18)[strictness] if not aggressive else (0.22, 0.16, 0.10)[strictness]
        spec["retention_rescue_gate"] = bool(not aggressive or slot in {"A", "B", "C"})
        spec["two_branch_background_guard"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["class_envelope_gate"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["class_envelope_gate_action"] = "uncertain"
        spec["known_coverage_weight"] = max(1.08 if not aggressive else 0.92, float(spec.get("known_coverage_weight", 0.0)))
        spec["source_ce"] = max(1.15 if not aggressive else 0.98, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(1.02 if not aggressive else 0.86, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.94 if not aggressive else 0.78, float(spec.get("old_neighborhood", 0.0)))
        spec["soft_proto"] = max(0.62 if not aggressive else 0.74, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.36 if not aggressive else 0.50, float(spec.get("soft_proto_boundary", 0.0)))
        spec["support_center_ce"] = max(0.20 if not aggressive else 0.30, float(spec.get("support_center_ce", 0.0)))
        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.04, 0.08, 0.12)[strictness]
        spec["unknown_moat"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.14, 0.20)[strictness]
        spec["unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.38, 0.48, 0.58)[strictness]
        spec["boundary_samples"] = max(4 if not aggressive else 6, int(spec.get("boundary_samples", 0)))
        spec["steps"] = min(88 if not aggressive else 96, max(60 if not aggressive else 68, int(spec.get("steps", 40)) + 8))
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"D", "E", "F"} else "low_rank")
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 3900
    return specs


def _oa_mse_evibg48_stage_specs() -> list[dict]:
    """48-row evidence-balanced old/seen-new/background route after next48eu."""

    specs = [dict(spec) for spec in _oa_mse_classfirst48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "evidence_balanced_known_background_competition_after_next48eu"
        spec["evidence_ref"] = (
            "next48eu_complete_negative_old_mean_0p2828_seen_new_mean_0p1594_unknown_far_mean_0p5267;"
            "class_first_known_prob_old_0p9821_seen_new_0p9814_unknown_0p9837"
        )
        spec["description"] = (
            "Conservative evidence-balanced arm: require target support evidence before a high class-first known "
            "probability can accept old/seen-new, so unknown-like rows cannot bypass background competition."
            if not aggressive
            else "Aggressive evidence-balanced arm: strengthen pseudo-background and LOOO pressure while still "
            "requiring old/seen-new support evidence before any known accept, testing whether learned rejection can "
            "avoid the class-first unknown-as-known failure."
        )
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "defer" if not aggressive else ("accept" if slot in {"A", "B", "C"} else "uncertain")
        spec["three_way_head_weight"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]
        spec["three_way_head_temperature"] = (0.22, 0.18, 0.14)[strictness] if not aggressive else (0.16, 0.12, 0.09)[strictness]
        spec["three_way_head_known_margin"] = (0.08, 0.10, 0.12)[strictness] if not aggressive else (0.10, 0.12, 0.14)[strictness]
        spec["three_way_head_background_margin"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.07, 0.09, 0.11)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.35 if not aggressive else 1.15
        spec["three_way_head_pseudo_ce_weight"] = (0.10, 0.16, 0.22)[strictness] if not aggressive else (0.20, 0.30, 0.40)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.85, 1.00, 1.15)[strictness] if not aggressive else (1.00, 1.20, 1.40)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.20, 0.30, 0.40)[strictness] if not aggressive else (0.35, 0.50, 0.65)[strictness]
        spec["three_way_accept_prob"] = (0.58, 0.62, 0.66)[strictness] if not aggressive else (0.52, 0.56, 0.60)[strictness]
        spec["three_way_reject_prob"] = (0.84, 0.80, 0.76)[strictness] if not aggressive else (0.76, 0.70, 0.64)[strictness]
        spec["three_way_defer_prob"] = (0.62, 0.58, 0.54)[strictness] if not aggressive else (0.56, 0.50, 0.44)[strictness]
        spec["three_way_known_background_margin"] = (-0.02, 0.00, 0.02)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
        spec["three_way_reject_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["three_way_old_seen_ambiguity_margin"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["three_way_known_floor_old_min_evidence_delta"] = (0.12, 0.08, 0.04)[strictness] if not aggressive else (0.08, 0.04, 0.00)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (0.02, -0.02, -0.06)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (0.06, 0.03, 0.00)[strictness] if not aggressive else (0.03, 0.00, -0.03)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (0.02, -0.02, -0.06)[strictness] if not aggressive else (-0.04, -0.10, -0.16)[strictness]
        spec["three_way_known_floor_seen_new_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_seen_new_min_anchor_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_seen_new_min_score_margin"] = (-0.02, -0.06, -0.10)[strictness] if not aggressive else (-0.08, -0.14, -0.20)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.998, 0.996, 0.994)[strictness] if not aggressive else (0.996, 0.992, 0.988)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.10, 0.85, 0.60)[strictness] if not aggressive else (0.85, 0.60, 0.40)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_max_background_score"] = (0.84, 0.80, 0.76)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["pre_reject_max_background_margin"] = (0.14, 0.10, 0.06)[strictness] if not aggressive else (0.08, 0.04, 0.00)[strictness]
        spec["pre_reject_defer_background_score"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.72, 0.66, 0.60)[strictness]
        spec["pre_reject_defer_background_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (0.04, 0.00, -0.04)[strictness]
        spec["pre_reject_reject_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.82, 0.76, 0.70)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]
        spec["retention_rescue_gate"] = False
        spec["two_branch_background_guard"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["class_envelope_gate"] = False
        spec["void_gate"] = bool(aggressive and slot in {"E", "F"})
        spec["void_background"] = 0.0 if not aggressive else (0.05, 0.08, 0.11)[strictness]
        spec["known_coverage_weight"] = max(1.10 if not aggressive else 0.92, float(spec.get("known_coverage_weight", 0.0)))
        spec["source_ce"] = max(1.12 if not aggressive else 0.96, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(1.00 if not aggressive else 0.84, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(0.92 if not aggressive else 0.76, float(spec.get("old_neighborhood", 0.0)))
        spec["support_center_ce"] = max(0.24 if not aggressive else 0.32, float(spec.get("support_center_ce", 0.0)))
        spec["soft_proto"] = max(0.54 if not aggressive else 0.66, float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max(0.34 if not aggressive else 0.46, float(spec.get("soft_proto_boundary", 0.0)))
        spec["soft_proto_boundary_margin"] = (0.16, 0.20, 0.24)[strictness] if not aggressive else (0.20, 0.26, 0.32)[strictness]
        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["source_looo_unknown_margin"] = (0.26, 0.32, 0.38)[strictness] if not aggressive else (0.40, 0.50, 0.60)[strictness]
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["unknown_margin"] = (0.34, 0.42, 0.50)[strictness] if not aggressive else (0.52, 0.64, 0.76)[strictness]
        spec["boundary_samples"] = max(8 if not aggressive else 12, int(spec.get("boundary_samples", 0)))
        spec["target_shift_samples"] = max(3 if not aggressive else 5, int(spec.get("target_shift_samples", 0)))
        spec["target_halo_samples"] = max(2 if not aggressive else 4, int(spec.get("target_halo_samples", 0)))
        spec["target_ring_samples"] = max(8 if not aggressive else 12, int(spec.get("target_ring_samples", 0)))
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"E", "F"} else "low_rank")
        spec["steps"] = min(88 if not aggressive else 96, max(64 if not aggressive else 72, int(spec.get("steps", 40)) + 8))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 4700
    return specs


def _oa_mse_softtarget48_stage_specs() -> list[dict]:
    """48-row prototype-mixture soft-target route after next48ev."""

    specs = [dict(spec) for spec in _oa_mse_evibg48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "prototype_mixture_soft_target_after_next48ev"
        spec["evidence_ref"] = (
            "next48ev_complete_negative_old_mean_0p1684_seen_new_mean_0p1497_unknown_far_mean_0p4122;"
            "evibg_known_prob_old_0p9839_seen_new_0p9792_unknown_0p9861;"
            "background_prob_high_for_all_groups"
        )
        spec["description"] = (
            "Prototype-mixture conservative arm: make class assignment depend on a soft mixture of same-class "
            "support prototypes plus high known-coverage/retention losses, then use background only as a late risk veto."
            if not aggressive
            else "Prototype-mixture aggressive arm: use top-3 soft prototype targets, stronger support-center geometry, "
            "source leave-one-old-out pseudo-unknown pressure, and a decoupled background risk veto to seek separation "
            "without repeating all-reject collapse."
        )
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else ("accept" if slot in {"A", "B", "C"} else "defer")
        spec["three_way_head_weight"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.16, 0.22, 0.28)[strictness]
        spec["three_way_head_temperature"] = (0.28, 0.24, 0.20)[strictness] if not aggressive else (0.20, 0.16, 0.12)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.45 if not aggressive else 1.25
        spec["three_way_head_pseudo_ce_weight"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.10, 0.16, 0.22)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.45, 0.60, 0.75)[strictness] if not aggressive else (0.70, 0.85, 1.00)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.06, 0.10, 0.16)[strictness] if not aggressive else (0.18, 0.28, 0.38)[strictness]
        spec["three_way_accept_prob"] = (0.66, 0.70, 0.74)[strictness] if not aggressive else (0.58, 0.62, 0.66)[strictness]
        spec["three_way_reject_prob"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["three_way_defer_prob"] = (0.76, 0.72, 0.68)[strictness] if not aggressive else (0.64, 0.58, 0.52)[strictness]
        spec["three_way_known_background_margin"] = (-0.32, -0.26, -0.20)[strictness] if not aggressive else (-0.18, -0.12, -0.06)[strictness]
        spec["three_way_reject_margin"] = (0.46, 0.38, 0.30)[strictness] if not aggressive else (0.26, 0.18, 0.10)[strictness]
        spec["three_way_old_seen_ambiguity_margin"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.9995, 0.9990, 0.9980)[strictness] if not aggressive else (0.9980, 0.9960, 0.9940)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.80, 1.50, 1.20)[strictness] if not aggressive else (1.20, 0.90, 0.65)[strictness]
        spec["three_way_known_floor_old_min_evidence_delta"] = (0.04, 0.02, 0.00)[strictness] if not aggressive else (0.02, -0.02, -0.06)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.03, -0.06)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (0.00, -0.04, -0.08)[strictness] if not aggressive else (-0.06, -0.12, -0.18)[strictness]
        spec["three_way_known_floor_seen_new_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_seen_new_min_anchor_delta"] = (0.00, -0.02, -0.04)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_seen_new_min_score_margin"] = (0.00, -0.04, -0.08)[strictness] if not aggressive else (-0.08, -0.14, -0.20)[strictness]

        spec["soft_proto"] = max((0.82, 0.88, 0.94)[strictness] if not aggressive else (0.94, 1.05, 1.16)[strictness], float(spec.get("soft_proto", 0.0)))
        spec["soft_proto_boundary"] = max((0.44, 0.52, 0.60)[strictness] if not aggressive else (0.60, 0.72, 0.84)[strictness], float(spec.get("soft_proto_boundary", 0.0)))
        spec["soft_proto_boundary_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.18, 0.24, 0.30)[strictness]
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 2 if not aggressive else 3
        spec["multiproto_temperature"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.06, 0.05)[strictness]
        spec["multiproto_score_weight"] = (1.15, 1.30, 1.45)[strictness] if not aggressive else (1.45, 1.70, 1.95)[strictness]
        spec["support_center_ce"] = max((0.38, 0.44, 0.50)[strictness] if not aggressive else (0.52, 0.62, 0.72)[strictness], float(spec.get("support_center_ce", 0.0)))
        spec["support_center_temperature"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.06, 0.05)[strictness]
        spec["support_center_margin"] = (0.10, 0.12, 0.14)[strictness] if not aggressive else (0.14, 0.18, 0.22)[strictness]
        spec["support_contrast"] = max((0.36, 0.42, 0.48)[strictness] if not aggressive else (0.50, 0.60, 0.70)[strictness], float(spec.get("support_contrast", 0.0)))

        spec["known_coverage_weight"] = max((1.35, 1.45, 1.55)[strictness] if not aggressive else (1.05, 1.15, 1.25)[strictness], float(spec.get("known_coverage_weight", 0.0)))
        spec["known_coverage_margin"] = (0.14, 0.16, 0.18)[strictness] if not aggressive else (0.10, 0.12, 0.14)[strictness]
        spec["known_coverage_min_affinity"] = (0.36, 0.40, 0.44)[strictness] if not aggressive else (0.30, 0.34, 0.38)[strictness]
        spec["source_ce"] = max(1.25 if not aggressive else 1.02, float(spec.get("source_ce", 0.0)))
        spec["old_bridge"] = max(1.12 if not aggressive else 0.92, float(spec.get("old_bridge", 0.0)))
        spec["old_neighborhood"] = max(1.02 if not aggressive else 0.82, float(spec.get("old_neighborhood", 0.0)))
        spec["old_surrogate_margin_weight"] = max(0.10 if not aggressive else 0.14, float(spec.get("old_surrogate_margin_weight", 0.0)))
        spec["old_surrogate_margin"] = max(0.12 if not aggressive else 0.18, float(spec.get("old_surrogate_margin", 0.0)))
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.02, 0.03, 0.04)[strictness] if not aggressive else (0.05, 0.07, 0.09)[strictness]
        spec["support_retention_guard_slack"] = (0.10, 0.12, 0.14)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["old_retention_quantile"] = (0.98, 0.96, 0.94)[strictness] if not aggressive else (0.94, 0.92, 0.90)[strictness]

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_max_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["pre_reject_max_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["pre_reject_defer_background_score"] = (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.74, 0.68, 0.62)[strictness]
        spec["pre_reject_defer_background_margin"] = (0.14, 0.10, 0.06)[strictness] if not aggressive else (0.04, 0.00, -0.04)[strictness]
        spec["pre_reject_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.32, 0.26, 0.20)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["two_branch_background_guard"] = bool(aggressive or slot in {"E", "F"})
        spec["two_branch_bg_min_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.72, 0.66, 0.60)[strictness]
        spec["two_branch_bg_min_margin"] = (0.16, 0.10, 0.04)[strictness] if not aggressive else (0.02, -0.04, -0.10)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["two_branch_seen_new_evidence_delta"] = (-0.16, -0.12, -0.08)[strictness] if not aggressive else (-0.10, -0.06, -0.02)[strictness]
        spec["retention_rescue_gate"] = True if not aggressive else bool(slot in {"A", "B", "C"})
        spec["retention_rescue_max_background_score"] = (0.82, 0.78, 0.74)[strictness] if not aggressive else (0.70, 0.66, 0.62)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.12, 0.08, 0.04)[strictness] if not aggressive else (0.02, -0.02, -0.06)[strictness]

        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.50, 0.62, 0.74)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["unknown_moat"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.28, 0.38, 0.48)[strictness]
        spec["unknown_margin"] = (0.28, 0.36, 0.44)[strictness] if not aggressive else (0.62, 0.74, 0.86)[strictness]
        spec["boundary_samples"] = max(6 if not aggressive else 14, int(spec.get("boundary_samples", 0)))
        spec["target_shift_samples"] = max(2 if not aggressive else 6, int(spec.get("target_shift_samples", 0)))
        spec["target_halo_samples"] = max(2 if not aggressive else 5, int(spec.get("target_halo_samples", 0)))
        spec["target_ring_samples"] = max(6 if not aggressive else 14, int(spec.get("target_ring_samples", 0)))
        spec["void_background"] = 0.0 if not aggressive else (0.06, 0.10, 0.14)[strictness]
        spec["void_gate"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["class_envelope_gate"] = False

        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"D", "E", "F"} else "low_rank")
        spec["steps"] = min(96 if not aggressive else 112, max(72 if not aggressive else 84, int(spec.get("steps", 40)) + 12))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 5600
    return specs


def _oa_mse_denshell48_stage_specs() -> list[dict]:
    """48-row class-conditional density-shell route after next48ex.

    The design is literature-inspired but CVS-specific: it does not copy a
    named few-shot/open-set method. It uses support-calibrated density shells as
    an inlier proof before any pseudo-background rejection.
    """

    specs = [dict(spec) for spec in _oa_mse_softtarget48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "class_conditional_density_shell_after_next48ex"
        spec["evidence_ref"] = (
            "next48ex_complete_negative_old_mean_0p2205_seen_new_mean_0p1323_unknown_far_mean_0p4813;"
            "negative_anchor_phase_transition_known_collapse_after_weight_0p34;"
            "score_table_background_and_known_probs_high_for_old_seen_new_unknown"
        )
        spec["description"] = (
            "Density-shell conservative arm: prove old/seen-new inlier membership from support-calibrated class shells "
            "before allowing any background veto; negative-anchor basin is disabled to avoid known collapse."
            if not aggressive
            else "Density-shell aggressive arm: combine source leave-one-old-out pseudo-unknown pressure with class-shell "
            "inlier-first arbitration, so rejection only fires when both old and seen-new shells fail."
        )
        spec["density_shell_gate"] = True
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain"
        spec["anchor_density_topk"] = 3 if not aggressive else (3 if slot in {"A", "B", "C"} else 4)
        spec["anchor_density_temperature"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.06, 0.05)[strictness]
        spec["anchor_density_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.04, 0.07, 0.10)[strictness]
        spec["anchor_density_margin_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["density_shell_old_min_evidence_delta"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["density_shell_old_min_anchor_delta"] = (-0.16, -0.12, -0.08)[strictness] if not aggressive else (-0.10, -0.06, -0.02)[strictness]
        spec["density_shell_old_min_density_delta"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
        spec["density_shell_seen_new_min_evidence_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["density_shell_seen_new_min_anchor_delta"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["density_shell_seen_new_min_density_delta"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
        spec["density_shell_accept_background_margin"] = (0.34, 0.26, 0.18)[strictness] if not aggressive else (0.18, 0.12, 0.06)[strictness]
        spec["density_shell_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["density_shell_reject_background_margin"] = (0.34, 0.26, 0.18)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["density_shell_reject_min_failed_shells"] = 3 if not aggressive else 2

        spec["negative_anchor_weight"] = 0.0
        spec["negative_anchor_margin"] = 0.08
        spec["negative_anchor_temperature"] = 0.12
        spec["void_background"] = 0.0 if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["void_gate"] = False
        spec["class_envelope_gate"] = False
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "defer"
        spec["pre_reject_max_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["pre_reject_max_background_margin"] = (0.26, 0.20, 0.14)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["pre_reject_reject_background_score"] = (0.99, 0.96, 0.92)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.38, 0.30, 0.22)[strictness] if not aggressive else (0.18, 0.12, 0.06)[strictness]
        spec["pre_reject_old_min_evidence_delta"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
        spec["pre_reject_seen_new_min_evidence_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]

        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["three_way_reject_prob"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["three_way_reject_margin"] = (0.42, 0.34, 0.26)[strictness] if not aggressive else (0.22, 0.16, 0.10)[strictness]
        spec["three_way_known_background_margin"] = (-0.36, -0.30, -0.24)[strictness] if not aggressive else (-0.18, -0.12, -0.06)[strictness]

        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (1.55, 1.65, 1.75)[strictness] if not aggressive else (1.15, 1.25, 1.35)[strictness])
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["support_retention_guard_slack"] = (0.14, 0.16, 0.18)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["source_looo_unknown_weight"] = (0.00, 0.03, 0.06)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.46, 0.58, 0.70)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.05, 0.06)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["unknown_moat"] = (0.02, 0.06, 0.10)[strictness] if not aggressive else (0.22, 0.32, 0.42)[strictness]
        spec["unknown_margin"] = (0.26, 0.32, 0.38)[strictness] if not aggressive else (0.54, 0.66, 0.78)[strictness]
        spec["target_ring_samples"] = max(6 if not aggressive else 12, int(spec.get("target_ring_samples", 0)))
        spec["target_halo_samples"] = max(2 if not aggressive else 6, int(spec.get("target_halo_samples", 0)))
        spec["target_shift_samples"] = max(2 if not aggressive else 5, int(spec.get("target_shift_samples", 0)))
        spec["adapter_selection_policy"] = "constrained_retention_risk"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"D", "E", "F"} else "low_rank")
        spec["steps"] = min(104 if not aggressive else 116, max(80 if not aggressive else 92, int(spec.get("steps", 40)) + 12))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 6500
    return specs


def _oa_mse_idcons48_stage_specs() -> list[dict]:
    """48-row identity-consensus OA-MSE route after next48ey.

    This is a CVS-specific route, not a copied paper method. The key change is
    decision order: old/seen-new identity evidence is fused first, and
    pseudo-background rejection fires only when both identity hypotheses fail.
    """

    specs = [dict(spec) for spec in _oa_mse_denshell48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "identity_consensus_after_next48ey"
        spec["evidence_ref"] = (
            "next48ey_corrected_negative_old_mean_0p1346_seen_new_mean_0p1132_unknown_rejection_mean_0p9323;"
            "density_shell_accept_mean_0p553_but_identity_wrong;"
            "support_fit_mean_0p968_old_query_retention_failed"
        )
        spec["description"] = (
            "Identity-consensus conservative arm: keep density-shell inlier evidence, then select old/seen-new labels "
            "with fused identity score before background veto; pseudo-background is not allowed to dominate known identity."
            if not aggressive
            else "Identity-consensus aggressive arm: retain identity-first arbitration while increasing source leave-one-old-out "
            "and pseudo-ring pressure, so unknown rejection is tested only after both known identities fail."
        )
        spec["identity_consensus_arbitration"] = True
        spec["density_shell_gate"] = True
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain"
        spec["anchor_density_topk"] = 3 if not aggressive else (4 if slot in {"D", "E", "F"} else 3)
        spec["anchor_density_temperature"] = (0.14, 0.11, 0.09)[strictness] if not aggressive else (0.08, 0.06, 0.05)[strictness]
        spec["anchor_density_quantile"] = (0.00, 0.01, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["anchor_density_margin_quantile"] = (0.00, 0.01, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]

        spec["identity_consensus_old_min_evidence_delta"] = (-0.16, -0.11, -0.06)[strictness] if not aggressive else (-0.08, -0.03, 0.02)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.22, -0.16, -0.10)[strictness] if not aggressive else (-0.12, -0.07, -0.02)[strictness]
        spec["identity_consensus_old_min_density_delta"] = (-0.18, -0.12, -0.06)[strictness] if not aggressive else (-0.08, -0.03, 0.02)[strictness]
        spec["identity_consensus_seen_new_min_evidence_delta"] = (-0.14, -0.09, -0.04)[strictness] if not aggressive else (-0.07, -0.02, 0.03)[strictness]
        spec["identity_consensus_seen_new_min_anchor_delta"] = (-0.18, -0.13, -0.08)[strictness] if not aggressive else (-0.10, -0.05, 0.00)[strictness]
        spec["identity_consensus_seen_new_min_density_delta"] = (-0.16, -0.10, -0.04)[strictness] if not aggressive else (-0.07, -0.02, 0.03)[strictness]
        spec["identity_consensus_min_identity_margin"] = (-0.18, -0.10, -0.02)[strictness] if not aggressive else (-0.08, -0.02, 0.04)[strictness]
        spec["identity_consensus_background_accept_margin"] = (0.42, 0.34, 0.26)[strictness] if not aggressive else (0.20, 0.14, 0.08)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.99, 0.96, 0.93)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.42, 0.34, 0.26)[strictness] if not aggressive else (0.18, 0.12, 0.06)[strictness]
        spec["identity_consensus_reject_min_identity_failures"] = 5 if not aggressive else 4

        spec["density_shell_old_min_evidence_delta"] = min(float(spec.get("density_shell_old_min_evidence_delta", -0.04)), spec["identity_consensus_old_min_evidence_delta"] + 0.04)
        spec["density_shell_seen_new_min_evidence_delta"] = min(float(spec.get("density_shell_seen_new_min_evidence_delta", -0.04)), spec["identity_consensus_seen_new_min_evidence_delta"] + 0.04)
        spec["density_shell_accept_background_margin"] = max(float(spec.get("density_shell_accept_background_margin", 0.18)), spec["identity_consensus_background_accept_margin"] - 0.04)
        spec["density_shell_reject_background_score"] = max(float(spec.get("density_shell_reject_background_score", 0.86)), spec["identity_consensus_reject_background_score"])
        spec["density_shell_reject_background_margin"] = max(float(spec.get("density_shell_reject_background_margin", 0.14)), spec["identity_consensus_reject_background_margin"])
        spec["density_shell_reject_min_failed_shells"] = 3 if not aggressive else 2

        spec["negative_anchor_weight"] = 0.0 if not aggressive else (0.04, 0.08, 0.12)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.03, 0.06, 0.09)[strictness]
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["three_way_reject_prob"] = (0.99, 0.96, 0.93)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["three_way_reject_margin"] = (0.46, 0.38, 0.30)[strictness] if not aggressive else (0.24, 0.16, 0.08)[strictness]
        spec["three_way_known_background_margin"] = (-0.44, -0.36, -0.28)[strictness] if not aggressive else (-0.20, -0.14, -0.08)[strictness]

        spec["pre_reject_defer_arbitration"] = False if not aggressive else bool(slot in {"D", "E", "F"})
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (1.80, 1.95, 2.10)[strictness] if not aggressive else (1.25, 1.40, 1.55)[strictness])
        spec["support_center_ce"] = max(float(spec.get("support_center_ce", 0.0)), (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness])
        spec["support_center_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.16, 0.22, 0.28)[strictness]
        spec["soft_proto"] = max(float(spec.get("soft_proto", 0.0)), (0.18, 0.22, 0.26)[strictness] if not aggressive else (0.22, 0.28, 0.34)[strictness])
        spec["soft_proto_topk"] = 3 if slot in {"C", "E", "F"} else 2
        spec["soft_proto_temperature"] = (0.16, 0.13, 0.10)[strictness] if not aggressive else (0.12, 0.09, 0.07)[strictness]
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = int(spec["soft_proto_topk"])
        spec["multiproto_temperature"] = spec["soft_proto_temperature"]
        spec["multiproto_score_weight"] = (1.20, 1.35, 1.50)[strictness] if not aggressive else (1.10, 1.30, 1.55)[strictness]
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.00, 0.01, 0.02)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["support_retention_guard_slack"] = (0.18, 0.20, 0.22)[strictness] if not aggressive else (0.10, 0.12, 0.14)[strictness]
        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.16, 0.24, 0.32)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.50, 0.62, 0.74)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.12, 0.16, 0.20)[strictness]
        spec["unknown_moat"] = (0.00, 0.04, 0.08)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.56, 0.68, 0.80)[strictness]
        spec["target_ring_samples"] = max(6 if not aggressive else 14, int(spec.get("target_ring_samples", 0)))
        spec["target_halo_samples"] = max(2 if not aggressive else 7, int(spec.get("target_halo_samples", 0)))
        spec["target_shift_samples"] = max(2 if not aggressive else 5, int(spec.get("target_shift_samples", 0)))
        spec["adapter_selection_policy"] = "identity_preserving" if not aggressive else "identity_preserving_risk"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if slot in {"D", "E", "F"} else "low_rank")
        spec["steps"] = min(112 if not aggressive else 124, max(88 if not aggressive else 96, int(spec.get("steps", 40)) + 16))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 7200
    return specs


def _oa_mse_conform48_stage_specs() -> list[dict]:
    """48-row support-conformal route after next48fa.

    next48fa showed that identity-first acceptance improves old retention but
    leaks unknowns, while global aggressive rejection collapses known coverage.
    This route adds a class-conditional support-conformal veto after identity
    consensus, so accepted rows must be explainable by the predicted class's
    own support/source geometry before pseudo-background can reject them.
    """

    specs = [dict(spec) for spec in _oa_mse_idcons48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_conformal_after_next48fa"
        spec["evidence_ref"] = (
            "next48fa_negative_old_mean_0p2412_seen_new_mean_0p1479_unknown_rejection_mean_0p5253;"
            "conservative_known_retention_unknown_false_accept;"
            "aggressive_unknown_reject_known_reject;"
            "support_acc_high_query_generalization_poor"
        )
        spec["description"] = (
            "Support-conformal conservative arm: retain identity-first known recovery, but accepted rows must pass a "
            "class-local support conformity floor before background-risk rejection can fire."
            if not aggressive
            else "Support-conformal aggressive arm: use tighter class-local conformity and background coupling to test "
            "whether unknown rejection can recover without the global reject collapse seen in next48fa."
        )
        spec["support_conformal_arbitration"] = True
        spec["identity_consensus_arbitration"] = True
        spec["density_shell_gate"] = True
        spec["support_conformal_calibration_quantile"] = (0.00, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.10)[strictness]
        spec["support_conformal_conformity_slack"] = (0.34, 0.28, 0.22)[strictness] if not aggressive else (0.22, 0.16, 0.10)[strictness]
        spec["support_conformal_anchor_margin_slack"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.12, 0.08, 0.04)[strictness]
        spec["support_conformal_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.82, 0.78, 0.74)[strictness]
        spec["support_conformal_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["support_conformal_hard_reject_margin"] = (0.34, 0.26, 0.18)[strictness] if not aggressive else (0.22, 0.16, 0.10)[strictness]
        spec["support_conformal_reject_min_failures"] = 3 if not aggressive else 2
        spec["support_conformal_reject_action"] = "reject" if not aggressive else ("defer" if strictness == 0 else "reject")

        spec["identity_consensus_background_accept_margin"] = min(
            float(spec.get("identity_consensus_background_accept_margin", 0.22)),
            (0.34, 0.28, 0.22)[strictness] if not aggressive else (0.18, 0.12, 0.08)[strictness],
        )
        spec["identity_consensus_reject_background_score"] = min(
            float(spec.get("identity_consensus_reject_background_score", 0.90)),
            (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.80, 0.76)[strictness],
        )
        spec["identity_consensus_reject_background_margin"] = min(
            float(spec.get("identity_consensus_reject_background_margin", 0.18)),
            (0.32, 0.24, 0.18)[strictness] if not aggressive else (0.14, 0.10, 0.06)[strictness],
        )
        spec["pre_reject_defer_arbitration"] = False if not aggressive else bool(strictness >= 1)
        spec["support_retention_guard_slack"] = max(float(spec.get("support_retention_guard_slack", 0.02)), 0.18 if not aggressive else 0.12)
        spec["source_looo_unknown_weight"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.18, 0.24, 0.30)[strictness]
        spec["unknown_moat"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (2.0, 2.1, 2.2)[strictness] if not aggressive else (1.45, 1.60, 1.75)[strictness])
        spec["adapter_selection_policy"] = "identity_preserving" if not aggressive else "identity_preserving_risk"
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 8100
    return specs


def _oa_mse_recon48_stage_specs() -> list[dict]:
    """48-row class-local reconstruction route after next48fb."""

    specs = [dict(spec) for spec in _oa_mse_conform48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_reconstruction_after_next48fb"
        spec["evidence_ref"] = (
            "next48fb_complete_diagnostic_negative_old_mean_0p2471_seen_new_mean_0p1413_unknown_rejection_mean_0p5375;"
            "conservative_unknown_to_old_high_0p7104;"
            "aggressive_old_reject_high_0p8722;"
            "support_acc_high_0p9698_query_boundary_unlearned"
        )
        spec["description"] = (
            "Reconstruction conservative arm: keep identity and support-conformal evidence, then require accepted rows "
            "to be reconstructable by their predicted class-local support subspace while using reciprocal boundary negatives only as a coupled risk signal."
            if not aggressive
            else "Reconstruction aggressive arm: tighten class-local residual ceilings and reciprocal boundary negatives to test whether unknown leakage can be reduced without the known-collapse pattern from next48fb."
        )
        spec["support_reconstruction_arbitration"] = True
        spec["support_conformal_arbitration"] = bool(strictness <= 1) if not aggressive else bool(strictness == 0)
        spec["identity_consensus_arbitration"] = True
        spec["density_shell_gate"] = True
        k_old = int(spec.get("k_old", 5))
        spec["support_reconstruction_rank"] = 1 if k_old <= 2 else (2 if strictness < 2 else 3)
        spec["support_reconstruction_residual_quantile"] = (0.98, 0.95, 0.90)[strictness] if not aggressive else (0.90, 0.85, 0.80)[strictness]
        spec["support_reconstruction_residual_slack"] = (0.14, 0.10, 0.07)[strictness] if not aggressive else (0.08, 0.05, 0.03)[strictness]
        spec["support_reconstruction_min_residual_floor"] = (0.08, 0.06, 0.05)[strictness] if not aggressive else (0.05, 0.04, 0.03)[strictness]
        spec["support_reconstruction_negative_scale"] = (0.45, 0.55, 0.65)[strictness] if not aggressive else (0.60, 0.70, 0.80)[strictness]
        spec["support_reconstruction_negative_margin"] = (0.12, 0.06, 0.00)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["support_reconstruction_hard_residual_margin"] = (0.18, 0.14, 0.10)[strictness] if not aggressive else (0.12, 0.08, 0.05)[strictness]
        spec["support_reconstruction_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.80, 0.76)[strictness]
        spec["support_reconstruction_background_margin"] = (0.28, 0.20, 0.12)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["support_reconstruction_reject_min_failures"] = 3 if not aggressive else 2
        spec["support_reconstruction_reject_action"] = "defer" if (aggressive and strictness == 0) else "reject"
        spec["pre_reject_defer_arbitration"] = False if not aggressive else bool(strictness == 2)
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (2.20, 2.35, 2.50)[strictness] if not aggressive else (1.50, 1.70, 1.90)[strictness])
        spec["support_center_ce"] = max(float(spec.get("support_center_ce", 0.0)), (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.34, 0.42, 0.50)[strictness])
        spec["support_center_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.20, 0.26, 0.32)[strictness]
        spec["soft_proto"] = max(float(spec.get("soft_proto", 0.0)), (0.22, 0.26, 0.30)[strictness] if not aggressive else (0.30, 0.38, 0.46)[strictness])
        spec["unknown_moat"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.28, 0.38, 0.48)[strictness]
        spec["unknown_margin"] = (0.32, 0.40, 0.48)[strictness] if not aggressive else (0.62, 0.76, 0.90)[strictness]
        spec["source_looo_unknown_weight"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["target_ring_samples"] = max(int(spec.get("target_ring_samples", 0)), 8 if not aggressive else 18)
        spec["target_halo_samples"] = max(int(spec.get("target_halo_samples", 0)), 4 if not aggressive else 9)
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if strictness >= 1 else "low_rank")
        spec["steps"] = min(132 if not aggressive else 148, max(96 if not aggressive else 112, int(spec.get("steps", 40)) + 20))
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 9100
    return specs


def _oa_mse_sourcerisk48_stage_specs() -> list[dict]:
    """48-row source-LOOO risk route after next48fc diagnostic failure.

    next48fc showed that support reconstruction residuals reject known rows
    more often than true unknown rows. This route therefore stops treating
    reconstruction as the main separator and promotes source leave-one-old-out
    impostor calibration into an explicit inference-time unknown-risk arbiter.
    """

    specs = [dict(spec) for spec in _oa_mse_idcons48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "source_looo_risk_arbitration_after_next48fc"
        spec["evidence_ref"] = (
            "next48fc_complete_diagnostic_negative_old_mean_0p2571_seen_new_mean_0p1323_unknown_rejection_mean_0p5122;"
            "support_reconstruction_rejected_old_841_new_300_unknown_260;"
            "residual_margins_overlap_old_new_unknown;"
            "loss_decreased_but_open_set_separator_not_learned"
        )
        spec["description"] = (
            "Source-risk conservative arm: identity-first known retention remains primary, while source leave-one-old-out "
            "impostor scores only veto accepted rows when class score, support evidence, and pseudo-background jointly indicate open-space risk."
            if not aggressive
            else "Source-risk aggressive arm: stronger source leave-one-old-out impostor calibration plus negative anchors and residual adapter "
            "tests whether unknown rejection can become learnable without target-unknown labels."
        )
        spec["support_reconstruction_arbitration"] = False
        spec["source_looo_risk_arbitration"] = True
        spec["identity_consensus_arbitration"] = True
        spec["density_shell_gate"] = bool(not aggressive or strictness <= 1)
        spec["support_conformal_arbitration"] = bool((not aggressive and strictness == 2) or (aggressive and strictness == 0))
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"

        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.18, 0.28, 0.38)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.52, 0.66, 0.80)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]
        spec["source_looo_max_samples_per_class"] = 18 if not aggressive else 36
        spec["source_looo_risk_quantile"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["source_looo_risk_slack"] = (0.10, 0.06, 0.03)[strictness] if not aggressive else (0.02, -0.02, -0.06)[strictness]
        spec["source_looo_risk_min_score_margin"] = (-0.06, -0.02, 0.02)[strictness] if not aggressive else (0.02, 0.06, 0.10)[strictness]
        spec["source_looo_risk_min_known_evidence_delta"] = (-0.16, -0.11, -0.06)[strictness] if not aggressive else (-0.08, -0.03, 0.02)[strictness]
        spec["source_looo_risk_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["source_looo_risk_background_margin"] = (0.30, 0.22, 0.14)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 3 if not aggressive else 2
        spec["source_looo_risk_reject_action"] = "defer" if (aggressive and strictness == 0) else "reject"

        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (2.30, 2.45, 2.60)[strictness] if not aggressive else (1.40, 1.60, 1.80)[strictness])
        spec["support_center_ce"] = max(float(spec.get("support_center_ce", 0.0)), (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.36, 0.46, 0.56)[strictness])
        spec["support_center_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["soft_proto"] = max(float(spec.get("soft_proto", 0.0)), (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.34, 0.46, 0.58)[strictness])
        spec["soft_proto_boundary"] = max(float(spec.get("soft_proto_boundary", 0.0)), (0.12, 0.18, 0.24)[strictness] if not aggressive else (0.28, 0.40, 0.52)[strictness])
        spec["soft_proto_boundary_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["soft_proto_topk"] = 3 if strictness >= 1 else 2
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = int(spec["soft_proto_topk"])
        spec["multiproto_score_weight"] = (1.30, 1.45, 1.60)[strictness] if not aggressive else (1.20, 1.45, 1.70)[strictness]

        spec["unknown_moat"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.30, 0.42, 0.54)[strictness]
        spec["unknown_margin"] = (0.30, 0.38, 0.46)[strictness] if not aggressive else (0.66, 0.82, 0.98)[strictness]
        spec["negative_anchor_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.18, 0.30, 0.42)[strictness]
        spec["negative_anchor_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.06, 0.10, 0.14)[strictness]
        spec["target_ring_samples"] = max(int(spec.get("target_ring_samples", 0)), 8 if not aggressive else 20)
        spec["target_halo_samples"] = max(int(spec.get("target_halo_samples", 0)), 4 if not aggressive else 10)
        spec["target_shift_samples"] = max(int(spec.get("target_shift_samples", 0)), 2 if not aggressive else 6)

        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.00, 0.01, 0.02)[strictness] if not aggressive else (0.04, 0.07, 0.10)[strictness]
        spec["support_retention_guard_slack"] = (0.22, 0.20, 0.18)[strictness] if not aggressive else (0.12, 0.10, 0.08)[strictness]
        spec["old_retention_quantile"] = (0.98, 0.96, 0.94)[strictness] if not aggressive else (0.90, 0.86, 0.82)[strictness]
        spec["adapter_selection_policy"] = "identity_preserving" if not aggressive else "identity_preserving_risk"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if strictness >= 1 else "low_rank")
        spec["steps"] = min(136 if not aggressive else 156, max(104 if not aggressive else 120, int(spec.get("steps", 40)) + 24))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 10300
    return specs


def _oa_mse_supportcv48_stage_specs() -> list[dict]:
    """48-row support-CV route after next48fe.

    next48fe reduced training loss and fitted support labels, but query old/new
    still collapsed. This route promotes query-free support leave-one-out
    generalization into the adapter selector, so low-rank adaptation is chosen
    only when it preserves class boundaries under held-out support probes.
    """

    specs = [dict(spec) for spec in _oa_mse_sourcerisk48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_cv_adapter_selection_after_next48fe"
        spec["evidence_ref"] = (
            "next48fe_complete_negative_old_mean_0p2131_seen_new_mean_0p1229_unknown_rejection_mean_0p5920;"
            "support_old_acc_0p9726_support_seen_new_acc_0p9240_query_old_new_collapse;"
            "old_vs_unknown_rejection_corr_minus_0p9127;"
            "source_looo_reject_old_more_than_unknown"
        )
        spec["description"] = (
            "Support-CV conservative arm: select adapter alpha by leave-one-out support generalization first, then old "
            "source/bridge retention, and only then pseudo-unknown risk; this targets the support-fit/query-collapse failure."
            if not aggressive
            else "Support-CV aggressive arm: keep the same leave-one-out selector but allows stronger pseudo-boundary pressure "
            "and residual adapters only after support-CV and old-retention signals remain viable."
        )
        spec["support_reconstruction_arbitration"] = False
        spec["source_looo_risk_arbitration"] = bool(aggressive or strictness >= 1)
        spec["identity_consensus_arbitration"] = True
        spec["density_shell_gate"] = bool(strictness <= 1)
        spec["support_conformal_arbitration"] = bool(not aggressive and strictness == 2)
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"

        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "support_cv_risk_balanced"
        spec["known_coverage_weight"] = (2.70, 2.90, 3.10)[strictness] if not aggressive else (1.80, 2.05, 2.30)[strictness]
        spec["support_center_ce"] = (0.42, 0.54, 0.66)[strictness] if not aggressive else (0.56, 0.74, 0.92)[strictness]
        spec["support_center_margin"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["soft_proto"] = (0.34, 0.42, 0.50)[strictness] if not aggressive else (0.46, 0.62, 0.78)[strictness]
        spec["soft_proto_boundary"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.30, 0.44, 0.58)[strictness]
        spec["soft_proto_boundary_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.22, 0.32, 0.42)[strictness]
        spec["soft_proto_topk"] = 3 if slot in {"C", "E", "F"} else 2
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = int(spec["soft_proto_topk"])
        spec["multiproto_temperature"] = 0.12 if not aggressive else 0.08
        spec["multiproto_score_weight"] = (1.45, 1.60, 1.75)[strictness] if not aggressive else (1.35, 1.60, 1.85)[strictness]

        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.12, 0.20, 0.28)[strictness]
        spec["source_looo_unknown_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.42, 0.54, 0.66)[strictness]
        spec["source_looo_interclass_margin"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["source_looo_risk_quantile"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["source_looo_risk_slack"] = (0.14, 0.10, 0.06)[strictness] if not aggressive else (0.06, 0.02, -0.02)[strictness]
        spec["source_looo_risk_background_score"] = (0.98, 0.95, 0.92)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["source_looo_risk_background_margin"] = (0.34, 0.26, 0.18)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 4 if not aggressive else 3
        spec["source_looo_risk_reject_action"] = "defer" if strictness == 0 else "reject"

        spec["unknown_moat"] = (0.00, 0.03, 0.06)[strictness] if not aggressive else (0.18, 0.28, 0.38)[strictness]
        spec["unknown_margin"] = (0.22, 0.28, 0.34)[strictness] if not aggressive else (0.48, 0.62, 0.76)[strictness]
        spec["negative_anchor_weight"] = 0.0 if not aggressive else (0.06, 0.12, 0.18)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.02, 0.05, 0.08)[strictness]
        spec["target_ring_samples"] = max(int(spec.get("target_ring_samples", 0)), 8 if not aggressive else 16)
        spec["target_halo_samples"] = max(int(spec.get("target_halo_samples", 0)), 4 if not aggressive else 8)
        spec["target_shift_samples"] = max(int(spec.get("target_shift_samples", 0)), 2 if not aggressive else 5)

        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.00, 0.01, 0.02)[strictness] if not aggressive else (0.03, 0.05, 0.07)[strictness]
        spec["support_retention_guard_slack"] = (0.26, 0.24, 0.22)[strictness] if not aggressive else (0.16, 0.13, 0.10)[strictness]
        spec["old_retention_quantile"] = (0.99, 0.98, 0.96)[strictness] if not aggressive else (0.94, 0.90, 0.86)[strictness]
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if strictness >= 1 else "low_rank")
        spec["steps"] = min(144 if not aggressive else 164, max(108 if not aggressive else 124, int(spec.get("steps", 40)) + 12))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 11400
    return specs


def _oa_mse_bgcap48_stage_specs() -> list[dict]:
    """48-row support-calibrated background-cap route after next48ff.

    next48ff proved that support-CV alone is not a query-generalization proxy:
    conservative rows kept known coverage but accepted unknown as old, while
    aggressive rows rejected unknown by rejecting nearly all known rows. This
    route adds a class-conditional support background cap to identity consensus
    so known acceptance must also stay within the support-calibrated pseudo-
    unknown risk envelope.
    """

    specs = [dict(spec) for spec in _oa_mse_supportcv48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_calibrated_background_cap_after_next48ff"
        spec["evidence_ref"] = (
            "next48ff_complete_negative_old_mean_0p2351_seen_new_mean_0p1316_unknown_rejection_mean_0p5646;"
            "conservative_unknown_far_0p791_aggressive_old_mean_0p0697;"
            "coverage_vs_unknown_far_corr_0p989;identity_consensus_accept_caused_unknown_to_old"
        )
        spec["description"] = (
            "BGCAP conservative arm: keep known-retention losses, but require identity-consensus accepts to pass a "
            "support-calibrated pseudo-background cap; add mild learnable background pressure instead of post-hoc reject."
            if not aggressive
            else "BGCAP aggressive arm: stronger background cap and pseudo-unknown pressure, paired with retention rescue so "
            "unknown rejection is not obtained by rejecting all old/seen-new queries."
        )
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_support_background_cap_quantile"] = (0.96, 0.93, 0.90)[strictness] if not aggressive else (0.90, 0.86, 0.82)[strictness]
        spec["identity_consensus_support_background_cap_slack"] = (0.14, 0.10, 0.07)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["identity_consensus_support_background_cap_min_anchors"] = 2
        spec["identity_consensus_background_accept_margin"] = (0.34, 0.28, 0.22)[strictness] if not aggressive else (0.18, 0.12, 0.08)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.80, 0.76)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.12, 0.08, 0.04)[strictness]
        spec["identity_consensus_reject_min_identity_failures"] = 4 if not aggressive else 3

        spec["void_background"] = (0.03, 0.05, 0.07)[strictness] if not aggressive else (0.08, 0.11, 0.14)[strictness]
        spec["negative_anchor_weight"] = (0.04, 0.07, 0.10)[strictness] if not aggressive else (0.14, 0.20, 0.26)[strictness]
        spec["negative_anchor_margin"] = (0.08, 0.10, 0.12)[strictness] if not aggressive else (0.14, 0.18, 0.22)[strictness]
        spec["negative_anchor_temperature"] = 0.12 if not aggressive else 0.10
        spec["three_way_head_weight"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.22, 0.28, 0.34)[strictness]
        spec["three_way_head_pseudo_ce_weight"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.30, 0.40, 0.50)[strictness]
        spec["three_way_accept_prob"] = (0.54, 0.58, 0.62)[strictness] if not aggressive else (0.60, 0.64, 0.68)[strictness]
        spec["three_way_reject_prob"] = (0.82, 0.80, 0.78)[strictness] if not aggressive else (0.78, 0.74, 0.70)[strictness]
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_max_background_score"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.70, 0.66, 0.62)[strictness]
        spec["pre_reject_reject_background_score"] = (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.80, 0.76, 0.72)[strictness]
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_max_background_score"] = (0.70, 0.66, 0.62)[strictness] if not aggressive else (0.62, 0.58, 0.54)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.02, -0.02, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["support_conformal_arbitration"] = bool(not aggressive)
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "support_cv_risk_balanced"
        spec["known_coverage_weight"] = (2.6, 2.8, 3.0)[strictness] if not aggressive else (2.1, 2.3, 2.5)[strictness]
        spec["unknown_moat"] = (0.05, 0.09, 0.13)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["unknown_margin"] = (0.30, 0.38, 0.46)[strictness] if not aggressive else (0.56, 0.68, 0.80)[strictness]
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 12500
    return specs


def _oa_mse_kret48_stage_specs() -> list[dict]:
    """48-row known-retention-first route after next48fg.

    next48fg improved unknown rejection by rejecting old/seen-new as well. This
    route gives support-consistent known neighborhoods a pre-reject retention
    path unless pseudo-background evidence is extreme.
    """

    specs = [dict(spec) for spec in _oa_mse_supportcv48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_neighborhood_known_retention_after_next48fg"
        spec["evidence_ref"] = (
            "next48fg_bgcap_complete_negative_old_mean_0p1580_seen_new_mean_0p0903;"
            "unknown_rejection_mean_0p7851_target_hits_0;"
            "pre_reject_defer_9538_identity_accept_3510_known_rejected_with_unknown"
        )
        spec["description"] = (
            "KRET conservative arm: make support-neighborhood evidence override non-extreme pseudo-background risk, "
            "so target-old and seen-new query rows are not lost before the verifier."
            if not aggressive
            else "KRET aggressive arm: keep strong pseudo-unknown pressure, but only allow background veto to beat known "
            "support evidence when the background score or margin is extreme."
        )
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = False
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_old_min_evidence_delta"] = (-0.06, -0.03, 0.00)[strictness] if not aggressive else (-0.01, 0.02, 0.05)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_delta"] = (-0.12, -0.09, -0.06)[strictness] if not aggressive else (-0.08, -0.05, -0.02)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_margin"] = (-0.08, -0.05, -0.02)[strictness] if not aggressive else (-0.04, -0.02, 0.00)[strictness]
        spec["pre_reject_support_retention_old_min_score_margin"] = (-0.16, -0.12, -0.08)[strictness] if not aggressive else (-0.10, -0.06, -0.02)[strictness]
        spec["pre_reject_support_retention_seen_new_min_evidence_delta"] = (-0.05, -0.02, 0.01)[strictness] if not aggressive else (0.00, 0.03, 0.06)[strictness]
        spec["pre_reject_support_retention_seen_new_min_anchor_delta"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.07, -0.04, -0.01)[strictness]
        spec["pre_reject_support_retention_seen_new_min_score_margin"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["pre_reject_support_retention_max_background_score"] = (0.99, 0.97, 0.95)[strictness] if not aggressive else (0.95, 0.92, 0.89)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.44, 0.36, 0.28)[strictness] if not aggressive else (0.26, 0.18, 0.10)[strictness]

        spec["pre_reject_max_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.80, 0.74, 0.68)[strictness]
        spec["pre_reject_max_background_margin"] = (0.26, 0.20, 0.14)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["pre_reject_defer_background_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.74, 0.68, 0.62)[strictness]
        spec["pre_reject_defer_background_margin"] = (0.20, 0.14, 0.08)[strictness] if not aggressive else (0.08, 0.02, -0.04)[strictness]
        spec["pre_reject_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.36, 0.28, 0.20)[strictness] if not aggressive else (0.18, 0.10, 0.04)[strictness]
        spec["pre_reject_defer_action"] = "uncertain" if not aggressive else ("uncertain" if slot in {"A", "B"} else "defer")

        spec["retention_rescue_gate"] = bool(not aggressive or strictness == 0)
        spec["retention_rescue_max_background_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.72, 0.68, 0.64)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.18, 0.12, 0.06)[strictness] if not aggressive else (0.04, 0.00, -0.04)[strictness]
        spec["source_looo_risk_arbitration"] = bool(aggressive and strictness >= 1)
        spec["source_looo_risk_reject_action"] = "defer" if aggressive and strictness == 1 else "reject"
        spec["support_conformal_arbitration"] = bool(not aggressive and strictness == 2)
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["known_coverage_weight"] = (3.05, 3.25, 3.45)[strictness] if not aggressive else (2.10, 2.35, 2.60)[strictness]
        spec["support_center_ce"] = (0.62, 0.74, 0.86)[strictness] if not aggressive else (0.74, 0.92, 1.10)[strictness]
        spec["support_center_margin"] = (0.18, 0.22, 0.26)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]
        spec["soft_proto"] = (0.52, 0.62, 0.72)[strictness] if not aggressive else (0.64, 0.82, 1.00)[strictness]
        spec["soft_proto_boundary"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.38, 0.52, 0.66)[strictness]
        spec["soft_proto_topk"] = 3
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 3
        spec["multiproto_score_weight"] = (1.70, 1.90, 2.10)[strictness] if not aggressive else (1.55, 1.80, 2.05)[strictness]
        spec["unknown_moat"] = (0.02, 0.05, 0.08)[strictness] if not aggressive else (0.22, 0.34, 0.46)[strictness]
        spec["unknown_margin"] = (0.24, 0.32, 0.40)[strictness] if not aggressive else (0.58, 0.72, 0.86)[strictness]
        spec["negative_anchor_weight"] = (0.00, 0.03, 0.06)[strictness] if not aggressive else (0.12, 0.20, 0.28)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.04, 0.08, 0.12)[strictness]
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "support_cv_risk_balanced"
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if strictness >= 1 else "low_rank")
        spec["steps"] = min(148 if not aggressive else 172, max(112 if not aggressive else 132, int(spec.get("steps", 40)) + 16))
        spec["target_ring_samples"] = max(int(spec.get("target_ring_samples", 0)), 8 if not aggressive else 18)
        spec["target_halo_samples"] = max(int(spec.get("target_halo_samples", 0)), 4 if not aggressive else 9)
        spec["target_shift_samples"] = max(int(spec.get("target_shift_samples", 0)), 2 if not aggressive else 5)
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 13600
    return specs


def _oa_mse_riskret48_stage_specs() -> list[dict]:
    """48-row source-risk-constrained known-retention route after next48fh.

    KRET48 recovered known accuracy only by re-accepting many rows that looked
    unknown-like under source leave-one-old-out risk. This route keeps the
    support-neighborhood retention idea, but requires it to pass source-domain
    impostor-risk evidence before it can override pseudo-background pressure.
    """

    specs = [dict(spec) for spec in _oa_mse_kret48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "source_risk_constrained_support_retention_after_next48fh"
        spec["evidence_ref"] = (
            "next48fh_kret_complete_negative_old_mean_0p3205_seen_new_mean_0p1486;"
            "unknown_rejection_mean_0p3500_unknown_far_mean_0p6500;"
            "old_unknown_far_corr_0p870_support_retention_unknown_far_corr_0p806"
        )
        spec["description"] = (
            "RISKRET conservative arm: keep support-neighborhood known retention, but block retention when source "
            "leave-one-old-out risk has already rejected the row or reports more than one failure."
            if not aggressive
            else "RISKRET aggressive arm: require zero source leave-one-old-out failures before support retention can "
            "override pseudo-background risk, targeting unknown FAR without returning to BGCAP known collapse."
        )
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer" if not aggressive and strictness == 0 else "reject"
        spec["source_looo_risk_reject_min_failures"] = (3, 2, 2)[strictness] if not aggressive else (2, 2, 1)[strictness]
        spec["source_looo_risk_quantile"] = (0.70, 0.74, 0.78)[strictness] if not aggressive else (0.78, 0.82, 0.86)[strictness]
        spec["source_looo_risk_slack"] = (-0.04, -0.02, 0.0)[strictness] if not aggressive else (-0.01, 0.0, 0.02)[strictness]
        spec["source_looo_risk_min_score_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["source_looo_risk_min_known_evidence_delta"] = (-0.04, -0.02, 0.00)[strictness] if not aggressive else (0.00, 0.02, 0.04)[strictness]
        spec["pre_reject_support_retention_require_source_looo_pass"] = True
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["pre_reject_support_retention_max_background_score"] = (
            min(float(spec.get("pre_reject_support_retention_max_background_score", 0.96)), (0.94, 0.92, 0.90)[strictness])
            if not aggressive
            else min(float(spec.get("pre_reject_support_retention_max_background_score", 0.90)), (0.88, 0.84, 0.80)[strictness])
        )
        spec["pre_reject_support_retention_max_background_margin"] = (
            min(float(spec.get("pre_reject_support_retention_max_background_margin", 0.30)), (0.22, 0.16, 0.10)[strictness])
            if not aggressive
            else min(float(spec.get("pre_reject_support_retention_max_background_margin", 0.10)), (0.08, 0.04, 0.00)[strictness])
        )
        spec["known_coverage_weight"] = (2.70, 2.90, 3.10)[strictness] if not aggressive else (1.80, 2.05, 2.30)[strictness]
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.34, 0.46, 0.58)[strictness]
        spec["negative_anchor_weight"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 900
    return specs


def _oa_mse_manifold48_stage_specs() -> list[dict]:
    """48-row support-manifold soft-prototype route after next48fi.

    RISKRET48 proved that post-hoc background risk can trade unknown rejection
    against known retention, but it does not teach the head a stable known
    manifold. This route keeps unknown query eval-only and adds a query-time
    same-class soft-prototype consistency check: a row must be explainable by a
    convex mixture of allowed same-class support/source anchors before it is
    trusted as known.
    """

    specs = [dict(spec) for spec in _oa_mse_riskret48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "support_manifold_soft_prototype_consistency_after_next48fi"
        spec["evidence_ref"] = (
            "next48fi_riskret_complete_negative_old_mean_0p2773_seen_new_mean_0p1524;"
            "unknown_rejection_mean_0p4545_old_unknown_far_corr_0p894;"
            "loss_normal_but_soft_proto_weighted_dominates_without_query_generalization"
        )
        spec["description"] = (
            "MANIFOLD conservative arm: keep support-CV/source-risk selection, but require accepted rows to be "
            "consistent with a same-class soft prototype mixture; failed rows defer to later verifier instead of hard reject."
            if not aggressive
            else "MANIFOLD aggressive arm: add a stricter same-class soft mixture consistency gate and stronger "
            "pseudo-background basin to test whether unknown leakage can fall without the all-known collapse."
        )
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 3 if int(spec.get("k_old", 5)) >= 10 else 2
        spec["multiproto_temperature"] = (0.10, 0.085, 0.07)[strictness] if not aggressive else (0.075, 0.06, 0.05)[strictness]
        spec["multiproto_score_weight"] = (0.55, 0.70, 0.85)[strictness] if not aggressive else (0.85, 1.0, 1.0)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_action"] = ("uncertain", "uncertain", "defer")[strictness] if not aggressive else ("defer", "reject", "reject")[strictness]
        spec["mixture_consistency_min_cos"] = (0.18, 0.22, 0.26)[strictness] if not aggressive else (0.28, 0.34, 0.40)[strictness]
        spec["mixture_consistency_max_residual"] = (0.16, 0.13, 0.10)[strictness] if not aggressive else (0.095, 0.075, 0.055)[strictness]
        spec["mixture_consistency_min_margin"] = (-2.0, -1.2, -0.6)[strictness] if not aggressive else (-0.4, 0.0, 0.3)[strictness]
        spec["soft_proto"] = max(float(spec.get("soft_proto", 0.0)), (0.55, 0.70, 0.85)[strictness] if not aggressive else (0.95, 1.10, 1.25)[strictness])
        spec["soft_proto_boundary"] = max(float(spec.get("soft_proto_boundary", 0.0)), (0.30, 0.40, 0.50)[strictness] if not aggressive else (0.55, 0.70, 0.85)[strictness])
        spec["soft_proto_boundary_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.40, 0.48, 0.56)[strictness]
        spec["support_center"] = max(float(spec.get("support_center", 0.0)), (0.70, 0.90, 1.10)[strictness] if not aggressive else (1.15, 1.35, 1.55)[strictness])
        spec["support_center_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.32, 0.40, 0.48)[strictness]
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (2.6, 2.9, 3.2)[strictness] if not aggressive else (2.0, 2.3, 2.6)[strictness])
        spec["unknown_moat"] = (0.08, 0.12, 0.18)[strictness] if not aggressive else (0.42, 0.56, 0.70)[strictness]
        spec["negative_anchor_weight"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.30, 0.42, 0.54)[strictness]
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "identity_preserving_risk_cv"
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer" if not aggressive else "reject"
        spec["pre_reject_support_retention_require_source_looo_pass"] = True
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 950
    return specs


def _oa_mse_h06_evid48_stage_specs() -> list[dict]:
    """48-row H06 latest-model multi-evidence route.

    This route keeps the Stage2 data protocol unchanged and switches only the
    ground/source backbone default to H06 low-prob hybrid latest_model.pth. It
    intentionally exposes many decision parameters in score tables first; later
    runs can prune the expensive or redundant evidence sources after attribution.
    """

    specs = [dict(spec) for spec in _oa_mse_manifold48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "h06_latest_multi_evidence_stage2_oa_mse"
        spec["evidence_ref"] = (
            "user_requested_h06_low_prob_hybrid_latest_model_default;"
            "next48fj_manifold_complete_48_metrics_48_score_tables;"
            "multi_evidence_old_drift_seen_new_support_knn_unknown_background"
        )
        spec["description"] = (
            "H06 conservative multi-evidence: use H06 latest_model source prototypes, old drift/support-quality, "
            "support-kNN density, soft-mixture head, identity consensus, and pre-reject defer to maximize known retention "
            "before pruning evidence sources."
            if not aggressive
            else "H06 aggressive multi-evidence: use H06 latest_model with stronger background, negative-anchor, "
            "three-way, support reconstruction/conformal, and coupled verifier veto to stress unknown rejection without "
            "using unknown query for fitting."
        )
        spec["source_proto_per_tx"] = 48 if not aggressive else 64
        spec["source_query_per_tx"] = 24 if not aggressive else 32
        spec["sfe_max_samples_per_tx"] = 240 if not aggressive else 320
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = (3, 3, 4)[strictness] if not aggressive else (4, 4, 5)[strictness]
        spec["multiproto_temperature"] = (0.095, 0.080, 0.065)[strictness] if not aggressive else (0.070, 0.055, 0.045)[strictness]
        spec["multiproto_score_weight"] = (0.70, 0.85, 1.00)[strictness] if not aggressive else (1.05, 1.25, 1.45)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_action"] = ("uncertain", "defer", "defer")[strictness] if not aggressive else ("defer", "reject", "reject")[strictness]
        spec["mixture_consistency_min_cos"] = (0.16, 0.20, 0.24)[strictness] if not aggressive else (0.26, 0.32, 0.38)[strictness]
        spec["mixture_consistency_max_residual"] = (0.18, 0.14, 0.11)[strictness] if not aggressive else (0.10, 0.075, 0.055)[strictness]
        spec["mixture_consistency_min_margin"] = (-1.8, -1.0, -0.4)[strictness] if not aggressive else (-0.3, 0.1, 0.35)[strictness]
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain" if not aggressive else ("uncertain", "reject", "reject")[strictness]
        spec["anchor_density_quantile"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.06, 0.08, 0.10)[strictness]
        spec["anchor_density_margin_quantile"] = (0.02, 0.03, 0.04)[strictness] if not aggressive else (0.05, 0.07, 0.09)[strictness]
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["support_conformal_arbitration"] = bool((not aggressive and strictness >= 1) or aggressive)
        spec["support_reconstruction_arbitration"] = bool(aggressive and strictness >= 1)
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_unknown_weight"] = max(float(spec.get("source_looo_unknown_weight", 0.0)), 0.18 if not aggressive else 0.34)
        spec["source_looo_risk_reject_action"] = "defer" if not aggressive else "reject"
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (2.8, 3.1, 3.4)[strictness] if not aggressive else (2.4, 2.8, 3.2)[strictness])
        spec["support_center"] = max(float(spec.get("support_center", 0.0)), (0.80, 1.00, 1.20)[strictness] if not aggressive else (1.20, 1.45, 1.70)[strictness])
        spec["support_center_margin"] = (0.16, 0.22, 0.28)[strictness] if not aggressive else (0.28, 0.36, 0.44)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(aggressive or strictness >= 1)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["retention_rescue_gate"] = True
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.66, 0.70, 0.74)[strictness] if not aggressive else (0.76, 0.82, 0.88)[strictness]
        spec["two_branch_bg_min_margin"] = (-0.04, 0.00, 0.04)[strictness] if not aggressive else (0.04, 0.10, 0.16)[strictness]
        spec["seen_new_registration_override"] = bool(int(spec.get("k_new", 0)) > 0)
        spec["seen_new_override_min_seen_vs_old_evidence_margin"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.06, 0.09, 0.12)[strictness]
        spec["three_way_decision_head"] = True
        spec["three_way_head_weight"] = max(float(spec.get("three_way_head_weight", 0.0)), (0.18, 0.26, 0.34)[strictness] if not aggressive else (0.40, 0.55, 0.70)[strictness])
        spec["three_way_decision_policy"] = "evidence_balanced" if not aggressive else "class_first"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "defer" if not aggressive else "accept"
        spec["three_way_reject_prob"] = (0.62, 0.66, 0.70)[strictness] if not aggressive else (0.72, 0.78, 0.84)[strictness]
        spec["three_way_known_background_margin"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.06, 0.10, 0.14)[strictness]
        spec["unknown_moat"] = (0.10, 0.16, 0.24)[strictness] if not aggressive else (0.46, 0.62, 0.78)[strictness]
        spec["unknown_margin"] = (0.46, 0.52, 0.58)[strictness] if not aggressive else (0.62, 0.72, 0.82)[strictness]
        spec["negative_anchor_weight"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.36, 0.50, 0.64)[strictness]
        spec["negative_anchor_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["siamese_unknown_veto"] = True
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["siamese_threshold"] = (0.62, 0.68, 0.74)[strictness] if not aggressive else (0.76, 0.82, 0.88)[strictness]
        spec["min_veto_failures"] = 4 if not aggressive else 3
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 5 if not aggressive else 4
        spec["guard_min_margin"] = (0.08, 0.12, 0.18)[strictness] if not aggressive else (0.18, 0.26, 0.34)[strictness]
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "identity_preserving_risk_cv"
        spec["old_retention_quantile"] = 0.80 if not aggressive else 0.76
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.82
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1240
    return specs


def _oa_mse_h06_arb48_stage_specs() -> list[dict]:
    """48-row H06 arbitration repair after next48fk.

    next48fk proved that H06 latest_model emits useful old-drift, support-kNN,
    background, and pair-verifier evidence, but the decision cascade did not use
    that evidence strongly enough. This route keeps the same Stage2-C protocol
    and turns those measurements into explicit old/new/unknown arbitration.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_evid48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "h06_arbitration_repair_after_next48fk"
        spec["evidence_ref"] = (
            "next48fk_h06_evid48_complete_negative;"
            "raw_old_mean_0p2382_raw_seen_new_mean_0p0368_unknown_far_0p3733;"
            "conservative_old_0p4009_far_0p6889_aggressive_old_0p0755_far_0p0576;"
            "pair_verifier_called_0p2339_veto_0p0_support_knn_seen_new_minus_old_mean_-0p3278"
        )
        spec["description"] = (
            "H06-ARB conservative: preserve old/seen-new support-consistent rows, but require seen-new override to "
            "pass support-kNN separation and make pair-verifier veto actionable before accepting ambiguous known rows."
            if not aggressive
            else "H06-ARB aggressive: prioritize unknown rejection with early pair-verifier/background veto, while "
            "allowing only support-kNN and source-risk-consistent known retention."
        )
        spec["stage2_max_active_per_gpu"] = 2
        spec["source_proto_per_tx"] = 64
        spec["source_query_per_tx"] = 32
        spec["sfe_max_samples_per_tx"] = 320
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["adapter_selection_policy"] = (
            "support_cv_constrained" if not aggressive else "identity_preserving_risk_cv"
        )
        spec["adapter_kind"] = "low_rank" if not aggressive else ("residual_mlp" if strictness >= 1 else "low_rank")
        spec["steps"] = min(104 if not aggressive else 128, max(72 if not aggressive else 88, int(spec.get("steps", 40))))

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = (3, 4, 4)[strictness] if not aggressive else (4, 5, 5)[strictness]
        spec["multiproto_temperature"] = (0.09, 0.075, 0.06)[strictness] if not aggressive else (0.065, 0.05, 0.04)[strictness]
        spec["multiproto_score_weight"] = (0.85, 1.00, 1.15)[strictness] if not aggressive else (1.20, 1.45, 1.70)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_action"] = ("uncertain", "defer", "defer")[strictness] if not aggressive else ("defer", "reject", "reject")[strictness]
        spec["mixture_consistency_min_cos"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.26, 0.32, 0.38)[strictness]
        spec["mixture_consistency_max_residual"] = (0.20, 0.16, 0.12)[strictness] if not aggressive else (0.10, 0.075, 0.055)[strictness]
        spec["mixture_consistency_min_margin"] = (-1.4, -0.8, -0.3)[strictness] if not aggressive else (-0.2, 0.15, 0.45)[strictness]

        spec["known_coverage_weight"] = (3.2, 3.5, 3.8)[strictness] if not aggressive else (2.2, 2.5, 2.8)[strictness]
        spec["support_center"] = (1.05, 1.25, 1.45)[strictness] if not aggressive else (1.35, 1.65, 1.95)[strictness]
        spec["support_center_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.32, 0.40, 0.48)[strictness]
        spec["soft_proto"] = (0.80, 0.95, 1.10)[strictness] if not aggressive else (1.10, 1.35, 1.60)[strictness]
        spec["soft_proto_boundary"] = (0.36, 0.46, 0.56)[strictness] if not aggressive else (0.62, 0.78, 0.94)[strictness]
        spec["soft_proto_boundary_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.42, 0.52, 0.62)[strictness]

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = True
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["pre_reject_support_retention_max_background_score"] = (
            (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        )
        spec["pre_reject_support_retention_max_background_margin"] = (
            (0.16, 0.10, 0.04)[strictness] if not aggressive else (0.04, -0.02, -0.08)[strictness]
        )
        spec["pre_reject_max_background_score"] = (
            (0.82, 0.78, 0.74)[strictness] if not aggressive else (0.70, 0.64, 0.58)[strictness]
        )
        spec["pre_reject_reject_background_score"] = (
            (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        )
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer" if not aggressive else "reject"
        spec["source_looo_risk_reject_min_failures"] = (2, 2, 1)[strictness] if not aggressive else (2, 1, 1)[strictness]
        spec["source_looo_risk_background_score"] = (0.84, 0.80, 0.76)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["source_looo_risk_background_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (0.06, 0.02, -0.04)[strictness]

        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.70, 0.74, 0.78)[strictness] if not aggressive else (0.78, 0.84, 0.90)[strictness]
        spec["two_branch_bg_min_margin"] = (-0.02, 0.02, 0.06)[strictness] if not aggressive else (0.06, 0.12, 0.18)[strictness]
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = (3, 2, 2)[strictness] if not aggressive else (2, 2, 1)[strictness]
        spec["guard_min_margin"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.12, 0.18, 0.24)[strictness]

        spec["siamese_unknown_veto"] = True
        spec["siamese_unknown_veto_mode"] = ("coupled", "any", "any")[strictness] if not aggressive else ("any", "any", "coupled")[strictness]
        spec["siamese_threshold"] = (0.56, 0.60, 0.64)[strictness] if not aggressive else (0.62, 0.68, 0.74)[strictness]
        spec["min_old_support_evidence_delta"] = (-0.12, -0.06, 0.00)[strictness] if not aggressive else (-0.02, 0.02, 0.06)[strictness]
        spec["min_old_surrogate_reject_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (0.00, 0.04, 0.08)[strictness]
        spec["min_energy_delta"] = (-10.0, -5.0, 0.0)[strictness] if not aggressive else (-2.0, 2.0, 6.0)[strictness]
        spec["min_mahalanobis_delta"] = (-30.0, -20.0, -10.0)[strictness] if not aggressive else (-15.0, -5.0, 5.0)[strictness]
        spec["min_accept_delta"] = (-18.0, -12.0, -6.0)[strictness] if not aggressive else (-8.0, -2.0, 4.0)[strictness]
        spec["min_old_support_anchor_margin"] = (0.00, 0.015, 0.030)[strictness] if not aggressive else (0.020, 0.040, 0.060)[strictness]
        spec["min_veto_failures"] = (2, 1, 1)[strictness] if not aggressive else (1, 1, 2)[strictness]

        if int(spec.get("k_new", 0)) > 0:
            spec["seen_new_registration_override"] = True
            spec["seen_new_override_min_seen_vs_old_evidence_margin"] = (
                (-0.02, 0.00, 0.03)[strictness] if not aggressive else (0.00, 0.04, 0.08)[strictness]
            )
            spec["seen_new_override_min_score_margin"] = (
                (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
            )
            spec["seen_new_override_max_background_score"] = (
                (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
            )
            spec["seen_new_override_max_background_margin"] = (
                (0.20, 0.14, 0.08)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
            )
            spec["seen_new_override_min_support_knn_seen_new_minus_old"] = (
                (-0.30, -0.25, -0.20)[strictness] if not aggressive else (-0.25, -0.20, -0.15)[strictness]
            )
            spec["seen_new_override_min_support_knn_margin"] = (
                (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.02, 0.04, 0.06)[strictness]
            )

        spec["unknown_moat"] = (0.14, 0.20, 0.28)[strictness] if not aggressive else (0.50, 0.68, 0.86)[strictness]
        spec["unknown_margin"] = (0.48, 0.56, 0.64)[strictness] if not aggressive else (0.66, 0.78, 0.90)[strictness]
        spec["negative_anchor_weight"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.42, 0.58, 0.74)[strictness]
        spec["negative_anchor_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["retention_rescue_gate"] = bool(not aggressive)
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["three_way_reject_prob"] = (0.66, 0.70, 0.74)[strictness] if not aggressive else (0.70, 0.76, 0.82)[strictness]
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.82
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1510
    return specs


def _oa_mse_h06_oldunk48_stage_specs() -> list[dict]:
    """48-row H06 old/unknown-only identity-first late-veto route.

    next48fm showed that stronger arbitration made pair-verifier veto active,
    but old/seen-new identity still collapsed when unknown gates were tightened.
    Per the latest user direction, this route removes target-domain new-class
    support from training and isolates target-old improvement plus unknown
    rejection before any later new-class registration experiment.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_arb48_stage_specs()]
    for idx, spec in enumerate(specs):
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        arm = (
            ("identity_only", "pair_late", "old_support_knn")
            if not aggressive
            else ("background_late", "pair_background", "full_stack")
        )[strictness]
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_unknown_identity_first_late_veto_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "next48fm_h06_arb48fix_complete_negative;"
            "old_mean_0p2193_old_max_0p5056_seen_new_mean_0p0413_unknown_far_mean_0p2861;"
            "pair_veto_active_0p0528_but_old_seen_new_identity_weak;"
            "user_requested_split_targets_old_unknown_first_new_class_later;"
            "no_target_new_support_or_training_mix"
        )
        spec["description"] = (
            "H06-OLDUNK conservative identity-first ablation: use target-old support, multi-prototype, "
            "support-kNN, drift and identity-consensus evidence before any hard unknown veto; no target-new support."
            if not aggressive
            else "H06-OLDUNK aggressive late-veto ablation: keep target-old identity assignment first, then test "
            "pair verifier and background gates as post-identity unknown veto modules; new classes are excluded."
        )

        # Keep the H06 latest_model route and feature budget, but reduce adapter
        # overfit pressure so support memorization does not dominate query old.
        spec["source_proto_per_tx"] = 72 if not aggressive else 80
        spec["source_query_per_tx"] = 36 if not aggressive else 40
        spec["sfe_max_samples_per_tx"] = 360 if not aggressive else 400
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "identity_preserving_cv"
        spec["adapter_kind"] = "low_rank" if strictness != 2 else "residual_mlp"
        spec["steps"] = min(96 if not aggressive else 112, max(64 if not aggressive else 76, int(spec.get("steps", 64))))

        # Identity assignment first: make known evidence permissive enough to
        # expose whether the identity features can separate old/new before veto.
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = (4, 4, 5)[strictness] if not aggressive else (4, 5, 5)[strictness]
        spec["multiproto_temperature"] = (0.085, 0.070, 0.055)[strictness] if not aggressive else (0.070, 0.055, 0.045)[strictness]
        spec["multiproto_score_weight"] = (1.00, 1.10, 1.25)[strictness] if not aggressive else (1.20, 1.35, 1.55)[strictness]
        spec["known_coverage_weight"] = (4.0, 4.3, 4.6)[strictness] if not aggressive else (3.1, 3.4, 3.7)[strictness]
        spec["support_center"] = (1.20, 1.40, 1.60)[strictness] if not aggressive else (1.45, 1.70, 1.95)[strictness]
        spec["support_center_margin"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.20, 0.26, 0.32)[strictness]
        spec["soft_proto"] = (1.00, 1.15, 1.30)[strictness] if not aggressive else (1.20, 1.40, 1.60)[strictness]
        spec["soft_proto_boundary"] = (0.28, 0.36, 0.44)[strictness] if not aggressive else (0.45, 0.58, 0.72)[strictness]
        spec["soft_proto_boundary_margin"] = (0.16, 0.22, 0.28)[strictness] if not aggressive else (0.28, 0.38, 0.48)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_action"] = "uncertain" if arm in {"identity_only", "support_knn"} else "defer"
        spec["mixture_consistency_min_cos"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.16, 0.20, 0.24)[strictness]
        spec["mixture_consistency_max_residual"] = (0.26, 0.22, 0.18)[strictness] if not aggressive else (0.18, 0.14, 0.10)[strictness]
        spec["mixture_consistency_min_margin"] = (-2.0, -1.4, -0.8)[strictness] if not aggressive else (-0.8, -0.3, 0.1)[strictness]
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain"
        spec["anchor_density_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.03, 0.05, 0.07)[strictness]
        spec["anchor_density_margin_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.03, 0.04, 0.06)[strictness]

        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_old_min_evidence_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.20, -0.16, -0.12)[strictness] if not aggressive else (-0.14, -0.10, -0.06)[strictness]
        spec["identity_consensus_old_min_density_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["identity_consensus_seen_new_min_evidence_delta"] = 0.0
        spec["identity_consensus_seen_new_min_anchor_delta"] = 0.0
        spec["identity_consensus_seen_new_min_density_delta"] = 0.0
        spec["identity_consensus_min_identity_margin"] = (-0.10, -0.06, -0.02)[strictness]
        spec["identity_consensus_background_accept_margin"] = (0.35, 0.30, 0.25)[strictness] if not aggressive else (0.26, 0.20, 0.14)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.98, 0.96, 0.94)[strictness] if not aggressive else (0.92, 0.88, 0.84)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.32, 0.26, 0.20)[strictness] if not aggressive else (0.20, 0.14, 0.08)[strictness]
        spec["identity_consensus_reject_min_identity_failures"] = 5 if not aggressive else 4

        # Early reject modules are softened to defer/uncertain so late-veto
        # modules can be attributed cleanly.
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = False if arm in {"identity_only", "old_support_knn"} else True
        spec["pre_reject_support_retention_source_looo_max_failures"] = 2 if not aggressive else 1
        spec["pre_reject_max_background_score"] = (0.96, 0.93, 0.90)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_max_background_margin"] = (0.30, 0.24, 0.18)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["pre_reject_defer_background_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.80, 0.74, 0.68)[strictness]
        spec["pre_reject_reject_background_score"] = (0.99, 0.97, 0.95)[strictness] if not aggressive else (0.94, 0.88, 0.82)[strictness]
        spec["source_looo_risk_arbitration"] = arm not in {"identity_only"}
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_risk_reject_min_failures"] = 3 if not aggressive else 2
        spec["source_looo_risk_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["source_looo_risk_background_margin"] = (0.16, 0.12, 0.08)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]

        # Module switches for attribution.
        pair_enabled = arm in {"pair_late", "pair_background", "full_stack"}
        background_enabled = arm in {"background_late", "pair_background", "full_stack"}
        support_knn_focus = arm in {"old_support_knn", "full_stack"}
        three_way_enabled = background_enabled

        spec["siamese_unknown_veto"] = pair_enabled
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["siamese_threshold"] = (0.50, 0.54, 0.58)[strictness] if not aggressive else (0.56, 0.62, 0.68)[strictness]
        spec["min_old_support_evidence_delta"] = (-0.20, -0.16, -0.12)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["min_old_surrogate_reject_delta"] = (-0.16, -0.12, -0.08)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["min_energy_delta"] = (-16.0, -12.0, -8.0)[strictness] if not aggressive else (-8.0, -4.0, 0.0)[strictness]
        spec["min_mahalanobis_delta"] = (-45.0, -35.0, -25.0)[strictness] if not aggressive else (-25.0, -15.0, -5.0)[strictness]
        spec["min_accept_delta"] = (-28.0, -22.0, -16.0)[strictness] if not aggressive else (-16.0, -10.0, -4.0)[strictness]
        spec["min_old_support_anchor_margin"] = (-0.015, 0.000, 0.015)[strictness] if not aggressive else (0.010, 0.025, 0.040)[strictness]
        spec["min_veto_failures"] = 4 if not aggressive else 3

        spec["old_unknown_acceptance_guard"] = background_enabled
        spec["guard_min_failures"] = 5 if not aggressive else 4
        spec["guard_min_margin"] = (0.18, 0.22, 0.26)[strictness] if not aggressive else (0.22, 0.28, 0.34)[strictness]
        spec["guard_min_old_support_evidence_delta"] = (-0.12, -0.08, -0.04)[strictness]
        spec["guard_min_old_surrogate_reject_delta"] = (-0.08, -0.04, 0.00)[strictness]
        spec["guard_min_energy_delta"] = (-8.0, -4.0, 0.0)[strictness]
        spec["guard_min_mahalanobis_delta"] = (-25.0, -15.0, -5.0)[strictness]
        spec["guard_min_accept_delta"] = (-18.0, -12.0, -6.0)[strictness]
        spec["guard_min_old_support_anchor_margin"] = (0.00, 0.015, 0.030)[strictness]
        spec["two_branch_background_guard"] = background_enabled
        spec["two_branch_bg_min_score"] = (0.86, 0.90, 0.94)[strictness] if not aggressive else (0.80, 0.86, 0.92)[strictness]
        spec["two_branch_bg_min_margin"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.08, 0.14, 0.20)[strictness]
        spec["void_gate"] = arm == "full_stack"
        spec["void_gate_min_score"] = (0.82, 0.86, 0.90)[strictness]
        spec["void_gate_min_margin"] = (0.08, 0.12, 0.16)[strictness]
        spec["three_way_decision_head"] = three_way_enabled
        spec["three_way_head_weight"] = (0.00, 0.18, 0.26)[strictness] if not aggressive else (0.30, 0.42, 0.55)[strictness]
        spec["three_way_decision_policy"] = "class_first"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["three_way_reject_prob"] = (0.82, 0.86, 0.90)[strictness] if not aggressive else (0.76, 0.82, 0.88)[strictness]
        spec["three_way_known_background_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]

        spec["seen_new_registration_override"] = False
        spec["seen_new_override_min_seen_vs_old_evidence_margin"] = 0.0
        spec["seen_new_override_min_score_margin"] = 0.0
        spec["seen_new_override_max_background_score"] = 1.0
        spec["seen_new_override_max_background_margin"] = 1.0
        spec["seen_new_override_min_support_knn_seen_new_minus_old"] = None
        spec["seen_new_override_min_support_knn_margin"] = None
        if support_knn_focus:
            spec["pre_reject_support_retention_old_min_evidence_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.02, 0.02, 0.06)[strictness]
            spec["pre_reject_support_retention_old_min_anchor_delta"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
            spec["pre_reject_support_retention_old_min_anchor_margin"] = (-0.04, -0.02, 0.00)[strictness] if not aggressive else (-0.01, 0.01, 0.03)[strictness]
            spec["pre_reject_support_retention_old_min_score_margin"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]

        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.30, 0.44, 0.58)[strictness]
        spec["unknown_margin"] = (0.42, 0.48, 0.54)[strictness] if not aggressive else (0.56, 0.66, 0.76)[strictness]
        spec["negative_anchor_weight"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.22, 0.34, 0.46)[strictness]
        spec["negative_anchor_margin"] = (0.08, 0.10, 0.12)[strictness] if not aggressive else (0.16, 0.22, 0.28)[strictness]
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_max_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.82
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1780
    return specs


def _oa_mse_h06_bgtrain48_stage_specs() -> list[dict]:
    """48-row H06 old/background learnable-reject route after next48fp.

    next48fp recovered target-old mean accuracy, but unknown FAR stayed high
    because most background modules were late vetoes and the old/unknown-only
    three-way head had no seen-new class, so its background loss was inactive.
    This route keeps Stage2-B old support only and makes the query-free
    pseudo-background objective explicit before testing new-class enrollment.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_oldunk48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        arm = (
            ("old_floor_bg_train", "source_risk_bg_train", "recon_bg_train")
            if not aggressive
            else ("hard_bg_train", "hard_recon_bg_train", "full_bg_train")
        )[strictness]
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_unknown_learned_background_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "next48fp_complete_old_unknown_only;"
            "old_mean_0p5020_old_max_0p6778_unknown_far_mean_0p8479_unknown_far_min_0p4000;"
            "score_table_unknown_group_gap_candidate_group_old_only;"
            "three_way_head_old_only_background_loss_repaired;"
            "new_class_enrollment_deferred_by_user_direction"
        )
        spec["description"] = (
            "H06-BGTRAIN conservative: train an old/background head from target-old support plus query-free "
            "pseudo-background anchors, keeping a permissive old evidence floor so recovered old accuracy is not "
            "thrown away by the rejector."
            if not aggressive
            else "H06-BGTRAIN aggressive: strengthen pseudo-background moat, source leave-one-old-out risk and "
            "support reconstruction before hard unknown rejection, while still using no target-new support."
        )

        # Stage2-B split: only target-old support is used; target-new remains
        # disabled and unknown samples stay eval-only.
        spec["source_proto_per_tx"] = 80 if not aggressive else 96
        spec["source_query_per_tx"] = 40 if not aggressive else 48
        spec["sfe_max_samples_per_tx"] = 400 if not aggressive else 480
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "identity_preserving_risk_cv"
        spec["adapter_kind"] = "low_rank" if not aggressive or strictness == 0 else "residual_mlp"
        spec["steps"] = min(112 if not aggressive else 132, max(80 if not aggressive else 96, int(spec.get("steps", 80)) + 12))

        # Keep the identity signal that worked in next48fp, but reduce
        # unconditional re-acceptance when pseudo-background evidence is high.
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = (4, 5, 5)[strictness] if not aggressive else (5, 5, 6)[strictness]
        spec["multiproto_temperature"] = (0.075, 0.060, 0.050)[strictness] if not aggressive else (0.055, 0.045, 0.038)[strictness]
        spec["multiproto_score_weight"] = (1.15, 1.30, 1.45)[strictness] if not aggressive else (1.40, 1.65, 1.90)[strictness]
        spec["known_coverage_weight"] = (4.4, 4.7, 5.0)[strictness] if not aggressive else (3.2, 3.6, 4.0)[strictness]
        spec["known_coverage_margin"] = (0.14, 0.16, 0.18)[strictness] if not aggressive else (0.18, 0.21, 0.24)[strictness]
        spec["known_coverage_min_affinity"] = (0.40, 0.43, 0.46)[strictness] if not aggressive else (0.44, 0.48, 0.52)[strictness]
        spec["support_center"] = (1.35, 1.55, 1.75)[strictness] if not aggressive else (1.65, 1.95, 2.20)[strictness]
        spec["support_center_margin"] = (0.16, 0.21, 0.26)[strictness] if not aggressive else (0.26, 0.34, 0.42)[strictness]
        spec["soft_proto"] = (1.10, 1.25, 1.40)[strictness] if not aggressive else (1.35, 1.60, 1.85)[strictness]
        spec["soft_proto_boundary"] = (0.36, 0.46, 0.56)[strictness] if not aggressive else (0.62, 0.78, 0.94)[strictness]
        spec["soft_proto_boundary_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.34, 0.44, 0.54)[strictness]

        # Make rejection learnable from pseudo-background, not just a post-hoc
        # threshold. Unknown query labels are still not used for fitting.
        spec["three_way_decision_head"] = True
        spec["three_way_head_weight"] = (0.32, 0.44, 0.56)[strictness] if not aggressive else (0.68, 0.86, 1.05)[strictness]
        spec["three_way_head_temperature"] = (0.24, 0.20, 0.17)[strictness] if not aggressive else (0.18, 0.15, 0.12)[strictness]
        spec["three_way_head_background_margin"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.18, 0.24, 0.30)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.75, 0.95, 1.15)[strictness] if not aggressive else (1.25, 1.55, 1.85)[strictness]
        spec["three_way_head_pseudo_ce_weight"] = (0.10, 0.16, 0.22)[strictness] if not aggressive else (0.28, 0.38, 0.50)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.14, 0.22, 0.30)[strictness] if not aggressive else (0.36, 0.50, 0.66)[strictness]
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else "defer"
        spec["three_way_accept_prob"] = (0.58, 0.62, 0.66)[strictness] if not aggressive else (0.64, 0.68, 0.72)[strictness]
        spec["three_way_reject_prob"] = (0.74, 0.78, 0.82)[strictness] if not aggressive else (0.70, 0.76, 0.82)[strictness]
        spec["three_way_reject_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["three_way_known_background_margin"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.10, 0.16, 0.22)[strictness]

        spec["pseudo_unknown_samples_per_pair"] = 6 if not aggressive else 8
        spec["pseudo_unknown_source_boundary_samples_per_pair"] = 10 if not aggressive else 14
        spec["target_shift_samples"] = 8 if not aggressive else 12
        spec["target_halo_samples"] = 8 if not aggressive else 12
        spec["target_ring_samples"] = 10 if not aggressive else 16
        spec["unknown_moat"] = (0.16, 0.24, 0.32)[strictness] if not aggressive else (0.44, 0.60, 0.76)[strictness]
        spec["unknown_margin"] = (0.50, 0.58, 0.66)[strictness] if not aggressive else (0.66, 0.78, 0.90)[strictness]
        spec["negative_anchor_weight"] = (0.10, 0.16, 0.22)[strictness] if not aggressive else (0.34, 0.48, 0.62)[strictness]
        spec["negative_anchor_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["void_background"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.13, 0.18)[strictness]

        spec["source_looo_unknown_weight"] = (0.22, 0.30, 0.38)[strictness] if not aggressive else (0.42, 0.56, 0.70)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.34, 0.44, 0.54)[strictness]
        spec["source_looo_interclass_margin"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.08, 0.11, 0.14)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer" if not aggressive else "reject"
        spec["source_looo_risk_quantile"] = (0.76, 0.80, 0.84)[strictness] if not aggressive else (0.82, 0.86, 0.90)[strictness]
        spec["source_looo_risk_slack"] = (-0.02, 0.00, 0.02)[strictness] if not aggressive else (0.00, 0.02, 0.04)[strictness]
        spec["source_looo_risk_min_score_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.10, 0.13, 0.16)[strictness]
        spec["source_looo_risk_background_score"] = (0.84, 0.80, 0.76)[strictness] if not aggressive else (0.76, 0.70, 0.64)[strictness]
        spec["source_looo_risk_background_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (0.06, 0.02, -0.02)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 3 if not aggressive else 2

        spec["support_reconstruction_arbitration"] = arm in {"recon_bg_train", "hard_recon_bg_train", "full_bg_train"}
        spec["support_reconstruction_rank"] = 2 if not aggressive else 3
        spec["support_reconstruction_residual_quantile"] = (0.96, 0.95, 0.94)[strictness] if not aggressive else (0.94, 0.92, 0.90)[strictness]
        spec["support_reconstruction_residual_slack"] = (0.05, 0.04, 0.03)[strictness] if not aggressive else (0.03, 0.02, 0.01)[strictness]
        spec["support_reconstruction_background_score"] = (0.84, 0.80, 0.76)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["support_reconstruction_background_margin"] = (0.10, 0.06, 0.02)[strictness] if not aggressive else (0.06, 0.02, -0.02)[strictness]
        spec["support_reconstruction_reject_min_failures"] = 2 if not aggressive else 1

        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_old_min_evidence_delta"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.06, -0.02, 0.02)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.18, 0.12, 0.06)[strictness] if not aggressive else (0.10, 0.04, -0.02)[strictness]
        spec["identity_consensus_reject_min_identity_failures"] = 4 if not aggressive else 3

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain" if not aggressive else "defer"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = True
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["pre_reject_support_retention_max_background_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.14, 0.08, 0.02)[strictness] if not aggressive else (0.06, 0.00, -0.06)[strictness]
        spec["pre_reject_reject_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.18, 0.12, 0.06)[strictness] if not aggressive else (0.10, 0.04, -0.02)[strictness]

        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.70, 0.64, 0.58)[strictness]
        spec["two_branch_bg_min_margin"] = (0.06, 0.02, -0.02)[strictness] if not aggressive else (0.02, -0.04, -0.10)[strictness]
        spec["siamese_unknown_veto"] = aggressive or strictness >= 1
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["siamese_threshold"] = (0.54, 0.60, 0.66)[strictness] if not aggressive else (0.64, 0.72, 0.80)[strictness]
        spec["min_veto_failures"] = 4 if not aggressive else 3
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 5 if not aggressive else 4
        spec["guard_min_margin"] = (0.16, 0.22, 0.28)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]

        spec["retention_rescue_gate"] = True
        spec["retention_rescue_max_background_score"] = (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.76, 0.70, 0.64)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.16, 0.10, 0.04)[strictness] if not aggressive else (0.06, 0.00, -0.06)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.82
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 1970
    return specs


def _oa_mse_h06_retold48_stage_specs() -> list[dict]:
    """48-row H06 old-retention-first calibrated rejector after next48fq.

    next48fq proved the repaired old/background head can reduce unknown FAR,
    but it over-rejected target-old query samples. This route keeps Stage2-B
    old/unknown-only and makes old retention the first decision stage:
    accept/defer old evidence before any pseudo-background hard reject, while
    using background probability only as a joint risk signal with support and
    drift diagnostics.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_bgtrain48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        slot = str(spec.get("slot", ""))
        arm = (
            ("old_floor_joint_risk", "identity_cap_joint_risk", "softmix_joint_risk")
            if not aggressive
            else ("balanced_joint_risk", "defer_recon_joint_risk", "hard_joint_risk")
        )[strictness]
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_unknown_old_retention_first_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "next48fp_old_mean_0p502_unknown_far_mean_0p848;"
            "next48fq_unknown_far_mean_0p424_old_mean_0p275;"
            "next48fq_old_bg_prob_0p923_unknown_bg_prob_0p955;"
            "support_reconstruction_and_conformal_hurt_old_more_than_unknown;"
            "new_class_enrollment_deferred_by_user_direction"
        )
        spec["description"] = (
            "H06-RETOLD conservative: keep H06 latest old/unknown-only Stage2-B, restore target-old retention "
            "with class-first known floor and identity-consensus background caps, then reject only when "
            "three-way background risk and old-evidence failure agree."
            if not aggressive
            else "H06-RETOLD aggressive-balanced: keep stronger pseudo-background pressure, but downgrade "
            "support reconstruction/conformal to defer diagnostics and require old-retention failure before "
            "hard unknown rejection."
        )

        # Keep target-new out of training and evaluation claims in this route.
        spec["source_proto_per_tx"] = 96 if not aggressive else 112
        spec["source_query_per_tx"] = 48 if not aggressive else 56
        spec["sfe_max_samples_per_tx"] = 480 if not aggressive else 560
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "constrained_retention_risk"
        spec["adapter_kind"] = "low_rank" if (not aggressive or slot in {"A", "B", "C"}) else "residual_mlp"
        spec["steps"] = min(116 if not aggressive else 132, max(88 if not aggressive else 100, int(spec.get("steps", 88)) + 4))

        # Old retention first: increase old/source/support constraints and
        # reduce the support-only overfit driver relative to BGTRAIN.
        spec["known_coverage_weight"] = (5.20, 5.60, 6.00)[strictness] if not aggressive else (4.20, 4.70, 5.20)[strictness]
        spec["known_coverage_margin"] = (0.16, 0.18, 0.20)[strictness] if not aggressive else (0.18, 0.21, 0.24)[strictness]
        spec["known_coverage_min_affinity"] = (0.42, 0.45, 0.48)[strictness] if not aggressive else (0.44, 0.48, 0.52)[strictness]
        spec["source_ce"] = (1.35, 1.45, 1.55)[strictness] if not aggressive else (1.12, 1.24, 1.36)[strictness]
        spec["old_bridge"] = (1.22, 1.34, 1.46)[strictness] if not aggressive else (1.02, 1.14, 1.26)[strictness]
        spec["old_neighborhood"] = (1.15, 1.25, 1.35)[strictness] if not aggressive else (0.96, 1.06, 1.16)[strictness]
        spec["old_surrogate_margin_weight"] = (0.12, 0.16, 0.20)[strictness] if not aggressive else (0.16, 0.22, 0.28)[strictness]
        spec["support_center_ce"] = (0.55, 0.65, 0.75)[strictness] if not aggressive else (0.70, 0.85, 1.00)[strictness]
        spec["support_center_margin"] = (0.10, 0.12, 0.14)[strictness] if not aggressive else (0.14, 0.18, 0.22)[strictness]
        spec["support_contrast"] = (0.38, 0.46, 0.54)[strictness] if not aggressive else (0.50, 0.62, 0.74)[strictness]
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["support_retention_guard_slack"] = (0.14, 0.16, 0.18)[strictness] if not aggressive else (0.10, 0.12, 0.14)[strictness]
        spec["old_retention_quantile"] = (0.99, 0.98, 0.97)[strictness] if not aggressive else (0.96, 0.94, 0.92)[strictness]

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = (5, 5, 6)[strictness] if not aggressive else (5, 6, 6)[strictness]
        spec["multiproto_temperature"] = (0.075, 0.060, 0.050)[strictness] if not aggressive else (0.060, 0.050, 0.042)[strictness]
        spec["multiproto_score_weight"] = (1.35, 1.55, 1.75)[strictness] if not aggressive else (1.55, 1.85, 2.15)[strictness]
        spec["soft_proto"] = (1.25, 1.45, 1.65)[strictness] if not aggressive else (1.45, 1.75, 2.05)[strictness]
        spec["soft_proto_boundary"] = (0.28, 0.34, 0.40)[strictness] if not aggressive else (0.40, 0.52, 0.64)[strictness]
        spec["soft_proto_boundary_margin"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.20, 0.26, 0.32)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = (0.22, 0.28, 0.34)[strictness] if not aggressive else (0.30, 0.38, 0.46)[strictness]
        spec["mixture_consistency_max_residual"] = (1.20, 1.00, 0.85)[strictness] if not aggressive else (0.95, 0.78, 0.62)[strictness]
        spec["mixture_consistency_min_margin"] = (-0.10, -0.05, 0.00)[strictness] if not aggressive else (-0.02, 0.04, 0.10)[strictness]
        spec["mixture_consistency_action"] = "uncertain"

        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_residual_quantile"] = (0.99, 0.98, 0.97)[strictness] if not aggressive else (0.97, 0.95, 0.93)[strictness]
        spec["class_envelope_score_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_margin_quantile"] = (0.01, 0.03, 0.05)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_evidence_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_residual_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_score_slack"] = (0.10, 0.07, 0.04)[strictness] if not aggressive else (0.06, 0.03, 0.00)[strictness]
        spec["class_envelope_margin_slack"] = (0.08, 0.06, 0.04)[strictness] if not aggressive else (0.04, 0.02, 0.00)[strictness]
        spec["class_envelope_min_failures"] = 2 if not aggressive else 1
        spec["class_envelope_gate_action"] = "uncertain" if not aggressive else "reject"

        spec["old_primary_gate"] = True
        spec["old_primary_require_soft_mixture"] = True
        spec["old_primary_require_support_knn"] = True
        spec["old_primary_require_support_knn_label_match"] = True
        spec["old_primary_require_class_envelope"] = True
        spec["old_primary_min_old_support_evidence_delta"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.03, 0.06, 0.09)[strictness]
        spec["old_primary_min_old_support_anchor_delta"] = (-0.04, -0.02, 0.00)[strictness] if not aggressive else (-0.01, 0.02, 0.05)[strictness]
        spec["old_primary_min_old_support_anchor_margin"] = (-0.02, 0.00, 0.02)[strictness] if not aggressive else (0.02, 0.05, 0.08)[strictness]
        spec["old_primary_min_score_margin"] = (-0.06, -0.02, 0.02)[strictness] if not aggressive else (0.00, 0.05, 0.10)[strictness]
        spec["old_primary_min_soft_mixture_margin"] = (-0.10, -0.05, 0.00)[strictness] if not aggressive else (-0.02, 0.04, 0.10)[strictness]
        spec["old_primary_min_soft_mixture_cos"] = (0.22, 0.28, 0.34)[strictness] if not aggressive else (0.30, 0.38, 0.46)[strictness]
        spec["old_primary_max_soft_mixture_residual"] = (1.20, 1.00, 0.85)[strictness] if not aggressive else (0.95, 0.78, 0.62)[strictness]
        spec["old_primary_min_support_knn_margin"] = (-0.04, 0.00, 0.04)[strictness] if not aggressive else (0.02, 0.06, 0.10)[strictness]
        spec["old_primary_max_support_knn_seen_new_minus_old"] = 0.0
        spec["old_primary_min_old_drift_cos"] = (0.35, 0.45, 0.55)[strictness] if not aggressive else (0.45, 0.55, 0.65)[strictness]
        spec["old_primary_max_old_drift_dist"] = (0.65, 0.55, 0.45)[strictness] if not aggressive else (0.55, 0.45, 0.35)[strictness]
        spec["old_primary_unknown_veto_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["old_primary_unknown_veto_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["old_primary_unknown_veto_min_sources"] = 1
        spec["old_primary_fail_action"] = "defer"
        spec["old_primary_unknown_veto_action"] = "defer" if not aggressive else "reject"

        # Background is a secondary joint-risk veto. We keep it learnable but
        # make direct hard reject require stronger probability/margin evidence.
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not aggressive else ("accept" if slot in {"A", "B", "C"} else "defer")
        spec["three_way_head_weight"] = (0.12, 0.18, 0.24)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["three_way_head_temperature"] = (0.32, 0.28, 0.24)[strictness] if not aggressive else (0.28, 0.23, 0.18)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.60 if not aggressive else 1.35
        spec["three_way_head_pseudo_ce_weight"] = (0.04, 0.07, 0.10)[strictness] if not aggressive else (0.10, 0.16, 0.22)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.35, 0.45, 0.55)[strictness] if not aggressive else (0.55, 0.70, 0.85)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.04, 0.08, 0.12)[strictness] if not aggressive else (0.14, 0.22, 0.30)[strictness]
        spec["three_way_accept_prob"] = (0.66, 0.70, 0.74)[strictness] if not aggressive else (0.62, 0.66, 0.70)[strictness]
        spec["three_way_reject_prob"] = (0.96, 0.94, 0.92)[strictness] if not aggressive else (0.90, 0.86, 0.82)[strictness]
        spec["three_way_defer_prob"] = (0.80, 0.76, 0.72)[strictness] if not aggressive else (0.70, 0.66, 0.62)[strictness]
        spec["three_way_reject_margin"] = (0.44, 0.38, 0.32)[strictness] if not aggressive else (0.32, 0.24, 0.16)[strictness]
        spec["three_way_known_background_margin"] = (-0.30, -0.24, -0.18)[strictness] if not aggressive else (-0.22, -0.16, -0.10)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.9995, 0.9990, 0.9980)[strictness] if not aggressive else (0.9980, 0.9960, 0.9940)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.80, 1.45, 1.10)[strictness] if not aggressive else (1.15, 0.85, 0.60)[strictness]
        spec["three_way_known_floor_old_min_evidence_delta"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (0.00, -0.03, -0.06)[strictness] if not aggressive else (-0.04, -0.08, -0.12)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (0.02, 0.00, -0.02)[strictness] if not aggressive else (0.00, -0.04, -0.08)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (-0.02, -0.06, -0.10)[strictness] if not aggressive else (-0.08, -0.14, -0.20)[strictness]

        # Unknown pressure is kept, but no longer allowed to dominate the head.
        spec["pseudo_unknown_samples_per_pair"] = 4 if not aggressive else 6
        spec["pseudo_unknown_source_boundary_samples_per_pair"] = 8 if not aggressive else 10
        spec["target_shift_samples"] = 4 if not aggressive else 7
        spec["target_halo_samples"] = 4 if not aggressive else 7
        spec["target_ring_samples"] = 6 if not aggressive else 10
        spec["unknown_moat"] = (0.06, 0.10, 0.14)[strictness] if not aggressive else (0.18, 0.28, 0.38)[strictness]
        spec["unknown_margin"] = (0.34, 0.42, 0.50)[strictness] if not aggressive else (0.46, 0.58, 0.70)[strictness]
        spec["negative_anchor_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.14, 0.20)[strictness]
        spec["negative_anchor_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.02, 0.04, 0.06)[strictness]
        spec["source_looo_unknown_weight"] = (0.00, 0.03, 0.06)[strictness] if not aggressive else (0.08, 0.14, 0.20)[strictness]
        spec["source_looo_unknown_margin"] = (0.22, 0.28, 0.34)[strictness] if not aggressive else (0.32, 0.42, 0.52)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_risk_quantile"] = (0.82, 0.86, 0.90)[strictness] if not aggressive else (0.86, 0.90, 0.94)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 4 if not aggressive else 3

        # Prior evidence showed these two modules misfire on old. Keep their
        # diagnostics but do not let them hard-reject old in this matrix.
        spec["support_reconstruction_arbitration"] = bool(aggressive and slot in {"D", "E", "F"})
        spec["support_reconstruction_reject_action"] = "defer"
        spec["support_reconstruction_reject_min_failures"] = 3
        spec["support_reconstruction_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["support_reconstruction_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["support_conformal_arbitration"] = bool(slot in {"B", "C", "E", "F"})
        spec["support_conformal_reject_action"] = "defer"
        spec["support_conformal_reject_min_failures"] = 3
        spec["support_conformal_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["support_conformal_background_margin"] = (0.20, 0.14, 0.08)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]

        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_support_background_cap_quantile"] = (0.98, 0.96, 0.94)[strictness] if not aggressive else (0.94, 0.91, 0.88)[strictness]
        spec["identity_consensus_support_background_cap_slack"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["identity_consensus_old_min_evidence_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.16, -0.12, -0.08)[strictness] if not aggressive else (-0.10, -0.06, -0.02)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.90, 0.84, 0.78)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.28, 0.22, 0.16)[strictness] if not aggressive else (0.18, 0.12, 0.06)[strictness]

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = False
        spec["pre_reject_support_retention_max_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.28, 0.22, 0.16)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]
        spec["pre_reject_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.90, 0.84, 0.78)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.30, 0.24, 0.18)[strictness] if not aggressive else (0.20, 0.14, 0.08)[strictness]

        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["two_branch_bg_min_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.20, -0.16, -0.12)[strictness] if not aggressive else (-0.14, -0.10, -0.06)[strictness]
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_max_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["siamese_unknown_veto"] = bool(aggressive and strictness >= 1)
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["min_veto_failures"] = 5 if not aggressive else 4
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 6 if not aggressive else 5
        spec["guard_min_margin"] = (0.22, 0.28, 0.34)[strictness] if not aggressive else (0.26, 0.34, 0.42)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2110
    return specs


def _oa_mse_h06_oldrelax48_stage_specs() -> list[dict]:
    """48-row H06 old-primary relaxed ablation after next48fs.

    next48fs showed useful old-primary structure in the A/B/C region, but its
    hard class-envelope and support-KNN/drift thresholds over-rejected target-old
    samples. This plan keeps the same old/unknown-only Stage2-B boundary and
    tests relaxed old acceptance: multi/soft prototype, support-KNN, drift,
    margin, and class-envelope evidence must agree for old accept; unknown-risk
    veto remains stronger than retention rescue.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_retold48_stage_specs()]
    arms = (
        "abc_soft_envelope_k5",
        "abc_support_knn_relaxed_k5",
        "abc_k10_soft_envelope_control",
        "abc_relaxed_aggressive_k5",
        "abc_margin_aggressive_k5",
        "abc_veto_aggressive_k5",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]

        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = 10 if arm.endswith("k10_soft_envelope_control") else 5
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_primary_consensus_gate_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "next48fs_old_primary_too_strict_old_mean_0p122_coverage_mean_0p184;"
            "conservative_old_0p241_far_0p301_aggressive_near_zero_coverage;"
            "keep_next48fs_A_B_C_region_relax_D_F_zero_coverage_failure;"
            "class_envelope_required_for_terminal_old_primary_consensus;"
            "retention_rescue_candidate_only_old_primary_final_accept;"
            "unknown_risk_veto_preempts_rescue_no_unknown_threshold_fit"
        )
        spec["description"] = (
            "H06-OLDRELAX conservative ablation: keep H06 latest old/unknown-only Stage2-B on "
            "target receiver 20-1, relax support-KNN and drift while requiring multi/soft prototype, "
            "margin, support-KNN, drift and soft class-envelope evidence to agree before accepting old; "
            "unknown-risk veto uses two evidence sources and overrides retention rescue."
            if not aggressive
            else "H06-OLDRELAX relatively aggressive ablation: stay near the useful next48fs A/B/C "
            "region with relaxed old-primary thresholds, but keep unknown-risk veto ahead of rescue "
            "and keep class envelope as uncertainty evidence rather than a hard reject."
        )

        spec["source_proto_per_tx"] = 104 if not aggressive else 112
        spec["source_query_per_tx"] = 52 if not aggressive else 56
        spec["sfe_max_samples_per_tx"] = 520 if not aggressive else 560
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "constrained_retention_risk"
        spec["adapter_kind"] = "low_rank"
        spec["steps"] = (104, 112, 120)[strictness] if not aggressive else (112, 124, 132)[strictness]

        spec["known_coverage_weight"] = (5.40, 5.80, 6.20)[strictness] if not aggressive else (4.70, 5.20, 5.70)[strictness]
        spec["known_coverage_margin"] = (0.14, 0.16, 0.18)[strictness] if not aggressive else (0.16, 0.19, 0.22)[strictness]
        spec["known_coverage_min_affinity"] = (0.40, 0.43, 0.46)[strictness] if not aggressive else (0.42, 0.46, 0.50)[strictness]
        spec["source_ce"] = (1.25, 1.35, 1.45)[strictness] if not aggressive else (1.05, 1.16, 1.28)[strictness]
        spec["old_bridge"] = (1.12, 1.22, 1.32)[strictness] if not aggressive else (0.96, 1.06, 1.16)[strictness]
        spec["old_neighborhood"] = (1.05, 1.15, 1.25)[strictness] if not aggressive else (0.90, 1.00, 1.10)[strictness]
        spec["old_surrogate_margin_weight"] = (0.10, 0.13, 0.16)[strictness] if not aggressive else (0.12, 0.17, 0.22)[strictness]
        spec["support_center_ce"] = (0.48, 0.58, 0.68)[strictness] if not aggressive else (0.58, 0.72, 0.86)[strictness]
        spec["support_center_margin"] = (0.08, 0.10, 0.12)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["support_contrast"] = (0.30, 0.38, 0.46)[strictness] if not aggressive else (0.40, 0.52, 0.64)[strictness]
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.02, 0.03, 0.04)[strictness] if not aggressive else (0.04, 0.06, 0.08)[strictness]
        spec["support_retention_guard_slack"] = (0.18, 0.20, 0.22)[strictness] if not aggressive else (0.14, 0.16, 0.18)[strictness]
        spec["old_retention_quantile"] = (0.97, 0.96, 0.95)[strictness] if not aggressive else (0.94, 0.92, 0.90)[strictness]

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if spec["k_old"] == 5 else 6
        spec["multiproto_temperature"] = (0.080, 0.070, 0.060)[strictness] if not aggressive else (0.070, 0.060, 0.052)[strictness]
        spec["multiproto_score_weight"] = (1.20, 1.40, 1.60)[strictness] if not aggressive else (1.40, 1.65, 1.90)[strictness]
        spec["soft_proto"] = (1.10, 1.30, 1.50)[strictness] if not aggressive else (1.30, 1.55, 1.80)[strictness]
        spec["soft_proto_boundary"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.32, 0.42, 0.52)[strictness]
        spec["soft_proto_boundary_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.16, 0.22, 0.28)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = (0.16, 0.22, 0.28)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["mixture_consistency_max_residual"] = (1.45, 1.25, 1.05)[strictness] if not aggressive else (1.20, 1.00, 0.82)[strictness]
        spec["mixture_consistency_min_margin"] = (-0.18, -0.12, -0.06)[strictness] if not aggressive else (-0.12, -0.04, 0.04)[strictness]
        spec["mixture_consistency_action"] = "uncertain"

        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.03, 0.05, 0.08)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_residual_quantile"] = (0.99, 0.98, 0.97)[strictness] if not aggressive else (0.98, 0.96, 0.94)[strictness]
        spec["class_envelope_score_quantile"] = (0.03, 0.05, 0.08)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_margin_quantile"] = (0.03, 0.05, 0.08)[strictness] if not aggressive else (0.05, 0.08, 0.12)[strictness]
        spec["class_envelope_evidence_slack"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["class_envelope_residual_slack"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["class_envelope_score_slack"] = (0.16, 0.12, 0.08)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["class_envelope_margin_slack"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["class_envelope_min_failures"] = 3 if not aggressive else 2
        spec["class_envelope_gate_action"] = "uncertain"

        spec["old_primary_gate"] = True
        spec["old_primary_require_soft_mixture"] = True
        spec["old_primary_require_support_knn"] = True
        spec["old_primary_require_support_knn_label_match"] = True
        spec["old_primary_require_class_envelope"] = True
        spec["old_primary_min_old_support_evidence_delta"] = (-0.06, -0.03, 0.00)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["old_primary_min_old_support_anchor_delta"] = (-0.10, -0.07, -0.04)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["old_primary_min_old_support_anchor_margin"] = (-0.06, -0.03, 0.00)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["old_primary_min_score_margin"] = (-0.14, -0.09, -0.04)[strictness] if not aggressive else (-0.10, -0.04, 0.02)[strictness]
        spec["old_primary_min_soft_mixture_margin"] = (-0.18, -0.12, -0.06)[strictness] if not aggressive else (-0.12, -0.04, 0.04)[strictness]
        spec["old_primary_min_soft_mixture_cos"] = (0.16, 0.22, 0.28)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["old_primary_max_soft_mixture_residual"] = (1.45, 1.25, 1.05)[strictness] if not aggressive else (1.20, 1.00, 0.82)[strictness]
        spec["old_primary_min_support_knn_margin"] = (-0.12, -0.08, -0.04)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["old_primary_max_support_knn_seen_new_minus_old"] = None
        spec["old_primary_min_old_drift_cos"] = (0.25, 0.32, 0.39)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["old_primary_max_old_drift_dist"] = (0.85, 0.78, 0.70)[strictness] if not aggressive else (0.88, 0.80, 0.72)[strictness]
        spec["old_primary_unknown_veto_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.82, 0.76, 0.70)[strictness]
        spec["old_primary_unknown_veto_background_margin"] = (0.18, 0.13, 0.08)[strictness] if not aggressive else (0.10, 0.04, -0.02)[strictness]
        spec["old_primary_unknown_veto_min_sources"] = 2 if not aggressive else 1
        spec["old_primary_fail_action"] = "defer"
        spec["old_primary_unknown_veto_action"] = "reject"

        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "defer"
        spec["three_way_head_weight"] = (0.10, 0.15, 0.20)[strictness] if not aggressive else (0.20, 0.30, 0.40)[strictness]
        spec["three_way_head_temperature"] = (0.34, 0.30, 0.26)[strictness] if not aggressive else (0.30, 0.25, 0.20)[strictness]
        spec["three_way_head_support_ce_weight"] = 1.50 if not aggressive else 1.30
        spec["three_way_head_pseudo_ce_weight"] = (0.03, 0.05, 0.08)[strictness] if not aggressive else (0.08, 0.13, 0.18)[strictness]
        spec["three_way_head_support_background_margin_weight"] = (0.28, 0.36, 0.44)[strictness] if not aggressive else (0.42, 0.56, 0.70)[strictness]
        spec["three_way_head_pseudo_margin_weight"] = (0.03, 0.06, 0.09)[strictness] if not aggressive else (0.10, 0.16, 0.22)[strictness]
        spec["three_way_accept_prob"] = (0.62, 0.66, 0.70)[strictness] if not aggressive else (0.60, 0.64, 0.68)[strictness]
        spec["three_way_reject_prob"] = (0.97, 0.95, 0.93)[strictness] if not aggressive else (0.91, 0.87, 0.83)[strictness]
        spec["three_way_defer_prob"] = (0.82, 0.78, 0.74)[strictness] if not aggressive else (0.72, 0.68, 0.64)[strictness]
        spec["three_way_reject_margin"] = (0.42, 0.36, 0.30)[strictness] if not aggressive else (0.30, 0.22, 0.14)[strictness]
        spec["three_way_known_background_margin"] = (-0.32, -0.26, -0.20)[strictness] if not aggressive else (-0.24, -0.18, -0.12)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.9995, 0.9990, 0.9980)[strictness] if not aggressive else (0.9980, 0.9960, 0.9940)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.70, 1.35, 1.00)[strictness] if not aggressive else (1.10, 0.80, 0.55)[strictness]

        spec["pseudo_unknown_samples_per_pair"] = 4 if not aggressive else 6
        spec["pseudo_unknown_source_boundary_samples_per_pair"] = 8 if not aggressive else 10
        spec["target_shift_samples"] = 4 if not aggressive else 6
        spec["target_halo_samples"] = 4 if not aggressive else 6
        spec["target_ring_samples"] = 6 if not aggressive else 9
        spec["unknown_moat"] = (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.18, 0.26, 0.34)[strictness]
        spec["unknown_margin"] = (0.32, 0.40, 0.48)[strictness] if not aggressive else (0.44, 0.56, 0.68)[strictness]
        spec["negative_anchor_weight"] = (0.02, 0.03, 0.04)[strictness] if not aggressive else (0.06, 0.10, 0.14)[strictness]
        spec["negative_anchor_margin"] = (0.05, 0.07, 0.09)[strictness] if not aggressive else (0.09, 0.12, 0.15)[strictness]
        spec["void_background"] = 0.0 if not aggressive else (0.01, 0.02, 0.03)[strictness]
        spec["source_looo_unknown_weight"] = (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.06, 0.10, 0.14)[strictness]
        spec["source_looo_unknown_margin"] = (0.20, 0.26, 0.32)[strictness] if not aggressive else (0.30, 0.38, 0.46)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_risk_quantile"] = (0.80, 0.84, 0.88)[strictness] if not aggressive else (0.84, 0.88, 0.92)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 4 if not aggressive else 3

        spec["support_reconstruction_arbitration"] = False
        spec["support_reconstruction_reject_action"] = "defer"
        spec["support_conformal_arbitration"] = False
        spec["support_conformal_reject_action"] = "defer"
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_support_background_cap_quantile"] = (0.98, 0.96, 0.94)[strictness] if not aggressive else (0.94, 0.91, 0.88)[strictness]
        spec["identity_consensus_support_background_cap_slack"] = (0.14, 0.12, 0.10)[strictness] if not aggressive else (0.10, 0.07, 0.04)[strictness]
        spec["identity_consensus_old_min_evidence_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.90, 0.84, 0.78)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.26, 0.20, 0.14)[strictness] if not aggressive else (0.16, 0.10, 0.04)[strictness]

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = False
        spec["pre_reject_support_retention_max_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]
        spec["pre_reject_reject_background_score"] = (0.98, 0.94, 0.90)[strictness] if not aggressive else (0.90, 0.84, 0.78)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.28, 0.22, 0.16)[strictness] if not aggressive else (0.18, 0.12, 0.06)[strictness]

        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.82, 0.76, 0.70)[strictness]
        spec["two_branch_bg_min_margin"] = (0.18, 0.13, 0.08)[strictness] if not aggressive else (0.10, 0.04, -0.02)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.22, -0.18, -0.14)[strictness] if not aggressive else (-0.16, -0.12, -0.08)[strictness]
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["old_primary_promote_rescue_candidates"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (-0.02, 0.00, 0.02)[strictness] if not aggressive else (0.00, 0.03, 0.06)[strictness]
        spec["retention_rescue_old_min_anchor_delta"] = (-0.06, -0.04, -0.02)[strictness] if not aggressive else (-0.03, 0.00, 0.03)[strictness]
        spec["retention_rescue_old_min_anchor_margin"] = (-0.04, -0.02, 0.00)[strictness] if not aggressive else (-0.01, 0.02, 0.05)[strictness]
        spec["retention_rescue_old_min_score_margin"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["retention_rescue_max_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.16, 0.11, 0.06)[strictness] if not aggressive else (0.06, 0.00, -0.06)[strictness]
        spec["siamese_unknown_veto"] = bool(aggressive and strictness >= 1)
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["min_veto_failures"] = 5 if not aggressive else 4
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 6 if not aggressive else 5
        spec["guard_min_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2290
    return specs


def _oa_mse_h06_oldgeom48_stage_specs() -> list[dict]:
    """48-row H06 old/unknown-only support-geometry repair.

    The preceding H06 separability diagnostic showed that prototype scores and
    support/source thresholds were not geometrically separable enough to promote
    more terminal gates. This plan keeps Stage2-B old/unknown boundaries, excludes
    target-new support/query enrollment, and first repairs support-center and
    target-shift geometry before applying old-primary arbitration.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_oldrelax48_stage_specs()]
    arms = (
        "support_center_shift_k5",
        "support_center_halo_k5",
        "soft_mixture_ring_k5",
        "support_manifold_k10",
        "background_ring_k5",
        "balanced_geometry_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]

        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = 10 if arm.endswith("k10") else 5
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_unknown_support_geometry_repair_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "h06_feature_separability_20260626_negative_geometry_not_separable;"
            "rho0p5_oracle_far005_old_mean_0p167_max_0p311;"
            "source_thresholds_high_unknown_far_and_support_thresholds_low_old_accept;"
            "repair_support_center_target_shift_halo_ring_before_more_terminal_gates;"
            "stage2b_old_unknown_only_target_new_excluded"
        )
        spec["description"] = (
            "H06-OLDGEOM conservative repair: keep target receiver 20-1 old/unknown-only Stage2-B, "
            "train support-center and soft prototype geometry with query-free target shift/halo/ring pseudo-unknowns, "
            "then use old-primary arbitration as a measurement gate rather than adding new target-new enrollment."
            if not aggressive
            else "H06-OLDGEOM aggressive repair: strengthen support-center geometry, target-ring pseudo-unknown pressure "
            "and soft-mixture consistency to test whether old/unknown separation improves before any Stage2-C path."
        )

        spec["source_proto_per_tx"] = 108 if not aggressive else 116
        spec["source_query_per_tx"] = 54 if not aggressive else 58
        spec["sfe_max_samples_per_tx"] = 540 if not aggressive else 580
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not aggressive else "residual_mlp"
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "constrained_retention_risk"
        spec["steps"] = (112, 124, 136)[strictness] if not aggressive else (124, 140, 156)[strictness]

        spec["source_ce"] = (1.18, 1.30, 1.42)[strictness] if not aggressive else (1.00, 1.14, 1.28)[strictness]
        spec["old_bridge"] = (1.18, 1.30, 1.42)[strictness] if not aggressive else (1.02, 1.16, 1.30)[strictness]
        spec["old_neighborhood"] = (1.10, 1.24, 1.38)[strictness] if not aggressive else (0.98, 1.12, 1.26)[strictness]
        spec["old_surrogate_margin_weight"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.18, 0.24, 0.30)[strictness]
        spec["support_center_ce"] = (0.74, 0.86, 0.98)[strictness] if not aggressive else (0.88, 1.06, 1.24)[strictness]
        spec["support_center_temperature"] = (0.070, 0.060, 0.052)[strictness] if not aggressive else (0.060, 0.052, 0.046)[strictness]
        spec["support_center_margin"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["support_contrast"] = (0.46, 0.58, 0.70)[strictness] if not aggressive else (0.60, 0.76, 0.92)[strictness]
        spec["known_coverage_weight"] = (4.20, 4.70, 5.20)[strictness] if not aggressive else (3.80, 4.40, 5.00)[strictness]
        spec["known_coverage_margin"] = (0.14, 0.17, 0.20)[strictness] if not aggressive else (0.18, 0.23, 0.28)[strictness]
        spec["known_coverage_min_affinity"] = (0.36, 0.40, 0.44)[strictness] if not aggressive else (0.38, 0.44, 0.50)[strictness]

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 4 if spec["k_old"] == 5 else 6
        spec["multiproto_temperature"] = (0.075, 0.065, 0.055)[strictness] if not aggressive else (0.065, 0.055, 0.048)[strictness]
        spec["multiproto_score_weight"] = (1.45, 1.70, 1.95)[strictness] if not aggressive else (1.70, 2.00, 2.30)[strictness]
        spec["soft_proto"] = (1.45, 1.70, 1.95)[strictness] if not aggressive else (1.70, 2.05, 2.40)[strictness]
        spec["soft_proto_topk"] = 3 if spec["k_old"] == 5 else 4
        spec["soft_proto_temperature"] = (0.080, 0.070, 0.060)[strictness] if not aggressive else (0.070, 0.060, 0.052)[strictness]
        spec["soft_proto_boundary"] = (0.30, 0.38, 0.46)[strictness] if not aggressive else (0.42, 0.54, 0.66)[strictness]
        spec["soft_proto_boundary_margin"] = (0.14, 0.18, 0.22)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = (0.22, 0.30, 0.38)[strictness] if not aggressive else (0.30, 0.40, 0.50)[strictness]
        spec["mixture_consistency_max_residual"] = (1.20, 1.05, 0.90)[strictness] if not aggressive else (1.05, 0.86, 0.70)[strictness]
        spec["mixture_consistency_min_margin"] = (-0.10, -0.04, 0.02)[strictness] if not aggressive else (-0.04, 0.04, 0.12)[strictness]
        spec["mixture_consistency_action"] = "uncertain"

        spec["pseudo_unknown_samples_per_pair"] = 4 if not aggressive else 6
        spec["pseudo_unknown_source_boundary_samples_per_pair"] = 10 if not aggressive else 14
        spec["pseudo_halo_override"] = 8 if not aggressive else 12
        spec["pseudo_ring_override"] = 12 if not aggressive else 18
        spec["unknown_moat"] = (0.14, 0.20, 0.26)[strictness] if not aggressive else (0.24, 0.34, 0.44)[strictness]
        spec["unknown_margin"] = (0.38, 0.48, 0.58)[strictness] if not aggressive else (0.54, 0.68, 0.82)[strictness]
        spec["negative_anchor_weight"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.10, 0.15, 0.20)[strictness]
        spec["negative_anchor_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.10, 0.14, 0.18)[strictness]
        spec["source_looo_unknown_weight"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["source_looo_unknown_margin"] = (0.24, 0.30, 0.36)[strictness] if not aggressive else (0.34, 0.44, 0.54)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_risk_quantile"] = (0.82, 0.86, 0.90)[strictness] if not aggressive else (0.86, 0.90, 0.94)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 4 if not aggressive else 3

        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.05, 0.08, 0.11)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["class_envelope_residual_quantile"] = (0.98, 0.97, 0.96)[strictness] if not aggressive else (0.97, 0.95, 0.93)[strictness]
        spec["class_envelope_score_quantile"] = (0.05, 0.08, 0.11)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["class_envelope_margin_quantile"] = (0.05, 0.08, 0.11)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["class_envelope_evidence_slack"] = (0.10, 0.08, 0.06)[strictness] if not aggressive else (0.06, 0.03, 0.00)[strictness]
        spec["class_envelope_residual_slack"] = (0.10, 0.08, 0.06)[strictness] if not aggressive else (0.06, 0.03, 0.00)[strictness]
        spec["class_envelope_score_slack"] = (0.12, 0.09, 0.06)[strictness] if not aggressive else (0.08, 0.04, 0.00)[strictness]
        spec["class_envelope_margin_slack"] = (0.10, 0.08, 0.06)[strictness] if not aggressive else (0.06, 0.03, 0.00)[strictness]
        spec["class_envelope_min_failures"] = 3 if not aggressive else 2
        spec["class_envelope_gate_action"] = "uncertain"

        spec["old_primary_gate"] = True
        spec["old_primary_require_soft_mixture"] = True
        spec["old_primary_require_support_knn"] = True
        spec["old_primary_require_support_knn_label_match"] = True
        spec["old_primary_require_class_envelope"] = True
        spec["old_primary_min_old_support_evidence_delta"] = (-0.06, -0.02, 0.02)[strictness] if not aggressive else (-0.02, 0.04, 0.10)[strictness]
        spec["old_primary_min_old_support_anchor_delta"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.06, 0.00, 0.06)[strictness]
        spec["old_primary_min_old_support_anchor_margin"] = (-0.06, -0.02, 0.02)[strictness] if not aggressive else (-0.02, 0.04, 0.10)[strictness]
        spec["old_primary_min_score_margin"] = (-0.12, -0.06, 0.00)[strictness] if not aggressive else (-0.06, 0.02, 0.10)[strictness]
        spec["old_primary_min_soft_mixture_margin"] = (-0.10, -0.04, 0.02)[strictness] if not aggressive else (-0.04, 0.04, 0.12)[strictness]
        spec["old_primary_min_soft_mixture_cos"] = spec["mixture_consistency_min_cos"]
        spec["old_primary_max_soft_mixture_residual"] = spec["mixture_consistency_max_residual"]
        spec["old_primary_min_support_knn_margin"] = (-0.10, -0.05, 0.00)[strictness] if not aggressive else (-0.05, 0.02, 0.09)[strictness]
        spec["old_primary_max_support_knn_seen_new_minus_old"] = None
        spec["old_primary_min_old_drift_cos"] = (0.30, 0.38, 0.46)[strictness] if not aggressive else (0.34, 0.44, 0.54)[strictness]
        spec["old_primary_max_old_drift_dist"] = (0.82, 0.74, 0.66)[strictness] if not aggressive else (0.76, 0.66, 0.56)[strictness]
        spec["old_primary_unknown_veto_background_score"] = (0.90, 0.86, 0.82)[strictness] if not aggressive else (0.80, 0.74, 0.68)[strictness]
        spec["old_primary_unknown_veto_background_margin"] = (0.16, 0.11, 0.06)[strictness] if not aggressive else (0.08, 0.02, -0.04)[strictness]
        spec["old_primary_unknown_veto_min_sources"] = 2 if not aggressive else 1
        spec["old_primary_fail_action"] = "defer"
        spec["old_primary_unknown_veto_action"] = "reject"
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = False

        spec["support_reconstruction_arbitration"] = False
        spec["support_conformal_arbitration"] = False
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_support_background_cap_quantile"] = (0.97, 0.95, 0.93)[strictness] if not aggressive else (0.93, 0.90, 0.87)[strictness]
        spec["identity_consensus_support_background_cap_slack"] = (0.12, 0.10, 0.08)[strictness] if not aggressive else (0.08, 0.05, 0.02)[strictness]
        spec["identity_consensus_old_min_evidence_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["identity_consensus_old_min_anchor_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]

        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if not aggressive else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "defer"
        spec["three_way_head_weight"] = (0.12, 0.18, 0.24)[strictness] if not aggressive else (0.22, 0.32, 0.42)[strictness]
        spec["three_way_head_temperature"] = (0.32, 0.28, 0.24)[strictness] if not aggressive else (0.28, 0.23, 0.18)[strictness]
        spec["three_way_accept_prob"] = (0.62, 0.66, 0.70)[strictness] if not aggressive else (0.60, 0.64, 0.68)[strictness]
        spec["three_way_reject_prob"] = (0.94, 0.91, 0.88)[strictness] if not aggressive else (0.88, 0.84, 0.80)[strictness]
        spec["three_way_defer_prob"] = (0.78, 0.74, 0.70)[strictness] if not aggressive else (0.70, 0.66, 0.62)[strictness]
        spec["three_way_reject_margin"] = (0.36, 0.30, 0.24)[strictness] if not aggressive else (0.26, 0.18, 0.10)[strictness]

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = False
        spec["pre_reject_support_retention_max_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.22, 0.16, 0.10)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["pre_reject_reject_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_reject_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]

        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.88, 0.84, 0.80)[strictness] if not aggressive else (0.78, 0.72, 0.66)[strictness]
        spec["two_branch_bg_min_margin"] = (0.16, 0.10, 0.04)[strictness] if not aggressive else (0.06, 0.00, -0.06)[strictness]
        spec["two_branch_old_support_evidence_delta"] = (-0.18, -0.14, -0.10)[strictness] if not aggressive else (-0.12, -0.08, -0.04)[strictness]
        spec["siamese_unknown_veto"] = bool(aggressive and strictness >= 1)
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["min_veto_failures"] = 5 if not aggressive else 4
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 6 if not aggressive else 5
        spec["guard_min_margin"] = (0.20, 0.26, 0.32)[strictness] if not aggressive else (0.24, 0.32, 0.40)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2470
    return specs


def _oa_mse_h06_oldconf48_stage_specs() -> list[dict]:
    """48-row H06 support-conformal old-retention repair after OLDGEOM.

    OLDGEOM reduced unknown FAR but lost old retention. This plan keeps the
    same Stage2-B old/unknown-only boundary and tests whether support-conformal
    and local reconstruction evidence can rescue target-old accepts without
    fitting thresholds on unknown query samples or adding target-new enrollment.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_oldgeom48_stage_specs()]
    arms = (
        "conformal_retention_k5",
        "reconstruction_retention_k5",
        "source_looo_guard_k5",
        "old_anchor_rescue_k10",
        "support_cv_conformal_k5",
        "background_capped_rescue_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]

        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = 10 if arm.endswith("k10") else 5
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_old_unknown_support_conformal_retention_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "h06_oldgeom_negative_old_mean_0p107_unknown_far_0p150_hmean_0p158;"
            "oldgeom_improved_unknown_far_but_reduced_old_retention_vs_oldprimary;"
            "fresh_repair_uses_support_conformal_and_reconstruction_not_oldgeom_replay;"
            "stage2b_old_unknown_only_target_new_excluded_unknown_query_eval_only"
        )
        spec["description"] = (
            "H06-OLDCONF conservative repair: keep receiver 20-1 old/unknown-only Stage2-B, "
            "use support-conformal and local reconstruction evidence to defer uncertain background-like samples, "
            "then allow old-primary to promote only support-consistent retention-rescue candidates."
            if not aggressive
            else "H06-OLDCONF aggressive repair: test higher old-retention rescue under explicit support-conformal, "
            "reconstruction, source-LOOO, and background-cap vetoes; target-new remains excluded."
        )

        spec["source_proto_per_tx"] = 112 if not aggressive else 120
        spec["source_query_per_tx"] = 56 if not aggressive else 60
        spec["sfe_max_samples_per_tx"] = 560 if not aggressive else 600
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not aggressive else "residual_mlp"
        spec["adapter_selection_policy"] = "support_cv_constrained" if not aggressive else "constrained_retention_risk"
        spec["steps"] = (116, 128, 140)[strictness] if not aggressive else (128, 144, 160)[strictness]

        spec["support_center_ce"] = (0.60, 0.72, 0.84)[strictness] if not aggressive else (0.70, 0.86, 1.02)[strictness]
        spec["support_center_temperature"] = (0.085, 0.075, 0.065)[strictness] if not aggressive else (0.075, 0.065, 0.055)[strictness]
        spec["support_center_margin"] = (0.10, 0.13, 0.16)[strictness] if not aggressive else (0.14, 0.20, 0.26)[strictness]
        spec["support_contrast"] = (0.34, 0.44, 0.54)[strictness] if not aggressive else (0.46, 0.60, 0.74)[strictness]
        spec["known_coverage_weight"] = (4.80, 5.20, 5.60)[strictness] if not aggressive else (4.10, 4.70, 5.30)[strictness]
        spec["known_coverage_margin"] = (0.12, 0.15, 0.18)[strictness] if not aggressive else (0.15, 0.20, 0.25)[strictness]
        spec["known_coverage_min_affinity"] = (0.34, 0.38, 0.42)[strictness] if not aggressive else (0.36, 0.42, 0.48)[strictness]

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if spec["k_old"] == 5 else 6
        spec["multiproto_temperature"] = (0.085, 0.075, 0.065)[strictness] if not aggressive else (0.075, 0.062, 0.052)[strictness]
        spec["multiproto_score_weight"] = (1.35, 1.55, 1.75)[strictness] if not aggressive else (1.55, 1.85, 2.15)[strictness]
        spec["soft_proto"] = (1.30, 1.55, 1.80)[strictness] if not aggressive else (1.55, 1.90, 2.25)[strictness]
        spec["soft_proto_boundary"] = (0.24, 0.32, 0.40)[strictness] if not aggressive else (0.34, 0.46, 0.58)[strictness]
        spec["soft_proto_boundary_margin"] = (0.10, 0.14, 0.18)[strictness] if not aggressive else (0.16, 0.24, 0.32)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = (0.18, 0.26, 0.34)[strictness] if not aggressive else (0.26, 0.36, 0.46)[strictness]
        spec["mixture_consistency_max_residual"] = (1.32, 1.14, 0.96)[strictness] if not aggressive else (1.12, 0.92, 0.74)[strictness]
        spec["mixture_consistency_min_margin"] = (-0.14, -0.08, -0.02)[strictness] if not aggressive else (-0.08, 0.00, 0.08)[strictness]
        spec["mixture_consistency_action"] = "uncertain"

        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.06, 0.09, 0.12)[strictness]
        spec["class_envelope_residual_quantile"] = (0.99, 0.98, 0.97)[strictness] if not aggressive else (0.98, 0.96, 0.94)[strictness]
        spec["class_envelope_score_quantile"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.06, 0.09, 0.12)[strictness]
        spec["class_envelope_margin_quantile"] = (0.04, 0.06, 0.08)[strictness] if not aggressive else (0.06, 0.09, 0.12)[strictness]
        spec["class_envelope_evidence_slack"] = (0.14, 0.11, 0.08)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["class_envelope_residual_slack"] = (0.14, 0.11, 0.08)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["class_envelope_score_slack"] = (0.16, 0.12, 0.08)[strictness] if not aggressive else (0.12, 0.07, 0.02)[strictness]
        spec["class_envelope_margin_slack"] = (0.14, 0.11, 0.08)[strictness] if not aggressive else (0.10, 0.06, 0.02)[strictness]
        spec["class_envelope_min_failures"] = 3 if not aggressive else 2
        spec["class_envelope_gate_action"] = "uncertain"

        spec["old_primary_gate"] = True
        spec["old_primary_require_soft_mixture"] = True
        spec["old_primary_require_support_knn"] = True
        spec["old_primary_require_support_knn_label_match"] = True
        spec["old_primary_require_class_envelope"] = True
        spec["old_primary_promote_rescue_candidates"] = True
        spec["old_primary_min_old_support_evidence_delta"] = (-0.10, -0.06, -0.02)[strictness] if not aggressive else (-0.06, 0.00, 0.06)[strictness]
        spec["old_primary_min_old_support_anchor_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.09, -0.03, 0.03)[strictness]
        spec["old_primary_min_old_support_anchor_margin"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.04, 0.02, 0.08)[strictness]
        spec["old_primary_min_score_margin"] = (-0.16, -0.10, -0.04)[strictness] if not aggressive else (-0.08, 0.00, 0.08)[strictness]
        spec["old_primary_min_soft_mixture_margin"] = (-0.14, -0.08, -0.02)[strictness] if not aggressive else (-0.08, 0.00, 0.08)[strictness]
        spec["old_primary_min_soft_mixture_cos"] = spec["mixture_consistency_min_cos"]
        spec["old_primary_max_soft_mixture_residual"] = spec["mixture_consistency_max_residual"]
        spec["old_primary_min_support_knn_margin"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.02, 0.04)[strictness]
        spec["old_primary_max_support_knn_seen_new_minus_old"] = None
        spec["old_primary_min_old_drift_cos"] = (0.24, 0.31, 0.38)[strictness] if not aggressive else (0.30, 0.40, 0.50)[strictness]
        spec["old_primary_max_old_drift_dist"] = (0.90, 0.82, 0.74)[strictness] if not aggressive else (0.82, 0.72, 0.62)[strictness]
        spec["old_primary_unknown_veto_background_score"] = (0.92, 0.88, 0.84)[strictness] if not aggressive else (0.84, 0.78, 0.72)[strictness]
        spec["old_primary_unknown_veto_background_margin"] = (0.20, 0.14, 0.08)[strictness] if not aggressive else (0.12, 0.06, 0.00)[strictness]
        spec["old_primary_unknown_veto_min_sources"] = 2 if not aggressive else 1
        spec["old_primary_fail_action"] = "defer"
        spec["old_primary_unknown_veto_action"] = "reject"

        spec["support_conformal_arbitration"] = True
        spec["support_conformal_calibration_quantile"] = (0.02, 0.04, 0.06)[strictness] if not aggressive else (0.05, 0.08, 0.10)[strictness]
        spec["support_conformal_conformity_slack"] = (0.26, 0.21, 0.16)[strictness] if not aggressive else (0.18, 0.13, 0.08)[strictness]
        spec["support_conformal_anchor_margin_slack"] = (0.18, 0.14, 0.10)[strictness] if not aggressive else (0.12, 0.08, 0.04)[strictness]
        spec["support_conformal_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["support_conformal_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]
        spec["support_conformal_hard_reject_margin"] = (0.30, 0.24, 0.18)[strictness] if not aggressive else (0.22, 0.16, 0.10)[strictness]
        spec["support_conformal_reject_min_failures"] = 3 if not aggressive else 2
        spec["support_conformal_reject_action"] = "defer" if not aggressive or strictness == 0 else "reject"

        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_rank"] = 2 if spec["k_old"] == 5 else 3
        spec["support_reconstruction_residual_quantile"] = (0.98, 0.95, 0.92)[strictness] if not aggressive else (0.94, 0.90, 0.86)[strictness]
        spec["support_reconstruction_residual_slack"] = (0.12, 0.09, 0.06)[strictness] if not aggressive else (0.08, 0.05, 0.03)[strictness]
        spec["support_reconstruction_min_residual_floor"] = (0.06, 0.05, 0.04)[strictness] if not aggressive else (0.05, 0.04, 0.03)[strictness]
        spec["support_reconstruction_negative_scale"] = (0.45, 0.55, 0.65)[strictness] if not aggressive else (0.60, 0.70, 0.80)[strictness]
        spec["support_reconstruction_negative_margin"] = (0.08, 0.03, -0.02)[strictness] if not aggressive else (0.02, -0.04, -0.10)[strictness]
        spec["support_reconstruction_hard_residual_margin"] = (0.16, 0.12, 0.08)[strictness] if not aggressive else (0.12, 0.08, 0.05)[strictness]
        spec["support_reconstruction_background_score"] = (0.94, 0.90, 0.86)[strictness] if not aggressive else (0.86, 0.80, 0.74)[strictness]
        spec["support_reconstruction_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not aggressive else (0.14, 0.08, 0.02)[strictness]
        spec["support_reconstruction_reject_min_failures"] = 3 if not aggressive else 2
        spec["support_reconstruction_reject_action"] = "defer" if not aggressive or strictness == 0 else "reject"

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(aggressive or strictness >= 1)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if not aggressive else 0
        spec["pre_reject_support_retention_old_min_evidence_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.03, 0.02, 0.07)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_delta"] = (-0.14, -0.10, -0.06)[strictness] if not aggressive else (-0.08, -0.04, 0.00)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_margin"] = (-0.08, -0.05, -0.02)[strictness] if not aggressive else (-0.04, -0.01, 0.02)[strictness]
        spec["pre_reject_support_retention_old_min_score_margin"] = (-0.16, -0.11, -0.06)[strictness] if not aggressive else (-0.09, -0.04, 0.01)[strictness]
        spec["pre_reject_support_retention_seen_new_min_evidence_delta"] = (0.00, 0.02, 0.04)[strictness]
        spec["pre_reject_support_retention_seen_new_min_anchor_delta"] = (-0.06, -0.03, 0.00)[strictness]
        spec["pre_reject_support_retention_seen_new_min_score_margin"] = (-0.10, -0.06, -0.02)[strictness]
        spec["pre_reject_support_retention_max_background_score"] = (0.96, 0.92, 0.88)[strictness] if not aggressive else (0.88, 0.82, 0.76)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.28, 0.22, 0.16)[strictness] if not aggressive else (0.18, 0.10, 0.02)[strictness]

        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (-0.04, -0.01, 0.02)[strictness] if not aggressive else (-0.01, 0.03, 0.07)[strictness]
        spec["retention_rescue_old_min_anchor_delta"] = (-0.08, -0.04, 0.00)[strictness] if not aggressive else (-0.04, 0.00, 0.04)[strictness]
        spec["retention_rescue_old_min_anchor_margin"] = (-0.04, -0.01, 0.02)[strictness] if not aggressive else (-0.01, 0.02, 0.05)[strictness]
        spec["retention_rescue_old_min_score_margin"] = (-0.10, -0.05, 0.00)[strictness] if not aggressive else (-0.04, 0.02, 0.08)[strictness]
        spec["retention_rescue_seen_new_min_evidence_delta"] = (0.02, 0.04, 0.06)[strictness]
        spec["retention_rescue_seen_new_min_anchor_delta"] = (-0.02, 0.00, 0.02)[strictness]
        spec["retention_rescue_seen_new_min_score_margin"] = (-0.06, -0.02, 0.02)[strictness]
        spec["retention_rescue_max_background_score"] = (0.86, 0.82, 0.78)[strictness] if not aggressive else (0.76, 0.70, 0.64)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.18, 0.12, 0.06)[strictness] if not aggressive else (0.08, 0.02, -0.04)[strictness]

        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = (0.84, 0.88, 0.92)[strictness] if not aggressive else (0.88, 0.92, 0.95)[strictness]
        spec["source_looo_risk_slack"] = (0.04, 0.03, 0.02)[strictness]
        spec["source_looo_risk_min_score_margin"] = (0.10, 0.13, 0.16)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 4 if not aggressive else 3
        spec["source_looo_risk_reject_action"] = "defer"
        spec["unknown_moat"] = (0.12, 0.17, 0.22)[strictness] if not aggressive else (0.20, 0.28, 0.36)[strictness]
        spec["unknown_margin"] = (0.34, 0.44, 0.54)[strictness] if not aggressive else (0.48, 0.62, 0.76)[strictness]
        spec["negative_anchor_weight"] = (0.03, 0.05, 0.07)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["negative_anchor_margin"] = (0.05, 0.07, 0.09)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness]
        spec["siamese_unknown_veto"] = bool(aggressive and strictness >= 1)
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["min_veto_failures"] = 5 if not aggressive else 4
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_failures"] = 6 if not aggressive else 5
        spec["guard_min_margin"] = (0.18, 0.24, 0.30)[strictness] if not aggressive else (0.22, 0.30, 0.38)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2710
    return specs


def _oa_mse_h06_oldbudget48_stage_specs() -> list[dict]:
    """48-row H06 old/unknown-only acceptance-budget repair after OLDCONF.

    OLDGEOM improved unknown rejection at the cost of old retention, while
    OLDCONF split into high-old/high-FAR conservative rows and zero-old
    aggressive rows. This plan keeps the Stage2-B old/unknown-only boundary and
    converts the hard gates into a softer accept/defer budget before any future
    reject tightening.
    """

    specs = [dict(spec) for spec in _oa_mse_h06_oldconf48_stage_specs()]
    arms = (
        "soft_accept_budget_k5",
        "old_support_budget_k5",
        "far_balanced_budget_k5",
        "low_far_rescue_k5",
        "defer_first_budget_k10",
        "strict_budget_probe_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        guarded = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]

        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = 10 if arm.endswith("k10") else 5
        spec["k_new"] = 0
        spec["target_new_leo_support"] = False
        spec["category"] = "aggressive" if guarded else "conservative"
        spec["optimization_category"] = "far_guarded" if guarded else "old_budget"
        spec["route_suffix"] = f"h06_old_unknown_acceptance_budget_{arm}"
        spec["ablation_arm"] = arm
        spec["evidence_ref"] = (
            "h06_oldconf_negative_old_mean_0p128_unknown_far_0p193_hmean_0p172;"
            "oldconf_conservative_rows_recover_old_but_leak_unknown;"
            "oldconf_aggressive_rows_reject_unknown_but_collapse_old;"
            "fresh_repair_uses_acceptance_budget_and_defer_first_arbitration;"
            "stage2b_old_unknown_only_target_new_excluded_unknown_query_eval_only"
        )
        spec["description"] = (
            "H06-OLDBUDGET old-retention budget repair: keep receiver 20-1 old/unknown-only Stage2-B, "
            "reuse support-conformal and reconstruction evidence as defer-first signals, and accept only old "
            "samples that pass a soft old-support budget before any unknown-risk veto."
            if not guarded
            else "H06-OLDBUDGET FAR-guarded repair: preserve the useful OLDCONF conservative old rescue, "
            "but tighten background and source-LOOO budget only after old-support evidence fails; target-new remains excluded."
        )

        spec["source_proto_per_tx"] = 116 if not guarded else 120
        spec["source_query_per_tx"] = 58 if not guarded else 60
        spec["sfe_max_samples_per_tx"] = 580 if not guarded else 600
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not guarded else "residual_mlp"
        spec["adapter_selection_policy"] = "support_cv_constrained" if not guarded else "identity_preserving_cv"
        spec["steps"] = (112, 124, 136)[strictness] if not guarded else (120, 132, 144)[strictness]

        spec["support_center_ce"] = (0.54, 0.62, 0.70)[strictness] if not guarded else (0.62, 0.74, 0.86)[strictness]
        spec["support_center_temperature"] = (0.095, 0.085, 0.075)[strictness] if not guarded else (0.085, 0.075, 0.065)[strictness]
        spec["support_center_margin"] = (0.08, 0.10, 0.12)[strictness] if not guarded else (0.11, 0.15, 0.19)[strictness]
        spec["support_contrast"] = (0.28, 0.36, 0.44)[strictness] if not guarded else (0.38, 0.50, 0.62)[strictness]
        spec["known_coverage_weight"] = (5.20, 5.60, 6.00)[strictness] if not guarded else (4.80, 5.30, 5.80)[strictness]
        spec["known_coverage_margin"] = (0.10, 0.12, 0.14)[strictness] if not guarded else (0.12, 0.16, 0.20)[strictness]
        spec["known_coverage_min_affinity"] = (0.32, 0.35, 0.38)[strictness] if not guarded else (0.34, 0.38, 0.42)[strictness]

        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if spec["k_old"] == 5 else 6
        spec["multiproto_temperature"] = (0.090, 0.080, 0.070)[strictness] if not guarded else (0.080, 0.070, 0.060)[strictness]
        spec["multiproto_score_weight"] = (1.20, 1.40, 1.60)[strictness] if not guarded else (1.35, 1.60, 1.85)[strictness]
        spec["soft_proto"] = (1.16, 1.36, 1.56)[strictness] if not guarded else (1.34, 1.62, 1.90)[strictness]
        spec["soft_proto_topk"] = 5 if spec["k_old"] == 5 else 6
        spec["soft_proto_temperature"] = (0.090, 0.080, 0.070)[strictness] if not guarded else (0.080, 0.068, 0.058)[strictness]
        spec["soft_proto_boundary"] = (0.18, 0.24, 0.30)[strictness] if not guarded else (0.26, 0.34, 0.42)[strictness]
        spec["soft_proto_boundary_margin"] = (0.08, 0.11, 0.14)[strictness] if not guarded else (0.12, 0.17, 0.22)[strictness]
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = (0.14, 0.20, 0.26)[strictness] if not guarded else (0.20, 0.28, 0.36)[strictness]
        spec["mixture_consistency_max_residual"] = (1.46, 1.28, 1.10)[strictness] if not guarded else (1.30, 1.08, 0.88)[strictness]
        spec["mixture_consistency_min_margin"] = (-0.18, -0.12, -0.06)[strictness] if not guarded else (-0.12, -0.04, 0.04)[strictness]
        spec["mixture_consistency_action"] = "uncertain"

        spec["class_envelope_gate"] = True
        spec["class_envelope_evidence_quantile"] = (0.03, 0.045, 0.06)[strictness] if not guarded else (0.045, 0.065, 0.085)[strictness]
        spec["class_envelope_residual_quantile"] = (0.995, 0.985, 0.975)[strictness] if not guarded else (0.985, 0.965, 0.945)[strictness]
        spec["class_envelope_score_quantile"] = (0.03, 0.045, 0.06)[strictness] if not guarded else (0.045, 0.065, 0.085)[strictness]
        spec["class_envelope_margin_quantile"] = (0.03, 0.045, 0.06)[strictness] if not guarded else (0.045, 0.065, 0.085)[strictness]
        spec["class_envelope_evidence_slack"] = (0.18, 0.15, 0.12)[strictness] if not guarded else (0.14, 0.10, 0.06)[strictness]
        spec["class_envelope_residual_slack"] = (0.18, 0.15, 0.12)[strictness] if not guarded else (0.14, 0.10, 0.06)[strictness]
        spec["class_envelope_score_slack"] = (0.20, 0.16, 0.12)[strictness] if not guarded else (0.16, 0.11, 0.06)[strictness]
        spec["class_envelope_margin_slack"] = (0.18, 0.15, 0.12)[strictness] if not guarded else (0.14, 0.10, 0.06)[strictness]
        spec["class_envelope_min_failures"] = 4 if not guarded else 3
        spec["class_envelope_gate_action"] = "uncertain"

        spec["old_primary_gate"] = True
        spec["old_primary_require_soft_mixture"] = True
        spec["old_primary_require_support_knn"] = True
        spec["old_primary_require_support_knn_label_match"] = True
        spec["old_primary_require_class_envelope"] = True
        spec["old_primary_promote_rescue_candidates"] = True
        spec["old_primary_min_old_support_evidence_delta"] = (-0.16, -0.11, -0.06)[strictness] if not guarded else (-0.10, -0.04, 0.02)[strictness]
        spec["old_primary_min_old_support_anchor_delta"] = (-0.20, -0.15, -0.10)[strictness] if not guarded else (-0.14, -0.08, -0.02)[strictness]
        spec["old_primary_min_old_support_anchor_margin"] = (-0.12, -0.08, -0.04)[strictness] if not guarded else (-0.08, -0.03, 0.02)[strictness]
        spec["old_primary_min_score_margin"] = (-0.22, -0.16, -0.10)[strictness] if not guarded else (-0.14, -0.07, 0.00)[strictness]
        spec["old_primary_min_soft_mixture_margin"] = (-0.18, -0.12, -0.06)[strictness] if not guarded else (-0.12, -0.04, 0.04)[strictness]
        spec["old_primary_min_soft_mixture_cos"] = spec["mixture_consistency_min_cos"]
        spec["old_primary_max_soft_mixture_residual"] = spec["mixture_consistency_max_residual"]
        spec["old_primary_min_support_knn_margin"] = (-0.18, -0.13, -0.08)[strictness] if not guarded else (-0.12, -0.06, 0.00)[strictness]
        spec["old_primary_max_support_knn_seen_new_minus_old"] = None
        spec["old_primary_min_old_drift_cos"] = (0.18, 0.24, 0.30)[strictness] if not guarded else (0.24, 0.32, 0.40)[strictness]
        spec["old_primary_max_old_drift_dist"] = (1.02, 0.92, 0.82)[strictness] if not guarded else (0.92, 0.80, 0.68)[strictness]
        spec["old_primary_unknown_veto_background_score"] = (0.96, 0.92, 0.88)[strictness] if not guarded else (0.90, 0.84, 0.78)[strictness]
        spec["old_primary_unknown_veto_background_margin"] = (0.28, 0.22, 0.16)[strictness] if not guarded else (0.18, 0.10, 0.02)[strictness]
        spec["old_primary_unknown_veto_min_sources"] = 3 if not guarded else 2
        spec["old_primary_fail_action"] = "defer"
        spec["old_primary_unknown_veto_action"] = "defer" if not guarded else "reject"

        spec["support_conformal_arbitration"] = True
        spec["support_conformal_calibration_quantile"] = (0.015, 0.03, 0.045)[strictness] if not guarded else (0.035, 0.055, 0.075)[strictness]
        spec["support_conformal_conformity_slack"] = (0.32, 0.27, 0.22)[strictness] if not guarded else (0.24, 0.18, 0.12)[strictness]
        spec["support_conformal_anchor_margin_slack"] = (0.24, 0.19, 0.14)[strictness] if not guarded else (0.16, 0.11, 0.06)[strictness]
        spec["support_conformal_background_score"] = spec["old_primary_unknown_veto_background_score"]
        spec["support_conformal_background_margin"] = spec["old_primary_unknown_veto_background_margin"]
        spec["support_conformal_hard_reject_margin"] = (0.36, 0.30, 0.24)[strictness] if not guarded else (0.28, 0.20, 0.12)[strictness]
        spec["support_conformal_reject_min_failures"] = 4 if not guarded else 3
        spec["support_conformal_reject_action"] = "defer"

        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_rank"] = 2 if spec["k_old"] == 5 else 3
        spec["support_reconstruction_residual_quantile"] = (0.995, 0.98, 0.965)[strictness] if not guarded else (0.98, 0.95, 0.92)[strictness]
        spec["support_reconstruction_residual_slack"] = (0.16, 0.12, 0.09)[strictness] if not guarded else (0.11, 0.08, 0.05)[strictness]
        spec["support_reconstruction_min_residual_floor"] = (0.07, 0.06, 0.05)[strictness] if not guarded else (0.06, 0.05, 0.04)[strictness]
        spec["support_reconstruction_negative_scale"] = (0.38, 0.48, 0.58)[strictness] if not guarded else (0.52, 0.62, 0.72)[strictness]
        spec["support_reconstruction_negative_margin"] = (0.12, 0.08, 0.04)[strictness] if not guarded else (0.06, 0.00, -0.06)[strictness]
        spec["support_reconstruction_hard_residual_margin"] = (0.22, 0.17, 0.12)[strictness] if not guarded else (0.16, 0.11, 0.07)[strictness]
        spec["support_reconstruction_background_score"] = spec["old_primary_unknown_veto_background_score"]
        spec["support_reconstruction_background_margin"] = spec["old_primary_unknown_veto_background_margin"]
        spec["support_reconstruction_reject_min_failures"] = 4 if not guarded else 3
        spec["support_reconstruction_reject_action"] = "defer"

        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(guarded and strictness >= 1)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 2 if not guarded else 1
        spec["pre_reject_support_retention_old_min_evidence_delta"] = (-0.16, -0.11, -0.06)[strictness] if not guarded else (-0.09, -0.03, 0.03)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_delta"] = (-0.22, -0.17, -0.12)[strictness] if not guarded else (-0.14, -0.08, -0.02)[strictness]
        spec["pre_reject_support_retention_old_min_anchor_margin"] = (-0.13, -0.09, -0.05)[strictness] if not guarded else (-0.08, -0.04, 0.00)[strictness]
        spec["pre_reject_support_retention_old_min_score_margin"] = (-0.24, -0.18, -0.12)[strictness] if not guarded else (-0.14, -0.08, -0.02)[strictness]
        spec["pre_reject_support_retention_max_background_score"] = (0.98, 0.94, 0.90)[strictness] if not guarded else (0.90, 0.84, 0.78)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.34, 0.28, 0.22)[strictness] if not guarded else (0.20, 0.12, 0.04)[strictness]

        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (-0.10, -0.06, -0.02)[strictness] if not guarded else (-0.05, 0.00, 0.05)[strictness]
        spec["retention_rescue_old_min_anchor_delta"] = (-0.14, -0.10, -0.06)[strictness] if not guarded else (-0.08, -0.04, 0.00)[strictness]
        spec["retention_rescue_old_min_anchor_margin"] = (-0.08, -0.05, -0.02)[strictness] if not guarded else (-0.04, -0.01, 0.02)[strictness]
        spec["retention_rescue_old_min_score_margin"] = (-0.16, -0.10, -0.04)[strictness] if not guarded else (-0.08, -0.02, 0.04)[strictness]
        spec["retention_rescue_max_background_score"] = (0.92, 0.88, 0.84)[strictness] if not guarded else (0.82, 0.76, 0.70)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.24, 0.18, 0.12)[strictness] if not guarded else (0.12, 0.06, 0.00)[strictness]

        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = (0.80, 0.84, 0.88)[strictness] if not guarded else (0.84, 0.88, 0.92)[strictness]
        spec["source_looo_risk_slack"] = (0.06, 0.045, 0.03)[strictness]
        spec["source_looo_risk_min_score_margin"] = (0.06, 0.09, 0.12)[strictness]
        spec["source_looo_risk_reject_min_failures"] = 5 if not guarded else 4
        spec["source_looo_risk_reject_action"] = "defer"
        spec["unknown_moat"] = (0.10, 0.14, 0.18)[strictness] if not guarded else (0.16, 0.22, 0.28)[strictness]
        spec["unknown_margin"] = (0.30, 0.38, 0.46)[strictness] if not guarded else (0.42, 0.54, 0.66)[strictness]
        spec["negative_anchor_weight"] = (0.02, 0.035, 0.05)[strictness] if not guarded else (0.05, 0.075, 0.10)[strictness]
        spec["negative_anchor_margin"] = (0.04, 0.055, 0.07)[strictness] if not guarded else (0.06, 0.09, 0.12)[strictness]
        spec["siamese_unknown_veto"] = bool(guarded and strictness >= 1)
        spec["siamese_unknown_veto_mode"] = "coupled"
        spec["min_veto_failures"] = 5 if not guarded else 4
        spec["old_unknown_acceptance_guard"] = bool(guarded)
        spec["guard_min_failures"] = 6 if not guarded else 5
        spec["guard_min_margin"] = (0.12, 0.18, 0.24)[strictness] if not guarded else (0.18, 0.26, 0.34)[strictness]
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.90
        spec["seen_new_acc_target"] = 0.75
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 2930
    return specs


def _oa_mse_h06_oldqual48_stage_specs() -> list[dict]:
    """H06 old/unknown support-quality repair after the OLDBUDGET diagnostic."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldbudget48_stage_specs()]
    arms = (
        "support_quality_cv_k5",
        "support_center_mix_k5",
        "prototype_compactness_k5",
        "support_reconstruction_k5",
        "prototype_geometry_k10",
        "support_quality_risk_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        prototype_arm = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]
        k_old = 10 if arm.endswith("k10") else 5
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = k_old
        spec["k_new"] = 0
        spec["category"] = "prototype_geometry" if prototype_arm else "support_quality"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_support_quality_prototype_construction_{arm}"
        spec["ablation_arm"] = arm
        spec["target_new_leo_support"] = False
        spec["source_proto_per_tx"] = 16 if k_old == 5 else 18
        spec["source_query_per_tx"] = 34 if k_old == 5 else 36
        spec["sfe_max_samples_per_tx"] = 50 if k_old == 5 else 54
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "residual_mlp" if prototype_arm else "low_rank"
        spec["adapter_selection_policy"] = (
            "identity_preserving_cv" if prototype_arm else "support_cv_constrained"
        )
        spec["steps"] = (112, 124, 136)[strictness] if prototype_arm else (104, 112, 120)[strictness]
        spec["align"] = round(0.14 + 0.01 * strictness + (0.01 if prototype_arm else 0.0), 3)
        spec["norm"] = round(0.020 + 0.002 * strictness, 3)
        spec["bce"] = round(0.044 + 0.003 * strictness, 3)
        spec["unknown"] = round(0.030 + 0.004 * strictness, 3)
        spec["pl"] = round(0.025 + 0.003 * strictness, 3)
        spec["maha"] = round(0.30 + 0.02 * strictness, 3)
        spec["unknown_moat"] = round(0.14 + 0.02 * strictness, 3)
        spec["unknown_margin"] = round(0.10 + 0.02 * strictness, 3)
        spec["negative_anchor_weight"] = round(0.055 + 0.008 * strictness, 3)
        spec["negative_anchor_margin"] = round(0.17 + 0.02 * strictness, 3)
        spec["support_center_ce"] = round(0.16 + 0.02 * strictness + (0.02 if prototype_arm else 0.0), 3)
        spec["support_center_temperature"] = round(0.52 - 0.03 * strictness, 3)
        spec["support_center_margin"] = round(0.12 + 0.02 * strictness, 3)
        spec["support_contrast"] = round(0.030 + 0.006 * strictness, 3)
        spec["known_coverage_weight"] = round(0.12 + 0.015 * strictness, 3)
        spec["known_coverage_margin"] = round(0.08 + 0.015 * strictness, 3)
        spec["known_coverage_min_affinity"] = round(0.18 + 0.02 * strictness, 3)
        spec["multiproto_score"] = True
        spec["multiproto_score_topk"] = 5 if prototype_arm else 4
        spec["multiproto_score_temperature"] = round(0.50 - 0.03 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.16 + 0.02 * strictness, 3)
        spec["soft_proto"] = True
        spec["soft_proto_topk"] = spec["multiproto_score_topk"]
        spec["soft_proto_temp"] = round(0.48 - 0.03 * strictness, 3)
        spec["soft_proto_weight"] = round(0.14 + 0.02 * strictness, 3)
        spec["soft_proto_boundary"] = True
        spec["soft_proto_boundary_margin"] = round(0.11 + 0.02 * strictness, 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.58 + 0.03 * strictness, 3)
        spec["mixture_consistency_max_residual"] = round(0.34 - 0.02 * strictness, 3)
        spec["mixture_consistency_margin"] = round(0.055 + 0.012 * strictness, 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 4 - min(strictness, 1)
        spec["old_primary_gate"] = False
        spec["old_primary_require_soft_mixture"] = False
        spec["old_primary_require_support_knn"] = False
        spec["old_primary_require_class_envelope"] = False
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = False
        spec["retention_rescue_candidate_only"] = False
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_alpha"] = round(0.14 - 0.02 * strictness, 3)
        spec["support_conformal_min_support"] = 3 if k_old == 5 else 5
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_max"] = round(0.48 - 0.03 * strictness, 3)
        spec["support_reconstruction_min_gap"] = round(0.035 + 0.01 * strictness, 3)
        spec["support_reconstruction_reject_action"] = "defer"
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_max"] = round(0.40 - 0.03 * strictness, 3)
        spec["source_looo_unknown_weight"] = round(0.055 + 0.008 * strictness, 3)
        spec["source_looo_risk_reject_min_failures"] = 5 - min(strictness, 1)
        spec["source_looo_risk_reject_action"] = "defer"
        spec["old_unknown_acceptance_guard"] = False
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.90
        spec["seen_new_acc_target"] = 0.75
        spec["risk_note"] = (
            "OLDBUDGET was negative because old retention improved while unknown FAR worsened "
            "and target_hit_count stayed 0; this route changes support-quality/prototype construction "
            "instead of stacking another terminal old-primary gate."
        )
        spec["description"] = (
            "Stage2-B H06 old/unknown-only support-quality/prototype construction repair; "
            "target-new support is excluded, unknown transmitters remain query-only, and support "
            "conformal/reconstruction checks defer rather than promote deployment claims."
        )
        spec["evidence_ref"] = (
            "oldbudget_negative_old_mean_0p226_unknown_far_0p304_target_hit0;"
            "old_retention_improved_unknown_far_worsened;"
            "fresh_h06_support_quality_prototype_construction_repair;"
            "stage2b_old_unknown_only_target_new_excluded"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 3190 + idx * 13
    return specs


def _oa_mse_h06_oldrisk48_stage_specs() -> list[dict]:
    """H06 query-free background-risk repair after the OLDQUAL diagnostic."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldqual48_stage_specs()]
    arms = (
        "bg_risk_cap_k5",
        "source_looo_bg_k5",
        "two_branch_joint_k5",
        "pre_reject_margin_k5",
        "unknown_separation_k10",
        "joint_veto_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        unknown_sep_arm = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]
        k_old = 10 if arm.endswith("k10") else 5
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = k_old
        spec["k_new"] = 0
        spec["category"] = "unknown_separability" if unknown_sep_arm else "query_free_background_risk"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_unknown_separability_query_free_background_risk_{arm}"
        spec["ablation_arm"] = arm
        spec["target_new_leo_support"] = False
        spec["source_proto_per_tx"] = 18 if k_old == 5 else 20
        spec["source_query_per_tx"] = 36 if k_old == 5 else 38
        spec["sfe_max_samples_per_tx"] = 54 if k_old == 5 else 58
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not unknown_sep_arm else "residual_mlp"
        spec["adapter_selection_policy"] = (
            "constrained_retention_risk" if not unknown_sep_arm else "identity_preserving_risk"
        )
        spec["steps"] = (96, 108, 120)[strictness] if not unknown_sep_arm else (112, 124, 136)[strictness]
        spec["align"] = round(0.11 + 0.01 * strictness, 3)
        spec["norm"] = round(0.018 + 0.002 * strictness, 3)
        spec["bce"] = round(0.035 + 0.004 * strictness, 3)
        spec["unknown"] = round(0.055 + 0.010 * strictness + (0.020 if unknown_sep_arm else 0.0), 3)
        spec["pl"] = round(0.018 + 0.003 * strictness, 3)
        spec["maha"] = round(0.26 + 0.03 * strictness, 3)
        spec["unknown_moat"] = round(0.28 + 0.06 * strictness + (0.12 if unknown_sep_arm else 0.0), 3)
        spec["unknown_margin"] = round(0.34 + 0.08 * strictness + (0.18 if unknown_sep_arm else 0.0), 3)
        spec["negative_anchor_weight"] = round(0.16 + 0.035 * strictness + (0.060 if unknown_sep_arm else 0.0), 3)
        spec["negative_anchor_margin"] = round(0.24 + 0.040 * strictness + (0.080 if unknown_sep_arm else 0.0), 3)
        spec["negative_anchor_temperature"] = round(0.12 - 0.015 * min(strictness, 2), 3)
        spec["negative_anchor_max_anchors"] = 256 if not unknown_sep_arm else 320
        spec["void_background"] = round(0.06 + 0.020 * strictness + (0.040 if unknown_sep_arm else 0.0), 3)
        spec["void_gate"] = bool(unknown_sep_arm or strictness >= 1)
        spec["void_gate_min_score"] = round(0.74 + 0.04 * strictness, 3)
        spec["void_gate_min_margin"] = round(0.04 + 0.04 * strictness, 3)
        spec["source_looo_unknown_weight"] = round(0.16 + 0.050 * strictness + (0.060 if unknown_sep_arm else 0.0), 3)
        spec["source_looo_unknown_margin"] = round(0.44 + 0.060 * strictness + (0.080 if unknown_sep_arm else 0.0), 3)
        spec["source_looo_interclass_margin"] = round(0.12 + 0.020 * strictness, 3)
        spec["source_looo_max_samples_per_class"] = 40 if not unknown_sep_arm else 48
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = round(0.88 + 0.025 * strictness, 3)
        spec["source_looo_risk_slack"] = round(0.015 + 0.010 * strictness, 3)
        spec["source_looo_risk_min_score_margin"] = round(0.08 + 0.030 * strictness, 3)
        spec["source_looo_risk_min_known_evidence_delta"] = round(-0.04 + 0.020 * strictness, 3)
        spec["source_looo_risk_background_score"] = round(0.68 + 0.040 * strictness, 3)
        spec["source_looo_risk_background_margin"] = round(0.00 + 0.040 * strictness, 3)
        spec["source_looo_risk_reject_min_failures"] = 3 + min(strictness, 1)
        spec["source_looo_risk_reject_action"] = "defer"
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = round(0.62 + 0.050 * strictness + (0.060 if unknown_sep_arm else 0.0), 3)
        spec["two_branch_bg_min_margin"] = round(-0.02 + 0.050 * strictness + (0.040 if unknown_sep_arm else 0.0), 3)
        spec["two_branch_old_support_evidence_delta"] = round(-0.08 + 0.020 * strictness, 3)
        spec["two_branch_old_anchor_delta"] = round(-0.06 + 0.020 * strictness, 3)
        spec["two_branch_old_anchor_margin"] = round(0.02 + 0.015 * strictness, 3)
        spec["two_branch_seen_new_evidence_delta"] = 0.0
        spec["two_branch_seen_new_anchor_delta"] = 0.0
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_max_background_score"] = round(0.70 + 0.040 * strictness, 3)
        spec["pre_reject_max_background_margin"] = round(0.00 + 0.040 * strictness, 3)
        spec["pre_reject_defer_background_score"] = round(0.64 + 0.040 * strictness, 3)
        spec["pre_reject_defer_background_margin"] = round(-0.06 + 0.040 * strictness, 3)
        spec["pre_reject_reject_background_score"] = round(0.76 + 0.040 * strictness, 3)
        spec["pre_reject_reject_background_margin"] = round(0.04 + 0.050 * strictness, 3)
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_max_background_score"] = round(0.70 + 0.030 * strictness, 3)
        spec["pre_reject_support_retention_max_background_margin"] = round(0.00 + 0.035 * strictness, 3)
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = round(0.06 + 0.015 * strictness, 3)
        spec["support_retention_guard_slack"] = round(0.10 + 0.020 * strictness, 3)
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = round(-0.08 + 0.020 * strictness, 3)
        spec["guard_min_old_surrogate_reject_delta"] = round(0.00 + 0.020 * strictness, 3)
        spec["guard_min_energy_delta"] = round(0.0 + 4.0 * strictness, 3)
        spec["guard_min_mahalanobis_delta"] = round(-8.0 + 4.0 * strictness, 3)
        spec["guard_min_accept_delta"] = round(-8.0 + 3.0 * strictness, 3)
        spec["guard_min_old_support_anchor_margin"] = round(0.035 + 0.010 * strictness, 3)
        spec["guard_min_best_old_score"] = round(-1.2 + 0.20 * strictness, 3)
        spec["guard_min_margin"] = round(0.34 + 0.080 * strictness + (0.060 if unknown_sep_arm else 0.0), 3)
        spec["guard_min_failures"] = 4 if not unknown_sep_arm else 5
        spec["support_center_ce"] = round(0.08 + 0.010 * strictness, 3)
        spec["support_contrast"] = round(0.020 + 0.004 * strictness, 3)
        spec["multiproto_score"] = True
        spec["multiproto_score_topk"] = 4
        spec["multiproto_score_temperature"] = round(0.46 - 0.02 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.10 + 0.015 * strictness, 3)
        spec["soft_proto"] = True
        spec["soft_proto_topk"] = 4
        spec["soft_proto_temp"] = round(0.44 - 0.02 * strictness, 3)
        spec["soft_proto_weight"] = round(0.10 + 0.015 * strictness, 3)
        spec["soft_proto_boundary"] = True
        spec["soft_proto_boundary_margin"] = round(0.08 + 0.015 * strictness, 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.50 + 0.03 * strictness, 3)
        spec["mixture_consistency_max_residual"] = round(0.42 - 0.02 * strictness, 3)
        spec["mixture_consistency_margin"] = round(0.035 + 0.010 * strictness, 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 3
        spec["old_primary_gate"] = False
        spec["old_primary_require_soft_mixture"] = False
        spec["old_primary_require_support_knn"] = False
        spec["old_primary_require_class_envelope"] = False
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = False
        spec["retention_rescue_candidate_only"] = False
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_reject_action"] = "defer"
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.90
        spec["seen_new_acc_target"] = 0.75
        spec["risk_note"] = (
            "OLDQUAL was negative because old retention rose but unknown FAR worsened badly and "
            "target_hit_count stayed 0; this route changes query-free background-risk separation "
            "and old/unknown joint veto instead of support-quality/prototype construction."
        )
        spec["description"] = (
            "Stage2-B H06 old/unknown-only query-free background-risk repair; target-new support "
            "is excluded, unknown transmitters remain query-only, and background/LOOO risk signals "
            "gate old accepts without fitting thresholds on unknown query."
        )
        spec["evidence_ref"] = (
            "oldqual_negative_old_mean_0p436_unknown_far_0p676_target_hit0;"
            "oldqual_old_retention_improved_unknown_far_worsened_severely;"
            "fresh_h06_unknown_separability_query_free_background_risk_repair;"
            "stage2b_old_unknown_only_target_new_excluded"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 3470 + idx * 17
    return specs


def _oa_mse_h06_oldfuse48_stage_specs() -> list[dict]:
    """H06 OLDQUAL/OLDRISK fusion plus rollback-calibration repair."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldrisk48_stage_specs()]
    arms = (
        "qual_risk_fusion_k5",
        "old_retention_risk_k5",
        "background_softcap_k5",
        "rollback_relax_k5",
        "hmean_calibration_k10",
        "rollback_strict_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        rollback_arm = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]
        k_old = 10 if arm.endswith("k10") else 5
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = k_old
        spec["k_new"] = 0
        spec["category"] = "rollback_calibration" if rollback_arm else "oldqual_oldrisk_fusion"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_oldqual_oldrisk_fusion_rollback_calibration_{arm}"
        spec["ablation_arm"] = arm
        spec["target_new_leo_support"] = False
        spec["source_proto_per_tx"] = 18 if k_old == 5 else 20
        spec["source_query_per_tx"] = 36 if k_old == 5 else 38
        spec["sfe_max_samples_per_tx"] = 54 if k_old == 5 else 58
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not rollback_arm else "residual_mlp"
        spec["adapter_selection_policy"] = (
            "constrained_retention_risk" if not rollback_arm else "identity_preserving_risk"
        )
        spec["steps"] = (104, 116, 128)[strictness] if not rollback_arm else (112, 124, 136)[strictness]
        spec["align"] = round(0.12 + 0.01 * strictness + (0.01 if rollback_arm else 0.0), 3)
        spec["norm"] = round(0.018 + 0.002 * strictness, 3)
        spec["bce"] = round(0.038 + 0.004 * strictness, 3)
        spec["unknown"] = round(0.045 + 0.008 * strictness + (0.010 if rollback_arm else 0.0), 3)
        spec["pl"] = round(0.020 + 0.003 * strictness, 3)
        spec["maha"] = round(0.26 + 0.025 * strictness, 3)
        spec["unknown_moat"] = round(0.22 + 0.04 * strictness + (0.06 if rollback_arm else 0.0), 3)
        spec["unknown_margin"] = round(0.26 + 0.06 * strictness + (0.10 if rollback_arm else 0.0), 3)
        spec["negative_anchor_weight"] = round(0.10 + 0.025 * strictness + (0.035 if rollback_arm else 0.0), 3)
        spec["negative_anchor_margin"] = round(0.20 + 0.035 * strictness + (0.045 if rollback_arm else 0.0), 3)
        spec["negative_anchor_temperature"] = round(0.14 - 0.015 * min(strictness, 2), 3)
        spec["negative_anchor_max_anchors"] = 288 if not rollback_arm else 320
        spec["void_background"] = round(0.035 + 0.015 * strictness + (0.025 if rollback_arm else 0.0), 3)
        spec["void_gate"] = bool(strictness >= 1 or rollback_arm)
        spec["void_gate_min_score"] = round(0.68 + 0.04 * strictness + (0.04 if rollback_arm else 0.0), 3)
        spec["void_gate_min_margin"] = round(-0.02 + 0.035 * strictness + (0.03 if rollback_arm else 0.0), 3)
        spec["source_looo_unknown_weight"] = round(0.10 + 0.035 * strictness + (0.035 if rollback_arm else 0.0), 3)
        spec["source_looo_unknown_margin"] = round(0.34 + 0.055 * strictness + (0.050 if rollback_arm else 0.0), 3)
        spec["source_looo_interclass_margin"] = round(0.10 + 0.015 * strictness, 3)
        spec["source_looo_max_samples_per_class"] = 40 if not rollback_arm else 48
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = round(0.84 + 0.025 * strictness + (0.015 if rollback_arm else 0.0), 3)
        spec["source_looo_risk_slack"] = round(0.025 + 0.010 * strictness, 3)
        spec["source_looo_risk_min_score_margin"] = round(0.04 + 0.025 * strictness, 3)
        spec["source_looo_risk_min_known_evidence_delta"] = round(-0.10 + 0.020 * strictness, 3)
        spec["source_looo_risk_background_score"] = round(0.64 + 0.035 * strictness + (0.040 if rollback_arm else 0.0), 3)
        spec["source_looo_risk_background_margin"] = round(-0.04 + 0.035 * strictness + (0.035 if rollback_arm else 0.0), 3)
        spec["source_looo_risk_reject_min_failures"] = 3 + min(strictness, 1) + (1 if rollback_arm else 0)
        spec["source_looo_risk_reject_action"] = "defer"
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = round(0.58 + 0.04 * strictness + (0.05 if rollback_arm else 0.0), 3)
        spec["two_branch_bg_min_margin"] = round(-0.06 + 0.04 * strictness + (0.04 if rollback_arm else 0.0), 3)
        spec["two_branch_old_support_evidence_delta"] = round(-0.10 + 0.020 * strictness, 3)
        spec["two_branch_old_anchor_delta"] = round(-0.08 + 0.020 * strictness, 3)
        spec["two_branch_old_anchor_margin"] = round(0.00 + 0.015 * strictness, 3)
        spec["two_branch_seen_new_evidence_delta"] = 0.0
        spec["two_branch_seen_new_anchor_delta"] = 0.0
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "uncertain"
        spec["pre_reject_max_background_score"] = round(0.68 + 0.035 * strictness + (0.030 if rollback_arm else 0.0), 3)
        spec["pre_reject_max_background_margin"] = round(-0.02 + 0.030 * strictness + (0.025 if rollback_arm else 0.0), 3)
        spec["pre_reject_defer_background_score"] = round(0.62 + 0.035 * strictness + (0.025 if rollback_arm else 0.0), 3)
        spec["pre_reject_defer_background_margin"] = round(-0.08 + 0.035 * strictness + (0.025 if rollback_arm else 0.0), 3)
        spec["pre_reject_reject_background_score"] = round(0.72 + 0.035 * strictness + (0.035 if rollback_arm else 0.0), 3)
        spec["pre_reject_reject_background_margin"] = round(0.00 + 0.040 * strictness + (0.035 if rollback_arm else 0.0), 3)
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_max_background_score"] = round(0.74 + 0.030 * strictness, 3)
        spec["pre_reject_support_retention_max_background_margin"] = round(0.02 + 0.035 * strictness, 3)
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(rollback_arm)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 2 if rollback_arm else 3
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = round(0.04 + 0.012 * strictness, 3)
        spec["support_retention_guard_slack"] = round(0.12 + 0.018 * strictness, 3)
        spec["support_center_ce"] = round(0.12 + 0.018 * strictness + (0.015 if not rollback_arm else 0.0), 3)
        spec["support_center_temperature"] = round(0.50 - 0.03 * strictness, 3)
        spec["support_center_margin"] = round(0.10 + 0.018 * strictness, 3)
        spec["support_contrast"] = round(0.026 + 0.005 * strictness, 3)
        spec["known_coverage_weight"] = round(0.10 + 0.012 * strictness + (0.02 if not rollback_arm else 0.0), 3)
        spec["known_coverage_margin"] = round(0.08 + 0.012 * strictness, 3)
        spec["known_coverage_min_affinity"] = round(0.20 + 0.02 * strictness, 3)
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if k_old == 10 else 4
        spec["multiproto_temperature"] = round(0.48 - 0.02 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.12 + 0.015 * strictness, 3)
        spec["soft_proto"] = round(0.12 + 0.015 * strictness, 3)
        spec["soft_proto_topk"] = spec["multiproto_topk"]
        spec["soft_proto_temperature"] = round(0.46 - 0.02 * strictness, 3)
        spec["soft_proto_boundary"] = round(0.08 + 0.012 * strictness, 3)
        spec["soft_proto_boundary_margin"] = round(0.09 + 0.014 * strictness, 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.54 + 0.025 * strictness, 3)
        spec["mixture_consistency_max_residual"] = round(0.38 - 0.018 * strictness, 3)
        spec["mixture_consistency_min_margin"] = round(0.045 + 0.010 * strictness, 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 3
        spec["old_primary_gate"] = False
        spec["old_primary_require_soft_mixture"] = False
        spec["old_primary_require_support_knn"] = False
        spec["old_primary_require_class_envelope"] = False
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = False
        spec["retention_rescue_candidate_only"] = False
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_alpha"] = round(0.12 - 0.015 * strictness, 3)
        spec["support_conformal_min_support"] = 3 if k_old == 5 else 5
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_max"] = round(0.50 - 0.025 * strictness, 3)
        spec["support_reconstruction_min_gap"] = round(0.030 + 0.008 * strictness, 3)
        spec["support_reconstruction_reject_action"] = "defer"
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = round(-0.10 + 0.020 * strictness, 3)
        spec["guard_min_old_surrogate_reject_delta"] = round(-0.02 + 0.020 * strictness, 3)
        spec["guard_min_energy_delta"] = round(-4.0 + 3.0 * strictness, 3)
        spec["guard_min_mahalanobis_delta"] = round(-12.0 + 4.0 * strictness, 3)
        spec["guard_min_accept_delta"] = round(-10.0 + 3.0 * strictness, 3)
        spec["guard_min_old_support_anchor_margin"] = round(0.025 + 0.010 * strictness, 3)
        spec["guard_min_best_old_score"] = round(-1.4 + 0.20 * strictness, 3)
        spec["guard_min_margin"] = round(0.24 + 0.060 * strictness + (0.050 if rollback_arm else 0.0), 3)
        spec["guard_min_failures"] = 3 + min(strictness, 1) + (1 if rollback_arm else 0)
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.90
        spec["seen_new_acc_target"] = 0.75
        spec["risk_note"] = (
            "OLDQUAL preserved old retention but leaked unknowns; OLDRISK reduced unknown FAR but "
            "dropped old retention and triggered rollback for every row. This route fuses support-quality "
            "old evidence with query-free risk and explicitly calibrates defer/rollback pressure."
        )
        spec["description"] = (
            "Stage2-B H06 old/unknown-only OLDQUAL+OLDRISK fusion and rollback-calibration repair; "
            "target-new support remains excluded, unknown transmitters remain query-only, and accept/defer "
            "thresholds are tested without fitting on unknown query labels."
        )
        spec["evidence_ref"] = (
            "oldqual_negative_old_mean_0p436_unknown_far_0p676_target_hit0;"
            "oldrisk_negative_old_mean_0p404_unknown_far_0p485_rollback48_target_hit0;"
            "fresh_h06_oldqual_oldrisk_fusion_rollback_calibration_repair;"
            "stage2b_old_unknown_only_target_new_excluded"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 3790 + idx * 19
    return specs


def _oa_mse_h06_rollsafe48_stage_specs() -> list[dict]:
    """H06 rollback-safe old-retention repair after OLDFUSE all-rollback."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldfuse48_stage_specs()]
    arms = (
        "retention_rescue_k5",
        "support_evidence_k5",
        "defer_bias_k5",
        "deployment_gate_k5",
        "hmean_probe_k10",
        "rollback_floor_k10",
    )
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        gate_arm = idx >= (len(specs) // 2)
        arm = arms[idx % len(arms)]
        k_old = 10 if arm.endswith("k10") else 5
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = k_old
        spec["k_new"] = 0
        spec["category"] = "deployment_gate_rescue" if gate_arm else "rollback_safe_retention"
        spec["optimization_category"] = spec["category"]
        spec["route_suffix"] = f"h06_rollsafe_retention_repair_{arm}"
        spec["ablation_arm"] = arm
        spec["target_new_leo_support"] = False
        spec["source_proto_per_tx"] = 20 if k_old == 5 else 22
        spec["source_query_per_tx"] = 38 if k_old == 5 else 40
        spec["sfe_max_samples_per_tx"] = 58 if k_old == 5 else 62
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if not gate_arm else "residual_mlp"
        spec["adapter_selection_policy"] = (
            "constrained_retention_risk" if not gate_arm else "identity_preserving_risk"
        )
        spec["steps"] = (96, 108, 120)[strictness] if not gate_arm else (104, 116, 128)[strictness]
        spec["align"] = round(0.115 + 0.010 * strictness + (0.005 if gate_arm else 0.0), 3)
        spec["norm"] = round(0.020 + 0.002 * strictness, 3)
        spec["bce"] = round(0.032 + 0.004 * strictness, 3)
        spec["unknown"] = round(0.038 + 0.006 * strictness + (0.008 if gate_arm else 0.0), 3)
        spec["pl"] = round(0.018 + 0.003 * strictness, 3)
        spec["maha"] = round(0.22 + 0.020 * strictness, 3)
        spec["unknown_moat"] = round(0.17 + 0.030 * strictness + (0.030 if gate_arm else 0.0), 3)
        spec["unknown_margin"] = round(0.20 + 0.045 * strictness + (0.060 if gate_arm else 0.0), 3)
        spec["negative_anchor_weight"] = round(0.075 + 0.020 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["negative_anchor_margin"] = round(0.16 + 0.030 * strictness + (0.030 if gate_arm else 0.0), 3)
        spec["negative_anchor_temperature"] = round(0.15 - 0.012 * min(strictness, 2), 3)
        spec["negative_anchor_max_anchors"] = 256 if not gate_arm else 288
        spec["void_background"] = round(0.025 + 0.010 * strictness + (0.018 if gate_arm else 0.0), 3)
        spec["void_gate"] = bool(gate_arm or strictness >= 2)
        spec["void_gate_min_score"] = round(0.62 + 0.030 * strictness + (0.030 if gate_arm else 0.0), 3)
        spec["void_gate_min_margin"] = round(-0.08 + 0.030 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["source_looo_unknown_weight"] = round(0.070 + 0.026 * strictness + (0.030 if gate_arm else 0.0), 3)
        spec["source_looo_unknown_margin"] = round(0.28 + 0.045 * strictness + (0.045 if gate_arm else 0.0), 3)
        spec["source_looo_interclass_margin"] = round(0.09 + 0.012 * strictness, 3)
        spec["source_looo_max_samples_per_class"] = 48 if not gate_arm else 52
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = round(0.80 + 0.020 * strictness + (0.015 if gate_arm else 0.0), 3)
        spec["source_looo_risk_slack"] = round(0.040 + 0.012 * strictness, 3)
        spec["source_looo_risk_min_score_margin"] = round(0.010 + 0.018 * strictness, 3)
        spec["source_looo_risk_min_known_evidence_delta"] = round(-0.18 + 0.030 * strictness, 3)
        spec["source_looo_risk_background_score"] = round(0.58 + 0.030 * strictness + (0.035 if gate_arm else 0.0), 3)
        spec["source_looo_risk_background_margin"] = round(-0.10 + 0.030 * strictness + (0.030 if gate_arm else 0.0), 3)
        spec["source_looo_risk_reject_min_failures"] = 4 if not gate_arm else 3 + min(strictness, 1)
        spec["source_looo_risk_reject_action"] = "defer"
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = round(0.52 + 0.035 * strictness + (0.045 if gate_arm else 0.0), 3)
        spec["two_branch_bg_min_margin"] = round(-0.14 + 0.035 * strictness + (0.035 if gate_arm else 0.0), 3)
        spec["two_branch_old_support_evidence_delta"] = round(-0.18 + 0.030 * strictness, 3)
        spec["two_branch_old_anchor_delta"] = round(-0.14 + 0.025 * strictness, 3)
        spec["two_branch_old_anchor_margin"] = round(-0.02 + 0.015 * strictness, 3)
        spec["two_branch_seen_new_evidence_delta"] = 0.0
        spec["two_branch_seen_new_anchor_delta"] = 0.0
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "defer"
        spec["pre_reject_max_background_score"] = round(0.76 + 0.025 * strictness + (0.020 if gate_arm else 0.0), 3)
        spec["pre_reject_max_background_margin"] = round(0.02 + 0.025 * strictness + (0.020 if gate_arm else 0.0), 3)
        spec["pre_reject_defer_background_score"] = round(0.56 + 0.030 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["pre_reject_defer_background_margin"] = round(-0.12 + 0.030 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["pre_reject_reject_background_score"] = round(0.82 + 0.025 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["pre_reject_reject_background_margin"] = round(0.06 + 0.030 * strictness + (0.025 if gate_arm else 0.0), 3)
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_old_min_evidence_delta"] = (
            (-0.14, -0.10, -0.06)[strictness] if not gate_arm else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["pre_reject_support_retention_old_min_anchor_delta"] = (
            (-0.18, -0.14, -0.10)[strictness] if not gate_arm else (-0.12, -0.08, -0.04)[strictness]
        )
        spec["pre_reject_support_retention_old_min_anchor_margin"] = (
            (-0.10, -0.07, -0.04)[strictness] if not gate_arm else (-0.06, -0.03, 0.00)[strictness]
        )
        spec["pre_reject_support_retention_old_min_score_margin"] = (
            (-0.20, -0.16, -0.12)[strictness] if not gate_arm else (-0.14, -0.10, -0.06)[strictness]
        )
        spec["pre_reject_support_retention_max_background_score"] = (
            (0.96, 0.93, 0.90)[strictness] if not gate_arm else (0.88, 0.84, 0.80)[strictness]
        )
        spec["pre_reject_support_retention_max_background_margin"] = (
            (0.24, 0.18, 0.12)[strictness] if not gate_arm else (0.12, 0.06, 0.00)[strictness]
        )
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(gate_arm and strictness >= 1)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 2 if gate_arm else 3
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = round(0.030 + 0.010 * strictness, 3)
        spec["support_retention_guard_slack"] = round(0.16 + 0.020 * strictness, 3)
        spec["support_center_ce"] = round(0.14 + 0.018 * strictness + (0.010 if not gate_arm else 0.0), 3)
        spec["support_center_temperature"] = round(0.54 - 0.035 * strictness, 3)
        spec["support_center_margin"] = round(0.08 + 0.015 * strictness, 3)
        spec["support_contrast"] = round(0.030 + 0.005 * strictness, 3)
        spec["known_coverage_weight"] = round(0.14 + 0.016 * strictness + (0.010 if not gate_arm else 0.0), 3)
        spec["known_coverage_margin"] = round(0.06 + 0.012 * strictness, 3)
        spec["known_coverage_min_affinity"] = round(0.16 + 0.018 * strictness, 3)
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if k_old == 10 else 4
        spec["multiproto_temperature"] = round(0.52 - 0.025 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.14 + 0.015 * strictness, 3)
        spec["soft_proto"] = round(0.14 + 0.014 * strictness, 3)
        spec["soft_proto_topk"] = spec["multiproto_topk"]
        spec["soft_proto_temperature"] = round(0.50 - 0.025 * strictness, 3)
        spec["soft_proto_boundary"] = round(0.06 + 0.010 * strictness, 3)
        spec["soft_proto_boundary_margin"] = round(0.07 + 0.012 * strictness, 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.46 + 0.020 * strictness, 3)
        spec["mixture_consistency_max_residual"] = round(0.46 - 0.018 * strictness, 3)
        spec["mixture_consistency_min_margin"] = round(0.020 + 0.010 * strictness, 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 3
        spec["old_primary_gate"] = False
        spec["old_primary_require_soft_mixture"] = False
        spec["old_primary_require_support_knn"] = False
        spec["old_primary_require_class_envelope"] = False
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (
            (-0.10, -0.06, -0.02)[strictness] if not gate_arm else (-0.04, 0.00, 0.04)[strictness]
        )
        spec["retention_rescue_old_min_anchor_delta"] = (
            (-0.14, -0.10, -0.06)[strictness] if not gate_arm else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["retention_rescue_old_min_anchor_margin"] = (
            (-0.08, -0.05, -0.02)[strictness] if not gate_arm else (-0.04, -0.01, 0.02)[strictness]
        )
        spec["retention_rescue_old_min_score_margin"] = (
            (-0.16, -0.12, -0.08)[strictness] if not gate_arm else (-0.10, -0.06, -0.02)[strictness]
        )
        spec["retention_rescue_max_background_score"] = (
            (0.92, 0.88, 0.84)[strictness] if not gate_arm else (0.80, 0.74, 0.68)[strictness]
        )
        spec["retention_rescue_max_background_margin"] = (
            (0.18, 0.12, 0.06)[strictness] if not gate_arm else (0.08, 0.02, -0.04)[strictness]
        )
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_alpha"] = round(0.14 - 0.015 * strictness, 3)
        spec["support_conformal_min_support"] = 3 if k_old == 5 else 5
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_max"] = round(0.56 - 0.025 * strictness, 3)
        spec["support_reconstruction_min_gap"] = round(0.020 + 0.008 * strictness, 3)
        spec["support_reconstruction_reject_action"] = "defer"
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = round(-0.18 + 0.030 * strictness, 3)
        spec["guard_min_old_surrogate_reject_delta"] = round(-0.08 + 0.025 * strictness, 3)
        spec["guard_min_energy_delta"] = round(-8.0 + 2.5 * strictness, 3)
        spec["guard_min_mahalanobis_delta"] = round(-22.0 + 4.0 * strictness, 3)
        spec["guard_min_accept_delta"] = round(-18.0 + 3.5 * strictness, 3)
        spec["guard_min_old_support_anchor_margin"] = round(0.000 + 0.010 * strictness, 3)
        spec["guard_min_best_old_score"] = round(-2.0 + 0.22 * strictness, 3)
        spec["guard_min_margin"] = round(0.14 + 0.040 * strictness + (0.040 if gate_arm else 0.0), 3)
        spec["guard_min_failures"] = 4 if not gate_arm else 3 + min(strictness, 1)
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.80
        spec["seen_new_acc_target"] = 0.75
        spec["stage2_priority_phase"] = "OLD80_FIRST"
        spec["old_acc_phase_gate"] = 0.80
        spec["secondary_objectives_after_old_gate"] = "NEXT_PHASE_SECONDARY_OBJECTIVES"
        spec["risk_note"] = (
            "OLDFUSE completed as a negative diagnostic with rollback on every row and no target hits. "
            "This route lowers hard reject pressure, enables candidate-only retention rescue, and keeps "
            "background/deployment gates as defer-first checks so old-retention signal can first reach "
            "the OLD80 gate before the next open-world objective phase."
        )
        spec["description"] = (
            "Stage2-B H06 OLD80_FIRST rollback-safe old-retention repair after OLDFUSE all-rollback; "
            "target-new support remains excluded, unknown transmitters remain query-only, and this "
            "stage first seeks old_acc>=0.80 before the next open-world objective phase."
        )
        spec["evidence_ref"] = (
            "oldfuse_negative_old_mean_0p025_unknown_far_0p000_rollback48_target_hit0;"
            "oldfuse_all_rollback_not_promotable;"
            "fresh_h06_rollback_safe_retention_repair;"
            "stage2b_old_unknown_only_target_new_excluded"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 4210 + idx * 23
    return specs


def _oa_mse_h06_oldhead48_stage_specs() -> list[dict]:
    """H06 OLD80 boundary repair from old-head sweep evidence."""

    specs = [dict(spec) for spec in _oa_mse_h06_rollsafe48_stage_specs()]
    arms = (
        ("oldhead_ridge_bridge", "ridge_k10"),
        ("oldhead_ridge_bridge", "ridge_k20"),
        ("oldhead_ridge_bridge", "ridge_k50_saturation"),
        ("oldhead_knn_density_guard", "knn_k10"),
        ("oldhead_knn_density_guard", "knn_k20"),
        ("oldhead_knn_density_guard", "knn_k50_saturation"),
    )
    for idx, spec in enumerate(specs):
        category, arm = arms[idx]
        strictness = idx % 3
        saturation = arm.endswith("saturation")
        k_old = (10, 20, 50)[strictness]
        ridge_bridge = category == "oldhead_ridge_bridge"
        spec["stage"] = "mse_subspace"
        spec["eval_protocol"] = "ftrc"
        spec["k_old"] = k_old
        spec["k_new"] = 0
        spec["category"] = category
        spec["optimization_category"] = category
        spec["route_suffix"] = f"h06_oldhead_boundary_repair_{arm}"
        spec["ablation_arm"] = arm
        spec["target_new_leo_support"] = False
        spec["source_proto_per_tx"] = 24 if saturation else 22
        spec["source_query_per_tx"] = 42 if saturation else 40
        spec["sfe_max_samples_per_tx"] = 88 if saturation else 68
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["stage2_max_active_per_gpu"] = 2
        spec["adapter_kind"] = "low_rank" if ridge_bridge else "residual_mlp"
        spec["adapter_selection_policy"] = (
            "support_cv_constrained" if ridge_bridge else "support_cv_risk_balanced"
        )
        spec["steps"] = (112, 124, 136)[strictness] if ridge_bridge else (104, 116, 128)[strictness]
        spec["align"] = round(0.125 + 0.010 * strictness + (0.008 if ridge_bridge else 0.0), 3)
        spec["norm"] = round(0.018 + 0.002 * strictness, 3)
        spec["bce"] = round(0.034 + 0.004 * strictness, 3)
        spec["unknown"] = round(0.052 + 0.010 * strictness + (0.020 if not ridge_bridge else 0.0), 3)
        spec["pl"] = round(0.016 + 0.003 * strictness, 3)
        spec["maha"] = round(0.24 + 0.025 * strictness, 3)
        spec["unknown_moat"] = round(0.20 + 0.035 * strictness + (0.045 if not ridge_bridge else 0.0), 3)
        spec["unknown_margin"] = round(0.24 + 0.050 * strictness + (0.065 if not ridge_bridge else 0.0), 3)
        spec["negative_anchor_weight"] = round(0.070 + 0.020 * strictness + (0.040 if not ridge_bridge else 0.0), 3)
        spec["negative_anchor_margin"] = round(0.18 + 0.030 * strictness + (0.045 if not ridge_bridge else 0.0), 3)
        spec["negative_anchor_temperature"] = round(0.14 - 0.012 * min(strictness, 2), 3)
        spec["negative_anchor_max_anchors"] = 288 if ridge_bridge else 320
        spec["void_background"] = round(0.020 + 0.008 * strictness + (0.030 if not ridge_bridge else 0.0), 3)
        spec["void_gate"] = bool(not ridge_bridge or strictness >= 2)
        spec["void_gate_min_score"] = round(0.64 + 0.030 * strictness + (0.050 if not ridge_bridge else 0.0), 3)
        spec["void_gate_min_margin"] = round(-0.06 + 0.025 * strictness + (0.040 if not ridge_bridge else 0.0), 3)
        spec["source_looo_unknown_weight"] = round(0.080 + 0.030 * strictness + (0.040 if not ridge_bridge else 0.0), 3)
        spec["source_looo_unknown_margin"] = round(0.30 + 0.050 * strictness + (0.060 if not ridge_bridge else 0.0), 3)
        spec["source_looo_interclass_margin"] = round(0.10 + 0.015 * strictness, 3)
        spec["source_looo_max_samples_per_class"] = 54 if ridge_bridge else 58
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_quantile"] = round(0.78 + 0.025 * strictness + (0.030 if not ridge_bridge else 0.0), 3)
        spec["source_looo_risk_slack"] = round(0.035 + 0.012 * strictness, 3)
        spec["source_looo_risk_min_score_margin"] = round(0.000 + 0.015 * strictness, 3)
        spec["source_looo_risk_min_known_evidence_delta"] = round(-0.22 + 0.035 * strictness, 3)
        spec["source_looo_risk_background_score"] = round(0.56 + 0.035 * strictness + (0.055 if not ridge_bridge else 0.0), 3)
        spec["source_looo_risk_background_margin"] = round(-0.12 + 0.035 * strictness + (0.050 if not ridge_bridge else 0.0), 3)
        spec["source_looo_risk_reject_min_failures"] = 4 if ridge_bridge else 3 + min(strictness, 1)
        spec["source_looo_risk_reject_action"] = "defer"
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = round(0.50 + 0.035 * strictness + (0.060 if not ridge_bridge else 0.0), 3)
        spec["two_branch_bg_min_margin"] = round(-0.16 + 0.035 * strictness + (0.055 if not ridge_bridge else 0.0), 3)
        spec["two_branch_old_support_evidence_delta"] = round(-0.20 + 0.030 * strictness, 3)
        spec["two_branch_old_anchor_delta"] = round(-0.16 + 0.025 * strictness, 3)
        spec["two_branch_old_anchor_margin"] = round(-0.03 + 0.015 * strictness, 3)
        spec["two_branch_seen_new_evidence_delta"] = 0.0
        spec["two_branch_seen_new_anchor_delta"] = 0.0
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_defer_action"] = "defer"
        spec["pre_reject_max_background_score"] = round(0.82 + 0.025 * strictness if ridge_bridge else 0.76 + 0.020 * strictness, 3)
        spec["pre_reject_max_background_margin"] = round(0.08 + 0.025 * strictness if ridge_bridge else 0.02 + 0.020 * strictness, 3)
        spec["pre_reject_defer_background_score"] = round(0.58 + 0.030 * strictness + (0.045 if not ridge_bridge else 0.0), 3)
        spec["pre_reject_defer_background_margin"] = round(-0.10 + 0.030 * strictness + (0.040 if not ridge_bridge else 0.0), 3)
        spec["pre_reject_reject_background_score"] = round(0.84 + 0.025 * strictness + (0.030 if not ridge_bridge else 0.0), 3)
        spec["pre_reject_reject_background_margin"] = round(0.06 + 0.030 * strictness + (0.030 if not ridge_bridge else 0.0), 3)
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_old_min_evidence_delta"] = (
            (-0.12, -0.08, -0.04)[strictness] if ridge_bridge else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["pre_reject_support_retention_old_min_anchor_delta"] = (
            (-0.16, -0.12, -0.08)[strictness] if ridge_bridge else (-0.12, -0.08, -0.04)[strictness]
        )
        spec["pre_reject_support_retention_old_min_anchor_margin"] = (
            (-0.08, -0.05, -0.02)[strictness] if ridge_bridge else (-0.04, -0.02, 0.00)[strictness]
        )
        spec["pre_reject_support_retention_old_min_score_margin"] = (
            (-0.18, -0.14, -0.10)[strictness] if ridge_bridge else (-0.12, -0.08, -0.04)[strictness]
        )
        spec["pre_reject_support_retention_max_background_score"] = (
            (0.94, 0.90, 0.86)[strictness] if ridge_bridge else (0.84, 0.78, 0.72)[strictness]
        )
        spec["pre_reject_support_retention_max_background_margin"] = (
            (0.20, 0.14, 0.08)[strictness] if ridge_bridge else (0.10, 0.04, -0.02)[strictness]
        )
        spec["pre_reject_support_retention_require_source_looo_pass"] = bool(not ridge_bridge and strictness >= 1)
        spec["pre_reject_support_retention_source_looo_max_failures"] = 3 if ridge_bridge else 2
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = round(0.025 + 0.010 * strictness, 3)
        spec["support_retention_guard_slack"] = round(0.14 + 0.020 * strictness, 3)
        spec["support_center_ce"] = round(0.16 + 0.018 * strictness + (0.010 if ridge_bridge else 0.0), 3)
        spec["support_center_temperature"] = round(0.52 - 0.035 * strictness, 3)
        spec["support_center_margin"] = round(0.09 + 0.015 * strictness, 3)
        spec["support_contrast"] = round(0.035 + 0.005 * strictness, 3)
        spec["known_coverage_weight"] = round(0.15 + 0.016 * strictness + (0.010 if ridge_bridge else 0.0), 3)
        spec["known_coverage_margin"] = round(0.07 + 0.012 * strictness, 3)
        spec["known_coverage_min_affinity"] = round(0.18 + 0.018 * strictness, 3)
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 6 if saturation else 5
        spec["multiproto_temperature"] = round(0.50 - 0.025 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.15 + 0.015 * strictness, 3)
        spec["soft_proto"] = round(0.15 + 0.014 * strictness, 3)
        spec["soft_proto_topk"] = spec["multiproto_topk"]
        spec["soft_proto_temperature"] = round(0.48 - 0.025 * strictness, 3)
        spec["soft_proto_boundary"] = round(0.07 + 0.010 * strictness, 3)
        spec["soft_proto_boundary_margin"] = round(0.08 + 0.012 * strictness, 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.48 + 0.020 * strictness, 3)
        spec["mixture_consistency_max_residual"] = round(0.44 - 0.018 * strictness, 3)
        spec["mixture_consistency_min_margin"] = round(0.025 + 0.010 * strictness, 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["density_shell_gate"] = True
        spec["density_shell_old_min_evidence_delta"] = (
            (-0.12, -0.08, -0.04)[strictness] if ridge_bridge else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["density_shell_old_min_anchor_delta"] = (
            (-0.16, -0.12, -0.08)[strictness] if ridge_bridge else (-0.12, -0.08, -0.04)[strictness]
        )
        spec["density_shell_old_min_density_delta"] = (
            (-0.12, -0.08, -0.04)[strictness] if ridge_bridge else (-0.06, -0.02, 0.02)[strictness]
        )
        spec["density_shell_accept_background_margin"] = (
            (0.26, 0.22, 0.18)[strictness] if ridge_bridge else (0.20, 0.14, 0.08)[strictness]
        )
        spec["density_shell_reject_background_score"] = (
            (0.90, 0.88, 0.86)[strictness] if ridge_bridge else (0.84, 0.80, 0.76)[strictness]
        )
        spec["density_shell_reject_background_margin"] = (
            (0.18, 0.14, 0.10)[strictness] if ridge_bridge else (0.12, 0.08, 0.04)[strictness]
        )
        spec["density_shell_reject_min_failed_shells"] = 3 if ridge_bridge else 2
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain" if ridge_bridge else "reject"
        spec["anchor_density_topk"] = 5 if saturation else 4
        spec["anchor_density_temperature"] = round(0.12 - 0.020 * strictness, 3)
        spec["anchor_density_quantile"] = round(0.020 + 0.020 * strictness + (0.020 if not ridge_bridge else 0.0), 3)
        spec["anchor_density_margin_quantile"] = round(0.020 + 0.015 * strictness + (0.020 if not ridge_bridge else 0.0), 3)
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 3
        spec["old_primary_gate"] = False
        spec["old_primary_require_soft_mixture"] = False
        spec["old_primary_require_support_knn"] = False
        spec["old_primary_require_class_envelope"] = False
        spec["old_primary_promote_rescue_candidates"] = False
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (
            (-0.08, -0.04, 0.00)[strictness] if ridge_bridge else (-0.04, 0.00, 0.04)[strictness]
        )
        spec["retention_rescue_old_min_anchor_delta"] = (
            (-0.12, -0.08, -0.04)[strictness] if ridge_bridge else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["retention_rescue_old_min_anchor_margin"] = (
            (-0.06, -0.03, 0.00)[strictness] if ridge_bridge else (-0.03, 0.00, 0.03)[strictness]
        )
        spec["retention_rescue_old_min_score_margin"] = (
            (-0.14, -0.10, -0.06)[strictness] if ridge_bridge else (-0.08, -0.04, 0.00)[strictness]
        )
        spec["retention_rescue_max_background_score"] = (
            (0.88, 0.84, 0.80)[strictness] if ridge_bridge else (0.76, 0.70, 0.64)[strictness]
        )
        spec["retention_rescue_max_background_margin"] = (
            (0.14, 0.08, 0.02)[strictness] if ridge_bridge else (0.04, -0.02, -0.08)[strictness]
        )
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_calibration_quantile"] = round(0.12 - 0.015 * strictness, 3)
        spec["support_conformal_conformity_slack"] = round(0.14 + 0.015 * strictness, 3)
        spec["support_conformal_anchor_margin_slack"] = round(0.06 + 0.012 * strictness, 3)
        spec["support_conformal_background_score"] = round(0.82 + 0.025 * strictness if ridge_bridge else 0.74 + 0.025 * strictness, 3)
        spec["support_conformal_background_margin"] = round(0.08 + 0.020 * strictness if ridge_bridge else 0.02 + 0.020 * strictness, 3)
        spec["support_conformal_hard_reject_margin"] = round(0.20 + 0.030 * strictness if ridge_bridge else 0.12 + 0.025 * strictness, 3)
        spec["support_conformal_reject_min_failures"] = 3 if ridge_bridge else 2
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_rank"] = 3 if saturation else 2
        spec["support_reconstruction_residual_quantile"] = round(0.94 - 0.015 * strictness, 3)
        spec["support_reconstruction_residual_slack"] = round(0.045 + 0.012 * strictness, 3)
        spec["support_reconstruction_min_residual_floor"] = round(0.030 + 0.010 * strictness, 3)
        spec["support_reconstruction_negative_scale"] = round(0.54 + 0.040 * strictness, 3)
        spec["support_reconstruction_negative_margin"] = round(-0.04 + 0.020 * strictness, 3)
        spec["support_reconstruction_hard_residual_margin"] = round(0.08 + 0.020 * strictness, 3)
        spec["support_reconstruction_background_score"] = round(0.82 + 0.030 * strictness if ridge_bridge else 0.74 + 0.030 * strictness, 3)
        spec["support_reconstruction_background_margin"] = round(0.08 + 0.025 * strictness if ridge_bridge else 0.02 + 0.025 * strictness, 3)
        spec["support_reconstruction_reject_min_failures"] = 3 if ridge_bridge else 2
        spec["support_reconstruction_reject_action"] = "defer"
        spec["three_way_decision_head"] = True
        spec["three_way_head_weight"] = round(0.040 + 0.010 * strictness + (0.012 if not ridge_bridge else 0.0), 3)
        spec["three_way_head_temperature"] = round(0.13 - 0.014 * strictness, 3)
        spec["three_way_head_known_margin"] = round(0.08 + 0.012 * strictness, 3)
        spec["three_way_head_background_margin"] = round(0.07 + 0.018 * strictness + (0.020 if not ridge_bridge else 0.0), 3)
        spec["three_way_decision_policy"] = "class_first" if ridge_bridge else "evidence_balanced"
        spec["three_way_accept_prob"] = round(0.48 + 0.025 * strictness, 3)
        spec["three_way_reject_prob"] = round(0.60 + 0.035 * strictness + (0.035 if not ridge_bridge else 0.0), 3)
        spec["three_way_defer_prob"] = round(0.44 + 0.025 * strictness, 3)
        spec["three_way_known_background_margin"] = round(0.00 + 0.025 * strictness, 3)
        spec["three_way_reject_margin"] = round(0.04 + 0.025 * strictness + (0.030 if not ridge_bridge else 0.0), 3)
        spec["three_way_defer_action"] = "defer"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if ridge_bridge else "defer"
        spec["three_way_known_floor_old_min_evidence_delta"] = (
            (0.00, -0.04, -0.08)[strictness] if ridge_bridge else (-0.04, -0.08, -0.12)[strictness]
        )
        spec["three_way_known_floor_old_min_anchor_delta"] = (
            (-0.04, -0.08, -0.12)[strictness] if ridge_bridge else (-0.08, -0.12, -0.16)[strictness]
        )
        spec["three_way_known_floor_old_min_anchor_margin"] = (
            (0.00, -0.03, -0.06)[strictness] if ridge_bridge else (-0.04, -0.08, -0.12)[strictness]
        )
        spec["three_way_known_floor_old_min_score_margin"] = (
            (-0.06, -0.10, -0.14)[strictness] if ridge_bridge else (-0.12, -0.18, -0.24)[strictness]
        )
        spec["three_way_known_floor_background_override_prob"] = (
            (0.998, 0.996, 0.994)[strictness] if ridge_bridge else (0.996, 0.992, 0.988)[strictness]
        )
        spec["three_way_known_floor_background_override_margin"] = (
            (1.20, 0.90, 0.65)[strictness] if ridge_bridge else (0.90, 0.65, 0.45)[strictness]
        )
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = (
            (-0.18, -0.14, -0.10)[strictness] if ridge_bridge else (-0.12, -0.08, -0.04)[strictness]
        )
        spec["guard_min_old_surrogate_reject_delta"] = (
            (-0.08, -0.04, 0.00)[strictness] if ridge_bridge else (-0.04, 0.00, 0.04)[strictness]
        )
        spec["guard_min_energy_delta"] = round(-8.5 + 2.5 * strictness + (2.0 if not ridge_bridge else 0.0), 3)
        spec["guard_min_mahalanobis_delta"] = round(-24.0 + 4.5 * strictness + (4.0 if not ridge_bridge else 0.0), 3)
        spec["guard_min_accept_delta"] = round(-19.0 + 3.5 * strictness + (4.0 if not ridge_bridge else 0.0), 3)
        spec["guard_min_old_support_anchor_margin"] = round(0.000 + 0.010 * strictness + (0.010 if not ridge_bridge else 0.0), 3)
        spec["guard_min_best_old_score"] = round(-2.1 + 0.24 * strictness + (0.20 if not ridge_bridge else 0.0), 3)
        spec["guard_min_margin"] = round(0.12 + 0.045 * strictness + (0.055 if not ridge_bridge else 0.0), 3)
        spec["guard_min_failures"] = 4 if ridge_bridge else 3 + min(strictness, 1)
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.80
        spec["seen_new_acc_target"] = 0.75
        spec["stage2_priority_phase"] = "OLD80_FIRST"
        spec["old_acc_phase_gate"] = 0.80
        spec["secondary_objectives_after_old_gate"] = "NEXT_PHASE_SECONDARY_OBJECTIVES"
        spec["risk_note"] = (
            "Old-head sweep showed ridge can preserve target-old recoverability near OLD80 at K50 but leaks "
            "unknowns, while KNN lowers FAR with weak old correct accept. This route combines old-head proof "
            "with query-free density, conformal, reconstruction, and source-risk boundary checks."
        )
        spec["description"] = (
            "Stage2-B H06 OLD80_FIRST old-head boundary repair beyond support threshold sweep; target-new "
            "support/query remain excluded, unknown query is eval-only, and K50 rows are higher-shot saturation "
            "diagnostics rather than strict few-shot claims."
        )
        spec["evidence_ref"] = (
            "oldhead_sweep_ridge_k50_old_full_mean_0p7374_max_0p8333_far_high;"
            "oldhead_sweep_knn_k50_far_lower_old_correct_accept_unstable;"
            "fresh_h06_oldhead_boundary_representation_repair;"
            "stage2b_old_unknown_only_target_new_excluded_unknown_query_eval_only"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 4670 + idx * 29
    return specs


def _oa_mse_h06_oldheadfar48_stage_specs() -> list[dict]:
    """H06 support-CV stability repair from OLDHEAD samplecap negative evidence."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldhead48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        k_old = int(spec["k_old"])
        saturation = k_old >= 50
        far_stability = idx >= (len(specs) // 2)
        spec["category"] = "oldhead_far_stability" if far_stability else "oldhead_support_cv_stability"
        spec["route_suffix"] = "h06_oldheadfar_support_cv_stability_repair_after_oldhead_negative"
        spec["adapter_kind"] = "residual_mlp" if far_stability else "low_rank"
        spec["adapter_selection_policy"] = "identity_preserving_risk_cv" if far_stability else "identity_preserving_cv"
        spec["steps"] = 34 + 4 * strictness + (6 if far_stability else 0) + (4 if saturation else 0)
        spec["unknown_threshold"] = round(0.988 + 0.003 * strictness + (0.004 if far_stability else 0.0), 3)
        spec["openmax_min_threshold"] = round(0.18 + 0.025 * strictness + (0.03 if far_stability else 0.0), 3)
        spec["source_ce"] = round(0.18 + 0.025 * strictness + (0.030 if far_stability else 0.0), 3)
        spec["unknown_moat"] = round(0.07 + 0.018 * strictness + (0.060 if far_stability else 0.0), 3)
        spec["unknown_margin"] = round(0.30 + 0.035 * strictness + (0.090 if far_stability else 0.0), 3)
        spec["boundary_samples"] = 2 + strictness + (2 if far_stability else 0)
        spec["boundary_offset"] = round(0.13 + 0.020 * strictness + (0.050 if far_stability else 0.0), 3)
        spec["source_boundary_samples"] = 1 + strictness + (1 if far_stability else 0)
        spec["source_boundary_offset"] = round(0.12 + 0.020 * strictness + (0.035 if far_stability else 0.0), 3)
        spec["target_shift_samples"] = 1 + strictness
        spec["target_shift_offset"] = round(0.14 + 0.018 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["target_halo_samples"] = 1 + strictness
        spec["target_halo_offset"] = round(0.22 + 0.025 * strictness + (0.040 if far_stability else 0.0), 3)
        spec["target_ring_samples"] = 1 + strictness + (1 if far_stability else 0)
        spec["target_ring_offset"] = round(0.30 + 0.035 * strictness + (0.070 if far_stability else 0.0), 3)
        spec["support_contrast"] = round(0.11 + 0.018 * strictness + (0.015 if far_stability else 0.0), 3)
        spec["old_bridge"] = round(0.12 + 0.020 * strictness, 3)
        spec["old_neighborhood"] = round(0.13 + 0.020 * strictness + (0.012 if saturation else 0.0), 3)
        spec["support_center_ce"] = round(0.10 + 0.020 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["support_center_margin_weight"] = round(0.08 + 0.015 * strictness + (0.015 if far_stability else 0.0), 3)
        spec["support_center_margin"] = round(0.08 + 0.020 * strictness + (0.030 if far_stability else 0.0), 3)
        spec["known_coverage_weight"] = round(0.015 + 0.008 * strictness, 3)
        spec["known_coverage_min_margin"] = round(-0.08 + 0.020 * strictness, 3)
        spec["known_coverage_background_margin"] = round(0.30 + 0.040 * strictness + (0.080 if far_stability else 0.0), 3)
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 5 if saturation else 4
        spec["multiproto_temperature"] = round(0.11 - 0.012 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.12 + 0.020 * strictness + (0.010 if far_stability else 0.0), 3)
        spec["soft_proto"] = round(0.12 + 0.020 * strictness, 3)
        spec["soft_proto_topk"] = spec["multiproto_topk"]
        spec["soft_proto_temperature"] = round(0.44 - 0.020 * strictness, 3)
        spec["soft_proto_boundary"] = round(0.04 + 0.010 * strictness, 3)
        spec["soft_proto_boundary_margin"] = round(0.06 + 0.012 * strictness + (0.010 if far_stability else 0.0), 3)
        spec["mixture_consistency_gate"] = True
        spec["mixture_consistency_min_cos"] = round(0.42 + 0.020 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["mixture_consistency_max_residual"] = round(0.48 - 0.018 * strictness - (0.030 if far_stability else 0.0), 3)
        spec["mixture_consistency_min_margin"] = round(0.015 + 0.010 * strictness + (0.010 if far_stability else 0.0), 3)
        spec["mixture_consistency_action"] = "uncertain"
        spec["density_shell_gate"] = True
        spec["density_shell_old_min_evidence_delta"] = (-0.18, -0.14, -0.10)[strictness] if not far_stability else (-0.10, -0.06, -0.02)[strictness]
        spec["density_shell_old_min_anchor_delta"] = (-0.20, -0.16, -0.12)[strictness] if not far_stability else (-0.14, -0.10, -0.06)[strictness]
        spec["density_shell_old_min_density_delta"] = (-0.18, -0.14, -0.10)[strictness] if not far_stability else (-0.10, -0.06, -0.02)[strictness]
        spec["density_shell_accept_background_margin"] = (0.32, 0.28, 0.24)[strictness] if not far_stability else (0.22, 0.16, 0.10)[strictness]
        spec["density_shell_reject_background_score"] = (0.94, 0.92, 0.90)[strictness] if not far_stability else (0.86, 0.82, 0.78)[strictness]
        spec["density_shell_reject_background_margin"] = (0.22, 0.18, 0.14)[strictness] if not far_stability else (0.16, 0.10, 0.04)[strictness]
        spec["density_shell_reject_min_failed_shells"] = 3 if not far_stability else 2
        spec["anchor_density_gate"] = True
        spec["anchor_density_action"] = "uncertain"
        spec["anchor_density_topk"] = 5 if saturation else 4
        spec["anchor_density_temperature"] = round(0.12 - 0.020 * strictness, 3)
        spec["anchor_density_quantile"] = round(0.015 + 0.015 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["anchor_density_margin_quantile"] = round(0.015 + 0.015 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["class_envelope_gate"] = True
        spec["class_envelope_action"] = "uncertain"
        spec["class_envelope_min_failures"] = 3 if not far_stability else 2
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["retention_rescue_old_min_evidence_delta"] = (-0.16, -0.12, -0.08)[strictness] if not far_stability else (-0.10, -0.06, -0.02)[strictness]
        spec["retention_rescue_old_min_anchor_delta"] = (-0.18, -0.14, -0.10)[strictness] if not far_stability else (-0.12, -0.08, -0.04)[strictness]
        spec["retention_rescue_old_min_anchor_margin"] = (-0.10, -0.07, -0.04)[strictness] if not far_stability else (-0.06, -0.03, 0.00)[strictness]
        spec["retention_rescue_old_min_score_margin"] = (-0.18, -0.14, -0.10)[strictness] if not far_stability else (-0.10, -0.06, -0.02)[strictness]
        spec["retention_rescue_max_background_score"] = (0.90, 0.86, 0.82)[strictness] if not far_stability else (0.78, 0.72, 0.66)[strictness]
        spec["retention_rescue_max_background_margin"] = (0.18, 0.12, 0.06)[strictness] if not far_stability else (0.06, 0.00, -0.06)[strictness]
        spec["identity_consensus_arbitration"] = True
        spec["identity_consensus_support_background_cap"] = True
        spec["identity_consensus_support_background_cap_quantile"] = round(0.82 - 0.020 * strictness - (0.030 if far_stability else 0.0), 3)
        spec["identity_consensus_support_background_cap_slack"] = round(0.10 + 0.020 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["identity_consensus_support_background_cap_min_anchors"] = 3 if saturation else 2
        spec["identity_consensus_background_accept_margin"] = (0.24, 0.20, 0.16)[strictness] if not far_stability else (0.16, 0.10, 0.04)[strictness]
        spec["identity_consensus_reject_background_score"] = (0.94, 0.92, 0.90)[strictness] if not far_stability else (0.88, 0.84, 0.80)[strictness]
        spec["identity_consensus_reject_background_margin"] = (0.24, 0.20, 0.16)[strictness] if not far_stability else (0.16, 0.10, 0.04)[strictness]
        spec["identity_consensus_reject_min_failures"] = 3 if not far_stability else 2
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_calibration_quantile"] = round(0.08 - 0.010 * strictness, 3)
        spec["support_conformal_conformity_slack"] = round(0.18 + 0.020 * strictness + (0.010 if far_stability else 0.0), 3)
        spec["support_conformal_anchor_margin_slack"] = round(0.08 + 0.014 * strictness, 3)
        spec["support_conformal_background_score"] = round(0.86 + 0.020 * strictness if not far_stability else 0.78 + 0.025 * strictness, 3)
        spec["support_conformal_background_margin"] = round(0.12 + 0.020 * strictness if not far_stability else 0.04 + 0.020 * strictness, 3)
        spec["support_conformal_hard_reject_margin"] = round(0.24 + 0.030 * strictness if not far_stability else 0.16 + 0.025 * strictness, 3)
        spec["support_conformal_reject_min_failures"] = 3 if not far_stability else 2
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_rank"] = 3 if saturation else 2
        spec["support_reconstruction_residual_quantile"] = round(0.96 - 0.010 * strictness, 3)
        spec["support_reconstruction_residual_slack"] = round(0.055 + 0.012 * strictness, 3)
        spec["support_reconstruction_min_residual_floor"] = round(0.030 + 0.010 * strictness, 3)
        spec["support_reconstruction_negative_scale"] = round(0.48 + 0.035 * strictness + (0.050 if far_stability else 0.0), 3)
        spec["support_reconstruction_negative_margin"] = round(-0.08 + 0.020 * strictness + (0.030 if far_stability else 0.0), 3)
        spec["support_reconstruction_hard_residual_margin"] = round(0.10 + 0.020 * strictness, 3)
        spec["support_reconstruction_background_score"] = round(0.86 + 0.025 * strictness if not far_stability else 0.78 + 0.030 * strictness, 3)
        spec["support_reconstruction_background_margin"] = round(0.12 + 0.020 * strictness if not far_stability else 0.04 + 0.025 * strictness, 3)
        spec["support_reconstruction_reject_min_failures"] = 3 if not far_stability else 2
        spec["support_reconstruction_reject_action"] = "defer"
        spec["three_way_decision_head"] = True
        spec["three_way_head_weight"] = round(0.035 + 0.010 * strictness + (0.015 if far_stability else 0.0), 3)
        spec["three_way_head_temperature"] = round(0.14 - 0.012 * strictness, 3)
        spec["three_way_head_known_margin"] = round(0.06 + 0.012 * strictness, 3)
        spec["three_way_head_background_margin"] = round(0.06 + 0.018 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["three_way_decision_policy"] = "class_first" if not far_stability else "evidence_balanced"
        spec["three_way_accept_prob"] = round(0.42 + 0.025 * strictness if not far_stability else 0.34 + 0.025 * strictness, 3)
        spec["three_way_reject_prob"] = round(0.62 + 0.035 * strictness if not far_stability else 0.70 + 0.040 * strictness, 3)
        spec["three_way_defer_prob"] = round(0.46 + 0.025 * strictness, 3)
        spec["three_way_known_background_margin"] = round(0.00 + 0.020 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["three_way_reject_margin"] = round(0.04 + 0.025 * strictness + (0.030 if far_stability else 0.0), 3)
        spec["three_way_defer_action"] = "defer"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if not far_stability else "defer"
        spec["three_way_known_floor_old_min_evidence_delta"] = (-0.12, -0.16, -0.20)[strictness] if not far_stability else (-0.06, -0.10, -0.14)[strictness]
        spec["three_way_known_floor_old_min_anchor_delta"] = (-0.14, -0.18, -0.22)[strictness] if not far_stability else (-0.10, -0.14, -0.18)[strictness]
        spec["three_way_known_floor_old_min_anchor_margin"] = (-0.08, -0.11, -0.14)[strictness] if not far_stability else (-0.04, -0.07, -0.10)[strictness]
        spec["three_way_known_floor_old_min_score_margin"] = (-0.18, -0.22, -0.26)[strictness] if not far_stability else (-0.10, -0.14, -0.18)[strictness]
        spec["three_way_known_floor_background_override_prob"] = (0.999, 0.998, 0.997)[strictness] if not far_stability else (0.996, 0.993, 0.990)[strictness]
        spec["three_way_known_floor_background_override_margin"] = (1.30, 1.00, 0.75)[strictness] if not far_stability else (0.85, 0.60, 0.40)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["pre_reject_support_retention_require_source_looo_pass"] = far_stability and strictness >= 1
        spec["pre_reject_support_retention_source_looo_max_failures"] = 1 if far_stability else 2
        spec["pre_reject_support_retention_max_background_score"] = (0.96, 0.94, 0.92)[strictness] if not far_stability else (0.88, 0.84, 0.80)[strictness]
        spec["pre_reject_support_retention_max_background_margin"] = (0.36, 0.30, 0.24)[strictness] if not far_stability else (0.18, 0.12, 0.06)[strictness]
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_reject_min_failures"] = 4 if not far_stability else 3
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.76, 0.80, 0.84)[strictness] if not far_stability else (0.66, 0.70, 0.74)[strictness]
        spec["two_branch_bg_min_margin"] = (0.06, 0.10, 0.14)[strictness] if not far_stability else (-0.02, 0.02, 0.06)[strictness]
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = (-0.22, -0.18, -0.14)[strictness] if not far_stability else (-0.12, -0.08, -0.04)[strictness]
        spec["guard_min_old_surrogate_reject_delta"] = (-0.12, -0.08, -0.04)[strictness] if not far_stability else (-0.04, 0.00, 0.04)[strictness]
        spec["guard_min_energy_delta"] = round(-10.0 + 2.5 * strictness + (3.0 if far_stability else 0.0), 3)
        spec["guard_min_mahalanobis_delta"] = round(-30.0 + 5.0 * strictness + (6.0 if far_stability else 0.0), 3)
        spec["guard_min_accept_delta"] = round(-24.0 + 4.0 * strictness + (5.0 if far_stability else 0.0), 3)
        spec["guard_min_old_support_anchor_margin"] = round(-0.010 + 0.010 * strictness + (0.020 if far_stability else 0.0), 3)
        spec["guard_min_best_old_score"] = round(-2.4 + 0.25 * strictness + (0.30 if far_stability else 0.0), 3)
        spec["guard_min_margin"] = round(0.08 + 0.040 * strictness + (0.050 if far_stability else 0.0), 3)
        spec["guard_min_failures"] = 5 if not far_stability else 3 + min(strictness, 1)
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.80
        spec["seen_new_acc_target"] = 0.75
        spec["stage2_priority_phase"] = "OLD80_FIRST"
        spec["old_acc_phase_gate"] = 0.80
        spec["secondary_objectives_after_old_gate"] = "UNKNOWN_FAR_AFTER_OLD80"
        spec["risk_note"] = (
            "OLDHEAD samplecap completed 48/48 but reached old80_count=0. This fresh route keeps "
            "target-old support-only calibration and changes the head selector to support-CV stability proof "
            "plus FAR-aware background caps, without using unknown query for fitting."
        )
        spec["description"] = (
            "Stage2-B H06 OLDHEADFAR support-CV stability repair: preserve old-head recoverability through "
            "target-old leave-one-out/stability proof first, then apply support-only background caps for unknown FAR. "
            "Target-new remains excluded and unknown query is eval-only."
        )
        spec["evidence_ref"] = (
            "oldhead_samplecap_complete_48of48_old80_count_0_old_max_0p2389_far_tradeoff;"
            "target_only_old_signal_exists_ridge_k50_old_max_0p8333_not_deployment_evidence;"
            "fresh_support_cv_stability_head_unknown_far_repair;"
            "stage2b_old_unknown_only_target_new_excluded_unknown_query_eval_only"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 5190 + idx * 31
    return specs


def _oa_mse_h06_oldrecov48_stage_specs() -> list[dict]:
    """H06 target-old recoverability repair after OLDHEADFAR negative diagnosis."""

    specs = [dict(spec) for spec in _oa_mse_h06_oldhead48_stage_specs()]
    arms = (
        ("oldrecov_ridge_head", "ridge_head_k10"),
        ("oldrecov_ridge_head", "ridge_head_k20"),
        ("oldrecov_ridge_head", "ridge_head_k50_saturation"),
        ("oldrecov_proto_bridge", "proto_bridge_k10"),
        ("oldrecov_proto_bridge", "proto_bridge_k20"),
        ("oldrecov_proto_bridge", "proto_bridge_k50_saturation"),
    )
    for idx, spec in enumerate(specs):
        category, arm = arms[idx]
        strictness = idx % 3
        saturation = arm.endswith("saturation")
        ridge_head = category == "oldrecov_ridge_head"
        spec["category"] = category
        spec["optimization_category"] = category
        spec["route_suffix"] = f"h06_oldrecov_target_old_recoverability_{arm}"
        spec["ablation_arm"] = arm
        spec["adapter_kind"] = "low_rank" if ridge_head else "residual_mlp"
        spec["adapter_selection_policy"] = "support_cv_constrained" if ridge_head else "identity_preserving"
        spec["steps"] = (118, 130, 142)[strictness] if ridge_head else (102, 114, 126)[strictness]
        spec["source_proto_per_tx"] = 26 if saturation and ridge_head else 24
        spec["source_query_per_tx"] = 44 if saturation else 40
        spec["sfe_max_samples_per_tx"] = 92 if saturation else 72
        spec["unknown_moat"] = round(0.045 + 0.014 * strictness + (0.018 if not ridge_head else 0.0), 3)
        spec["unknown_margin"] = round(0.24 + 0.030 * strictness + (0.045 if not ridge_head else 0.0), 3)
        spec["negative_anchor_weight"] = round(0.030 + 0.012 * strictness + (0.020 if not ridge_head else 0.0), 3)
        spec["negative_anchor_margin"] = round(0.12 + 0.020 * strictness + (0.030 if not ridge_head else 0.0), 3)
        spec["void_background"] = round(0.010 + 0.006 * strictness + (0.016 if not ridge_head else 0.0), 3)
        spec["void_gate"] = False
        spec["known_coverage_weight"] = round(0.22 + 0.024 * strictness + (0.018 if ridge_head else 0.0), 3)
        spec["known_coverage_margin"] = round(0.08 + 0.014 * strictness, 3)
        spec["known_coverage_min_affinity"] = round(0.20 + 0.018 * strictness, 3)
        spec["support_center_ce"] = round(0.20 + 0.020 * strictness + (0.020 if ridge_head else 0.0), 3)
        spec["support_center_temperature"] = round(0.54 - 0.035 * strictness, 3)
        spec["support_center_margin"] = round(0.10 + 0.014 * strictness, 3)
        spec["support_contrast"] = round(0.060 + 0.010 * strictness + (0.015 if ridge_head else 0.0), 3)
        spec["old_bridge"] = round(0.18 + 0.022 * strictness + (0.025 if ridge_head else 0.0), 3)
        spec["old_neighborhood"] = round(0.16 + 0.020 * strictness + (0.012 if saturation else 0.0), 3)
        spec["multiproto_score"] = True
        spec["multiproto_topk"] = 6 if saturation else 5
        spec["multiproto_temperature"] = round(0.52 - 0.025 * strictness, 3)
        spec["multiproto_score_weight"] = round(0.18 + 0.016 * strictness + (0.018 if not ridge_head else 0.0), 3)
        spec["soft_proto"] = round(0.18 + 0.016 * strictness + (0.018 if not ridge_head else 0.0), 3)
        spec["soft_proto_topk"] = spec["multiproto_topk"]
        spec["soft_proto_temperature"] = round(0.50 - 0.025 * strictness, 3)
        spec["soft_proto_boundary"] = round(0.08 + 0.010 * strictness, 3)
        spec["soft_proto_boundary_margin"] = round(0.08 + 0.012 * strictness, 3)
        spec["density_shell_reject_min_failed_shells"] = 4 if ridge_head else 3
        spec["anchor_density_action"] = "uncertain"
        spec["class_envelope_action"] = "uncertain"
        spec["retention_rescue_gate"] = True
        spec["retention_rescue_candidate_only"] = True
        spec["support_conformal_arbitration"] = True
        spec["support_conformal_reject_action"] = "defer"
        spec["support_reconstruction_arbitration"] = True
        spec["support_reconstruction_reject_action"] = "defer"
        spec["three_way_decision_head"] = True
        spec["three_way_decision_policy"] = "class_first" if ridge_head else "evidence_balanced"
        spec["three_way_known_floor"] = True
        spec["three_way_known_floor_action"] = "accept" if ridge_head else "defer"
        spec["pre_reject_defer_arbitration"] = True
        spec["pre_reject_support_neighborhood_retention"] = True
        spec["source_looo_risk_arbitration"] = True
        spec["source_looo_risk_reject_action"] = "defer"
        spec["source_looo_reject_min_failures"] = 4 if ridge_head else 3
        spec["two_branch_background_guard"] = True
        spec["two_branch_bg_min_score"] = (0.82, 0.84, 0.86)[strictness] if ridge_head else (0.76, 0.79, 0.82)[strictness]
        spec["two_branch_bg_min_margin"] = (0.08, 0.11, 0.14)[strictness] if ridge_head else (0.02, 0.05, 0.08)[strictness]
        spec["old_unknown_acceptance_guard"] = True
        spec["guard_min_old_support_evidence_delta"] = (-0.24, -0.20, -0.16)[strictness] if ridge_head else (-0.18, -0.14, -0.10)[strictness]
        spec["guard_min_old_surrogate_reject_delta"] = (-0.14, -0.10, -0.06)[strictness] if ridge_head else (-0.10, -0.06, -0.02)[strictness]
        spec["guard_min_energy_delta"] = round(-12.0 + 2.5 * strictness + (2.0 if not ridge_head else 0.0), 3)
        spec["guard_min_mahalanobis_delta"] = round(-34.0 + 5.0 * strictness + (4.0 if not ridge_head else 0.0), 3)
        spec["guard_min_accept_delta"] = round(-26.0 + 4.0 * strictness + (3.0 if not ridge_head else 0.0), 3)
        spec["guard_min_old_support_anchor_margin"] = round(-0.020 + 0.010 * strictness + (0.010 if not ridge_head else 0.0), 3)
        spec["guard_min_best_old_score"] = round(-2.7 + 0.26 * strictness + (0.18 if not ridge_head else 0.0), 3)
        spec["guard_min_margin"] = round(0.06 + 0.035 * strictness + (0.035 if not ridge_head else 0.0), 3)
        spec["guard_min_failures"] = 5 if ridge_head else 4
        spec["seen_new_registration_override"] = False
        spec["old_acc_target"] = 0.80
        spec["seen_new_acc_target"] = 0.75
        spec["stage2_priority_phase"] = "OLD80_FIRST"
        spec["old_acc_phase_gate"] = 0.80
        spec["secondary_objectives_after_old_gate"] = "UNKNOWN_FAR_AFTER_OLD80"
        spec["risk_note"] = (
            "OLDHEADFAR completed as a negative diagnostic: unknown FAR improved only with collapsed target-old "
            "accuracy and coverage. This route restores target-old support/query recoverability first, then treats "
            "unknown FAR as a post-OLD80 secondary objective."
        )
        spec["description"] = (
            "Stage2-B H06 OLDRECOV target-old recoverability repair after OLDHEADFAR: target-new support remains "
            "excluded, unknown query remains eval-only, and K50 rows are higher-shot saturation diagnostics."
        )
        spec["evidence_ref"] = (
            "oldheadfar_negative_old_mean_0p0163_old_max_0p1111_old80_count0;"
            "target_old_only_ridge_k50_old_max_0p8333_upper_bound_not_deployment;"
            "fresh_h06_target_old_recoverability_first_repair;"
            "stage2b_old_unknown_only_target_new_excluded_unknown_query_eval_only"
        )
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 6210 + idx * 37
    return specs


def _oa_mse_neganchor48_stage_specs() -> list[dict]:
    """48-row negative-anchor background-basin route after next48ew."""

    specs = [dict(spec) for spec in _oa_mse_softtarget48_stage_specs()]
    for idx, spec in enumerate(specs):
        strictness = idx % 3
        aggressive = idx >= (len(specs) // 2)
        spec["category"] = "aggressive" if aggressive else "conservative"
        spec["route_suffix"] = "negative_anchor_background_basin_after_next48ew"
        spec["evidence_ref"] = (
            "next48ew_complete_negative_old_mean_0p2294_seen_new_mean_0p1510_unknown_far_mean_0p5024;"
            "conservative_old_retention_leaked_unknown_aggressive_unknown_reject_rejected_known;"
            "known_prob_and_bg_prob_high_for_old_seen_new_unknown"
        )
        spec["description"] = (
            "Negative-anchor conservative arm: add a weak query-free background-anchor basin to reduce unknown leakage "
            "while retaining next48ew's only useful old/seen-new retention behavior."
            if not aggressive
            else "Negative-anchor aggressive arm: use stronger background-anchor basin, residual adapter and pseudo-ring negatives "
            "to test whether unknown rejection can improve without collapsing old and seen-new coverage."
        )
        spec["negative_anchor_weight"] = (0.10, 0.16, 0.22)[strictness] if not aggressive else (0.34, 0.46, 0.58)[strictness]
        spec["negative_anchor_margin"] = (0.06, 0.08, 0.10)[strictness] if not aggressive else (0.12, 0.16, 0.20)[strictness]
        spec["negative_anchor_temperature"] = (0.16, 0.14, 0.12)[strictness] if not aggressive else (0.10, 0.08, 0.06)[strictness]
        spec["negative_anchor_max_anchors"] = 192 if not aggressive else 320
        spec["void_background"] = max(float(spec.get("void_background", 0.0)), (0.00, 0.02, 0.04)[strictness] if not aggressive else (0.08, 0.12, 0.16)[strictness])
        spec["unknown_moat"] = max(float(spec.get("unknown_moat", 0.0)), (0.08, 0.12, 0.16)[strictness] if not aggressive else (0.36, 0.48, 0.60)[strictness])
        spec["unknown_margin"] = max(float(spec.get("unknown_margin", 0.0)), (0.34, 0.42, 0.50)[strictness] if not aggressive else (0.70, 0.84, 0.96)[strictness])
        spec["known_coverage_weight"] = max(float(spec.get("known_coverage_weight", 0.0)), (1.55, 1.65, 1.75)[strictness] if not aggressive else (1.10, 1.25, 1.40)[strictness])
        spec["support_retention_guard"] = True
        spec["support_retention_guard_quantile"] = (0.01, 0.02, 0.03)[strictness] if not aggressive else (0.05, 0.08, 0.11)[strictness]
        spec["support_retention_guard_slack"] = (0.12, 0.14, 0.16)[strictness] if not aggressive else (0.08, 0.10, 0.12)[strictness]
        spec["pre_reject_defer_arbitration"] = True
        spec["two_branch_background_guard"] = bool(aggressive or strictness == 2)
        spec["two_branch_bg_min_score"] = (0.90, 0.84, 0.78)[strictness] if not aggressive else (0.70, 0.62, 0.54)[strictness]
        spec["two_branch_bg_min_margin"] = (0.20, 0.12, 0.04)[strictness] if not aggressive else (0.00, -0.08, -0.16)[strictness]
        spec["source_looo_unknown_weight"] = max(float(spec.get("source_looo_unknown_weight", 0.0)), (0.00, 0.04, 0.08)[strictness] if not aggressive else (0.18, 0.26, 0.34)[strictness])
        spec["target_ring_samples"] = max(int(spec.get("target_ring_samples", 0)), 8 if not aggressive else 16)
        spec["target_halo_samples"] = max(int(spec.get("target_halo_samples", 0)), 3 if not aggressive else 7)
        spec["adapter_kind"] = "low_rank" if not aggressive else "residual_mlp"
        spec["steps"] = min(104 if not aggressive else 120, max(76 if not aggressive else 92, int(spec.get("steps", 40)) + 8))
        spec["old_acc_target"] = 0.95
        spec["seen_new_acc_target"] = 0.80
        spec["stage2_max_active_per_gpu"] = 2
        spec["query_per_tx"] = 30
        spec["target_old_query_per_tx"] = 30
        spec["seed_offset"] = int(spec.get("seed_offset", 0)) + 6500
    return specs


def _receiver_spec_for_candidate(index: int) -> dict:
    return dict(PHASE2_TARGET_RECEIVER_POOL[int(index) % len(PHASE2_TARGET_RECEIVER_POOL)])


def _candidate_category_pair(rows: Sequence[Candidate]) -> tuple[str, str] | None:
    counts: dict[str, int] = {}
    for candidate in rows:
        category = str(candidate.optimization_category or "unknown")
        counts[category] = counts.get(category, 0) + 1
    nonzero = [(category, count) for category, count in counts.items() if count > 0]
    if len(nonzero) != 2:
        return None
    nonzero.sort(key=lambda item: (-item[1], item[0]))
    return nonzero[0][0], nonzero[1][0]


def _phase1_ground_training_candidates(plan: str, base_rows: Sequence[Candidate]) -> list[Candidate]:
    category_pair = _candidate_category_pair(base_rows) or ("conservative", "aggressive")
    rows: list[Candidate] = []
    for gpu in range(8):
        category = category_pair[gpu % 2]
        variant = "source_prototype_geometry" if gpu % 2 == 0 else "receiver_distribution_mask_audit"
        rows.append(
            Candidate(
                cid=f"PHASE1_GROUND_PROTO_MASK_{plan}_GPU{gpu}_A",
                protocol="Safe-SSDG-CVS-R01",
                k=0,
                target_visibility="source_only_ground_training_no_target_receiver",
                label_set_relation="Y_old_source_only",
                update_module=(
                    "source_domain_tx_prototypes+receiver_domain_feature_distribution+"
                    "mask_auxiliary_geometry_audit"
                ),
                metrics=(
                    "strict_udu,worst_receiver,sat_mean_5,sat_floor_5,"
                    "prototype_radius,tx_margin_violation,rx_probe_on_z_tx,tx_probe_on_z_rx"
                ),
                command_kind="phase1_safe_ssdg_ground_train",
                gpu=gpu,
                description=(
                    "Source-only Phase1 ground DG optimization row: use CEN51 as non-regression "
                    "experience while adding prototype/mask/feature-distribution evidence; target receiver "
                    "samples remain forbidden for training and threshold fitting."
                ),
                slot=f"GPU{gpu}/A",
                epochs=200,
                seed=260627 + gpu,
                source_tx_ids="0,1,2,3,4,5",
                source_rxs="${CEN51_TRAIN_RXS}",
                target_receiver_ids="",
                new_tx_ids="__NONE__",
                unknown_tx_ids="__NONE__",
                route_family="SAFE_SSDG_CVS_R01",
                loss_profile="safe_ssdg_source_only_paic_plus_prototype_mask_distribution_audit",
                optimization_category=category,
                lane="phase1_ground_dg",
                phase_axis="Phase1-GroundDG",
                phase1_variant=variant,
                phase1_design_report_ref=(
                    "C:/Users/lh594/Downloads/"
                    "PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md"
                ),
                phase1_enable_ground_prototype_stats=True,
                phase1_enable_feature_distribution_audit=True,
                phase1_enable_feature_masks_aux=True,
                phase1_enable_txrx_geometry_audit=True,
            )
        )
    return rows


def _phase1_gpu0_jointsafe_candidates() -> list[Candidate]:
    rows: list[Candidate] = []
    for idx, (variant, spec) in enumerate(PHASE1_GPU0_JOINTSAFE_VARIANTS.items()):
        rows.append(
            Candidate(
                cid=f"PHASE1_GPU0_JOINTSAFE_{variant.upper()}",
                protocol="Safe-SSDG-CVS-R01",
                k=0,
                target_visibility="source_only_ground_training_no_target_receiver",
                label_set_relation="Y_old_source_only",
                update_module="gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit",
                metrics=(
                    "joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,"
                    "sat_strict_mean_3,pseudo_precision,paic_guard"
                ),
                command_kind="phase1_safe_ssdg_ground_train",
                gpu=idx,
                description=str(spec["description"]),
                slot=f"GPU{idx}/A",
                epochs=int(spec["label_epochs"]) + int(spec["pseudo_epochs"]),
                seed=260629 + idx,
                source_tx_ids="0,1,2,3,4,5",
                source_rxs="${CEN51_TRAIN_RXS}",
                target_receiver_ids="",
                new_tx_ids="__NONE__",
                unknown_tx_ids="__NONE__",
                route_family="SAFE_SSDG_CVS_R01",
                loss_profile="gpu0_a_joint_safe_paic_guard_plus_prototype_mask_audit_only",
                optimization_category="conservative" if idx < 2 else "aggressive",
                lane="phase1_ground_dg",
                phase_axis="Phase1-GroundDG",
                phase1_variant=variant,
                phase1_design_report_ref=(
                    "C:/Users/lh594/Downloads/"
                    "PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md"
                ),
                phase1_enable_ground_prototype_stats=True,
                phase1_enable_feature_distribution_audit=True,
                phase1_enable_feature_masks_aux=True,
                phase1_enable_txrx_geometry_audit=True,
            )
        )
    return rows


def _with_phase1_ground_rows(plan: str, rows: list[Candidate]) -> list[Candidate]:
    if not rows:
        return rows
    if not str(plan).upper().startswith("OA_MSE_"):
        return rows
    if len(rows) != 48:
        return rows
    if _candidate_category_pair(rows) is None:
        return rows
    return _phase1_ground_training_candidates(plan, rows) + rows


def make_candidates(*, plan: str = "SMOKE") -> list[Candidate]:
    plan = str(plan).upper()
    if plan == "PHASE1_GPU0_JOINTSAFE4":
        return _phase1_gpu0_jointsafe_candidates()
    if plan == "SMOKE":
        return [
            Candidate(
                cid="SFE_ZID_PROTO_K5_SYNTH",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_satellite_support_labeled",
                label_set_relation="Y_T_has_unknown_new_tx",
                update_module="new_prototype_plus_unknown_gate",
                metrics="full_accuracy,accepted_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate",
                command_kind="feature_sfe_synthetic",
                gpu=0,
                description="Synthetic z_id smoke for new-TX enrollment core logic; support features represent samples after H_sg o R_sat.",
            ),
            Candidate(
                cid="FTRC_SAT_RXTX_K2_LABELED_BASE",
                protocol="CVS-FTRC",
                k=2,
                target_visibility="target_receiver_satellite_support_labeled",
                label_set_relation="Y_T_equals_Y_S",
                update_module="target_receiver_labeled_adapter_with_source_anchor",
                metrics="full_accuracy,accepted_accuracy,coverage,strict_udu,sat_floor,ECE",
                command_kind="target_adapt_labeled_rxtx",
                gpu=0,
                description="N607 smoke candidate using train_target_adapt.py with per-RX-per-TX labeled support after synthetic star-ground channel.",
                epochs=1,
                adapt_steps_per_epoch=2,
                eval_max_batches=1,
                sat_eval_max_batches=1,
                loss_profile="supervised_ce_plus_source_anchor_no_semisupervised",
            ),
        ]
    if plan == "CORE":
        rows: list[Candidate] = []
        for k, gpu in [(1, 0), (2, 1), (5, 2), (10, 3), (20, 4)]:
            rows.append(
                Candidate(
                    cid=f"SFE_ZID_PROTO_K{k}_SYNTH",
                    protocol="CVS-SFE",
                    k=k,
                    target_visibility="new_class_satellite_support_labeled",
                    label_set_relation="Y_T_has_unknown_new_tx",
                    update_module="new_prototype_plus_unknown_gate",
                    metrics="full_accuracy,accepted_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate",
                    command_kind="feature_sfe_synthetic",
                    gpu=gpu,
                    description=f"Feature-level new-TX enrollment sanity check at K={k}; support features represent samples after H_sg o R_sat.",
                )
            )
        for k, gpu in [(1, 0), (2, 1), (5, 2), (10, 3), (20, 4)]:
            rows.append(
                Candidate(
                    cid=f"FTRC_SAT_RXTX_K{k}_LABELED_BASE",
                    protocol="CVS-FTRC",
                    k=k,
                    target_visibility="target_receiver_satellite_support_labeled",
                    label_set_relation="Y_T_equals_Y_S",
                    update_module="target_receiver_labeled_adapter_with_source_anchor",
                    metrics="full_accuracy,accepted_accuracy,coverage,strict_udu,sat_floor,ECE",
                    command_kind="target_adapt_labeled_rxtx",
                    gpu=gpu,
                    description=f"Target receiver calibration with K={k} samples per RX/TX after synthetic star-ground channel.",
                    epochs=20,
                    adapt_steps_per_epoch=20,
                    eval_max_batches=0,
                    sat_eval_max_batches=0,
                    loss_profile="supervised_ce_plus_source_anchor_no_semisupervised",
                )
            )
        return rows
    if plan == "WISIG_NEWCLASS":
        return [
            Candidate(
                cid="SFE_WISIG_NEW_TX_K5_STRICT",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="frozen_z_id_export_plus_new_prototype_unknown_gate",
                metrics="full_accuracy,accepted_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,tx_overlap_audit",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=0,
                description="Export checkpoint z_id features from source and other WiSig subset TXs, assert transmitter identity non-overlap, then run CVS-SFE.",
                seed=1342,
            )
        ]
    if plan == "WISIG_NEWCLASS_CARD8":
        rows: list[Candidate] = []
        for gpu, k, threshold, seed, suffix in [
            (0, 1, 0.70, 1338, "K1_T070"),
            (1, 2, 0.70, 1339, "K2_T070"),
            (2, 5, 0.70, 1342, "K5_T070_BASE"),
            (3, 10, 0.70, 1347, "K10_T070"),
            (4, 20, 0.70, 1357, "K20_T070"),
            (5, 5, 0.60, 1362, "K5_T060_COVERAGE"),
            (6, 5, 0.80, 1367, "K5_T080_PRECISION"),
            (7, 5, 0.70, 1372, "K5_T070_UNKNOWN_AUDIT"),
        ]:
            rows.append(
                Candidate(
                    cid=f"SFE_WISIG_NEW_TX_{suffix}",
                    protocol="CVS-SFE",
                    k=k,
                    target_visibility="new_class_wisig_support_labeled",
                    label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                    update_module="frozen_z_id_export_plus_new_prototype_unknown_gate",
                    metrics="full_accuracy,accepted_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,rollback_gate,tx_overlap_audit,split_overlap_audit",
                    command_kind="feature_sfe_wisig_nonoverlap",
                    gpu=gpu,
                    description=f"One-card WiSig new-TX SFE validation at K={k}, cosine threshold={threshold:.2f}; GPU7 requires audited UNKNOWN_TX_IDS before launch.",
                    seed=seed,
                    unknown_threshold=threshold,
                    gate_mode="cosine",
                )
            )
        return rows
    if plan == "WISIG_ENHANCED_CARD8":
        return [
            Candidate(
                cid="SFE_WISIG_GATE_COSINE_K5",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_cosine_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,lifecycle_state",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=0,
                description="Baseline cosine threshold gate with lifecycle/rollback telemetry.",
                seed=1342,
                gate_mode="cosine",
            ),
            Candidate(
                cid="SFE_WISIG_GATE_MARGIN_K5",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_margin_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,lifecycle_state",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=1,
                description="Margin gate rejects ambiguous new/old boundaries.",
                seed=1343,
                gate_mode="combined",
                min_margin=0.05,
            ),
            Candidate(
                cid="SFE_WISIG_GATE_MAHAL_K5",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_mahalanobis_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,mahalanobis_gate",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=2,
                description="Diagonal Mahalanobis gate over frozen z_id prototype statistics.",
                seed=1344,
                gate_mode="mahalanobis",
                max_mahalanobis=8.0,
            ),
            Candidate(
                cid="SFE_WISIG_GATE_OPENMAX_K5",
                protocol="CVS-SFE",
                k=5,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_openmax_tail_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,openmax_tail_gate",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=3,
                description="OpenMax-style class tail gate fitted only on support/source calibration splits.",
                seed=1345,
                gate_mode="openmax",
                openmax_quantile=1.0,
            ),
            Candidate(
                cid="SFE_WISIG_GATE_COMBINED_K10",
                protocol="CVS-SFE",
                k=10,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_combined_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,combined_gate",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=4,
                description="Combined cosine/margin/Mahalanobis/OpenMax-style gate at K=10.",
                seed=1347,
                gate_mode="combined",
                min_margin=0.05,
                max_mahalanobis=8.0,
                openmax_quantile=1.0,
            ),
            Candidate(
                cid="SFE_WISIG_GATE_COMBINED_K20",
                protocol="CVS-SFE",
                k=20,
                target_visibility="new_class_wisig_support_labeled",
                label_set_relation="Y_T_has_explicit_nonoverlap_tx",
                update_module="new_prototype_combined_gate_lifecycle_rollback",
                metrics="full_accuracy,coverage,new_class_accuracy,old_class_accuracy,unknown_rejection_rate,rollback_gate,combined_gate",
                command_kind="feature_sfe_wisig_nonoverlap",
                gpu=5,
                description="Combined gate high-K upper-bound sanity check.",
                seed=1357,
                gate_mode="combined",
                min_margin=0.05,
                max_mahalanobis=8.0,
                openmax_quantile=1.0,
            ),
            Candidate(
                cid="FTRC_WISIG_FEATURE_ADAPTER_K2",
                protocol="CVS-FTRC",
                k=2,
                target_visibility="target_receiver_satellite_support_labeled",
                label_set_relation="Y_T_equals_Y_S",
                update_module="feature_residual_adapter_with_source_anchor_and_rollback",
                metrics="target_tx,overall_tx,sat_mean,rollback_gate,adapter_delta_logit_norm",
                command_kind="target_adapt_labeled_rxtx",
                gpu=6,
                description="Frozen-backbone feature residual adapter; only adapter/logit delta modules train.",
                epochs=20,
                adapt_steps_per_epoch=20,
                eval_max_batches=0,
                sat_eval_max_batches=0,
                loss_profile="supervised_ce_plus_source_anchor_no_semisupervised",
                target_adapter_type="feature_residual",
                adapter_bottleneck=16,
                freeze_base_stats=True,
            ),
            Candidate(
                cid="FTRC_WISIG_LOGIT_LORA_K2",
                protocol="CVS-FTRC",
                k=2,
                target_visibility="target_receiver_satellite_support_labeled",
                label_set_relation="Y_T_equals_Y_S",
                update_module="logit_lora_adapter_with_source_anchor_and_rollback",
                metrics="target_tx,overall_tx,sat_mean,rollback_gate,adapter_delta_logit_norm",
                command_kind="target_adapt_labeled_rxtx",
                gpu=7,
                description="Frozen-backbone low-rank logit LoRA delta; no full backbone fine-tuning.",
                epochs=20,
                adapt_steps_per_epoch=20,
                eval_max_batches=0,
                sat_eval_max_batches=0,
                loss_profile="supervised_ce_plus_source_anchor_no_semisupervised",
                target_adapter_type="logit_lora",
                adapter_rank=4,
                freeze_base_stats=True,
            ),
        ]
    if plan in {
        "OA_MSE_CARD3",
        "OA_MSE_PROXY32",
        "OA_MSE_BOUNDARY32",
        "OA_MSE_UNCERTAIN32",
        "OA_MSE_VETO32",
        "OA_MSE_CLASSCOND32",
        "OA_MSE_CALGUARD32",
        "OA_MSE_BALANCE64",
        "OA_MSE_SOFTMIX64",
        "OA_MSE_VOID64",
        "OA_MSE_SOFTVOID128",
        "OA_MSE_ANCHORGUARD128",
        "OA_MSE_MIXHEAD128",
        "OA_MSE_STRUCT48",
        "OA_MSE_SIMPLIFIED48",
        "OA_MSE_RETENTION48",
        "OA_MSE_SUPPORTRET48",
        "OA_MSE_TWOBRANCH48",
        "OA_MSE_REGHEAD48",
        "OA_MSE_GEOM48",
        "OA_MSE_TRIAGE48",
        "OA_MSE_LOOO48",
        "OA_MSE_CONSTRAIN48",
        "OA_MSE_ENVELOPE48",
        "OA_MSE_RESCUE48",
        "OA_MSE_PREREJECT48",
        "OA_MSE_THREEWAY48",
        "OA_MSE_COVFLOOR48",
        "OA_MSE_CLASSFIRST48",
        "OA_MSE_EVIBG48",
        "OA_MSE_SOFTTARGET48",
        "OA_MSE_NEGANCHOR48",
        "OA_MSE_DENSHELL48",
        "OA_MSE_IDCONS48",
        "OA_MSE_CONFORM48",
        "OA_MSE_RECON48",
        "OA_MSE_SOURCERISK48",
        "OA_MSE_SUPPORTCV48",
        "OA_MSE_BGCAP48",
        "OA_MSE_KRET48",
        "OA_MSE_RISKRET48",
        "OA_MSE_MANIFOLD48",
        "OA_MSE_H06_EVID48",
        "OA_MSE_H06_ARB48",
        "OA_MSE_H06_OLDUNK48",
        "OA_MSE_H06_BGTRAIN48",
        "OA_MSE_H06_RETOLD48",
        "OA_MSE_H06_OLDFIRST48",
        "OA_MSE_H06_OLDRELAX48",
        "OA_MSE_H06_OLDGEOM48",
        "OA_MSE_H06_OLDCONF48",
        "OA_MSE_H06_OLDBUDGET48",
        "OA_MSE_H06_OLDQUAL48",
        "OA_MSE_H06_OLDRISK48",
        "OA_MSE_H06_OLDFUSE48",
        "OA_MSE_H06_ROLLSAFE48",
        "OA_MSE_H06_OLDHEAD48",
        "OA_MSE_H06_OLDHEADFAR48",
        "OA_MSE_H06_OLDRECOV48",
    }:
        base = {
            "protocol": "CVS-OA-MSE",
            "label_set_relation": "Y_T_has_explicit_nonoverlap_tx",
            "command_kind": "feature_oa_mse_wisig_nonoverlap",
            "gate_mode": "oa_mse",
            "route_family": "OA_MSE_HEAD",
            "unknown_tx_ids": "${OA_MSE_UNKNOWN_TX_IDS}",
            "source_rxs": "${CEN51_TRAIN_RXS}",
            "target_receiver_ids": "${TARGET_RECEIVER_IDS}",
            "ground_model_label": (
                "CEN51_R04_H06_LOW_PROB_HYBRID_R010"
                if plan in {"OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"}
                else "BEX02_fishr002_mixed_e170"
            ),
            "ground_model_default_ckpt": (
                H06_LOW_PROB_HYBRID_LATEST_CKPT
                if plan in {"OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"}
                else DEFAULT_BEX02_TEACHER_CKPT
            ),
            "unknown_leo_query": True,
            "openmax_quantile": 1.0,
            "onboard_low_compute_training": True,
            "compute_budget_profile": "feature_level_low_rank_adapter_rank2_max40steps_no_backbone_update",
            "adapter_trainable_params_cap": 4096,
            "max_adapt_steps": 40,
            "oa_mse_source_anchor_weight": 0.05,
            "oa_mse_source_ce_weight": 0.10,
            "oa_mse_unknown_moat_weight": 0.10,
            "oa_mse_unknown_moat_margin": 0.45,
            "pseudo_unknown_samples_per_pair": 4,
            "pseudo_unknown_offset_scale": 0.15,
            "pseudo_unknown_source_boundary_samples_per_pair": 2,
            "pseudo_unknown_source_boundary_offset_scale": 0.18,
            "pseudo_unknown_target_shift_samples_per_class": 2,
            "pseudo_unknown_target_shift_offset_scale": 0.20,
            "pseudo_unknown_target_halo_samples_per_class": 2,
            "pseudo_unknown_target_halo_offset_scale": 0.35,
            "pseudo_unknown_target_ring_samples_per_class": 3,
            "pseudo_unknown_target_ring_offset_scale": 0.45,
            "oa_mse_old_bridge_weight": 0.15,
            "old_bridge_samples_per_class": 3,
            "old_bridge_max_mix": 0.85,
            "oa_mse_support_contrast_weight": 0.12,
            "old_support_contrast_negative_margin": 0.78,
            "old_support_contrast_positive_margin": 0.88,
            "oa_mse_soft_proto_weight": 0.08,
            "soft_proto_topk": 2,
            "soft_proto_temperature": 0.10,
            "oa_mse_soft_proto_boundary_weight": 0.0,
            "soft_proto_boundary_margin": 0.15,
            "oa_mse_void_background_weight": 0.0,
            "oa_mse_void_gate": False,
            "oa_mse_void_gate_min_score": 0.55,
            "oa_mse_void_gate_min_margin": 0.05,
            "oa_mse_old_neighborhood_weight": 0.10,
            "old_neighborhood_samples_per_class": 2,
            "old_neighborhood_radius": 0.06,
            "oa_mse_old_surrogate_margin_weight": 0.05,
            "old_surrogate_margin": 0.10,
            "oa_mse_source_looo_unknown_weight": 0.0,
            "source_looo_unknown_margin": 0.35,
            "source_looo_interclass_margin": 0.08,
            "source_looo_max_samples_per_class": 24,
            "oa_mse_source_looo_risk_arbitration": False,
            "source_looo_risk_quantile": 0.85,
            "source_looo_risk_slack": 0.0,
            "source_looo_risk_min_score_margin": 0.02,
            "source_looo_risk_min_known_evidence_delta": -0.08,
            "source_looo_risk_background_score": 0.86,
            "source_looo_risk_background_margin": 0.10,
            "source_looo_risk_reject_min_failures": 2,
            "source_looo_risk_reject_action": "reject",
            "old_surrogate_evidence_margin": 0.0,
            "old_surrogate_reject_relax": 0.0,
            "oa_mse_siamese_quantile": 0.10,
            "oa_mse_siamese_accept_threshold": 0.50,
            "oa_mse_siamese_unknown_veto": False,
            "oa_mse_siamese_unknown_veto_mode": "any",
            "oa_mse_siamese_min_old_support_evidence_delta": None,
            "oa_mse_siamese_min_old_surrogate_reject_delta": None,
            "oa_mse_siamese_min_energy_delta": None,
            "oa_mse_siamese_min_mahalanobis_delta": None,
            "oa_mse_siamese_min_accept_delta": None,
            "oa_mse_siamese_min_old_support_anchor_margin": None,
            "oa_mse_siamese_min_veto_failures": 1,
            "oa_mse_old_unknown_acceptance_guard": False,
            "oa_mse_old_unknown_guard_min_old_support_evidence_delta": None,
            "oa_mse_old_unknown_guard_min_old_surrogate_reject_delta": None,
            "oa_mse_old_unknown_guard_min_energy_delta": None,
            "oa_mse_old_unknown_guard_min_mahalanobis_delta": None,
            "oa_mse_old_unknown_guard_min_accept_delta": None,
            "oa_mse_old_unknown_guard_min_old_support_anchor_margin": None,
            "oa_mse_old_unknown_guard_min_best_old_score": None,
            "oa_mse_old_unknown_guard_min_margin": None,
            "oa_mse_old_unknown_guard_min_failures": 1,
            "oa_mse_adapter_selection_policy": (
                "target_boundary_guard"
                if plan in {
                    "OA_MSE_BOUNDARY32",
                    "OA_MSE_UNCERTAIN32",
                    "OA_MSE_VETO32",
                    "OA_MSE_CLASSCOND32",
                    "OA_MSE_CALGUARD32",
                    "OA_MSE_BALANCE64",
                    "OA_MSE_SOFTMIX64",
                    "OA_MSE_VOID64",
                    "OA_MSE_SOFTVOID128",
                    "OA_MSE_ANCHORGUARD128",
                    "OA_MSE_MIXHEAD128",
                    "OA_MSE_STRUCT48",
                    "OA_MSE_SIMPLIFIED48",
                    "OA_MSE_RETENTION48",
                    "OA_MSE_SUPPORTRET48",
                    "OA_MSE_TWOBRANCH48",
                    "OA_MSE_REGHEAD48",
                    "OA_MSE_GEOM48",
                    "OA_MSE_TRIAGE48",
                    "OA_MSE_LOOO48",
                    "OA_MSE_CONSTRAIN48",
                    "OA_MSE_ENVELOPE48",
                    "OA_MSE_RESCUE48",
                    "OA_MSE_PREREJECT48",
                    "OA_MSE_THREEWAY48",
                    "OA_MSE_COVFLOOR48",
                    "OA_MSE_CLASSFIRST48",
                    "OA_MSE_EVIBG48",
                    "OA_MSE_SOFTTARGET48",
                    "OA_MSE_NEGANCHOR48",
                    "OA_MSE_DENSHELL48",
                    "OA_MSE_IDCONS48",
                }
                else "proxy_line_search"
            ),
            "oa_mse_adapter_alpha_eval_sweep": True,
            "old_anchor_override_min_quality": 0.55,
            "old_retention_quantile": 0.95,
            "oa_mse_support_retention_guard": False,
            "support_retention_guard_quantile": 0.05,
            "support_retention_guard_slack": 0.02,
            "oa_mse_class_envelope_gate": False,
            "class_envelope_evidence_quantile": 0.05,
            "class_envelope_residual_quantile": 0.95,
            "class_envelope_score_quantile": 0.05,
            "class_envelope_margin_quantile": 0.05,
            "class_envelope_evidence_slack": 0.02,
            "class_envelope_residual_slack": 0.02,
            "class_envelope_score_slack": 0.05,
            "class_envelope_margin_slack": 0.02,
            "class_envelope_min_failures": 1,
            "class_envelope_gate_action": "reject",
            "oa_mse_density_shell_gate": False,
            "density_shell_old_min_evidence_delta": -0.04,
            "density_shell_old_min_anchor_delta": -0.08,
            "density_shell_old_min_density_delta": -0.06,
            "density_shell_seen_new_min_evidence_delta": -0.04,
            "density_shell_seen_new_min_anchor_delta": -0.08,
            "density_shell_seen_new_min_density_delta": -0.06,
            "density_shell_accept_background_margin": 0.18,
            "density_shell_reject_background_score": 0.86,
            "density_shell_reject_background_margin": 0.14,
            "density_shell_reject_min_failed_shells": 2,
            "oa_mse_identity_consensus_arbitration": False,
            "identity_consensus_old_min_evidence_delta": -0.06,
            "identity_consensus_old_min_anchor_delta": -0.10,
            "identity_consensus_old_min_density_delta": -0.08,
            "identity_consensus_seen_new_min_evidence_delta": -0.04,
            "identity_consensus_seen_new_min_anchor_delta": -0.08,
            "identity_consensus_seen_new_min_density_delta": -0.06,
            "identity_consensus_min_identity_margin": -0.05,
            "identity_consensus_background_accept_margin": 0.22,
            "identity_consensus_reject_background_score": 0.90,
            "identity_consensus_reject_background_margin": 0.18,
            "identity_consensus_reject_min_identity_failures": 4,
            "oa_mse_support_conformal_arbitration": False,
            "support_conformal_calibration_quantile": 0.05,
            "support_conformal_conformity_slack": 0.12,
            "support_conformal_anchor_margin_slack": 0.06,
            "support_conformal_background_score": 0.82,
            "support_conformal_background_margin": 0.08,
            "support_conformal_hard_reject_margin": 0.18,
            "support_conformal_reject_min_failures": 2,
            "support_conformal_reject_action": "reject",
            "pre_reject_support_neighborhood_retention": False,
            "pre_reject_support_retention_old_min_evidence_delta": 0.02,
            "pre_reject_support_retention_old_min_anchor_delta": -0.04,
            "pre_reject_support_retention_old_min_anchor_margin": -0.02,
            "pre_reject_support_retention_old_min_score_margin": -0.04,
            "pre_reject_support_retention_seen_new_min_evidence_delta": 0.02,
            "pre_reject_support_retention_seen_new_min_anchor_delta": -0.04,
            "pre_reject_support_retention_seen_new_min_score_margin": -0.08,
            "pre_reject_support_retention_max_background_score": 0.96,
            "pre_reject_support_retention_max_background_margin": 0.30,
            "oa_mse_retention_rescue_gate": False,
            "retention_rescue_old_min_evidence_delta": 0.02,
            "retention_rescue_old_min_anchor_delta": -0.01,
            "retention_rescue_old_min_anchor_margin": 0.0,
            "retention_rescue_old_min_score_margin": 0.0,
            "retention_rescue_seen_new_min_evidence_delta": 0.02,
            "retention_rescue_seen_new_min_anchor_delta": 0.0,
            "retention_rescue_seen_new_min_score_margin": -0.02,
            "retention_rescue_max_background_score": 0.70,
            "retention_rescue_max_background_margin": 0.06,
            "oa_mse_two_branch_background_guard": False,
            "two_branch_bg_min_score": 0.62,
            "two_branch_bg_min_margin": -0.02,
            "two_branch_old_support_evidence_delta": 0.0,
            "two_branch_old_anchor_delta": -0.02,
            "two_branch_old_anchor_margin": 0.0,
            "two_branch_seen_new_evidence_delta": 0.0,
            "two_branch_seen_new_anchor_delta": 0.0,
            "oa_mse_support_center_ce_weight": 0.0,
            "support_center_temperature": 0.10,
            "support_center_margin": 0.10,
            "old_acc_target": 0.90,
            "seen_new_acc_target": 0.75,
            "weibull_evt_required": True,
            "target_adapter_required": True,
            "pseudo_unknown_energy_required": True,
            "seen_new_evidence_gate_required": True,
            "seen_new_anchor_gate_required": True,
            "siamese_verifier_required": True,
            "accepted_only_online_update_required": True,
            "oa_mse_onboard_adaptation_bundle": ONBOARD_ADAPTATION_BUNDLE,
            }
        if plan == "OA_MSE_CARD3":
            return [
            Candidate(
                cid="OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN",
                k=0,
                gpu=0,
                target_visibility="source_old_only_with_leo_unknown_query_eval",
                update_module="mse_lite_source_old_masks_energy_gate",
                metrics="full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,energy,subspace_residual,decision_histogram",
                description="Stage2-A OA-MSE-lite: source old classes only; target-new and unknown query rows are eval-only and never calibrate thresholds.",
                oa_mse_stage="mse_lite",
                eval_protocol="source_open_set",
                source_target_fusion_policy="source_old_only_no_target_support",
                fusion_inputs="source_old_prototypes,source_old_covariance,source_old_masks",
                target_old_leo_support=False,
                target_new_leo_support=False,
                seed=1441,
                **base,
            ),
            Candidate(
                cid="OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD",
                k=2,
                gpu=1,
                target_visibility="target_old_leo_support_labeled_unknown_eval_only",
                update_module="mse_subspace_target_old_radius_and_orbit_basis",
                metrics="full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,energy,subspace_residual,decision_histogram",
                description="Stage2-B OA-MSE-subspace: target-old support updates old-class masks/radii and U_orbit; target-new support is forbidden.",
                oa_mse_stage="mse_subspace",
                eval_protocol="ftrc",
                source_target_fusion_policy="source_old_plus_target_old_labeled_support_only",
                fusion_inputs="source_old_prototypes,target_old_support,old_class_radius,U_orbit",
                target_old_leo_support=True,
                target_new_leo_support=False,
                target_old_tx_ids="${TARGET_OLD_TX_IDS}",
                target_old_support_per_tx=2,
                target_old_query_per_tx=50,
                seed=1442,
                **base,
            ),
            Candidate(
                cid="OA_MSE_STAGE2C_HEAD_SEEN_NEW",
                k=5,
                gpu=2,
                target_visibility="target_old_and_seen_new_leo_support_labeled_unknown_eval_only",
                update_module="oa_mse_head_old_subspace_seen_new_registration_uncertain_defer",
                metrics="full_accuracy,coverage,old_class_accuracy,new_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,energy,subspace_residual,decision_histogram",
                description="Stage2-C OA-MSE-Head: target-old calibration plus explicit seen-new support registration; unknown query remains eval-only.",
                oa_mse_stage="oa_mse_head",
                eval_protocol="sfe",
                source_target_fusion_policy="source_old_plus_target_old_support_plus_seen_new_support_no_unknown_fit",
                fusion_inputs="source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate",
                target_old_leo_support=True,
                target_new_leo_support=True,
                target_old_tx_ids="${TARGET_OLD_TX_IDS}",
                target_old_support_per_tx=5,
                target_new_support_per_tx=5,
                target_old_query_per_tx=50,
                seed=1443,
                **base,
            ),
        ]
        rows: list[Candidate] = []
        stage_specs = [
            {
                "slot": "A",
                "stage": "mse_subspace",
                "eval_protocol": "ftrc",
                "k_old": 2,
                "k_new": 0,
                "steps": 20,
                "unknown_threshold": 0.92,
                "openmax_quantile": 0.99,
                "openmax_min_threshold": 0.08,
                "source_ce": 0.18,
                "unknown_moat": 0.14,
                "unknown_margin": 0.50,
                "boundary_samples": 3,
                "boundary_offset": 0.18,
                "soft_proto": 0.06,
                "support_contrast": 0.10,
                "old_bridge": 0.12,
                "old_neighborhood": 0.12,
                "description": "Proxy alpha selection for conservative target-old retention under unknown-risk control.",
            },
            {
                "slot": "B",
                "stage": "oa_mse_head",
                "eval_protocol": "sfe",
                "k_old": 2,
                "k_new": 2,
                "steps": 20,
                "unknown_threshold": 0.96,
                "openmax_quantile": 1.0,
                "openmax_min_threshold": 0.10,
                "source_ce": 0.16,
                "unknown_moat": 0.12,
                "unknown_margin": 0.45,
                "boundary_samples": 3,
                "boundary_offset": 0.20,
                "soft_proto": 0.08,
                "support_contrast": 0.12,
                "old_bridge": 0.12,
                "old_neighborhood": 0.10,
                "description": "Proxy alpha selection for low-shot seen-new registration without unknown query calibration.",
            },
            {
                "slot": "C",
                "stage": "mse_subspace",
                "eval_protocol": "ftrc",
                "k_old": 5,
                "k_new": 0,
                "steps": 40,
                "unknown_threshold": 0.94,
                "openmax_quantile": 0.985,
                "openmax_min_threshold": 0.10,
                "source_ce": 0.24,
                "unknown_moat": 0.22,
                "unknown_margin": 0.58,
                "boundary_samples": 5,
                "boundary_offset": 0.16,
                "soft_proto": 0.10,
                "support_contrast": 0.16,
                "old_bridge": 0.14,
                "old_neighborhood": 0.18,
                "description": "Proxy alpha selection with stronger old-class moat and source-boundary pressure.",
            },
            {
                "slot": "D",
                "stage": "oa_mse_head",
                "eval_protocol": "sfe",
                "k_old": 5,
                "k_new": 5,
                "steps": 30,
                "unknown_threshold": 0.98,
                "openmax_quantile": 1.0,
                "openmax_min_threshold": 0.12,
                "source_ce": 0.20,
                "unknown_moat": 0.18,
                "unknown_margin": 0.55,
                "boundary_samples": 5,
                "boundary_offset": 0.18,
                "soft_proto": 0.10,
                "support_contrast": 0.14,
                "old_bridge": 0.14,
                "old_neighborhood": 0.16,
                "description": "Proxy alpha selection for K5 seen-new registration with stricter rejection risk.",
            },
        ]
        if plan in {
            "OA_MSE_BALANCE64",
            "OA_MSE_SOFTMIX64",
            "OA_MSE_VOID64",
            "OA_MSE_SOFTVOID128",
            "OA_MSE_ANCHORGUARD128",
            "OA_MSE_MIXHEAD128",
            "OA_MSE_STRUCT48",
            "OA_MSE_SIMPLIFIED48",
            "OA_MSE_RETENTION48",
                    "OA_MSE_SUPPORTRET48",
                    "OA_MSE_TWOBRANCH48",
                    "OA_MSE_REGHEAD48",
                    "OA_MSE_GEOM48",
                    "OA_MSE_TRIAGE48",
                    "OA_MSE_LOOO48",
                    "OA_MSE_CONSTRAIN48",
                    "OA_MSE_ENVELOPE48",
                    "OA_MSE_RESCUE48",
                    "OA_MSE_PREREJECT48",
                    "OA_MSE_THREEWAY48",
                    "OA_MSE_COVFLOOR48",
                    "OA_MSE_CLASSFIRST48",
                    "OA_MSE_EVIBG48",
                    "OA_MSE_SOFTTARGET48",
                    "OA_MSE_NEGANCHOR48",
                    "OA_MSE_DENSHELL48",
                    "OA_MSE_IDCONS48",
                    "OA_MSE_CONFORM48",
                    "OA_MSE_RECON48",
                    "OA_MSE_SOURCERISK48",
                    "OA_MSE_SUPPORTCV48",
                    "OA_MSE_BGCAP48",
                    "OA_MSE_KRET48",
                    "OA_MSE_RISKRET48",
                    "OA_MSE_MANIFOLD48",
                    "OA_MSE_H06_EVID48",
                    "OA_MSE_H06_ARB48",
                    "OA_MSE_H06_OLDUNK48",
                    "OA_MSE_H06_BGTRAIN48",
                    "OA_MSE_H06_RETOLD48",
                    "OA_MSE_H06_OLDFIRST48",
                    "OA_MSE_H06_OLDRELAX48",
                    "OA_MSE_H06_OLDGEOM48",
                    "OA_MSE_H06_OLDCONF48",
                    "OA_MSE_H06_OLDBUDGET48",
                    "OA_MSE_H06_OLDQUAL48",
                    "OA_MSE_H06_OLDRISK48",
                    "OA_MSE_H06_OLDFUSE48",
                    "OA_MSE_H06_ROLLSAFE48",
                    "OA_MSE_H06_OLDHEAD48",
                    "OA_MSE_H06_OLDHEADFAR48",
                    "OA_MSE_H06_OLDRECOV48",
                }:
            stage_specs = [
                {
                    "slot": "A",
                    "stage": "mse_subspace",
                    "eval_protocol": "ftrc",
                    "k_old": 2,
                    "k_new": 0,
                    "steps": 20,
                    "unknown_threshold": 0.90,
                    "openmax_quantile": 0.985,
                    "openmax_min_threshold": 0.06,
                    "source_ce": 0.24,
                    "unknown_moat": 0.12,
                    "unknown_margin": 0.44,
                    "boundary_samples": 3,
                    "boundary_offset": 0.18,
                    "soft_proto": 0.12,
                    "soft_proto_boundary": 0.18,
                    "soft_proto_boundary_margin": 0.12,
                    "support_contrast": 0.12,
                    "old_bridge": 0.18,
                    "old_neighborhood": 0.20,
                    "reject_relax": 0.34,
                    "siamese_quantile": 0.10,
                    "siamese_threshold": 0.58,
                    "old_retention_quantile": 0.84,
                    "description": "Balanced old-rescue OA-MSE: relax old-surrogate hard rejection and prioritize target-old retention before unknown-risk veto.",
                },
                {
                    "slot": "B",
                    "stage": "oa_mse_head",
                    "eval_protocol": "sfe",
                    "k_old": 2,
                    "k_new": 2,
                    "steps": 24,
                    "unknown_threshold": 0.94,
                    "openmax_quantile": 0.995,
                    "openmax_min_threshold": 0.08,
                    "source_ce": 0.20,
                    "unknown_moat": 0.10,
                    "unknown_margin": 0.42,
                    "boundary_samples": 3,
                    "boundary_offset": 0.20,
                    "soft_proto": 0.14,
                    "soft_proto_boundary": 0.20,
                    "soft_proto_boundary_margin": 0.14,
                    "support_contrast": 0.10,
                    "old_bridge": 0.16,
                    "old_neighborhood": 0.16,
                    "reject_relax": 0.38,
                    "siamese_quantile": 0.15,
                    "siamese_threshold": 0.60,
                    "old_retention_quantile": 0.84,
                    "description": "Balanced seen-new rescue OA-MSE: make low-shot seen-new evidence less likely to be rejected while keeping unknown query eval-only.",
                },
                {
                    "slot": "C",
                    "stage": "mse_subspace",
                    "eval_protocol": "ftrc",
                    "k_old": 5,
                    "k_new": 0,
                    "steps": 32,
                    "unknown_threshold": 0.92,
                    "openmax_quantile": 0.99,
                    "openmax_min_threshold": 0.08,
                    "source_ce": 0.26,
                    "unknown_moat": 0.16,
                    "unknown_margin": 0.48,
                    "boundary_samples": 4,
                    "boundary_offset": 0.18,
                    "soft_proto": 0.14,
                    "soft_proto_boundary": 0.20,
                    "soft_proto_boundary_margin": 0.14,
                    "support_contrast": 0.14,
                    "old_bridge": 0.18,
                    "old_neighborhood": 0.22,
                    "reject_relax": 0.34,
                    "siamese_quantile": 0.10,
                    "siamese_threshold": 0.62,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.18,
                    "min_old_surrogate_reject_delta": -0.08,
                    "min_energy_delta": -25.0,
                    "min_mahalanobis_delta": -60.0,
                    "min_accept_delta": -40.0,
                    "min_old_support_anchor_margin": 0.015,
                    "min_veto_failures": 4,
                    "old_retention_quantile": 0.82,
                    "description": "K5 old-rescue with late coupled risk veto: reject only when several diagnostics and anchor margin are weak together.",
                },
                {
                    "slot": "D",
                    "stage": "oa_mse_head",
                    "eval_protocol": "sfe",
                    "k_old": 5,
                    "k_new": 5,
                    "steps": 36,
                    "unknown_threshold": 0.95,
                    "openmax_quantile": 0.995,
                    "openmax_min_threshold": 0.08,
                    "source_ce": 0.22,
                    "unknown_moat": 0.14,
                    "unknown_margin": 0.46,
                    "boundary_samples": 4,
                    "boundary_offset": 0.20,
                    "soft_proto": 0.16,
                    "soft_proto_boundary": 0.22,
                    "soft_proto_boundary_margin": 0.16,
                    "support_contrast": 0.12,
                    "old_bridge": 0.16,
                    "old_neighborhood": 0.18,
                    "reject_relax": 0.40,
                    "siamese_quantile": 0.15,
                    "siamese_threshold": 0.64,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.16,
                    "min_old_surrogate_reject_delta": -0.06,
                    "min_energy_delta": -20.0,
                    "min_mahalanobis_delta": -50.0,
                    "min_accept_delta": -35.0,
                    "min_old_support_anchor_margin": 0.020,
                    "min_veto_failures": 4,
                    "old_retention_quantile": 0.82,
                    "description": "K5 seen-new rescue with late coupled risk veto and stronger soft prototype mixture.",
                },
                {
                    "slot": "E",
                    "stage": "mse_subspace",
                    "eval_protocol": "ftrc",
                    "k_old": 10,
                    "k_new": 0,
                    "steps": 40,
                    "unknown_threshold": 0.92,
                    "openmax_quantile": 0.99,
                    "openmax_min_threshold": 0.08,
                    "source_ce": 0.28,
                    "unknown_moat": 0.18,
                    "unknown_margin": 0.50,
                    "boundary_samples": 5,
                    "boundary_offset": 0.16,
                    "soft_proto": 0.16,
                    "soft_proto_boundary": 0.22,
                    "soft_proto_boundary_margin": 0.16,
                    "support_contrast": 0.16,
                    "old_bridge": 0.20,
                    "old_neighborhood": 0.24,
                    "reject_relax": 0.36,
                    "siamese_quantile": 0.10,
                    "siamese_threshold": 0.66,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.12,
                    "min_old_surrogate_reject_delta": -0.02,
                    "min_energy_delta": -10.0,
                    "min_mahalanobis_delta": -35.0,
                    "min_accept_delta": -25.0,
                    "min_old_support_anchor_margin": 0.025,
                    "min_veto_failures": 4,
                    "old_unknown_acceptance_guard": True,
                    "guard_min_old_support_evidence_delta": -0.20,
                    "guard_min_old_surrogate_reject_delta": -0.10,
                    "guard_min_energy_delta": -20.0,
                    "guard_min_mahalanobis_delta": -60.0,
                    "guard_min_accept_delta": -45.0,
                    "guard_min_old_support_anchor_margin": 0.010,
                    "guard_min_best_old_score": -3.0,
                    "guard_min_margin": 0.10,
                    "guard_min_failures": 6,
                    "old_retention_quantile": 0.80,
                    "description": "K10 old-retention saturation probe with very late post-accept guard to test whether more target-old support separates old from unknown.",
                },
                {
                    "slot": "F",
                    "stage": "oa_mse_head",
                    "eval_protocol": "sfe",
                    "k_old": 10,
                    "k_new": 10,
                    "steps": 40,
                    "unknown_threshold": 0.96,
                    "openmax_quantile": 1.0,
                    "openmax_min_threshold": 0.10,
                    "source_ce": 0.24,
                    "unknown_moat": 0.16,
                    "unknown_margin": 0.48,
                    "boundary_samples": 5,
                    "boundary_offset": 0.18,
                    "soft_proto": 0.18,
                    "soft_proto_boundary": 0.24,
                    "soft_proto_boundary_margin": 0.18,
                    "support_contrast": 0.14,
                    "old_bridge": 0.18,
                    "old_neighborhood": 0.20,
                    "reject_relax": 0.42,
                    "siamese_quantile": 0.15,
                    "siamese_threshold": 0.68,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.10,
                    "min_old_surrogate_reject_delta": 0.00,
                    "min_energy_delta": -5.0,
                    "min_mahalanobis_delta": -25.0,
                    "min_accept_delta": -20.0,
                    "min_old_support_anchor_margin": 0.030,
                    "min_veto_failures": 4,
                    "old_unknown_acceptance_guard": True,
                    "guard_min_old_support_evidence_delta": -0.18,
                    "guard_min_old_surrogate_reject_delta": -0.08,
                    "guard_min_energy_delta": -15.0,
                    "guard_min_mahalanobis_delta": -50.0,
                    "guard_min_accept_delta": -35.0,
                    "guard_min_old_support_anchor_margin": 0.015,
                    "guard_min_best_old_score": -2.5,
                    "guard_min_margin": 0.10,
                    "guard_min_failures": 6,
                    "old_retention_quantile": 0.80,
                    "description": "K10 old+seen-new saturation probe with soft prototypes and very late unknown guard.",
                },
                {
                    "slot": "G",
                    "stage": "mse_subspace",
                    "eval_protocol": "ftrc",
                    "k_old": 20,
                    "k_new": 0,
                    "steps": 40,
                    "unknown_threshold": 0.93,
                    "openmax_quantile": 0.995,
                    "openmax_min_threshold": 0.10,
                    "source_ce": 0.30,
                    "unknown_moat": 0.20,
                    "unknown_margin": 0.52,
                    "boundary_samples": 6,
                    "boundary_offset": 0.16,
                    "soft_proto": 0.18,
                    "soft_proto_boundary": 0.24,
                    "soft_proto_boundary_margin": 0.18,
                    "support_contrast": 0.18,
                    "old_bridge": 0.22,
                    "old_neighborhood": 0.24,
                    "reject_relax": 0.38,
                    "siamese_quantile": 0.10,
                    "siamese_threshold": 0.70,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.08,
                    "min_old_surrogate_reject_delta": 0.02,
                    "min_energy_delta": 0.0,
                    "min_mahalanobis_delta": -15.0,
                    "min_accept_delta": -10.0,
                    "min_old_support_anchor_margin": 0.035,
                    "min_veto_failures": 4,
                    "old_unknown_acceptance_guard": True,
                    "guard_min_old_support_evidence_delta": -0.12,
                    "guard_min_old_surrogate_reject_delta": -0.02,
                    "guard_min_energy_delta": -5.0,
                    "guard_min_mahalanobis_delta": -25.0,
                    "guard_min_accept_delta": -20.0,
                    "guard_min_old_support_anchor_margin": 0.020,
                    "guard_min_best_old_score": -2.0,
                    "guard_min_margin": 0.20,
                    "guard_min_failures": 5,
                    "old_retention_quantile": 0.78,
                    "description": "K20 higher-shot old-calibration saturation probe: test whether more target-old labels recover query old without full backbone updates.",
                },
                {
                    "slot": "H",
                    "stage": "oa_mse_head",
                    "eval_protocol": "sfe",
                    "k_old": 20,
                    "k_new": 20,
                    "steps": 40,
                    "unknown_threshold": 0.97,
                    "openmax_quantile": 1.0,
                    "openmax_min_threshold": 0.12,
                    "source_ce": 0.26,
                    "unknown_moat": 0.18,
                    "unknown_margin": 0.50,
                    "boundary_samples": 6,
                    "boundary_offset": 0.18,
                    "soft_proto": 0.20,
                    "soft_proto_boundary": 0.26,
                    "soft_proto_boundary_margin": 0.20,
                    "support_contrast": 0.16,
                    "old_bridge": 0.20,
                    "old_neighborhood": 0.22,
                    "reject_relax": 0.44,
                    "siamese_quantile": 0.15,
                    "siamese_threshold": 0.72,
                    "siamese_unknown_veto": True,
                    "siamese_unknown_veto_mode": "coupled",
                    "min_old_support_evidence_delta": -0.06,
                    "min_old_surrogate_reject_delta": 0.04,
                    "min_energy_delta": 5.0,
                    "min_mahalanobis_delta": -10.0,
                    "min_accept_delta": -5.0,
                    "min_old_support_anchor_margin": 0.040,
                    "min_veto_failures": 4,
                    "old_unknown_acceptance_guard": True,
                    "guard_min_old_support_evidence_delta": -0.10,
                    "guard_min_old_surrogate_reject_delta": 0.00,
                    "guard_min_energy_delta": 0.0,
                    "guard_min_mahalanobis_delta": -20.0,
                    "guard_min_accept_delta": -15.0,
                    "guard_min_old_support_anchor_margin": 0.025,
                    "guard_min_best_old_score": -1.5,
                    "guard_min_margin": 0.25,
                    "guard_min_failures": 5,
                    "old_retention_quantile": 0.78,
                    "description": "K20 old+seen-new higher-shot saturation probe with explicit output-space balance and late unknown guard.",
                },
            ]
            if plan == "OA_MSE_STRUCT48":
                stage_specs = _oa_mse_struct48_stage_specs()
            if plan == "OA_MSE_SIMPLIFIED48":
                stage_specs = _oa_mse_simplified48_stage_specs()
            if plan == "OA_MSE_RETENTION48":
                stage_specs = _oa_mse_retention48_stage_specs()
            if plan == "OA_MSE_SUPPORTRET48":
                stage_specs = _oa_mse_supportret48_stage_specs()
            if plan == "OA_MSE_TWOBRANCH48":
                stage_specs = _oa_mse_twobranch48_stage_specs()
            if plan == "OA_MSE_REGHEAD48":
                stage_specs = _oa_mse_reghead48_stage_specs()
            if plan == "OA_MSE_GEOM48":
                stage_specs = _oa_mse_geom48_stage_specs()
            if plan == "OA_MSE_TRIAGE48":
                stage_specs = _oa_mse_triage48_stage_specs()
            if plan == "OA_MSE_LOOO48":
                stage_specs = _oa_mse_looo48_stage_specs()
            if plan == "OA_MSE_CONSTRAIN48":
                stage_specs = _oa_mse_constrain48_stage_specs()
            if plan == "OA_MSE_ENVELOPE48":
                stage_specs = _oa_mse_envelope48_stage_specs()
            if plan == "OA_MSE_RESCUE48":
                stage_specs = _oa_mse_rescue48_stage_specs()
            if plan == "OA_MSE_PREREJECT48":
                stage_specs = _oa_mse_prereject48_stage_specs()
            if plan == "OA_MSE_THREEWAY48":
                stage_specs = _oa_mse_threeway48_stage_specs()
            if plan == "OA_MSE_COVFLOOR48":
                stage_specs = _oa_mse_covfloor48_stage_specs()
            if plan == "OA_MSE_CLASSFIRST48":
                stage_specs = _oa_mse_classfirst48_stage_specs()
            if plan == "OA_MSE_EVIBG48":
                stage_specs = _oa_mse_evibg48_stage_specs()
            if plan == "OA_MSE_SOFTTARGET48":
                stage_specs = _oa_mse_softtarget48_stage_specs()
            if plan == "OA_MSE_NEGANCHOR48":
                stage_specs = _oa_mse_neganchor48_stage_specs()
            if plan == "OA_MSE_DENSHELL48":
                stage_specs = _oa_mse_denshell48_stage_specs()
            if plan == "OA_MSE_IDCONS48":
                stage_specs = _oa_mse_idcons48_stage_specs()
            if plan == "OA_MSE_CONFORM48":
                stage_specs = _oa_mse_conform48_stage_specs()
            if plan == "OA_MSE_RECON48":
                stage_specs = _oa_mse_recon48_stage_specs()
            if plan == "OA_MSE_SOURCERISK48":
                stage_specs = _oa_mse_sourcerisk48_stage_specs()
            if plan == "OA_MSE_KRET48":
                stage_specs = _oa_mse_kret48_stage_specs()
            if plan == "OA_MSE_RISKRET48":
                stage_specs = _oa_mse_riskret48_stage_specs()
            if plan == "OA_MSE_MANIFOLD48":
                stage_specs = _oa_mse_manifold48_stage_specs()
            if plan == "OA_MSE_H06_EVID48":
                stage_specs = _oa_mse_h06_evid48_stage_specs()
            if plan == "OA_MSE_H06_ARB48":
                stage_specs = _oa_mse_h06_arb48_stage_specs()
            if plan == "OA_MSE_H06_OLDUNK48":
                stage_specs = _oa_mse_h06_oldunk48_stage_specs()
            if plan == "OA_MSE_H06_BGTRAIN48":
                stage_specs = _oa_mse_h06_bgtrain48_stage_specs()
            if plan in {"OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48"}:
                stage_specs = _oa_mse_h06_retold48_stage_specs()
            if plan == "OA_MSE_H06_OLDRELAX48":
                stage_specs = _oa_mse_h06_oldrelax48_stage_specs()
            if plan == "OA_MSE_H06_OLDGEOM48":
                stage_specs = _oa_mse_h06_oldgeom48_stage_specs()
            if plan == "OA_MSE_H06_OLDCONF48":
                stage_specs = _oa_mse_h06_oldconf48_stage_specs()
            if plan == "OA_MSE_H06_OLDBUDGET48":
                stage_specs = _oa_mse_h06_oldbudget48_stage_specs()
            if plan == "OA_MSE_H06_OLDQUAL48":
                stage_specs = _oa_mse_h06_oldqual48_stage_specs()
            if plan == "OA_MSE_H06_OLDRISK48":
                stage_specs = _oa_mse_h06_oldrisk48_stage_specs()
            if plan == "OA_MSE_H06_OLDFUSE48":
                stage_specs = _oa_mse_h06_oldfuse48_stage_specs()
            if plan == "OA_MSE_H06_ROLLSAFE48":
                stage_specs = _oa_mse_h06_rollsafe48_stage_specs()
            if plan == "OA_MSE_H06_OLDHEAD48":
                stage_specs = _oa_mse_h06_oldhead48_stage_specs()
            if plan == "OA_MSE_H06_OLDHEADFAR48":
                stage_specs = _oa_mse_h06_oldheadfar48_stage_specs()
            if plan == "OA_MSE_H06_OLDRECOV48":
                stage_specs = _oa_mse_h06_oldrecov48_stage_specs()
            elif plan == "OA_MSE_BGCAP48":
                stage_specs = _oa_mse_bgcap48_stage_specs()
            elif plan == "OA_MSE_SUPPORTCV48":
                stage_specs = _oa_mse_supportcv48_stage_specs()
            if plan in {"OA_MSE_SOFTVOID128", "OA_MSE_ANCHORGUARD128", "OA_MSE_MIXHEAD128"}:
                extended_specs = []
                for rep, suffix in enumerate(("I", "J", "K", "L", "M", "N", "O", "P")):
                    spec = dict(stage_specs[rep])
                    spec["slot"] = suffix
                    spec["seed_offset"] = 100 + rep
                    spec["steps"] = min(56, int(spec["steps"]) + (8 if rep >= 4 else 4))
                    spec["unknown_threshold"] = min(0.99, float(spec["unknown_threshold"]) + 0.01)
                    spec["openmax_quantile"] = min(1.0, float(spec["openmax_quantile"]) + 0.002)
                    spec["soft_proto"] = min(0.24, float(spec["soft_proto"]) + 0.03)
                    spec["soft_proto_boundary"] = min(0.24, float(spec.get("soft_proto_boundary", 0.0)) + 0.02)
                    spec["support_contrast"] = min(0.24, float(spec["support_contrast"]) + 0.02)
                    spec["old_bridge"] = min(0.24, float(spec["old_bridge"]) + 0.02)
                    spec["old_neighborhood"] = min(0.26, float(spec["old_neighborhood"]) + 0.02)
                    spec["old_retention_quantile"] = max(0.76, float(spec.get("old_retention_quantile", 0.82)) - 0.02)
                    spec["description"] = (
                        "High-occupancy soft-void OA-MSE: combine soft prototype mixture boundaries "
                        "with a weak pseudo-unknown background competitor, preserving old-class retention first."
                    )
                    extended_specs.append(spec)
                stage_specs.extend(extended_specs)
            if plan == "OA_MSE_ANCHORGUARD128":
                for idx, spec in enumerate(stage_specs):
                    k_old = int(spec["k_old"])
                    late = idx >= 8
                    strictness = idx % 8
                    spec["description"] = (
                        "Anchor-guard OA-MSE: preserve target-old support geometry with source-old replay, "
                        "then reject accepted old-like outputs only when multiple query-free old/unknown diagnostics fail."
                    )
                    spec["steps"] = min(48, max(20, int(spec["steps"]) - (4 if k_old <= 5 else 0) + (4 if late else 0)))
                    spec["source_ce"] = min(0.40, float(spec["source_ce"]) + 0.08)
                    spec["unknown_moat"] = min(0.36, max(float(spec["unknown_moat"]) + 0.04, 0.24 + 0.01 * strictness))
                    spec["unknown_margin"] = min(0.68, max(float(spec["unknown_margin"]) + 0.04, 0.56 + 0.01 * strictness))
                    spec["soft_proto"] = min(0.22, max(float(spec["soft_proto"]), 0.14 + 0.01 * (strictness % 4)))
                    spec["soft_proto_boundary"] = min(0.16, max(float(spec.get("soft_proto_boundary", 0.0)), 0.08 + 0.01 * (strictness % 4)))
                    spec["support_contrast"] = min(0.22, float(spec["support_contrast"]) + 0.04)
                    spec["old_bridge"] = min(0.28, float(spec["old_bridge"]) + 0.06)
                    spec["old_neighborhood"] = min(0.30, float(spec["old_neighborhood"]) + 0.06)
                    spec["reject_relax"] = 0.32 if k_old <= 5 else 0.28
                    spec["old_retention_quantile"] = 0.82 if k_old <= 5 else 0.80
                    spec["siamese_unknown_veto"] = True
                    spec["siamese_unknown_veto_mode"] = "coupled"
                    spec["siamese_threshold"] = min(0.76, max(float(spec.get("siamese_threshold", 0.60)), 0.62 + 0.015 * strictness))
                    spec["min_old_support_evidence_delta"] = -0.06 + 0.02 * (strictness % 4)
                    spec["min_old_surrogate_reject_delta"] = -0.02 + 0.02 * (strictness % 4)
                    spec["min_energy_delta"] = -4.0 + 2.0 * (strictness % 4)
                    spec["min_mahalanobis_delta"] = -20.0 + 5.0 * (strictness % 4)
                    spec["min_accept_delta"] = -10.0 + 2.5 * (strictness % 4)
                    spec["min_old_support_anchor_margin"] = 0.020 + 0.010 * (strictness % 4)
                    spec["min_veto_failures"] = 4 if strictness < 4 else 3
                    spec["old_unknown_acceptance_guard"] = True
                    spec["guard_min_old_support_evidence_delta"] = -0.02 + 0.02 * (strictness % 4)
                    spec["guard_min_old_surrogate_reject_delta"] = 0.00 + 0.02 * (strictness % 4)
                    spec["guard_min_energy_delta"] = -2.0 + 2.0 * (strictness % 4)
                    spec["guard_min_mahalanobis_delta"] = -15.0 + 5.0 * (strictness % 4)
                    spec["guard_min_accept_delta"] = -8.0 + 2.0 * (strictness % 4)
                    spec["guard_min_old_support_anchor_margin"] = 0.020 + 0.010 * (strictness % 4)
                    spec["guard_min_best_old_score"] = -1.0 + 0.25 * (strictness % 4)
                    spec["guard_min_margin"] = 0.08 + 0.04 * (strictness % 4)
                    spec["guard_min_failures"] = 4 if strictness < 4 else 3
                    spec["void_background"] = 0.03 + 0.01 * strictness
                    spec["void_gate"] = True
                    spec["void_gate_min_score"] = 0.60 + 0.015 * strictness
                    spec["void_gate_min_margin"] = -0.02 + 0.02 * (strictness % 4)
            if plan == "OA_MSE_MIXHEAD128":
                for idx, spec in enumerate(stage_specs):
                    k_old = int(spec["k_old"])
                    strictness = idx % 8
                    spec["description"] = (
                        "Multi-prototype score-head OA-MSE: score each class with a query-conditioned soft mixture "
                        "of same-class support anchors, preserving old-query accuracy before raising rejection strictness."
                    )
                    spec["steps"] = min(52, max(24, int(spec["steps"]) + (4 if idx >= 8 else 0)))
                    spec["source_ce"] = min(0.36, float(spec["source_ce"]) + 0.06)
                    spec["unknown_moat"] = min(0.30, max(float(spec["unknown_moat"]), 0.18 + 0.01 * strictness))
                    spec["unknown_margin"] = min(0.58, max(float(spec["unknown_margin"]), 0.46 + 0.01 * strictness))
                    spec["soft_proto"] = min(0.24, max(float(spec["soft_proto"]), 0.16 + 0.01 * (strictness % 4)))
                    spec["soft_proto_boundary"] = min(0.18, max(float(spec.get("soft_proto_boundary", 0.0)), 0.10 + 0.01 * (strictness % 4)))
                    spec["support_contrast"] = min(0.20, float(spec["support_contrast"]) + 0.03)
                    spec["old_bridge"] = min(0.26, float(spec["old_bridge"]) + 0.05)
                    spec["old_neighborhood"] = min(0.28, float(spec["old_neighborhood"]) + 0.05)
                    spec["reject_relax"] = 0.38 if k_old <= 5 else 0.34
                    spec["old_retention_quantile"] = 0.78 if k_old <= 5 else 0.76
                    spec["multiproto_score"] = True
                    spec["multiproto_topk"] = 2 if k_old <= 5 else 3
                    spec["multiproto_temperature"] = 0.08 + 0.01 * (strictness % 3)
                    spec["multiproto_score_weight"] = (0.55, 0.70, 0.85, 1.00)[strictness % 4]
                    spec["siamese_unknown_veto"] = strictness >= 4
                    spec["siamese_unknown_veto_mode"] = "coupled"
                    spec["siamese_threshold"] = min(0.72, max(float(spec.get("siamese_threshold", 0.60)), 0.58 + 0.015 * strictness))
                    spec["min_old_support_evidence_delta"] = -0.14 + 0.02 * (strictness % 4)
                    spec["min_old_surrogate_reject_delta"] = -0.08 + 0.02 * (strictness % 4)
                    spec["min_energy_delta"] = -10.0 + 2.0 * (strictness % 4)
                    spec["min_mahalanobis_delta"] = -35.0 + 5.0 * (strictness % 4)
                    spec["min_accept_delta"] = -24.0 + 4.0 * (strictness % 4)
                    spec["min_old_support_anchor_margin"] = 0.010 + 0.005 * (strictness % 4)
                    spec["min_veto_failures"] = 5 if strictness < 4 else 4
                    spec["old_unknown_acceptance_guard"] = strictness >= 4
                    spec["guard_min_old_support_evidence_delta"] = -0.10 + 0.02 * (strictness % 4)
                    spec["guard_min_old_surrogate_reject_delta"] = -0.06 + 0.02 * (strictness % 4)
                    spec["guard_min_energy_delta"] = -8.0 + 2.0 * (strictness % 4)
                    spec["guard_min_mahalanobis_delta"] = -30.0 + 5.0 * (strictness % 4)
                    spec["guard_min_accept_delta"] = -20.0 + 4.0 * (strictness % 4)
                    spec["guard_min_old_support_anchor_margin"] = 0.010 + 0.005 * (strictness % 4)
                    spec["guard_min_best_old_score"] = -2.0 + 0.25 * (strictness % 4)
                    spec["guard_min_margin"] = 0.04 + 0.03 * (strictness % 4)
                    spec["guard_min_failures"] = 5 if strictness < 4 else 4
                    spec["void_background"] = 0.02 + 0.005 * strictness
                    spec["void_gate"] = strictness >= 4
                    spec["void_gate_min_score"] = 0.54 + 0.015 * strictness
                    spec["void_gate_min_margin"] = -0.05 + 0.015 * (strictness % 4)
            if plan == "OA_MSE_BALANCE64":
                for spec in stage_specs:
                    spec["soft_proto_boundary"] = 0.0
                    spec["soft_proto_boundary_margin"] = 0.15
            elif plan not in {"OA_MSE_STRUCT48", "OA_MSE_SIMPLIFIED48", "OA_MSE_RETENTION48", "OA_MSE_SUPPORTRET48", "OA_MSE_TWOBRANCH48", "OA_MSE_REGHEAD48", "OA_MSE_LOOO48", "OA_MSE_CONSTRAIN48", "OA_MSE_ENVELOPE48", "OA_MSE_RESCUE48", "OA_MSE_PREREJECT48", "OA_MSE_THREEWAY48", "OA_MSE_COVFLOOR48", "OA_MSE_CLASSFIRST48", "OA_MSE_EVIBG48", "OA_MSE_SOFTTARGET48", "OA_MSE_NEGANCHOR48", "OA_MSE_DENSHELL48", "OA_MSE_IDCONS48", "OA_MSE_CONFORM48", "OA_MSE_RECON48", "OA_MSE_SOURCERISK48", "OA_MSE_SUPPORTCV48", "OA_MSE_BGCAP48", "OA_MSE_KRET48", "OA_MSE_RISKRET48", "OA_MSE_MANIFOLD48", "OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"}:
                for spec in stage_specs:
                    spec["description"] = (
                        "Soft-mix prototype boundary OA-MSE: train the adapter toward convex same-class "
                        "prototype mixtures while separating other-class anchors and keeping unknown query eval-only."
                    )
                    spec["soft_proto"] = min(0.30, float(spec["soft_proto"]) + 0.06)
                    spec["support_contrast"] = min(0.24, float(spec["support_contrast"]) + 0.04)
        if plan in {
            "OA_MSE_BOUNDARY32",
            "OA_MSE_UNCERTAIN32",
            "OA_MSE_VETO32",
            "OA_MSE_CLASSCOND32",
            "OA_MSE_CALGUARD32",
            "OA_MSE_BALANCE64",
            "OA_MSE_SOFTMIX64",
            "OA_MSE_VOID64",
            "OA_MSE_SOFTVOID128",
            "OA_MSE_ANCHORGUARD128",
            "OA_MSE_MIXHEAD128",
            "OA_MSE_SIMPLIFIED48",
            "OA_MSE_RETENTION48",
            "OA_MSE_SUPPORTRET48",
            "OA_MSE_TWOBRANCH48",
            "OA_MSE_REGHEAD48",
        }:
            for spec in stage_specs:
                if plan not in {"OA_MSE_BALANCE64", "OA_MSE_MIXHEAD128", "OA_MSE_STRUCT48", "OA_MSE_SIMPLIFIED48", "OA_MSE_RETENTION48", "OA_MSE_SUPPORTRET48", "OA_MSE_TWOBRANCH48", "OA_MSE_REGHEAD48", "OA_MSE_LOOO48", "OA_MSE_CONSTRAIN48", "OA_MSE_ENVELOPE48", "OA_MSE_RESCUE48", "OA_MSE_PREREJECT48", "OA_MSE_THREEWAY48", "OA_MSE_COVFLOOR48", "OA_MSE_CLASSFIRST48", "OA_MSE_EVIBG48", "OA_MSE_SOFTTARGET48", "OA_MSE_NEGANCHOR48", "OA_MSE_DENSHELL48", "OA_MSE_IDCONS48", "OA_MSE_CONFORM48", "OA_MSE_RECON48", "OA_MSE_SOURCERISK48", "OA_MSE_SUPPORTCV48", "OA_MSE_BGCAP48", "OA_MSE_KRET48", "OA_MSE_RISKRET48", "OA_MSE_MANIFOLD48", "OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48"}:
                    spec["description"] = (
                        "Boundary-guard alpha selection with stronger class-constrained soft prototype mixture, "
                        "old-retention anchors, and protocol-safe pseudo-unknown pressure."
                    )
                    spec["boundary_samples"] = int(spec["boundary_samples"]) + 2
                    spec["unknown_moat"] = min(0.32, float(spec["unknown_moat"]) + 0.06)
                    spec["unknown_margin"] = min(0.66, float(spec["unknown_margin"]) + 0.06)
                    spec["soft_proto"] = min(0.20, float(spec["soft_proto"]) + 0.06)
                    spec["support_contrast"] = min(0.24, float(spec["support_contrast"]) + 0.04)
                    spec["old_bridge"] = min(0.22, float(spec["old_bridge"]) + 0.04)
                    spec["old_neighborhood"] = min(0.24, float(spec["old_neighborhood"]) + 0.04)
        if plan in {"OA_MSE_UNCERTAIN32", "OA_MSE_VETO32", "OA_MSE_CLASSCOND32", "OA_MSE_CALGUARD32"}:
            for idx, spec in enumerate(stage_specs):
                spec["description"] = (
                    "Old-surrogate uncertain-band repair: keep query-free surrogate calibration, "
                    "but route middle-band old evidence to the support-only Siamese verifier instead of hard reject."
                )
                spec["reject_relax"] = (0.18, 0.22, 0.26, 0.30)[idx]
                spec["siamese_quantile"] = (0.10, 0.15, 0.10, 0.15)[idx]
                spec["siamese_threshold"] = (0.62, 0.68, 0.72, 0.78)[idx]
                spec["old_retention_quantile"] = (0.92, 0.92, 0.90, 0.90)[idx]
        if plan == "OA_MSE_VETO32":
            veto_specs = (
                (-0.04, 0.02, -0.20, -60.0, -30.0, 0.68),
                (0.00, 0.05, -0.10, -40.0, -20.0, 0.72),
                (0.04, 0.08, 0.00, -20.0, -10.0, 0.76),
                (0.08, 0.12, 0.02, 0.0, 0.0, 0.80),
            )
            for idx, spec in enumerate(stage_specs):
                (
                    support_delta,
                    surrogate_delta,
                    energy_delta,
                    mahalanobis_delta,
                    accept_delta,
                    siamese_threshold,
                ) = veto_specs[idx]
                spec["description"] = (
                    "Unknown-risk veto OA-MSE repair: preserve uncertain-band old recovery, "
                    "but reject Siamese-verified rows when old evidence/risk diagnostics remain weak."
                )
                spec["siamese_unknown_veto"] = True
                spec["siamese_threshold"] = siamese_threshold
                spec["min_old_support_evidence_delta"] = support_delta
                spec["min_old_surrogate_reject_delta"] = surrogate_delta
                spec["min_energy_delta"] = energy_delta
                spec["min_mahalanobis_delta"] = mahalanobis_delta
                spec["min_accept_delta"] = accept_delta
        if plan == "OA_MSE_CLASSCOND32":
            coupled_specs = (
                (-0.20, -0.10, -0.50, -100.0, -50.0, 0.02, 3, 0.66),
                (-0.15, -0.05, -0.35, -80.0, -40.0, 0.04, 3, 0.70),
                (-0.10, 0.00, -0.20, -60.0, -30.0, 0.06, 2, 0.74),
                (-0.05, 0.02, -0.10, -40.0, -20.0, 0.08, 2, 0.78),
            )
            for idx, spec in enumerate(stage_specs):
                (
                    support_delta,
                    surrogate_delta,
                    energy_delta,
                    mahalanobis_delta,
                    accept_delta,
                    anchor_margin,
                    min_failures,
                    siamese_threshold,
                ) = coupled_specs[idx]
                spec["description"] = (
                    "Class-conditional OA-MSE repair: keep uncertain-band old recovery, "
                    "but reject Siamese rows only when support-anchor class margin is weak and multiple risk diagnostics fail."
                )
                spec["siamese_unknown_veto"] = True
                spec["siamese_unknown_veto_mode"] = "coupled"
                spec["siamese_threshold"] = siamese_threshold
                spec["min_old_support_evidence_delta"] = support_delta
                spec["min_old_surrogate_reject_delta"] = surrogate_delta
                spec["min_energy_delta"] = energy_delta
                spec["min_mahalanobis_delta"] = mahalanobis_delta
                spec["min_accept_delta"] = accept_delta
                spec["min_old_support_anchor_margin"] = anchor_margin
                spec["min_veto_failures"] = min_failures
        if plan == "OA_MSE_CALGUARD32":
            guard_specs = (
                (-0.08, 0.05, 40.0, -20.0, 0.00, 0.02, -0.50, 0.50, 3, 0.70),
                (-0.02, 0.10, 45.0, 0.0, 0.01, 0.04, 0.00, 1.00, 3, 0.74),
                (0.04, 0.16, 50.0, 10.0, 0.02, 0.06, 0.20, 2.00, 2, 0.78),
                (0.10, 0.22, 55.0, 20.0, 0.03, 0.08, 0.40, 3.00, 2, 0.82),
            )
            for idx, spec in enumerate(stage_specs):
                (
                    support_delta,
                    surrogate_delta,
                    energy_delta,
                    mahalanobis_delta,
                    accept_delta,
                    anchor_margin,
                    best_old_score,
                    margin,
                    min_failures,
                    siamese_threshold,
                ) = guard_specs[idx]
                spec["description"] = (
                    "Calibrated old/unknown guard OA-MSE repair: keep support-only adaptation, "
                    "then re-check accepted old-like outputs with query-free old evidence before online update."
                )
                spec["siamese_unknown_veto"] = True
                spec["siamese_unknown_veto_mode"] = "any"
                spec["siamese_threshold"] = siamese_threshold
                spec["min_old_support_evidence_delta"] = support_delta
                spec["min_old_surrogate_reject_delta"] = surrogate_delta
                spec["min_energy_delta"] = energy_delta
                spec["min_mahalanobis_delta"] = mahalanobis_delta
                spec["min_accept_delta"] = accept_delta
                spec["old_unknown_acceptance_guard"] = True
                spec["guard_min_old_support_evidence_delta"] = support_delta
                spec["guard_min_old_surrogate_reject_delta"] = surrogate_delta
                spec["guard_min_energy_delta"] = energy_delta
                spec["guard_min_mahalanobis_delta"] = mahalanobis_delta
                spec["guard_min_accept_delta"] = accept_delta
                spec["guard_min_old_support_anchor_margin"] = anchor_margin
                spec["guard_min_best_old_score"] = best_old_score
                spec["guard_min_margin"] = margin
                spec["guard_min_failures"] = min_failures
        if plan in {"OA_MSE_VOID64", "OA_MSE_SOFTVOID128"}:
            void_specs = [
                (0.08, 0.50, -0.08),
                (0.12, 0.54, -0.04),
                (0.16, 0.58, 0.00),
                (0.22, 0.62, 0.04),
                (0.28, 0.66, 0.08),
                (0.34, 0.70, 0.12),
                (0.40, 0.74, 0.16),
                (0.48, 0.78, 0.20),
            ]
            if plan == "OA_MSE_SOFTVOID128":
                void_specs.extend(
                    [
                        (0.05, 0.50, -0.12),
                        (0.07, 0.53, -0.08),
                        (0.09, 0.56, -0.04),
                        (0.11, 0.59, 0.00),
                        (0.13, 0.62, 0.04),
                        (0.15, 0.65, 0.08),
                        (0.17, 0.68, 0.12),
                        (0.20, 0.71, 0.16),
                    ]
                )
            for idx, spec in enumerate(stage_specs):
                void_weight, void_score, void_margin = void_specs[idx]
                spec["description"] = (
                    "Soft-void OA-MSE repair: train pseudo-unknown as a weak background competitor, "
                    "keep soft prototype boundaries, and apply a query-label-free void gate after known-class acceptance."
                )
                if plan == "OA_MSE_VOID64":
                    spec["soft_proto"] = min(0.16, float(spec["soft_proto"]))
                    spec["soft_proto_boundary"] = min(0.06, float(spec.get("soft_proto_boundary", 0.0)))
                spec["unknown_moat"] = min(0.42, max(float(spec["unknown_moat"]), 0.22 + 0.02 * idx))
                spec["unknown_margin"] = min(0.72, max(float(spec["unknown_margin"]), 0.54 + 0.02 * idx))
                spec["void_background"] = float(void_weight)
                spec["void_gate"] = True
                spec["void_gate_min_score"] = float(void_score)
                spec["void_gate_min_margin"] = float(void_margin)
        for gpu in range(8):
            for spec_idx, spec in enumerate(stage_specs):
                is_seen_new = spec["stage"] == "oa_mse_head"
                k_old = int(spec["k_old"])
                k_new = int(spec["k_new"])
                seed = 98100 + gpu * 10 + spec_idx
                slot = str(spec["slot"])
                candidate_kwargs = dict(base)
                receiver_spec = (
                    _receiver_spec_for_candidate(gpu * len(stage_specs) + spec_idx)
                    if plan in {"OA_MSE_GEOM48", "OA_MSE_TRIAGE48", "OA_MSE_CONSTRAIN48", "OA_MSE_ENVELOPE48", "OA_MSE_RESCUE48", "OA_MSE_PREREJECT48", "OA_MSE_THREEWAY48", "OA_MSE_COVFLOOR48", "OA_MSE_CLASSFIRST48", "OA_MSE_EVIBG48", "OA_MSE_SOFTTARGET48", "OA_MSE_NEGANCHOR48", "OA_MSE_DENSHELL48", "OA_MSE_IDCONS48", "OA_MSE_CONFORM48", "OA_MSE_RECON48", "OA_MSE_SOURCERISK48", "OA_MSE_SUPPORTCV48", "OA_MSE_BGCAP48", "OA_MSE_KRET48", "OA_MSE_RISKRET48", "OA_MSE_MANIFOLD48"}
                    else {
                        "label": PHASE2_TARGET_RECEIVER_LABEL,
                        "manysig_rx_index": PHASE2_MANYSIG_TARGET_RX_INDEX,
                        "manytx_rx_index": PHASE2_MANYTX_TARGET_RX_INDEX,
                    }
                )
                candidate_kwargs.update(
                    {
                        "target_receiver_ids": str(receiver_spec["label"]),
                        "target_receiver_label": str(receiver_spec["label"]),
                        "manysig_target_rx_index": str(receiver_spec["manysig_rx_index"]),
                        "manytx_target_rx_index": str(receiver_spec["manytx_rx_index"]),
                        "max_adapt_steps": int(spec["steps"]),
                        "unknown_threshold": float(spec["unknown_threshold"]),
                        "openmax_quantile": float(spec["openmax_quantile"]),
                        "openmax_min_threshold": float(spec["openmax_min_threshold"]),
                        "oa_mse_source_ce_weight": float(spec["source_ce"]),
                        "oa_mse_unknown_moat_weight": float(spec["unknown_moat"]),
                        "oa_mse_unknown_moat_margin": float(spec["unknown_margin"]),
                        "pseudo_unknown_source_boundary_samples_per_pair": int(spec["boundary_samples"]),
                        "pseudo_unknown_source_boundary_offset_scale": float(spec["boundary_offset"]),
                        "pseudo_unknown_target_shift_samples_per_class": 4 if k_old <= 2 else 6,
                        "pseudo_unknown_target_shift_offset_scale": 0.24 if k_old <= 2 else 0.22,
                        "pseudo_unknown_target_halo_samples_per_class": int(
                            spec.get("pseudo_halo_override", 4 if k_old <= 2 else 6)
                        ),
                        "pseudo_unknown_target_halo_offset_scale": 0.35 if is_seen_new else 0.32,
                        "pseudo_unknown_target_ring_samples_per_class": int(
                            spec.get("pseudo_ring_override", 4 if k_old <= 2 else 6)
                        ),
                        "pseudo_unknown_target_ring_offset_scale": 0.45 if is_seen_new else 0.38,
                        "oa_mse_support_contrast_weight": float(spec["support_contrast"]),
                        "oa_mse_support_center_ce_weight": float(spec.get("support_center_ce", 0.0)),
                        "support_center_temperature": float(spec.get("support_center_temperature", 0.10)),
                        "support_center_margin": float(spec.get("support_center_margin", 0.10)),
                        "oa_mse_soft_proto_weight": float(spec["soft_proto"]),
                        "soft_proto_topk": int(spec.get("soft_proto_topk", 2)),
                        "soft_proto_temperature": float(spec.get("soft_proto_temperature", 0.10)),
                        "oa_mse_soft_proto_boundary_weight": float(spec.get("soft_proto_boundary", 0.0)),
                        "soft_proto_boundary_margin": float(spec.get("soft_proto_boundary_margin", 0.15)),
                        "oa_mse_multiproto_score": bool(spec.get("multiproto_score", False)),
                        "multiproto_topk": int(spec.get("multiproto_topk", 2 if k_old <= 5 else 3)),
                        "multiproto_temperature": float(spec.get("multiproto_temperature", 0.10)),
                        "multiproto_score_weight": float(spec.get("multiproto_score_weight", 1.0)),
                        "oa_mse_mixture_consistency_gate": bool(spec.get("mixture_consistency_gate", False)),
                        "mixture_consistency_min_cos": float(spec.get("mixture_consistency_min_cos", -1.0)),
                        "mixture_consistency_max_residual": float(spec.get("mixture_consistency_max_residual", 1.0e6)),
                        "mixture_consistency_min_margin": float(spec.get("mixture_consistency_min_margin", -1.0e6)),
                        "mixture_consistency_action": str(spec.get("mixture_consistency_action", "uncertain")),
                        "optimization_category": str(spec.get("category", "conservative")),
                        "oa_mse_adapter_kind": str(spec.get("adapter_kind", "low_rank")),
                        "oa_mse_anchor_density_gate": bool(spec.get("anchor_density_gate", False)),
                        "anchor_density_topk": int(spec.get("anchor_density_topk", 3)),
                        "anchor_density_temperature": float(spec.get("anchor_density_temperature", 0.08)),
                        "anchor_density_min_quantile": float(spec.get("anchor_density_quantile", 0.05)),
                        "anchor_density_margin_quantile": float(spec.get("anchor_density_margin_quantile", 0.05)),
                        "anchor_density_gate_action": str(spec.get("anchor_density_action", "uncertain")),
                        "oa_mse_class_envelope_gate": bool(spec.get("class_envelope_gate", False)),
                        "class_envelope_evidence_quantile": float(spec.get("class_envelope_evidence_quantile", 0.05)),
                        "class_envelope_residual_quantile": float(spec.get("class_envelope_residual_quantile", 0.95)),
                        "class_envelope_score_quantile": float(spec.get("class_envelope_score_quantile", 0.05)),
                        "class_envelope_margin_quantile": float(spec.get("class_envelope_margin_quantile", 0.05)),
                        "class_envelope_evidence_slack": float(spec.get("class_envelope_evidence_slack", 0.02)),
                        "class_envelope_residual_slack": float(spec.get("class_envelope_residual_slack", 0.02)),
                        "class_envelope_score_slack": float(spec.get("class_envelope_score_slack", 0.05)),
                        "class_envelope_margin_slack": float(spec.get("class_envelope_margin_slack", 0.02)),
                        "class_envelope_min_failures": int(spec.get("class_envelope_min_failures", 1)),
                        "class_envelope_gate_action": str(spec.get("class_envelope_gate_action", "reject")),
                        "oa_mse_old_primary_gate": bool(spec.get("old_primary_gate", False)),
                        "old_primary_min_old_support_evidence_delta": float(
                            spec.get("old_primary_min_old_support_evidence_delta", 0.0)
                        ),
                        "old_primary_min_old_support_anchor_delta": float(
                            spec.get("old_primary_min_old_support_anchor_delta", -0.02)
                        ),
                        "old_primary_min_old_support_anchor_margin": float(
                            spec.get("old_primary_min_old_support_anchor_margin", 0.0)
                        ),
                        "old_primary_min_score_margin": float(spec.get("old_primary_min_score_margin", 0.0)),
                        "old_primary_require_soft_mixture": bool(spec.get("old_primary_require_soft_mixture", False)),
                        "old_primary_min_soft_mixture_margin": float(
                            spec.get("old_primary_min_soft_mixture_margin", -1.0e6)
                        ),
                        "old_primary_min_soft_mixture_cos": float(spec.get("old_primary_min_soft_mixture_cos", -1.0)),
                        "old_primary_max_soft_mixture_residual": float(
                            spec.get("old_primary_max_soft_mixture_residual", 1.0e6)
                        ),
                        "old_primary_require_support_knn": bool(spec.get("old_primary_require_support_knn", False)),
                        "old_primary_require_support_knn_label_match": bool(
                            spec.get("old_primary_require_support_knn_label_match", True)
                        ),
                        "old_primary_min_support_knn_margin": float(
                            spec.get("old_primary_min_support_knn_margin", 0.0)
                        ),
                        "old_primary_max_support_knn_seen_new_minus_old": spec.get(
                            "old_primary_max_support_knn_seen_new_minus_old"
                        ),
                        "old_primary_min_old_drift_cos": float(spec.get("old_primary_min_old_drift_cos", -1.0)),
                        "old_primary_max_old_drift_dist": float(spec.get("old_primary_max_old_drift_dist", 1.0e6)),
                        "old_primary_require_class_envelope": bool(
                            spec.get("old_primary_require_class_envelope", False)
                        ),
                        "old_primary_unknown_veto_background_score": float(
                            spec.get("old_primary_unknown_veto_background_score", 0.86)
                        ),
                        "old_primary_unknown_veto_background_margin": float(
                            spec.get("old_primary_unknown_veto_background_margin", 0.10)
                        ),
                        "old_primary_unknown_veto_min_sources": int(
                            spec.get("old_primary_unknown_veto_min_sources", 1)
                        ),
                        "old_primary_fail_action": str(spec.get("old_primary_fail_action", "defer")),
                        "old_primary_unknown_veto_action": str(spec.get("old_primary_unknown_veto_action", "reject")),
                        "old_primary_promote_rescue_candidates": bool(
                            spec.get("old_primary_promote_rescue_candidates", False)
                        ),
                        "oa_mse_density_shell_gate": bool(spec.get("density_shell_gate", False)),
                        "density_shell_old_min_evidence_delta": float(spec.get("density_shell_old_min_evidence_delta", -0.04)),
                        "density_shell_old_min_anchor_delta": float(spec.get("density_shell_old_min_anchor_delta", -0.08)),
                        "density_shell_old_min_density_delta": float(spec.get("density_shell_old_min_density_delta", -0.06)),
                        "density_shell_seen_new_min_evidence_delta": float(spec.get("density_shell_seen_new_min_evidence_delta", -0.04)),
                        "density_shell_seen_new_min_anchor_delta": float(spec.get("density_shell_seen_new_min_anchor_delta", -0.08)),
                        "density_shell_seen_new_min_density_delta": float(spec.get("density_shell_seen_new_min_density_delta", -0.06)),
                        "density_shell_accept_background_margin": float(spec.get("density_shell_accept_background_margin", 0.18)),
                        "density_shell_reject_background_score": float(spec.get("density_shell_reject_background_score", 0.86)),
                        "density_shell_reject_background_margin": float(spec.get("density_shell_reject_background_margin", 0.14)),
                        "density_shell_reject_min_failed_shells": int(spec.get("density_shell_reject_min_failed_shells", 2)),
                        "oa_mse_identity_consensus_arbitration": bool(spec.get("identity_consensus_arbitration", False)),
                        "identity_consensus_old_min_evidence_delta": float(spec.get("identity_consensus_old_min_evidence_delta", -0.06)),
                        "identity_consensus_old_min_anchor_delta": float(spec.get("identity_consensus_old_min_anchor_delta", -0.10)),
                        "identity_consensus_old_min_density_delta": float(spec.get("identity_consensus_old_min_density_delta", -0.08)),
                        "identity_consensus_seen_new_min_evidence_delta": float(spec.get("identity_consensus_seen_new_min_evidence_delta", -0.04)),
                        "identity_consensus_seen_new_min_anchor_delta": float(spec.get("identity_consensus_seen_new_min_anchor_delta", -0.08)),
                        "identity_consensus_seen_new_min_density_delta": float(spec.get("identity_consensus_seen_new_min_density_delta", -0.06)),
                        "identity_consensus_min_identity_margin": float(spec.get("identity_consensus_min_identity_margin", -0.05)),
                        "identity_consensus_background_accept_margin": float(spec.get("identity_consensus_background_accept_margin", 0.22)),
                        "identity_consensus_reject_background_score": float(spec.get("identity_consensus_reject_background_score", 0.90)),
                        "identity_consensus_reject_background_margin": float(spec.get("identity_consensus_reject_background_margin", 0.18)),
                        "identity_consensus_reject_min_identity_failures": int(spec.get("identity_consensus_reject_min_identity_failures", 4)),
                        "identity_consensus_support_background_cap": bool(spec.get("identity_consensus_support_background_cap", False)),
                        "identity_consensus_support_background_cap_quantile": float(spec.get("identity_consensus_support_background_cap_quantile", 0.90)),
                        "identity_consensus_support_background_cap_slack": float(spec.get("identity_consensus_support_background_cap_slack", 0.05)),
                        "identity_consensus_support_background_cap_min_anchors": int(spec.get("identity_consensus_support_background_cap_min_anchors", 2)),
                        "oa_mse_support_conformal_arbitration": bool(spec.get("support_conformal_arbitration", False)),
                        "support_conformal_calibration_quantile": float(spec.get("support_conformal_calibration_quantile", 0.05)),
                        "support_conformal_conformity_slack": float(spec.get("support_conformal_conformity_slack", 0.12)),
                        "support_conformal_anchor_margin_slack": float(spec.get("support_conformal_anchor_margin_slack", 0.06)),
                        "support_conformal_background_score": float(spec.get("support_conformal_background_score", 0.82)),
                        "support_conformal_background_margin": float(spec.get("support_conformal_background_margin", 0.08)),
                        "support_conformal_hard_reject_margin": float(spec.get("support_conformal_hard_reject_margin", 0.18)),
                        "support_conformal_reject_min_failures": int(spec.get("support_conformal_reject_min_failures", 2)),
                        "support_conformal_reject_action": str(spec.get("support_conformal_reject_action", "reject")),
                        "oa_mse_support_reconstruction_arbitration": bool(spec.get("support_reconstruction_arbitration", False)),
                        "support_reconstruction_rank": int(spec.get("support_reconstruction_rank", 2)),
                        "support_reconstruction_residual_quantile": float(spec.get("support_reconstruction_residual_quantile", 0.95)),
                        "support_reconstruction_residual_slack": float(spec.get("support_reconstruction_residual_slack", 0.04)),
                        "support_reconstruction_min_residual_floor": float(spec.get("support_reconstruction_min_residual_floor", 0.03)),
                        "support_reconstruction_negative_scale": float(spec.get("support_reconstruction_negative_scale", 0.55)),
                        "support_reconstruction_negative_margin": float(spec.get("support_reconstruction_negative_margin", -0.02)),
                        "support_reconstruction_hard_residual_margin": float(spec.get("support_reconstruction_hard_residual_margin", 0.08)),
                        "support_reconstruction_background_score": float(spec.get("support_reconstruction_background_score", 0.86)),
                        "support_reconstruction_background_margin": float(spec.get("support_reconstruction_background_margin", 0.12)),
                        "support_reconstruction_reject_min_failures": int(spec.get("support_reconstruction_reject_min_failures", 2)),
                        "support_reconstruction_reject_action": str(spec.get("support_reconstruction_reject_action", "reject")),
                        "oa_mse_three_way_decision_head": bool(spec.get("three_way_decision_head", False)),
                        "oa_mse_three_way_head_weight": float(spec.get("three_way_head_weight", 0.0)),
                        "three_way_head_temperature": float(spec.get("three_way_head_temperature", 0.10)),
                        "three_way_head_known_margin": float(spec.get("three_way_head_known_margin", 0.08)),
                        "three_way_head_background_margin": float(spec.get("three_way_head_background_margin", 0.08)),
                        "three_way_head_support_ce_weight": float(spec.get("three_way_head_support_ce_weight", 1.0)),
                        "three_way_head_pseudo_ce_weight": float(spec.get("three_way_head_pseudo_ce_weight", 0.35)),
                        "three_way_head_support_background_margin_weight": float(spec.get("three_way_head_support_background_margin_weight", 1.0)),
                        "three_way_head_pseudo_margin_weight": float(spec.get("three_way_head_pseudo_margin_weight", 0.50)),
                        "three_way_accept_prob": float(spec.get("three_way_accept_prob", 0.50)),
                        "three_way_reject_prob": float(spec.get("three_way_reject_prob", 0.55)),
                        "three_way_defer_prob": float(spec.get("three_way_defer_prob", 0.45)),
                        "three_way_known_background_margin": float(spec.get("three_way_known_background_margin", 0.02)),
                        "three_way_reject_margin": float(spec.get("three_way_reject_margin", 0.04)),
                        "three_way_old_seen_ambiguity_margin": float(spec.get("three_way_old_seen_ambiguity_margin", 0.04)),
                        "three_way_defer_action": str(spec.get("three_way_defer_action", "uncertain")),
                        "three_way_decision_policy": str(spec.get("three_way_decision_policy", "background_competition")),
                        "three_way_known_floor": bool(spec.get("three_way_known_floor", False)),
                        "three_way_known_floor_action": str(spec.get("three_way_known_floor_action", "defer")),
                        "three_way_known_floor_old_min_evidence_delta": float(spec.get("three_way_known_floor_old_min_evidence_delta", -0.04)),
                        "three_way_known_floor_old_min_anchor_delta": float(spec.get("three_way_known_floor_old_min_anchor_delta", -0.08)),
                        "three_way_known_floor_old_min_anchor_margin": float(spec.get("three_way_known_floor_old_min_anchor_margin", -0.04)),
                        "three_way_known_floor_old_min_score_margin": float(spec.get("three_way_known_floor_old_min_score_margin", -0.12)),
                        "three_way_known_floor_seen_new_min_evidence_delta": float(spec.get("three_way_known_floor_seen_new_min_evidence_delta", -0.04)),
                        "three_way_known_floor_seen_new_min_anchor_delta": float(spec.get("three_way_known_floor_seen_new_min_anchor_delta", -0.08)),
                        "three_way_known_floor_seen_new_min_score_margin": float(spec.get("three_way_known_floor_seen_new_min_score_margin", -0.12)),
                        "three_way_known_floor_background_override_prob": float(spec.get("three_way_known_floor_background_override_prob", 0.995)),
                        "three_way_known_floor_background_override_margin": float(spec.get("three_way_known_floor_background_override_margin", 1.0)),
                        "oa_mse_pre_reject_defer_arbitration": bool(spec.get("pre_reject_defer_arbitration", False)),
                        "pre_reject_old_min_evidence_delta": float(spec.get("pre_reject_old_min_evidence_delta", 0.0)),
                        "pre_reject_old_min_anchor_delta": float(spec.get("pre_reject_old_min_anchor_delta", -0.02)),
                        "pre_reject_old_min_anchor_margin": float(spec.get("pre_reject_old_min_anchor_margin", 0.0)),
                        "pre_reject_old_min_score_margin": float(spec.get("pre_reject_old_min_score_margin", -0.02)),
                        "pre_reject_seen_new_min_evidence_delta": float(spec.get("pre_reject_seen_new_min_evidence_delta", 0.0)),
                        "pre_reject_seen_new_min_anchor_delta": float(spec.get("pre_reject_seen_new_min_anchor_delta", 0.0)),
                        "pre_reject_seen_new_min_score_margin": float(spec.get("pre_reject_seen_new_min_score_margin", -0.05)),
                        "pre_reject_max_background_score": float(spec.get("pre_reject_max_background_score", 0.74)),
                        "pre_reject_max_background_margin": float(spec.get("pre_reject_max_background_margin", 0.10)),
                        "pre_reject_defer_background_score": float(spec.get("pre_reject_defer_background_score", 0.70)),
                        "pre_reject_defer_background_margin": float(spec.get("pre_reject_defer_background_margin", 0.04)),
                        "pre_reject_reject_background_score": float(spec.get("pre_reject_reject_background_score", 0.82)),
                        "pre_reject_reject_background_margin": float(spec.get("pre_reject_reject_background_margin", 0.12)),
                        "pre_reject_defer_action": str(spec.get("pre_reject_defer_action", "uncertain")),
                        "pre_reject_support_neighborhood_retention": bool(spec.get("pre_reject_support_neighborhood_retention", False)),
                        "pre_reject_support_retention_old_min_evidence_delta": float(spec.get("pre_reject_support_retention_old_min_evidence_delta", 0.02)),
                        "pre_reject_support_retention_old_min_anchor_delta": float(spec.get("pre_reject_support_retention_old_min_anchor_delta", -0.04)),
                        "pre_reject_support_retention_old_min_anchor_margin": float(spec.get("pre_reject_support_retention_old_min_anchor_margin", -0.02)),
                        "pre_reject_support_retention_old_min_score_margin": float(spec.get("pre_reject_support_retention_old_min_score_margin", -0.04)),
                        "pre_reject_support_retention_seen_new_min_evidence_delta": float(spec.get("pre_reject_support_retention_seen_new_min_evidence_delta", 0.02)),
                        "pre_reject_support_retention_seen_new_min_anchor_delta": float(spec.get("pre_reject_support_retention_seen_new_min_anchor_delta", -0.04)),
                        "pre_reject_support_retention_seen_new_min_score_margin": float(spec.get("pre_reject_support_retention_seen_new_min_score_margin", -0.08)),
                        "pre_reject_support_retention_max_background_score": float(spec.get("pre_reject_support_retention_max_background_score", 0.96)),
                        "pre_reject_support_retention_max_background_margin": float(spec.get("pre_reject_support_retention_max_background_margin", 0.30)),
                        "pre_reject_support_retention_require_source_looo_pass": bool(
                            spec.get("pre_reject_support_retention_require_source_looo_pass", False)
                        ),
                        "pre_reject_support_retention_source_looo_max_failures": int(
                            spec.get("pre_reject_support_retention_source_looo_max_failures", 0)
                        ),
                        "oa_mse_retention_rescue_gate": bool(spec.get("retention_rescue_gate", False)),
                        "retention_rescue_old_min_evidence_delta": float(spec.get("retention_rescue_old_min_evidence_delta", 0.02)),
                        "retention_rescue_old_min_anchor_delta": float(spec.get("retention_rescue_old_min_anchor_delta", -0.01)),
                        "retention_rescue_old_min_anchor_margin": float(spec.get("retention_rescue_old_min_anchor_margin", 0.0)),
                        "retention_rescue_old_min_score_margin": float(spec.get("retention_rescue_old_min_score_margin", 0.0)),
                        "retention_rescue_seen_new_min_evidence_delta": float(spec.get("retention_rescue_seen_new_min_evidence_delta", 0.02)),
                        "retention_rescue_seen_new_min_anchor_delta": float(spec.get("retention_rescue_seen_new_min_anchor_delta", 0.0)),
                        "retention_rescue_seen_new_min_score_margin": float(spec.get("retention_rescue_seen_new_min_score_margin", -0.02)),
                        "retention_rescue_max_background_score": float(spec.get("retention_rescue_max_background_score", 0.70)),
                        "retention_rescue_max_background_margin": float(spec.get("retention_rescue_max_background_margin", 0.06)),
                        "retention_rescue_candidate_only": bool(spec.get("retention_rescue_candidate_only", False)),
                        "oa_mse_void_background_weight": float(spec.get("void_background", 0.0)),
                        "oa_mse_negative_anchor_weight": float(spec.get("negative_anchor_weight", 0.0)),
                        "negative_anchor_margin": float(spec.get("negative_anchor_margin", 0.12)),
                        "negative_anchor_temperature": float(spec.get("negative_anchor_temperature", 0.10)),
                        "negative_anchor_max_anchors": int(spec.get("negative_anchor_max_anchors", 256)),
                        "oa_mse_void_gate": bool(spec.get("void_gate", False)),
                        "oa_mse_void_gate_min_score": float(spec.get("void_gate_min_score", 0.55)),
                        "oa_mse_void_gate_min_margin": float(spec.get("void_gate_min_margin", 0.05)),
                        "oa_mse_old_bridge_weight": float(spec["old_bridge"]),
                        "old_bridge_samples_per_class": 3 if k_old <= 2 else 4,
                        "old_bridge_max_mix": 0.82 if k_old <= 2 else 0.78,
                        "oa_mse_old_neighborhood_weight": float(spec["old_neighborhood"]),
                        "old_neighborhood_samples_per_class": 3 if k_old <= 2 else 4,
                        "old_neighborhood_radius": 0.08 if k_old <= 2 else 0.05,
                        "oa_mse_old_surrogate_margin_weight": float(
                            spec.get("old_surrogate_margin_weight", 0.06 if k_old <= 2 else 0.10)
                        ),
                        "old_surrogate_margin": float(spec.get("old_surrogate_margin", 0.08 if k_old <= 2 else 0.10)),
                        "oa_mse_source_looo_unknown_weight": float(spec.get("source_looo_unknown_weight", 0.0)),
                        "source_looo_unknown_margin": float(spec.get("source_looo_unknown_margin", 0.35)),
                        "source_looo_interclass_margin": float(spec.get("source_looo_interclass_margin", 0.08)),
                        "source_looo_max_samples_per_class": int(spec.get("source_looo_max_samples_per_class", 24)),
                        "oa_mse_source_looo_risk_arbitration": bool(spec.get("source_looo_risk_arbitration", False)),
                        "source_looo_risk_quantile": float(spec.get("source_looo_risk_quantile", 0.85)),
                        "source_looo_risk_slack": float(spec.get("source_looo_risk_slack", 0.0)),
                        "source_looo_risk_min_score_margin": float(spec.get("source_looo_risk_min_score_margin", 0.02)),
                        "source_looo_risk_min_known_evidence_delta": float(spec.get("source_looo_risk_min_known_evidence_delta", -0.08)),
                        "source_looo_risk_background_score": float(spec.get("source_looo_risk_background_score", 0.86)),
                        "source_looo_risk_background_margin": float(spec.get("source_looo_risk_background_margin", 0.10)),
                        "source_looo_risk_reject_min_failures": int(spec.get("source_looo_risk_reject_min_failures", 2)),
                        "source_looo_risk_reject_action": str(spec.get("source_looo_risk_reject_action", "reject")),
                        "oa_mse_known_coverage_weight": float(spec.get("known_coverage_weight", 0.0)),
                        "known_coverage_margin": float(spec.get("known_coverage_margin", 0.12)),
                        "known_coverage_min_affinity": float(spec.get("known_coverage_min_affinity", 0.35)),
                        "known_coverage_max_samples": int(spec.get("known_coverage_max_samples", 256)),
                        "old_surrogate_evidence_margin": 0.04 if k_old <= 2 else 0.06,
                        "old_surrogate_reject_relax": float(spec.get("reject_relax", 0.0)),
                        "oa_mse_siamese_quantile": float(spec.get("siamese_quantile", 0.10)),
                        "oa_mse_siamese_accept_threshold": float(spec.get("siamese_threshold", 0.50)),
                        "oa_mse_siamese_unknown_veto": bool(spec.get("siamese_unknown_veto", False)),
                        "oa_mse_siamese_unknown_veto_mode": str(spec.get("siamese_unknown_veto_mode", "any")),
                        "oa_mse_siamese_min_old_support_evidence_delta": spec.get(
                            "min_old_support_evidence_delta"
                        ),
                        "oa_mse_siamese_min_old_surrogate_reject_delta": spec.get(
                            "min_old_surrogate_reject_delta"
                        ),
                        "oa_mse_siamese_min_energy_delta": spec.get("min_energy_delta"),
                        "oa_mse_siamese_min_mahalanobis_delta": spec.get("min_mahalanobis_delta"),
                        "oa_mse_siamese_min_accept_delta": spec.get("min_accept_delta"),
                        "oa_mse_siamese_min_old_support_anchor_margin": spec.get("min_old_support_anchor_margin"),
                        "oa_mse_siamese_min_veto_failures": int(spec.get("min_veto_failures", 1)),
                        "oa_mse_old_unknown_acceptance_guard": bool(spec.get("old_unknown_acceptance_guard", False)),
                        "oa_mse_old_unknown_guard_min_old_support_evidence_delta": spec.get(
                            "guard_min_old_support_evidence_delta"
                        ),
                        "oa_mse_old_unknown_guard_min_old_surrogate_reject_delta": spec.get(
                            "guard_min_old_surrogate_reject_delta"
                        ),
                        "oa_mse_old_unknown_guard_min_energy_delta": spec.get("guard_min_energy_delta"),
                        "oa_mse_old_unknown_guard_min_mahalanobis_delta": spec.get(
                            "guard_min_mahalanobis_delta"
                        ),
                        "oa_mse_old_unknown_guard_min_accept_delta": spec.get("guard_min_accept_delta"),
                        "oa_mse_old_unknown_guard_min_old_support_anchor_margin": spec.get(
                            "guard_min_old_support_anchor_margin"
                        ),
                        "oa_mse_old_unknown_guard_min_best_old_score": spec.get("guard_min_best_old_score"),
                        "oa_mse_old_unknown_guard_min_margin": spec.get("guard_min_margin"),
                        "oa_mse_old_unknown_guard_min_failures": int(spec.get("guard_min_failures", 1)),
                        "old_anchor_override_min_quality": float(
                            spec.get("old_anchor_override_min_quality", 0.55 if is_seen_new else 0.60)
                        ),
                        "old_retention_quantile": float(
                            spec.get("old_retention_quantile", 0.90 if k_old <= 2 else 0.88)
                        ),
                        "oa_mse_support_retention_guard": bool(spec.get("support_retention_guard", False)),
                        "support_retention_guard_quantile": float(spec.get("support_retention_guard_quantile", 0.05)),
                        "support_retention_guard_slack": float(spec.get("support_retention_guard_slack", 0.02)),
                        "oa_mse_two_branch_background_guard": bool(spec.get("two_branch_background_guard", False)),
                        "two_branch_bg_min_score": float(spec.get("two_branch_bg_min_score", 0.62)),
                        "two_branch_bg_min_margin": float(spec.get("two_branch_bg_min_margin", -0.02)),
                        "two_branch_old_support_evidence_delta": float(
                            spec.get("two_branch_old_support_evidence_delta", 0.0)
                        ),
                        "two_branch_old_anchor_delta": float(spec.get("two_branch_old_anchor_delta", -0.02)),
                        "two_branch_old_anchor_margin": float(spec.get("two_branch_old_anchor_margin", 0.0)),
                        "two_branch_seen_new_evidence_delta": float(
                            spec.get("two_branch_seen_new_evidence_delta", 0.0)
                        ),
                        "two_branch_seen_new_anchor_delta": float(
                            spec.get("two_branch_seen_new_anchor_delta", 0.0)
                        ),
                        "oa_mse_seen_new_registration_override": bool(
                            spec.get("seen_new_registration_override", False)
                        ),
                        "seen_new_override_min_evidence_delta": float(
                            spec.get("seen_new_override_min_evidence_delta", 0.0)
                        ),
                        "seen_new_override_min_anchor_delta": float(
                            spec.get("seen_new_override_min_anchor_delta", 0.0)
                        ),
                        "seen_new_override_min_affinity_delta": float(
                            spec.get("seen_new_override_min_affinity_delta", -0.02)
                        ),
                        "seen_new_override_min_residual_delta": float(
                            spec.get("seen_new_override_min_residual_delta", -0.02)
                        ),
                        "seen_new_override_min_score_margin": float(
                            spec.get("seen_new_override_min_score_margin", -0.10)
                        ),
                        "seen_new_override_min_seen_vs_old_evidence_margin": float(
                            spec.get("seen_new_override_min_seen_vs_old_evidence_margin", 0.02)
                        ),
                        "seen_new_override_max_background_score": float(
                            spec.get("seen_new_override_max_background_score", 0.72)
                        ),
                        "seen_new_override_max_background_margin": float(
                            spec.get("seen_new_override_max_background_margin", 0.08)
                        ),
                        "seen_new_override_min_support_knn_seen_new_minus_old": spec.get(
                            "seen_new_override_min_support_knn_seen_new_minus_old"
                        ),
                        "seen_new_override_min_support_knn_margin": spec.get(
                            "seen_new_override_min_support_knn_margin"
                        ),
                        "oa_mse_adapter_selection_policy": str(
                            spec.get("adapter_selection_policy", base["oa_mse_adapter_selection_policy"])
                        ),
                        "old_acc_target": float(spec.get("old_acc_target", base["old_acc_target"])),
                        "seen_new_acc_target": float(spec.get("seen_new_acc_target", base["seen_new_acc_target"])),
                        "stage2_priority_phase": str(spec.get("stage2_priority_phase", "")),
                        "old_acc_phase_gate": float(spec.get("old_acc_phase_gate", 0.0)),
                        "secondary_objectives_after_old_gate": str(spec.get("secondary_objectives_after_old_gate", "")),
                    }
                )
                if plan in {"OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"}:
                    candidate_kwargs["new_tx_ids"] = "__NONE__"
                rows.append(
                    Candidate(
                        cid=(
                            f"OA_MSE_CLASSCOND32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_CLASSCOND32"
                            else
                            f"OA_MSE_BALANCE64_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_BALANCE64"
                            else
                            f"OA_MSE_SOFTMIX64_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SOFTMIX64"
                            else
                            f"OA_MSE_VOID64_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_VOID64"
                            else
                            f"OA_MSE_SOFTVOID128_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SOFTVOID128"
                            else
                            f"OA_MSE_MIXHEAD128_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_MIXHEAD128"
                            else
                            f"OA_MSE_STRUCT48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_STRUCT48"
                            else
                            f"OA_MSE_SIMPLIFIED48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SIMPLIFIED48"
                            else
                            f"OA_MSE_RETENTION48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_RETENTION48"
                            else
                            f"OA_MSE_SUPPORTRET48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SUPPORTRET48"
                            else
                            f"OA_MSE_TWOBRANCH48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_TWOBRANCH48"
                            else
                            f"OA_MSE_REGHEAD48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_REGHEAD48"
                            else
                            f"OA_MSE_GEOM48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_GEOM48"
                            else
                            f"OA_MSE_TRIAGE48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_TRIAGE48"
                            else
                            f"OA_MSE_LOOO48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_LOOO48"
                            else
                            f"OA_MSE_CONSTRAIN48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_CONSTRAIN48"
                            else
                            f"OA_MSE_ENVELOPE48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_ENVELOPE48"
                            else
                            f"OA_MSE_RESCUE48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_RESCUE48"
                            else
                            f"OA_MSE_PREREJECT48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_PREREJECT48"
                            else
                            f"OA_MSE_THREEWAY48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_THREEWAY48"
                            else
                            f"OA_MSE_COVFLOOR48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_COVFLOOR48"
                            else
                            f"OA_MSE_CLASSFIRST48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_CLASSFIRST48"
                            else
                            f"OA_MSE_EVIBG48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_EVIBG48"
                            else
                            f"OA_MSE_SOFTTARGET48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SOFTTARGET48"
                            else
                            f"OA_MSE_NEGANCHOR48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_NEGANCHOR48"
                            else
                            f"OA_MSE_DENSHELL48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_DENSHELL48"
                            else
                            f"OA_MSE_IDCONS48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_IDCONS48"
                            else
                            f"OA_MSE_CONFORM48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_CONFORM48"
                            else
                            f"OA_MSE_RECON48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_RECON48"
                            else
                            f"OA_MSE_SOURCERISK48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SOURCERISK48"
                            else
                            f"OA_MSE_RISKRET48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_RISKRET48"
                            else
                            f"OA_MSE_MANIFOLD48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_MANIFOLD48"
                            else
                            f"OA_MSE_H06_EVID48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_EVID48"
                            else
                            f"OA_MSE_H06_ARB48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_ARB48"
                            else
                            f"OA_MSE_H06_OLDUNK48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDUNK48"
                            else
                            f"OA_MSE_H06_BGTRAIN48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_BGTRAIN48"
                            else
                            f"OA_MSE_H06_OLDFIRST48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDFIRST48"
                            else
                            f"OA_MSE_H06_OLDRELAX48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDRELAX48"
                            else
                            f"OA_MSE_H06_OLDGEOM48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDGEOM48"
                            else
                            f"OA_MSE_H06_OLDCONF48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDCONF48"
                            else
                            f"OA_MSE_H06_OLDBUDGET48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDBUDGET48"
                            else
                            f"OA_MSE_H06_OLDQUAL48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDQUAL48"
                            else
                            f"OA_MSE_H06_OLDRISK48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDRISK48"
                            else
                            f"OA_MSE_H06_OLDFUSE48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDFUSE48"
                            else
                            f"OA_MSE_H06_ROLLSAFE48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_ROLLSAFE48"
                            else
                            f"OA_MSE_H06_OLDHEAD48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDHEAD48"
                            else
                            f"OA_MSE_H06_OLDHEADFAR48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDHEADFAR48"
                            else
                            f"OA_MSE_H06_OLDRECOV48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_OLDRECOV48"
                            else
                            f"OA_MSE_H06_RETOLD48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_H06_RETOLD48"
                            else
                            f"OA_MSE_KRET48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_KRET48"
                            else
                            f"OA_MSE_BGCAP48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_BGCAP48"
                            else f"OA_MSE_SUPPORTCV48_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_SUPPORTCV48"
                            else
                            f"OA_MSE_ANCHORGUARD128_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_ANCHORGUARD128"
                            else
                            f"OA_MSE_CALGUARD32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_CALGUARD32"
                            else
                            f"OA_MSE_VETO32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_VETO32"
                            else
                            f"OA_MSE_UNCERTAIN32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_UNCERTAIN32"
                            else
                            f"OA_MSE_BOUNDARY32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                            if plan == "OA_MSE_BOUNDARY32"
                            else f"OA_MSE_PROXY32_GPU{gpu}_{slot}_{spec['stage'].upper()}_KOLD{k_old}_KNEW{k_new}"
                        ),
                        k=k_new if is_seen_new else k_old,
                        gpu=gpu,
                        slot=f"GPU{gpu}/{slot}",
                        target_visibility=(
                            "target_old_and_seen_new_leo_support_labeled_unknown_eval_only"
                            if is_seen_new
                            else "target_old_leo_support_labeled_unknown_eval_only"
                        ),
                        update_module=(
                            (
                                "class_conditional_veto_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_CLASSCOND32"
                                else "balanced_rescue_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_BALANCE64"
                                else "softmix_boundary_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_SOFTMIX64"
                                else "void_background_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_VOID64"
                                else "softvoid_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_SOFTVOID128"
                                else "multiproto_score_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_MIXHEAD128"
                                else "structural_density_residual_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_STRUCT48"
                                else "simplified_leo_residual_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_SIMPLIFIED48"
                                else "retention_risk_balanced_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_RETENTION48"
                                else "support_retention_guarded_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_SUPPORTRET48"
                                else "two_branch_background_guard_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_TWOBRANCH48"
                                else "explicit_seen_new_registration_head_oa_mse"
                                if plan == "OA_MSE_REGHEAD48"
                                else "support_center_geometry_registration_oa_mse_head"
                                if plan == "OA_MSE_GEOM48"
                                else "triage_split_objective_oa_mse_head_seen_new_rescue"
                                if plan == "OA_MSE_TRIAGE48"
                                else "source_leave_one_old_out_meta_unknown_oa_mse_head"
                                if plan == "OA_MSE_LOOO48"
                                else "known_coverage_constrained_oa_mse_head"
                                if plan == "OA_MSE_CONSTRAIN48"
                                else "source_support_class_envelope_oa_mse_head"
                                if plan == "OA_MSE_ENVELOPE48"
                                else "post_reject_retention_rescue_oa_mse_head"
                                if plan == "OA_MSE_RESCUE48"
                                else "pre_reject_defer_arbitration_oa_mse_head"
                                if plan == "OA_MSE_PREREJECT48"
                                else "structural_three_way_old_seen_background_oa_mse_head"
                                if plan == "OA_MSE_THREEWAY48"
                                else "coverage_floor_three_way_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_COVFLOOR48"
                                else "class_first_known_assignment_oa_mse_head"
                                if plan == "OA_MSE_CLASSFIRST48"
                                else "negative_anchor_background_basin_oa_mse_head"
                                if plan == "OA_MSE_NEGANCHOR48"
                                else "density_shell_inlier_first_oa_mse_head"
                                if plan == "OA_MSE_DENSHELL48"
                                else "identity_consensus_oa_mse_head"
                                if plan == "OA_MSE_IDCONS48"
                                else "support_conformal_identity_oa_mse_head"
                                if plan == "OA_MSE_CONFORM48"
                                else "support_reconstruction_identity_oa_mse_head"
                                if plan == "OA_MSE_RECON48"
                                else "source_looo_risk_identity_oa_mse_head"
                                if plan == "OA_MSE_SOURCERISK48"
                                else "source_risk_constrained_known_retention_oa_mse_head"
                                if plan == "OA_MSE_RISKRET48"
                                else "support_manifold_soft_prototype_consistency_mse_subspace"
                                if plan == "OA_MSE_MANIFOLD48"
                                else "h06_arbitration_repair_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_H06_ARB48"
                                else "h06_multi_evidence_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_H06_EVID48"
                                else "h06_old_unknown_only_identity_first_late_veto"
                                if plan == "OA_MSE_H06_OLDUNK48"
                                else "h06_old_unknown_learned_background_rejector"
                                if plan == "OA_MSE_H06_BGTRAIN48"
                                else "h06_old_primary_gate_terminal_unknown_veto"
                                if plan == "OA_MSE_H06_OLDFIRST48"
                                else "h06_old_primary_relaxed_ablation_terminal_unknown_veto"
                                if plan == "OA_MSE_H06_OLDRELAX48"
                                else "h06_old_unknown_support_geometry_repair"
                                if plan == "OA_MSE_H06_OLDGEOM48"
                                else "h06_old_unknown_support_conformal_retention_repair"
                                if plan == "OA_MSE_H06_OLDCONF48"
                                else "h06_old_unknown_acceptance_budget_repair"
                                if plan == "OA_MSE_H06_OLDBUDGET48"
                                else "h06_old_unknown_support_quality_prototype_construction_repair"
                                if plan == "OA_MSE_H06_OLDQUAL48"
                                else "h06_unknown_separability_query_free_background_risk_repair"
                                if plan == "OA_MSE_H06_OLDRISK48"
                                else "h06_oldqual_oldrisk_fusion_rollback_calibration_repair"
                                if plan == "OA_MSE_H06_OLDFUSE48"
                                else "h06_rollback_safe_retention_repair"
                                if plan == "OA_MSE_H06_ROLLSAFE48"
                                else "h06_oldhead_boundary_repair"
                                if plan == "OA_MSE_H06_OLDHEAD48"
                                else "h06_oldheadfar_support_cv_stability_repair"
                                if plan == "OA_MSE_H06_OLDHEADFAR48"
                                else "h06_oldrecov_target_old_recoverability_repair"
                                if plan == "OA_MSE_H06_OLDRECOV48"
                                else "h06_old_retention_first_calibrated_unknown_veto"
                                if plan == "OA_MSE_H06_RETOLD48"
                                else "support_background_cap_identity_oa_mse_head"
                                if plan == "OA_MSE_BGCAP48"
                                else "support_neighborhood_known_retention_oa_mse_head"
                                if plan == "OA_MSE_KRET48"
                                else "support_cv_selected_oa_mse_head"
                                if plan == "OA_MSE_SUPPORTCV48"
                                else "anchorguard_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_ANCHORGUARD128"
                                else "old_unknown_calibrated_guard_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_CALGUARD32"
                                else "unknown_veto_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_VETO32"
                                else "uncertain_band_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_UNCERTAIN32"
                                else "boundary_guard_oa_mse_head_seen_new_registration"
                                if plan == "OA_MSE_BOUNDARY32"
                                else "proxy_alpha_selected_oa_mse_head_seen_new_registration"
                            )
                            if is_seen_new
                            else (
                                "class_conditional_veto_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_CLASSCOND32"
                                else "balanced_rescue_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_BALANCE64"
                                else "softmix_boundary_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_SOFTMIX64"
                                else "void_background_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_VOID64"
                                else "softvoid_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_SOFTVOID128"
                                else "multiproto_score_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_MIXHEAD128"
                                else "structural_density_residual_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_STRUCT48"
                                else "simplified_leo_residual_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_SIMPLIFIED48"
                                else "retention_risk_balanced_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_RETENTION48"
                                else "support_retention_guarded_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_SUPPORTRET48"
                                else "two_branch_background_guard_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_TWOBRANCH48"
                                else "registration_head_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_REGHEAD48"
                                else "support_center_geometry_registration_mse_subspace_target_old"
                                if plan == "OA_MSE_GEOM48"
                                else "triage_split_objective_mse_subspace_old_unknown_boundary"
                                if plan == "OA_MSE_TRIAGE48"
                                else "source_leave_one_old_out_meta_unknown_mse_subspace"
                                if plan == "OA_MSE_LOOO48"
                                else "known_coverage_constrained_mse_subspace"
                                if plan == "OA_MSE_CONSTRAIN48"
                                else "source_support_class_envelope_mse_subspace"
                                if plan == "OA_MSE_ENVELOPE48"
                                else "post_reject_retention_rescue_mse_subspace"
                                if plan == "OA_MSE_RESCUE48"
                                else "pre_reject_defer_arbitration_mse_subspace"
                                if plan == "OA_MSE_PREREJECT48"
                                else "structural_three_way_old_seen_background_mse_subspace"
                                if plan == "OA_MSE_THREEWAY48"
                                else "coverage_floor_three_way_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_COVFLOOR48"
                                else "class_first_known_assignment_mse_subspace"
                                if plan == "OA_MSE_CLASSFIRST48"
                                else "negative_anchor_background_basin_mse_subspace"
                                if plan == "OA_MSE_NEGANCHOR48"
                                else "density_shell_inlier_first_mse_subspace"
                                if plan == "OA_MSE_DENSHELL48"
                                else "identity_consensus_mse_subspace"
                                if plan == "OA_MSE_IDCONS48"
                                else "support_conformal_identity_mse_subspace"
                                if plan == "OA_MSE_CONFORM48"
                                else "support_reconstruction_identity_mse_subspace"
                                if plan == "OA_MSE_RECON48"
                                else "source_looo_risk_identity_mse_subspace"
                                if plan == "OA_MSE_SOURCERISK48"
                                else "source_risk_constrained_known_retention_mse_subspace"
                                if plan == "OA_MSE_RISKRET48"
                                else "h06_arbitration_repair_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_H06_ARB48"
                                else "h06_old_unknown_only_identity_first_late_veto_mse_subspace"
                                if plan == "OA_MSE_H06_OLDUNK48"
                                else "h06_old_unknown_learned_background_mse_subspace"
                                if plan == "OA_MSE_H06_BGTRAIN48"
                                else "h06_old_primary_gate_terminal_unknown_veto_mse_subspace"
                                if plan == "OA_MSE_H06_OLDFIRST48"
                                else "h06_old_primary_relaxed_ablation_mse_subspace"
                                if plan == "OA_MSE_H06_OLDRELAX48"
                                else "h06_old_unknown_support_geometry_mse_subspace"
                                if plan == "OA_MSE_H06_OLDGEOM48"
                                else "h06_old_unknown_support_conformal_retention_mse_subspace"
                                if plan == "OA_MSE_H06_OLDCONF48"
                                else "h06_old_unknown_acceptance_budget_mse_subspace"
                                if plan == "OA_MSE_H06_OLDBUDGET48"
                                else "h06_old_unknown_support_quality_prototype_construction_mse_subspace"
                                if plan == "OA_MSE_H06_OLDQUAL48"
                                else "h06_unknown_separability_query_free_background_risk_mse_subspace"
                                if plan == "OA_MSE_H06_OLDRISK48"
                                else "h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace"
                                if plan == "OA_MSE_H06_OLDFUSE48"
                                else "h06_rollback_safe_retention_mse_subspace"
                                if plan == "OA_MSE_H06_ROLLSAFE48"
                                else "h06_oldhead_boundary_repair_mse_subspace"
                                if plan == "OA_MSE_H06_OLDHEAD48"
                                else "h06_oldheadfar_support_cv_stability_mse_subspace"
                                if plan == "OA_MSE_H06_OLDHEADFAR48"
                                else "h06_oldrecov_target_old_recoverability_mse_subspace"
                                if plan == "OA_MSE_H06_OLDRECOV48"
                                else "h06_old_retention_first_calibrated_unknown_veto_mse_subspace"
                                if plan == "OA_MSE_H06_RETOLD48"
                                else "support_background_cap_identity_mse_subspace"
                                if plan == "OA_MSE_BGCAP48"
                                else "support_neighborhood_known_retention_mse_subspace"
                                if plan == "OA_MSE_KRET48"
                                else "support_cv_selected_mse_subspace"
                                if plan == "OA_MSE_SUPPORTCV48"
                                else "anchorguard_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_ANCHORGUARD128"
                                else "old_unknown_calibrated_guard_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_CALGUARD32"
                                else "unknown_veto_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_VETO32"
                                else "uncertain_band_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_UNCERTAIN32"
                                else "boundary_guard_mse_subspace_target_old_retention"
                                if plan == "OA_MSE_BOUNDARY32"
                                else "proxy_alpha_selected_mse_subspace_target_old_retention"
                            )
                        ),
                        metrics=(
                            "full_accuracy,coverage,old_class_accuracy,new_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha"
                            if is_seen_new
                            else "full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha"
                        ),
                        description=str(spec["description"]),
                        oa_mse_stage=str(spec["stage"]),
                        eval_protocol=str(spec["eval_protocol"]),
                        source_target_fusion_policy=(
                            "source_old_plus_target_old_support_plus_seen_new_support_no_unknown_fit"
                            if is_seen_new
                            else "source_old_plus_target_old_labeled_support_only"
                        ),
                        fusion_inputs=(
                            (
                                "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector,pseudo_unknown_void_background_gate"
                                if plan == "OA_MSE_VOID64"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector,soft_multi_prototype_score_head,pseudo_unknown_void_background_gate"
                                if plan == "OA_MSE_MIXHEAD128"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,residual_mlp_adapter_when_selected"
                                if plan == "OA_MSE_STRUCT48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,guarded_void_background,residual_mlp_adapter_when_selected"
                                if plan == "OA_MSE_SIMPLIFIED48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,retention_risk_balanced_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_RETENTION48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_SUPPORTRET48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_TWOBRANCH48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,seen_new_registration_override,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_REGHEAD48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,support_center_leave_one_out_metric_loss,seen_new_registration_override,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_GEOM48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,split_objective_triage,old_retention_arm,unknown_boundary_arm,seen_new_rescue_arm,support_center_leave_one_out_metric_loss,seen_new_registration_override,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_TRIAGE48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_leave_one_old_out_meta_unknown_loss,support_center_leave_one_out_metric_loss,seen_new_registration_override,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_LOOO48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,support_evidence_overridden_background_gates,seen_new_registration_override,soft_multi_prototype_score_head,anchor_density_uncertain_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_CONSTRAIN48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_support_class_envelope_gate,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,seen_new_registration_override,soft_multi_prototype_score_head,anchor_density_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_ENVELOPE48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_support_class_envelope_gate,post_reject_retention_rescue,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,seen_new_registration_override,soft_multi_prototype_score_head,anchor_density_gate,pseudo_background_cap,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_RESCUE48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_support_class_envelope_gate,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,seen_new_registration_override,soft_multi_prototype_score_head,anchor_density_uncertain_gate,pseudo_background_cap,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_PREREJECT48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,three_way_old_seen_background_head_loss,three_way_decision_head,pseudo_background_anchors,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_THREEWAY48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,three_way_old_seen_background_head_loss,known_evidence_floor_before_background_reject,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_COVFLOOR48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,class_first_known_assignment,three_way_old_seen_background_head_loss,known_evidence_floor_before_background_veto,pre_reject_defer_arbitration,retention_rescue,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,seen_new_registration_override,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_CLASSFIRST48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,evidence_balanced_old_seen_background_head,known_support_evidence_required_for_accept,pseudo_background_competition,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_EVIBG48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,prototype_mixture_soft_target,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,decoupled_background_risk_veto,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_SOFTTARGET48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,negative_anchor_background_basin,prototype_mixture_soft_target,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,decoupled_background_risk_veto,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_NEGANCHOR48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,class_conditional_density_shell_inlier_gate,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_DENSHELL48"
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,identity_consensus_arbitration,class_conditional_density_shell_inlier_gate,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_IDCONS48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,identity_consensus_arbitration,support_conformal_arbitration,class_conditional_support_floor,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_CONFORM48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,identity_consensus_arbitration,support_reconstruction_arbitration,class_local_low_rank_residual,reciprocal_boundary_negatives,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_RECON48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_leave_one_old_out_impostor_risk_arbitration,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,negative_anchor_background_basin_when_aggressive,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_SOURCERISK48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,source_leave_one_old_out_impostor_risk_arbitration,source_risk_constrained_support_retention,pre_reject_defer_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_RISKRET48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,support_manifold_soft_mixture_consistency,source_risk_constrained_support_retention,pre_reject_defer_arbitration,soft_multi_prototype_score_head,known_coverage_margin_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_MANIFOLD48"
                            else "h06_latest_source_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,old_drift_support_quality,support_knn,seen_new_support_knn_gate,source_looo_risk,pre_reject_defer,pair_verifier_actionable_veto,three_way_background,soft_multi_prototype_score_head,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_ARB48"
                            else "h06_latest_source_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,old_drift_support_quality,support_knn,identity_consensus,pre_reject_defer,three_way_background,pair_verifier,soft_multi_prototype_score_head,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_EVID48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_drift_support_quality,old_support_knn,identity_consensus,pair_verifier_late_unknown_veto,background_late_unknown_veto,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDUNK48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_background_two_way_head,query_free_pseudo_background,source_looo_risk,support_reconstruction_when_enabled,old_drift_support_quality,old_support_knn,identity_consensus,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_BGTRAIN48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,soft_multi_prototype_score_head,old_support_knn,old_drift_gate,class_envelope,old_primary_terminal_gate,unknown_risk_veto_blocks_rescue,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDFIRST48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,support_center_geometry,target_shift_halo_ring_pseudo_unknowns,soft_multi_prototype_score_head,mixture_consistency,old_primary_measurement_gate,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDGEOM48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,support_conformal_arbitration,support_reconstruction_arbitration,pre_reject_support_retention,retention_rescue_candidate_only,old_primary_promote_rescue_candidates,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDCONF48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,soft_acceptance_budget,support_conformal_defer,support_reconstruction_defer,old_primary_rescue_budget,source_looo_defer_budget,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDBUDGET48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_quality_prototype_construction,support_center_leave_one_out_metric_loss,soft_multi_prototype_score_head,mixture_consistency,support_conformal_defer,support_reconstruction_defer,source_looo_quality_probe,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDQUAL48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,query_free_background_risk,source_looo_risk,two_branch_pseudo_background_guard,pre_reject_defer,unknown_score_joint_veto,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDRISK48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,oldqual_oldrisk_fusion,oldqual_support_quality,oldrisk_query_free_background_risk,rollback_calibration,pre_reject_defer,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDFUSE48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,rollback_safe_retention,retention_rescue_candidate_only,defer_first_deployment_gate,pre_reject_support_retention,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_ROLLSAFE48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,oldhead_ridge_recoverability,knn_density_boundary_risk,density_shell_guard,support_conformal_reconstruction,three_way_old_background_head,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDHEAD48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_cv_stability_head,identity_preserving_cv_selector,support_background_cap,density_shell_guard,support_conformal_reconstruction,three_way_old_background_head,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDHEADFAR48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,target_old_recoverability_first,target_only_ridge_upper_bound,oldrecov_ridge_head,oldrecov_proto_bridge,support_conformal_reconstruction_defer,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDRECOV48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_proof_first,old_drift_support_quality,identity_consensus_background_cap,three_way_background_prob_as_soft_risk,unknown_score_joint_veto,support_reconstruction_conformal_defer_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_RETOLD48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,support_calibrated_background_cap,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,retention_rescue,negative_anchor_background_basin,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_BGCAP48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,support_neighborhood_known_retention,pre_reject_defer_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,source_leave_one_old_out_risk_when_enabled,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_KRET48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,support_leave_one_out_adapter_selector,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,source_leave_one_old_out_risk_when_enabled,simplified_leo_residual_channel,multi_target_receiver_pool"
                            if plan == "OA_MSE_SUPPORTCV48"
                            else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector,soft_prototype_mixture,pseudo_unknown_void_background_gate"
                                if plan in {"OA_MSE_SOFTVOID128", "OA_MSE_ANCHORGUARD128"}
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,target_boundary_guard_selector"
                                if plan
                                in {
                                    "OA_MSE_BOUNDARY32",
                                    "OA_MSE_UNCERTAIN32",
                                    "OA_MSE_VETO32",
                                    "OA_MSE_CLASSCOND32",
                                    "OA_MSE_CALGUARD32",
                                    "OA_MSE_BALANCE64",
                                    "OA_MSE_SOFTMIX64",
                                    "OA_MSE_VOID64",
                                    "OA_MSE_SOFTVOID128",
                                    "OA_MSE_ANCHORGUARD128",
                                    "OA_MSE_MIXHEAD128",
                                    "OA_MSE_STRUCT48",
                                    "OA_MSE_SIMPLIFIED48",
                                    "OA_MSE_RETENTION48",
                                    "OA_MSE_SUPPORTRET48",
                                    "OA_MSE_TWOBRANCH48",
                                    "OA_MSE_REGHEAD48",
                                    "OA_MSE_GEOM48",
                                    "OA_MSE_TRIAGE48",
                                    "OA_MSE_LOOO48",
                                    "OA_MSE_CONSTRAIN48",
                                    "OA_MSE_ENVELOPE48",
                                    "OA_MSE_RESCUE48",
                                    "OA_MSE_PREREJECT48",
                                    "OA_MSE_THREEWAY48",
                                }
                                else "source_old_prototypes,target_old_support,seen_new_support,U_orbit,class_masks,energy_gate,proxy_alpha_selector"
                            )
                            if is_seen_new
                            else (
                                "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector,pseudo_unknown_void_background_gate"
                                if plan == "OA_MSE_VOID64"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector,soft_multi_prototype_score_head,pseudo_unknown_void_background_gate"
                                if plan == "OA_MSE_MIXHEAD128"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,residual_mlp_adapter_when_selected"
                                if plan == "OA_MSE_STRUCT48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,guarded_void_background,residual_mlp_adapter_when_selected"
                                if plan == "OA_MSE_SIMPLIFIED48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,retention_risk_balanced_selector,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_RETENTION48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_SUPPORTRET48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_TWOBRANCH48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,seen_new_registration_override_inactive_without_seen_support,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel"
                                if plan == "OA_MSE_REGHEAD48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,support_center_leave_one_out_metric_loss,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_GEOM48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,split_objective_triage,old_retention_arm,unknown_boundary_arm,support_center_leave_one_out_metric_loss,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_TRIAGE48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_leave_one_old_out_meta_unknown_loss,support_center_leave_one_out_metric_loss,retention_risk_balanced_selector,support_retention_guarded_surrogate_reject_gate,two_branch_pseudo_background_risk_veto,soft_multi_prototype_score_head,anchor_density_one_class_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_LOOO48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,support_evidence_overridden_background_gates,soft_multi_prototype_score_head,anchor_density_uncertain_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_CONSTRAIN48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_support_class_envelope_gate,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,anchor_density_gate,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_ENVELOPE48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_support_class_envelope_gate,post_reject_retention_rescue,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,anchor_density_gate,pseudo_background_cap,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_RESCUE48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_support_class_envelope_gate,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,anchor_density_uncertain_gate,pseudo_background_cap,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_PREREJECT48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,three_way_old_seen_background_head_loss,three_way_decision_head,pseudo_background_anchors,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_THREEWAY48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,three_way_old_seen_background_head_loss,known_evidence_floor_before_background_reject,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_COVFLOOR48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,class_first_known_assignment,three_way_old_seen_background_head_loss,known_evidence_floor_before_background_veto,pre_reject_defer_arbitration,retention_rescue,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_CLASSFIRST48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,evidence_balanced_old_background_head,known_support_evidence_required_for_accept,pseudo_background_competition,pre_reject_defer_arbitration,known_coverage_margin_loss,source_leave_one_old_out_meta_unknown_loss,constrained_retention_risk_selector,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_EVIBG48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,prototype_mixture_soft_target,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,decoupled_background_risk_veto,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_SOFTTARGET48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,negative_anchor_background_basin,prototype_mixture_soft_target,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,decoupled_background_risk_veto,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_NEGANCHOR48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,class_conditional_density_shell_inlier_gate,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_DENSHELL48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,identity_consensus_arbitration,class_conditional_density_shell_inlier_gate,soft_multi_prototype_score_head,support_center_geometry,known_coverage_margin_loss,support_retention_guard,source_leave_one_old_out_meta_unknown_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_IDCONS48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,identity_consensus_arbitration,support_conformal_arbitration,class_conditional_support_floor,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_CONFORM48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,identity_consensus_arbitration,support_reconstruction_arbitration,class_local_low_rank_residual,reciprocal_boundary_negatives,soft_multi_prototype_score_head,support_center_geometry,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_RECON48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_leave_one_old_out_impostor_risk_arbitration,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,negative_anchor_background_basin_when_aggressive,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_SOURCERISK48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,source_leave_one_old_out_impostor_risk_arbitration,source_risk_constrained_support_retention,pre_reject_defer_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_RISKRET48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,support_manifold_soft_mixture_consistency,source_risk_constrained_support_retention,pre_reject_defer_arbitration,soft_multi_prototype_score_head,known_coverage_margin_loss,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_MANIFOLD48"
                                else "h06_latest_source_prototypes,target_old_support,old_class_radius,U_orbit,old_drift_support_quality,support_knn,source_looo_risk,pre_reject_support_retention,pair_verifier_actionable_veto,three_way_background,soft_multi_prototype_score_head,simplified_leo_residual_channel"
                                if plan == "OA_MSE_H06_ARB48"
                                else "h06_latest_source_prototypes,target_old_support,old_class_radius,U_orbit,old_drift_support_quality,support_knn,source_risk_constrained_retention,pre_reject_defer,pair_verifier,soft_multi_prototype_score_head,simplified_leo_residual_channel"
                                if plan == "OA_MSE_H06_EVID48"
                                else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,old_drift_support_quality,old_support_knn,identity_consensus,pair_verifier_late_unknown_veto,background_late_unknown_veto,simplified_leo_residual_channel"
                                if plan == "OA_MSE_H06_OLDUNK48"
                                else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,old_background_two_way_head,query_free_pseudo_background,source_looo_risk,support_reconstruction_when_enabled,old_drift_support_quality,old_support_knn,identity_consensus,simplified_leo_residual_channel"
                                if plan == "OA_MSE_H06_BGTRAIN48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,soft_multi_prototype_score_head,old_support_knn,old_drift_gate,class_envelope,old_primary_terminal_gate,unknown_risk_veto_blocks_rescue,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDFIRST48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,soft_multi_prototype_score_head,old_support_knn,old_drift_gate,class_envelope_required_old_primary_consensus,retention_rescue_candidate_only,old_primary_terminal_gate,pre_reject_unknown_risk_veto_blocks_rescue,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDRELAX48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_center_geometry,target_shift_halo_ring_pseudo_unknowns,soft_multi_prototype_score_head,mixture_consistency,old_primary_measurement_gate,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDGEOM48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_conformal_arbitration,support_reconstruction_arbitration,pre_reject_support_retention,retention_rescue_candidate_only,old_primary_promote_rescue_candidates,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDCONF48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,soft_acceptance_budget,support_conformal_defer,support_reconstruction_defer,old_primary_rescue_budget,source_looo_defer_budget,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDBUDGET48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_quality_prototype_construction,support_center_leave_one_out_metric_loss,soft_multi_prototype_score_head,mixture_consistency,support_conformal_defer,support_reconstruction_defer,source_looo_quality_probe,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDQUAL48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,query_free_background_risk,source_looo_risk,two_branch_pseudo_background_guard,pre_reject_defer,unknown_score_joint_veto,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDRISK48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,oldqual_oldrisk_fusion,oldqual_support_quality,oldrisk_query_free_background_risk,rollback_calibration,pre_reject_defer,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDFUSE48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,rollback_safe_retention,retention_rescue_candidate_only,defer_first_deployment_gate,pre_reject_support_retention,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,soft_multi_prototype_score_head,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_ROLLSAFE48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,oldhead_ridge_recoverability,knn_density_boundary_risk,density_shell_guard,support_conformal_reconstruction,three_way_old_background_head,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDHEAD48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,support_cv_stability_head,identity_preserving_cv_selector,support_background_cap,density_shell_guard,support_conformal_reconstruction,three_way_old_background_head,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDHEADFAR48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,target_old_recoverability_first,target_only_ridge_upper_bound,oldrecov_ridge_head,oldrecov_proto_bridge,support_conformal_reconstruction_defer,retention_rescue_candidate_only,source_looo_risk,two_branch_pseudo_background_guard,old_unknown_acceptance_guard,unknown_query_eval_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_OLDRECOV48"
                            else "h06_latest_source_prototypes,target_old_support_only,no_target_new_support,old_class_radius,U_orbit,old_proof_first,old_drift_support_quality,identity_consensus_background_cap,three_way_background_prob_as_soft_risk,unknown_score_joint_veto,support_reconstruction_conformal_defer_only,simplified_leo_residual_channel"
                            if plan == "OA_MSE_H06_RETOLD48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,support_calibrated_background_cap,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,retention_rescue,negative_anchor_background_basin,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_BGCAP48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,support_neighborhood_known_retention,pre_reject_defer_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,source_leave_one_old_out_risk_when_enabled,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_KRET48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,support_leave_one_out_adapter_selector,identity_consensus_arbitration,prototype_mixture_soft_target,soft_multi_prototype_score_head,known_coverage_margin_loss,source_leave_one_old_out_risk_when_enabled,simplified_leo_residual_channel,multi_target_receiver_pool"
                                if plan == "OA_MSE_SUPPORTCV48"
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector,soft_prototype_mixture,pseudo_unknown_void_background_gate"
                                if plan in {"OA_MSE_SOFTVOID128", "OA_MSE_ANCHORGUARD128"}
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,target_boundary_guard_selector"
                                if plan
                                in {
                                    "OA_MSE_BOUNDARY32",
                                    "OA_MSE_UNCERTAIN32",
                                    "OA_MSE_VETO32",
                                    "OA_MSE_CLASSCOND32",
                                    "OA_MSE_CALGUARD32",
                                    "OA_MSE_BALANCE64",
                                    "OA_MSE_SOFTMIX64",
                                    "OA_MSE_VOID64",
                                    "OA_MSE_SOFTVOID128",
                                    "OA_MSE_ANCHORGUARD128",
                                    "OA_MSE_MIXHEAD128",
                                    "OA_MSE_STRUCT48",
                                    "OA_MSE_SIMPLIFIED48",
                                    "OA_MSE_RETENTION48",
                                    "OA_MSE_SUPPORTRET48",
                                    "OA_MSE_TWOBRANCH48",
                                    "OA_MSE_REGHEAD48",
                                    "OA_MSE_GEOM48",
                                    "OA_MSE_TRIAGE48",
                                    "OA_MSE_LOOO48",
                                    "OA_MSE_CONSTRAIN48",
                                    "OA_MSE_ENVELOPE48",
                                    "OA_MSE_RESCUE48",
                                    "OA_MSE_PREREJECT48",
                                    "OA_MSE_THREEWAY48",
                                }
                                else "source_old_prototypes,target_old_support,old_class_radius,U_orbit,proxy_alpha_selector"
                            )
                        ),
                        target_old_leo_support=True,
                        target_new_leo_support=is_seen_new,
                        target_old_tx_ids="${TARGET_OLD_TX_IDS}",
                        target_old_support_per_tx=k_old,
                        target_new_support_per_tx=k_new,
                        source_proto_per_tx=int(spec.get("source_proto_per_tx", 20)),
                        source_query_per_tx=int(spec.get("source_query_per_tx", 20)),
                        target_old_query_per_tx=int(spec.get("target_old_query_per_tx", 50)),
                        query_per_tx=int(spec.get("query_per_tx", 50)),
                        sfe_max_samples_per_tx=int(spec.get("sfe_max_samples_per_tx", 200)),
                        stage2_max_active_per_gpu=(
                            6
                            if plan == "OA_MSE_SOFTVOID128"
                            else 4
                            if plan in {"OA_MSE_BALANCE64", "OA_MSE_SOFTMIX64", "OA_MSE_VOID64", "OA_MSE_ANCHORGUARD128", "OA_MSE_MIXHEAD128", "OA_MSE_STRUCT48", "OA_MSE_SIMPLIFIED48"}
                            else 2
                            if plan in {"OA_MSE_RETENTION48", "OA_MSE_SUPPORTRET48", "OA_MSE_TWOBRANCH48", "OA_MSE_REGHEAD48", "OA_MSE_GEOM48", "OA_MSE_TRIAGE48", "OA_MSE_LOOO48", "OA_MSE_CONSTRAIN48", "OA_MSE_ENVELOPE48", "OA_MSE_RESCUE48", "OA_MSE_PREREJECT48", "OA_MSE_THREEWAY48", "OA_MSE_COVFLOOR48", "OA_MSE_CLASSFIRST48", "OA_MSE_EVIBG48", "OA_MSE_SOFTTARGET48", "OA_MSE_NEGANCHOR48", "OA_MSE_DENSHELL48", "OA_MSE_IDCONS48", "OA_MSE_CONFORM48", "OA_MSE_RECON48", "OA_MSE_SOURCERISK48", "OA_MSE_SUPPORTCV48", "OA_MSE_BGCAP48", "OA_MSE_KRET48", "OA_MSE_RISKRET48", "OA_MSE_MANIFOLD48", "OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"}
                            else None
                        ),
                        seed=seed + int(spec.get("seed_offset", 0)),
                        **candidate_kwargs,
                    )
                )
        return _with_phase1_ground_rows(plan, rows)
    raise ValueError(f"unknown plan: {plan}")


def _candidate_command(candidate: Candidate) -> str:
    if candidate.command_kind == "phase1_safe_ssdg_ground_train":
        epochs = int(candidate.epochs or 200)
        variant_spec = PHASE1_GPU0_JOINTSAFE_VARIANTS.get(str(candidate.phase1_variant or "").lower())
        if variant_spec is not None:
            label_epochs = int(variant_spec["label_epochs"])
            pseudo_epochs = int(variant_spec["pseudo_epochs"])
            ema = "true" if bool(variant_spec["use_ema_teacher"]) else "false"
            return (
                "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" CUDA_VISIBLE_DEVICES=\"${GPU}\" "
                "\"${PYTHON}\" -u \"${ROOT}/code/SSDG/train_ssdg.py\" "
                "--wisig_pkl \"${WISIG_PKL}\" --split_mode tx_rx_day_1_7_2 "
                "--labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 "
                f"--output_dir \"${{RUNS_ROOT}}/{candidate.cid}\" --epochs {epochs} "
                f"--label_epochs {label_epochs} --pseudo_epochs {pseudo_epochs} --from_scratch true "
                "--best_metric joint_safe --enable_joint_safe_guard true "
                "--one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true "
                "--paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 "
                "--paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 "
                "--use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true "
                "--use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true "
                "--lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 "
                "--lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 "
                "--use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only "
                f"--sat_train_scenario leo_clear_weak --sat_train_scenarios {SIMPLIFIED_LEO_SCENARIOS} "
                f"--sat_view_schedule \"{PHASE1_PAIC_SAT_VIEW_SCHEDULE}\" "
                f"--sat_cons_start_epoch 80 --lambda_sat_cls {float(variant_spec['lambda_sat_cls']):.6g} "
                f"--lambda_sat_cons {float(variant_spec['lambda_sat_cons']):.6g} "
                f"--lambda_u {float(variant_spec['lambda_u']):.6g} --lambda_ent 0.01 "
                f"--lambda_domain {float(variant_spec['lambda_domain']):.6g} --lambda_adv 0.35 "
                f"--lambda_group_ce {float(variant_spec['lambda_group_ce']):.6g} "
                f"--lambda_fishr {float(variant_spec['lambda_fishr']):.6g} "
                f"--tau_min {float(variant_spec['tau_min']):.6g} --tau_max {float(variant_spec['tau_max']):.6g} "
                f"--pseudo_quantile {float(variant_spec['pseudo_quantile']):.6g} --use_ema_teacher {ema} "
                f"--eval_sat_channel true --eval_sat_scenarios {SIMPLIFIED_LEO_SCENARIOS} "
                "--sat_eval_max_batches -1 --device cuda:0 "
                f"--seed {int(candidate.seed)}"
            )
        return (
            "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" CUDA_VISIBLE_DEVICES=\"${GPU}\" "
            "\"${PYTHON}\" -u \"${ROOT}/code/SSDG/train_ssdg.py\" "
            "--wisig_pkl \"${WISIG_PKL}\" --split_mode tx_rx_day_1_7_2 "
            "--labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 "
            f"--output_dir \"${{RUNS_ROOT}}/{candidate.cid}\" --epochs {epochs} --from_scratch true "
            "--use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only "
            f"--sat_train_scenario leo_clear_weak --sat_train_scenarios {SIMPLIFIED_LEO_SCENARIOS} "
            f"--sat_view_schedule \"{PHASE1_PAIC_SAT_VIEW_SCHEDULE}\" "
            "--sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 "
            f"--eval_sat_channel true --eval_sat_scenarios {SIMPLIFIED_LEO_SCENARIOS} "
            "--sat_eval_max_batches -1 --device cuda:0 "
            f"--seed {int(candidate.seed)}"
        )
    if candidate.command_kind == "feature_sfe_synthetic":
        return (
            "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" "
            "\"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" "
            f"--protocol sfe --dry_run_synthetic --shots {candidate.k} "
            "--unknown_threshold 0.70 "
            f"--output_json \"${{RUNS_ROOT}}/{candidate.cid}/metrics.json\""
        )
    if candidate.command_kind == "target_adapt_labeled_rxtx":
        epochs = int(candidate.epochs or 20)
        adapt_steps = int(candidate.adapt_steps_per_epoch or 20)
        eval_max_batches = int(candidate.eval_max_batches)
        sat_eval_max_batches = int(candidate.sat_eval_max_batches)
        return (
            "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" CUDA_VISIBLE_DEVICES=\"${GPU}\" "
            "\"${PYTHON}\" -u \"${ROOT}/code/train_target_adapt.py\" "
            "--teacher_ckpt \"${TEACHER_CKPT}\" "
            f"--output_dir \"${{RUNS_ROOT}}/{candidate.cid}\" "
            "--dataset wisig --wisig_pkl \"${WISIG_PKL}\" --wisig_equalized 1 --wisig_domain rx_day "
            "--target_loader test_unseen_day_unseen_rx "
            "--target_channel_view satellite "
            "--target_label_mode labeled "
            f"--target_samples_per_rx_tx {candidate.k} "
            "--target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit "
            f"--epochs {epochs} --adapt_steps_per_epoch {adapt_steps} --target_batch_size 32 "
            "--lr_adapt 1e-4 --entropy_weight 0 --consistency_weight 0 --pseudo_weight 0 "
            "--anchor_weight 0.05 --eval_detail_every 5 "
            f"--target_adapter_type {candidate.target_adapter_type} "
            f"--adapter_rank {int(candidate.adapter_rank)} --adapter_bottleneck {int(candidate.adapter_bottleneck)} "
            f"--adapter_alpha {float(candidate.adapter_alpha)} --adapter_dropout {float(candidate.adapter_dropout)} "
            f"--freeze_base_stats {'true' if bool(candidate.freeze_base_stats) else 'false'} "
            f"--update_norm {'false' if candidate.target_adapter_type != 'logit_calibration' else 'true'} "
            f"--update_classifier {'false' if candidate.target_adapter_type != 'logit_calibration' else 'false'} "
            "--rollback_enabled true "
            "--eval_sat_channel true --eval_sat_on test_unseen_day_unseen_rx "
            "--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit "
            f"--sat_eval_max_batches {sat_eval_max_batches} --eval_max_batches {eval_max_batches}"
        )
    if candidate.command_kind == "feature_sfe_wisig_nonoverlap":
        source_tx_ids = candidate.source_tx_ids or "${SOURCE_TX_IDS}"
        new_tx_ids = candidate.new_tx_ids or "${NEW_TX_IDS}"
        unknown_tx_ids = candidate.unknown_tx_ids or "${UNKNOWN_TX_IDS}"
        gate_args = (
            "--unknown_threshold " + str(float(candidate.unknown_threshold)) + " "
            "--gate_mode " + str(candidate.gate_mode) + " "
            "--openmax_tail_size " + str(int(candidate.openmax_tail_size)) + " "
            "--openmax_quantile " + str(float(candidate.openmax_quantile)) + " "
            "--openmax_min_threshold " + str(float(candidate.openmax_min_threshold)) + " "
        )
        if candidate.min_margin is not None:
            gate_args += "--min_margin " + str(float(candidate.min_margin)) + " "
        if candidate.max_mahalanobis is not None:
            gate_args += "--max_mahalanobis " + str(float(candidate.max_mahalanobis)) + " "
        return (
            "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" CUDA_VISIBLE_DEVICES=\"${GPU}\" "
            "bash -lc \"set -euo pipefail; "
            "mkdir -p \\\"${RUNS_ROOT}/" + candidate.cid + "\\\"; "
            "\\\"${PYTHON}\\\" -u \\\"${ROOT}/code/export_spaceborne_features.py\\\" "
            "--ckpt \\\"${TEACHER_CKPT}\\\" "
            "--wisig_pkl \\\"${WISIG_PKL}\\\" "
            "--new_wisig_pkl \\\"${NEW_WISIG_PKL}\\\" "
            "--out_npz \\\"${RUNS_ROOT}/" + candidate.cid + "/features.npz\\\" "
            "--feature_name z_id "
            "--source_tx_ids \\\"" + source_tx_ids + "\\\" "
            "--new_tx_ids \\\"" + new_tx_ids + "\\\" "
            "--unknown_tx_ids \\\"" + unknown_tx_ids + "\\\" "
            "--wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 "
            "--max_samples_per_combo \\\"" + str(candidate.sfe_max_samples_per_combo) + "\\\" "
            "--max_samples_per_tx \\\"" + str(candidate.sfe_max_samples_per_tx) + "\\\" "
            "--batch_size \\\"" + str(candidate.export_batch_size) + "\\\" "
            "--device cuda:0 --seed " + str(int(candidate.seed)) + "; "
            "\\\"${PYTHON}\\\" -u \\\"${ROOT}/code/eval_spaceborne_fewshot.py\\\" "
            "--protocol sfe "
            "--feature_npz \\\"${RUNS_ROOT}/" + candidate.cid + "/features.npz\\\" "
            "--output_json \\\"${RUNS_ROOT}/" + candidate.cid + "/metrics.json\\\" "
            "--manifest_json \\\"${RUNS_ROOT}/" + candidate.cid + "/manifest.json\\\" "
            "--source_tx_ids \\\"" + source_tx_ids + "\\\" "
            "--new_tx_ids \\\"" + new_tx_ids + "\\\" "
            "--unknown_tx_ids \\\"" + unknown_tx_ids + "\\\" "
            "--shots " + str(candidate.k) + " "
            "--source_proto_per_tx \\\"" + str(candidate.source_proto_per_tx) + "\\\" "
            "--source_query_per_tx \\\"" + str(candidate.source_query_per_tx) + "\\\" "
            "--query_per_tx \\\"" + str(candidate.query_per_tx) + "\\\" "
            + gate_args +
            "--seed " + str(int(candidate.seed)) + "\""
        )
    if candidate.command_kind == "feature_oa_mse_wisig_nonoverlap":
        source_tx_ids = candidate.source_tx_ids or "${SOURCE_TX_IDS}"
        target_old_tx_ids = candidate.target_old_tx_ids or ("${TARGET_OLD_TX_IDS}" if candidate.target_old_leo_support else "")
        disable_new_tx = str(candidate.new_tx_ids or "").strip().upper() in {"__NONE__", "NONE", "OLD_UNKNOWN_ONLY"}
        new_tx_ids = "" if disable_new_tx else (candidate.new_tx_ids or "${NEW_TX_IDS}")
        unknown_tx_ids = candidate.unknown_tx_ids or "${OA_MSE_UNKNOWN_TX_IDS}"
        source_rxs = candidate.source_rxs or "${CEN51_TRAIN_RXS}"
        target_receiver_ids = candidate.target_receiver_ids or "${TARGET_RECEIVER_IDS}"
        eval_protocol = candidate.eval_protocol or {
            "mse_lite": "source_open_set",
            "mse_subspace": "ftrc",
            "oa_mse_head": "sfe",
        }.get(str(candidate.oa_mse_stage), "sfe")
        eval_shots = 0 if str(eval_protocol) == "ftrc" else int(candidate.k)
        target_old_export_args = ""
        target_old_eval_args = ""
        if target_old_tx_ids:
            target_old_export_args = (
                "--target_old_tx_ids \\\"" + target_old_tx_ids + "\\\" "
                "--target_old_rxs \\\"" + target_receiver_ids + "\\\" "
                "--target_old_channel_view satellite "
            )
            target_old_eval_args = (
                "--target_old_tx_ids \\\"" + target_old_tx_ids + "\\\" "
                "--target_old_support_per_tx \\\"" + str(int(candidate.target_old_support_per_tx)) + "\\\" "
                "--target_old_query_per_tx \\\"" + str(int(candidate.target_old_query_per_tx)) + "\\\" "
            )
        target_new_export_args = ""
        target_new_eval_args = ""
        if new_tx_ids:
            target_new_export_args = (
                "--new_tx_ids \\\"" + new_tx_ids + "\\\" "
                "--new_rxs \\\"" + target_receiver_ids + "\\\" "
                "--target_new_channel_view satellite "
                "--target_new_sat_scenarios " + str(candidate.target_channel_scenarios) + " "
            )
            target_new_eval_args = "--new_tx_ids \\\"" + new_tx_ids + "\\\" "
        gate_args = (
            "--unknown_threshold " + str(float(candidate.unknown_threshold)) + " "
            "--gate_mode oa_mse "
            "--openmax_tail_size " + str(int(candidate.openmax_tail_size)) + " "
            "--openmax_quantile " + str(float(candidate.openmax_quantile)) + " "
            "--openmax_min_threshold " + str(float(candidate.openmax_min_threshold)) + " "
        )
        siamese_veto_args = "--oa_mse_siamese_unknown_veto " if bool(candidate.oa_mse_siamese_unknown_veto) else ""
        if bool(candidate.oa_mse_siamese_unknown_veto):
            siamese_veto_args += "--oa_mse_siamese_unknown_veto_mode " + str(candidate.oa_mse_siamese_unknown_veto_mode) + " "
        optional_veto_thresholds = (
            ("--oa_mse_siamese_min_old_support_evidence_delta", candidate.oa_mse_siamese_min_old_support_evidence_delta),
            (
                "--oa_mse_siamese_min_old_surrogate_reject_delta",
                candidate.oa_mse_siamese_min_old_surrogate_reject_delta,
            ),
            ("--oa_mse_siamese_min_energy_delta", candidate.oa_mse_siamese_min_energy_delta),
            ("--oa_mse_siamese_min_mahalanobis_delta", candidate.oa_mse_siamese_min_mahalanobis_delta),
            ("--oa_mse_siamese_min_accept_delta", candidate.oa_mse_siamese_min_accept_delta),
            ("--oa_mse_siamese_min_old_support_anchor_margin", candidate.oa_mse_siamese_min_old_support_anchor_margin),
        )
        for flag, value in optional_veto_thresholds:
            if value is not None:
                siamese_veto_args += flag + " " + str(float(value)) + " "
        if bool(candidate.oa_mse_siamese_unknown_veto):
            siamese_veto_args += "--oa_mse_siamese_min_veto_failures " + str(int(candidate.oa_mse_siamese_min_veto_failures)) + " "
        old_unknown_guard_args = (
            "--oa_mse_old_unknown_acceptance_guard " if bool(candidate.oa_mse_old_unknown_acceptance_guard) else ""
        )
        optional_old_unknown_guard_thresholds = (
            (
                "--oa_mse_old_unknown_guard_min_old_support_evidence_delta",
                candidate.oa_mse_old_unknown_guard_min_old_support_evidence_delta,
            ),
            (
                "--oa_mse_old_unknown_guard_min_old_surrogate_reject_delta",
                candidate.oa_mse_old_unknown_guard_min_old_surrogate_reject_delta,
            ),
            ("--oa_mse_old_unknown_guard_min_energy_delta", candidate.oa_mse_old_unknown_guard_min_energy_delta),
            (
                "--oa_mse_old_unknown_guard_min_mahalanobis_delta",
                candidate.oa_mse_old_unknown_guard_min_mahalanobis_delta,
            ),
            ("--oa_mse_old_unknown_guard_min_accept_delta", candidate.oa_mse_old_unknown_guard_min_accept_delta),
            (
                "--oa_mse_old_unknown_guard_min_old_support_anchor_margin",
                candidate.oa_mse_old_unknown_guard_min_old_support_anchor_margin,
            ),
            ("--oa_mse_old_unknown_guard_min_best_old_score", candidate.oa_mse_old_unknown_guard_min_best_old_score),
            ("--oa_mse_old_unknown_guard_min_margin", candidate.oa_mse_old_unknown_guard_min_margin),
        )
        for flag, value in optional_old_unknown_guard_thresholds:
            if value is not None:
                old_unknown_guard_args += flag + " " + str(float(value)) + " "
        if bool(candidate.oa_mse_old_unknown_acceptance_guard):
            old_unknown_guard_args += (
                "--oa_mse_old_unknown_guard_min_failures "
                + str(int(candidate.oa_mse_old_unknown_guard_min_failures))
                + " "
            )
        void_gate_args = "--oa_mse_void_gate " if bool(candidate.oa_mse_void_gate) else ""
        two_branch_guard_args = (
            "--oa_mse_two_branch_background_guard "
            if bool(candidate.oa_mse_two_branch_background_guard)
            else ""
        )
        seen_new_registration_args = (
            "--oa_mse_seen_new_registration_override "
            if bool(candidate.oa_mse_seen_new_registration_override)
            else ""
        )
        for flag, value in (
            (
                "--seen_new_override_min_support_knn_seen_new_minus_old",
                candidate.seen_new_override_min_support_knn_seen_new_minus_old,
            ),
            ("--seen_new_override_min_support_knn_margin", candidate.seen_new_override_min_support_knn_margin),
        ):
            if value is not None:
                seen_new_registration_args += flag + " " + str(float(value)) + " "
        return (
            "env PYTHONPATH=\"${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" CUDA_VISIBLE_DEVICES=\"${GPU}\" "
            "bash -lc \"set -euo pipefail; "
            "mkdir -p \\\"${RUNS_ROOT}/" + candidate.cid + "\\\"; "
            "\\\"${PYTHON}\\\" -u \\\"${ROOT}/code/export_spaceborne_features.py\\\" "
            "--ckpt \\\"${TEACHER_CKPT}\\\" "
            "--wisig_pkl \\\"${WISIG_PKL}\\\" "
            "--new_wisig_pkl \\\"${NEW_WISIG_PKL}\\\" "
            "--out_npz \\\"${RUNS_ROOT}/" + candidate.cid + "/features.npz\\\" "
            "--feature_name z_id "
            "--source_tx_ids \\\"" + source_tx_ids + "\\\" "
            "--source_rxs \\\"" + source_rxs + "\\\" "
            + target_old_export_args +
            target_new_export_args +
            "--unknown_tx_ids \\\"" + unknown_tx_ids + "\\\" "
            "--star_ground_channel_impl " + str(candidate.star_ground_channel_impl) + " "
            "--target_old_sat_scenarios " + str(candidate.target_channel_scenarios) + " "
            "--wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 "
            "--max_samples_per_combo \\\"" + str(candidate.sfe_max_samples_per_combo) + "\\\" "
            "--max_samples_per_tx \\\"" + str(candidate.sfe_max_samples_per_tx) + "\\\" "
            "--batch_size \\\"" + str(candidate.export_batch_size) + "\\\" "
            "--device cuda:0 --seed " + str(int(candidate.seed)) + "; "
            "\\\"${PYTHON}\\\" -u \\\"${ROOT}/code/eval_spaceborne_fewshot.py\\\" "
            "--protocol " + str(eval_protocol) + " "
            "--feature_npz \\\"${RUNS_ROOT}/" + candidate.cid + "/features.npz\\\" "
            "--output_json \\\"${RUNS_ROOT}/" + candidate.cid + "/metrics.json\\\" "
            "--manifest_json \\\"${RUNS_ROOT}/" + candidate.cid + "/manifest.json\\\" "
            "--score_table_csv \\\"${RUNS_ROOT}/" + candidate.cid + "/score_table.csv\\\" "
            "--source_tx_ids \\\"" + source_tx_ids + "\\\" "
            + target_old_eval_args +
            target_new_eval_args +
            "--unknown_tx_ids \\\"" + unknown_tx_ids + "\\\" "
            "--shots " + str(eval_shots) + " "
            "--source_proto_per_tx \\\"" + str(candidate.source_proto_per_tx) + "\\\" "
            "--source_query_per_tx \\\"" + str(candidate.source_query_per_tx) + "\\\" "
            "--query_per_tx \\\"" + str(candidate.query_per_tx) + "\\\" "
            + gate_args +
            "--oa_mse_adapter_rank 2 "
            "--oa_mse_adapter_kind " + str(candidate.oa_mse_adapter_kind) + " "
            "--oa_mse_adapter_steps " + str(int(candidate.max_adapt_steps)) + " "
            "--oa_mse_adapter_selection_policy " + str(candidate.oa_mse_adapter_selection_policy) + " "
            + ("--oa_mse_adapter_alpha_eval_sweep " if bool(candidate.oa_mse_adapter_alpha_eval_sweep) else "")
            + "--oa_mse_source_anchor_weight " + str(float(candidate.oa_mse_source_anchor_weight)) + " "
            "--oa_mse_source_ce_weight " + str(float(candidate.oa_mse_source_ce_weight)) + " "
            "--oa_mse_unknown_moat_weight " + str(float(candidate.oa_mse_unknown_moat_weight)) + " "
            "--oa_mse_unknown_moat_margin " + str(float(candidate.oa_mse_unknown_moat_margin)) + " "
            "--pseudo_unknown_samples_per_pair " + str(int(candidate.pseudo_unknown_samples_per_pair)) + " "
            "--pseudo_unknown_offset_scale " + str(float(candidate.pseudo_unknown_offset_scale)) + " "
            "--pseudo_unknown_source_boundary_samples_per_pair " + str(int(candidate.pseudo_unknown_source_boundary_samples_per_pair)) + " "
            "--pseudo_unknown_source_boundary_offset_scale " + str(float(candidate.pseudo_unknown_source_boundary_offset_scale)) + " "
            "--pseudo_unknown_target_shift_samples_per_class " + str(int(candidate.pseudo_unknown_target_shift_samples_per_class)) + " "
            "--pseudo_unknown_target_shift_offset_scale " + str(float(candidate.pseudo_unknown_target_shift_offset_scale)) + " "
            "--pseudo_unknown_target_halo_samples_per_class " + str(int(candidate.pseudo_unknown_target_halo_samples_per_class)) + " "
            "--pseudo_unknown_target_halo_offset_scale " + str(float(candidate.pseudo_unknown_target_halo_offset_scale)) + " "
            "--pseudo_unknown_target_ring_samples_per_class " + str(int(candidate.pseudo_unknown_target_ring_samples_per_class)) + " "
            "--pseudo_unknown_target_ring_offset_scale " + str(float(candidate.pseudo_unknown_target_ring_offset_scale)) + " "
            "--oa_mse_old_bridge_weight " + str(float(candidate.oa_mse_old_bridge_weight)) + " "
            "--old_bridge_samples_per_class " + str(int(candidate.old_bridge_samples_per_class)) + " "
            "--old_bridge_max_mix " + str(float(candidate.old_bridge_max_mix)) + " "
            "--oa_mse_support_contrast_weight " + str(float(candidate.oa_mse_support_contrast_weight)) + " "
            "--old_support_contrast_negative_margin " + str(float(candidate.old_support_contrast_negative_margin)) + " "
            "--old_support_contrast_positive_margin " + str(float(candidate.old_support_contrast_positive_margin)) + " "
            "--oa_mse_support_center_ce_weight " + str(float(candidate.oa_mse_support_center_ce_weight)) + " "
            "--support_center_temperature " + str(float(candidate.support_center_temperature)) + " "
            "--support_center_margin " + str(float(candidate.support_center_margin)) + " "
            "--oa_mse_soft_proto_weight " + str(float(candidate.oa_mse_soft_proto_weight)) + " "
            "--soft_proto_topk " + str(int(candidate.soft_proto_topk)) + " "
            "--soft_proto_temperature " + str(float(candidate.soft_proto_temperature)) + " "
            "--oa_mse_soft_proto_boundary_weight " + str(float(candidate.oa_mse_soft_proto_boundary_weight)) + " "
            "--soft_proto_boundary_margin " + str(float(candidate.soft_proto_boundary_margin)) + " "
            + ("--oa_mse_three_way_decision_head " if bool(candidate.oa_mse_three_way_decision_head) else "")
            + "--oa_mse_three_way_head_weight " + str(float(candidate.oa_mse_three_way_head_weight)) + " "
            + "--three_way_head_temperature " + str(float(candidate.three_way_head_temperature)) + " "
            + "--three_way_head_known_margin " + str(float(candidate.three_way_head_known_margin)) + " "
            + "--three_way_head_background_margin " + str(float(candidate.three_way_head_background_margin)) + " "
            + "--three_way_head_support_ce_weight " + str(float(candidate.three_way_head_support_ce_weight)) + " "
            + "--three_way_head_pseudo_ce_weight " + str(float(candidate.three_way_head_pseudo_ce_weight)) + " "
            + "--three_way_head_support_background_margin_weight " + str(float(candidate.three_way_head_support_background_margin_weight)) + " "
            + "--three_way_head_pseudo_margin_weight " + str(float(candidate.three_way_head_pseudo_margin_weight)) + " "
            + "--three_way_accept_prob " + str(float(candidate.three_way_accept_prob)) + " "
            + "--three_way_reject_prob " + str(float(candidate.three_way_reject_prob)) + " "
            + "--three_way_defer_prob " + str(float(candidate.three_way_defer_prob)) + " "
            + "--three_way_known_background_margin " + str(float(candidate.three_way_known_background_margin)) + " "
            + "--three_way_reject_margin " + str(float(candidate.three_way_reject_margin)) + " "
            + "--three_way_old_seen_ambiguity_margin " + str(float(candidate.three_way_old_seen_ambiguity_margin)) + " "
            + "--three_way_defer_action " + str(candidate.three_way_defer_action) + " "
            + "--three_way_decision_policy " + str(candidate.three_way_decision_policy) + " "
            + ("--three_way_known_floor " if bool(candidate.three_way_known_floor) else "")
            + "--three_way_known_floor_action " + str(candidate.three_way_known_floor_action) + " "
            + "--three_way_known_floor_old_min_evidence_delta " + str(float(candidate.three_way_known_floor_old_min_evidence_delta)) + " "
            + "--three_way_known_floor_old_min_anchor_delta " + str(float(candidate.three_way_known_floor_old_min_anchor_delta)) + " "
            + "--three_way_known_floor_old_min_anchor_margin " + str(float(candidate.three_way_known_floor_old_min_anchor_margin)) + " "
            + "--three_way_known_floor_old_min_score_margin " + str(float(candidate.three_way_known_floor_old_min_score_margin)) + " "
            + "--three_way_known_floor_seen_new_min_evidence_delta " + str(float(candidate.three_way_known_floor_seen_new_min_evidence_delta)) + " "
            + "--three_way_known_floor_seen_new_min_anchor_delta " + str(float(candidate.three_way_known_floor_seen_new_min_anchor_delta)) + " "
            + "--three_way_known_floor_seen_new_min_score_margin " + str(float(candidate.three_way_known_floor_seen_new_min_score_margin)) + " "
            + "--three_way_known_floor_background_override_prob " + str(float(candidate.three_way_known_floor_background_override_prob)) + " "
            + "--three_way_known_floor_background_override_margin " + str(float(candidate.three_way_known_floor_background_override_margin)) + " "
            + ("--oa_mse_multiproto_score " if bool(candidate.oa_mse_multiproto_score) else "")
            + "--multiproto_topk " + str(int(candidate.multiproto_topk)) + " "
            + "--multiproto_temperature " + str(float(candidate.multiproto_temperature)) + " "
            + "--multiproto_score_weight " + str(float(candidate.multiproto_score_weight)) + " "
            + ("--oa_mse_mixture_consistency_gate " if bool(candidate.oa_mse_mixture_consistency_gate) else "")
            + "--mixture_consistency_min_cos " + str(float(candidate.mixture_consistency_min_cos)) + " "
            + "--mixture_consistency_max_residual " + str(float(candidate.mixture_consistency_max_residual)) + " "
            + "--mixture_consistency_min_margin " + str(float(candidate.mixture_consistency_min_margin)) + " "
            + "--mixture_consistency_action " + str(candidate.mixture_consistency_action) + " "
            + ("--oa_mse_anchor_density_gate " if bool(candidate.oa_mse_anchor_density_gate) else "")
            + "--anchor_density_topk " + str(int(candidate.anchor_density_topk)) + " "
            + "--anchor_density_temperature " + str(float(candidate.anchor_density_temperature)) + " "
            + "--anchor_density_min_quantile " + str(float(candidate.anchor_density_min_quantile)) + " "
            + "--anchor_density_margin_quantile " + str(float(candidate.anchor_density_margin_quantile)) + " "
            + "--anchor_density_gate_action " + str(candidate.anchor_density_gate_action) + " "
            + ("--oa_mse_class_envelope_gate " if bool(candidate.oa_mse_class_envelope_gate) else "")
            + "--class_envelope_evidence_quantile " + str(float(candidate.class_envelope_evidence_quantile)) + " "
            + "--class_envelope_residual_quantile " + str(float(candidate.class_envelope_residual_quantile)) + " "
            + "--class_envelope_score_quantile " + str(float(candidate.class_envelope_score_quantile)) + " "
            + "--class_envelope_margin_quantile " + str(float(candidate.class_envelope_margin_quantile)) + " "
            + "--class_envelope_evidence_slack " + str(float(candidate.class_envelope_evidence_slack)) + " "
            + "--class_envelope_residual_slack " + str(float(candidate.class_envelope_residual_slack)) + " "
            + "--class_envelope_score_slack " + str(float(candidate.class_envelope_score_slack)) + " "
            + "--class_envelope_margin_slack " + str(float(candidate.class_envelope_margin_slack)) + " "
            + "--class_envelope_min_failures " + str(int(candidate.class_envelope_min_failures)) + " "
            + "--class_envelope_gate_action " + str(candidate.class_envelope_gate_action) + " "
            + ("--oa_mse_old_primary_gate " if bool(candidate.oa_mse_old_primary_gate) else "")
            + "--old_primary_min_old_support_evidence_delta " + str(float(candidate.old_primary_min_old_support_evidence_delta)) + " "
            + "--old_primary_min_old_support_anchor_delta " + str(float(candidate.old_primary_min_old_support_anchor_delta)) + " "
            + "--old_primary_min_old_support_anchor_margin " + str(float(candidate.old_primary_min_old_support_anchor_margin)) + " "
            + "--old_primary_min_score_margin " + str(float(candidate.old_primary_min_score_margin)) + " "
            + ("--old_primary_require_soft_mixture " if bool(candidate.old_primary_require_soft_mixture) else "")
            + "--old_primary_min_soft_mixture_margin " + str(float(candidate.old_primary_min_soft_mixture_margin)) + " "
            + "--old_primary_min_soft_mixture_cos " + str(float(candidate.old_primary_min_soft_mixture_cos)) + " "
            + "--old_primary_max_soft_mixture_residual " + str(float(candidate.old_primary_max_soft_mixture_residual)) + " "
            + ("--old_primary_require_support_knn " if bool(candidate.old_primary_require_support_knn) else "")
            + (
                ""
                if bool(candidate.old_primary_require_support_knn_label_match)
                else "--old_primary_no_support_knn_label_match "
            )
            + "--old_primary_min_support_knn_margin " + str(float(candidate.old_primary_min_support_knn_margin)) + " "
            + (
                "--old_primary_max_support_knn_seen_new_minus_old "
                + str(float(candidate.old_primary_max_support_knn_seen_new_minus_old))
                + " "
                if candidate.old_primary_max_support_knn_seen_new_minus_old is not None
                else ""
            )
            + "--old_primary_min_old_drift_cos " + str(float(candidate.old_primary_min_old_drift_cos)) + " "
            + "--old_primary_max_old_drift_dist " + str(float(candidate.old_primary_max_old_drift_dist)) + " "
            + ("--old_primary_require_class_envelope " if bool(candidate.old_primary_require_class_envelope) else "")
            + "--old_primary_unknown_veto_background_score " + str(float(candidate.old_primary_unknown_veto_background_score)) + " "
            + "--old_primary_unknown_veto_background_margin " + str(float(candidate.old_primary_unknown_veto_background_margin)) + " "
            + "--old_primary_unknown_veto_min_sources " + str(int(candidate.old_primary_unknown_veto_min_sources)) + " "
            + "--old_primary_fail_action " + str(candidate.old_primary_fail_action) + " "
            + "--old_primary_unknown_veto_action " + str(candidate.old_primary_unknown_veto_action) + " "
            + ("--old_primary_promote_rescue_candidates " if bool(candidate.old_primary_promote_rescue_candidates) else "")
            + ("--oa_mse_density_shell_gate " if bool(candidate.oa_mse_density_shell_gate) else "")
            + "--density_shell_old_min_evidence_delta " + str(float(candidate.density_shell_old_min_evidence_delta)) + " "
            + "--density_shell_old_min_anchor_delta " + str(float(candidate.density_shell_old_min_anchor_delta)) + " "
            + "--density_shell_old_min_density_delta " + str(float(candidate.density_shell_old_min_density_delta)) + " "
            + "--density_shell_seen_new_min_evidence_delta " + str(float(candidate.density_shell_seen_new_min_evidence_delta)) + " "
            + "--density_shell_seen_new_min_anchor_delta " + str(float(candidate.density_shell_seen_new_min_anchor_delta)) + " "
            + "--density_shell_seen_new_min_density_delta " + str(float(candidate.density_shell_seen_new_min_density_delta)) + " "
            + "--density_shell_accept_background_margin " + str(float(candidate.density_shell_accept_background_margin)) + " "
            + "--density_shell_reject_background_score " + str(float(candidate.density_shell_reject_background_score)) + " "
            + "--density_shell_reject_background_margin " + str(float(candidate.density_shell_reject_background_margin)) + " "
            + "--density_shell_reject_min_failed_shells " + str(int(candidate.density_shell_reject_min_failed_shells)) + " "
            + ("--oa_mse_identity_consensus_arbitration " if bool(candidate.oa_mse_identity_consensus_arbitration) else "")
            + "--identity_consensus_old_min_evidence_delta " + str(float(candidate.identity_consensus_old_min_evidence_delta)) + " "
            + "--identity_consensus_old_min_anchor_delta " + str(float(candidate.identity_consensus_old_min_anchor_delta)) + " "
            + "--identity_consensus_old_min_density_delta " + str(float(candidate.identity_consensus_old_min_density_delta)) + " "
            + "--identity_consensus_seen_new_min_evidence_delta " + str(float(candidate.identity_consensus_seen_new_min_evidence_delta)) + " "
            + "--identity_consensus_seen_new_min_anchor_delta " + str(float(candidate.identity_consensus_seen_new_min_anchor_delta)) + " "
            + "--identity_consensus_seen_new_min_density_delta " + str(float(candidate.identity_consensus_seen_new_min_density_delta)) + " "
            + "--identity_consensus_min_identity_margin " + str(float(candidate.identity_consensus_min_identity_margin)) + " "
            + "--identity_consensus_background_accept_margin " + str(float(candidate.identity_consensus_background_accept_margin)) + " "
            + "--identity_consensus_reject_background_score " + str(float(candidate.identity_consensus_reject_background_score)) + " "
            + "--identity_consensus_reject_background_margin " + str(float(candidate.identity_consensus_reject_background_margin)) + " "
            + "--identity_consensus_reject_min_identity_failures " + str(int(candidate.identity_consensus_reject_min_identity_failures)) + " "
            + ("--identity_consensus_support_background_cap " if bool(candidate.identity_consensus_support_background_cap) else "")
            + "--identity_consensus_support_background_cap_quantile " + str(float(candidate.identity_consensus_support_background_cap_quantile)) + " "
            + "--identity_consensus_support_background_cap_slack " + str(float(candidate.identity_consensus_support_background_cap_slack)) + " "
            + "--identity_consensus_support_background_cap_min_anchors " + str(int(candidate.identity_consensus_support_background_cap_min_anchors)) + " "
            + ("--oa_mse_support_conformal_arbitration " if bool(candidate.oa_mse_support_conformal_arbitration) else "")
            + "--support_conformal_calibration_quantile " + str(float(candidate.support_conformal_calibration_quantile)) + " "
            + "--support_conformal_conformity_slack " + str(float(candidate.support_conformal_conformity_slack)) + " "
            + "--support_conformal_anchor_margin_slack " + str(float(candidate.support_conformal_anchor_margin_slack)) + " "
            + "--support_conformal_background_score " + str(float(candidate.support_conformal_background_score)) + " "
            + "--support_conformal_background_margin " + str(float(candidate.support_conformal_background_margin)) + " "
            + "--support_conformal_hard_reject_margin " + str(float(candidate.support_conformal_hard_reject_margin)) + " "
            + "--support_conformal_reject_min_failures " + str(int(candidate.support_conformal_reject_min_failures)) + " "
            + "--support_conformal_reject_action " + str(candidate.support_conformal_reject_action) + " "
            + ("--oa_mse_support_reconstruction_arbitration " if bool(candidate.oa_mse_support_reconstruction_arbitration) else "")
            + "--support_reconstruction_rank " + str(int(candidate.support_reconstruction_rank)) + " "
            + "--support_reconstruction_residual_quantile " + str(float(candidate.support_reconstruction_residual_quantile)) + " "
            + "--support_reconstruction_residual_slack " + str(float(candidate.support_reconstruction_residual_slack)) + " "
            + "--support_reconstruction_min_residual_floor " + str(float(candidate.support_reconstruction_min_residual_floor)) + " "
            + "--support_reconstruction_negative_scale " + str(float(candidate.support_reconstruction_negative_scale)) + " "
            + "--support_reconstruction_negative_margin " + str(float(candidate.support_reconstruction_negative_margin)) + " "
            + "--support_reconstruction_hard_residual_margin " + str(float(candidate.support_reconstruction_hard_residual_margin)) + " "
            + "--support_reconstruction_background_score " + str(float(candidate.support_reconstruction_background_score)) + " "
            + "--support_reconstruction_background_margin " + str(float(candidate.support_reconstruction_background_margin)) + " "
            + "--support_reconstruction_reject_min_failures " + str(int(candidate.support_reconstruction_reject_min_failures)) + " "
            + "--support_reconstruction_reject_action " + str(candidate.support_reconstruction_reject_action) + " "
            + ("--oa_mse_pre_reject_defer_arbitration " if bool(candidate.oa_mse_pre_reject_defer_arbitration) else "")
            + "--pre_reject_old_min_evidence_delta " + str(float(candidate.pre_reject_old_min_evidence_delta)) + " "
            + "--pre_reject_old_min_anchor_delta " + str(float(candidate.pre_reject_old_min_anchor_delta)) + " "
            + "--pre_reject_old_min_anchor_margin " + str(float(candidate.pre_reject_old_min_anchor_margin)) + " "
            + "--pre_reject_old_min_score_margin " + str(float(candidate.pre_reject_old_min_score_margin)) + " "
            + "--pre_reject_seen_new_min_evidence_delta " + str(float(candidate.pre_reject_seen_new_min_evidence_delta)) + " "
            + "--pre_reject_seen_new_min_anchor_delta " + str(float(candidate.pre_reject_seen_new_min_anchor_delta)) + " "
            + "--pre_reject_seen_new_min_score_margin " + str(float(candidate.pre_reject_seen_new_min_score_margin)) + " "
            + "--pre_reject_max_background_score " + str(float(candidate.pre_reject_max_background_score)) + " "
            + "--pre_reject_max_background_margin " + str(float(candidate.pre_reject_max_background_margin)) + " "
            + "--pre_reject_defer_background_score " + str(float(candidate.pre_reject_defer_background_score)) + " "
            + "--pre_reject_defer_background_margin " + str(float(candidate.pre_reject_defer_background_margin)) + " "
            + "--pre_reject_reject_background_score " + str(float(candidate.pre_reject_reject_background_score)) + " "
            + "--pre_reject_reject_background_margin " + str(float(candidate.pre_reject_reject_background_margin)) + " "
            + "--pre_reject_defer_action " + str(candidate.pre_reject_defer_action) + " "
            + ("--pre_reject_support_neighborhood_retention " if bool(candidate.pre_reject_support_neighborhood_retention) else "")
            + "--pre_reject_support_retention_old_min_evidence_delta " + str(float(candidate.pre_reject_support_retention_old_min_evidence_delta)) + " "
            + "--pre_reject_support_retention_old_min_anchor_delta " + str(float(candidate.pre_reject_support_retention_old_min_anchor_delta)) + " "
            + "--pre_reject_support_retention_old_min_anchor_margin " + str(float(candidate.pre_reject_support_retention_old_min_anchor_margin)) + " "
            + "--pre_reject_support_retention_old_min_score_margin " + str(float(candidate.pre_reject_support_retention_old_min_score_margin)) + " "
            + "--pre_reject_support_retention_seen_new_min_evidence_delta " + str(float(candidate.pre_reject_support_retention_seen_new_min_evidence_delta)) + " "
            + "--pre_reject_support_retention_seen_new_min_anchor_delta " + str(float(candidate.pre_reject_support_retention_seen_new_min_anchor_delta)) + " "
            + "--pre_reject_support_retention_seen_new_min_score_margin " + str(float(candidate.pre_reject_support_retention_seen_new_min_score_margin)) + " "
            + "--pre_reject_support_retention_max_background_score " + str(float(candidate.pre_reject_support_retention_max_background_score)) + " "
            + "--pre_reject_support_retention_max_background_margin " + str(float(candidate.pre_reject_support_retention_max_background_margin)) + " "
            + ("--pre_reject_support_retention_require_source_looo_pass " if bool(candidate.pre_reject_support_retention_require_source_looo_pass) else "")
            + "--pre_reject_support_retention_source_looo_max_failures " + str(int(candidate.pre_reject_support_retention_source_looo_max_failures)) + " "
            + ("--oa_mse_retention_rescue_gate " if bool(candidate.oa_mse_retention_rescue_gate) else "")
            + "--retention_rescue_old_min_evidence_delta " + str(float(candidate.retention_rescue_old_min_evidence_delta)) + " "
            + "--retention_rescue_old_min_anchor_delta " + str(float(candidate.retention_rescue_old_min_anchor_delta)) + " "
            + "--retention_rescue_old_min_anchor_margin " + str(float(candidate.retention_rescue_old_min_anchor_margin)) + " "
            + "--retention_rescue_old_min_score_margin " + str(float(candidate.retention_rescue_old_min_score_margin)) + " "
            + "--retention_rescue_seen_new_min_evidence_delta " + str(float(candidate.retention_rescue_seen_new_min_evidence_delta)) + " "
            + "--retention_rescue_seen_new_min_anchor_delta " + str(float(candidate.retention_rescue_seen_new_min_anchor_delta)) + " "
            + "--retention_rescue_seen_new_min_score_margin " + str(float(candidate.retention_rescue_seen_new_min_score_margin)) + " "
            + "--retention_rescue_max_background_score " + str(float(candidate.retention_rescue_max_background_score)) + " "
            + "--retention_rescue_max_background_margin " + str(float(candidate.retention_rescue_max_background_margin)) + " "
            + ("--retention_rescue_candidate_only " if bool(candidate.retention_rescue_candidate_only) else "")
            + "--oa_mse_void_background_weight " + str(float(candidate.oa_mse_void_background_weight)) + " "
            + "--oa_mse_negative_anchor_weight " + str(float(candidate.oa_mse_negative_anchor_weight)) + " "
            + "--negative_anchor_margin " + str(float(candidate.negative_anchor_margin)) + " "
            + "--negative_anchor_temperature " + str(float(candidate.negative_anchor_temperature)) + " "
            + "--negative_anchor_max_anchors " + str(int(candidate.negative_anchor_max_anchors)) + " "
            + void_gate_args +
            "--oa_mse_void_gate_min_score " + str(float(candidate.oa_mse_void_gate_min_score)) + " "
            "--oa_mse_void_gate_min_margin " + str(float(candidate.oa_mse_void_gate_min_margin)) + " "
            "--oa_mse_old_neighborhood_weight " + str(float(candidate.oa_mse_old_neighborhood_weight)) + " "
            "--old_neighborhood_samples_per_class " + str(int(candidate.old_neighborhood_samples_per_class)) + " "
            "--old_neighborhood_radius " + str(float(candidate.old_neighborhood_radius)) + " "
            "--oa_mse_old_surrogate_margin_weight " + str(float(candidate.oa_mse_old_surrogate_margin_weight)) + " "
            "--old_surrogate_margin " + str(float(candidate.old_surrogate_margin)) + " "
            "--oa_mse_source_looo_unknown_weight " + str(float(candidate.oa_mse_source_looo_unknown_weight)) + " "
            "--source_looo_unknown_margin " + str(float(candidate.source_looo_unknown_margin)) + " "
            "--source_looo_interclass_margin " + str(float(candidate.source_looo_interclass_margin)) + " "
            "--source_looo_max_samples_per_class " + str(int(candidate.source_looo_max_samples_per_class)) + " "
            + ("--oa_mse_source_looo_risk_arbitration " if bool(candidate.oa_mse_source_looo_risk_arbitration) else "")
            + "--source_looo_risk_quantile " + str(float(candidate.source_looo_risk_quantile)) + " "
            "--source_looo_risk_slack " + str(float(candidate.source_looo_risk_slack)) + " "
            "--source_looo_risk_min_score_margin " + str(float(candidate.source_looo_risk_min_score_margin)) + " "
            "--source_looo_risk_min_known_evidence_delta " + str(float(candidate.source_looo_risk_min_known_evidence_delta)) + " "
            "--source_looo_risk_background_score " + str(float(candidate.source_looo_risk_background_score)) + " "
            "--source_looo_risk_background_margin " + str(float(candidate.source_looo_risk_background_margin)) + " "
            "--source_looo_risk_reject_min_failures " + str(int(candidate.source_looo_risk_reject_min_failures)) + " "
            "--source_looo_risk_reject_action " + str(candidate.source_looo_risk_reject_action) + " "
            "--oa_mse_known_coverage_weight " + str(float(candidate.oa_mse_known_coverage_weight)) + " "
            "--known_coverage_margin " + str(float(candidate.known_coverage_margin)) + " "
            "--known_coverage_min_affinity " + str(float(candidate.known_coverage_min_affinity)) + " "
            "--known_coverage_max_samples " + str(int(candidate.known_coverage_max_samples)) + " "
            "--old_surrogate_evidence_margin " + str(float(candidate.old_surrogate_evidence_margin)) + " "
            "--old_surrogate_reject_relax " + str(float(candidate.old_surrogate_reject_relax)) + " "
            "--oa_mse_siamese_quantile " + str(float(candidate.oa_mse_siamese_quantile)) + " "
            "--oa_mse_siamese_accept_threshold " + str(float(candidate.oa_mse_siamese_accept_threshold)) + " "
            + siamese_veto_args +
            old_unknown_guard_args +
            "--old_anchor_override_min_quality " + str(float(candidate.old_anchor_override_min_quality)) + " "
            "--old_retention_quantile " + str(float(candidate.old_retention_quantile)) + " "
            + ("--oa_mse_support_retention_guard " if bool(candidate.oa_mse_support_retention_guard) else "")
            + "--support_retention_guard_quantile " + str(float(candidate.support_retention_guard_quantile)) + " "
            "--support_retention_guard_slack " + str(float(candidate.support_retention_guard_slack)) + " "
            + two_branch_guard_args
            + "--two_branch_bg_min_score " + str(float(candidate.two_branch_bg_min_score)) + " "
            "--two_branch_bg_min_margin " + str(float(candidate.two_branch_bg_min_margin)) + " "
            "--two_branch_old_support_evidence_delta " + str(float(candidate.two_branch_old_support_evidence_delta)) + " "
            "--two_branch_old_anchor_delta " + str(float(candidate.two_branch_old_anchor_delta)) + " "
            "--two_branch_old_anchor_margin " + str(float(candidate.two_branch_old_anchor_margin)) + " "
            "--two_branch_seen_new_evidence_delta " + str(float(candidate.two_branch_seen_new_evidence_delta)) + " "
            "--two_branch_seen_new_anchor_delta " + str(float(candidate.two_branch_seen_new_anchor_delta)) + " "
            + seen_new_registration_args
            + "--seen_new_override_min_evidence_delta " + str(float(candidate.seen_new_override_min_evidence_delta)) + " "
            "--seen_new_override_min_anchor_delta " + str(float(candidate.seen_new_override_min_anchor_delta)) + " "
            "--seen_new_override_min_affinity_delta " + str(float(candidate.seen_new_override_min_affinity_delta)) + " "
            "--seen_new_override_min_residual_delta " + str(float(candidate.seen_new_override_min_residual_delta)) + " "
            "--seen_new_override_min_score_margin " + str(float(candidate.seen_new_override_min_score_margin)) + " "
            "--seen_new_override_min_seen_vs_old_evidence_margin " + str(float(candidate.seen_new_override_min_seen_vs_old_evidence_margin)) + " "
            "--seen_new_override_max_background_score " + str(float(candidate.seen_new_override_max_background_score)) + " "
            "--seen_new_override_max_background_margin " + str(float(candidate.seen_new_override_max_background_margin)) + " "
            "--old_acc_target " + str(float(candidate.old_acc_target)) + " "
            "--seen_new_acc_target " + str(float(candidate.seen_new_acc_target)) + " "
            "--seed " + str(int(candidate.seed)) + "\""
        )
    raise ValueError(f"unknown command kind: {candidate.command_kind}")


def _launcher_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    def key(candidate: Candidate) -> tuple[int, int, str]:
        slot_text = str(candidate.slot or "")
        slot_name = slot_text.split("/", 1)[1] if "/" in slot_text else slot_text
        slot_rank = ord(slot_name[0]) - ord("A") if slot_name else 999
        return (slot_rank, int(candidate.gpu), candidate.cid)

    return sorted(candidates, key=key)


def render_launcher(run_id: str, candidates: Sequence[Candidate]) -> str:
    stage2_max_active_values = {
        int(c.stage2_max_active_per_gpu)
        for c in candidates
        if c.stage2_max_active_per_gpu is not None
    }
    stage2_max_active_default = (
        str(stage2_max_active_values.pop())
        if len(stage2_max_active_values) == 1
        else "3"
    )
    teacher_defaults = {str(c.ground_model_default_ckpt) for c in candidates if str(c.ground_model_default_ckpt)}
    teacher_ckpt_default = teacher_defaults.pop() if len(teacher_defaults) == 1 else DEFAULT_BEX02_TEACHER_CKPT
    blocks = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"',
        'PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"',
        f'RUN_ID="${{RUN_ID:-{run_id}}}"',
        'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        f'TEACHER_CKPT="${{TEACHER_CKPT:-{teacher_ckpt_default}}}"',
        'WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"',
        'NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"',
        'SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"',
        'TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"',
        f'NEW_TX_IDS="${{NEW_TX_IDS:-{PHASE2_TARGET_NEW_TX_LABELS}}}"',
        'UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-}"',
        f'OA_MSE_UNKNOWN_TX_IDS="${{OA_MSE_UNKNOWN_TX_IDS:-{PHASE2_UNKNOWN_TX_LABELS}}}"',
        'CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"',
        f'TARGET_RECEIVER_IDS="${{TARGET_RECEIVER_IDS:-{PHASE2_TARGET_RECEIVER_LABEL}}}"',
        'SFE_MAX_SAMPLES_PER_COMBO="${SFE_MAX_SAMPLES_PER_COMBO:-0}"',
        'SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"',
        'SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"',
        'SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"',
        'SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"',
        'SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"',
        f'STAGE2_MAX_ACTIVE_PER_GPU="${{STAGE2_MAX_ACTIVE_PER_GPU:-{stage2_max_active_default}}}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        "",
        'for arg in "$@"; do',
        '  case "${arg}" in',
        "    --dry-run) DRY_RUN=1 ;;",
        '    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        'if [[ "${DRY_RUN}" != "1" ]]; then',
        '  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"',
        "fi",
        'echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=' + str(len(candidates)) + '"',
        "PIDS=()",
        "NAMES=()",
        "GPUS=()",
        "STATUS=0",
        "",
        "reap_finished() {",
        "  local idx pid rc",
        '  for idx in "${!PIDS[@]}"; do',
        '    pid="${PIDS[${idx}]}"',
        '    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then',
        '      if wait "${pid}"; then',
        '        echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${pid} status=0"',
        "      else",
        "        rc=$?",
        '        echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${pid} status=${rc}" >&2',
        "        STATUS=${rc}",
        "      fi",
        '      PIDS[${idx}]=""; NAMES[${idx}]=""; GPUS[${idx}]=""',
        "    fi",
        "  done",
        "}",
        "",
        "active_for_gpu() {",
        "  local gpu=\"$1\" idx pid count=0",
        '  for idx in "${!PIDS[@]}"; do',
        '    pid="${PIDS[${idx}]}"',
        '    if [[ -n "${pid}" && "${GPUS[${idx}]}" == "${gpu}" ]] && kill -0 "${pid}" 2>/dev/null; then',
        "      count=$((count + 1))",
        "    fi",
        "  done",
        '  echo "${count}"',
        "}",
        "",
        "wait_for_gpu_slot() {",
        "  local gpu=\"$1\" active",
        "  while true; do",
        "    reap_finished",
        '    active="$(active_for_gpu "${gpu}")"',
        '    if (( active < STAGE2_MAX_ACTIVE_PER_GPU )); then',
        "      break",
        "    fi",
        '    echo "[SPACEBORNE-FSDA-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"',
        "    sleep 5",
        "  done",
        "}",
        "",
    ]
    for candidate in _launcher_order(candidates):
        cmd = _candidate_command(candidate)
        blocks.extend(
            [
                f'echo "[SPACEBORNE-FSDA-CANDIDATE] id={candidate.cid} protocol={candidate.protocol} k={candidate.k} target_visibility={candidate.target_visibility} label_set_relation={candidate.label_set_relation}"',
                f'GPU="{candidate.gpu}"',
                f'CMD=({cmd})',
                'printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\\n"',
                'if [[ "${DRY_RUN}" != "1" ]]; then',
                '  wait_for_gpu_slot "${GPU}"',
                f'  mkdir -p "${{RUNS_ROOT}}/{candidate.cid}"',
                f'  ("${{CMD[@]}}" > "${{LOG_ROOT}}/{candidate.cid}.out" 2>&1) &',
                '  pid="$!"',
                '  PIDS+=("${pid}")',
                f'  NAMES+=("{candidate.cid}")',
                '  GPUS+=("${GPU}")',
                f'  echo "[SPACEBORNE-FSDA-LAUNCHED] id={candidate.cid} pid=${{pid}} gpu=${{GPU}} log=${{LOG_ROOT}}/{candidate.cid}.out"',
                "fi",
                "",
            ]
        )
    blocks.extend(
        [
            'if [[ "${DRY_RUN}" != "1" ]]; then',
            '  for idx in "${!PIDS[@]}"; do',
            '    if [[ -n "${PIDS[${idx}]}" ]] && wait "${PIDS[${idx}]}"; then',
            '      echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=0"',
            "    else",
            "      rc=$?",
            '      if [[ -n "${PIDS[${idx}]}" ]]; then',
            '        echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=${rc}" >&2',
            "        STATUS=${rc}",
            "      fi",
            "    fi",
            "  done",
            '  exit "${STATUS}"',
            "fi",
            "",
        ]
    )
    return "\n".join(blocks)


def render_report(run_id: str, candidates: Sequence[Candidate]) -> str:
    ground_models = sorted({f"{c.ground_model_label} ({c.ground_model_default_ckpt})" for c in candidates})
    phase1_count = sum(1 for c in candidates if c.command_kind == "phase1_safe_ssdg_ground_train")
    rows = [
        "| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |",
        "|---|---:|---|---:|---|---|---|---|---|",
    ]
    for c in candidates:
        gate_or_adapter = c.gate_mode if c.protocol in {"CVS-SFE", "CVS-OA-MSE"} else c.target_adapter_type
        rows.append(
            f"| `{c.cid}` | {c.gpu} | {c.protocol} | {c.k} | `{gate_or_adapter}` | `{c.target_visibility}` | `{c.label_set_relation}` | `{c.update_module}` | {c.metrics} |"
        )
    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## Objective",
            "",
            (
                "Validate Phase1 source-only ground DG representation optimization together with "
                "deployment-time Phase2 few-shot adaptation paths."
                if phase1_count
                else "Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:"
            ),
            (
                "Phase1 rows track source-domain TX prototypes, receiver feature distribution, mask auxiliaries, "
                "and TX/RX geometry without using target receiver samples."
                if phase1_count
                else "new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`)."
            ),
            f"Ground model default: {', '.join(ground_models)}.",
            "",
            "## Candidate Matrix",
            "",
            *rows,
            "",
            "## Verification Contract",
            "",
            "- Framework match: the ground-stage model is the existing generalized CVS-RFFI metric space checkpoint; this launcher only covers satellite-deployed few-shot procedures.",
            "- No semi-supervised target adaptation is used: target receiver adaptation is labeled-only and commands set `--entropy_weight 0`, `--consistency_weight 0`, and `--pseudo_weight 0`.",
            "- `CVS-SFE` is a feature-level validation over frozen `z_id` prototypes; the support features stand for samples already affected by `H_sg o R_sat`, and it must report `full_accuracy`, `accepted_accuracy`, `coverage`, `new_class_accuracy`, `old_class_accuracy`, and `unknown_rejection_rate`.",
            "- `CVS-FTRC` uses target receiver support after explicit star-ground channel synthesis (`--target_channel_view satellite`) and is not strict DG; it must be reported separately from source-only DG tables.",
            "- OA-MSE rows are staged as Stage2-A MSE-lite, Stage2-B MSE-subspace, and Stage2-C OA-MSE-Head; unknown query samples are eval-only and cannot fit thresholds.",
            "- Future star-ground augmentation uses `star_ground_channel_impl=simplified_leo_residual` with `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak`; legacy five-scenario LEO is control-only unless explicitly marked.",
            "- OA-MSE launchable rows must carry the combined onboard adaptation bundle: Weibull EVT, target adapter, pseudo-unknown energy, seen-new evidence gate, ambiguous-only Siamese verifier, accepted-only online update, and Stage2 receiver-domain separation.",
            "- Gate and adapter variants must record their candidate-level parameters in `matrix.json`; rollback decisions are deployment gates, not post-hoc notes.",
            "- Any accepted-only metric must be shown with its full denominator and coverage.",
            "- Satellite metrics are stress-test metrics unless real in-orbit IQ is explicitly used.",
            "",
            "## Launch",
            "",
            "Run local/remote dry-run first:",
            "",
            "```bash",
            f"bash code/scripts/launch_{run_id}.sh --dry-run",
            "```",
            "",
            "After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.",
            "",
        ]
    )


def _oa_mse_stage2_mode(candidate: Candidate) -> str:
    if candidate.oa_mse_stage == "mse_lite":
        return "Stage2-A_zero_label_deploy"
    if candidate.oa_mse_stage == "mse_subspace":
        return "Stage2-B_old_label_calibration"
    if candidate.oa_mse_stage == "oa_mse_head":
        return "Stage2-C_old_new_enrollment"
    return "Phase2_spaceborne_fewshot_adaptation"


def _phase1_optimizer_matrix_item(run_id: str, candidate: Candidate, idx: int) -> dict:
    exact_command = _candidate_command(candidate)
    command_hash = hashlib.sha256(exact_command.encode("utf-8")).hexdigest()[:16]
    variant_spec = PHASE1_GPU0_JOINTSAFE_VARIANTS.get(str(candidate.phase1_variant or "").lower())
    parameters = {
        "epochs": int(candidate.epochs or 200),
        "phase1_candidate": "Safe-SSDG-CVS-R01",
        "entrypoint": "python ${ROOT}/code/SSDG/train_ssdg.py",
        "split_mode": "tx_rx_day_1_7_2",
        "labeled_ratio": 0.10,
        "unlabeled_ratio": 0.70,
        "source_val_ratio": 0.20,
        "phase1_design_report_ref": candidate.phase1_design_report_ref,
        "prototype_mask_modules": PHASE1_GROUND_PROTO_MASK_MODULES,
        "phase1_enable_ground_prototype_stats": True,
        "phase1_enable_feature_distribution_audit": True,
        "phase1_enable_feature_masks_aux": True,
        "phase1_enable_txrx_geometry_audit": True,
        "phase1_prototype_loss_weight": 0.0,
        "phase1_mask_aux_loss_weight": 0.0,
        "phase1_geometry_loss_weight": 0.0,
        "phase1_distribution_audit_only": True,
        "phase1_star_ground_aug_default_enabled": True,
        "phase1_star_ground_aug_route_family": "CVS-SAT-PAIC",
        "phase1_star_ground_aug_mode": "concat_sat_ce_only_paic_curriculum",
        "star_ground_channel_impl": STAR_GROUND_CHANNEL_IMPL,
        "use_concat_sat_channel_aug": True,
        "concat_sat_ce_only": True,
        "use_sat_consistency": True,
        "sat_train_scenarios": SIMPLIFIED_LEO_SCENARIOS,
        "sat_view_schedule": PHASE1_PAIC_SAT_VIEW_SCHEDULE,
        "best_metric": "joint_safe" if variant_spec is not None else "clean_val_tx",
        "joint_checkpoint_policy": "joint_safe_guarded" if variant_spec is not None else "legacy_best_metric",
        "protected_metrics": (
            "strict_udu,receiver_floor,sat_mean_tx,sat_floor_tx,sat_strict_mean,sat_strict_floor,val_tx"
            if variant_spec is not None
            else ""
        ),
        "one_epoch_drop_guard": True if variant_spec is not None else False,
        "one_epoch_drop_guard_pp": 2.0 if variant_spec is not None else 0.0,
        "paic_late_variance_guard": True if variant_spec is not None else False,
        "paic_guard_sat_ce_delta": 0.12 if variant_spec is not None else 0.0,
        "paic_guard_grad_delta": 3.0 if variant_spec is not None else 0.0,
        "paic_guard_reliable_drop": 0.01 if variant_spec is not None else 0.0,
        "safe_best_path": "best_joint_safe_ssdg.pth" if variant_spec is not None else "",
        "safe_latest_path": "latest_safe_ssdg.pth" if variant_spec is not None else "",
        "active_loss_markers_expected": (
            "[JOINT-GUARD],[SAFE-CKPT],[PROTO-TX],[PROTO-RX],[MASK],[BATCH-GEOM],[TXRX-ANOVA]"
            if variant_spec is not None
            else ""
        ),
        "zero_weight_telemetry_expected": True,
    }
    if variant_spec is not None:
        parameters.update(
            {
                "label_epochs": int(variant_spec["label_epochs"]),
                "pseudo_epochs": int(variant_spec["pseudo_epochs"]),
                "lambda_u": float(variant_spec["lambda_u"]),
                "tau_min": float(variant_spec["tau_min"]),
                "tau_max": float(variant_spec["tau_max"]),
                "pseudo_quantile": float(variant_spec["pseudo_quantile"]),
                "use_ema_teacher": bool(variant_spec["use_ema_teacher"]),
                "lambda_sat_cls": float(variant_spec["lambda_sat_cls"]),
                "lambda_sat_cons": float(variant_spec["lambda_sat_cons"]),
                "lambda_group_ce": float(variant_spec["lambda_group_ce"]),
                "lambda_fishr": float(variant_spec["lambda_fishr"]),
                "lambda_domain": float(variant_spec["lambda_domain"]),
            }
        )
    item = asdict(candidate)
    item.update(
        {
            "candidate_id": candidate.cid,
            "slot": candidate.slot,
            "category": str(candidate.optimization_category),
            "lane": "phase1_ground_dg",
            "phase_axis": "Phase1-GroundDG",
            "stage2_mode": "NOT_APPLICABLE",
            "parent_run": run_id,
            "lineage": f"{run_id}:phase1_ground_proto_mask",
            "route_signature": (
                f"safe_ssdg_cvs_r01_phase1_ground_proto_mask_{candidate.phase1_variant}_{candidate.gpu}"
            ).lower(),
            "retirement_status": "not_retired",
            "invalidity_status": "not_invalidated",
            "principle_rejection_ref": "none",
            "experimental_rejection_ref": "none",
            "retirement_evidence_count": 0,
            "retirement_evidence_refs": [],
            "replacement_reason": "enable_phase1_ground_prototype_mask_feature_distribution_lane",
            "hypothesis": candidate.description,
            "control": (
                "source-only ground DG; CEN51 success is non-regression baseline, "
                "not a narrowed route family; target receiver and unknown query are forbidden in training"
            ),
            "key_changes": [
                "joint_safe_checkpoint_guard" if variant_spec is not None else "source_domain_tx_prototypes",
                "paic_late_variance_guard" if variant_spec is not None else "receiver_domain_feature_distribution_audit",
                "prototype_mask_geometry_audit_only",
                "gpu0_a_late_pseudo_repair" if variant_spec is not None else "tx_rx_geometry_audit",
            ],
            "parameters": parameters,
            "estimated_run_path": f"/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}/{candidate.cid}",
            "estimated_log_path": f"/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}/{candidate.cid}.out",
            "cross_domain_target_metric": (
                "strict_udu,worst_receiver,source_receiver_floor,prototype_radius,"
                "tx_margin_violation,rx_probe_leakage"
            ),
            "satellite_channel_target_metric": "sat_mean_5,sat_floor_5 with no CEN51 floor regression",
            "allowed_tradeoff": "no target receiver training use; no unknown query threshold fit; no full strong-loss enablement",
            "must_not_regress_floor": PHASE1_CEN51_NON_REGRESSION_FLOORS,
            "comparability_status": "CEN51_COMPARABLE_SOURCE_ONLY_NON_REGRESSION",
            "expected_failure_signals": (
                "prototype_radius_expands,tx_margin_violation_increases,rx_probe_on_z_tx_high,"
                "tx_probe_on_z_rx_high,sat_floor_5_regresses,worst_receiver_regresses"
            ),
            "fallback_or_alternative": (
                "keep Phase1 losses at zero-weight audit; only promote prototype/mask/geometry losses after "
                "source-only diagnostics beat CEN51 non-regression floors"
            ),
            "exact_command": exact_command,
            "launchability_status": "phase1_launchable_training_candidate_full_200e_pending_remote_gates",
            "runtime_class": "phase1_training",
            "registry_key": f"{run_id}:{candidate.cid}",
            "command_hash": command_hash,
            "ground_dg_claim_scope": "source_only",
            "source_ssl_split": "L_s/U_s/Val_s source receivers only; rho_label=0.10; no target receiver samples",
            "no_target_receiver_in_training": True,
            "cen51_base_checkpoint_or_config": (
                "matched_CEN51_R04 source-only split and CEN51 success experience; "
                "used as non-regression/comparison, not as route restriction"
            ),
            "cen51_parent_run_or_control": "matched_CEN51_R04_control",
            "phase1_non_regression_target": "matched_CEN51_R04",
            "optimization_target": (
                "exceed matched CEN51_R04 on source-only DG and satellite-channel metrics while improving "
                "feature-space prototype/radius/receiver-distribution diagnostics"
            ),
            "target_lift_over_cen51": (
                "sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0; strict_udu_delta_pp>=0; "
                "receiver_floor_delta_pp>=0; prototype_radius_not_expanded"
            ),
            "satellite_channel_lift_target": "sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0",
            "pseudo_precision_audit_target": "precision>=0.95; coverage_by_class_receiver_reported_as_risk_metric",
            "CEN51_COMPARABLE": True,
            "pseudo_coverage_is_risk_metric": True,
            "satellite_channel_primary_metric": True,
            "forbid_meta_learning_dg_mainline": True,
            "phase1_star_ground_aug_policy": "default_on",
            "phase1_star_ground_aug_default_enabled": True,
            "phase1_star_ground_aug_route_family": "CVS-SAT-PAIC",
            "phase1_star_ground_aug_mode": "concat_sat_ce_only_paic_curriculum",
            "use_concat_sat_channel_aug": True,
            "concat_sat_ce_only": True,
            "use_sat_consistency": True,
            "sat_view_schedule": PHASE1_PAIC_SAT_VIEW_SCHEDULE,
            "best_metric": "joint_safe" if variant_spec is not None else "clean_val_tx",
            "joint_checkpoint_policy": "joint_safe_guarded" if variant_spec is not None else "legacy_best_metric",
            "protected_metrics": parameters["protected_metrics"],
            "one_epoch_drop_guard": bool(variant_spec is not None),
            "rollback_policy": "block_unsafe_best_and_keep_latest_safe_checkpoint" if variant_spec is not None else "not_enabled",
            "paic_late_variance_guard": bool(variant_spec is not None),
            "sat_cons_start_epoch": 80 if variant_spec is not None else 60,
            "lambda_sat_cls": float(variant_spec["lambda_sat_cls"]) if variant_spec is not None else 1.0,
            "lambda_sat_cons": float(variant_spec["lambda_sat_cons"]) if variant_spec is not None else 0.03,
            "star_ground_aug_exploration_axis": (
                "GPU0_A joint-safe late pseudo repair plus source-domain prototype/mask/feature-distribution audit"
                if variant_spec is not None
                else "late weak z_id consistency plus source-domain prototype/mask/feature-distribution audit"
            ),
            "phase1_ground_prototype_mask_openworld_enabled": True,
            "phase1_ground_feature_distribution_objective": True,
            "source_domain_prototype_outputs_required": True,
            "phase1_prototype_loss_weight": 0.0,
            "phase1_mask_aux_loss_weight": 0.0,
            "phase1_geometry_loss_weight": 0.0,
            "phase1_distribution_audit_only": True,
            "active_loss_markers_expected": parameters["active_loss_markers_expected"],
            "zero_weight_telemetry_expected": True,
            "target_receiver_usage": "forbidden_in_phase1",
            "unknown_query_role": "eval_only_not_available_in_phase1_training",
            "clean_view_role": "control_only",
            "dataset_role": "terrestrial_source_proxy",
            "evidence_level": "source_receiver_x_transmitter_ground_dg_proxy_non_deployment_claim",
            "deployment_success_claim_allowed": False,
        }
    )
    return item


def _optimizer_matrix_item(run_id: str, candidate: Candidate, idx: int) -> dict:
    if candidate.command_kind == "phase1_safe_ssdg_ground_train":
        return _phase1_optimizer_matrix_item(run_id, candidate, idx)

    item = asdict(candidate)
    mode = _oa_mse_stage2_mode(candidate)
    target_receiver_label = candidate.target_receiver_label or PHASE2_TARGET_RECEIVER_LABEL
    target_receiver_ids = candidate.target_receiver_ids or PHASE2_TARGET_RECEIVER_ALIAS
    if target_receiver_ids == "${TARGET_RECEIVER_IDS}":
        target_receiver_ids = PHASE2_TARGET_RECEIVER_ALIAS
    manysig_target_rx_index = candidate.manysig_target_rx_index or PHASE2_MANYSIG_TARGET_RX_INDEX
    manytx_target_rx_index = candidate.manytx_target_rx_index or PHASE2_MANYTX_TARGET_RX_INDEX
    new_tx_disabled = str(candidate.new_tx_ids or "").strip().upper() in {"__NONE__", "NONE", "OLD_UNKNOWN_ONLY"}
    target_new_tx_ids = "" if new_tx_disabled else PHASE2_TARGET_NEW_TX_LABELS
    target_new_tx_indices = "" if new_tx_disabled else PHASE2_TARGET_NEW_TX_INDICES
    target_old_support = ""
    target_new_support = ""
    old_split = (
        f"support=empty; query=target_old_{target_receiver_ids}; target_receiver_label="
        f"{target_receiver_label}; source_old_prototypes_only"
    )
    new_split = (
        f"support=empty; query=target_new_{target_receiver_ids}; target_receiver_label="
        f"{target_receiver_label}; threshold_fit=forbidden"
    )
    if new_tx_disabled:
        new_split = "not_applicable_old_unknown_only"
    k_shot = ""
    if "Stage2-B" in mode:
        target_old_support = (
            f"target_old support tx=0-5 rx={target_receiver_ids} target_receiver_label="
            f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
            f"scenarios={candidate.target_channel_scenarios} support_per_tx={candidate.target_old_support_per_tx}"
        )
        old_split = (
            f"support=target_old_fewshot; query=target_old_{target_receiver_ids}; target_receiver_label="
            f"{target_receiver_label}; support_per_tx={candidate.target_old_support_per_tx}"
        )
        k_shot = candidate.target_old_support_per_tx
        item["k"] = candidate.target_old_support_per_tx
    elif "Stage2-C" in mode:
        target_old_support = (
            f"target_old support tx=0-5 rx={target_receiver_ids} target_receiver_label="
            f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
            f"scenarios={candidate.target_channel_scenarios} support_per_tx={candidate.target_old_support_per_tx}"
        )
        target_new_support = (
            "target_new seen support tx="
            f"{PHASE2_TARGET_NEW_TX_LABELS} rx={target_receiver_ids} target_receiver_label="
            f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
            f"scenarios={candidate.target_channel_scenarios} support_per_tx={candidate.target_new_support_per_tx}"
        )
        old_split = (
            f"support=target_old_fewshot; query=target_old_{target_receiver_ids}; target_receiver_label="
            f"{target_receiver_label}; support_per_tx={candidate.target_old_support_per_tx}"
        )
        new_split = (
            f"support=target_new_fewshot; query=target_new_{target_receiver_ids}; target_receiver_label="
            f"{target_receiver_label}; support_per_tx={candidate.target_new_support_per_tx}; unknown_query=eval_only"
        )
        k_shot = candidate.k

    item.update(
        {
            "candidate_id": candidate.cid,
            "slot": candidate.slot,
            "category": str(candidate.optimization_category),
            "stage2_priority_phase": candidate.stage2_priority_phase,
            "old_acc_phase_gate": candidate.old_acc_phase_gate,
            "secondary_objectives_after_old_gate": candidate.secondary_objectives_after_old_gate,
            "lane": "phase2_spaceborne_fsl",
            "phase_axis": "Phase2-Spaceborne-FSL",
            "stage2_mode": mode,
            "parent_run": run_id,
            "lineage": f"{run_id}:oa_mse_card3",
            "route_signature": f"oa_mse_head_{candidate.oa_mse_stage}_{candidate.cid}".lower(),
            "retirement_status": "not_retired",
            "invalidity_status": "not_invalidated",
            "principle_rejection_ref": "none",
            "experimental_rejection_ref": "none",
            "retirement_evidence_count": 0,
            "retirement_evidence_refs": [],
            "replacement_reason": "repaired_phase2_oa_mse_local_hook",
            "hypothesis": candidate.description,
            "control": "source-only/no-target-support control and unknown-query eval-only guard",
            "key_changes": [candidate.update_module, candidate.oa_mse_stage],
            "parameters": {
                "gate_mode": candidate.gate_mode,
                "oa_mse_stage": candidate.oa_mse_stage,
                "optimization_category": candidate.optimization_category,
                "stage2_priority_phase": candidate.stage2_priority_phase,
                "old_acc_phase_gate": candidate.old_acc_phase_gate,
                "secondary_objectives_after_old_gate": candidate.secondary_objectives_after_old_gate,
                "oa_mse_adapter_kind": candidate.oa_mse_adapter_kind,
                "max_adapt_steps": candidate.max_adapt_steps,
                "oa_mse_source_anchor_weight": candidate.oa_mse_source_anchor_weight,
                "oa_mse_source_ce_weight": candidate.oa_mse_source_ce_weight,
                "oa_mse_unknown_moat_weight": candidate.oa_mse_unknown_moat_weight,
                "oa_mse_unknown_moat_margin": candidate.oa_mse_unknown_moat_margin,
                "pseudo_unknown_samples_per_pair": candidate.pseudo_unknown_samples_per_pair,
                "pseudo_unknown_offset_scale": candidate.pseudo_unknown_offset_scale,
                "pseudo_unknown_source_boundary_samples_per_pair": candidate.pseudo_unknown_source_boundary_samples_per_pair,
                "pseudo_unknown_source_boundary_offset_scale": candidate.pseudo_unknown_source_boundary_offset_scale,
                "pseudo_unknown_target_shift_samples_per_class": candidate.pseudo_unknown_target_shift_samples_per_class,
                "pseudo_unknown_target_shift_offset_scale": candidate.pseudo_unknown_target_shift_offset_scale,
                "pseudo_unknown_target_halo_samples_per_class": candidate.pseudo_unknown_target_halo_samples_per_class,
                "pseudo_unknown_target_halo_offset_scale": candidate.pseudo_unknown_target_halo_offset_scale,
                "pseudo_unknown_target_ring_samples_per_class": candidate.pseudo_unknown_target_ring_samples_per_class,
                "pseudo_unknown_target_ring_offset_scale": candidate.pseudo_unknown_target_ring_offset_scale,
                "oa_mse_old_bridge_weight": candidate.oa_mse_old_bridge_weight,
                "old_bridge_samples_per_class": candidate.old_bridge_samples_per_class,
                "old_bridge_max_mix": candidate.old_bridge_max_mix,
                "oa_mse_support_contrast_weight": candidate.oa_mse_support_contrast_weight,
                "old_support_contrast_negative_margin": candidate.old_support_contrast_negative_margin,
                "old_support_contrast_positive_margin": candidate.old_support_contrast_positive_margin,
                "oa_mse_support_center_ce_weight": candidate.oa_mse_support_center_ce_weight,
                "support_center_temperature": candidate.support_center_temperature,
                "support_center_margin": candidate.support_center_margin,
                "oa_mse_soft_proto_weight": candidate.oa_mse_soft_proto_weight,
                "soft_proto_topk": candidate.soft_proto_topk,
                "soft_proto_temperature": candidate.soft_proto_temperature,
                "oa_mse_soft_proto_boundary_weight": candidate.oa_mse_soft_proto_boundary_weight,
                "soft_proto_boundary_margin": candidate.soft_proto_boundary_margin,
                "oa_mse_multiproto_score": candidate.oa_mse_multiproto_score,
                "multiproto_topk": candidate.multiproto_topk,
                "multiproto_temperature": candidate.multiproto_temperature,
                "multiproto_score_weight": candidate.multiproto_score_weight,
                "oa_mse_mixture_consistency_gate": candidate.oa_mse_mixture_consistency_gate,
                "mixture_consistency_min_cos": candidate.mixture_consistency_min_cos,
                "mixture_consistency_max_residual": candidate.mixture_consistency_max_residual,
                "mixture_consistency_min_margin": candidate.mixture_consistency_min_margin,
                "mixture_consistency_action": candidate.mixture_consistency_action,
                "oa_mse_anchor_density_gate": candidate.oa_mse_anchor_density_gate,
                "anchor_density_topk": candidate.anchor_density_topk,
                "anchor_density_temperature": candidate.anchor_density_temperature,
                "anchor_density_min_quantile": candidate.anchor_density_min_quantile,
                "anchor_density_margin_quantile": candidate.anchor_density_margin_quantile,
                "anchor_density_gate_action": candidate.anchor_density_gate_action,
                "oa_mse_void_background_weight": candidate.oa_mse_void_background_weight,
                "oa_mse_void_gate": candidate.oa_mse_void_gate,
                "oa_mse_void_gate_min_score": candidate.oa_mse_void_gate_min_score,
                "oa_mse_void_gate_min_margin": candidate.oa_mse_void_gate_min_margin,
                "oa_mse_old_neighborhood_weight": candidate.oa_mse_old_neighborhood_weight,
                "old_neighborhood_samples_per_class": candidate.old_neighborhood_samples_per_class,
                "old_neighborhood_radius": candidate.old_neighborhood_radius,
                "oa_mse_old_surrogate_margin_weight": candidate.oa_mse_old_surrogate_margin_weight,
                "old_surrogate_margin": candidate.old_surrogate_margin,
                "oa_mse_source_looo_unknown_weight": candidate.oa_mse_source_looo_unknown_weight,
                "source_looo_unknown_margin": candidate.source_looo_unknown_margin,
                "source_looo_interclass_margin": candidate.source_looo_interclass_margin,
                "source_looo_max_samples_per_class": candidate.source_looo_max_samples_per_class,
                "oa_mse_known_coverage_weight": candidate.oa_mse_known_coverage_weight,
                "known_coverage_margin": candidate.known_coverage_margin,
                "known_coverage_min_affinity": candidate.known_coverage_min_affinity,
                "known_coverage_max_samples": candidate.known_coverage_max_samples,
                "old_surrogate_evidence_margin": candidate.old_surrogate_evidence_margin,
                "old_surrogate_reject_relax": candidate.old_surrogate_reject_relax,
                "oa_mse_siamese_quantile": candidate.oa_mse_siamese_quantile,
                "oa_mse_siamese_accept_threshold": candidate.oa_mse_siamese_accept_threshold,
                "oa_mse_siamese_unknown_veto": candidate.oa_mse_siamese_unknown_veto,
                "oa_mse_siamese_unknown_veto_mode": candidate.oa_mse_siamese_unknown_veto_mode,
                "oa_mse_siamese_min_old_support_evidence_delta": (
                    candidate.oa_mse_siamese_min_old_support_evidence_delta
                ),
                "oa_mse_siamese_min_old_surrogate_reject_delta": (
                    candidate.oa_mse_siamese_min_old_surrogate_reject_delta
                ),
                "oa_mse_siamese_min_energy_delta": candidate.oa_mse_siamese_min_energy_delta,
                "oa_mse_siamese_min_mahalanobis_delta": candidate.oa_mse_siamese_min_mahalanobis_delta,
                "oa_mse_siamese_min_accept_delta": candidate.oa_mse_siamese_min_accept_delta,
                "oa_mse_siamese_min_old_support_anchor_margin": (
                    candidate.oa_mse_siamese_min_old_support_anchor_margin
                ),
                "oa_mse_siamese_min_veto_failures": candidate.oa_mse_siamese_min_veto_failures,
                "oa_mse_old_unknown_acceptance_guard": candidate.oa_mse_old_unknown_acceptance_guard,
                "oa_mse_old_unknown_guard_min_old_support_evidence_delta": (
                    candidate.oa_mse_old_unknown_guard_min_old_support_evidence_delta
                ),
                "oa_mse_old_unknown_guard_min_old_surrogate_reject_delta": (
                    candidate.oa_mse_old_unknown_guard_min_old_surrogate_reject_delta
                ),
                "oa_mse_old_unknown_guard_min_energy_delta": candidate.oa_mse_old_unknown_guard_min_energy_delta,
                "oa_mse_old_unknown_guard_min_mahalanobis_delta": (
                    candidate.oa_mse_old_unknown_guard_min_mahalanobis_delta
                ),
                "oa_mse_old_unknown_guard_min_accept_delta": candidate.oa_mse_old_unknown_guard_min_accept_delta,
                "oa_mse_old_unknown_guard_min_old_support_anchor_margin": (
                    candidate.oa_mse_old_unknown_guard_min_old_support_anchor_margin
                ),
                "oa_mse_old_unknown_guard_min_best_old_score": candidate.oa_mse_old_unknown_guard_min_best_old_score,
                "oa_mse_old_unknown_guard_min_margin": candidate.oa_mse_old_unknown_guard_min_margin,
                "oa_mse_old_unknown_guard_min_failures": candidate.oa_mse_old_unknown_guard_min_failures,
                "stage2_max_active_per_gpu": candidate.stage2_max_active_per_gpu,
                "oa_mse_adapter_selection_policy": candidate.oa_mse_adapter_selection_policy,
                "oa_mse_adapter_alpha_eval_sweep": candidate.oa_mse_adapter_alpha_eval_sweep,
                "old_anchor_override_min_quality": candidate.old_anchor_override_min_quality,
                "old_retention_quantile": candidate.old_retention_quantile,
                "oa_mse_support_retention_guard": candidate.oa_mse_support_retention_guard,
                "support_retention_guard_quantile": candidate.support_retention_guard_quantile,
                "support_retention_guard_slack": candidate.support_retention_guard_slack,
                "oa_mse_two_branch_background_guard": candidate.oa_mse_two_branch_background_guard,
                "two_branch_bg_min_score": candidate.two_branch_bg_min_score,
                "two_branch_bg_min_margin": candidate.two_branch_bg_min_margin,
                "two_branch_old_support_evidence_delta": candidate.two_branch_old_support_evidence_delta,
                "two_branch_old_anchor_delta": candidate.two_branch_old_anchor_delta,
                "two_branch_old_anchor_margin": candidate.two_branch_old_anchor_margin,
                "two_branch_seen_new_evidence_delta": candidate.two_branch_seen_new_evidence_delta,
                "two_branch_seen_new_anchor_delta": candidate.two_branch_seen_new_anchor_delta,
                "oa_mse_seen_new_registration_override": candidate.oa_mse_seen_new_registration_override,
                "seen_new_override_min_evidence_delta": candidate.seen_new_override_min_evidence_delta,
                "seen_new_override_min_anchor_delta": candidate.seen_new_override_min_anchor_delta,
                "seen_new_override_min_affinity_delta": candidate.seen_new_override_min_affinity_delta,
                "seen_new_override_min_residual_delta": candidate.seen_new_override_min_residual_delta,
                "seen_new_override_min_score_margin": candidate.seen_new_override_min_score_margin,
                "seen_new_override_min_seen_vs_old_evidence_margin": (
                    candidate.seen_new_override_min_seen_vs_old_evidence_margin
                ),
                "seen_new_override_max_background_score": candidate.seen_new_override_max_background_score,
                "seen_new_override_max_background_margin": candidate.seen_new_override_max_background_margin,
                "seen_new_override_min_support_knn_seen_new_minus_old": (
                    candidate.seen_new_override_min_support_knn_seen_new_minus_old
                ),
                "seen_new_override_min_support_knn_margin": candidate.seen_new_override_min_support_knn_margin,
                "oa_mse_old_primary_gate": candidate.oa_mse_old_primary_gate,
                "old_primary_min_old_support_evidence_delta": candidate.old_primary_min_old_support_evidence_delta,
                "old_primary_min_old_support_anchor_delta": candidate.old_primary_min_old_support_anchor_delta,
                "old_primary_min_old_support_anchor_margin": candidate.old_primary_min_old_support_anchor_margin,
                "old_primary_min_score_margin": candidate.old_primary_min_score_margin,
                "old_primary_require_soft_mixture": candidate.old_primary_require_soft_mixture,
                "old_primary_min_soft_mixture_margin": candidate.old_primary_min_soft_mixture_margin,
                "old_primary_min_soft_mixture_cos": candidate.old_primary_min_soft_mixture_cos,
                "old_primary_max_soft_mixture_residual": candidate.old_primary_max_soft_mixture_residual,
                "old_primary_require_support_knn": candidate.old_primary_require_support_knn,
                "old_primary_require_support_knn_label_match": candidate.old_primary_require_support_knn_label_match,
                "old_primary_min_support_knn_margin": candidate.old_primary_min_support_knn_margin,
                "old_primary_max_support_knn_seen_new_minus_old": (
                    candidate.old_primary_max_support_knn_seen_new_minus_old
                ),
                "old_primary_min_old_drift_cos": candidate.old_primary_min_old_drift_cos,
                "old_primary_max_old_drift_dist": candidate.old_primary_max_old_drift_dist,
                "old_primary_require_class_envelope": candidate.old_primary_require_class_envelope,
                "old_primary_promote_rescue_candidates": candidate.old_primary_promote_rescue_candidates,
                "old_primary_unknown_veto_background_score": candidate.old_primary_unknown_veto_background_score,
                "old_primary_unknown_veto_background_margin": candidate.old_primary_unknown_veto_background_margin,
                "old_primary_unknown_veto_min_sources": candidate.old_primary_unknown_veto_min_sources,
                "old_primary_fail_action": candidate.old_primary_fail_action,
                "old_primary_unknown_veto_action": candidate.old_primary_unknown_veto_action,
                "oa_mse_density_shell_gate": candidate.oa_mse_density_shell_gate,
                "density_shell_old_min_evidence_delta": candidate.density_shell_old_min_evidence_delta,
                "density_shell_old_min_anchor_delta": candidate.density_shell_old_min_anchor_delta,
                "density_shell_old_min_density_delta": candidate.density_shell_old_min_density_delta,
                "density_shell_seen_new_min_evidence_delta": candidate.density_shell_seen_new_min_evidence_delta,
                "density_shell_seen_new_min_anchor_delta": candidate.density_shell_seen_new_min_anchor_delta,
                "density_shell_seen_new_min_density_delta": candidate.density_shell_seen_new_min_density_delta,
                "density_shell_accept_background_margin": candidate.density_shell_accept_background_margin,
                "density_shell_reject_background_score": candidate.density_shell_reject_background_score,
                "density_shell_reject_background_margin": candidate.density_shell_reject_background_margin,
                "density_shell_reject_min_failed_shells": candidate.density_shell_reject_min_failed_shells,
                "oa_mse_identity_consensus_arbitration": candidate.oa_mse_identity_consensus_arbitration,
                "identity_consensus_scope": "post_density_shell_pre_pre_reject_identity_first_old_seen_new_background_arbitration",
                "identity_consensus_old_min_evidence_delta": candidate.identity_consensus_old_min_evidence_delta,
                "identity_consensus_old_min_anchor_delta": candidate.identity_consensus_old_min_anchor_delta,
                "identity_consensus_old_min_density_delta": candidate.identity_consensus_old_min_density_delta,
                "identity_consensus_seen_new_min_evidence_delta": candidate.identity_consensus_seen_new_min_evidence_delta,
                "identity_consensus_seen_new_min_anchor_delta": candidate.identity_consensus_seen_new_min_anchor_delta,
                "identity_consensus_seen_new_min_density_delta": candidate.identity_consensus_seen_new_min_density_delta,
                "identity_consensus_min_identity_margin": candidate.identity_consensus_min_identity_margin,
                "identity_consensus_background_accept_margin": candidate.identity_consensus_background_accept_margin,
                "identity_consensus_reject_background_score": candidate.identity_consensus_reject_background_score,
                "identity_consensus_reject_background_margin": candidate.identity_consensus_reject_background_margin,
                "identity_consensus_reject_min_identity_failures": candidate.identity_consensus_reject_min_identity_failures,
                "identity_consensus_support_background_cap": candidate.identity_consensus_support_background_cap,
                "identity_consensus_support_background_cap_quantile": candidate.identity_consensus_support_background_cap_quantile,
                "identity_consensus_support_background_cap_slack": candidate.identity_consensus_support_background_cap_slack,
                "identity_consensus_support_background_cap_min_anchors": candidate.identity_consensus_support_background_cap_min_anchors,
                "oa_mse_support_conformal_arbitration": candidate.oa_mse_support_conformal_arbitration,
                "support_conformal_scope": "post_identity_consensus_class_conditional_support_conformal_veto",
                "support_conformal_calibration_quantile": candidate.support_conformal_calibration_quantile,
                "support_conformal_conformity_slack": candidate.support_conformal_conformity_slack,
                "support_conformal_anchor_margin_slack": candidate.support_conformal_anchor_margin_slack,
                "support_conformal_background_score": candidate.support_conformal_background_score,
                "support_conformal_background_margin": candidate.support_conformal_background_margin,
                "support_conformal_hard_reject_margin": candidate.support_conformal_hard_reject_margin,
                "support_conformal_reject_min_failures": candidate.support_conformal_reject_min_failures,
                "support_conformal_reject_action": candidate.support_conformal_reject_action,
                "oa_mse_support_reconstruction_arbitration": candidate.oa_mse_support_reconstruction_arbitration,
                "support_reconstruction_scope": "post_support_conformal_class_local_reconstruction_residual_and_reciprocal_boundary_negative_veto",
                "support_reconstruction_rank": candidate.support_reconstruction_rank,
                "support_reconstruction_residual_quantile": candidate.support_reconstruction_residual_quantile,
                "support_reconstruction_residual_slack": candidate.support_reconstruction_residual_slack,
                "support_reconstruction_min_residual_floor": candidate.support_reconstruction_min_residual_floor,
                "support_reconstruction_negative_scale": candidate.support_reconstruction_negative_scale,
                "support_reconstruction_negative_margin": candidate.support_reconstruction_negative_margin,
                "support_reconstruction_hard_residual_margin": candidate.support_reconstruction_hard_residual_margin,
                "support_reconstruction_background_score": candidate.support_reconstruction_background_score,
                "support_reconstruction_background_margin": candidate.support_reconstruction_background_margin,
                "support_reconstruction_reject_min_failures": candidate.support_reconstruction_reject_min_failures,
                "support_reconstruction_reject_action": candidate.support_reconstruction_reject_action,
                "oa_mse_three_way_decision_head": candidate.oa_mse_three_way_decision_head,
                "three_way_decision_head_scope": (
                    "post_class_envelope_class_first_known_assignment_then_background_veto"
                    if candidate.three_way_decision_policy == "class_first"
                    else "post_class_envelope_pre_pre_reject_old_seen_new_pseudo_background_competition"
                ),
                "oa_mse_three_way_head_weight": candidate.oa_mse_three_way_head_weight,
                "three_way_head_temperature": candidate.three_way_head_temperature,
                "three_way_head_known_margin": candidate.three_way_head_known_margin,
                "three_way_head_background_margin": candidate.three_way_head_background_margin,
                "three_way_head_support_ce_weight": candidate.three_way_head_support_ce_weight,
                "three_way_head_pseudo_ce_weight": candidate.three_way_head_pseudo_ce_weight,
                "three_way_head_support_background_margin_weight": candidate.three_way_head_support_background_margin_weight,
                "three_way_head_pseudo_margin_weight": candidate.three_way_head_pseudo_margin_weight,
                "three_way_accept_prob": candidate.three_way_accept_prob,
                "three_way_reject_prob": candidate.three_way_reject_prob,
                "three_way_defer_prob": candidate.three_way_defer_prob,
                "three_way_known_background_margin": candidate.three_way_known_background_margin,
                "three_way_reject_margin": candidate.three_way_reject_margin,
                "three_way_old_seen_ambiguity_margin": candidate.three_way_old_seen_ambiguity_margin,
                "three_way_defer_action": candidate.three_way_defer_action,
                "three_way_decision_policy": candidate.three_way_decision_policy,
                "three_way_known_floor": candidate.three_way_known_floor,
                "three_way_known_floor_action": candidate.three_way_known_floor_action,
                "three_way_known_floor_old_min_evidence_delta": candidate.three_way_known_floor_old_min_evidence_delta,
                "three_way_known_floor_old_min_anchor_delta": candidate.three_way_known_floor_old_min_anchor_delta,
                "three_way_known_floor_old_min_anchor_margin": candidate.three_way_known_floor_old_min_anchor_margin,
                "three_way_known_floor_old_min_score_margin": candidate.three_way_known_floor_old_min_score_margin,
                "three_way_known_floor_seen_new_min_evidence_delta": candidate.three_way_known_floor_seen_new_min_evidence_delta,
                "three_way_known_floor_seen_new_min_anchor_delta": candidate.three_way_known_floor_seen_new_min_anchor_delta,
                "three_way_known_floor_seen_new_min_score_margin": candidate.three_way_known_floor_seen_new_min_score_margin,
                "three_way_known_floor_background_override_prob": candidate.three_way_known_floor_background_override_prob,
                "three_way_known_floor_background_override_margin": candidate.three_way_known_floor_background_override_margin,
                "pre_reject_support_neighborhood_retention": candidate.pre_reject_support_neighborhood_retention,
                "pre_reject_support_retention_old_min_evidence_delta": candidate.pre_reject_support_retention_old_min_evidence_delta,
                "pre_reject_support_retention_old_min_anchor_delta": candidate.pre_reject_support_retention_old_min_anchor_delta,
                "pre_reject_support_retention_old_min_anchor_margin": candidate.pre_reject_support_retention_old_min_anchor_margin,
                "pre_reject_support_retention_old_min_score_margin": candidate.pre_reject_support_retention_old_min_score_margin,
                "pre_reject_support_retention_seen_new_min_evidence_delta": candidate.pre_reject_support_retention_seen_new_min_evidence_delta,
                "pre_reject_support_retention_seen_new_min_anchor_delta": candidate.pre_reject_support_retention_seen_new_min_anchor_delta,
                "pre_reject_support_retention_seen_new_min_score_margin": candidate.pre_reject_support_retention_seen_new_min_score_margin,
                "pre_reject_support_retention_max_background_score": candidate.pre_reject_support_retention_max_background_score,
                "pre_reject_support_retention_max_background_margin": candidate.pre_reject_support_retention_max_background_margin,
                "pre_reject_support_retention_require_source_looo_pass": candidate.pre_reject_support_retention_require_source_looo_pass,
                "pre_reject_support_retention_source_looo_max_failures": candidate.pre_reject_support_retention_source_looo_max_failures,
                "target_receiver_ids": target_receiver_ids,
                "target_receiver_labels": target_receiver_label,
                "manysig_target_rx_index": manysig_target_rx_index,
                "manytx_target_rx_index": manytx_target_rx_index,
                "star_ground_channel_impl": candidate.star_ground_channel_impl,
                "target_channel_scenarios": candidate.target_channel_scenarios,
                "target_old_tx_ids": "0,1,2,3,4,5",
                "target_new_tx_ids": target_new_tx_ids,
                "target_old_support_per_tx": candidate.target_old_support_per_tx,
                "target_new_support_per_tx": candidate.target_new_support_per_tx,
                "target_old_query_per_tx": candidate.target_old_query_per_tx,
                "query_per_tx": candidate.query_per_tx,
                "manytx_receiver_specific_query_cap": candidate.query_per_tx,
                "manytx_query_cap_source": (
                    "next48ej_receiver_specific_manytx_unknown_query_availability_min30"
                    if candidate.query_per_tx <= 30
                    else "default"
                ),
                "target_old_k": candidate.target_old_support_per_tx,
                "target_new_k": candidate.target_new_support_per_tx,
                "k_shot_interpretation": (
                    "Stage2-B old/unknown-only higher-shot/saturation diagnostic: target-new support and query are excluded"
                    if new_tx_disabled and candidate.target_old_support_per_tx > 20
                    else
                    "Stage2-B old/unknown-only: target-new support and query are excluded"
                    if new_tx_disabled
                    else "Stage2-C uses symmetric target-old and seen-new support K; Stage2-B records old support K only"
                ),
                "target_new_tx_indices": target_new_tx_indices,
                "target_new_tx_labels": target_new_tx_ids,
                "unknown_tx_ids": PHASE2_UNKNOWN_TX_LABELS,
                "unknown_tx_indices": PHASE2_UNKNOWN_TX_INDICES,
                "unknown_tx_labels": PHASE2_UNKNOWN_TX_LABELS,
            },
            "gpu": f"GPU{candidate.gpu}",
            "estimated_run_path": f"/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}/{candidate.cid}",
            "estimated_log_path": f"/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}/{candidate.cid}.out",
            "cross_domain_target_metric": "old_acc/seen_new_acc by target receiver x transmitter",
            "satellite_channel_target_metric": (
                "unknown_FAR<=0.05 with per-scenario simplified LEO residual score table"
                if candidate.star_ground_channel_impl == STAR_GROUND_CHANNEL_IMPL
                else "unknown_FAR<=0.05 with per-scenario satellite/LEO score table"
            ),
            "allowed_tradeoff": "bounded coverage/defer is allowed only if unknown_FAR and old retention remain within gates",
            "must_not_regress_floor": (
                "OLD80_FIRST phase gate: old_acc>=0.80 before next open-world objectives; "
                "unknown_FAR<=0.05; no seen-new claim in Stage2-B"
                if candidate.stage2_priority_phase == "OLD80_FIRST"
                else "old_acc>=0.90; seen_new_acc>=0.75 for Stage2-C; unknown_FAR<=0.05"
            ),
            "comparability_status": f"COMPARABLE_STAGE2_TARGET_RECEIVER_{target_receiver_label}",
            "expected_failure_signals": "unknown_FAR>0.05; old-class forgetting; excessive uncertain/defer; adapter cost blow-up",
            "fallback_or_alternative": "downgrade to local repair if ManyTx tx_list labels or receiver-specific sample availability drift",
            "exact_command": f"bash code/scripts/launch_{run_id}.sh # candidate={candidate.cid}",
            "registry_key": f"{run_id}:{candidate.cid}",
            "command_hash": hashlib.sha256(
                (
                    f"{run_id}:{candidate.cid}:{candidate.gpu}:{candidate.command_kind}:"
                    f"{candidate.seed}:{candidate.target_old_support_per_tx}:"
                    f"{candidate.target_new_support_per_tx}:{candidate.update_module}:"
                    f"{target_receiver_label}:{manysig_target_rx_index}:{manytx_target_rx_index}:"
                    f"{candidate.star_ground_channel_impl}:{candidate.target_channel_scenarios}:"
                    f"{candidate.oa_mse_support_center_ce_weight}:"
                    f"{candidate.support_center_temperature}:"
                    f"{candidate.support_center_margin}:"
                    f"{candidate.oa_mse_known_coverage_weight}:"
                    f"{candidate.known_coverage_margin}:"
                    f"{candidate.known_coverage_min_affinity}:"
                    f"{candidate.known_coverage_max_samples}:"
                    f"{candidate.oa_mse_support_retention_guard}:"
                    f"{candidate.support_retention_guard_quantile}:"
                    f"{candidate.support_retention_guard_slack}:"
                    f"{candidate.oa_mse_two_branch_background_guard}:"
                    f"{candidate.two_branch_bg_min_score}:"
                    f"{candidate.two_branch_bg_min_margin}:"
                    f"{candidate.two_branch_old_support_evidence_delta}:"
                    f"{candidate.two_branch_old_anchor_delta}:"
                    f"{candidate.two_branch_old_anchor_margin}:"
                    f"{candidate.two_branch_seen_new_evidence_delta}:"
                    f"{candidate.two_branch_seen_new_anchor_delta}:"
                    f"{candidate.oa_mse_seen_new_registration_override}:"
                    f"{candidate.seen_new_override_min_evidence_delta}:"
                    f"{candidate.seen_new_override_min_anchor_delta}:"
                    f"{candidate.oa_mse_density_shell_gate}:"
                    f"{candidate.density_shell_old_min_evidence_delta}:"
                    f"{candidate.density_shell_seen_new_min_evidence_delta}:"
                    f"{candidate.density_shell_accept_background_margin}:"
                    f"{candidate.density_shell_reject_background_score}:"
                    f"{candidate.oa_mse_identity_consensus_arbitration}:"
                    f"{candidate.identity_consensus_old_min_evidence_delta}:"
                    f"{candidate.identity_consensus_old_min_anchor_delta}:"
                    f"{candidate.identity_consensus_old_min_density_delta}:"
                    f"{candidate.identity_consensus_seen_new_min_evidence_delta}:"
                    f"{candidate.identity_consensus_seen_new_min_anchor_delta}:"
                    f"{candidate.identity_consensus_seen_new_min_density_delta}:"
                    f"{candidate.identity_consensus_min_identity_margin}:"
                    f"{candidate.identity_consensus_background_accept_margin}:"
                    f"{candidate.identity_consensus_reject_background_score}:"
                    f"{candidate.identity_consensus_reject_background_margin}:"
                    f"{candidate.identity_consensus_reject_min_identity_failures}:"
                    f"{candidate.identity_consensus_support_background_cap}:"
                    f"{candidate.identity_consensus_support_background_cap_quantile}:"
                    f"{candidate.identity_consensus_support_background_cap_slack}:"
                    f"{candidate.identity_consensus_support_background_cap_min_anchors}:"
                    f"{candidate.oa_mse_support_conformal_arbitration}:"
                    f"{candidate.support_conformal_calibration_quantile}:"
                    f"{candidate.support_conformal_conformity_slack}:"
                    f"{candidate.support_conformal_anchor_margin_slack}:"
                    f"{candidate.support_conformal_background_score}:"
                    f"{candidate.support_conformal_background_margin}:"
                    f"{candidate.support_conformal_hard_reject_margin}:"
                    f"{candidate.support_conformal_reject_min_failures}:"
                    f"{candidate.support_conformal_reject_action}:"
                    f"{candidate.oa_mse_support_reconstruction_arbitration}:"
                    f"{candidate.support_reconstruction_rank}:"
                    f"{candidate.support_reconstruction_residual_quantile}:"
                    f"{candidate.support_reconstruction_residual_slack}:"
                    f"{candidate.support_reconstruction_min_residual_floor}:"
                    f"{candidate.support_reconstruction_negative_scale}:"
                    f"{candidate.support_reconstruction_negative_margin}:"
                    f"{candidate.support_reconstruction_hard_residual_margin}:"
                    f"{candidate.support_reconstruction_background_score}:"
                    f"{candidate.support_reconstruction_background_margin}:"
                    f"{candidate.support_reconstruction_reject_min_failures}:"
                    f"{candidate.support_reconstruction_reject_action}:"
                    f"{candidate.seen_new_override_min_affinity_delta}:"
                    f"{candidate.seen_new_override_min_residual_delta}:"
                    f"{candidate.seen_new_override_min_score_margin}:"
                    f"{candidate.seen_new_override_min_seen_vs_old_evidence_margin}:"
                    f"{candidate.seen_new_override_max_background_score}:"
                    f"{candidate.seen_new_override_max_background_margin}:"
                    f"{candidate.seen_new_override_min_support_knn_seen_new_minus_old}:"
                    f"{candidate.seen_new_override_min_support_knn_margin}:"
                    f"{candidate.oa_mse_three_way_decision_head}:"
                    f"{candidate.oa_mse_three_way_head_weight}:"
                    f"{candidate.three_way_head_temperature}:"
                    f"{candidate.three_way_accept_prob}:"
                    f"{candidate.three_way_reject_prob}:"
                    f"{candidate.three_way_known_background_margin}:"
                    f"{candidate.three_way_reject_margin}:"
                    f"{candidate.three_way_old_seen_ambiguity_margin}:"
                    f"{candidate.three_way_defer_action}:"
                    f"{candidate.three_way_decision_policy}:"
                    f"{candidate.three_way_head_support_ce_weight}:"
                    f"{candidate.three_way_head_pseudo_ce_weight}:"
                    f"{candidate.three_way_head_support_background_margin_weight}:"
                    f"{candidate.three_way_head_pseudo_margin_weight}:"
                    f"{candidate.three_way_known_floor}:"
                    f"{candidate.three_way_known_floor_action}:"
                    f"{candidate.three_way_known_floor_old_min_evidence_delta}:"
                    f"{candidate.three_way_known_floor_seen_new_min_evidence_delta}:"
                    f"{candidate.three_way_known_floor_background_override_prob}:"
                    f"{candidate.three_way_known_floor_background_override_margin}:"
                    f"{candidate.oa_mse_pre_reject_defer_arbitration}:"
                    f"{candidate.pre_reject_old_min_evidence_delta}:"
                    f"{candidate.pre_reject_seen_new_min_evidence_delta}:"
                    f"{candidate.pre_reject_max_background_score}:"
                    f"{candidate.pre_reject_reject_background_score}:"
                    f"{candidate.pre_reject_defer_action}:"
                    f"{candidate.pre_reject_support_neighborhood_retention}:"
                    f"{candidate.pre_reject_support_retention_old_min_evidence_delta}:"
                    f"{candidate.pre_reject_support_retention_seen_new_min_evidence_delta}:"
                    f"{candidate.pre_reject_support_retention_max_background_score}:"
                    f"{candidate.pre_reject_support_retention_max_background_margin}"
                ).encode("utf-8")
            ).hexdigest()[:16],
            "launchability_status": "phase2_launchable_wisig_manytx_resolved_labels_verified_sample_pool",
            "defer_reason": "",
            "sample_availability_status": f"CONFIRMED_PHASE2_POOL_MANYTX_EQ1_RX_{target_receiver_label}",
            "manytx_sample_audit": PHASE2_MANYTX_SAMPLE_AUDIT,
            "source_tx_ids": "0,1,2,3,4,5",
            "target_old_tx_ids": "0,1,2,3,4,5",
            "target_new_tx_ids": target_new_tx_ids,
            "model_output_semantics": (
                "old_label,reject,uncertain,defer" if new_tx_disabled else candidate.model_output_semantics
            ),
            "target_old_support_per_tx": candidate.target_old_support_per_tx,
            "target_new_support_per_tx": candidate.target_new_support_per_tx,
            "target_old_k": candidate.target_old_support_per_tx,
            "target_new_k": candidate.target_new_support_per_tx,
            "k_shot_interpretation": (
                "Stage2-B old/unknown-only higher-shot/saturation diagnostic: target-new support and query are excluded"
                if new_tx_disabled and candidate.target_old_support_per_tx > 20
                else
                "Stage2-B old/unknown-only: target-new support and query are excluded"
                if new_tx_disabled
                else "Stage2-C symmetric target-old and seen-new support; Stage2-A has no target support; Stage2-B old support only"
            ),
            "target_new_tx_indices": target_new_tx_indices,
            "new_tx_ids": target_new_tx_ids,
            "unknown_tx_ids": PHASE2_UNKNOWN_TX_LABELS,
            "unknown_tx_indices": PHASE2_UNKNOWN_TX_INDICES,
            "target_new_tx_labels": target_new_tx_ids,
            "unknown_tx_labels": PHASE2_UNKNOWN_TX_LABELS,
            "target_unknown_tx_ids": PHASE2_UNKNOWN_TX_LABELS,
            "target_unknown_tx_labels": PHASE2_UNKNOWN_TX_LABELS,
            "cen51_train_rxs": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "cen51_train_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "cen51_train_receiver_labels": PHASE2_SOURCE_RECEIVER_LABELS,
            "source_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "source_receiver_labels": PHASE2_SOURCE_RECEIVER_LABELS,
            "source_rxs": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "target_receiver_ids": target_receiver_ids,
            "target_receiver_label": target_receiver_label,
            "target_receiver_labels": target_receiver_label,
            "manysig_target_rx_index": manysig_target_rx_index,
            "manytx_target_rx_index": manytx_target_rx_index,
            "target_channel_view": "satellite/LEO",
            "star_ground_channel_impl": candidate.star_ground_channel_impl,
            "target_channel_scenarios": candidate.target_channel_scenarios,
            "clean_view_role": "control_only",
            "dataset_role": "terrestrial_proxy",
            "evidence_level": "receiver_x_transmitter_proxy_stress",
            "deployment_success_claim_allowed": False,
            "support_query_split_verified": True,
            "receiver_disjoint_verified": True,
            "tx_split_disjoint_verified": True,
            "target_old_leo_support": target_old_support,
            "target_old_leo_query": (
                f"target_old query tx=0-5 rx={target_receiver_ids} target_receiver_label="
                f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
                f"scenarios={candidate.target_channel_scenarios}"
            ),
            "target_new_leo_support": target_new_support,
            "target_new_leo_query": (
                "not_applicable_old_unknown_only"
                if new_tx_disabled
                else (
                    "target_new query tx="
                    f"{PHASE2_TARGET_NEW_TX_LABELS} rx={target_receiver_ids} target_receiver_label="
                    f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
                    f"scenarios={candidate.target_channel_scenarios}"
                )
            ),
            "unknown_leo_query": (
                "unknown query tx="
                f"{PHASE2_UNKNOWN_TX_LABELS} rx={target_receiver_ids} target_receiver_label="
                f"{target_receiver_label} star_ground_channel_impl={candidate.star_ground_channel_impl} "
                f"scenarios={candidate.target_channel_scenarios}"
            ),
            "k_shot": k_shot,
            "stage2c_success_metric_bundle": (
                "old_acc,seen_new_acc,H_old_new,unknown_FAR,new_acc_drop_pp,"
                "old_to_seen_new,seen_new_to_old,unknown_to_seen_new"
                if "Stage2-C" in mode
                else "not_applicable"
            ),
            "score_table_required_columns": "candidate_label,candidate_group,predicted_group,outcome_code,best_old_label,best_seen_new_label,best_old_score,best_seen_new_score,seen_new_minus_old_score,old_drift_cos,old_drift_dist,old_effective_rho,old_support_count,old_support_compactness,old_support_anchor_similarity,old_support_anchor_delta,old_support_evidence,old_support_evidence_delta,old_surrogate_evidence_delta,old_surrogate_reject_evidence_delta,old_support_quality,old_support_quality_delta,anchor_density,anchor_density_margin,anchor_density_delta,anchor_density_margin_delta,support_knn_label,support_knn_score,support_knn_margin,support_knn_old_score,support_knn_seen_new_score,support_knn_seen_new_minus_old,support_knn_topk,soft_mixture_score,soft_mixture_cos,soft_mixture_residual,soft_mixture_maha,soft_mixture_score_margin,soft_mixture_consistency_pass,class_envelope_evidence,class_envelope_reject,old_primary_label,old_primary_evidence_delta,old_primary_score_margin,old_primary_support_knn_pass,old_primary_drift_pass,old_primary_class_envelope_pass,old_primary_consistency_pass,old_primary_unknown_veto,old_primary_blocked_accept,old_primary_rescue_promoted,old_primary_rescue_blocked,density_shell_old_label,density_shell_seen_new_label,density_shell_accept,density_shell_reject,identity_consensus_old_label,identity_consensus_seen_new_label,identity_consensus_chosen_label,identity_consensus_old_score,identity_consensus_seen_new_score,identity_consensus_margin,identity_consensus_background_score,identity_consensus_background_margin,identity_consensus_support_background_cap,identity_consensus_support_background_cap_pass,identity_consensus_accept,identity_consensus_reject,support_conformal_label,support_conformal_score,support_conformal_floor,support_conformal_margin,support_conformal_anchor_margin,support_conformal_anchor_margin_floor,support_conformal_background_score,support_conformal_background_margin,support_conformal_pass,support_conformal_reject,support_conformal_failure_count,support_reconstruction_label,support_reconstruction_residual,support_reconstruction_residual_ceiling,support_reconstruction_residual_margin,support_reconstruction_center_cosine,support_reconstruction_negative_score,support_reconstruction_negative_margin,support_reconstruction_background_score,support_reconstruction_background_margin,support_reconstruction_pass,support_reconstruction_reject,support_reconstruction_failure_count,pair_verifier_label,pair_verifier_prob,pair_verifier_threshold,pair_verifier_called,pair_verifier_veto,three_way_label,three_way_old_score,three_way_seen_new_score,three_way_background_score,three_way_old_prob,three_way_seen_new_prob,three_way_background_prob,three_way_known_background_gap,three_way_background_margin,three_way_old_seen_gap,three_way_background_available,three_way_evidence_balanced_known_evidence,three_way_accept,three_way_reject,three_way_defer,pre_reject_arbitration_label,pre_reject_arbitration_score,pre_reject_arbitration_margin,pre_reject_arbitration_evidence_delta,pre_reject_arbitration_anchor_delta,pre_reject_arbitration_anchor_margin,pre_reject_arbitration_background_score,pre_reject_arbitration_background_margin,pre_reject_arbitration_background_available,pre_reject_arbitration_background_accept_ok,pre_reject_arbitration_background_defer_risk,pre_reject_arbitration_background_reject_risk,pre_reject_arbitration_evidence_ok,pre_reject_arbitration_support_retention,pre_reject_arbitration_extreme_background,pre_reject_arbitration_accept,pre_reject_arbitration_reject,pre_reject_arbitration_defer,pre_reject_arbitration_uncertain,retention_rescue_label,retention_rescue_evidence_delta,retention_rescue_background_score,retention_rescue_background_margin,retention_rescue_eligible,retention_rescue_accept,two_branch_background_score,two_branch_background_margin,two_branch_background_risk,two_branch_support_override,two_branch_background_reject,seen_new_override_label,seen_new_override_evidence_delta,seen_new_override_anchor_delta,seen_new_override_affinity_delta,seen_new_override_residual_delta,seen_new_override_seen_minus_old_evidence,seen_new_override_seen_minus_old_score,seen_new_override_background_score,seen_new_override_background_margin,seen_new_override_background_risk,seen_new_override_support_knn_seen_new_minus_old,seen_new_override_support_knn_margin,seen_new_override_support_knn_pass,seen_new_registration_override,min_accept_delta,seen_new_evidence,seen_new_anchor_similarity,seen_new_anchor_delta,unknown_score_kind,unknown_score",
            "old_support_query_split": old_split,
            "new_support_query_split": new_split,
            "cen51_base_checkpoint_or_config": "CEN51 strongest-generalization checkpoint; verify concrete path before completion claim",
            "threshold_selection_label_scope": candidate.threshold_selection_label_scope,
            "oa_mse_support_retention_guard": candidate.oa_mse_support_retention_guard,
            "support_retention_guard_quantile": candidate.support_retention_guard_quantile,
            "support_retention_guard_slack": candidate.support_retention_guard_slack,
            "support_retention_guard_scope": "target_old_support_only_no_unknown_query_threshold_fit",
            "oa_mse_two_branch_background_guard": candidate.oa_mse_two_branch_background_guard,
            "two_branch_background_guard_scope": "query_free_pseudo_background_risk_veto_with_old_seen_support_override_no_unknown_query_fit",
            "two_branch_bg_min_score": candidate.two_branch_bg_min_score,
            "two_branch_bg_min_margin": candidate.two_branch_bg_min_margin,
            "two_branch_old_support_evidence_delta": candidate.two_branch_old_support_evidence_delta,
            "two_branch_old_anchor_delta": candidate.two_branch_old_anchor_delta,
            "two_branch_old_anchor_margin": candidate.two_branch_old_anchor_margin,
            "two_branch_seen_new_evidence_delta": candidate.two_branch_seen_new_evidence_delta,
            "two_branch_seen_new_anchor_delta": candidate.two_branch_seen_new_anchor_delta,
            "oa_mse_pre_reject_defer_arbitration": candidate.oa_mse_pre_reject_defer_arbitration,
            "pre_reject_defer_arbitration_scope": "post_class_envelope_pre_siamese_known_vs_pseudo_background_no_unknown_query_fit",
            "pre_reject_old_min_evidence_delta": candidate.pre_reject_old_min_evidence_delta,
            "pre_reject_old_min_anchor_delta": candidate.pre_reject_old_min_anchor_delta,
            "pre_reject_old_min_anchor_margin": candidate.pre_reject_old_min_anchor_margin,
            "pre_reject_old_min_score_margin": candidate.pre_reject_old_min_score_margin,
            "pre_reject_seen_new_min_evidence_delta": candidate.pre_reject_seen_new_min_evidence_delta,
            "pre_reject_seen_new_min_anchor_delta": candidate.pre_reject_seen_new_min_anchor_delta,
            "pre_reject_seen_new_min_score_margin": candidate.pre_reject_seen_new_min_score_margin,
            "pre_reject_max_background_score": candidate.pre_reject_max_background_score,
            "pre_reject_max_background_margin": candidate.pre_reject_max_background_margin,
            "pre_reject_defer_background_score": candidate.pre_reject_defer_background_score,
            "pre_reject_defer_background_margin": candidate.pre_reject_defer_background_margin,
            "pre_reject_reject_background_score": candidate.pre_reject_reject_background_score,
            "pre_reject_reject_background_margin": candidate.pre_reject_reject_background_margin,
            "pre_reject_defer_action": candidate.pre_reject_defer_action,
            "pre_reject_support_neighborhood_retention": candidate.pre_reject_support_neighborhood_retention,
            "pre_reject_support_retention_old_min_evidence_delta": candidate.pre_reject_support_retention_old_min_evidence_delta,
            "pre_reject_support_retention_old_min_anchor_delta": candidate.pre_reject_support_retention_old_min_anchor_delta,
            "pre_reject_support_retention_old_min_anchor_margin": candidate.pre_reject_support_retention_old_min_anchor_margin,
            "pre_reject_support_retention_old_min_score_margin": candidate.pre_reject_support_retention_old_min_score_margin,
            "pre_reject_support_retention_seen_new_min_evidence_delta": candidate.pre_reject_support_retention_seen_new_min_evidence_delta,
            "pre_reject_support_retention_seen_new_min_anchor_delta": candidate.pre_reject_support_retention_seen_new_min_anchor_delta,
            "pre_reject_support_retention_seen_new_min_score_margin": candidate.pre_reject_support_retention_seen_new_min_score_margin,
            "pre_reject_support_retention_max_background_score": candidate.pre_reject_support_retention_max_background_score,
            "pre_reject_support_retention_max_background_margin": candidate.pre_reject_support_retention_max_background_margin,
            "oa_mse_retention_rescue_gate": candidate.oa_mse_retention_rescue_gate,
            "retention_rescue_scope": "post_reject_known_support_rescue_with_pseudo_background_cap_no_unknown_query_fit",
            "retention_rescue_old_min_evidence_delta": candidate.retention_rescue_old_min_evidence_delta,
            "retention_rescue_old_min_anchor_delta": candidate.retention_rescue_old_min_anchor_delta,
            "retention_rescue_old_min_anchor_margin": candidate.retention_rescue_old_min_anchor_margin,
            "retention_rescue_old_min_score_margin": candidate.retention_rescue_old_min_score_margin,
            "retention_rescue_seen_new_min_evidence_delta": candidate.retention_rescue_seen_new_min_evidence_delta,
            "retention_rescue_seen_new_min_anchor_delta": candidate.retention_rescue_seen_new_min_anchor_delta,
            "retention_rescue_seen_new_min_score_margin": candidate.retention_rescue_seen_new_min_score_margin,
            "retention_rescue_max_background_score": candidate.retention_rescue_max_background_score,
            "retention_rescue_max_background_margin": candidate.retention_rescue_max_background_margin,
            "retention_rescue_candidate_only": candidate.retention_rescue_candidate_only,
            "oa_mse_seen_new_registration_override": candidate.oa_mse_seen_new_registration_override,
            "seen_new_registration_override_scope": "target_seen_new_support_geometry_only_before_unknown_guard_no_unknown_query_fit",
            "seen_new_override_min_evidence_delta": candidate.seen_new_override_min_evidence_delta,
            "seen_new_override_min_anchor_delta": candidate.seen_new_override_min_anchor_delta,
            "seen_new_override_min_affinity_delta": candidate.seen_new_override_min_affinity_delta,
            "seen_new_override_min_residual_delta": candidate.seen_new_override_min_residual_delta,
            "seen_new_override_min_score_margin": candidate.seen_new_override_min_score_margin,
            "seen_new_override_min_seen_vs_old_evidence_margin": (
                candidate.seen_new_override_min_seen_vs_old_evidence_margin
            ),
            "seen_new_override_max_background_score": candidate.seen_new_override_max_background_score,
            "seen_new_override_max_background_margin": candidate.seen_new_override_max_background_margin,
            "seen_new_override_min_support_knn_seen_new_minus_old": (
                candidate.seen_new_override_min_support_knn_seen_new_minus_old
            ),
            "seen_new_override_min_support_knn_margin": candidate.seen_new_override_min_support_knn_margin,
            "seen_new_evidence_gate_calibration_scope": (
                "not_applicable_old_unknown_only"
                if new_tx_disabled
                else "target_old_support+target_new_support+pseudo_unknown_only"
            ),
            "seen_new_evidence_gate_unknown_query_calibration": False,
            "unknown_query_eval_only": True,
            "target_new_query_not_threshold_fit": True,
            "support_center_geometry_registration": candidate.oa_mse_support_center_ce_weight > 0,
            "support_center_geometry_scope": (
                "target_old_support_only_no_unknown_query_fit"
                if new_tx_disabled
                else "target_old_support+target_seen_new_support_only_no_unknown_query_fit"
            ),
        }
    )
    return item


def matrix_payload(run_id: str, candidates: Sequence[Candidate]) -> dict:
    phase1_count = sum(1 for c in candidates if c.command_kind == "phase1_safe_ssdg_ground_train")
    phase2_candidates = [c for c in candidates if c.command_kind != "phase1_safe_ssdg_ground_train"]
    phase2_count = len(phase2_candidates)
    phase1_only = bool(candidates) and phase1_count > 0 and phase2_count == 0
    phase2_only = bool(candidates) and phase1_count == 0 and all("oa_mse" in c.command_kind for c in candidates)
    target_new_disabled = bool(candidates) and all(
        str(c.new_tx_ids or "").strip().upper() in {"__NONE__", "NONE", "OLD_UNKNOWN_ONLY"}
        for c in candidates
    )
    policy_candidate = phase2_candidates[0] if phase2_candidates else (candidates[0] if candidates else None)
    return {
        "run_id": run_id,
        "n607_run_id": run_id,
        "expected_count": len(candidates),
        "lane_quota_mode": "phase1_retry_only" if phase1_only else ("phase2_only" if phase2_only else "canonical_mixed"),
        "phase1_rows_expected": phase1_count,
        "phase2_rows_expected": phase2_count,
        "phase2_gpu_utilization_policy": {
            "phase2_only": phase2_only,
            "max_active_per_gpu": policy_candidate.stage2_max_active_per_gpu if policy_candidate else None,
            "queued_rows_per_gpu": phase2_count // 8 if phase2_count and phase2_count % 8 == 0 else None,
        },
        "star_ground_channel_policy": {
            "default_impl": policy_candidate.star_ground_channel_impl if policy_candidate else STAR_GROUND_CHANNEL_IMPL,
            "target_channel_view": "satellite/LEO",
            "target_channel_scenarios": policy_candidate.target_channel_scenarios if policy_candidate else SIMPLIFIED_LEO_SCENARIOS,
            "legacy_scenarios": LEGACY_LEO_SCENARIOS,
        },
        "phase1_ground_dg_policy": {
            "status": "enabled" if phase1_count else "not_included_for_this_plan",
            "rows": phase1_count,
            "source_only": True,
            "cen51_role": "non_regression_experience_not_route_narrowing",
            "prototype_mask_modules": PHASE1_GROUND_PROTO_MASK_MODULES,
            "feature_distribution_objective": bool(phase1_count),
            "target_receiver_usage": "forbidden_in_phase1_training",
        },
        "stage2_sample_protocol": {
            "status": "active",
            "old_tx_ids": [0, 1, 2, 3, 4, 5],
            "cen51_train_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "source_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "source_receiver_labels": PHASE2_SOURCE_RECEIVER_LABELS,
            "target_receiver_label": PHASE2_TARGET_RECEIVER_LABEL,
            "target_receiver_pool_labels": [item["label"] for item in PHASE2_TARGET_RECEIVER_POOL],
            "target_receiver_ids": PHASE2_TARGET_RECEIVER_ALIAS,
            "target_new_tx_labels": "" if target_new_disabled else PHASE2_TARGET_NEW_TX_LABELS,
            "unknown_tx_labels": PHASE2_UNKNOWN_TX_LABELS,
            "recommended_k_shot_anchors": [1, 2, 5, 10, 15, 20, 50],
            "few_shot_upper_bound": 20,
        },
        "candidates": [_optimizer_matrix_item(run_id, c, idx) for idx, c in enumerate(candidates)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--plan", default="SMOKE", choices=["SMOKE", "CORE", "WISIG_NEWCLASS", "WISIG_NEWCLASS_CARD8", "WISIG_ENHANCED_CARD8", "PHASE1_GPU0_JOINTSAFE4", "OA_MSE_CARD3", "OA_MSE_PROXY32", "OA_MSE_BOUNDARY32", "OA_MSE_UNCERTAIN32", "OA_MSE_VETO32", "OA_MSE_CLASSCOND32", "OA_MSE_CALGUARD32", "OA_MSE_BALANCE64", "OA_MSE_SOFTMIX64", "OA_MSE_VOID64", "OA_MSE_SOFTVOID128", "OA_MSE_ANCHORGUARD128", "OA_MSE_MIXHEAD128", "OA_MSE_STRUCT48", "OA_MSE_SIMPLIFIED48", "OA_MSE_RETENTION48", "OA_MSE_SUPPORTRET48", "OA_MSE_TWOBRANCH48", "OA_MSE_REGHEAD48", "OA_MSE_GEOM48", "OA_MSE_TRIAGE48", "OA_MSE_LOOO48", "OA_MSE_CONSTRAIN48", "OA_MSE_ENVELOPE48", "OA_MSE_RESCUE48", "OA_MSE_PREREJECT48", "OA_MSE_THREEWAY48", "OA_MSE_COVFLOOR48", "OA_MSE_CLASSFIRST48", "OA_MSE_EVIBG48", "OA_MSE_SOFTTARGET48", "OA_MSE_NEGANCHOR48", "OA_MSE_DENSHELL48", "OA_MSE_IDCONS48", "OA_MSE_CONFORM48", "OA_MSE_RECON48", "OA_MSE_SOURCERISK48", "OA_MSE_SUPPORTCV48", "OA_MSE_BGCAP48", "OA_MSE_KRET48", "OA_MSE_RISKRET48", "OA_MSE_MANIFOLD48", "OA_MSE_H06_EVID48", "OA_MSE_H06_ARB48", "OA_MSE_H06_OLDUNK48", "OA_MSE_H06_BGTRAIN48", "OA_MSE_H06_RETOLD48", "OA_MSE_H06_OLDFIRST48", "OA_MSE_H06_OLDRELAX48", "OA_MSE_H06_OLDGEOM48", "OA_MSE_H06_OLDCONF48", "OA_MSE_H06_OLDBUDGET48", "OA_MSE_H06_OLDQUAL48", "OA_MSE_H06_OLDRISK48", "OA_MSE_H06_OLDFUSE48", "OA_MSE_H06_ROLLSAFE48", "OA_MSE_H06_OLDHEAD48", "OA_MSE_H06_OLDHEADFAR48", "OA_MSE_H06_OLDRECOV48"])
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Emit only the eight Phase1 Safe-SSDG rows for a retry run; intended for state-authorized Phase1 repair after a completed mixed run.",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"spaceborne_fewshot_da_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidates = make_candidates(plan=args.plan)
    if args.phase1_only:
        candidates = [c for c in candidates if c.command_kind == "phase1_safe_ssdg_ground_train"]
        if not candidates:
            raise ValueError(f"plan {args.plan} has no Phase1 Safe-SSDG rows for --phase1-only")
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"

    script_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps(matrix_payload(run_id, candidates), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(run_id, candidates), encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": len(candidates),
                "launcher": str(script_path),
                "matrix": str(matrix_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

