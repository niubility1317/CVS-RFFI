"""Truth-free full-query evaluation for the frozen D92-E0D five-arm matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
from cvsrffi.stage2_registration_resource_probe import measure_registration_call
from cvsrffi.stage2_d92_e0d_slim import (
    D92E0DSlimArmSpec,
    D92_E0D_ARMS,
    build_d92_e0d_fit,
    expected_total_component_fit_count,
)
from scripts import probe_d92_registration_balanced_covariance as d92_probe


SCHEMA_BY_ARM = {
    arm_id: f"cvs.phase2.d92_e0d.{arm_id.lower()}.full_query_evaluation.v1"
    for arm_id in D92_E0D_ARMS
}
_RESOURCE_FIELDS = (
    "schema",
    "registration_wall_time_ns",
    "registration_process_cpu_time_ns",
    "registration_baseline_rss_bytes",
    "registration_peak_rss_bytes",
    "registration_incremental_peak_working_set_bytes",
    "rss_sampler",
)
_FORBIDDEN_QUERY_ACCESS_FIELDS = (
    "d92_e0d_query_fit_access",
    "d92_e0d_query_update_access",
    "d92_e0d_query_selection_access",
    "d92_e0d_query_truth_access",
    "d92_e0d_query_role_oracle_access",
    "d92_e0d_query_class_quota_access",
    "d92_e0d_query_global_reassignment",
)
_STATE_ARRAY_NAMES = (
    "log_diag_fp32",
    "coef1_qint8",
    "coef2_qint8",
    "scale1_fp16",
    "scale2_fp16",
    "intercept_fp16",
    "coef_fp32",
    "intercept_fp32",
)
_OCF_ARM_IDS = frozenset(
    {"E0_OCF25", "E0_OCF50", "E0_FULL_MAXMIN_FLOORBOOST"}
)
_OCF_RECEIPT_FIELDS = (
    "d92_e0d_ocf_active",
    "d92_e0d_ocf_lambda",
    "d92_e0d_ocf_support_alignment_affine_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound",
)
_OCF_MAC_RECEIPT_FIELDS = _OCF_RECEIPT_FIELDS[2:5]
_FLOORBOOST_ARM_IDS = frozenset({"E0_FULL_MAXMIN_FLOORBOOST"})
_NEWGUARD_ARM_IDS = frozenset({"E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"})
_PARETO_DISTILL_ARM_IDS = frozenset({"E0_FULL_BLOCK_PARETO_DISTILL"})
_CSOAS_ARM_IDS = frozenset({"E0_FULL_CSOAS"})
_CCOC_ARM_IDS = frozenset({"E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"})
_TECHNICAL_SUPPORT_RECEIPT_ARM_IDS = frozenset({"E0_FULL_ONLY"}) | _CCOC_ARM_IDS


def _ccoc_raw_fields(*suffixes: str) -> frozenset[str]:
    """Return one explicit CCOC raw-receipt field collection."""

    return frozenset(f"d92_ccoc_{suffix}" for suffix in suffixes)


_CCOC_RAW_LIFECYCLE_FIELDS = _ccoc_raw_fields(
    "active",
    "fallback_active",
    "fallback_reason",
    "candidate_attempt_fit_count",
    "fallback_reference_fit_count",
    "candidate_statistic_receipt_available",
    "fallback_reference_full_head_byte_exact",
    "paired_e0_codec_state_equal",
    "query_rows_used",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_truth_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
_CCOC_RAW_ACTIVE_STATISTIC_FIELDS = _ccoc_raw_fields(
    "formula_revision",
    "formula",
    "old_rho",
    "new_rho",
    "old_group_class_count",
    "new_group_class_count",
    "old_offblock_norm_min",
    "old_offblock_norm_max",
    "new_offblock_norm_min",
    "new_offblock_norm_max",
    "old_pairwise_cosine_raw",
    "new_pairwise_cosine_raw",
    "canonicalization",
    "canonicalization_tie_policy",
    "crossblock_passes_per_class",
    "upper_block_count",
    "covariance_symmetric",
    "full_endpoint_reused",
    "full_endpoint_reuse",
    "additional_fit_count",
    "additional_full_fit_count",
    "additional_block_fit_count",
    "additional_loo_fit_count",
    "additional_fisher_fit_count",
    "additional_scan_count",
    "block_fit_count",
    "loo_fit_count",
    "fisher_fit_count",
    "scan_count",
    "hyperparameter_scan_count",
    "weight_scan_count",
    "dense_solve_count",
    "cholesky_check_count",
    "cholesky_endpoint_check_count",
    "cholesky_final_check_count",
    "cholesky_pass",
    "old_endpoint_cholesky_min_diagonal",
    "new_endpoint_cholesky_min_diagonal",
    "final_cholesky_min_diagonal",
    "support_macs_upper_bound",
    "workspace_upper_accumulators_bytes",
    "workspace_cross_block_buffer_bytes",
    "workspace_residual_buffer_bytes",
    "workspace_numeric_bytes_upper_bound",
    "workspace_frozen_k10_numeric_bytes_upper_bound",
    "workspace_candidate_covariance_result_bytes",
    "workspace_candidate_covariance_block_workspace_bytes",
    "workspace_candidate_covariance_row_workspace_bytes",
    "workspace_candidate_covariance_full_buffer_count_upper_bound",
    "workspace_candidate_covariance_mix_live_bytes_upper_bound",
    "support_transient_bytes_upper_bound",
    "persistent_state_bytes_delta",
    "persistent_bytes_delta",
    "query_state_bytes_delta",
    "query_bytes_delta",
    "query_macs_delta",
    "query_macs",
)
_CCOC_RAW_ACTIVE_COMPILE_FIELDS = _ccoc_raw_fields(
    "compile_solve_count",
    "full_solve_count",
    "full_dense_288_solve_count",
    "compiled_cholesky_check_count",
    "covariance_equation_residual_max",
)
_CCOC_RAW_ACTIVE_FIELDS = (
    _CCOC_RAW_LIFECYCLE_FIELDS
    | _CCOC_RAW_ACTIVE_STATISTIC_FIELDS
    | _CCOC_RAW_ACTIVE_COMPILE_FIELDS
)
_CCOC_RAW_NUMERIC_FALLBACK_WITH_STATISTICS_FIELDS = (
    _CCOC_RAW_ACTIVE_FIELDS - _CCOC_RAW_ACTIVE_COMPILE_FIELDS
)
_CCOC_RAW_NUMERIC_FALLBACK_WITHOUT_STATISTICS_FIELDS = _ccoc_raw_fields(
    "active",
    "fallback_active",
    "fallback_reason",
    "formula_revision",
    "formula",
    "old_rho",
    "new_rho",
    "old_group_class_count",
    "new_group_class_count",
    "canonicalization",
    "canonicalization_tie_policy",
    "full_endpoint_reused",
    "full_endpoint_reuse",
    "additional_fit_count",
    "additional_full_fit_count",
    "additional_block_fit_count",
    "additional_loo_fit_count",
    "additional_fisher_fit_count",
    "additional_scan_count",
    "hyperparameter_scan_count",
    "weight_scan_count",
    "dense_solve_count",
    "cholesky_check_count",
    "cholesky_pass",
    "support_macs_upper_bound",
    "support_transient_bytes_upper_bound",
    "persistent_state_bytes_delta",
    "persistent_bytes_delta",
    "query_state_bytes_delta",
    "query_bytes_delta",
    "query_macs_delta",
    "query_macs",
    "candidate_attempt_fit_count",
    "fallback_reference_fit_count",
    "candidate_statistic_receipt_available",
    "fallback_reference_full_head_byte_exact",
    "paired_e0_codec_state_equal",
    "query_rows_used",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_truth_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
_CCOC_RAW_CODEC_FALLBACK_FIELDS = _ccoc_raw_fields(
    "active",
    "fallback_active",
    "fallback_reason",
    "formula_revision",
    "formula",
    "old_rho",
    "new_rho",
    "old_group_class_count",
    "new_group_class_count",
    "candidate_attempt_fit_count",
    "fallback_reference_fit_count",
    "candidate_statistic_receipt_available",
    "fallback_reference_full_head_byte_exact",
    "paired_e0_codec_state_equal",
    "query_rows_used",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_truth_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
    "codec_fallback_component_execution_count",
    "codec_fallback_scope",
)
_CCOC_RAW_INACTIVE_FIELDS = _ccoc_raw_fields(
    "active",
    "fallback_active",
    "fallback_reason",
    "formula_revision",
    "status",
    "old_rho",
    "new_rho",
    "old_group_class_count",
    "new_group_class_count",
    "canonicalization",
    "canonicalization_tie_policy",
    "full_endpoint_reused",
    "full_endpoint_reuse",
    "additional_fit_count",
    "additional_full_fit_count",
    "additional_block_fit_count",
    "additional_loo_fit_count",
    "additional_fisher_fit_count",
    "additional_scan_count",
    "hyperparameter_scan_count",
    "weight_scan_count",
    "dense_solve_count",
    "cholesky_check_count",
    "cholesky_pass",
    "support_macs_upper_bound",
    "support_transient_bytes_upper_bound",
    "persistent_state_bytes_delta",
    "persistent_bytes_delta",
    "query_state_bytes_delta",
    "query_bytes_delta",
    "query_macs_delta",
    "query_macs",
    "query_rows_used",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_truth_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
_CCOC_MIRROR_G0_FIELDS = frozenset(
    {
        "d92_e0d_ccoc_g0_eligible",
        "d92_e0d_ccoc_g0_block_reason",
    }
)
_CCOC_MIRROR_INACTIVE_ONLY_FIELDS = _ccoc_raw_fields(
    "candidate_attempt_fit_count",
    "fallback_reference_fit_count",
    "candidate_statistic_receipt_available",
    "fallback_reference_full_head_byte_exact",
    "paired_e0_codec_state_equal",
)
_CCOC_MIRROR_INACTIVE_ONLY_FIELDS = frozenset(
    field.replace("d92_ccoc_", "d92_e0d_ccoc_")
    for field in _CCOC_MIRROR_INACTIVE_ONLY_FIELDS
)
_CCOC_COVARIANCE_MIX_WORKSPACE_EXPECTED = {
    "d92_ccoc_workspace_candidate_covariance_result_bytes": 288 * 288 * 8,
    "d92_ccoc_workspace_candidate_covariance_block_workspace_bytes": 160 * 160 * 8,
    "d92_ccoc_workspace_candidate_covariance_row_workspace_bytes": 160 * 8,
    "d92_ccoc_workspace_candidate_covariance_full_buffer_count_upper_bound": 1,
    "d92_ccoc_workspace_candidate_covariance_mix_live_bytes_upper_bound": (
        288 * 288 * 8 + 160 * 160 * 8 + 160 * 8
    ),
}
_TPCE_ARM_IDS = frozenset({"E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"})
_TPCE_RECEIPT_SUFFIXES = (
    "active",
    "fallback_active",
    "fallback_reason",
    "quantile",
    "quantile_method",
    "state_postprocess_mode",
    "direct_state_publish",
    "requantize_call_count",
    "e0_state_sha256",
    "final_state_sha256",
    "changed_code2_count",
    "requested_atomic_exchange_count",
    "applied_atomic_exchange_count",
    "aggregate_saturation_count",
    "generated_atomic_exchange_count",
    "selected_atomic_exchange_count",
    "rejected_atomic_exchange_count",
    "greedy_step_count",
    "code1_byte_exact",
    "scale1_byte_exact",
    "scale2_byte_exact",
    "intercept_byte_exact",
    "log_diag_byte_exact",
    "old_tail_count_by_class",
    "pooled_new_tail_count",
    "tied_competitor_relation_count",
    "guard_tolerance",
    "old_tail_gain_by_class",
    "old_tail_min_gain",
    "pooled_new_cross_tail_gain",
    "pooled_new_allclass_tail_gain",
    "old_to_new_hinge_delta",
    "new_to_old_hinge_delta",
    "support_guard_pass",
    "class_permutation_equivariant",
    "old_group_uniform_shift",
    "support_score_macs_upper_bound",
    "support_coordinate_comparisons_upper_bound",
    "support_macs_upper_bound",
    "support_transient_bytes_upper_bound",
    "persistent_state_bytes_delta",
    "component_fit_count",
)
_TPCE_RECEIPT_FIELDS = tuple(
    f"d92_e0d_tpce_{suffix}" for suffix in _TPCE_RECEIPT_SUFFIXES
)
_TCRA_ARM_IDS = frozenset({"E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"})
_TCRA_RECEIPT_SUFFIXES = (
    "active",
    "fallback_active",
    "fallback_reason",
    "quantile",
    "quantile_method",
    "state_postprocess_mode",
    "final_gate_revision",
    "direct_state_publish",
    "requantize_call_count",
    "e0_state_sha256",
    "final_state_sha256",
    "modified_state_field_names",
    "changed_code2_count",
    "state_delta_code2_l1",
    "requested_atomic_ascent_count",
    "applied_atomic_ascent_count",
    "generated_atomic_ascent_count",
    "selected_atomic_ascent_count",
    "rejected_atomic_ascent_count",
    "prefix_guard_rejected_count",
    "greedy_step_count",
    "aggregate_saturation_count",
    "code1_byte_exact",
    "scale1_byte_exact",
    "scale2_byte_exact",
    "intercept_byte_exact",
    "log_diag_byte_exact",
    "coef2_byte_exact",
    "old_tail_count_by_class",
    "pooled_new_tail_count",
    "guard_tolerance",
    "old_tail_gain_by_class",
    "old_tail_min_gain",
    "old_tail_gain_sum",
    "old_tail_strict_positive_count",
    "pooled_new_cross_tail_gain",
    "pooled_new_allclass_tail_gain",
    "old_to_new_hinge_delta",
    "new_to_old_hinge_delta",
    "support_guard_pass",
    "safe_directional_pass",
    "true_class_row_only",
    "competitor_code_decrement_count",
    "class_permutation_equivariant",
    "row_permutation_invariant",
    "old_group_uniform_shift",
    "support_full_score_evaluation_count",
    "support_analytic_candidate_evaluation_count",
    "support_score_macs_upper_bound",
    "support_coordinate_comparisons_upper_bound",
    "support_macs_upper_bound",
    "support_transient_bytes_upper_bound",
    "persistent_state_bytes_delta",
    "component_fit_count",
    "query_rows_used",
    "query_macs",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_truth_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
_TCRA_RECEIPT_FIELDS = tuple(
    f"d92_e0d_tcra_{suffix}" for suffix in _TCRA_RECEIPT_SUFFIXES
)
_FLOORBOOST_RECEIPT_FIELDS = (
    "d92_e0d_floorboost_active",
    "d92_e0d_floorboost_lambda",
    "d92_e0d_floorboost_quantile",
    "d92_e0d_floorboost_quantile_method",
    "d92_e0d_floorboost_kappa",
    "d92_e0d_floorboost_fallback_active",
    "d92_e0d_floorboost_fallback_reason",
    "d92_e0d_floorboost_new_rows_byte_exact",
    "d92_e0d_floorboost_full_head_byte_exact",
    "d92_e0d_floorboost_old_bias_zero_sum_residual_abs",
    "d92_e0d_floorboost_old_intercept_mean_residual_abs",
    "d92_e0d_floorboost_max_abs_delta_over_rms",
    "d92_e0d_floorboost_full_old_rms",
    "d92_e0d_floorboost_retention_score_by_old_class",
    "d92_e0d_floorboost_registration_drift_by_old_class",
    "d92_e0d_floorboost_delta_bias_by_old_class",
    "d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound",
    "d92_e0d_floorboost_support_retention_affine_macs_upper_bound",
    "d92_e0d_floorboost_support_bias_calibration_macs_upper_bound",
    "d92_e0d_floorboost_support_macs_upper_bound",
    "d92_e0d_floorboost_support_transient_bytes_upper_bound",
    "d92_e0d_floorboost_persistent_state_bytes_delta",
)
_NEWGUARD_RECEIPT_FIELDS = (
    "d92_e0d_newguard_active",
    "d92_e0d_newguard_fallback_active",
    "d92_e0d_newguard_fallback_reason",
    "d92_e0d_newguard_full_component_fit_count",
    "d92_e0d_newguard_new_rows_byte_exact",
    "d92_e0d_newguard_deployment_new_rows_byte_exact",
    "d92_e0d_newguard_tau_old_envelope_shift",
    "d92_e0d_newguard_deployment_protection_pass",
    "d92_e0d_newguard_full_head_byte_exact",
    "d92_e0d_newguard_deployment_strength_scale",
    "d92_e0d_newguard_deployment_candidate_count",
    "d92_e0d_newguard_deployment_full_head_byte_exact",
    "d92_e0d_newguard_deployment_codec_roundtrip_count",
    "d92_e0d_newguard_deployment_codec_macs_upper_bound",
    "d92_e0d_newguard_nullspace_rank",
    "d92_e0d_newguard_rank_threshold",
    "d92_e0d_newguard_max_abs_Xnew_internal_residual",
    "d92_e0d_newguard_closure_tolerance",
    "d92_e0d_newguard_old_group_zero_sum_residual_max_abs",
    "d92_e0d_newguard_new_support_old_envelope_change_max_abs_error",
    "d92_e0d_newguard_deployment_max_abs_Xnew_internal_residual",
    "d92_e0d_newguard_deployment_old_group_zero_sum_residual_max_abs",
    "d92_e0d_newguard_deployment_new_support_old_envelope_change_max_abs_error",
    "d92_e0d_newguard_protection_tolerance",
    "d92_e0d_newguard_new_support_min_margin_change",
    "d92_e0d_newguard_new_support_old_envelope_change_max",
    "d92_e0d_newguard_deployment_new_support_min_margin_change",
    "d92_e0d_newguard_deployment_new_support_old_envelope_change_max",
    "d92_e0d_newguard_tail_margin_change_by_old_class",
    "d92_e0d_newguard_deployment_tail_margin_change_by_old_class",
    "d92_e0d_newguard_residual_l2_by_old_class",
    "d92_e0d_newguard_maxmin_objective",
    "d92_e0d_newguard_trust_region_utilization",
    "d92_e0d_newguard_support_optimization_macs_upper_bound",
    "d92_e0d_newguard_support_transient_bytes_upper_bound",
    "d92_e0d_newguard_persistent_state_bytes_delta",
    "d92_e0d_newguard_query_rows_used",
    "d92_e0d_newguard_query_macs",
    "d92_e0d_newguard_query_fit_access",
    "d92_e0d_newguard_query_update_access",
    "d92_e0d_newguard_query_selection_access",
    "d92_e0d_newguard_query_truth_access",
    "d92_e0d_newguard_query_role_oracle_access",
    "d92_e0d_newguard_query_class_quota_access",
    "d92_e0d_newguard_query_global_reassignment",
)
_PARETO_DISTILL_RECEIPT_FIELDS = (
    "d92_e0d_pareto_distill_mode",
    "d92_e0d_pareto_distill_active",
    "d92_e0d_pareto_distill_fallback_active",
    "d92_e0d_pareto_distill_fallback_reason",
    "d92_e0d_pareto_distill_local_valid",
    "d92_e0d_pareto_distill_full_head_byte_exact",
    "d92_e0d_pareto_distill_deployed_support_constraints_pass",
    "d92_e0d_pareto_distill_deployed_full_head_byte_exact",
    "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs",
    "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum",
    "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass",
    "d92_e0d_pareto_distill_deployed_e0_affine_sha256",
    "d92_e0d_pareto_distill_deployed_candidate_affine_sha256",
    "d92_e0d_pareto_distill_full_solve_count",
    "d92_e0d_pareto_distill_block_solve_count",
    "d92_e0d_pareto_distill_loo_fit_count",
    "d92_e0d_pareto_distill_fisher_fit_count",
    "d92_e0d_pareto_distill_component_fit_count",
    "d92_e0d_pareto_distill_covariance_estimation_count",
    "d92_e0d_pareto_distill_robust_center_transform_count",
    "d92_e0d_pareto_distill_query_rows_used",
    "d92_e0d_pareto_distill_query_macs",
    "d92_e0d_pareto_distill_query_fit_access",
    "d92_e0d_pareto_distill_query_update_access",
    "d92_e0d_pareto_distill_query_selection_access",
    "d92_e0d_pareto_distill_query_truth_access",
    "d92_e0d_pareto_distill_query_role_oracle_access",
    "d92_e0d_pareto_distill_query_class_quota_access",
    "d92_e0d_pareto_distill_query_global_reassignment",
)


class D92E0DQueryEvaluationError(ValueError):
    """Raised when an E0D prediction or protocol closure drifts."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _state_fingerprint_sha256(state: Any) -> str:
    """Hash every persisted state array with its fixed schema metadata."""

    try:
        metadata = {
            "schema": str(state.schema),
            "classes": [str(value) for value in state.classes],
            "old_class_count": int(state.old_class_count),
            "covariance_policy": str(state.covariance_policy),
            "arrays": list(_STATE_ARRAY_NAMES),
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError("D92-E0D state metadata drift") from error
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(metadata))
    for name in _STATE_ARRAY_NAMES:
        try:
            array = np.ascontiguousarray(np.asarray(getattr(state, name)))
        except (AttributeError, TypeError, ValueError) as error:
            raise D92E0DQueryEvaluationError(
                f"D92-E0D state array missing: {name}"
            ) from error
        if array.ndim == 0 or not np.isfinite(array).all():
            raise D92E0DQueryEvaluationError(
                f"D92-E0D state array is invalid: {name}"
            )
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_bytes({"shape": list(array.shape)}))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _final_state_technical_support_receipt(
    result: Any,
    *,
    arm_id: str,
    old_support_features: Any,
    old_support_labels: Any,
    old_classes: Any,
    new_support_features: Any,
    new_support_labels: Any,
    new_classes: Any,
) -> dict[str, Any]:
    """Derive an ephemeral final-state support receipt without retaining rows."""

    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

    if arm_id in _CCOC_ARM_IDS:
        receipt_schema = "cvs.phase2.d92_ccoc.support_state_receipt.v1"
    elif arm_id == "E0_FULL_ONLY":
        receipt_schema = "cvs.phase2.d92_e0d.support_state_receipt.v1"
    else:
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support receipt arm drift"
        )
    state = result.state
    old_registry = tuple(str(value) for value in old_classes)
    new_registry = tuple(str(value) for value in new_classes)
    registry = old_registry + new_registry
    if (
        tuple(str(value) for value in state.classes) != registry
        or int(state.old_class_count) != len(old_registry)
        or len(old_registry) != int(OLD_CLASS_COUNT)
    ):
        raise D92E0DQueryEvaluationError("D92-E0D technical support registry drift")
    try:
        old_rows = np.asarray(old_support_features, dtype=np.float32)
        new_rows = np.asarray(new_support_features, dtype=np.float32)
        rows = np.concatenate((old_rows, new_rows), axis=0)
        labels = np.concatenate(
            (
                np.asarray(old_support_labels).astype(str),
                np.asarray(new_support_labels).astype(str),
            )
        )
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support input drift"
        ) from error
    if (
        rows.ndim != 2
        or rows.shape[1] != 288
        or labels.shape != (len(rows),)
        or not np.isfinite(rows).all()
    ):
        raise D92E0DQueryEvaluationError("D92-E0D technical support shape drift")
    class_index = {handle: index for index, handle in enumerate(registry)}
    try:
        targets = np.asarray(
            [class_index[str(value)] for value in labels.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support label drift"
        ) from error
    support_counts = np.bincount(targets, minlength=len(registry))
    if len(support_counts) != len(registry) or np.any(support_counts <= 0):
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support class closure drift"
        )
    transformed = d42._transform(rows, state.log_diag_fp32)
    coefficients = d42.decode_d42_coefficients(state)
    intercept = np.asarray(state.intercept_fp16, dtype=np.float32)
    if (
        transformed.shape != rows.shape
        or coefficients.shape != (len(registry), 288)
        or intercept.shape != (len(registry),)
        or not np.isfinite(transformed).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise D92E0DQueryEvaluationError("D92-E0D technical support state drift")
    scores = transformed @ coefficients.T + intercept
    if not np.isfinite(scores).all():
        raise D92E0DQueryEvaluationError("D92-E0D technical support score drift")
    canonical: list[tuple[str, bytes, int, str]] = []
    for row_index, target in enumerate(targets.tolist()):
        handle = registry[int(target)]
        row_bytes = np.ascontiguousarray(
            transformed[row_index], dtype=np.float32
        ).tobytes(order="C")
        row_digest = hashlib.sha256()
        row_digest.update(b"cvs.phase2.d92.ccoc.support-row.v1\0")
        row_digest.update(handle.encode("utf-8"))
        row_digest.update(row_bytes)
        canonical.append((handle, row_bytes, row_index, row_digest.hexdigest()))
    canonical.sort(key=lambda row: (row[0], row[1]))
    identity_digest = hashlib.sha256()
    margins: list[dict[str, Any]] = []
    for handle, row_bytes, row_index, row_handle in canonical:
        target = int(targets[row_index])
        opposite = (
            np.arange(len(registry)) >= int(OLD_CLASS_COUNT)
            if target < int(OLD_CLASS_COUNT)
            else np.arange(len(registry)) < int(OLD_CLASS_COUNT)
        )
        margin = float(scores[row_index, target] - np.max(scores[row_index, opposite]))
        if not np.isfinite(margin):
            raise D92E0DQueryEvaluationError(
                "D92-E0D technical support margin drift"
            )
        identity_digest.update(handle.encode("utf-8"))
        identity_digest.update(row_bytes)
        margins.append(
            {
                "support_handle": row_handle,
                "canonical_row_handle": row_handle,
                "class_handle": handle,
                "cross_group_margin": margin,
            }
        )
    block_names = ("z160", "fft96", "rf32")
    support_block_absmax = {
        name: float(np.max(np.abs(transformed[:, block])))
        for name, block in zip(block_names, d42.BLOCK_SLICES)
    }
    scale1_block_max_abs = np.max(
        np.abs(np.asarray(state.scale1_fp16, dtype=np.float32)), axis=0
    )
    scale2_block_max_abs = np.max(
        np.abs(np.asarray(state.scale2_fp16, dtype=np.float32)), axis=0
    )
    if (
        not np.isfinite(np.asarray(tuple(support_block_absmax.values()))).all()
        or not np.isfinite(scale1_block_max_abs).all()
        or not np.isfinite(scale2_block_max_abs).all()
    ):
        raise D92E0DQueryEvaluationError("D92-E0D technical support scale drift")
    state_fingerprint = _state_fingerprint_sha256(state)
    return {
        "schema": receipt_schema,
        "old_class_count": int(OLD_CLASS_COUNT),
        "registered_class_count": int(len(registry)),
        "canonical_class_handles": tuple(sorted(registry)),
        "canonical_support_identity_sha256": identity_digest.hexdigest(),
        "canonical_support_handles": tuple(row[3] for row in canonical),
        "cross_group_margin_by_support_handle": tuple(margins),
        "A_b": support_block_absmax,
        "support_block_absmax": support_block_absmax,
        "after_state_fingerprint_sha256": state_fingerprint,
        "final_d42_coefficient_bias_state_sha256": state_fingerprint,
        "scale1_block_max_abs": tuple(float(value) for value in scale1_block_max_abs),
        "scale2_block_max_abs": tuple(float(value) for value in scale2_block_max_abs),
        "query_access": False,
        "truth_access": False,
    }


def _resource_receipt(audit: dict[str, Any]) -> dict[str, Any]:
    if any(field not in audit for field in _RESOURCE_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D registration resource receipt drift")
    result = {field: audit[field] for field in _RESOURCE_FIELDS}
    if (
        result["schema"] != "cvs.phase2.registration_resource_receipt.v1"
        or int(result["registration_wall_time_ns"]) < 0
        or int(result["registration_process_cpu_time_ns"]) < 0
        or int(result["registration_baseline_rss_bytes"]) <= 0
        or int(result["registration_peak_rss_bytes"])
        < int(result["registration_baseline_rss_bytes"])
        or int(result["registration_incremental_peak_working_set_bytes"]) < 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D registration resource values drift")
    return result


def _ocf_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
) -> dict[str, Any]:
    floorboost_fallback = bool(
        arm.arm_id in _FLOORBOOST_ARM_IDS
        and audit.get("d92_e0d_floorboost_fallback_active") is True
    )
    expected_active = bool(
        registered
        and int(k_shot) > 2
        and arm.arm_id in _OCF_ARM_IDS
        and not floorboost_fallback
    )
    if any(field not in audit for field in _OCF_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D OCF support receipt drift")
    receipt = {field: audit[field] for field in _OCF_RECEIPT_FIELDS}
    if receipt["d92_e0d_ocf_active"] is not expected_active:
        raise D92E0DQueryEvaluationError("D92-E0D OCF support receipt drift")
    if not expected_active:
        if (
            receipt["d92_e0d_ocf_lambda"] is not None
            or receipt[
                "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound"
            ]
            is not None
        ):
            raise D92E0DQueryEvaluationError("D92-E0D OCF support receipt drift")
        for field in _OCF_MAC_RECEIPT_FIELDS:
            value = receipt[field]
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise D92E0DQueryEvaluationError(
                    "D92-E0D OCF support receipt drift"
                ) from error
            if not np.isfinite(numeric) or numeric != 0.0:
                raise D92E0DQueryEvaluationError(
                    "D92-E0D OCF support receipt drift"
                )
        return receipt
    try:
        expected_lambda = float(arm.ocf_lambda)
        actual_lambda = float(receipt["d92_e0d_ocf_lambda"])
        affine_macs_upper_bound = float(
            receipt[
                "d92_e0d_ocf_support_alignment_affine_macs_upper_bound"
            ]
        )
        contrast_mix_macs_upper_bound = float(
            receipt[
                "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"
            ]
        )
        macs_upper_bound = float(
            receipt["d92_e0d_ocf_support_alignment_macs_upper_bound"]
        )
        transient_bytes_upper_bound = float(
            receipt["d92_e0d_ocf_support_alignment_transient_bytes_upper_bound"]
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError(
            "D92-E0D OCF support receipt drift"
        ) from error
    expected_affine_macs = int(
        2
        * (OLD_CLASS_COUNT * int(k_shot))
        * OLD_CLASS_COUNT
        * 288
    )
    expected_contrast_mix_macs = int(
        5 * OLD_CLASS_COUNT * (288 + 1)
    )
    if (
        not np.isfinite(expected_lambda)
        or not np.isfinite(actual_lambda)
        or actual_lambda != expected_lambda
        or not np.isfinite(affine_macs_upper_bound)
        or affine_macs_upper_bound < 0.0
        or affine_macs_upper_bound != expected_affine_macs
        or not np.isfinite(contrast_mix_macs_upper_bound)
        or contrast_mix_macs_upper_bound < 0.0
        or contrast_mix_macs_upper_bound != expected_contrast_mix_macs
        or not np.isfinite(macs_upper_bound)
        or macs_upper_bound < 0.0
        or macs_upper_bound
        != affine_macs_upper_bound + contrast_mix_macs_upper_bound
        or not np.isfinite(transient_bytes_upper_bound)
        or transient_bytes_upper_bound < 0.0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D OCF support receipt drift")
    return receipt


def _floorboost_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    """Validate the four-state floorboost receipt without reading query data."""

    if any(field not in audit for field in _FLOORBOOST_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    receipt = {field: audit[field] for field in _FLOORBOOST_RECEIPT_FIELDS}
    applies = arm.arm_id in _FLOORBOOST_ARM_IDS
    active_state = bool(registered and int(k_shot) > 2 and applies)
    if not applies:
        if (
            receipt["d92_e0d_floorboost_active"] is not False
            or receipt["d92_e0d_floorboost_fallback_active"] is not False
            or receipt["d92_e0d_floorboost_fallback_reason"] is not None
        ):
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
        return receipt
    if not registered:
        expected_reason = "NOT_REGISTERED_STATE"
    elif int(k_shot) <= 2:
        expected_reason = "K1_K2_EXACT_D92_FULL_ALIAS"
    else:
        expected_reason = None
    fallback_active = receipt["d92_e0d_floorboost_fallback_active"]
    active = receipt["d92_e0d_floorboost_active"]
    reason = receipt["d92_e0d_floorboost_fallback_reason"]
    if not active_state:
        if active is not False or fallback_active is not False or reason != expected_reason:
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
        return receipt
    if fallback_active is True:
        if active is not False or not isinstance(reason, str) or not reason:
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    elif fallback_active is False:
        if active is not True or reason is not None:
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    else:
        raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    try:
        values = {
            "lambda": float(receipt["d92_e0d_floorboost_lambda"]),
            "quantile": float(receipt["d92_e0d_floorboost_quantile"]),
            "kappa": float(receipt["d92_e0d_floorboost_kappa"]),
            "zero_sum": float(receipt["d92_e0d_floorboost_old_bias_zero_sum_residual_abs"]),
            "old_mean": float(receipt["d92_e0d_floorboost_old_intercept_mean_residual_abs"]),
            "max_delta": float(receipt["d92_e0d_floorboost_max_abs_delta_over_rms"]),
            "ocf_macs": float(receipt["d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound"]),
            "retention_macs": float(receipt["d92_e0d_floorboost_support_retention_affine_macs_upper_bound"]),
            "bias_macs": float(receipt["d92_e0d_floorboost_support_bias_calibration_macs_upper_bound"]),
            "total_macs": float(receipt["d92_e0d_floorboost_support_macs_upper_bound"]),
            "transient": float(receipt["d92_e0d_floorboost_support_transient_bytes_upper_bound"]),
            "persistent_state_delta": int(
                receipt["d92_e0d_floorboost_persistent_state_bytes_delta"]
            ),
        }
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift") from error
    expected_ocf = int(
        2 * (OLD_CLASS_COUNT * int(k_shot)) * OLD_CLASS_COUNT * 288
        + 5 * OLD_CLASS_COUNT * (288 + 1)
    )
    expected_retention = int(OLD_CLASS_COUNT * int(k_shot) * int(class_count) * 288)
    expected_bias = int(6 * OLD_CLASS_COUNT)
    if (
        receipt["d92_e0d_floorboost_quantile_method"] != "lower"
        or not all(np.isfinite(value) for value in values.values())
        or values["lambda"] != float(arm.ocf_lambda)
        or values["quantile"] != float(arm.floorboost_quantile)
        or values["kappa"] != float(arm.floorboost_kappa)
        or values["zero_sum"] < 0.0
        or values["old_mean"] < 0.0
        or values["max_delta"] < 0.0
        or values["max_delta"] > values["kappa"] + 1.0e-5
        or values["ocf_macs"] != expected_ocf
        or values["retention_macs"] != expected_retention
        or values["bias_macs"] != expected_bias
        or values["total_macs"]
        != values["ocf_macs"] + values["retention_macs"] + values["bias_macs"]
        or values["transient"] < 0.0
        or receipt["d92_e0d_floorboost_new_rows_byte_exact"] is not True
        or values["persistent_state_delta"] != 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    if fallback_active is True:
        if (
            receipt["d92_e0d_floorboost_full_head_byte_exact"] is not True
            or values["max_delta"] != 0.0
            or any(
                receipt[field] is not None
                for field in (
                    "d92_e0d_floorboost_retention_score_by_old_class",
                    "d92_e0d_floorboost_registration_drift_by_old_class",
                    "d92_e0d_floorboost_delta_bias_by_old_class",
                )
            )
        ):
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
        rms_value = receipt["d92_e0d_floorboost_full_old_rms"]
        if rms_value is not None:
            try:
                numeric_rms = float(rms_value)
            except (TypeError, ValueError) as error:
                raise D92E0DQueryEvaluationError(
                    "D92-E0D floorboost receipt drift"
                ) from error
            if not np.isfinite(numeric_rms) or numeric_rms <= 0.0:
                raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    else:
        try:
            numeric_rms = float(receipt["d92_e0d_floorboost_full_old_rms"])
        except (TypeError, ValueError) as error:
            raise D92E0DQueryEvaluationError(
                "D92-E0D floorboost receipt drift"
            ) from error
        if (
            not np.isfinite(numeric_rms)
            or numeric_rms <= 0.0
            or values["max_delta"] > values["kappa"] + 1.0e-5
        ):
            raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
        for field in (
            "d92_e0d_floorboost_retention_score_by_old_class",
            "d92_e0d_floorboost_registration_drift_by_old_class",
            "d92_e0d_floorboost_delta_bias_by_old_class",
        ):
            values_by_class = np.asarray(receipt[field], dtype=np.float64)
            if (
                values_by_class.shape != (OLD_CLASS_COUNT,)
                or not np.isfinite(values_by_class).all()
            ):
                raise D92E0DQueryEvaluationError("D92-E0D floorboost receipt drift")
    return receipt


def _newguard_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    """Validate NewGuard's support-only FP32 and deployed-head closure receipt."""

    if arm.arm_id not in _NEWGUARD_ARM_IDS:
        return {}
    if any(field not in audit for field in _NEWGUARD_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D NewGuard receipt drift")
    receipt = {field: audit[field] for field in _NEWGUARD_RECEIPT_FIELDS}
    active_state = bool(registered and int(k_shot) > 2)
    active = receipt["d92_e0d_newguard_active"]
    fallback = receipt["d92_e0d_newguard_fallback_active"]
    reason = receipt["d92_e0d_newguard_fallback_reason"]
    if not active_state:
        expected_reason = (
            "NOT_REGISTERED_STATE"
            if not registered
            else "K1_K2_EXACT_D92_FULL_ALIAS"
        )
        if (
            active is not False
            or fallback is not False
            or reason != expected_reason
            or receipt["d92_e0d_newguard_deployment_strength_scale"] is not None
            or int(receipt["d92_e0d_newguard_deployment_candidate_count"]) != 0
            or receipt["d92_e0d_newguard_deployment_full_head_byte_exact"]
            is not True
        ):
            raise D92E0DQueryEvaluationError("D92-E0D NewGuard receipt drift")
        return receipt
    if int(receipt["d92_e0d_newguard_full_component_fit_count"]) != 1:
        raise D92E0DQueryEvaluationError("D92-E0D NewGuard FULL inventory drift")
    if fallback is True:
        if (
            active is not False
            or not isinstance(reason, str)
            or not reason
            or receipt["d92_e0d_newguard_full_head_byte_exact"] is not True
            or receipt["d92_e0d_newguard_deployment_strength_scale"] is not None
            or int(receipt["d92_e0d_newguard_deployment_candidate_count"])
            not in (0, 1)
            or int(receipt["d92_e0d_newguard_deployment_codec_roundtrip_count"])
            not in (0, 2)
            or receipt["d92_e0d_newguard_deployment_full_head_byte_exact"]
            is not True
        ):
            raise D92E0DQueryEvaluationError("D92-E0D NewGuard fallback receipt drift")
        return receipt
    if fallback is not False or active is not True or reason is not None:
        raise D92E0DQueryEvaluationError("D92-E0D NewGuard receipt drift")
    try:
        numeric = {
            "tau": float(receipt["d92_e0d_newguard_tau_old_envelope_shift"]),
            "rank": int(receipt["d92_e0d_newguard_nullspace_rank"]),
            "threshold": float(receipt["d92_e0d_newguard_rank_threshold"]),
            "xnew": float(
                receipt["d92_e0d_newguard_max_abs_Xnew_internal_residual"]
            ),
            "closure_tolerance": float(
                receipt["d92_e0d_newguard_closure_tolerance"]
            ),
            "old_group_zero_sum": float(
                receipt["d92_e0d_newguard_old_group_zero_sum_residual_max_abs"]
            ),
            "raw_envelope_error": float(
                receipt[
                    "d92_e0d_newguard_new_support_old_envelope_change_max_abs_error"
                ]
            ),
            "deployed_xnew": float(
                receipt[
                    "d92_e0d_newguard_deployment_max_abs_Xnew_internal_residual"
                ]
            ),
            "deployed_old_group_zero_sum": float(
                receipt[
                    "d92_e0d_newguard_deployment_old_group_zero_sum_residual_max_abs"
                ]
            ),
            "deployed_envelope_error": float(
                receipt[
                    "d92_e0d_newguard_deployment_new_support_old_envelope_change_max_abs_error"
                ]
            ),
            "new_margin": float(
                receipt["d92_e0d_newguard_new_support_min_margin_change"]
            ),
            "deployed_new_margin": float(
                receipt[
                    "d92_e0d_newguard_deployment_new_support_min_margin_change"
                ]
            ),
            "raw_envelope": float(
                receipt[
                    "d92_e0d_newguard_new_support_old_envelope_change_max"
                ]
            ),
            "deployed_envelope": float(
                receipt[
                    "d92_e0d_newguard_deployment_new_support_old_envelope_change_max"
                ]
            ),
            "protection_tolerance": float(
                receipt["d92_e0d_newguard_protection_tolerance"]
            ),
            "objective": float(receipt["d92_e0d_newguard_maxmin_objective"]),
            "trust": float(receipt["d92_e0d_newguard_trust_region_utilization"]),
            "macs": int(
                receipt["d92_e0d_newguard_support_optimization_macs_upper_bound"]
            ),
            "transient": int(
                receipt["d92_e0d_newguard_support_transient_bytes_upper_bound"]
            ),
            "state_delta": int(
                receipt["d92_e0d_newguard_persistent_state_bytes_delta"]
            ),
            "query_macs": int(receipt["d92_e0d_newguard_query_macs"]),
            "scale": float(
                receipt["d92_e0d_newguard_deployment_strength_scale"]
            ),
            "attempts": int(
                receipt["d92_e0d_newguard_deployment_candidate_count"]
            ),
            "codec_roundtrips": int(
                receipt["d92_e0d_newguard_deployment_codec_roundtrip_count"]
            ),
            "codec_macs": int(
                receipt["d92_e0d_newguard_deployment_codec_macs_upper_bound"]
            ),
        }
        tail = np.asarray(
            receipt["d92_e0d_newguard_tail_margin_change_by_old_class"],
            dtype=np.float64,
        )
        deployed_tail = np.asarray(
            receipt[
                "d92_e0d_newguard_deployment_tail_margin_change_by_old_class"
            ],
            dtype=np.float64,
        )
        residual_norm = np.asarray(
            receipt["d92_e0d_newguard_residual_l2_by_old_class"],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError("D92-E0D NewGuard receipt drift") from error
    if (
        receipt["d92_e0d_newguard_new_rows_byte_exact"] is not True
        or receipt["d92_e0d_newguard_deployment_new_rows_byte_exact"] is not True
        or receipt["d92_e0d_newguard_deployment_protection_pass"] is not True
        or receipt["d92_e0d_newguard_deployment_full_head_byte_exact"] is not False
        or not all(np.isfinite(value) for value in numeric.values())
        or numeric["tau"] > 0.0
        or numeric["rank"] <= 0
        or numeric["threshold"] <= 0.0
        or numeric["closure_tolerance"] <= 0.0
        or numeric["xnew"] < 0.0
        or numeric["xnew"] > numeric["closure_tolerance"]
        or numeric["old_group_zero_sum"] < 0.0
        or numeric["old_group_zero_sum"] > numeric["closure_tolerance"]
        or numeric["raw_envelope_error"] < 0.0
        or numeric["raw_envelope_error"] > numeric["closure_tolerance"]
        or numeric["deployed_xnew"] < 0.0
        or numeric["deployed_xnew"] > numeric["closure_tolerance"]
        or numeric["deployed_old_group_zero_sum"] < 0.0
        or numeric["deployed_old_group_zero_sum"] > numeric["closure_tolerance"]
        or numeric["deployed_envelope_error"] < 0.0
        or numeric["deployed_envelope_error"] > numeric["closure_tolerance"]
        or numeric["protection_tolerance"]
        != float(1024.0 * np.finfo(np.float32).eps)
        or numeric["new_margin"] < -numeric["protection_tolerance"]
        or numeric["deployed_new_margin"] < -numeric["protection_tolerance"]
        or numeric["raw_envelope"] > numeric["protection_tolerance"]
        or numeric["deployed_envelope"] > numeric["protection_tolerance"]
        or numeric["objective"] < 0.0
        or numeric["trust"] < 0.0
        or numeric["trust"] > 1.0 + 1.0e-6
        or numeric["macs"] < 0
        or numeric["transient"] < 0
        or numeric["state_delta"] != 0
        or numeric["query_macs"] != int(class_count) * 288
        or numeric["scale"] != 1.0
        or numeric["attempts"] != 1
        or numeric["codec_roundtrips"] != 2
        or numeric["codec_macs"] <= 0
        or tail.shape != (OLD_CLASS_COUNT,)
        or deployed_tail.shape != (OLD_CLASS_COUNT,)
        or residual_norm.shape != (OLD_CLASS_COUNT,)
        or not np.isfinite(tail).all()
        or not np.isfinite(deployed_tail).all()
        or not np.isfinite(residual_norm).all()
        or np.any(tail < -numeric["protection_tolerance"])
        or np.any(deployed_tail < -numeric["protection_tolerance"])
        or np.any(residual_norm < 0.0)
        or any(
            receipt[field] is not False
            for field in (
                "d92_e0d_newguard_query_fit_access",
                "d92_e0d_newguard_query_update_access",
                "d92_e0d_newguard_query_selection_access",
                "d92_e0d_newguard_query_truth_access",
                "d92_e0d_newguard_query_role_oracle_access",
                "d92_e0d_newguard_query_class_quota_access",
                "d92_e0d_newguard_query_global_reassignment",
            )
        )
        or int(receipt["d92_e0d_newguard_query_rows_used"]) != 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D NewGuard receipt drift")
    return receipt


def _pareto_distill_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    """Validate the shared-statistics Pareto receipt without query access."""

    if arm.arm_id not in _PARETO_DISTILL_ARM_IDS:
        return {}
    if any(field not in audit for field in _PARETO_DISTILL_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D Pareto Distill receipt drift")
    receipt = {field: audit[field] for field in _PARETO_DISTILL_RECEIPT_FIELDS}
    active_state = bool(registered and int(k_shot) > 2)
    active = receipt["d92_e0d_pareto_distill_active"]
    fallback = receipt["d92_e0d_pareto_distill_fallback_active"]
    reason = receipt["d92_e0d_pareto_distill_fallback_reason"]
    cross_group_change = receipt[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
    ]
    cross_group_quantum = receipt[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
    ]
    cross_group_quantum_pass = receipt[
        "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"
    ]
    fixed_counts = (
        int(receipt["d92_e0d_pareto_distill_full_solve_count"]),
        int(receipt["d92_e0d_pareto_distill_block_solve_count"]),
        int(receipt["d92_e0d_pareto_distill_loo_fit_count"]),
        int(receipt["d92_e0d_pareto_distill_fisher_fit_count"]),
        int(receipt["d92_e0d_pareto_distill_component_fit_count"]),
    )
    if receipt["d92_e0d_pareto_distill_mode"] != "pareto_distill":
        raise D92E0DQueryEvaluationError("D92-E0D Pareto Distill mode drift")
    if any(
        receipt[field] is not False
        for field in (
            "d92_e0d_pareto_distill_query_fit_access",
            "d92_e0d_pareto_distill_query_update_access",
            "d92_e0d_pareto_distill_query_selection_access",
            "d92_e0d_pareto_distill_query_truth_access",
            "d92_e0d_pareto_distill_query_role_oracle_access",
            "d92_e0d_pareto_distill_query_class_quota_access",
            "d92_e0d_pareto_distill_query_global_reassignment",
        )
    ) or int(receipt["d92_e0d_pareto_distill_query_rows_used"]) != 0:
        raise D92E0DQueryEvaluationError("D92-E0D Pareto Distill query closure drift")
    if int(receipt["d92_e0d_pareto_distill_query_macs"]) != int(class_count) * 288:
        raise D92E0DQueryEvaluationError("D92-E0D Pareto Distill query MAC drift")
    if not active_state:
        expected_reason = (
            "NOT_REGISTERED_STATE"
            if not registered
            else "K1_K2_EXACT_D92_FULL_ALIAS"
        )
        if (
            active is not False
            or fallback is not False
            or reason != expected_reason
            or receipt["d92_e0d_pareto_distill_local_valid"] is not False
            or fixed_counts != (0, 0, 0, 0, 0)
            or receipt["d92_e0d_pareto_distill_deployed_e0_affine_sha256"]
            is not None
            or receipt[
                "d92_e0d_pareto_distill_deployed_candidate_affine_sha256"
            ]
            is not None
            or cross_group_change is not None
            or cross_group_quantum is not None
            or cross_group_quantum_pass is not None
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D Pareto Distill inactive receipt drift"
            )
        return receipt
    if (
        fixed_counts != (1, 1, 0, 0, 2)
        or int(receipt["d92_e0d_pareto_distill_covariance_estimation_count"])
        != 1
        or int(receipt["d92_e0d_pareto_distill_robust_center_transform_count"])
        != 1
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill shared-count receipt drift"
        )
    if fallback is True:
        if (
            active is not False
            or receipt["d92_e0d_pareto_distill_local_valid"] is not False
            or not isinstance(reason, str)
            or not reason
            or receipt["d92_e0d_pareto_distill_full_head_byte_exact"] is not True
            or receipt[
                "d92_e0d_pareto_distill_deployed_support_constraints_pass"
            ]
            is not False
            or receipt[
                "d92_e0d_pareto_distill_deployed_full_head_byte_exact"
            ]
            is not True
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D Pareto Distill fallback receipt drift"
            )
        if cross_group_quantum_pass is not False:
            raise D92E0DQueryEvaluationError(
                "D92-E0D Pareto Distill fallback quantum receipt drift"
            )
        for value, positive in (
            (cross_group_change, False),
            (cross_group_quantum, True),
        ):
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise D92E0DQueryEvaluationError(
                        "D92-E0D Pareto Distill fallback quantum receipt drift"
                    ) from error
                if not np.isfinite(numeric) or numeric < 0.0 or (
                    positive and numeric <= 0.0
                ):
                    raise D92E0DQueryEvaluationError(
                        "D92-E0D Pareto Distill fallback quantum receipt drift"
                    )
        return receipt
    if (
        fallback is not False
        or active is not True
        or reason is not None
        or receipt["d92_e0d_pareto_distill_local_valid"] is not True
        or receipt["d92_e0d_pareto_distill_full_head_byte_exact"] is not False
        or receipt[
            "d92_e0d_pareto_distill_deployed_support_constraints_pass"
        ]
        is not True
        or receipt[
            "d92_e0d_pareto_distill_deployed_full_head_byte_exact"
        ]
        is not False
    ):
        raise D92E0DQueryEvaluationError("D92-E0D Pareto Distill active receipt drift")
    try:
        quantum = float(cross_group_quantum)
        change = float(cross_group_change)
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill active quantum receipt drift"
        ) from error
    if (
        cross_group_quantum_pass is not True
        or not np.isfinite(quantum)
        or not np.isfinite(change)
        or quantum <= 0.0
        or change < quantum
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill active quantum receipt drift"
        )
    return receipt


def _pareto_deployed_state_closure(
    state: Any,
    receipt: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
) -> dict[str, Any]:
    """Close the previewed D42 affine head against the final deployed state."""

    if arm.arm_id not in _PARETO_DISTILL_ARM_IDS:
        return {}
    active = receipt["d92_e0d_pareto_distill_active"] is True
    fallback = receipt["d92_e0d_pareto_distill_fallback_active"] is True
    if not active and not fallback:
        return {
            "d92_e0d_pareto_distill_deployed_head_state_closure_pass": None,
            "d92_e0d_pareto_distill_deployed_head_state_affine_sha256": None,
        }
    expected_field = (
        "d92_e0d_pareto_distill_deployed_candidate_affine_sha256"
        if active
        else "d92_e0d_pareto_distill_deployed_e0_affine_sha256"
    )
    expected = receipt.get(expected_field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill deployed preview hash drift"
        )
    try:
        from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

        coefficient = d42.decode_d42_coefficients(state)
        intercept = np.asarray(state.intercept_fp16, dtype=np.float32)
    except (AttributeError, TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill deployed state decode drift"
        ) from error
    if (
        coefficient.ndim != 2
        or intercept.shape != (coefficient.shape[0],)
        or not np.isfinite(coefficient).all()
        or not np.isfinite(intercept).all()
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill deployed state decode drift"
        )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coefficient, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(intercept, dtype=np.float32).tobytes())
    actual = digest.hexdigest()
    if actual != expected:
        raise D92E0DQueryEvaluationError(
            "D92-E0D Pareto Distill deployed preview/state mismatch"
        )
    return {
        "d92_e0d_pareto_distill_deployed_head_state_closure_pass": True,
        "d92_e0d_pareto_distill_deployed_head_state_affine_sha256": actual,
    }


def _csoas_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    """Validate frozen CSOAS lifecycle and independently checkable statistics."""

    if arm.arm_id not in _CSOAS_ARM_IDS:
        return {}
    lifecycle = (
        "d92_csoas_active",
        "d92_csoas_fallback_active",
        "d92_csoas_fallback_reason",
        "d92_csoas_candidate_attempt_fit_count",
        "d92_csoas_fallback_reference_fit_count",
        "d92_csoas_candidate_statistic_receipt_available",
        "d92_csoas_fallback_reference_full_head_byte_exact",
        "d92_csoas_paired_e0_codec_state_equal",
        "d92_csoas_query_rows_used",
        "d92_csoas_query_fit_access",
        "d92_csoas_query_update_access",
        "d92_csoas_query_selection_access",
        "d92_csoas_query_truth_access",
        "d92_csoas_query_role_oracle_access",
        "d92_csoas_query_class_quota_access",
        "d92_csoas_query_global_reassignment",
        "d92_e0d_csoas_g0_eligible",
        "d92_e0d_csoas_g0_block_reason",
    )
    if any(field not in audit for field in lifecycle):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS receipt missing")
    receipt = {field: audit[field] for field in lifecycle}
    if (
        receipt["d92_csoas_paired_e0_codec_state_equal"] is not None
        or int(receipt["d92_csoas_query_rows_used"]) != 0
        or any(receipt[field] is not False for field in lifecycle[9:16])
        or receipt["d92_e0d_csoas_g0_eligible"] is not False
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS query receipt drift")
    enabled = bool(registered and int(k_shot) > 2)
    if not enabled:
        reason = (
            "NOT_REGISTERED_STATE"
            if not registered
            else "K1_K2_EXACT_D92_FULL_ALIAS"
        )
        if (
            receipt["d92_csoas_active"] is not False
            or receipt["d92_csoas_fallback_active"] is not False
            or receipt["d92_csoas_fallback_reason"] != reason
            or int(receipt["d92_csoas_candidate_attempt_fit_count"]) != 0
            or int(receipt["d92_csoas_fallback_reference_fit_count"]) != 0
            or receipt["d92_csoas_candidate_statistic_receipt_available"] is not False
            or receipt["d92_csoas_fallback_reference_full_head_byte_exact"]
            is not None
            or receipt["d92_e0d_csoas_g0_block_reason"] != reason
        ):
            raise D92E0DQueryEvaluationError("D92-E0D CSOAS alias receipt drift")
        return receipt
    if int(receipt["d92_csoas_candidate_attempt_fit_count"]) != 1:
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS candidate count drift")
    if receipt["d92_csoas_fallback_active"] is True:
        if (
            receipt["d92_csoas_active"] is not False
            or not isinstance(receipt["d92_csoas_fallback_reason"], str)
            or not receipt["d92_csoas_fallback_reason"]
            or int(receipt["d92_csoas_fallback_reference_fit_count"]) != 1
            or receipt["d92_csoas_fallback_reference_full_head_byte_exact"]
            is not True
            or receipt["d92_e0d_csoas_g0_block_reason"]
            != "NUMERIC_FALLBACK_EXACT_E0"
        ):
            raise D92E0DQueryEvaluationError("D92-E0D CSOAS fallback receipt drift")
        return receipt
    if (
        receipt["d92_csoas_active"] is not True
        or receipt["d92_csoas_fallback_active"] is not False
        or receipt["d92_csoas_fallback_reason"] is not None
        or int(receipt["d92_csoas_fallback_reference_fit_count"]) != 0
        or receipt["d92_csoas_candidate_statistic_receipt_available"] is not True
        or receipt["d92_csoas_fallback_reference_full_head_byte_exact"] is not None
        or receipt["d92_e0d_csoas_g0_block_reason"]
        != "PENDING_DEPLOYED_CODEC_PAIRED_E0"
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS active receipt drift")
    statistics = (
        "d92_csoas_support_rows",
        "d92_csoas_class_count",
        "d92_csoas_k_shot",
        "d92_csoas_old_class_count",
        "d92_csoas_new_class_count",
        "d92_csoas_normalized_cauchy_weight_by_class",
        "d92_csoas_weighted_center_by_class",
        "d92_csoas_effective_sample_size_by_class",
        "d92_csoas_scatter_trace_by_class",
        "d92_csoas_oas_tau_by_class",
        "d92_csoas_oas_alpha_by_class",
        "d92_csoas_oas_denominator_by_class",
        "d92_csoas_oas_rho_by_class",
        "d92_csoas_shrunk_trace_by_class",
        "d92_csoas_old_group_trace",
        "d92_csoas_new_group_trace",
        "d92_csoas_final_trace",
        "d92_csoas_final_eigenvalue_min",
        "d92_csoas_final_eigenvalue_max",
        "d92_csoas_spd_pass",
        "d92_csoas_live_class_scatter_buffers",
        "d92_csoas_class_matrix_stack",
        "d92_csoas_single_scatter_buffer_bytes",
        "d92_csoas_final_covariance_buffer_bytes",
        "d92_csoas_support_scatter_macs_upper_bound",
        "d92_csoas_support_transient_bytes_upper_bound",
    )
    if any(field not in audit for field in statistics):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS statistic receipt missing")
    receipt.update({field: audit[field] for field in statistics})
    try:
        weights = np.asarray(receipt[statistics[5]], dtype=np.float64)
        centers = np.asarray(receipt[statistics[6]], dtype=np.float64)
        effective = np.asarray(receipt[statistics[7]], dtype=np.float64)
        traces = np.asarray(receipt[statistics[8]], dtype=np.float64)
        tau = np.asarray(receipt[statistics[9]], dtype=np.float64)
        alpha = np.asarray(receipt[statistics[10]], dtype=np.float64)
        denominator = np.asarray(receipt[statistics[11]], dtype=np.float64)
        rho = np.asarray(receipt[statistics[12]], dtype=np.float64)
        shrunk = np.asarray(receipt[statistics[13]], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS statistic receipt drift") from error
    vectors = (effective, traces, tau, alpha, denominator, rho, shrunk)
    expected_tau = traces / 288.0
    expected_denominator = (effective + 1.0) * (
        alpha - expected_tau * expected_tau / 288.0
    )
    expected_rho = np.ones_like(expected_denominator)
    valid_denominator = expected_denominator > 0.0
    expected_rho[valid_denominator] = np.clip(
        (alpha[valid_denominator] + expected_tau[valid_denominator] ** 2)
        / expected_denominator[valid_denominator],
        0.0,
        1.0,
    )
    if (
        int(receipt[statistics[0]]) != int(class_count) * int(k_shot)
        or int(receipt[statistics[1]]) != int(class_count)
        or int(receipt[statistics[2]]) != int(k_shot)
        or int(receipt[statistics[3]]) != int(OLD_CLASS_COUNT)
        or int(receipt[statistics[4]]) != int(class_count) - int(OLD_CLASS_COUNT)
        or weights.shape != (int(class_count), int(k_shot))
        or centers.shape != (int(class_count), 288)
        or any(value.shape != (int(class_count),) for value in vectors)
        or not all(np.isfinite(value).all() for value in (weights, centers, *vectors))
        or np.any(weights <= 0.0)
        or not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
        or not np.allclose(
            effective, 1.0 / np.sum(weights * weights, axis=1), rtol=1.0e-10, atol=1.0e-12
        )
        or np.any(rho < 0.0)
        or np.any(rho > 1.0)
        or not np.allclose(tau, expected_tau, rtol=1.0e-10, atol=1.0e-12)
        or not np.allclose(
            denominator, expected_denominator, rtol=1.0e-10, atol=1.0e-12
        )
        or not np.allclose(rho, expected_rho, rtol=1.0e-10, atol=1.0e-12)
        or not np.allclose(traces, shrunk, rtol=1.0e-10, atol=1.0e-12)
        or receipt["d92_csoas_spd_pass"] is not True
        or float(receipt["d92_csoas_final_eigenvalue_min"]) <= 0.0
        or float(receipt["d92_csoas_final_eigenvalue_max"])
        < float(receipt["d92_csoas_final_eigenvalue_min"])
        or int(receipt["d92_csoas_live_class_scatter_buffers"]) != 1
        or receipt["d92_csoas_class_matrix_stack"] is not False
        or any(int(receipt[field]) <= 0 for field in statistics[-4:])
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS statistic receipt drift")
    expected_trace = 0.5 * (
        float(receipt["d92_csoas_old_group_trace"])
        + float(receipt["d92_csoas_new_group_trace"])
    )
    if not np.isfinite(expected_trace) or not np.isclose(
        float(receipt["d92_csoas_final_trace"]), expected_trace, rtol=1.0e-10, atol=1.0e-12
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CSOAS trace receipt drift")
    return receipt


def _validate_ccoc_covariance_mix_workspace_receipt(
    raw: Mapping[str, Any],
) -> None:
    """Verify the frozen live CCOC covariance-mix allocation accounting."""

    for field, expected in _CCOC_COVARIANCE_MIX_WORKSPACE_EXPECTED.items():
        value = raw.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            or int(value) != expected
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D CCOC covariance mix workspace receipt drift"
            )


def _ccoc_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    """Close the CCOC support receipt without consulting query-side data."""

    if arm.arm_id not in _CCOC_ARM_IDS:
        return {}
    raw = {
        key: value
        for key, value in audit.items()
        if key.startswith("d92_ccoc_")
    }
    mirrored = {
        key: value
        for key, value in audit.items()
        if key.startswith("d92_e0d_ccoc_")
    }
    enabled = bool(registered and int(k_shot) > 2)
    raw_fields = frozenset(raw)
    if not enabled:
        expected_raw_fields = _CCOC_RAW_INACTIVE_FIELDS
    elif (
        raw.get("d92_ccoc_active") is True
        and raw.get("d92_ccoc_fallback_active") is False
    ):
        expected_raw_fields = _CCOC_RAW_ACTIVE_FIELDS
    elif raw.get("d92_ccoc_fallback_active") is True:
        if (
            raw.get("d92_ccoc_codec_fallback_scope")
            == "whole_d42_retry_before_and_after"
        ):
            expected_raw_fields = _CCOC_RAW_CODEC_FALLBACK_FIELDS
        elif raw.get("d92_ccoc_candidate_statistic_receipt_available") is True:
            expected_raw_fields = _CCOC_RAW_NUMERIC_FALLBACK_WITH_STATISTICS_FIELDS
        else:
            expected_raw_fields = _CCOC_RAW_NUMERIC_FALLBACK_WITHOUT_STATISTICS_FIELDS
    else:
        raise D92E0DQueryEvaluationError("D92-E0D CCOC lifecycle field drift")
    expected_mirror_fields = frozenset(
        field.replace("d92_ccoc_", "d92_e0d_ccoc_")
        for field in expected_raw_fields
    ) | _CCOC_MIRROR_G0_FIELDS
    if not enabled:
        expected_mirror_fields |= _CCOC_MIRROR_INACTIVE_ONLY_FIELDS
    if (
        raw_fields != expected_raw_fields
        or frozenset(mirrored) != expected_mirror_fields
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D CCOC receipt field collection drift"
        )
    for raw_field in expected_raw_fields:
        mirror_field = raw_field.replace("d92_ccoc_", "d92_e0d_ccoc_")
        if (
            (enabled or raw_field != "d92_ccoc_fallback_reason")
            and mirrored[mirror_field] != raw[raw_field]
        ):
            raise D92E0DQueryEvaluationError("D92-E0D CCOC raw/mirror drift")
    frozen_workspace_field = (
        "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound"
    )
    if frozen_workspace_field in raw:
        try:
            frozen_workspace = int(raw[frozen_workspace_field])
        except (TypeError, ValueError) as error:
            raise D92E0DQueryEvaluationError(
                "D92-E0D CCOC frozen workspace receipt drift"
            ) from error
        if frozen_workspace != 334_336:
            raise D92E0DQueryEvaluationError(
                "D92-E0D CCOC frozen workspace receipt drift"
            )
    if not enabled:
        raw_reason = (
            "before_exact_d81"
            if not registered
            else "k1_k2_exact_d81_fallback"
        )
        mirror_reason = (
            "NOT_REGISTERED_STATE"
            if not registered
            else "K1_K2_EXACT_D92_FULL_ALIAS"
        )
        if (
            raw.get("d92_ccoc_active") is not False
            or raw.get("d92_ccoc_fallback_active") is not False
            or raw.get("d92_ccoc_fallback_reason") != raw_reason
            or raw.get("d92_ccoc_old_rho") is not None
            or raw.get("d92_ccoc_new_rho") is not None
            or mirrored["d92_e0d_ccoc_active"] is not False
            or mirrored["d92_e0d_ccoc_fallback_active"] is not False
            or mirrored["d92_e0d_ccoc_fallback_reason"] != mirror_reason
            or int(mirrored["d92_e0d_ccoc_candidate_attempt_fit_count"]) != 0
            or int(mirrored["d92_e0d_ccoc_fallback_reference_fit_count"])
            != 0
            or mirrored["d92_e0d_ccoc_candidate_statistic_receipt_available"]
            is not False
            or mirrored["d92_e0d_ccoc_fallback_reference_full_head_byte_exact"]
            is not None
            or mirrored["d92_e0d_ccoc_paired_e0_codec_state_equal"] is not None
            or mirrored["d92_e0d_ccoc_g0_eligible"] is not False
            or mirrored["d92_e0d_ccoc_g0_block_reason"] != mirror_reason
        ):
            raise D92E0DQueryEvaluationError("D92-E0D CCOC alias receipt drift")
        return {field: mirrored[field] for field in sorted(expected_mirror_fields)}

    lifecycle = (
        "d92_ccoc_active",
        "d92_ccoc_fallback_active",
        "d92_ccoc_fallback_reason",
        "d92_ccoc_candidate_attempt_fit_count",
        "d92_ccoc_fallback_reference_fit_count",
        "d92_ccoc_candidate_statistic_receipt_available",
        "d92_ccoc_fallback_reference_full_head_byte_exact",
        "d92_ccoc_paired_e0_codec_state_equal",
        "d92_ccoc_query_rows_used",
        "d92_ccoc_query_fit_access",
        "d92_ccoc_query_update_access",
        "d92_ccoc_query_selection_access",
        "d92_ccoc_query_truth_access",
        "d92_ccoc_query_role_oracle_access",
        "d92_ccoc_query_class_quota_access",
        "d92_ccoc_query_global_reassignment",
    )
    if any(field not in raw for field in lifecycle):
        raise D92E0DQueryEvaluationError("D92-E0D CCOC lifecycle missing")
    if (
        int(raw["d92_ccoc_candidate_attempt_fit_count"]) != 1
        or raw["d92_ccoc_paired_e0_codec_state_equal"] is not None
        or int(raw["d92_ccoc_query_rows_used"]) != 0
        or any(raw[field] is not False for field in lifecycle[9:])
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CCOC support/query drift")
    fallback = raw["d92_ccoc_fallback_active"]
    active = raw["d92_ccoc_active"]
    if fallback is True:
        if raw["d92_ccoc_candidate_statistic_receipt_available"] is True:
            _validate_ccoc_covariance_mix_workspace_receipt(raw)
        if (
            active is not False
            or not isinstance(raw["d92_ccoc_fallback_reason"], str)
            or not raw["d92_ccoc_fallback_reason"]
            or int(raw["d92_ccoc_fallback_reference_fit_count"]) != 1
            or raw["d92_ccoc_fallback_reference_full_head_byte_exact"] is not True
            or mirrored["d92_e0d_ccoc_g0_eligible"] is not False
            or mirrored["d92_e0d_ccoc_g0_block_reason"]
            != "NUMERIC_FALLBACK_EXACT_E0"
        ):
            raise D92E0DQueryEvaluationError("D92-E0D CCOC fallback receipt drift")
        return {field: mirrored[field] for field in sorted(expected_mirror_fields)}
    if (
        fallback is not False
        or active is not True
        or raw["d92_ccoc_fallback_reason"] is not None
        or int(raw["d92_ccoc_fallback_reference_fit_count"]) != 0
        or raw["d92_ccoc_candidate_statistic_receipt_available"] is not True
        or raw["d92_ccoc_fallback_reference_full_head_byte_exact"] is not None
        or mirrored["d92_e0d_ccoc_g0_eligible"] is not True
        or mirrored["d92_e0d_ccoc_g0_block_reason"] is not None
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CCOC active receipt drift")
    _validate_ccoc_covariance_mix_workspace_receipt(raw)
    statistics = (
        "d92_ccoc_formula_revision",
        "d92_ccoc_formula",
        "d92_ccoc_old_rho",
        "d92_ccoc_new_rho",
        "d92_ccoc_old_group_class_count",
        "d92_ccoc_new_group_class_count",
        "d92_ccoc_old_offblock_norm_min",
        "d92_ccoc_old_offblock_norm_max",
        "d92_ccoc_new_offblock_norm_min",
        "d92_ccoc_new_offblock_norm_max",
        "d92_ccoc_old_pairwise_cosine_raw",
        "d92_ccoc_new_pairwise_cosine_raw",
        "d92_ccoc_canonicalization",
        "d92_ccoc_canonicalization_tie_policy",
        "d92_ccoc_crossblock_passes_per_class",
        "d92_ccoc_upper_block_count",
        "d92_ccoc_covariance_symmetric",
        "d92_ccoc_full_endpoint_reused",
        "d92_ccoc_full_endpoint_reuse",
        "d92_ccoc_additional_fit_count",
        "d92_ccoc_additional_full_fit_count",
        "d92_ccoc_additional_block_fit_count",
        "d92_ccoc_additional_loo_fit_count",
        "d92_ccoc_additional_fisher_fit_count",
        "d92_ccoc_additional_scan_count",
        "d92_ccoc_block_fit_count",
        "d92_ccoc_loo_fit_count",
        "d92_ccoc_fisher_fit_count",
        "d92_ccoc_scan_count",
        "d92_ccoc_hyperparameter_scan_count",
        "d92_ccoc_weight_scan_count",
        "d92_ccoc_dense_solve_count",
        "d92_ccoc_compile_solve_count",
        "d92_ccoc_full_solve_count",
        "d92_ccoc_full_dense_288_solve_count",
        "d92_ccoc_cholesky_check_count",
        "d92_ccoc_cholesky_endpoint_check_count",
        "d92_ccoc_cholesky_final_check_count",
        "d92_ccoc_cholesky_pass",
        "d92_ccoc_old_endpoint_cholesky_min_diagonal",
        "d92_ccoc_new_endpoint_cholesky_min_diagonal",
        "d92_ccoc_final_cholesky_min_diagonal",
        "d92_ccoc_support_macs_upper_bound",
        "d92_ccoc_workspace_upper_accumulators_bytes",
        "d92_ccoc_workspace_cross_block_buffer_bytes",
        "d92_ccoc_workspace_residual_buffer_bytes",
        "d92_ccoc_workspace_numeric_bytes_upper_bound",
        "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound",
        "d92_ccoc_support_transient_bytes_upper_bound",
        "d92_ccoc_persistent_state_bytes_delta",
        "d92_ccoc_persistent_bytes_delta",
        "d92_ccoc_query_state_bytes_delta",
        "d92_ccoc_query_bytes_delta",
        "d92_ccoc_query_macs_delta",
        "d92_ccoc_query_macs",
    )
    if any(field not in raw for field in statistics):
        raise D92E0DQueryEvaluationError("D92-E0D CCOC statistic receipt missing")
    try:
        rho = np.asarray(
            [raw["d92_ccoc_old_rho"], raw["d92_ccoc_new_rho"]],
            dtype=np.float64,
        )
        norms = np.asarray(
            [
                raw["d92_ccoc_old_offblock_norm_min"],
                raw["d92_ccoc_old_offblock_norm_max"],
                raw["d92_ccoc_new_offblock_norm_min"],
                raw["d92_ccoc_new_offblock_norm_max"],
            ],
            dtype=np.float64,
        )
        cholesky = np.asarray(
            [
                raw["d92_ccoc_old_endpoint_cholesky_min_diagonal"],
                raw["d92_ccoc_new_endpoint_cholesky_min_diagonal"],
                raw["d92_ccoc_final_cholesky_min_diagonal"],
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise D92E0DQueryEvaluationError("D92-E0D CCOC numeric receipt drift") from error
    zero_counts = (
        "d92_ccoc_additional_fit_count",
        "d92_ccoc_additional_full_fit_count",
        "d92_ccoc_additional_block_fit_count",
        "d92_ccoc_additional_loo_fit_count",
        "d92_ccoc_additional_fisher_fit_count",
        "d92_ccoc_additional_scan_count",
        "d92_ccoc_block_fit_count",
        "d92_ccoc_loo_fit_count",
        "d92_ccoc_fisher_fit_count",
        "d92_ccoc_scan_count",
        "d92_ccoc_hyperparameter_scan_count",
        "d92_ccoc_weight_scan_count",
        "d92_ccoc_persistent_state_bytes_delta",
        "d92_ccoc_persistent_bytes_delta",
        "d92_ccoc_query_state_bytes_delta",
        "d92_ccoc_query_bytes_delta",
        "d92_ccoc_query_macs_delta",
        "d92_ccoc_query_macs",
    )
    if (
        raw["d92_ccoc_formula_revision"] != "pairwise_cosine_v1"
        or raw["d92_ccoc_canonicalization"]
        != "lexicographic_float32_row_bytes_then_float64_reduce"
        or raw["d92_ccoc_canonicalization_tie_policy"]
        != "float32_row_bytes_then_float64_row_bytes_duplicate_class_handle_fail_closed"
        or int(raw["d92_ccoc_old_group_class_count"]) != int(OLD_CLASS_COUNT)
        or int(raw["d92_ccoc_new_group_class_count"])
        != int(class_count) - int(OLD_CLASS_COUNT)
        or not np.isfinite(rho).all()
        or np.any(rho < 0.0)
        or np.any(rho > 1.0)
        or not np.isfinite(norms).all()
        or np.any(norms <= 0.0)
        or norms[0] > norms[1]
        or norms[2] > norms[3]
        or not np.isfinite(cholesky).all()
        or np.any(cholesky <= 0.0)
        or int(raw["d92_ccoc_crossblock_passes_per_class"]) != 2
        or int(raw["d92_ccoc_upper_block_count"]) != 3
        or raw["d92_ccoc_covariance_symmetric"] is not True
        or raw["d92_ccoc_full_endpoint_reused"] is not True
        or raw["d92_ccoc_full_endpoint_reuse"] is not True
        or any(int(raw[field]) != 0 for field in zero_counts)
        or int(raw["d92_ccoc_dense_solve_count"]) != 1
        or int(raw["d92_ccoc_compile_solve_count"]) != 1
        or int(raw["d92_ccoc_full_solve_count"]) != 1
        or int(raw["d92_ccoc_full_dense_288_solve_count"]) != 1
        or int(raw["d92_ccoc_cholesky_check_count"]) != 3
        or int(raw["d92_ccoc_cholesky_endpoint_check_count"]) != 2
        or int(raw["d92_ccoc_cholesky_final_check_count"]) != 1
        or raw["d92_ccoc_cholesky_pass"] is not True
        or int(raw["d92_ccoc_support_macs_upper_bound"]) <= 0
        or int(raw["d92_ccoc_workspace_numeric_bytes_upper_bound"]) <= 0
        or int(raw["d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound"])
        != 334_336
        or int(raw["d92_ccoc_support_transient_bytes_upper_bound"])
        != int(raw["d92_ccoc_workspace_numeric_bytes_upper_bound"])
    ):
        raise D92E0DQueryEvaluationError("D92-E0D CCOC statistic receipt drift")
    return {field: mirrored[field] for field in sorted(expected_mirror_fields)}


def _tpce_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    state: Any,
) -> dict[str, Any]:
    """Validate the direct D42 code-state postprocessor receipt."""

    if arm.arm_id not in _TPCE_ARM_IDS:
        return {}
    if any(field not in audit for field in _TPCE_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D TPCE receipt missing")
    receipt = {field: audit[field] for field in _TPCE_RECEIPT_FIELDS}
    prefix = "d92_e0d_tpce_"

    def finite(name: str, *, lower: float | None = None) -> float:
        value = receipt[prefix + name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or (lower is not None and float(value) < lower)
        ):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D TPCE {name} receipt drift"
            )
        return float(value)

    def sha(name: str) -> str:
        value = receipt[prefix + name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D TPCE {name} receipt drift"
            )
        return value

    for name in (
        "code1_byte_exact",
        "scale1_byte_exact",
        "scale2_byte_exact",
        "intercept_byte_exact",
        "log_diag_byte_exact",
    ):
        if receipt[prefix + name] is not True:
            raise D92E0DQueryEvaluationError("D92-E0D TPCE state guard drift")
    if (
        receipt[prefix + "state_postprocess_mode"] != "d42_tpce"
        or receipt[prefix + "direct_state_publish"] is not True
        or receipt[prefix + "requantize_call_count"] != 0
        or receipt[prefix + "quantile"] != 0.20
        or receipt[prefix + "quantile_method"] != "lower"
        or receipt[prefix + "class_permutation_equivariant"] is not True
        or receipt[prefix + "old_group_uniform_shift"] is not False
        or receipt[prefix + "persistent_state_bytes_delta"] != 0
        or receipt[prefix + "component_fit_count"] != 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TPCE frozen receipt drift")
    from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as tpce

    final_sha = sha("final_state_sha256")
    e0_sha = sha("e0_state_sha256")
    if tpce.d42_tpce_state_sha256(state) != final_sha:
        raise D92E0DQueryEvaluationError("D92-E0D TPCE deployed state SHA drift")

    active_state = bool(registered and int(k_shot) > 2)
    active = receipt[prefix + "active"]
    fallback = receipt[prefix + "fallback_active"]
    reason = receipt[prefix + "fallback_reason"]
    if not active_state:
        if (
            active is not False
            or fallback is not False
            or reason != "K1_K2_EXACT_D92_FULL_ALIAS"
            or final_sha != e0_sha
            or any(
                receipt[prefix + name] != 0
                for name in (
                    "changed_code2_count",
                    "requested_atomic_exchange_count",
                    "applied_atomic_exchange_count",
                    "aggregate_saturation_count",
                    "generated_atomic_exchange_count",
                    "selected_atomic_exchange_count",
                    "rejected_atomic_exchange_count",
                    "greedy_step_count",
                    "support_score_macs_upper_bound",
                    "support_coordinate_comparisons_upper_bound",
                    "support_macs_upper_bound",
                    "support_transient_bytes_upper_bound",
                )
            )
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TPCE alias receipt drift")
        return receipt

    for name in (
        "requested_atomic_exchange_count",
        "aggregate_saturation_count",
        "generated_atomic_exchange_count",
        "selected_atomic_exchange_count",
        "rejected_atomic_exchange_count",
        "greedy_step_count",
        "support_score_macs_upper_bound",
        "support_coordinate_comparisons_upper_bound",
        "support_macs_upper_bound",
        "support_transient_bytes_upper_bound",
    ):
        finite(name, lower=0.0)
    if fallback is True:
        generated = int(finite("generated_atomic_exchange_count", lower=0.0))
        selected = int(finite("selected_atomic_exchange_count", lower=0.0))
        rejected = int(finite("rejected_atomic_exchange_count", lower=0.0))
        greedy_steps = int(finite("greedy_step_count", lower=0.0))
        if (
            active is not False
            or not isinstance(reason, str)
            or not reason
            or final_sha != e0_sha
            or receipt[prefix + "changed_code2_count"] != 0
            or receipt[prefix + "applied_atomic_exchange_count"] != 0
            or receipt[prefix + "support_guard_pass"] is not False
            or generated != selected + rejected
            or greedy_steps != selected
            or (
                reason == "aggregate_saturation"
                and finite("aggregate_saturation_count", lower=0.0) <= 0.0
            )
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TPCE fallback receipt drift")
        return receipt
    generated = int(finite("generated_atomic_exchange_count", lower=1.0))
    selected = int(finite("selected_atomic_exchange_count", lower=1.0))
    rejected = int(finite("rejected_atomic_exchange_count", lower=0.0))
    greedy_steps = int(finite("greedy_step_count", lower=1.0))
    requested = int(finite("requested_atomic_exchange_count", lower=1.0))
    applied = int(finite("applied_atomic_exchange_count", lower=1.0))
    if (
        requested != applied
        or generated != selected + rejected
        or selected != greedy_steps
        or selected != requested
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D TPCE active atomic receipt drift"
        )
    if (
        active is not True
        or fallback is not False
        or reason is not None
        or final_sha == e0_sha
        or receipt[prefix + "support_guard_pass"] is not True
        or finite("changed_code2_count", lower=1.0) < 1.0
        or finite("aggregate_saturation_count", lower=0.0) != 0.0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TPCE active receipt drift")
    old_counts = receipt[prefix + "old_tail_count_by_class"]
    old_gains = receipt[prefix + "old_tail_gain_by_class"]
    if (
        not isinstance(old_counts, list)
        or len(old_counts) != OLD_CLASS_COUNT
        or any(int(value) <= 0 for value in old_counts)
        or not isinstance(old_gains, list)
        or len(old_gains) != OLD_CLASS_COUNT
        or int(finite("pooled_new_tail_count", lower=1.0)) <= 0
        or int(finite("tied_competitor_relation_count", lower=1.0)) <= 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TPCE fixed-tail receipt drift")
    tolerance = finite("guard_tolerance", lower=0.0)
    if (
        any(
            not np.isfinite(float(value)) or float(value) <= tolerance
            for value in old_gains
        )
        or finite("old_tail_min_gain") <= tolerance
        or finite("pooled_new_cross_tail_gain") <= tolerance
        or finite("pooled_new_allclass_tail_gain") < -tolerance
        or finite("old_to_new_hinge_delta") > tolerance
        or finite("new_to_old_hinge_delta") > tolerance
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TPCE support guard drift")
    return receipt


def _tcra_support_receipt(
    audit: dict[str, Any],
    *,
    arm: D92E0DSlimArmSpec,
    registered: bool,
    k_shot: int,
    state: Any,
) -> dict[str, Any]:
    """Validate the frozen true-class-only D42 postprocessor receipt."""

    if arm.arm_id not in _TCRA_ARM_IDS:
        return {}
    if any(field not in audit for field in _TCRA_RECEIPT_FIELDS):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA receipt missing")
    receipt = {field: audit[field] for field in _TCRA_RECEIPT_FIELDS}
    prefix = "d92_e0d_tcra_"

    def finite(name: str, *, lower: float | None = None) -> float:
        value = receipt[prefix + name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or (lower is not None and float(value) < lower)
        ):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D TCRA {name} receipt drift"
            )
        return float(value)

    def integer(name: str, *, lower: int = 0) -> int:
        value = finite(name, lower=float(lower))
        if value != float(int(value)):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D TCRA {name} receipt drift"
            )
        return int(value)

    def sha(name: str) -> str:
        value = receipt[prefix + name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D TCRA {name} receipt drift"
            )
        return value

    for name in (
        "code1_byte_exact",
        "scale1_byte_exact",
        "scale2_byte_exact",
        "intercept_byte_exact",
        "log_diag_byte_exact",
        "true_class_row_only",
        "class_permutation_equivariant",
        "row_permutation_invariant",
    ):
        if receipt[prefix + name] is not True:
            raise D92E0DQueryEvaluationError("D92-E0D TCRA state guard drift")
    if (
        receipt[prefix + "state_postprocess_mode"] != "d42_tcra"
        or receipt[prefix + "final_gate_revision"] != "safe_directional_v2"
        or receipt[prefix + "direct_state_publish"] is not True
        or receipt[prefix + "requantize_call_count"] != 0
        or receipt[prefix + "quantile"] != 0.20
        or receipt[prefix + "quantile_method"] != "lower"
        or receipt[prefix + "competitor_code_decrement_count"] != 0
        or receipt[prefix + "old_group_uniform_shift"] is not False
        or receipt[prefix + "persistent_state_bytes_delta"] != 0
        or receipt[prefix + "component_fit_count"] != 0
        or receipt[prefix + "query_rows_used"] != 0
        or receipt[prefix + "query_macs"] != 0
        or any(
            receipt[prefix + f"query_{name}"] is not False
            for name in (
                "fit_access",
                "update_access",
                "selection_access",
                "truth_access",
                "role_oracle_access",
                "class_quota_access",
                "global_reassignment",
            )
        )
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA frozen receipt drift")

    from cvsrffi import stage2_d92_d42_tail_class_row_ascent as tcra

    final_sha = sha("final_state_sha256")
    e0_sha = sha("e0_state_sha256")
    if tcra.d42_tcra_state_sha256(state) != final_sha:
        raise D92E0DQueryEvaluationError("D92-E0D TCRA deployed state SHA drift")
    active_state = bool(registered and int(k_shot) > 2)
    active = receipt[prefix + "active"]
    fallback = receipt[prefix + "fallback_active"]
    reason = receipt[prefix + "fallback_reason"]
    generated = integer("generated_atomic_ascent_count")
    selected = integer("selected_atomic_ascent_count")
    rejected = integer("rejected_atomic_ascent_count")
    prefix_rejected = integer("prefix_guard_rejected_count")
    steps = integer("greedy_step_count")
    requested = integer("requested_atomic_ascent_count")
    applied = integer("applied_atomic_ascent_count")
    changed = integer("changed_code2_count")
    state_l1 = integer("state_delta_code2_l1")
    saturation = integer("aggregate_saturation_count")
    full_scores = integer("support_full_score_evaluation_count")
    analytic = integer("support_analytic_candidate_evaluation_count")
    score_macs = integer("support_score_macs_upper_bound")
    integer("support_coordinate_comparisons_upper_bound")
    support_macs = integer("support_macs_upper_bound")
    integer("support_transient_bytes_upper_bound")
    modified_fields = receipt[prefix + "modified_state_field_names"]
    if (
        generated != selected + rejected
        or requested != selected
        or prefix_rejected > rejected
        or steps != selected + prefix_rejected
        or support_macs < score_macs
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA atomic receipt drift")

    support_guard_pass = receipt[prefix + "support_guard_pass"]
    safe_directional_pass = receipt[prefix + "safe_directional_pass"]
    if (
        not isinstance(support_guard_pass, bool)
        or not isinstance(safe_directional_pass, bool)
        or safe_directional_pass is not support_guard_pass
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D TCRA safe-directional receipt drift"
        )
    old_gains = receipt[prefix + "old_tail_gain_by_class"]
    if old_gains is None:
        if any(
            receipt[prefix + name] is not None
            for name in (
                "guard_tolerance",
                "old_tail_min_gain",
                "old_tail_gain_sum",
                "old_tail_strict_positive_count",
                "pooled_new_cross_tail_gain",
                "pooled_new_allclass_tail_gain",
                "old_to_new_hinge_delta",
                "new_to_old_hinge_delta",
            )
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D TCRA empty support summary drift"
            )
    else:
        if (
            not isinstance(old_gains, list)
            or len(old_gains) != OLD_CLASS_COUNT
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in old_gains
            )
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D TCRA old-tail receipt drift"
            )
        summary_tolerance = finite("guard_tolerance", lower=0.0)
        expected_sum = float(
            np.sum(np.asarray(old_gains, dtype=np.float64), dtype=np.float64)
        )
        expected_count = int(
            np.sum(
                np.asarray(old_gains, dtype=np.float64) > summary_tolerance
            )
        )
        if (
            finite("old_tail_min_gain") != float(min(old_gains))
            or finite("old_tail_gain_sum") != expected_sum
            or integer("old_tail_strict_positive_count") != expected_count
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D TCRA old-tail summary drift"
            )

    if not active_state:
        if (
            active is not False
            or fallback is not False
            or reason != "K1_K2_EXACT_D92_FULL_ALIAS"
            or final_sha != e0_sha
            or modified_fields != []
            or any(
                value != 0
                for value in (
                    generated,
                    selected,
                    rejected,
                    prefix_rejected,
                    steps,
                    requested,
                    applied,
                    changed,
                    state_l1,
                    saturation,
                    full_scores,
                    analytic,
                    score_macs,
                    support_macs,
                )
            )
            or receipt[prefix + "coef2_byte_exact"] is not True
            or receipt[prefix + "support_guard_pass"] is not False
            or safe_directional_pass is not False
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TCRA alias receipt drift")
        return receipt

    if fallback is True:
        if (
            active is not False
            or not isinstance(reason, str)
            or not reason
            or final_sha != e0_sha
            or modified_fields != []
            or applied != 0
            or changed != 0
            or state_l1 != 0
            or receipt[prefix + "coef2_byte_exact"] is not True
            or receipt[prefix + "support_guard_pass"] is not False
            or safe_directional_pass is not False
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TCRA fallback receipt drift")
        return receipt
    if fallback is not False:
        raise D92E0DQueryEvaluationError("D92-E0D TCRA fallback flag drift")

    if (
        active is not True
        or reason is not None
        or generated <= 0
        or selected <= 0
        or applied != selected
        or changed != selected
        or state_l1 != changed
        or prefix_rejected < 0
        or final_sha == e0_sha
        or modified_fields != ["coef2_qint8"]
        or receipt[prefix + "coef2_byte_exact"] is not False
        or receipt[prefix + "support_guard_pass"] is not True
        or safe_directional_pass is not True
        or saturation < 0
        or full_scores != steps + 2
        or analytic <= 0
        or score_macs <= 0
        or support_macs <= 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA active receipt drift")
    old_counts = receipt[prefix + "old_tail_count_by_class"]
    if (
        not isinstance(old_counts, list)
        or len(old_counts) != OLD_CLASS_COUNT
        or any(int(value) <= 0 for value in old_counts)
        or not isinstance(old_gains, list)
        or len(old_gains) != OLD_CLASS_COUNT
        or integer("pooled_new_tail_count", lower=1) <= 0
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA fixed-tail receipt drift")
    tolerance = finite("guard_tolerance", lower=0.0)
    if (
        any(
            float(value) < -tolerance
            for value in old_gains
        )
        or finite("old_tail_min_gain") < -tolerance
        or finite("pooled_new_cross_tail_gain") < -tolerance
        or finite("pooled_new_allclass_tail_gain") < -tolerance
        or finite("old_to_new_hinge_delta") > tolerance
        or finite("new_to_old_hinge_delta") > tolerance
        or finite("old_tail_gain_sum") <= tolerance
        or integer("old_tail_strict_positive_count") <= 0
        or max(float(value) for value in old_gains) <= tolerance
    ):
        raise D92E0DQueryEvaluationError("D92-E0D TCRA support guard drift")
    return receipt


def _audit_d92_e0d_fit(
    result: Any,
    *,
    arm: D92E0DSlimArmSpec,
    scenario: str,
    k_shot: int,
    old_count: int,
    class_count: int,
) -> dict[str, Any]:
    geometry = result.geometry_audit
    before = dict(geometry.get("before_covariance_audit", {}))
    after = dict(geometry.get("final_covariance_audit", {}))
    registered_d_mode_active = int(k_shot) > 2 and not arm.e_enabled
    expected_after_e = arm.e_enabled if int(k_shot) > 2 else True
    expected_after_mode = (
        arm.registered_d_mode
        if registered_d_mode_active
        else "d92_full_alias"
    )
    checks = (
        (before, False, True, "d92_full_alias", False),
        (
            after,
            True,
            expected_after_e,
            expected_after_mode,
            registered_d_mode_active,
        ),
    )
    for audit, registered, expected_e, expected_mode, expected_mode_active in checks:
        _ocf_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
        )
        _floorboost_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
            class_count=class_count if registered else old_count,
        )
        _newguard_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
            class_count=class_count if registered else old_count,
        )
        _pareto_distill_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
            class_count=class_count if registered else old_count,
        )
        _csoas_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
            class_count=class_count if registered else old_count,
        )
        _ccoc_support_receipt(
            audit,
            arm=arm,
            registered=registered,
            k_shot=k_shot,
            class_count=class_count if registered else old_count,
        )
        inventory = audit.get("d92_e0d_actual_component_inventory")
        if (
            audit.get("d92_registration_state_support_only") is not True
            or int(audit.get("d92_query_rows_used", -1)) != 0
            or audit.get("d92_query_role_oracle_access") is not False
            or audit.get("d92_scene_receiver_seed_specific_branch") is not False
            or audit.get("d92_class_id_specific_formula") is not False
            or any(audit.get(field) is not False for field in _FORBIDDEN_QUERY_ACCESS_FIELDS)
            or audit.get("d92_e0d_arm_id") != arm.arm_id
            or audit.get("d92_e0d_candidate_id") != arm.candidate_id
            or audit.get("d92_e0d_B_enabled") is not True
            or audit.get("d92_e0d_E_enabled") is not arm.e_enabled
            or audit.get("d92_e0d_B_effective") is not True
            or audit.get("d92_e0d_E_effective") is not expected_e
            or audit.get("d92_e0d_registered_state") is not registered
            or audit.get("d92_e0d_registered_d_mode") != arm.registered_d_mode
            or audit.get("d92_e0d_registered_d_mode_active") is not expected_mode_active
            or audit.get("d92_e0d_registered_d_mode_effective") != expected_mode
            or audit.get("d92_e0d_finite_output_pass") is not True
            or not isinstance(inventory, dict)
            or int(inventory.get("actual_component_fit_count", -1))
            != int(audit.get("d92_e0d_actual_component_fit_count", -2))
        ):
            raise D92E0DQueryEvaluationError("D92-E0D protocol closure drift")
    after_csoas_receipt = _csoas_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        class_count=class_count,
    )
    after_ccoc_receipt = _ccoc_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        class_count=class_count,
    )
    if int(k_shot) > 2:
        expected_total = expected_total_component_fit_count(
            k_shot, arm_id=arm.arm_id
        )
        expected_actual = expected_total // 2
        if after_csoas_receipt.get("d92_csoas_fallback_active") is True:
            if (
                after.get("d92_csoas_codec_fallback_scope")
                == "whole_d42_retry_before_and_after"
            ):
                expected_total = 4
                expected_actual = 4
            else:
                expected_total = 3
                expected_actual = 2
        if after_ccoc_receipt.get("d92_e0d_ccoc_fallback_active") is True:
            if (
                after.get("d92_ccoc_codec_fallback_scope")
                == "whole_d42_retry_before_and_after"
            ):
                inventory = after.get("d92_e0d_actual_component_inventory")
                if not isinstance(inventory, dict):
                    raise D92E0DQueryEvaluationError(
                        "D92-E0D CCOC codec inventory drift"
                    )
                expected_actual = int(
                    inventory.get("actual_component_fit_count", -1)
                )
                expected_total = expected_actual
                if (
                    expected_actual < 1
                    or int(
                        after.get(
                            "d92_ccoc_codec_fallback_component_execution_count",
                            -1,
                        )
                    )
                    != expected_actual
                ):
                    raise D92E0DQueryEvaluationError(
                        "D92-E0D CCOC codec inventory drift"
                    )
            else:
                expected_total = 3
                expected_actual = 2
        if (
            int(after.get("d92_e0d_total_component_fit_count", -1))
            != expected_total
            or int(after.get("d92_e0d_actual_component_fit_count", -1))
            != expected_actual
            or after.get("d92_e0d_two_state_registered_count_applies") is not True
        ):
            raise D92E0DQueryEvaluationError("D92-E0D registered fit-count drift")
    if int(after.get("d92_e0d_query_macs", -1)) != int(class_count) * 288:
        raise D92E0DQueryEvaluationError("D92-E0D query affine MAC drift")

    def transform_receipt(
        audit: dict[str, Any], *, expected_class_count: int
    ) -> tuple[float, float]:
        transform = audit.get("d81_transform_audit")
        if not isinstance(transform, dict):
            raise D92E0DQueryEvaluationError("D92-E0D D81 transform audit missing")
        try:
            center_shift = float(transform["center_shift_l2_max"])
            effective_samples = np.asarray(
                transform["effective_sample_size_by_class"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise D92E0DQueryEvaluationError(
                "D92-E0D D81 transform receipt drift"
            ) from error
        if (
            transform.get("schema")
            != "cvs.phase2.d81.support_center_translation.v1"
            or int(transform.get("support_rows", -1))
            != int(expected_class_count) * int(k_shot)
            or int(transform.get("class_count", -1)) != int(expected_class_count)
            or int(transform.get("k_shot", -1)) != int(k_shot)
            or transform.get("uses_outer_held_or_query") is not False
            or int(transform.get("query_rows_used", -1)) != 0
            or not np.isfinite(center_shift)
            or center_shift < 0.0
            or effective_samples.ndim != 1
            or effective_samples.size != int(expected_class_count)
            or not np.isfinite(effective_samples).all()
            or np.any(effective_samples <= 0.0)
            or np.any(effective_samples > float(k_shot) + 1e-9)
        ):
            raise D92E0DQueryEvaluationError(
                "D92-E0D D81 transform receipt drift"
            )
        return center_shift, float(np.min(effective_samples))

    before_transform = transform_receipt(before, expected_class_count=old_count)
    after_transform = transform_receipt(after, expected_class_count=class_count)
    after_ocf_receipt = _ocf_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
    )
    after_floorboost_receipt = _floorboost_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        class_count=class_count,
    )
    after_newguard_receipt = _newguard_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        class_count=class_count,
    )
    after_pareto_distill_receipt = _pareto_distill_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        class_count=class_count,
    )
    after_tpce_receipt = _tpce_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        state=result.state,
    )
    after_tcra_receipt = _tcra_support_receipt(
        after,
        arm=arm,
        registered=True,
        k_shot=k_shot,
        state=result.state,
    )
    after_pareto_deployed_state_closure = _pareto_deployed_state_closure(
        result.state,
        after_pareto_distill_receipt,
        arm=arm,
    )
    before_resource = _resource_receipt(before)
    after_resource = _resource_receipt(after)
    return {
        "scenario": str(scenario),
        "k_shot": int(k_shot),
        "old_class_count": int(old_count),
        "registered_class_count": int(class_count),
        "k1_unit_covariance_fallback": bool(
            geometry["k1_unit_covariance_fallback"]
        ),
        "before_covariance_policy": str(before.get("covariance_policy")),
        "after_covariance_policy": str(after.get("covariance_policy")),
        "before_center_shift_l2_max": before_transform[0],
        "after_center_shift_l2_max": after_transform[0],
        "before_effective_sample_size_min": before_transform[1],
        "after_effective_sample_size_min": after_transform[1],
        "before_state_bytes": int(result.before_state.persistent_state_bytes),
        "after_state_bytes": int(result.state.persistent_state_bytes),
        "before_state_fingerprint_sha256": _state_fingerprint_sha256(
            result.before_state
        ),
        "after_state_fingerprint_sha256": _state_fingerprint_sha256(result.state),
        "training_trace": [dict(row) for row in result.training_trace],
        "resource_audit": dict(result.resource_audit),
        "arm_id": arm.arm_id,
        "candidate_id": arm.candidate_id,
        "before_effective_B": True,
        "before_effective_E": True,
        "after_effective_B": True,
        "after_effective_E": expected_after_e,
        "before_registered_d_mode_effective": "d92_full_alias",
        "after_registered_d_mode_effective": expected_after_mode,
        "after_state_postprocess_mode": (
            "d42_tpce"
            if arm.arm_id in _TPCE_ARM_IDS and int(k_shot) > 2
            else (
                "d42_tcra"
                if arm.arm_id in _TCRA_ARM_IDS and int(k_shot) > 2
                else None
            )
        ),
        "before_total_component_fit_count": int(
            before["d92_e0d_total_component_fit_count"]
        ),
        "after_total_component_fit_count": int(
            after["d92_e0d_total_component_fit_count"]
        ),
        "after_actual_component_inventory": dict(
            after["d92_e0d_actual_component_inventory"]
        ),
        "before_registration_resource": before_resource,
        "after_registration_resource": after_resource,
        "d92_e0d_ocf_active": after_ocf_receipt["d92_e0d_ocf_active"],
        "d92_e0d_ocf_lambda": after_ocf_receipt["d92_e0d_ocf_lambda"],
        "d92_e0d_ocf_support_alignment_affine_macs_upper_bound": (
            after_ocf_receipt[
                "d92_e0d_ocf_support_alignment_affine_macs_upper_bound"
            ]
        ),
        "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound": (
            after_ocf_receipt[
                "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"
            ]
        ),
        "d92_e0d_ocf_support_alignment_macs_upper_bound": after_ocf_receipt[
            "d92_e0d_ocf_support_alignment_macs_upper_bound"
        ],
        "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": after_ocf_receipt[
            "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound"
        ],
        **after_floorboost_receipt,
        **after_newguard_receipt,
        **after_pareto_distill_receipt,
        **after_csoas_receipt,
        **after_ccoc_receipt,
        **after_pareto_deployed_state_closure,
        **after_tpce_receipt,
        **after_tcra_receipt,
        "query_macs": int(after["d92_e0d_query_macs"]),
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "query_global_reassignment": False,
    }


def run_d92_e0d_query_evaluation(
    *,
    arm_id: str,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str,
    before_apply_package_root: str | Path,
    before_apply_seal_path: str | Path,
    before_apply_seal_sha256: str,
    after_enrollment_package_root: str | Path,
    after_enrollment_seal_path: str | Path,
    after_enrollment_seal_sha256: str,
    after_apply_package_root: str | Path,
    after_apply_seal_path: str | Path,
    after_apply_seal_sha256: str,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    output_root: str | Path,
    device: str,
    technical_support_receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one frozen arm without exposing a truth-side input surface."""

    from cvsrffi import stage2_d81_query_evaluation as d81_eval
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe

    try:
        arm = D92_E0D_ARMS[str(arm_id)]
    except KeyError as error:
        raise D92E0DQueryEvaluationError(f"unknown D92-E0D arm: {arm_id}") from error
    if technical_support_receipt_sink is not None and not callable(
        technical_support_receipt_sink
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support receipt sink must be callable"
        )
    if (
        technical_support_receipt_sink is not None
        and arm.arm_id not in _TECHNICAL_SUPPORT_RECEIPT_ARM_IDS
    ):
        raise D92E0DQueryEvaluationError(
            "D92-E0D technical support receipt sink requires E0 or CCOC"
        )
    support_receipt_enabled = technical_support_receipt_sink is not None
    original_builder = d81_probe.build_d81_fit
    original_candidate = d81_eval.CANDIDATE_D81
    original_schema = d81_eval.SCHEMA
    original_audit = d81_eval._audit_fit
    original_d42_fit = d81_eval.fit_d42_unified_shrinkage_lda
    after_state_codec_errors: list[Exception] = []
    ccoc_codec_execution_ledger: list[dict[str, Any]] = []
    ccoc_codec_retry_active = False
    pending_support_receipts: dict[int, dict[str, Any]] = {}

    def builder(d42: Any, basis: Any, weights: Any, ground_audit: dict[str, Any]):
        return build_d92_e0d_fit(
            d42,
            basis,
            weights,
            ground_audit,
            arm_id=arm.arm_id,
        )

    def codec_fallback_builder(
        d42: Any, basis: Any, weights: Any, ground_audit: dict[str, Any]
    ):
        """Use the only legal codec-recovery reference after CSOAS fails."""

        reference_fit, call_records, transform_records = build_d92_e0d_fit(
            d42,
            basis,
            weights,
            ground_audit,
            arm_id="E0_FULL_ONLY",
        )

        def fallback_fit(
            rows: np.ndarray,
            labels: np.ndarray,
            class_count: int,
            k_shot: int,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            coefficient, intercept, reference_audit = reference_fit(
                rows, labels, class_count, k_shot
            )
            audit_copy = dict(reference_audit)
            registered = int(class_count) > int(OLD_CLASS_COUNT)
            audit_copy.update(
                {
                    # The published candidate remains CSOAS even though this
                    # exceptional state uses the byte-exact E0 reference.
                    "d92_e0d_arm_id": arm.arm_id,
                    "d92_e0d_candidate_id": arm.candidate_id,
                    "d92_e0d_registered_d_mode": arm.registered_d_mode,
                }
            )
            if not registered:
                audit_copy.update(
                    {
                        "d92_csoas_active": False,
                        "d92_csoas_fallback_active": False,
                        "d92_csoas_fallback_reason": "NOT_REGISTERED_STATE",
                        "d92_csoas_candidate_attempt_fit_count": 0,
                        "d92_csoas_fallback_reference_fit_count": 0,
                        "d92_csoas_candidate_statistic_receipt_available": False,
                        "d92_csoas_fallback_reference_full_head_byte_exact": None,
                        "d92_csoas_paired_e0_codec_state_equal": None,
                        "d92_csoas_query_rows_used": 0,
                        "d92_csoas_query_fit_access": False,
                        "d92_csoas_query_update_access": False,
                        "d92_csoas_query_selection_access": False,
                        "d92_csoas_query_truth_access": False,
                        "d92_csoas_query_role_oracle_access": False,
                        "d92_csoas_query_class_quota_access": False,
                        "d92_csoas_query_global_reassignment": False,
                        "d92_e0d_csoas_g0_eligible": False,
                        "d92_e0d_csoas_g0_block_reason": "NOT_REGISTERED_STATE",
                    }
                )
                return coefficient, intercept, audit_copy
            calls = [
                {
                    "arm": "full",
                    "class_count": int(OLD_CLASS_COUNT),
                    "k_shot": int(k_shot),
                    "status": "csoas_codec_candidate_before_e0",
                    "active": True,
                },
                {
                    "arm": "full",
                    "class_count": int(class_count),
                    "k_shot": int(k_shot),
                    "status": "csoas_codec_numeric_attempt",
                    "active": True,
                },
                {
                    "arm": "full",
                    "class_count": int(OLD_CLASS_COUNT),
                    "k_shot": int(k_shot),
                    "status": "csoas_codec_retry_before_e0",
                    "active": True,
                },
                {
                    "arm": "full",
                    "class_count": int(class_count),
                    "k_shot": int(k_shot),
                    "status": "csoas_codec_retry_after_e0_reference",
                    "active": True,
                },
            ]
            inventory = dict(audit_copy["d92_e0d_actual_component_inventory"])
            inventory.update(
                {
                    "actual_component_fit_count": 4,
                    "actual_component_calls": calls,
                    "full_component_fit_count": 4,
                }
            )
            audit_copy.update(
                {
                    "d92_e0d_actual_component_fit_count": 4,
                    "d92_e0d_actual_component_inventory": inventory,
                    "d92_e0d_total_component_fit_count": 4,
                    "d92_e0d_two_state_registered_count_applies": True,
                    "d92_e0d_registered_d_mode_active": True,
                    "d92_e0d_registered_d_mode_effective": arm.registered_d_mode,
                    "d92_csoas_active": False,
                    "d92_csoas_fallback_active": True,
                    "d92_csoas_fallback_reason": "D42_CODEC_NUMERIC_RETRY_E0_FULL",
                    "d92_csoas_candidate_attempt_fit_count": 1,
                    "d92_csoas_fallback_reference_fit_count": 1,
                    "d92_csoas_candidate_statistic_receipt_available": False,
                    "d92_csoas_fallback_reference_full_head_byte_exact": True,
                    "d92_csoas_paired_e0_codec_state_equal": None,
                    "d92_csoas_query_rows_used": 0,
                    "d92_csoas_query_fit_access": False,
                    "d92_csoas_query_update_access": False,
                    "d92_csoas_query_selection_access": False,
                    "d92_csoas_query_truth_access": False,
                    "d92_csoas_query_role_oracle_access": False,
                    "d92_csoas_query_class_quota_access": False,
                    "d92_csoas_query_global_reassignment": False,
                    "d92_e0d_csoas_g0_eligible": False,
                    "d92_e0d_csoas_g0_block_reason": "NUMERIC_FALLBACK_EXACT_E0",
                    # The safe whole-D42 retry re-executes its before E0 fit;
                    # expose all four component executions instead of hiding it.
                    "d92_csoas_codec_fallback_component_execution_count": 4,
                    "d92_csoas_codec_fallback_scope": "whole_d42_retry_before_and_after",
                }
            )
            return coefficient, intercept, audit_copy

        return fallback_fit, call_records, transform_records

    def ccoc_codec_fallback_builder(
        d42: Any, basis: Any, weights: Any, ground_audit: dict[str, Any]
    ):
        """Retry the whole D42 path with E0 and a real CCOC execution ledger."""

        nonlocal ccoc_codec_retry_active
        ccoc_codec_retry_active = True
        reference_fit, call_records, transform_records = build_d92_e0d_fit(
            d42,
            basis,
            weights,
            ground_audit,
            arm_id="E0_FULL_ONLY",
        )

        def inactive_receipt(class_count: int, k_shot: int) -> dict[str, Any]:
            raw = d92_probe.ccoc_inactive_receipt(
                int(class_count), int(k_shot), old_class_count=OLD_CLASS_COUNT
            )
            mirrored = {
                key.replace("d92_ccoc_", "d92_e0d_ccoc_"): value
                for key, value in raw.items()
            }
            mirrored.update(
                {
                    "d92_e0d_ccoc_fallback_reason": "NOT_REGISTERED_STATE",
                    "d92_e0d_ccoc_candidate_attempt_fit_count": 0,
                    "d92_e0d_ccoc_fallback_reference_fit_count": 0,
                    "d92_e0d_ccoc_candidate_statistic_receipt_available": False,
                    "d92_e0d_ccoc_fallback_reference_full_head_byte_exact": None,
                    "d92_e0d_ccoc_paired_e0_codec_state_equal": None,
                    "d92_e0d_ccoc_g0_eligible": False,
                    "d92_e0d_ccoc_g0_block_reason": "NOT_REGISTERED_STATE",
                }
            )
            return {**raw, **mirrored}

        def fallback_receipt(class_count: int) -> dict[str, Any]:
            raw = {
                "d92_ccoc_active": False,
                "d92_ccoc_fallback_active": True,
                "d92_ccoc_fallback_reason": "D42_CODEC_NUMERIC_RETRY_E0_FULL",
                "d92_ccoc_formula_revision": "pairwise_cosine_v1",
                "d92_ccoc_formula": (
                    "Sigma=0.5*mix(Sigma_old,rho_old)+0.5*mix(Sigma_new,rho_new)"
                ),
                "d92_ccoc_old_rho": None,
                "d92_ccoc_new_rho": None,
                "d92_ccoc_old_group_class_count": int(OLD_CLASS_COUNT),
                "d92_ccoc_new_group_class_count": int(class_count)
                - int(OLD_CLASS_COUNT),
                "d92_ccoc_candidate_attempt_fit_count": 1,
                "d92_ccoc_fallback_reference_fit_count": 1,
                "d92_ccoc_candidate_statistic_receipt_available": False,
                "d92_ccoc_fallback_reference_full_head_byte_exact": True,
                "d92_ccoc_paired_e0_codec_state_equal": None,
                "d92_ccoc_query_rows_used": 0,
                "d92_ccoc_query_fit_access": False,
                "d92_ccoc_query_update_access": False,
                "d92_ccoc_query_selection_access": False,
                "d92_ccoc_query_truth_access": False,
                "d92_ccoc_query_role_oracle_access": False,
                "d92_ccoc_query_class_quota_access": False,
                "d92_ccoc_query_global_reassignment": False,
                "d92_ccoc_codec_fallback_component_execution_count": len(
                    ccoc_codec_execution_ledger
                ),
                "d92_ccoc_codec_fallback_scope": (
                    "whole_d42_retry_before_and_after"
                ),
            }
            mirrored = {
                key.replace("d92_ccoc_", "d92_e0d_ccoc_"): value
                for key, value in raw.items()
            }
            mirrored.update(
                {
                    "d92_e0d_ccoc_g0_eligible": False,
                    "d92_e0d_ccoc_g0_block_reason": "NUMERIC_FALLBACK_EXACT_E0",
                }
            )
            return {**raw, **mirrored}

        def fallback_fit(
            rows: np.ndarray,
            labels: np.ndarray,
            class_count: int,
            k_shot: int,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            coefficient, intercept, reference_audit = reference_fit(
                rows, labels, class_count, k_shot
            )
            audit_copy = dict(reference_audit)
            registered = int(class_count) > int(OLD_CLASS_COUNT)
            ccoc_codec_execution_ledger.append(
                {
                    "arm": "full",
                    "class_count": int(class_count),
                    "k_shot": int(k_shot),
                    "status": (
                        "ccoc_codec_retry_after_e0_reference"
                        if registered
                        else "ccoc_codec_retry_before_e0"
                    ),
                    "active": True,
                }
            )
            audit_copy.update(
                {
                    "d92_e0d_arm_id": arm.arm_id,
                    "d92_e0d_candidate_id": arm.candidate_id,
                    "d92_e0d_registered_d_mode": arm.registered_d_mode,
                }
            )
            if not registered:
                audit_copy.update(inactive_receipt(class_count, k_shot))
                return coefficient, intercept, audit_copy
            calls = [dict(record) for record in ccoc_codec_execution_ledger]
            actual_count = len(calls)
            inventory = dict(audit_copy["d92_e0d_actual_component_inventory"])
            inventory.update(
                {
                    "actual_component_fit_count": actual_count,
                    "actual_component_calls": calls,
                    "full_component_fit_count": actual_count,
                }
            )
            audit_copy.update(
                {
                    "d92_e0d_actual_component_fit_count": actual_count,
                    "d92_e0d_actual_component_inventory": inventory,
                    "d92_e0d_total_component_fit_count": actual_count,
                    "d92_e0d_two_state_registered_count_applies": True,
                    "d92_e0d_registered_d_mode_active": True,
                    "d92_e0d_registered_d_mode_effective": arm.registered_d_mode,
                    **fallback_receipt(class_count),
                }
            )
            return coefficient, intercept, audit_copy

        return fallback_fit, call_records, transform_records

    def is_d42_codec_numerical_error(error: BaseException) -> bool:
        from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

        return isinstance(error, d42.D42UnifiedShrinkageLDAError) and str(error) in {
            "D42 quantization scale overflow",
            "D42 coefficient decode became non-finite",
            "D42 intercept FP16 overflow",
        }

    def fit_with_csoas_codec_guard(*args: Any, **kwargs: Any) -> Any:
        """Identify only the registered int8 final-codec numerical boundary."""

        from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

        after_state_codec_errors.clear()
        original_compile_state = d42._compile_state

        def guarded_compile_state(
            classes: tuple[str, ...],
            old_class_count: int,
            *compile_args: Any,
            precision: str,
            **compile_kwargs: Any,
        ) -> Any:
            if (
                arm.arm_id in _CCOC_ARM_IDS
                and str(precision).lower() == "int8"
                and not ccoc_codec_retry_active
            ):
                ccoc_codec_execution_ledger.append(
                    {
                        "arm": "full",
                        "class_count": int(len(classes)),
                        "k_shot": 0,
                        "status": (
                            "ccoc_codec_numeric_attempt"
                            if len(classes) > int(old_class_count)
                            else "ccoc_codec_candidate_before_e0"
                        ),
                        "active": True,
                    }
                )
            try:
                return original_compile_state(
                    classes,
                    old_class_count,
                    *compile_args,
                    precision=precision,
                    **compile_kwargs,
                )
            except Exception as error:
                if (
                    len(classes) > int(old_class_count)
                    and str(precision).lower() == "int8"
                    and is_d42_codec_numerical_error(error)
                ):
                    after_state_codec_errors.append(error)
                raise

        try:
            d42._compile_state = guarded_compile_state
            return original_d42_fit(*args, **kwargs)
        finally:
            d42._compile_state = original_compile_state

    def fit_with_final_state_technical_support_receipt(
        old_support_features: Any,
        old_support_labels: Any,
        old_classes: Any,
        new_support_features: Any,
        new_support_labels: Any,
        new_classes: Any,
        **kwargs: Any,
    ) -> Any:
        """Capture a transient receipt after the arm's native final-state fit."""

        native_fit = (
            fit_with_csoas_codec_guard
            if arm.arm_id in _CCOC_ARM_IDS
            else original_d42_fit
        )
        result = native_fit(
            old_support_features,
            old_support_labels,
            old_classes,
            new_support_features,
            new_support_labels,
            new_classes,
            **kwargs,
        )
        receipt = _final_state_technical_support_receipt(
            result,
            arm_id=arm.arm_id,
            old_support_features=old_support_features,
            old_support_labels=old_support_labels,
            old_classes=old_classes,
            new_support_features=new_support_features,
            new_support_labels=new_support_labels,
            new_classes=new_classes,
        )
        pending_support_receipts[id(result)] = receipt
        return result

    def audit(
        result: Any,
        *,
        scenario: str,
        k_shot: int,
        old_count: int,
        class_count: int,
    ) -> dict[str, Any]:
        row = _audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario=scenario,
            k_shot=k_shot,
            old_count=old_count,
            class_count=class_count,
        )
        if support_receipt_enabled:
            try:
                receipt = pending_support_receipts.pop(id(result))
            except KeyError as error:
                raise D92E0DQueryEvaluationError(
                    "D92-E0D technical support receipt drift"
                ) from error
            technical_support_receipt_sink(
                {
                    **receipt,
                    "scene": str(scenario),
                    "arm_id": arm.arm_id,
                    "candidate_id": arm.candidate_id,
                }
            )
        return row

    def fit_with_state_postprocess(
        old_support_features: Any,
        old_support_labels: Any,
        old_classes: Any,
        new_support_features: Any,
        new_support_labels: Any,
        new_classes: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_d42_fit(
            old_support_features,
            old_support_labels,
            old_classes,
            new_support_features,
            new_support_labels,
            new_classes,
            **kwargs,
        )
        if arm.arm_id not in (_TPCE_ARM_IDS | _TCRA_ARM_IDS):
            return result
        from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

        is_tcra = arm.arm_id in _TCRA_ARM_IDS
        if is_tcra:
            from cvsrffi import stage2_d92_d42_tail_class_row_ascent as postprocess
        else:
            from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as postprocess
        receipt_name = "TCRA" if is_tcra else "TPCE"

        old_rows = np.asarray(old_support_features, dtype=np.float32)
        new_rows = np.asarray(new_support_features, dtype=np.float32)
        old_registry = tuple(str(value) for value in old_classes)
        new_registry = tuple(str(value) for value in new_classes)
        registry = old_registry + new_registry
        if tuple(result.state.classes) != registry:
            raise D92E0DQueryEvaluationError(
                f"D92-E0D {receipt_name} registry drift"
            )
        mapping = {handle: index for index, handle in enumerate(registry)}
        try:
            targets = np.asarray(
                [
                    mapping[str(value)]
                    for value in np.concatenate(
                        [
                            np.asarray(old_support_labels).astype(str),
                            np.asarray(new_support_labels).astype(str),
                        ]
                    ).tolist()
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise D92E0DQueryEvaluationError(
                f"D92-E0D {receipt_name} support registry drift"
            ) from error
        class_counts = np.bincount(targets, minlength=len(registry))
        if (
            len(class_counts) != len(registry)
            or np.any(class_counts <= 0)
            or len(set(int(value) for value in class_counts.tolist())) != 1
        ):
            raise D92E0DQueryEvaluationError(
                f"D92-E0D {receipt_name} K closure drift"
            )
        k_value = int(class_counts[0])
        if k_value <= 2:
            candidate_state = result.state
            postprocess_audit = (
                postprocess.d42_tcra_inactive_receipt(result.state)
                if is_tcra
                else postprocess.d42_tpce_inactive_receipt(result.state)
            )
            post_resource = None
        else:
            def run_state_postprocess() -> Any:
                all_rows = np.concatenate([old_rows, new_rows], axis=0).astype(
                    np.float32
                )
                transformed = d42._transform(
                    all_rows, result.state.log_diag_fp32
                )
                return (
                    postprocess.apply_d42_tail_class_row_ascent(
                        result.state,
                        transformed,
                        targets,
                        old_class_count=len(old_registry),
                    )
                    if is_tcra
                    else postprocess.apply_d42_tail_pair_code_exchange(
                        result.state,
                        transformed,
                        targets,
                        old_class_count=len(old_registry),
                    )
                )

            measured, post_resource = measure_registration_call(
                run_state_postprocess
            )
            candidate_state, postprocess_audit = measured
        source_prefix = "d92_tcra_" if is_tcra else "d92_tpce_"
        formal_prefix = "d92_e0d_tcra_" if is_tcra else "d92_e0d_tpce_"
        formal_receipt = {
            key.replace(source_prefix, formal_prefix): value
            for key, value in postprocess_audit.items()
            if key.startswith(source_prefix)
        }
        geometry = dict(result.geometry_audit)
        final_audit = dict(geometry.get("final_covariance_audit", {}))
        if post_resource is not None:
            if any(field not in final_audit for field in _RESOURCE_FIELDS):
                raise D92E0DQueryEvaluationError(
                    f"D92-E0D {receipt_name} base resource receipt drift"
                )
            base_baseline = int(final_audit["registration_baseline_rss_bytes"])
            combined_peak = max(
                int(final_audit["registration_peak_rss_bytes"]),
                int(post_resource["registration_peak_rss_bytes"]),
            )
            final_audit.update(
                {
                    "registration_wall_time_ns": int(
                        final_audit["registration_wall_time_ns"]
                    )
                    + int(post_resource["registration_wall_time_ns"]),
                    "registration_process_cpu_time_ns": int(
                        final_audit["registration_process_cpu_time_ns"]
                    )
                    + int(post_resource["registration_process_cpu_time_ns"]),
                    "registration_peak_rss_bytes": combined_peak,
                    "registration_incremental_peak_working_set_bytes": max(
                        0, combined_peak - base_baseline
                    ),
                }
            )
        final_audit.update(formal_receipt)
        geometry["final_covariance_audit"] = final_audit
        return replace(result, state=candidate_state, geometry_audit=geometry)

    try:
        d81_probe.build_d81_fit = builder
        d81_eval.CANDIDATE_D81 = arm.candidate_id
        d81_eval.SCHEMA = SCHEMA_BY_ARM[arm.arm_id]
        d81_eval._audit_fit = audit
        if support_receipt_enabled:
            d81_eval.fit_d42_unified_shrinkage_lda = (
                fit_with_final_state_technical_support_receipt
            )
        elif arm.arm_id in (_TPCE_ARM_IDS | _TCRA_ARM_IDS):
            d81_eval.fit_d42_unified_shrinkage_lda = fit_with_state_postprocess
        elif arm.arm_id in _CCOC_ARM_IDS:
            d81_eval.fit_d42_unified_shrinkage_lda = fit_with_csoas_codec_guard
        elif arm.arm_id in _CSOAS_ARM_IDS:
            d81_eval.fit_d42_unified_shrinkage_lda = fit_with_csoas_codec_guard
        evaluation_kwargs = {
            "before_enrollment_package_root": before_enrollment_package_root,
            "before_enrollment_seal_path": before_enrollment_seal_path,
            "before_enrollment_seal_sha256": before_enrollment_seal_sha256,
            "before_apply_package_root": before_apply_package_root,
            "before_apply_seal_path": before_apply_seal_path,
            "before_apply_seal_sha256": before_apply_seal_sha256,
            "after_enrollment_package_root": after_enrollment_package_root,
            "after_enrollment_seal_path": after_enrollment_seal_path,
            "after_enrollment_seal_sha256": after_enrollment_seal_sha256,
            "after_apply_package_root": after_apply_package_root,
            "after_apply_seal_path": after_apply_seal_path,
            "after_apply_seal_sha256": after_apply_seal_sha256,
            "ground_component_dir": ground_component_dir,
            "ground_manifest_sha256": ground_manifest_sha256,
            "output_root": output_root,
            "device": device,
        }
        try:
            result = d81_eval.run_d81_query_evaluation(**evaluation_kwargs)
        except Exception as error:
            if (
                arm.arm_id not in (_CSOAS_ARM_IDS | _CCOC_ARM_IDS)
                or not after_state_codec_errors
                or error is not after_state_codec_errors[-1]
            ):
                raise
            # D81 only creates the output after all three scene fits close, so
            # this retry cannot overwrite a partially published artifact.
            if arm.arm_id in _CCOC_ARM_IDS:
                d81_probe.build_d81_fit = ccoc_codec_fallback_builder
                result = {
                    **d81_eval.run_d81_query_evaluation(**evaluation_kwargs),
                    "d92_e0d_ccoc_codec_numeric_fallback": True,
                    "d92_e0d_ccoc_codec_fallback_component_execution_count": len(
                        ccoc_codec_execution_ledger
                    ),
                    "d92_e0d_ccoc_codec_fallback_scope": (
                        "whole_d42_retry_before_and_after"
                    ),
                }
            else:
                d81_probe.build_d81_fit = codec_fallback_builder
                result = {
                    **d81_eval.run_d81_query_evaluation(**evaluation_kwargs),
                    "d92_csoas_codec_numeric_fallback": True,
                    "d92_csoas_codec_fallback_component_execution_count": 4,
                    "d92_csoas_codec_fallback_scope": "whole_d42_retry_before_and_after",
                }
    finally:
        d81_probe.build_d81_fit = original_builder
        d81_eval.CANDIDATE_D81 = original_candidate
        d81_eval.SCHEMA = original_schema
        d81_eval._audit_fit = original_audit
        d81_eval.fit_d42_unified_shrinkage_lda = original_d42_fit
        pending_support_receipts.clear()
    if (
        result.get("candidate") != arm.candidate_id
        or result.get("schema") != SCHEMA_BY_ARM[arm.arm_id]
    ):
        raise D92E0DQueryEvaluationError("D92-E0D result identity drift")
    return {**result, "arm_id": arm.arm_id}


__all__ = [
    "D92E0DQueryEvaluationError",
    "SCHEMA_BY_ARM",
    "_audit_d92_e0d_fit",
    "_state_fingerprint_sha256",
    "run_d92_e0d_query_evaluation",
]
