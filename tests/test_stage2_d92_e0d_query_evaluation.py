from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d81_query_evaluation as d81_eval
from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_e0d_query_evaluation as e0d_eval
from cvsrffi import stage2_d92_e0d_slim as slim
from cvsrffi.stage2_d92_cross_class_offblock_consensus import (
    ccoc_inactive_receipt,
)
from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe


def _state(*, classes: int, old_count: int, marker: int):
    state, _ = d42._compile_state(
        tuple(f"tx_{index}" for index in range(classes)),
        old_count,
        np.full(288, marker, dtype=np.float32),
        np.full((classes, 288), marker + 4, dtype=np.float32),
        np.full(classes, marker + 5, dtype=np.float32),
        "sklearn_lsqr_auto_shrinkage_equal_prior",
        precision="int8",
    )
    return state


def _tpce_state(class_count: int = 11):
    code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
    for index in range(class_count):
        code1[index, index] = 1
    return d42.D42UnifiedShrinkageLDAState(
        schema=d42.SCHEMA_INT8,
        classes=tuple(f"tx_{index}" for index in range(class_count)),
        old_class_count=OLD_CLASS_COUNT,
        log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
        coef1_qint8=code1,
        coef2_qint8=np.zeros_like(code1),
        scale1_fp16=np.ones(
            (class_count, len(d42.BLOCK_SLICES)), dtype=np.float16
        ),
        scale2_fp16=np.full(
            (class_count, len(d42.BLOCK_SLICES)), 0.25, dtype=np.float16
        ),
        intercept_fp16=np.zeros(class_count, dtype=np.float16),
        coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
    )


def _tpce_support(
    class_count: int = 11, k_shot: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for class_index in range(class_count):
        for sample_index in range(k_shot):
            row = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.0 + 0.01 * sample_index)
            row[160 + class_index] = np.float32(0.5 + 0.01 * sample_index)
            row[256 + class_index] = np.float32(0.25 + 0.01 * sample_index)
            rows.append(row)
            targets.append(class_index)
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


def _deployed_affine_sha256(state) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(d42.decode_d42_coefficients(state), dtype=np.float32).tobytes()
    )
    digest.update(
        np.ascontiguousarray(state.intercept_fp16, dtype=np.float32).tobytes()
    )
    return digest.hexdigest()


def _resource(
    arm: slim.D92E0DSlimArmSpec, *, registered: bool, k_shot: int
) -> dict:
    active = registered and k_shot > 2 and not arm.e_enabled
    csoas_arm = arm.arm_id == "E0_FULL_CSOAS"
    csoas_active = bool(csoas_arm and registered and k_shot > 2)
    csoas_receipt = (
        _active_csoas_receipt(class_count=11, k_shot=k_shot)
        if csoas_active
        else (
            {
                "d92_csoas_active": False,
                "d92_csoas_fallback_active": False,
                "d92_csoas_fallback_reason": (
                    "NOT_REGISTERED_STATE"
                    if not registered
                    else "K1_K2_EXACT_D92_FULL_ALIAS"
                ),
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
                "d92_e0d_csoas_g0_block_reason": (
                    "NOT_REGISTERED_STATE"
                    if not registered
                    else "K1_K2_EXACT_D92_FULL_ALIAS"
                ),
            }
            if csoas_arm
            else {}
        )
    )
    floorboost_applies = arm.arm_id == "E0_FULL_MAXMIN_FLOORBOOST"
    floorboost_active = bool(
        floorboost_applies and registered and k_shot > 2
    )
    floorboost_reason = (
        "NOT_REGISTERED_STATE"
        if floorboost_applies and not registered
        else (
            "K1_K2_EXACT_D92_FULL_ALIAS"
            if floorboost_applies and k_shot <= 2
            else None
        )
    )
    ocf_active = (
        registered
        and k_shot > 2
        and arm.arm_id
        in {"E0_OCF25", "E0_OCF50", "E0_FULL_MAXMIN_FLOORBOOST"}
    )
    effective_mode = arm.registered_d_mode if active else "d92_full_alias"
    total = (
        slim.expected_total_component_fit_count(k_shot, arm_id=arm.arm_id)
        if registered and k_shot > 2
        else 1
    )
    actual = total // 2 if registered and k_shot > 2 else 1
    class_count = 11 if registered else 6
    affine_macs = (
        2 * (OLD_CLASS_COUNT * k_shot) * OLD_CLASS_COUNT * 288
        if ocf_active
        else None
    )
    contrast_mix_macs = (
        5 * OLD_CLASS_COUNT * (288 + 1) if ocf_active else None
    )
    floorboost_ocf_macs = (
        affine_macs + contrast_mix_macs if floorboost_active else None
    )
    floorboost_retention_macs = (
        OLD_CLASS_COUNT * k_shot * class_count * 288
        if floorboost_active
        else None
    )
    floorboost_bias_macs = 6 * OLD_CLASS_COUNT if floorboost_active else None
    newguard_applies = arm.arm_id == "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"
    newguard_active = bool(newguard_applies and registered and k_shot > 2)
    newguard_reason = (
        None
        if newguard_active
        else (
            "NOT_REGISTERED_STATE"
            if newguard_applies and not registered
            else (
                "K1_K2_EXACT_D92_FULL_ALIAS"
                if newguard_applies
                else "MODE_NOT_SELECTED"
            )
        )
    )
    pareto_applies = arm.arm_id == "E0_FULL_BLOCK_PARETO_DISTILL"
    pareto_active = bool(pareto_applies and registered and k_shot > 2)
    pareto_reason = (
        None
        if pareto_active
        else (
            "NOT_REGISTERED_STATE"
            if pareto_applies and not registered
            else (
                "K1_K2_EXACT_D92_FULL_ALIAS" if pareto_applies else None
            )
        )
    )
    return {
        **csoas_receipt,
        "d92_registration_state_support_only": True,
        "d92_query_rows_used": 0,
        "d92_query_role_oracle_access": False,
        "d92_scene_receiver_seed_specific_branch": False,
        "d92_class_id_specific_formula": False,
        "d92_e0d_arm_id": arm.arm_id,
        "d92_e0d_candidate_id": arm.candidate_id,
        "d92_e0d_B_enabled": True,
        "d92_e0d_E_enabled": arm.e_enabled,
        "d92_e0d_B_effective": True,
        "d92_e0d_E_effective": True if not active else arm.e_enabled,
        "d92_e0d_registered_state": registered,
        "d92_e0d_registered_d_mode": arm.registered_d_mode,
        "d92_e0d_registered_d_mode_active": active,
        "d92_e0d_registered_d_mode_effective": effective_mode,
        "d92_e0d_k1_k2_exact_full_alias": k_shot <= 2,
        "d92_e0d_actual_component_fit_count": actual,
        "d92_e0d_actual_component_inventory": {
            "schema": "cvs.phase2.d92.actual_component_fit_inventory.v1",
            "actual_component_fit_count": actual,
            "actual_component_calls": [],
        },
        "d92_e0d_total_component_fit_count": total,
        "d92_e0d_two_state_registered_count_applies": registered and k_shot > 2,
        "d92_e0d_query_macs": (11 if registered else 6) * 288,
        "d92_e0d_query_fit_access": False,
        "d92_e0d_query_update_access": False,
        "d92_e0d_query_selection_access": False,
        "d92_e0d_query_truth_access": False,
        "d92_e0d_query_role_oracle_access": False,
        "d92_e0d_query_class_quota_access": False,
        "d92_e0d_query_global_reassignment": False,
        "d92_e0d_finite_output_pass": True,
        "d92_e0d_ocf_active": ocf_active,
        "d92_e0d_ocf_lambda": arm.ocf_lambda if ocf_active else None,
        "d92_e0d_ocf_support_alignment_affine_macs_upper_bound": affine_macs,
        "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound": (
            contrast_mix_macs
        ),
        "d92_e0d_ocf_support_alignment_macs_upper_bound": (
            affine_macs + contrast_mix_macs if ocf_active else None
        ),
        "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": (
            864 if ocf_active else None
        ),
        "d92_e0d_floorboost_active": floorboost_active,
        "d92_e0d_floorboost_lambda": 0.25 if floorboost_active else None,
        "d92_e0d_floorboost_quantile": 0.20 if floorboost_active else None,
        "d92_e0d_floorboost_quantile_method": (
            "lower" if floorboost_active else None
        ),
        "d92_e0d_floorboost_kappa": 0.35 if floorboost_active else None,
        "d92_e0d_floorboost_fallback_active": False,
        "d92_e0d_floorboost_fallback_reason": floorboost_reason,
        "d92_e0d_floorboost_new_rows_byte_exact": (
            True if floorboost_active else None
        ),
        "d92_e0d_floorboost_full_head_byte_exact": (
            False if floorboost_active else None
        ),
        "d92_e0d_floorboost_old_bias_zero_sum_residual_abs": (
            0.0 if floorboost_active else None
        ),
        "d92_e0d_floorboost_old_intercept_mean_residual_abs": (
            0.0 if floorboost_active else None
        ),
        "d92_e0d_floorboost_max_abs_delta_over_rms": (
            0.35 if floorboost_active else None
        ),
        "d92_e0d_floorboost_full_old_rms": (
            1.0 if floorboost_active else None
        ),
        "d92_e0d_floorboost_retention_score_by_old_class": (
            [-1.8, 0.2, 0.6, 2.6, 3.8, 5.0] if floorboost_active else None
        ),
        "d92_e0d_floorboost_registration_drift_by_old_class": (
            [1.8, 0.8, 1.4, 0.4, 0.2, 0.0] if floorboost_active else None
        ),
        "d92_e0d_floorboost_delta_bias_by_old_class": (
            [0.35, 0.1, 0.05, -0.1, -0.15, -0.25]
            if floorboost_active
            else None
        ),
        "d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound": (
            floorboost_ocf_macs
        ),
        "d92_e0d_floorboost_support_retention_affine_macs_upper_bound": (
            floorboost_retention_macs
        ),
        "d92_e0d_floorboost_support_bias_calibration_macs_upper_bound": (
            floorboost_bias_macs
        ),
        "d92_e0d_floorboost_support_macs_upper_bound": (
            floorboost_ocf_macs + floorboost_retention_macs + floorboost_bias_macs
            if floorboost_active
            else None
        ),
        "d92_e0d_floorboost_support_transient_bytes_upper_bound": (
            1_024 if floorboost_active else None
        ),
        "d92_e0d_floorboost_persistent_state_bytes_delta": 0,
        "d92_e0d_newguard_active": newguard_active,
        "d92_e0d_newguard_fallback_active": False,
        "d92_e0d_newguard_fallback_reason": newguard_reason,
        "d92_e0d_newguard_full_component_fit_count": 1,
        "d92_e0d_newguard_new_rows_byte_exact": True,
        "d92_e0d_newguard_deployment_new_rows_byte_exact": (
            True if newguard_active else None
        ),
        "d92_e0d_newguard_protection_tolerance": (
            1024.0 * np.finfo(np.float32).eps if newguard_active else None
        ),
        "d92_e0d_newguard_closure_tolerance": (
            1.0e-5 if newguard_active else None
        ),
        "d92_e0d_newguard_old_group_zero_sum_residual_max_abs": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_new_support_old_envelope_change_max_abs_error": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_max_abs_Xnew_internal_residual": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_old_group_zero_sum_residual_max_abs": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_new_support_old_envelope_change_max_abs_error": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_new_support_old_envelope_change_max": (
            -0.01 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_new_support_old_envelope_change_max": (
            -0.01 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_new_support_min_margin_change": (
            0.01 if newguard_active else None
        ),
        "d92_e0d_newguard_tau_old_envelope_shift": (
            -0.01 if newguard_active else 0.0
        ),
        "d92_e0d_newguard_deployment_protection_pass": newguard_active,
        "d92_e0d_newguard_full_head_byte_exact": not newguard_active,
        "d92_e0d_newguard_deployment_strength_scale": (
            1.0 if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_candidate_count": (
            1 if newguard_active else 0
        ),
        "d92_e0d_newguard_deployment_full_head_byte_exact": not newguard_active,
        "d92_e0d_newguard_deployment_codec_roundtrip_count": (
            2 if newguard_active else 0
        ),
        "d92_e0d_newguard_deployment_codec_macs_upper_bound": (
            2 * 8 * (11 if registered else 6) * 288 if newguard_active else 0
        ),
        "d92_e0d_newguard_nullspace_rank": 10 if newguard_active else None,
        "d92_e0d_newguard_rank_threshold": 1.0e-8 if newguard_active else None,
        "d92_e0d_newguard_max_abs_Xnew_internal_residual": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_new_support_min_margin_change": (
            0.01 if newguard_active else None
        ),
        "d92_e0d_newguard_tail_margin_change_by_old_class": (
            [0.0] * OLD_CLASS_COUNT if newguard_active else None
        ),
        "d92_e0d_newguard_deployment_tail_margin_change_by_old_class": (
            [0.0] * OLD_CLASS_COUNT if newguard_active else None
        ),
        "d92_e0d_newguard_residual_l2_by_old_class": (
            [0.01] * OLD_CLASS_COUNT if newguard_active else None
        ),
        "d92_e0d_newguard_maxmin_objective": (
            0.0 if newguard_active else None
        ),
        "d92_e0d_newguard_trust_region_utilization": (
            0.25 if newguard_active else None
        ),
        "d92_e0d_newguard_support_optimization_macs_upper_bound": (
            1_024 if newguard_active else 0
        ),
        "d92_e0d_newguard_support_transient_bytes_upper_bound": (
            4_096 if newguard_active else 0
        ),
        "d92_e0d_newguard_persistent_state_bytes_delta": 0,
        "d92_e0d_newguard_query_rows_used": 0,
        "d92_e0d_newguard_query_macs": (11 if registered else 6) * 288,
        "d92_e0d_newguard_query_fit_access": False,
        "d92_e0d_newguard_query_update_access": False,
        "d92_e0d_newguard_query_selection_access": False,
        "d92_e0d_newguard_query_truth_access": False,
        "d92_e0d_newguard_query_role_oracle_access": False,
        "d92_e0d_newguard_query_class_quota_access": False,
        "d92_e0d_newguard_query_global_reassignment": False,
        "d92_e0d_pareto_distill_mode": "pareto_distill",
        "d92_e0d_pareto_distill_active": pareto_active,
        "d92_e0d_pareto_distill_fallback_active": False,
        "d92_e0d_pareto_distill_fallback_reason": pareto_reason,
        "d92_e0d_pareto_distill_local_valid": pareto_active,
        "d92_e0d_pareto_distill_full_head_byte_exact": not pareto_active,
        "d92_e0d_pareto_distill_deployed_support_constraints_pass": pareto_active,
        "d92_e0d_pareto_distill_deployed_full_head_byte_exact": not pareto_active,
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs": (
            0.25 if pareto_active else None
        ),
        "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum": (
            0.125 if pareto_active else None
        ),
        "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass": (
            True if pareto_active else None
        ),
        "d92_e0d_pareto_distill_deployed_e0_affine_sha256": None,
        "d92_e0d_pareto_distill_deployed_candidate_affine_sha256": None,
        "d92_e0d_pareto_distill_full_solve_count": 1 if pareto_active else 0,
        "d92_e0d_pareto_distill_block_solve_count": 1 if pareto_active else 0,
        "d92_e0d_pareto_distill_loo_fit_count": 0,
        "d92_e0d_pareto_distill_fisher_fit_count": 0,
        "d92_e0d_pareto_distill_component_fit_count": 2 if pareto_active else 0,
        "d92_e0d_pareto_distill_covariance_estimation_count": 1 if pareto_active else 0,
        "d92_e0d_pareto_distill_robust_center_transform_count": 1 if pareto_active else 0,
        "d92_e0d_pareto_distill_query_rows_used": 0,
        "d92_e0d_pareto_distill_query_macs": (11 if registered else 6) * 288,
        "d92_e0d_pareto_distill_query_fit_access": False,
        "d92_e0d_pareto_distill_query_update_access": False,
        "d92_e0d_pareto_distill_query_selection_access": False,
        "d92_e0d_pareto_distill_query_truth_access": False,
        "d92_e0d_pareto_distill_query_role_oracle_access": False,
        "d92_e0d_pareto_distill_query_class_quota_access": False,
        "d92_e0d_pareto_distill_query_global_reassignment": False,
        "d81_transform_audit": {
            "schema": "cvs.phase2.d81.support_center_translation.v1",
            "support_rows": class_count * k_shot,
            "class_count": class_count,
            "k_shot": k_shot,
            "uses_outer_held_or_query": False,
            "query_rows_used": 0,
            "center_shift_l2_max": 0.0 if k_shot == 1 else 0.125,
            "effective_sample_size_by_class": (
                [1.0] * class_count
                if k_shot == 1
                else [min(4.42187, float(k_shot) - 0.5)]
                + [float(k_shot) - 0.5] * (class_count - 1)
            ),
        },
        "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
        "schema": "cvs.phase2.registration_resource_receipt.v1",
        "registration_wall_time_ns": 5_000,
        "registration_process_cpu_time_ns": 4_000,
        "registration_baseline_rss_bytes": 100,
        "registration_peak_rss_bytes": 180,
        "registration_incremental_peak_working_set_bytes": 80,
        "rss_sampler": "synthetic",
    }


def _result(
    arm: slim.D92E0DSlimArmSpec, *, after_marker: int = 2, k_shot: int = 5
):
    before_state = _state(classes=6, old_count=6, marker=1)
    state = _state(classes=11, old_count=6, marker=after_marker)
    before = _resource(arm, registered=False, k_shot=k_shot)
    after = _resource(arm, registered=True, k_shot=k_shot)
    if arm.arm_id == "E0_FULL_BLOCK_PARETO_DISTILL":
        if after["d92_e0d_pareto_distill_active"]:
            after["d92_e0d_pareto_distill_deployed_candidate_affine_sha256"] = (
                _deployed_affine_sha256(state)
            )
        elif after["d92_e0d_pareto_distill_fallback_active"]:
            after["d92_e0d_pareto_distill_deployed_e0_affine_sha256"] = (
                _deployed_affine_sha256(state)
            )
    return SimpleNamespace(
        geometry_audit={
            "k1_unit_covariance_fallback": False,
            "before_covariance_audit": before,
            "final_covariance_audit": after,
        },
        before_state=before_state,
        state=state,
        training_trace=[],
        resource_audit={"trainable_parameters": 0},
    )


def _allowed_kwargs() -> dict[str, str]:
    return {
        "before_enrollment_package_root": "be",
        "before_enrollment_seal_path": "bes",
        "before_enrollment_seal_sha256": "a" * 64,
        "before_apply_package_root": "ba",
        "before_apply_seal_path": "bas",
        "before_apply_seal_sha256": "b" * 64,
        "after_enrollment_package_root": "ae",
        "after_enrollment_seal_path": "aes",
        "after_enrollment_seal_sha256": "c" * 64,
        "after_apply_package_root": "aa",
        "after_apply_seal_path": "aas",
        "after_apply_seal_sha256": "d" * 64,
        "ground_component_dir": "ground",
        "ground_manifest_sha256": "e" * 64,
        "output_root": "out",
        "device": "cpu",
    }


def _active_csoas_receipt(*, class_count: int = 11, k_shot: int = 5) -> dict:
    """Minimal independently recomputable CSOAS support-only receipt."""

    weights = [[1.0 / k_shot] * k_shot for _ in range(class_count)]
    trace = [2.0] * class_count
    tau = 2.0 / 288.0
    alpha = 1.0e-4
    denominator = (float(k_shot) + 1.0) * (alpha - tau * tau / 288.0)
    rho = min(1.0, (alpha + tau * tau) / denominator)
    return {
        "d92_csoas_active": True,
        "d92_csoas_fallback_active": False,
        "d92_csoas_fallback_reason": None,
        "d92_csoas_candidate_attempt_fit_count": 1,
        "d92_csoas_fallback_reference_fit_count": 0,
        "d92_csoas_candidate_statistic_receipt_available": True,
        "d92_csoas_fallback_reference_full_head_byte_exact": None,
        "d92_csoas_paired_e0_codec_state_equal": None,
        "d92_csoas_support_rows": class_count * k_shot,
        "d92_csoas_class_count": class_count,
        "d92_csoas_k_shot": k_shot,
        "d92_csoas_old_class_count": OLD_CLASS_COUNT,
        "d92_csoas_new_class_count": class_count - OLD_CLASS_COUNT,
        "d92_csoas_normalized_cauchy_weight_by_class": weights,
        "d92_csoas_weighted_center_by_class": [[0.0] * 288] * class_count,
        "d92_csoas_effective_sample_size_by_class": [float(k_shot)] * class_count,
        "d92_csoas_scatter_trace_by_class": trace,
        "d92_csoas_oas_tau_by_class": [tau] * class_count,
        "d92_csoas_oas_alpha_by_class": [alpha] * class_count,
        "d92_csoas_oas_denominator_by_class": [denominator] * class_count,
        "d92_csoas_oas_rho_by_class": [rho] * class_count,
        "d92_csoas_shrunk_trace_by_class": trace,
        "d92_csoas_old_group_trace": 2.0,
        "d92_csoas_new_group_trace": 2.0,
        "d92_csoas_final_trace": 2.0,
        "d92_csoas_final_eigenvalue_min": 1.0e-3,
        "d92_csoas_final_eigenvalue_max": 1.0,
        "d92_csoas_spd_pass": True,
        "d92_csoas_live_class_scatter_buffers": 1,
        "d92_csoas_class_matrix_stack": False,
        "d92_csoas_single_scatter_buffer_bytes": 288 * 288 * 8,
        "d92_csoas_final_covariance_buffer_bytes": 288 * 288 * 8,
        "d92_csoas_support_scatter_macs_upper_bound": 1,
        "d92_csoas_support_transient_bytes_upper_bound": 1,
        "d92_csoas_query_rows_used": 0,
        "d92_csoas_query_fit_access": False,
        "d92_csoas_query_update_access": False,
        "d92_csoas_query_selection_access": False,
        "d92_csoas_query_truth_access": False,
        "d92_csoas_query_role_oracle_access": False,
        "d92_csoas_query_class_quota_access": False,
        "d92_csoas_query_global_reassignment": False,
        "d92_e0d_csoas_g0_eligible": False,
        "d92_e0d_csoas_g0_block_reason": "PENDING_DEPLOYED_CODEC_PAIRED_E0",
    }


def _ccoc_receipt(
    *, registered: bool, class_count: int, k_shot: int, fallback: bool = False
) -> dict:
    """Return a literal support-only CCOC receipt for query-audit tests."""

    if not registered or k_shot <= 2:
        raw = ccoc_inactive_receipt(
            class_count,
            k_shot,
            old_class_count=OLD_CLASS_COUNT,
        )
        reason = (
            "NOT_REGISTERED_STATE"
            if not registered
            else "K1_K2_EXACT_D92_FULL_ALIAS"
        )
        mirrored = {
            key.replace("d92_ccoc_", "d92_e0d_ccoc_"): value
            for key, value in raw.items()
        }
        mirrored.update(
            {
                "d92_e0d_ccoc_fallback_reason": reason,
                "d92_e0d_ccoc_candidate_attempt_fit_count": 0,
                "d92_e0d_ccoc_fallback_reference_fit_count": 0,
                "d92_e0d_ccoc_candidate_statistic_receipt_available": False,
                "d92_e0d_ccoc_fallback_reference_full_head_byte_exact": None,
                "d92_e0d_ccoc_paired_e0_codec_state_equal": None,
                "d92_e0d_ccoc_g0_eligible": False,
                "d92_e0d_ccoc_g0_block_reason": reason,
            }
        )
        return {**raw, **mirrored}

    active = not fallback
    reason = (
        "injected_ccoc_numeric_failure" if fallback else None
    )
    raw = {
        "d92_ccoc_active": active,
        "d92_ccoc_fallback_active": bool(fallback),
        "d92_ccoc_fallback_reason": reason,
        "d92_ccoc_formula_revision": "pairwise_cosine_v1",
        "d92_ccoc_formula": (
            "Sigma=0.5*mix(Sigma_old,rho_old)+0.5*mix(Sigma_new,rho_new)"
        ),
        "d92_ccoc_old_rho": 0.25,
        "d92_ccoc_new_rho": 0.75,
        "d92_ccoc_old_group_class_count": OLD_CLASS_COUNT,
        "d92_ccoc_new_group_class_count": class_count - OLD_CLASS_COUNT,
        "d92_ccoc_old_offblock_norm_min": 1.0,
        "d92_ccoc_old_offblock_norm_max": 2.0,
        "d92_ccoc_new_offblock_norm_min": 1.5,
        "d92_ccoc_new_offblock_norm_max": 2.5,
        "d92_ccoc_old_pairwise_cosine_raw": 0.25,
        "d92_ccoc_new_pairwise_cosine_raw": 0.75,
        "d92_ccoc_canonicalization": (
            "lexicographic_float32_row_bytes_then_float64_reduce"
        ),
        "d92_ccoc_canonicalization_tie_policy": (
            "float32_row_bytes_then_float64_row_bytes_"
            "duplicate_class_handle_fail_closed"
        ),
        "d92_ccoc_crossblock_passes_per_class": 2,
        "d92_ccoc_upper_block_count": 3,
        "d92_ccoc_covariance_symmetric": True,
        "d92_ccoc_full_endpoint_reused": True,
        "d92_ccoc_full_endpoint_reuse": True,
        "d92_ccoc_additional_fit_count": 0,
        "d92_ccoc_additional_full_fit_count": 0,
        "d92_ccoc_additional_block_fit_count": 0,
        "d92_ccoc_additional_loo_fit_count": 0,
        "d92_ccoc_additional_fisher_fit_count": 0,
        "d92_ccoc_additional_scan_count": 0,
        "d92_ccoc_block_fit_count": 0,
        "d92_ccoc_loo_fit_count": 0,
        "d92_ccoc_fisher_fit_count": 0,
        "d92_ccoc_scan_count": 0,
        "d92_ccoc_hyperparameter_scan_count": 0,
        "d92_ccoc_weight_scan_count": 0,
        "d92_ccoc_dense_solve_count": 1 if active else 0,
        "d92_ccoc_compile_solve_count": 1,
        "d92_ccoc_full_solve_count": 1,
        "d92_ccoc_full_dense_288_solve_count": 1,
        "d92_ccoc_compiled_cholesky_check_count": 0,
        "d92_ccoc_covariance_equation_residual_max": 0.0,
        "d92_ccoc_cholesky_check_count": 3,
        "d92_ccoc_cholesky_endpoint_check_count": 2,
        "d92_ccoc_cholesky_final_check_count": 1,
        "d92_ccoc_cholesky_pass": True,
        "d92_ccoc_old_endpoint_cholesky_min_diagonal": 0.2,
        "d92_ccoc_new_endpoint_cholesky_min_diagonal": 0.3,
        "d92_ccoc_final_cholesky_min_diagonal": 0.4,
        "d92_ccoc_support_macs_upper_bound": 1_000,
        "d92_ccoc_workspace_upper_accumulators_bytes": 1_024,
        "d92_ccoc_workspace_cross_block_buffer_bytes": 512,
        "d92_ccoc_workspace_residual_buffer_bytes": 256,
        "d92_ccoc_workspace_numeric_bytes_upper_bound": 1_792,
        "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound": 334_336,
        "d92_ccoc_workspace_candidate_covariance_result_bytes": (
            288 * 288 * 8
        ),
        "d92_ccoc_workspace_candidate_covariance_block_workspace_bytes": (
            160 * 160 * 8
        ),
        "d92_ccoc_workspace_candidate_covariance_row_workspace_bytes": 160 * 8,
        "d92_ccoc_workspace_candidate_covariance_full_buffer_count_upper_bound": 1,
        "d92_ccoc_workspace_candidate_covariance_mix_live_bytes_upper_bound": (
            288 * 288 * 8 + 160 * 160 * 8 + 160 * 8
        ),
        "d92_ccoc_support_transient_bytes_upper_bound": 1_792,
        "d92_ccoc_persistent_state_bytes_delta": 0,
        "d92_ccoc_persistent_bytes_delta": 0,
        "d92_ccoc_query_state_bytes_delta": 0,
        "d92_ccoc_query_bytes_delta": 0,
        "d92_ccoc_query_macs_delta": 0,
        "d92_ccoc_query_macs": 0,
        "d92_ccoc_query_rows_used": 0,
        "d92_ccoc_query_fit_access": False,
        "d92_ccoc_query_update_access": False,
        "d92_ccoc_query_selection_access": False,
        "d92_ccoc_query_truth_access": False,
        "d92_ccoc_query_role_oracle_access": False,
        "d92_ccoc_query_class_quota_access": False,
        "d92_ccoc_query_global_reassignment": False,
        "d92_ccoc_candidate_attempt_fit_count": 1,
        "d92_ccoc_fallback_reference_fit_count": 1 if fallback else 0,
        "d92_ccoc_candidate_statistic_receipt_available": True,
        "d92_ccoc_fallback_reference_full_head_byte_exact": (
            True if fallback else None
        ),
        "d92_ccoc_paired_e0_codec_state_equal": None,
    }
    if fallback:
        for field in (
            "d92_ccoc_compile_solve_count",
            "d92_ccoc_full_solve_count",
            "d92_ccoc_full_dense_288_solve_count",
            "d92_ccoc_compiled_cholesky_check_count",
            "d92_ccoc_covariance_equation_residual_max",
        ):
            raw.pop(field)
    mirrored = {
        key.replace("d92_ccoc_", "d92_e0d_ccoc_"): value
        for key, value in raw.items()
    }
    mirrored.update(
        {
            "d92_e0d_ccoc_fallback_reason": reason,
            "d92_e0d_ccoc_g0_eligible": active,
            "d92_e0d_ccoc_g0_block_reason": (
                (
                    None
                    if active
                    else (
                        "NUMERIC_FALLBACK_EXACT_E0" if fallback else reason
                    )
                )
            ),
        }
    )
    return {**raw, **mirrored}


def test_audit_keeps_existing_resource_fields_and_adds_state_fingerprints():
    """Would fail if a state array was omitted from the per-scene parity receipt."""

    arm = slim.D92_E0D_ARMS["E0_FULL_ONLY"]
    row = e0d_eval._audit_d92_e0d_fit(
        _result(arm),
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    changed = e0d_eval._audit_d92_e0d_fit(
        _result(arm, after_marker=3),
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["after_total_component_fit_count"] == 2
    assert row["query_macs"] == 11 * 288
    assert row["after_registration_resource"][
        "registration_incremental_peak_working_set_bytes"
    ] == 80
    assert row["query_role_oracle_access"] is False
    assert row["query_class_quota_access"] is False
    assert row["query_global_reassignment"] is False
    assert row["before_center_shift_l2_max"] == 0.125
    assert row["after_center_shift_l2_max"] == 0.125
    assert row["before_effective_sample_size_min"] == 4.42187
    assert row["after_effective_sample_size_min"] == 4.42187
    assert len(row["before_state_fingerprint_sha256"]) == 64
    assert len(row["after_state_fingerprint_sha256"]) == 64
    assert row["after_state_fingerprint_sha256"] != changed[
        "after_state_fingerprint_sha256"
    ]


def test_csoas_query_audit_closes_active_receipt_and_rejects_rho_tamper():
    """Would fail if CSOAS rows could bypass support-only receipt closure."""

    arm = slim.D92_E0D_ARMS["E0_FULL_CSOAS"]
    result = _result(arm)
    receipt = result.geometry_audit["final_covariance_audit"]
    receipt.update(_active_csoas_receipt())
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    assert row["d92_csoas_active"] is True
    assert row["d92_csoas_fallback_active"] is False
    assert row["d92_csoas_paired_e0_codec_state_equal"] is None
    receipt["d92_csoas_oas_rho_by_class"][0] = 0.5
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CSOAS"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_csoas_codec_numeric_error_retries_once_with_e0_reference_builder(
    monkeypatch,
):
    """Would fail if a D42 codec overflow escaped or retried CSOAS again."""

    build_arms: list[str] = []

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        return d42._compile_state(
            tuple(f"tx_{index}" for index in range(11)),
            OLD_CLASS_COUNT,
            None,
            None,
            None,
            "frozen",
            precision="int8",
        )

    def overflow_compile(*_args, **_kwargs):
        raise d42.D42UnifiedShrinkageLDAError("D42 intercept FP16 overflow")

    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        d81_probe.build_d81_fit(None, None, None, {})
        if calls == 1:
            d81_eval.fit_d42_unified_shrinkage_lda()
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d42, "_compile_state", overflow_compile)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_CSOAS", **_allowed_kwargs()
    )
    assert calls == 2
    assert build_arms == ["E0_FULL_CSOAS", "E0_FULL_ONLY"]
    assert result["d92_csoas_codec_numeric_fallback"] is True


def test_csoas_before_state_codec_numeric_error_does_not_fallback(monkeypatch):
    """Would fail if a before-state codec error were mislabeled after-CSOAS."""

    build_arms: list[str] = []

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        return d42._compile_state(
            tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT)),
            OLD_CLASS_COUNT,
            None,
            None,
            None,
            "frozen",
            precision="int8",
        )

    def overflow_compile(*_args, **_kwargs):
        raise d42.D42UnifiedShrinkageLDAError("D42 intercept FP16 overflow")

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        return d81_eval.fit_d42_unified_shrinkage_lda()

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d42, "_compile_state", overflow_compile)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    with pytest.raises(d42.D42UnifiedShrinkageLDAError, match="FP16 overflow"):
        e0d_eval.run_d92_e0d_query_evaluation(
            arm_id="E0_FULL_CSOAS", **_allowed_kwargs()
        )
    assert build_arms == ["E0_FULL_CSOAS"]


def test_csoas_query_audit_reports_four_whole_retry_component_executions():
    """Would fail if an all-D42 codec retry concealed its repeated before fit."""

    arm = slim.D92_E0D_ARMS["E0_FULL_CSOAS"]
    result = _result(arm)
    receipt = result.geometry_audit["final_covariance_audit"]
    receipt.update(
        {
            "d92_csoas_active": False,
            "d92_csoas_fallback_active": True,
            "d92_csoas_fallback_reason": "D42_CODEC_NUMERIC_RETRY_E0_FULL",
            "d92_csoas_candidate_attempt_fit_count": 1,
            "d92_csoas_fallback_reference_fit_count": 1,
            "d92_csoas_candidate_statistic_receipt_available": False,
            "d92_csoas_fallback_reference_full_head_byte_exact": True,
            "d92_e0d_csoas_g0_block_reason": "NUMERIC_FALLBACK_EXACT_E0",
            "d92_e0d_actual_component_fit_count": 4,
            "d92_e0d_actual_component_inventory": {
                "schema": "cvs.phase2.d92.actual_component_fit_inventory.v1",
                "actual_component_fit_count": 4,
                "actual_component_calls": [],
            },
            "d92_e0d_total_component_fit_count": 4,
            "d92_csoas_codec_fallback_component_execution_count": 4,
            "d92_csoas_codec_fallback_scope": "whole_d42_retry_before_and_after",
        }
    )
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_rain_weak",
        k_shot=5,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    assert row["after_total_component_fit_count"] == 4
    assert row["after_actual_component_inventory"]["actual_component_fit_count"] == 4


def test_csoas_codec_structural_error_does_not_fallback(monkeypatch):
    """Would fail if registry/state drift were silently turned into E0."""

    def fake_run(**_kwargs):
        raise d42.D42UnifiedShrinkageLDAError("D42 state drift")

    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    with pytest.raises(d42.D42UnifiedShrinkageLDAError, match="state drift"):
        e0d_eval.run_d92_e0d_query_evaluation(
            arm_id="E0_FULL_CSOAS", **_allowed_kwargs()
        )


@pytest.mark.parametrize(
    ("k_shot", "expected_affine", "expected_mix", "expected_total"),
    (
        (5, 103_680, 8_670, 112_350),
        (10, 207_360, 8_670, 216_030),
    ),
)
def test_audit_exposes_active_ocf_support_receipt_from_after_state(
    k_shot,
    expected_affine,
    expected_mix,
    expected_total,
):
    """Would fail if the formal fit row dropped its active OCF support receipt."""

    arm = slim.D92_E0D_ARMS["E0_OCF25"]
    row = e0d_eval._audit_d92_e0d_fit(
        _result(arm, k_shot=k_shot),
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=k_shot,
        old_count=6,
        class_count=11,
    )
    affine_formula = 2 * (OLD_CLASS_COUNT * k_shot) * OLD_CLASS_COUNT * 288
    mix_formula = 5 * OLD_CLASS_COUNT * (288 + 1)
    assert OLD_CLASS_COUNT == 6
    assert (affine_formula, mix_formula, affine_formula + mix_formula) == (
        expected_affine,
        expected_mix,
        expected_total,
    )
    assert row["d92_e0d_ocf_active"] is True
    assert row["d92_e0d_ocf_lambda"] == pytest.approx(0.25)
    assert (
        row["d92_e0d_ocf_support_alignment_affine_macs_upper_bound"]
        == affine_formula
    )
    assert (
        row["d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"]
        == mix_formula
    )
    assert row["d92_e0d_ocf_support_alignment_macs_upper_bound"] == (
        affine_formula + mix_formula
    )
    assert row["d92_e0d_ocf_support_alignment_transient_bytes_upper_bound"] == 864


def test_audit_exposes_active_floorboost_receipt_with_lower_quantile():
    """Would fail if the formal row lost the frozen floorboost support-only receipt."""

    arm = slim.D92_E0D_ARMS["E0_FULL_MAXMIN_FLOORBOOST"]
    row = e0d_eval._audit_d92_e0d_fit(
        _result(arm),
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["d92_e0d_floorboost_active"] is True
    assert row["d92_e0d_floorboost_lambda"] == pytest.approx(0.25)
    assert row["d92_e0d_floorboost_quantile"] == pytest.approx(0.20)
    assert row["d92_e0d_floorboost_quantile_method"] == "lower"
    assert row["d92_e0d_floorboost_kappa"] == pytest.approx(0.35)
    assert row["d92_e0d_floorboost_fallback_active"] is False
    assert row["d92_e0d_floorboost_new_rows_byte_exact"] is True
    assert row["d92_e0d_floorboost_persistent_state_bytes_delta"] == 0
    assert row["d92_e0d_floorboost_support_macs_upper_bound"] == (
        row["d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound"]
        + row["d92_e0d_floorboost_support_retention_affine_macs_upper_bound"]
        + row["d92_e0d_floorboost_support_bias_calibration_macs_upper_bound"]
    )


def test_audit_rejects_active_ocf_total_macs_sum_tamper():
    """Would fail if a positive but inconsistent OCF MAC total were accepted."""

    arm = slim.D92_E0D_ARMS["E0_OCF25"]
    result = _result(arm)
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_ocf_support_alignment_macs_upper_bound"
    ] += 1
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="OCF"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


@pytest.mark.parametrize(
    "field",
    (
        "d92_e0d_ocf_support_alignment_affine_macs_upper_bound",
        "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound",
    ),
)
def test_audit_rejects_sum_consistent_ocf_k_formula_tamper(field):
    """Would fail if consistent positive fields could evade the frozen K formula."""

    arm = slim.D92_E0D_ARMS["E0_OCF25"]
    result = _result(arm)
    receipt = result.geometry_audit["final_covariance_audit"]
    receipt[field] += 1
    receipt["d92_e0d_ocf_support_alignment_macs_upper_bound"] += 1
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="OCF"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_audit_keeps_low_k_ocf_mac_receipt_inactive():
    """Would fail if the exact-full K2 lifecycle reported active OCF work."""

    arm = slim.D92_E0D_ARMS["E0_OCF25"]
    row = e0d_eval._audit_d92_e0d_fit(
        _result(arm, k_shot=2),
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=2,
        old_count=6,
        class_count=11,
    )
    assert row["d92_e0d_ocf_active"] is False
    assert row["d92_e0d_ocf_lambda"] is None
    assert row["d92_e0d_ocf_support_alignment_affine_macs_upper_bound"] is None
    assert (
        row["d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"]
        is None
    )
    assert row["d92_e0d_ocf_support_alignment_macs_upper_bound"] is None


@pytest.mark.parametrize(
    ("arm_id", "k_shot"),
    (("E0_FULL_ONLY", 5), ("E0_OCF25", 2)),
)
@pytest.mark.parametrize(
    "field",
    (
        "d92_e0d_ocf_support_alignment_affine_macs_upper_bound",
        "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound",
        "d92_e0d_ocf_support_alignment_macs_upper_bound",
    ),
)
def test_audit_rejects_inactive_nonzero_mac_receipt(arm_id, k_shot, field):
    """Would fail if an inactive arm carried any nonzero OCF MAC receipt."""

    arm = slim.D92_E0D_ARMS[arm_id]
    result = _result(arm, k_shot=k_shot)
    result.geometry_audit["final_covariance_audit"][field] = 1
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="OCF"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=k_shot,
            old_count=6,
            class_count=11,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("d92_e0d_ocf_lambda", 0.5),
        ("d92_e0d_ocf_support_alignment_affine_macs_upper_bound", -1),
        (
            "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound",
            float("nan"),
        ),
        ("d92_e0d_ocf_support_alignment_macs_upper_bound", None),
        ("d92_e0d_ocf_support_alignment_transient_bytes_upper_bound", float("nan")),
    ),
)
def test_audit_rejects_invalid_active_ocf_support_receipt(field, invalid_value):
    """Would fail if a malformed active OCF receipt reached formal analysis."""

    arm = slim.D92_E0D_ARMS["E0_OCF25"]
    result = _result(arm)
    result.geometry_audit["final_covariance_audit"][field] = invalid_value
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="OCF"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_audit_rejects_query_selection_access():
    arm = slim.D92_E0D_ARMS["D92_FULL"]
    result = _result(arm)
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_query_selection_access"
    ] = True
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="protocol"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_audit_rejects_missing_d81_transform_receipt():
    arm = slim.D92_E0D_ARMS["D92_FULL"]
    result = _result(arm)
    del result.geometry_audit["before_covariance_audit"]["d81_transform_audit"]
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="transform"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_evaluator_installs_arm_identity_and_restores_all_monkeypatches(monkeypatch):
    originals = (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    )
    observed = []

    def fake_builder(_d42, _basis, _weights, _audit, *, arm_id, **_kwargs):
        observed.append(arm_id)
        return "fit", [], []

    def fake_run(**_kwargs):
        fit, _, _ = d81_probe.build_d81_fit(None, None, None, {})
        assert fit == "fit"
        return {
            "candidate": d81_eval.CANDIDATE_D81,
            "schema": d81_eval.SCHEMA,
        }

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_builder)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FIXED50", **_allowed_kwargs()
    )
    assert result["candidate"] == "d92_e0d_e0_fixed50"
    assert result["arm_id"] == "E0_FIXED50"
    assert observed == ["E0_FIXED50"]
    assert (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    ) == originals


def test_evaluator_does_not_accept_truth_role_quota_or_score_arguments():
    with pytest.raises(TypeError):
        e0d_eval.run_d92_e0d_query_evaluation(
            arm_id="D92_FULL", truth_path="forbidden", **_allowed_kwargs()
        )


def test_tpce_evaluator_invokes_direct_state_postprocess_and_closes_receipt(
    monkeypatch,
):
    """Would fail if the production entry registered TPCE but never executed it."""

    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"]
    base_state = _tpce_state()
    rows, targets = _tpce_support()
    classes = tuple(base_state.classes)
    old_mask = targets < OLD_CLASS_COUNT
    registration_measure_active = {"value": False}
    original_measure = e0d_eval.measure_registration_call
    original_transform = d42._transform

    def tracked_measure(function):
        def measured_function():
            registration_measure_active["value"] = True
            try:
                return function()
            finally:
                registration_measure_active["value"] = False

        return original_measure(measured_function)

    def tracked_transform(*args, **kwargs):
        assert registration_measure_active["value"] is True
        return original_transform(*args, **kwargs)

    def fake_d42_fit(*_args, **_kwargs):
        base = _result(arm, k_shot=5)
        return d42.D42UnifiedShrinkageLDAResult(
            before_state=base.before_state,
            state=base_state,
            matched_fp32_before_state=base.before_state,
            matched_fp32_state=base_state,
            training_trace=tuple(base.training_trace),
            geometry_audit=base.geometry_audit,
            resource_audit=base.resource_audit,
        )

    def fake_run(**_kwargs):
        fitted = d81_eval.fit_d42_unified_shrinkage_lda(
            rows[old_mask],
            np.asarray(classes)[targets[old_mask]],
            classes[:OLD_CLASS_COUNT],
            rows[~old_mask],
            np.asarray(classes)[targets[~old_mask]],
            classes[OLD_CLASS_COUNT:],
        )
        row = d81_eval._audit_fit(
            fitted,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=len(classes),
        )
        return {
            "candidate": d81_eval.CANDIDATE_D81,
            "schema": d81_eval.SCHEMA,
            "tpce_fit_row": row,
        }

    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    monkeypatch.setattr(e0d_eval, "measure_registration_call", tracked_measure)
    monkeypatch.setattr(d42, "_transform", tracked_transform)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id=arm.arm_id, **_allowed_kwargs()
    )
    row = result["tpce_fit_row"]
    assert row["d92_e0d_tpce_active"] is True
    assert row["d92_e0d_tpce_fallback_active"] is False
    assert row["d92_e0d_tpce_final_state_sha256"] != row[
        "d92_e0d_tpce_e0_state_sha256"
    ]
    generated = row["d92_e0d_tpce_generated_atomic_exchange_count"]
    selected = row["d92_e0d_tpce_selected_atomic_exchange_count"]
    rejected = row["d92_e0d_tpce_rejected_atomic_exchange_count"]
    assert generated == selected + rejected
    assert selected == row["d92_e0d_tpce_greedy_step_count"]
    assert selected == row["d92_e0d_tpce_requested_atomic_exchange_count"]
    assert selected == row["d92_e0d_tpce_applied_atomic_exchange_count"]
    assert row["after_total_component_fit_count"] == 2
    assert row["after_actual_component_inventory"][
        "actual_component_fit_count"
    ] == 1
    assert row["after_state_postprocess_mode"] == "d42_tpce"
    assert row["query_fit_access"] is False
    assert row["query_truth_access"] is False


def test_tpce_query_audit_rejects_nonfinite_old_tail_gain():
    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"]
    state = _tpce_state()
    rows, targets = _tpce_support()
    from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as tpce

    candidate, base_receipt = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=OLD_CLASS_COUNT
    )
    result = _result(arm, k_shot=5)
    result.state = candidate
    result.geometry_audit["final_covariance_audit"].update(
        {
            key.replace("d92_tpce_", "d92_e0d_tpce_"): value
            for key, value in base_receipt.items()
        }
    )
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_tpce_old_tail_gain_by_class"
    ][0] = float("nan")
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="support guard"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_tpce_query_audit_rejects_greedy_subset_count_drift():
    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"]
    state = _tpce_state()
    rows, targets = _tpce_support()
    from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as tpce

    candidate, base_receipt = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=OLD_CLASS_COUNT
    )
    result = _result(arm, k_shot=5)
    result.state = candidate
    result.geometry_audit["final_covariance_audit"].update(
        {
            key.replace("d92_tpce_", "d92_e0d_tpce_"): value
            for key, value in base_receipt.items()
        }
    )
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_tpce_rejected_atomic_exchange_count"
    ] += 1
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="atomic"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_tcra_evaluator_runs_direct_postprocess_with_zero_query_access(
    monkeypatch,
):
    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"]
    base_state = _tpce_state()
    rows, targets = _tpce_support()
    classes = tuple(base_state.classes)
    old_mask = targets < OLD_CLASS_COUNT

    def fake_d42_fit(*_args, **_kwargs):
        base = _result(arm, k_shot=5)
        return d42.D42UnifiedShrinkageLDAResult(
            before_state=base.before_state,
            state=base_state,
            matched_fp32_before_state=base.before_state,
            matched_fp32_state=base_state,
            training_trace=tuple(base.training_trace),
            geometry_audit=base.geometry_audit,
            resource_audit=base.resource_audit,
        )

    def fake_run(**_kwargs):
        fitted = d81_eval.fit_d42_unified_shrinkage_lda(
            rows[old_mask],
            np.asarray(classes)[targets[old_mask]],
            classes[:OLD_CLASS_COUNT],
            rows[~old_mask],
            np.asarray(classes)[targets[~old_mask]],
            classes[OLD_CLASS_COUNT:],
        )
        row = d81_eval._audit_fit(
            fitted,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=len(classes),
        )
        return {
            "candidate": d81_eval.CANDIDATE_D81,
            "schema": d81_eval.SCHEMA,
            "tcra_fit_row": row,
        }

    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id=arm.arm_id, **_allowed_kwargs()
    )

    row = result["tcra_fit_row"]
    assert row["after_state_postprocess_mode"] == "d42_tcra"
    assert row["d92_e0d_tcra_active"] is True
    assert row["d92_e0d_tcra_component_fit_count"] == 0
    assert row["d92_e0d_tcra_modified_state_field_names"] == ["coef2_qint8"]
    assert row["d92_e0d_tcra_generated_atomic_ascent_count"] == row[
        "d92_e0d_tcra_selected_atomic_ascent_count"
    ] + row["d92_e0d_tcra_rejected_atomic_ascent_count"]
    assert row["d92_e0d_tcra_final_gate_revision"] == "safe_directional_v2"
    assert row["d92_e0d_tcra_old_tail_gain_sum"] == float(
        np.sum(
            np.asarray(row["d92_e0d_tcra_old_tail_gain_by_class"]),
            dtype=np.float64,
        )
    )
    assert row["d92_e0d_tcra_old_tail_strict_positive_count"] > 0
    assert row["d92_e0d_tcra_safe_directional_pass"] is True
    assert row["after_total_component_fit_count"] == 2
    assert row["after_actual_component_inventory"]["actual_component_fit_count"] == 1
    assert all(
        row[f"d92_e0d_tcra_query_{name}"] is False
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
    assert row["query_truth_access"] is False


@pytest.mark.parametrize("k_shot", [1, 2])
def test_tcra_query_audit_keeps_low_k_full_alias(k_shot):
    from cvsrffi import stage2_d92_d42_tail_class_row_ascent as tcra

    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"]
    state = _tpce_state()
    result = _result(arm, k_shot=k_shot)
    result.state = state
    result.geometry_audit["final_covariance_audit"].update(
        {
            key.replace("d92_tcra_", "d92_e0d_tcra_"): value
            for key, value in tcra.d42_tcra_inactive_receipt(state).items()
        }
    )
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=k_shot,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    assert row["after_state_postprocess_mode"] is None
    assert row["d92_e0d_tcra_active"] is False
    assert row["d92_e0d_tcra_fallback_active"] is False
    assert row["d92_e0d_tcra_fallback_reason"] == "K1_K2_EXACT_D92_FULL_ALIAS"
    assert row["d92_e0d_tcra_final_gate_revision"] == "safe_directional_v2"
    assert row["d92_e0d_tcra_old_tail_gain_sum"] is None
    assert row["d92_e0d_tcra_old_tail_strict_positive_count"] is None
    assert row["d92_e0d_tcra_safe_directional_pass"] is False


def test_tcra_query_audit_rejects_query_access_tamper():
    from cvsrffi import stage2_d92_d42_tail_class_row_ascent as tcra

    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"]
    state = _tpce_state()
    rows, targets = _tpce_support()
    candidate, receipt = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=OLD_CLASS_COUNT
    )
    result = _result(arm, k_shot=5)
    result.state = candidate
    result.geometry_audit["final_covariance_audit"].update(
        {
            key.replace("d92_tcra_", "d92_e0d_tcra_"): value
            for key, value in receipt.items()
        }
    )
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_tcra_query_selection_access"
    ] = True
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="TCRA"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("final_gate_revision", "strict_positive_v1"),
        ("old_tail_gain_sum", -1.0),
        ("old_tail_strict_positive_count", 0),
        ("safe_directional_pass", False),
    ],
)
def test_tcra_query_audit_rejects_safe_directional_receipt_tamper(
    field, tampered
):
    from cvsrffi import stage2_d92_d42_tail_class_row_ascent as tcra

    arm = slim.D92_E0D_ARMS["E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"]
    state = _tpce_state()
    rows, targets = _tpce_support()
    candidate, receipt = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=OLD_CLASS_COUNT
    )
    tolerance = receipt["d92_tcra_guard_tolerance"]
    old_gains = receipt["d92_tcra_old_tail_gain_by_class"]
    receipt.update(
        {
            "d92_tcra_final_gate_revision": "safe_directional_v2",
            "d92_tcra_old_tail_gain_sum": float(
                np.sum(np.asarray(old_gains), dtype=np.float64)
            ),
            "d92_tcra_old_tail_strict_positive_count": sum(
                value > tolerance for value in old_gains
            ),
            "d92_tcra_safe_directional_pass": True,
        }
    )
    result = _result(arm, k_shot=5)
    result.state = candidate
    result.geometry_audit["final_covariance_audit"].update(
        {
            key.replace("d92_tcra_", "d92_e0d_tcra_"): value
            for key, value in receipt.items()
        }
    )
    result.geometry_audit["final_covariance_audit"][
        f"d92_e0d_tcra_{field}"
    ] = tampered

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="TCRA"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_newguard_query_audit_requires_support_only_protection_receipt():
    """Would fail if formal NewGuard rows omitted its deployed-head protection proof."""

    arm = slim.D92_E0D_ARMS["E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"]
    result = _result(arm)
    receipt = result.geometry_audit["final_covariance_audit"]
    receipt.update(
        {
            "d92_e0d_newguard_active": True,
            "d92_e0d_newguard_fallback_active": False,
            "d92_e0d_newguard_fallback_reason": None,
            "d92_e0d_newguard_full_component_fit_count": 1,
            "d92_e0d_newguard_new_rows_byte_exact": True,
            "d92_e0d_newguard_deployment_new_rows_byte_exact": True,
            "d92_e0d_newguard_tau_old_envelope_shift": -0.01,
            "d92_e0d_newguard_deployment_protection_pass": True,
            "d92_e0d_newguard_protection_tolerance": (
                1024.0 * np.finfo(np.float32).eps
            ),
            "d92_e0d_newguard_closure_tolerance": 1.0e-5,
            "d92_e0d_newguard_max_abs_Xnew_internal_residual": 0.0,
            "d92_e0d_newguard_old_group_zero_sum_residual_max_abs": 0.0,
            "d92_e0d_newguard_new_support_old_envelope_change_max_abs_error": 0.0,
            "d92_e0d_newguard_deployment_max_abs_Xnew_internal_residual": 0.0,
            "d92_e0d_newguard_deployment_old_group_zero_sum_residual_max_abs": 0.0,
            "d92_e0d_newguard_deployment_new_support_old_envelope_change_max_abs_error": 0.0,
            "d92_e0d_newguard_new_support_old_envelope_change_max": -0.01,
            "d92_e0d_newguard_deployment_new_support_old_envelope_change_max": -0.01,
            "d92_e0d_newguard_deployment_new_support_min_margin_change": 0.01,
            "d92_e0d_newguard_deployment_strength_scale": 1.0,
            "d92_e0d_newguard_deployment_candidate_count": 1,
            "d92_e0d_newguard_deployment_full_head_byte_exact": False,
            "d92_e0d_newguard_deployment_codec_roundtrip_count": 2,
            "d92_e0d_newguard_deployment_codec_macs_upper_bound": 50_688,
            "d92_e0d_newguard_persistent_state_bytes_delta": 0,
        }
    )
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["d92_e0d_newguard_full_component_fit_count"] == 1
    assert row["d92_e0d_newguard_deployment_protection_pass"] is True
    assert row["d92_e0d_newguard_deployment_strength_scale"] == 1.0
    assert row["d92_e0d_newguard_deployment_candidate_count"] == 1
    assert row["d92_e0d_newguard_deployment_full_head_byte_exact"] is False
    assert row["d92_e0d_newguard_deployment_codec_roundtrip_count"] == 2
    assert row["d92_e0d_newguard_deployment_codec_macs_upper_bound"] == 50_688


def test_pareto_query_audit_closes_4_2_receipt_and_rejects_query_access_tamper():
    """Would fail if formal Pareto rows accepted query access or bad 4/2 state."""

    arm = slim.D92_E0D_ARMS["E0_FULL_BLOCK_PARETO_DISTILL"]
    result = _result(arm)
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_rain_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["after_total_component_fit_count"] == 4
    assert row["query_macs"] == 11 * 288
    assert row["d92_e0d_pareto_distill_active"] is True
    assert row["d92_e0d_pareto_distill_deployed_support_constraints_pass"] is True
    assert row["d92_e0d_pareto_distill_deployed_full_head_byte_exact"] is False
    assert row[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
    ] == 0.125
    assert (
        row["d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"]
        >= row["d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"]
    )
    assert row["d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"] is True
    assert row["d92_e0d_pareto_distill_deployed_head_state_closure_pass"] is True
    assert row["d92_e0d_pareto_distill_deployed_head_state_affine_sha256"] == (
        _deployed_affine_sha256(result.state)
    )

    tampered = _result(arm)
    tampered.geometry_audit["final_covariance_audit"][
        "d92_e0d_pareto_distill_query_selection_access"
    ] = True
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="Pareto Distill query"):
        e0d_eval._audit_d92_e0d_fit(
            tampered,
            arm=arm,
            scenario="leo_rain_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )

    quantum_tampered = _result(arm)
    quantum_tampered.geometry_audit["final_covariance_audit"][
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
    ] = 0.01
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="quantum"):
        e0d_eval._audit_d92_e0d_fit(
            quantum_tampered,
            arm=arm,
            scenario="leo_rain_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_pareto_query_audit_rejects_final_deployed_head_hash_tamper():
    """Would fail if a D42 final state could diverge from the previewed head."""

    arm = slim.D92_E0D_ARMS["E0_FULL_BLOCK_PARETO_DISTILL"]
    result = _result(arm)
    result.geometry_audit["final_covariance_audit"][
        "d92_e0d_pareto_distill_deployed_candidate_affine_sha256"
    ] = "0" * 64
    with pytest.raises(
        e0d_eval.D92E0DQueryEvaluationError, match="Pareto Distill deployed"
    ):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_rain_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_pareto_query_audit_uses_e0_reference_for_numeric_fallback():
    """Would fail if an exact-E0 fallback skipped its deployed reference closure."""

    arm = slim.D92_E0D_ARMS["E0_FULL_BLOCK_PARETO_DISTILL"]
    result = _result(arm)
    receipt = result.geometry_audit["final_covariance_audit"]
    receipt.update(
        {
            "d92_e0d_pareto_distill_active": False,
            "d92_e0d_pareto_distill_fallback_active": True,
            "d92_e0d_pareto_distill_fallback_reason": "numeric_fallback",
            "d92_e0d_pareto_distill_local_valid": False,
            "d92_e0d_pareto_distill_full_head_byte_exact": True,
            "d92_e0d_pareto_distill_deployed_support_constraints_pass": False,
            "d92_e0d_pareto_distill_deployed_full_head_byte_exact": True,
            "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs": 0.01,
            "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum": 0.125,
            "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass": False,
            "d92_e0d_pareto_distill_deployed_candidate_affine_sha256": None,
            "d92_e0d_pareto_distill_deployed_e0_affine_sha256": (
                _deployed_affine_sha256(result.state)
            ),
        }
    )
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_rain_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["d92_e0d_pareto_distill_deployed_head_state_closure_pass"] is True
    assert row["d92_e0d_pareto_distill_deployed_head_state_affine_sha256"] == (
        _deployed_affine_sha256(result.state)
    )


def test_pareto_query_audit_keeps_low_k_alias_reason():
    """Would fail if K1 alias evidence lost its frozen lifecycle reason."""

    arm = slim.D92_E0D_ARMS["E0_FULL_BLOCK_PARETO_DISTILL"]
    row = e0d_eval._audit_d92_e0d_fit(
        _result(arm, k_shot=1),
        arm=arm,
        scenario="leo_rain_weak",
        k_shot=1,
        old_count=6,
        class_count=11,
    )
    assert row["d92_e0d_pareto_distill_fallback_reason"] == (
        "K1_K2_EXACT_D92_FULL_ALIAS"
    )
    assert row[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
    ] is None
    assert row["d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"] is None
    assert row["d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"] is None


def _ccoc_result(*, fallback: bool = False):
    arm = slim.D92_E0D_ARMS["E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"]
    result = _result(arm, k_shot=10)
    result.geometry_audit["before_covariance_audit"].update(
        _ccoc_receipt(registered=False, class_count=OLD_CLASS_COUNT, k_shot=10)
    )
    result.geometry_audit["final_covariance_audit"].update(
        _ccoc_receipt(
            registered=True,
            class_count=11,
            k_shot=10,
            fallback=fallback,
        )
    )
    if fallback:
        after = result.geometry_audit["final_covariance_audit"]
        after["d92_e0d_actual_component_fit_count"] = 2
        after["d92_e0d_total_component_fit_count"] = 3
        after["d92_e0d_actual_component_inventory"] = {
            "schema": "cvs.phase2.d92.actual_component_fit_inventory.v1",
            "actual_component_fit_count": 2,
            "actual_component_calls": [
                {"arm": "full", "status": "ccoc_numeric_attempt"},
                {"arm": "full", "status": "ccoc_exact_e0_reference"},
            ],
        }
    return arm, result


def test_ccoc_query_audit_closes_complete_support_receipt():
    """Would fail if CCOC active rows omitted rho, SPD, inventory, or no-query proof."""

    arm, result = _ccoc_result()
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=10,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    assert row["d92_e0d_ccoc_active"] is True
    assert row["d92_e0d_ccoc_g0_eligible"] is True
    assert row["d92_e0d_ccoc_old_rho"] == pytest.approx(0.25)
    assert row["d92_e0d_ccoc_new_rho"] == pytest.approx(0.75)
    assert row["d92_e0d_ccoc_dense_solve_count"] == 1
    assert row["d92_e0d_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound"] == (
        334_336
    )
    assert row["d92_e0d_ccoc_query_global_reassignment"] is False


def test_ccoc_query_audit_recomputes_active_covariance_mix_workspace_receipt():
    """Would fail if Query accepted a non-v7 CCOC covariance-mix receipt."""

    arm, result = _ccoc_result()
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=10,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    expected = {
        "d92_e0d_ccoc_workspace_candidate_covariance_result_bytes": 288 * 288 * 8,
        "d92_e0d_ccoc_workspace_candidate_covariance_block_workspace_bytes": (
            160 * 160 * 8
        ),
        "d92_e0d_ccoc_workspace_candidate_covariance_row_workspace_bytes": 160
        * 8,
        "d92_e0d_ccoc_workspace_candidate_covariance_full_buffer_count_upper_bound": 1,
        "d92_e0d_ccoc_workspace_candidate_covariance_mix_live_bytes_upper_bound": (
            288 * 288 * 8 + 160 * 160 * 8 + 160 * 8
        ),
    }
    assert {field: row[field] for field in expected} == expected


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("d92_ccoc_old_rho", 1.5),
        ("d92_ccoc_additional_fit_count", 1),
        ("d92_ccoc_canonicalization", "input_row_order"),
        ("d92_ccoc_dense_solve_count", 2),
        ("d92_ccoc_persistent_state_bytes_delta", 1),
        ("d92_ccoc_query_macs_delta", 1),
        ("d92_ccoc_query_selection_access", True),
    ),
)
def test_ccoc_query_audit_rejects_rho_inventory_and_query_tamper(field, tampered):
    """Would fail if a raw CCOC receipt could diverge from its mirrored closure."""

    arm, result = _ccoc_result()
    result.geometry_audit["final_covariance_audit"][field] = tampered
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize("delta", (-1, 1))
def test_ccoc_query_audit_rejects_frozen_k10_workspace_byte_drift(delta):
    """Would fail if Query accepted either side of the frozen 334336-byte bound."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    raw_field = "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound"
    mirror_field = raw_field.replace("d92_ccoc_", "d92_e0d_ccoc_")
    audit[raw_field] = int(audit[raw_field]) + delta
    audit[mirror_field] = int(audit[mirror_field]) + delta

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize(
    ("suffix", "tampered"),
    (
        ("candidate_covariance_result_bytes", 288 * 288 * 8 - 8),
        ("candidate_covariance_block_workspace_bytes", 0),
        ("candidate_covariance_row_workspace_bytes", 1.5),
        ("candidate_covariance_full_buffer_count_upper_bound", 2),
        ("candidate_covariance_mix_live_bytes_upper_bound", 1),
    ),
)
def test_ccoc_query_audit_rejects_covariance_mix_workspace_tamper(
    suffix, tampered
):
    """Would fail if Query did not independently recompute v7 mix resources."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    raw_field = f"d92_ccoc_workspace_{suffix}"
    mirror_field = raw_field.replace("d92_ccoc_", "d92_e0d_ccoc_")
    audit[raw_field] = tampered
    audit[mirror_field] = tampered

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_ccoc_query_audit_rejects_missing_covariance_mix_workspace_fields():
    """Would fail if a v7 field pair could disappear without rejection."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    raw_field = "d92_ccoc_workspace_candidate_covariance_result_bytes"
    mirror_field = raw_field.replace("d92_ccoc_", "d92_e0d_ccoc_")
    audit.pop(raw_field)
    audit.pop(mirror_field)

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize("removed_prefix", ("d92_ccoc_", "d92_e0d_ccoc_"))
def test_ccoc_query_audit_rejects_single_sided_covariance_mix_workspace_mirror(
    removed_prefix,
):
    """Would fail if an approved v7 field could lose one side of its mirror."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    suffix = "workspace_candidate_covariance_result_bytes"
    audit.pop(f"{removed_prefix}{suffix}")

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize(
    ("raw_field", "mirror_field", "value"),
    (
        (
            "d92_ccoc_unknown_receipt_field",
            "d92_e0d_ccoc_unknown_receipt_field",
            "forged",
        ),
        (
            "d92_ccoc_raw_support_rows",
            "d92_e0d_ccoc_raw_support_rows",
            ((0.0, 1.0),),
        ),
    ),
)
def test_ccoc_query_audit_rejects_unknown_or_raw_support_receipt_fields(
    raw_field, mirror_field, value
):
    """Would fail if unapproved CCOC fields crossed the fit-audit boundary."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    audit[raw_field] = value
    audit[mirror_field] = value

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


@pytest.mark.parametrize(
    ("raw_field", "mirror_field", "value"),
    (
        ("d92_ccoc_raw_support_rows", None, ((0.0, 1.0),)),
        (None, "d92_e0d_ccoc_raw_support_rows", ((0.0, 1.0),)),
    ),
)
def test_ccoc_query_audit_rejects_single_sided_or_missing_mirror_fields(
    raw_field, mirror_field, value
):
    """Would fail if a CCOC raw/mirror collection could be one-sided."""

    arm, result = _ccoc_result()
    audit = result.geometry_audit["final_covariance_audit"]
    if raw_field is not None:
        audit[raw_field] = value
    if mirror_field is not None:
        audit[mirror_field] = value

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_ccoc_query_audit_rejects_missing_mirrored_statistic_field():
    """Would fail if an approved raw field could omit its mirrored closure."""

    arm, result = _ccoc_result()
    result.geometry_audit["final_covariance_audit"].pop("d92_e0d_ccoc_old_rho")

    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_ccoc_query_audit_accepts_core_numeric_e0_fallback_but_blocks_g0():
    """Would fail if a two-after-fit CCOC fallback were treated as active G0."""

    arm, result = _ccoc_result(fallback=True)
    row = e0d_eval._audit_d92_e0d_fit(
        result,
        arm=arm,
        scenario="leo_clear_weak",
        k_shot=10,
        old_count=OLD_CLASS_COUNT,
        class_count=11,
    )
    assert row["after_total_component_fit_count"] == 3
    assert row["d92_e0d_ccoc_active"] is False
    assert row["d92_e0d_ccoc_fallback_active"] is True
    assert row["d92_e0d_ccoc_g0_eligible"] is False
    assert row["d92_e0d_ccoc_g0_block_reason"] == "NUMERIC_FALLBACK_EXACT_E0"


def test_ccoc_codec_numeric_error_retries_once_with_e0_reference_builder(
    monkeypatch,
):
    """Would fail if a registered-after CCOC codec error retried the candidate."""

    build_arms: list[str] = []

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        return d42._compile_state(
            tuple(f"tx_{index}" for index in range(11)),
            OLD_CLASS_COUNT,
            None,
            None,
            None,
            "frozen",
            precision="int8",
        )

    def overflow_compile(*_args, **_kwargs):
        raise d42.D42UnifiedShrinkageLDAError("D42 intercept FP16 overflow")

    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        d81_probe.build_d81_fit(None, None, None, {})
        if calls == 1:
            d81_eval.fit_d42_unified_shrinkage_lda()
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d42, "_compile_state", overflow_compile)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS", **_allowed_kwargs()
    )
    assert calls == 2
    assert build_arms == [
        "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        "E0_FULL_ONLY",
    ]
    assert result["d92_e0d_ccoc_codec_numeric_fallback"] is True
    assert result["d92_e0d_ccoc_codec_fallback_component_execution_count"] == 1


def test_ccoc_codec_fallback_mirrors_live_ledger_and_closes_query_audit(
    monkeypatch,
):
    """Would fail if the real retry ledger had raw-only codec receipt fields."""

    arm = slim.D92_E0D_ARMS["E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"]
    reference_result = _result(slim.D92_E0D_ARMS["E0_FULL_ONLY"], k_shot=10)
    reference_before = dict(
        reference_result.geometry_audit["before_covariance_audit"]
    )
    reference_after = dict(
        reference_result.geometry_audit["final_covariance_audit"]
    )
    retry_template = _result(arm, k_shot=10)
    build_arms: list[str] = []
    audited_rows: list[dict] = []
    retry_results: list[SimpleNamespace] = []

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)

        def reference_fit(_rows, _labels, class_count, k_shot):
            source = (
                reference_before
                if int(class_count) == OLD_CLASS_COUNT
                else reference_after
            )
            return (
                np.zeros((int(class_count), d42.FEATURE_DIM), dtype=np.float32),
                np.zeros(int(class_count), dtype=np.float32),
                dict(source),
            )

        return reference_fit, [], []

    def fake_d42_fit(*_args, **_kwargs):
        return d42._compile_state(
            tuple(f"tx_{index}" for index in range(11)),
            OLD_CLASS_COUNT,
            None,
            None,
            None,
            "frozen",
            precision="int8",
        )

    def overflow_compile(*_args, **_kwargs):
        raise d42.D42UnifiedShrinkageLDAError("D42 intercept FP16 overflow")

    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        fit, _, _ = d81_probe.build_d81_fit(None, None, None, {})
        if calls == 1:
            return d81_eval.fit_d42_unified_shrinkage_lda()
        before = fit(
            np.zeros((OLD_CLASS_COUNT * 10, d42.FEATURE_DIM), dtype=np.float32),
            np.zeros(OLD_CLASS_COUNT * 10, dtype=np.int64),
            OLD_CLASS_COUNT,
            10,
        )
        after = fit(
            np.zeros((11 * 10, d42.FEATURE_DIM), dtype=np.float32),
            np.zeros(11 * 10, dtype=np.int64),
            11,
            10,
        )
        retry_result = retry_template
        retry_result.geometry_audit["before_covariance_audit"] = before[2]
        retry_result.geometry_audit["final_covariance_audit"] = after[2]
        retry_results.append(retry_result)
        audited_rows.append(
            d81_eval._audit_fit(
                retry_result,
                scenario="leo_clear_weak",
                k_shot=10,
                old_count=OLD_CLASS_COUNT,
                class_count=11,
            )
        )
        return {
            "candidate": d81_eval.CANDIDATE_D81,
            "schema": d81_eval.SCHEMA,
        }

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d42, "_compile_state", overflow_compile)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id=arm.arm_id,
        **_allowed_kwargs(),
    )

    assert build_arms == [arm.arm_id, "E0_FULL_ONLY"]
    assert result["d92_e0d_ccoc_codec_numeric_fallback"] is True
    assert len(audited_rows) == 1
    row = audited_rows[0]
    after = retry_results[0].geometry_audit["final_covariance_audit"]
    assert after["d92_ccoc_codec_fallback_component_execution_count"] == 3
    assert after["d92_e0d_ccoc_codec_fallback_component_execution_count"] == 3
    assert after["d92_e0d_ccoc_codec_fallback_scope"] == (
        "whole_d42_retry_before_and_after"
    )
    assert row["d92_e0d_ccoc_codec_fallback_component_execution_count"] == 3
    assert row["d92_e0d_ccoc_codec_fallback_scope"] == (
        "whole_d42_retry_before_and_after"
    )
    assert "d92_ccoc_codec_fallback_component_execution_count" not in row
    assert row["d92_e0d_ccoc_g0_eligible"] is False
    assert row["d92_e0d_ccoc_g0_block_reason"] == "NUMERIC_FALLBACK_EXACT_E0"

    after["d92_e0d_ccoc_codec_fallback_component_execution_count"] = 4
    with pytest.raises(e0d_eval.D92E0DQueryEvaluationError, match="CCOC"):
        e0d_eval._audit_d92_e0d_fit(
            retry_results[0],
            arm=arm,
            scenario="leo_clear_weak",
            k_shot=10,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )


def test_ccoc_technical_support_receipt_sink_is_support_only_and_state_stable(
    monkeypatch,
):
    """Would fail if the optional CCOC sink persisted support or read query state."""

    captured: list[dict] = []
    state = _state(classes=11, old_count=OLD_CLASS_COUNT, marker=3)
    before_fingerprint = e0d_eval._state_fingerprint_sha256(state)
    result_object = SimpleNamespace(state=state)
    old_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT))
    new_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT, 11))
    rows, targets = _tpce_support(class_count=11, k_shot=3)

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        assert arm_id == "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        return result_object

    def fake_audit(_result, **_kwargs):
        return {"support_only": True}

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        fitted = d81_eval.fit_d42_unified_shrinkage_lda(
            rows[targets < OLD_CLASS_COUNT],
            np.asarray(
                [old_classes[index] for index in targets[targets < OLD_CLASS_COUNT]]
            ),
            old_classes,
            rows[targets >= OLD_CLASS_COUNT],
            np.asarray(
                [
                    new_classes[index - OLD_CLASS_COUNT]
                    for index in targets[targets >= OLD_CLASS_COUNT]
                ]
            ),
            new_classes,
        )
        assert fitted is result_object
        assert e0d_eval._state_fingerprint_sha256(state) == before_fingerprint
        d81_eval._audit_fit(
            fitted,
            scenario="leo_clear_weak",
            k_shot=3,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(e0d_eval, "_audit_d92_e0d_fit", fake_audit)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        technical_support_receipt_sink=captured.append,
        **_allowed_kwargs(),
    )

    assert len(captured) == 1
    receipt = captured[0]
    assert receipt["schema"] == "cvs.phase2.d92_ccoc.support_state_receipt.v1"
    assert receipt["scene"] == "leo_clear_weak"
    assert receipt["after_state_fingerprint_sha256"] == before_fingerprint
    assert receipt["query_access"] is False
    assert receipt["truth_access"] is False
    assert len(receipt["canonical_support_handles"]) == len(rows)
    assert len(receipt["cross_group_margin_by_support_handle"]) == len(rows)
    assert set(receipt["support_block_absmax"]) == {"z160", "fft96", "rf32"}
    assert len(receipt["scale1_block_max_abs"]) == 3
    assert len(receipt["scale2_block_max_abs"]) == 3
    assert set(receipt).isdisjoint(
        {"support_features", "query_features", "query_tokens", "truth"}
    )


def test_ccoc_none_sink_skips_technical_support_capture(monkeypatch):
    """The default sink must add no support receipt work to a CCOC evaluation."""

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        assert arm_id == "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        return SimpleNamespace()

    def unexpected_capture(*_args, **_kwargs):
        raise AssertionError("default None sink must not capture support")

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        d81_eval.fit_d42_unified_shrinkage_lda()
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(e0d_eval, "_final_state_technical_support_receipt", unexpected_capture)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS", **_allowed_kwargs()
    )

    assert result["candidate"] == "d92_e0_full_cross_class_offblock_consensus"


def test_e0_technical_support_receipt_sink_uses_native_final_fit_without_ccoc_side_effects(
    monkeypatch,
):
    """E0 receipt capture must not enter any CCOC fit, codec, or mirror path."""

    captured: list[dict] = []
    state = _state(classes=11, old_count=OLD_CLASS_COUNT, marker=7)
    fingerprint = e0d_eval._state_fingerprint_sha256(state)
    result_object = SimpleNamespace(state=state)
    old_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT))
    new_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT, 11))
    rows, targets = _tpce_support(class_count=11, k_shot=3)
    build_arms: list[str] = []
    fit_calls = 0
    original_ccoc_arms = e0d_eval._CCOC_ARM_IDS

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return result_object

    def fake_audit(_result, **_kwargs):
        return {"e0_native_fit_count": fit_calls}

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        fitted = d81_eval.fit_d42_unified_shrinkage_lda(
            rows[targets < OLD_CLASS_COUNT],
            np.asarray(
                [old_classes[index] for index in targets[targets < OLD_CLASS_COUNT]]
            ),
            old_classes,
            rows[targets >= OLD_CLASS_COUNT],
            np.asarray(
                [
                    new_classes[index - OLD_CLASS_COUNT]
                    for index in targets[targets >= OLD_CLASS_COUNT]
                ]
            ),
            new_classes,
        )
        assert fitted is result_object
        assert e0d_eval._state_fingerprint_sha256(state) == fingerprint
        audit = d81_eval._audit_fit(
            fitted,
            scenario="leo_clear_weak",
            k_shot=3,
            old_count=OLD_CLASS_COUNT,
            class_count=11,
        )
        assert audit == {"e0_native_fit_count": 1}
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(e0d_eval, "_audit_d92_e0d_fit", fake_audit)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_ONLY",
        technical_support_receipt_sink=captured.append,
        **_allowed_kwargs(),
    )

    assert build_arms == ["E0_FULL_ONLY"]
    assert fit_calls == 1
    assert e0d_eval._CCOC_ARM_IDS == original_ccoc_arms
    assert set(result).isdisjoint(
        {
            "d92_ccoc_active",
            "d92_ccoc_fallback_active",
            "d92_e0d_ccoc_active",
            "d92_e0d_ccoc_fallback_active",
            "d92_e0d_ccoc_codec_numeric_fallback",
        }
    )
    assert len(captured) == 1
    receipt = captured[0]
    assert receipt["schema"] == "cvs.phase2.d92_e0d.support_state_receipt.v1"
    assert receipt["arm_id"] == "E0_FULL_ONLY"
    assert receipt["after_state_fingerprint_sha256"] == fingerprint
    assert receipt["final_d42_coefficient_bias_state_sha256"] == fingerprint
    assert receipt["query_access"] is False
    assert receipt["truth_access"] is False
    assert set(receipt).isdisjoint(
        {
            "support_features",
            "query_features",
            "query_tokens",
            "truth",
            "d92_ccoc_active",
            "d92_e0d_ccoc_active",
            "d92_e0d_ccoc_codec_numeric_fallback",
        }
    )


@pytest.mark.parametrize(
    "message",
    ("D42 state drift", "D42 intercept FP16 overflow"),
)
def test_e0_technical_sink_preserves_native_errors_without_ccoc_fallback(
    monkeypatch, message: str
):
    """E0 structural and codec errors must stay on the original E0 error path."""

    old_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT))
    new_classes = tuple(f"tx_{index}" for index in range(OLD_CLASS_COUNT, 11))
    rows, targets = _tpce_support(class_count=11, k_shot=3)
    build_arms: list[str] = []
    original_ccoc_arms = e0d_eval._CCOC_ARM_IDS

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        build_arms.append(arm_id)
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        raise d42.D42UnifiedShrinkageLDAError(message)

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        return d81_eval.fit_d42_unified_shrinkage_lda(
            rows[targets < OLD_CLASS_COUNT],
            np.asarray(
                [old_classes[index] for index in targets[targets < OLD_CLASS_COUNT]]
            ),
            old_classes,
            rows[targets >= OLD_CLASS_COUNT],
            np.asarray(
                [
                    new_classes[index - OLD_CLASS_COUNT]
                    for index in targets[targets >= OLD_CLASS_COUNT]
                ]
            ),
            new_classes,
        )

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    with pytest.raises(d42.D42UnifiedShrinkageLDAError, match=message):
        e0d_eval.run_d92_e0d_query_evaluation(
            arm_id="E0_FULL_ONLY",
            technical_support_receipt_sink=lambda _receipt: None,
            **_allowed_kwargs(),
        )

    assert build_arms == ["E0_FULL_ONLY"]
    assert e0d_eval._CCOC_ARM_IDS == original_ccoc_arms


def test_e0_none_sink_skips_support_receipt_capture(monkeypatch):
    """The default E0 call must not add a support transform or receipt capture."""

    fit_calls = 0

    def fake_build(_d42, _basis, _weights, _ground_audit, *, arm_id):
        assert arm_id == "E0_FULL_ONLY"
        return (lambda *_args: (None, None, {})), [], []

    def fake_d42_fit(*_args, **_kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return SimpleNamespace()

    def unexpected_transform(*_args, **_kwargs):
        raise AssertionError("default None sink must not transform support")

    def fake_run(**_kwargs):
        d81_probe.build_d81_fit(None, None, None, {})
        d81_eval.fit_d42_unified_shrinkage_lda()
        return {"candidate": d81_eval.CANDIDATE_D81, "schema": d81_eval.SCHEMA}

    monkeypatch.setattr(e0d_eval, "build_d92_e0d_fit", fake_build)
    monkeypatch.setattr(d81_eval, "fit_d42_unified_shrinkage_lda", fake_d42_fit)
    monkeypatch.setattr(d42, "_transform", unexpected_transform)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = e0d_eval.run_d92_e0d_query_evaluation(
        arm_id="E0_FULL_ONLY", **_allowed_kwargs()
    )

    assert result["candidate"] == "d92_e0d_e0_full_only"
    assert fit_calls == 1
