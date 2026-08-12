from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d81_query_evaluation as d81_eval
from cvsrffi import stage2_d92_e0d_query_evaluation as e0d_eval
from cvsrffi import stage2_d92_e0d_slim as slim
from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe


def _state(*, classes: int, old_count: int, marker: int):
    return SimpleNamespace(
        schema="cvs.phase2.d42.unified_shrinkage_lda_state.v1",
        classes=tuple(f"tx_{index}" for index in range(classes)),
        old_class_count=old_count,
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
        log_diag_fp32=np.full(288, marker, dtype=np.float32),
        coef1_qint8=np.full((classes, 288), marker, dtype=np.int8),
        coef2_qint8=np.full((classes, 288), -marker, dtype=np.int8),
        scale1_fp16=np.full((classes, 3), marker + 1, dtype=np.float16),
        scale2_fp16=np.full((classes, 3), marker + 2, dtype=np.float16),
        intercept_fp16=np.full(classes, marker + 3, dtype=np.float16),
        coef_fp32=np.full((classes, 288), marker + 4, dtype=np.float32),
        intercept_fp32=np.full(classes, marker + 5, dtype=np.float32),
        persistent_state_bytes=classes * 100,
    )


def _resource(
    arm: slim.D92E0DSlimArmSpec, *, registered: bool, k_shot: int
) -> dict:
    active = registered and k_shot > 2 and not arm.e_enabled
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
    return {
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
        "d92_e0d_newguard_tau_old_envelope_shift": (
            -0.01 if newguard_active else 0.0
        ),
        "d92_e0d_newguard_deployment_protection_pass": newguard_active,
        "d92_e0d_newguard_full_head_byte_exact": not newguard_active,
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
    return SimpleNamespace(
        geometry_audit={
            "k1_unit_covariance_fallback": False,
            "before_covariance_audit": _resource(
                arm, registered=False, k_shot=k_shot
            ),
            "final_covariance_audit": _resource(
                arm, registered=True, k_shot=k_shot
            ),
        },
        before_state=_state(classes=6, old_count=6, marker=1),
        state=_state(classes=11, old_count=6, marker=after_marker),
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
