from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d81_query_evaluation as d81_eval
from cvsrffi import stage2_d92_e0d_query_evaluation as e0d_eval
from cvsrffi import stage2_d92_e0d_slim as slim
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
    effective_mode = arm.registered_d_mode if active else "d92_full_alias"
    total = (
        slim.expected_total_component_fit_count(k_shot, arm_id=arm.arm_id)
        if registered and k_shot > 2
        else 1
    )
    actual = total // 2 if registered and k_shot > 2 else 1
    class_count = 11 if registered else 6
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
                else [4.42187]
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


def _result(arm: slim.D92E0DSlimArmSpec, *, after_marker: int = 2):
    return SimpleNamespace(
        geometry_audit={
            "k1_unit_covariance_fallback": False,
            "before_covariance_audit": _resource(
                arm, registered=False, k_shot=5
            ),
            "final_covariance_audit": _resource(arm, registered=True, k_shot=5),
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
