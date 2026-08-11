from __future__ import annotations

import json
from pathlib import Path

from cvsrffi.stage2_d92_floorboost_hard11_analysis import (
    HISTORICAL_BASELINE_SHA256,
    _floorboost_receipt,
    _fit_resource,
    decide_verdict,
)


QUERY_ZERO_FIELDS = (
    "query_truth_access",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)


def test_historical_baseline_identity_is_frozen() -> None:
    path = Path(
        r"E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv"
    )
    assert HISTORICAL_BASELINE_SHA256 == "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
    assert path.is_file()


def test_verdict_branches_keep_floor_and_forgetting_distinct() -> None:
    assert decide_verdict({"complete_artifact_closure": True, "performance_outer_closure": True, "all_advance_core": True, "all_forgetting": True, "revision_gate_passed": False, "hard_reject": False}) == "ADVANCE_TO_FULL125"
    assert decide_verdict({"complete_artifact_closure": True, "performance_outer_closure": True, "all_advance_core": False, "all_forgetting": False, "revision_gate_passed": True, "hard_reject": False}) == "REVISE_ONCE_FLOORBOOST"
    assert decide_verdict({"complete_artifact_closure": True, "performance_outer_closure": True, "all_advance_core": False, "all_forgetting": True, "revision_gate_passed": False, "hard_reject": True}) == "REJECT_FLOORBOOST"


def test_k1_receipt_is_exact_d92_full_alias(tmp_path: Path) -> None:
    root = tmp_path / "job" / "diag" / "after"
    root.mkdir(parents=True)
    rows = []
    for scene in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        row = {
            "scenario": scene,
            "arm_id": "E0_FULL_MAXMIN_FLOORBOOST",
            "candidate_id": "d92_e0_full_maxmin_floorboost",
            "after_total_component_fit_count": 3,
            "after_actual_component_inventory": {"actual_component_fit_count": 3},
            "query_macs": 7488,
            "after_state_bytes": 18503,
            "after_registered_d_mode_effective": "d92_full_alias",
            "after_registration_resource": {
                "registration_wall_time_ns": 1.0,
                "registration_process_cpu_time_ns": 2.0,
                "registration_incremental_peak_working_set_bytes": 3.0,
            },
            "d92_e0d_floorboost_active": False,
            # The exact K1/K2 alias does not produce floorboost statistics.
            "d92_e0d_floorboost_lambda": None,
            "d92_e0d_floorboost_quantile": None,
            "d92_e0d_floorboost_quantile_method": None,
            "d92_e0d_floorboost_kappa": None,
            "d92_e0d_floorboost_fallback_active": False,
            "d92_e0d_floorboost_fallback_reason": "K1_K2_EXACT_D92_FULL_ALIAS",
            "d92_e0d_floorboost_new_rows_byte_exact": None,
            "d92_e0d_floorboost_old_bias_zero_sum_residual_abs": None,
            "d92_e0d_floorboost_old_intercept_mean_residual_abs": None,
            "d92_e0d_floorboost_max_abs_delta_over_rms": None,
            "d92_e0d_floorboost_full_old_rms": None,
            "d92_e0d_floorboost_retention_score_by_old_class": None,
            "d92_e0d_floorboost_registration_drift_by_old_class": None,
            "d92_e0d_floorboost_delta_bias_by_old_class": None,
            "d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound": None,
            "d92_e0d_floorboost_support_retention_affine_macs_upper_bound": None,
            "d92_e0d_floorboost_support_bias_calibration_macs_upper_bound": None,
            "d92_e0d_floorboost_support_macs_upper_bound": None,
            "d92_e0d_floorboost_support_transient_bytes_upper_bound": None,
            "d92_e0d_floorboost_persistent_state_bytes_delta": None,
        }
        row.update({field: False for field in QUERY_ZERO_FIELDS})
        rows.append(row)
    (root / "fit_audit.json").write_text(json.dumps(rows), encoding="utf-8")
    result = _fit_resource(tmp_path / "job", 1)
    assert (result["fit_count"], result["actual_fit_count"]) == (3, 3)
    assert result["registered_d_mode"] == "d92_full_alias"


def test_k10_numeric_fallback_allows_absent_per_class_diagnostics() -> None:
    row = {
        "d92_e0d_floorboost_active": False,
        "d92_e0d_floorboost_lambda": 0.25,
        "d92_e0d_floorboost_quantile": 0.20,
        "d92_e0d_floorboost_quantile_method": "lower",
        "d92_e0d_floorboost_kappa": 0.35,
        "d92_e0d_floorboost_fallback_active": True,
        "d92_e0d_floorboost_fallback_reason": "NUMERIC_FAIL_CLOSE",
        "d92_e0d_floorboost_new_rows_byte_exact": True,
        "d92_e0d_floorboost_old_bias_zero_sum_residual_abs": 0.0,
        "d92_e0d_floorboost_old_intercept_mean_residual_abs": 0.0,
        "d92_e0d_floorboost_max_abs_delta_over_rms": 0.0,
        "d92_e0d_floorboost_full_old_rms": 1.0,
        "d92_e0d_floorboost_retention_score_by_old_class": None,
        "d92_e0d_floorboost_registration_drift_by_old_class": None,
        "d92_e0d_floorboost_delta_bias_by_old_class": None,
        "d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound": 1,
        "d92_e0d_floorboost_support_retention_affine_macs_upper_bound": 1,
        "d92_e0d_floorboost_support_bias_calibration_macs_upper_bound": 1,
        "d92_e0d_floorboost_support_macs_upper_bound": 3,
        "d92_e0d_floorboost_support_transient_bytes_upper_bound": 1,
        "d92_e0d_floorboost_persistent_state_bytes_delta": 0,
    }
    _floorboost_receipt(row, 10)
