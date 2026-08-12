"""Truth-free full-query evaluation for the frozen D92-E0D five-arm matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
from cvsrffi.stage2_registration_resource_probe import measure_registration_call
from cvsrffi.stage2_d92_e0d_slim import (
    D92E0DSlimArmSpec,
    D92_E0D_ARMS,
    build_d92_e0d_fit,
    expected_total_component_fit_count,
)


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
    "d92_e0d_newguard_deployment_backtrack_scale",
    "d92_e0d_newguard_deployment_attempt_count",
    "d92_e0d_newguard_deployment_full_head_byte_exact",
    "d92_e0d_newguard_deployment_codec_roundtrip_count",
    "d92_e0d_newguard_deployment_codec_macs_upper_bound",
    "d92_e0d_newguard_nullspace_rank",
    "d92_e0d_newguard_rank_threshold",
    "d92_e0d_newguard_max_abs_Xnew_internal_residual",
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
            or receipt["d92_e0d_newguard_deployment_backtrack_scale"] is not None
            or int(receipt["d92_e0d_newguard_deployment_attempt_count"]) != 0
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
            or receipt["d92_e0d_newguard_deployment_backtrack_scale"] is not None
            or int(receipt["d92_e0d_newguard_deployment_attempt_count"]) <= 0
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
                receipt["d92_e0d_newguard_deployment_backtrack_scale"]
            ),
            "attempts": int(
                receipt["d92_e0d_newguard_deployment_attempt_count"]
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
        or numeric["xnew"] < 0.0
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
        or numeric["scale"] <= 0.0
        or numeric["scale"] > 128.0
        or numeric["attempts"] <= 0
        or numeric["attempts"] > 20
        or numeric["codec_roundtrips"] != numeric["attempts"] + 1
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
        "support_score_macs_upper_bound",
        "support_coordinate_comparisons_upper_bound",
        "support_macs_upper_bound",
        "support_transient_bytes_upper_bound",
    ):
        finite(name, lower=0.0)
    if fallback is True:
        if (
            active is not False
            or not isinstance(reason, str)
            or not reason
            or final_sha != e0_sha
            or receipt[prefix + "changed_code2_count"] != 0
            or receipt[prefix + "applied_atomic_exchange_count"] != 0
            or receipt[prefix + "support_guard_pass"] is not False
            or (
                reason == "aggregate_saturation"
                and finite("aggregate_saturation_count", lower=0.0) <= 0.0
            )
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TPCE fallback receipt drift")
        return receipt
    if (
        active is not True
        or fallback is not False
        or reason is not None
        or final_sha == e0_sha
        or receipt[prefix + "support_guard_pass"] is not True
        or finite("changed_code2_count", lower=1.0) < 1.0
        or finite("requested_atomic_exchange_count", lower=1.0)
        != finite("applied_atomic_exchange_count", lower=1.0)
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
    if int(k_shot) > 2:
        expected_total = expected_total_component_fit_count(
            k_shot, arm_id=arm.arm_id
        )
        if (
            int(after.get("d92_e0d_total_component_fit_count", -1))
            != expected_total
            or int(after.get("d92_e0d_actual_component_fit_count", -1))
            != expected_total // 2
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
            else None
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
        **after_pareto_deployed_state_closure,
        **after_tpce_receipt,
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
) -> dict[str, Any]:
    """Run one frozen arm without exposing a truth-side input surface."""

    from cvsrffi import stage2_d81_query_evaluation as d81_eval
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe

    try:
        arm = D92_E0D_ARMS[str(arm_id)]
    except KeyError as error:
        raise D92E0DQueryEvaluationError(f"unknown D92-E0D arm: {arm_id}") from error
    original_builder = d81_probe.build_d81_fit
    original_candidate = d81_eval.CANDIDATE_D81
    original_schema = d81_eval.SCHEMA
    original_audit = d81_eval._audit_fit
    original_d42_fit = d81_eval.fit_d42_unified_shrinkage_lda

    def builder(d42: Any, basis: Any, weights: Any, ground_audit: dict[str, Any]):
        return build_d92_e0d_fit(
            d42,
            basis,
            weights,
            ground_audit,
            arm_id=arm.arm_id,
        )

    def audit(
        result: Any,
        *,
        scenario: str,
        k_shot: int,
        old_count: int,
        class_count: int,
    ) -> dict[str, Any]:
        return _audit_d92_e0d_fit(
            result,
            arm=arm,
            scenario=scenario,
            k_shot=k_shot,
            old_count=old_count,
            class_count=class_count,
        )

    def fit_with_tpce(
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
        if arm.arm_id not in _TPCE_ARM_IDS:
            return result
        from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
        from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as tpce

        old_rows = np.asarray(old_support_features, dtype=np.float32)
        new_rows = np.asarray(new_support_features, dtype=np.float32)
        old_registry = tuple(str(value) for value in old_classes)
        new_registry = tuple(str(value) for value in new_classes)
        registry = old_registry + new_registry
        if tuple(result.state.classes) != registry:
            raise D92E0DQueryEvaluationError("D92-E0D TPCE registry drift")
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
                "D92-E0D TPCE support registry drift"
            ) from error
        all_rows = np.concatenate([old_rows, new_rows], axis=0).astype(np.float32)
        transformed = d42._transform(all_rows, result.state.log_diag_fp32)
        class_counts = np.bincount(targets, minlength=len(registry))
        if (
            len(class_counts) != len(registry)
            or np.any(class_counts <= 0)
            or len(set(int(value) for value in class_counts.tolist())) != 1
        ):
            raise D92E0DQueryEvaluationError("D92-E0D TPCE K closure drift")
        k_value = int(class_counts[0])
        if k_value <= 2:
            candidate_state = result.state
            tpce_audit = tpce.d42_tpce_inactive_receipt(result.state)
            post_resource = None
        else:
            measured, post_resource = measure_registration_call(
                lambda: tpce.apply_d42_tail_pair_code_exchange(
                    result.state,
                    transformed,
                    targets,
                    old_class_count=len(old_registry),
                )
            )
            candidate_state, tpce_audit = measured
        formal_receipt = {
            key.replace("d92_tpce_", "d92_e0d_tpce_"): value
            for key, value in tpce_audit.items()
            if key.startswith("d92_tpce_")
        }
        geometry = dict(result.geometry_audit)
        final_audit = dict(geometry.get("final_covariance_audit", {}))
        if post_resource is not None:
            if any(field not in final_audit for field in _RESOURCE_FIELDS):
                raise D92E0DQueryEvaluationError(
                    "D92-E0D TPCE base resource receipt drift"
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
        if arm.arm_id in _TPCE_ARM_IDS:
            d81_eval.fit_d42_unified_shrinkage_lda = fit_with_tpce
        result = d81_eval.run_d81_query_evaluation(
            before_enrollment_package_root=before_enrollment_package_root,
            before_enrollment_seal_path=before_enrollment_seal_path,
            before_enrollment_seal_sha256=before_enrollment_seal_sha256,
            before_apply_package_root=before_apply_package_root,
            before_apply_seal_path=before_apply_seal_path,
            before_apply_seal_sha256=before_apply_seal_sha256,
            after_enrollment_package_root=after_enrollment_package_root,
            after_enrollment_seal_path=after_enrollment_seal_path,
            after_enrollment_seal_sha256=after_enrollment_seal_sha256,
            after_apply_package_root=after_apply_package_root,
            after_apply_seal_path=after_apply_seal_path,
            after_apply_seal_sha256=after_apply_seal_sha256,
            ground_component_dir=ground_component_dir,
            ground_manifest_sha256=ground_manifest_sha256,
            output_root=output_root,
            device=device,
        )
    finally:
        d81_probe.build_d81_fit = original_builder
        d81_eval.CANDIDATE_D81 = original_candidate
        d81_eval.SCHEMA = original_schema
        d81_eval._audit_fit = original_audit
        d81_eval.fit_d42_unified_shrinkage_lda = original_d42_fit
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
