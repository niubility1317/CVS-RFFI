"""Truth-free full-query evaluation for the frozen D92-E0D five-arm matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
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
_OCF_ARM_IDS = frozenset({"E0_OCF25", "E0_OCF50"})
_OCF_RECEIPT_FIELDS = (
    "d92_e0d_ocf_active",
    "d92_e0d_ocf_lambda",
    "d92_e0d_ocf_support_alignment_affine_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_macs_upper_bound",
    "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound",
)
_OCF_MAC_RECEIPT_FIELDS = _OCF_RECEIPT_FIELDS[2:5]


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
    expected_active = (
        registered and int(k_shot) > 2 and arm.arm_id in _OCF_ARM_IDS
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

    try:
        d81_probe.build_d81_fit = builder
        d81_eval.CANDIDATE_D81 = arm.candidate_id
        d81_eval.SCHEMA = SCHEMA_BY_ARM[arm.arm_id]
        d81_eval._audit_fit = audit
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
