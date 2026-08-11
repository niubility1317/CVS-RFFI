"""Truth-free full-query evaluation for the registered-state D92-BE arms."""

from __future__ import annotations

from typing import Any

from cvsrffi.stage2_d92_be_slim import (
    D92BESlimArmSpec,
    D92_BE_ARMS,
    build_d92_be_fit,
    expected_total_component_fit_count,
)


SCHEMA_BY_ARM = {
    arm_id: f"cvs.phase2.d92_be.{arm_id.lower()}.full_query_evaluation.v1"
    for arm_id in D92_BE_ARMS
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
_FORBIDDEN_TRUE_FIELDS = (
    "d92_be_query_fit_access",
    "d92_be_query_update_access",
    "d92_be_query_selection_access",
    "d92_be_query_role_oracle_access",
    "d92_be_query_class_quota_access",
    "d92_be_query_global_reassignment",
)


class D92BEQueryEvaluationError(ValueError):
    """Raised when a D92-BE prediction or protocol closure drifts."""


def _resource_receipt(audit: dict[str, Any]) -> dict[str, Any]:
    if any(field not in audit for field in _RESOURCE_FIELDS):
        raise D92BEQueryEvaluationError("D92-BE registration resource receipt drift")
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
        raise D92BEQueryEvaluationError("D92-BE registration resource values drift")
    return result


def _audit_d92_be_fit(
    result: Any,
    *,
    arm: D92BESlimArmSpec,
    scenario: str,
    k_shot: int,
    old_count: int,
    class_count: int,
) -> dict[str, Any]:
    geometry = result.geometry_audit
    before = dict(geometry.get("before_covariance_audit", {}))
    after = dict(geometry.get("final_covariance_audit", {}))
    expected_after_b = True if int(k_shot) <= 2 else arm.b_enabled
    expected_after_e = True if int(k_shot) <= 2 else arm.e_enabled
    for audit, registered, expected_b, expected_e in (
        (before, False, True, True),
        (after, True, expected_after_b, expected_after_e),
    ):
        if (
            audit.get("d92_registration_state_support_only") is not True
            or int(audit.get("d92_query_rows_used", -1)) != 0
            or audit.get("d92_query_role_oracle_access") is not False
            or audit.get("d92_scene_receiver_seed_specific_branch") is not False
            or audit.get("d92_class_id_specific_formula") is not False
            or any(audit.get(field) is not False for field in _FORBIDDEN_TRUE_FIELDS)
            or audit.get("d92_be_registered_state") is not registered
            or audit.get("d92_be_B_enabled") is not arm.b_enabled
            or audit.get("d92_be_E_enabled") is not arm.e_enabled
            or audit.get("d92_be_B_effective") is not expected_b
            or audit.get("d92_be_E_effective") is not expected_e
            or audit.get("d92_be_finite_output_pass") is not True
        ):
            raise D92BEQueryEvaluationError("D92-BE protocol closure drift")
    if int(k_shot) > 2:
        expected_count = expected_total_component_fit_count(
            k_shot, e_enabled=arm.e_enabled
        )
        if int(after.get("d92_be_total_component_fit_count", -1)) != expected_count:
            raise D92BEQueryEvaluationError("D92-BE registered fit-count drift")
    if int(after.get("d92_be_query_macs", -1)) != int(class_count) * 288:
        raise D92BEQueryEvaluationError("D92-BE query affine MAC drift")

    def center_shift(audit: dict[str, Any]) -> float:
        transform = audit.get("d81_transform_audit")
        if isinstance(transform, dict):
            return float(transform.get("center_shift_l2_max", 0.0))
        return 0.0

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
        "before_center_shift_l2_max": center_shift(before),
        "after_center_shift_l2_max": center_shift(after),
        "before_effective_sample_size_min": float(k_shot),
        "after_effective_sample_size_min": float(k_shot),
        "before_state_bytes": int(result.before_state.persistent_state_bytes),
        "after_state_bytes": int(result.state.persistent_state_bytes),
        "training_trace": [dict(row) for row in result.training_trace],
        "resource_audit": dict(result.resource_audit),
        "arm_id": arm.arm_id,
        "before_effective_B": True,
        "before_effective_E": True,
        "after_effective_B": expected_after_b,
        "after_effective_E": expected_after_e,
        "before_total_component_fit_count": int(
            before["d92_be_total_component_fit_count"]
        ),
        "after_total_component_fit_count": int(
            after["d92_be_total_component_fit_count"]
        ),
        "before_registration_resource": before_resource,
        "after_registration_resource": after_resource,
        "query_macs": int(after["d92_be_query_macs"]),
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }


def run_d92_be_query_evaluation(
    *, arm_id: str, **kwargs: Any
) -> dict[str, Any]:
    """Run one frozen arm without ever accepting a truth-side argument."""

    from cvsrffi import stage2_d81_query_evaluation as d81_eval
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe

    try:
        arm = D92_BE_ARMS[str(arm_id)]
    except KeyError as error:
        raise D92BEQueryEvaluationError(f"unknown D92-BE arm: {arm_id}") from error
    original_builder = d81_probe.build_d81_fit
    original_candidate = d81_eval.CANDIDATE_D81
    original_schema = d81_eval.SCHEMA
    original_audit = d81_eval._audit_fit

    def builder(d42: Any, basis: Any, weights: Any, ground_audit: dict[str, Any]):
        return build_d92_be_fit(
            d42,
            basis,
            weights,
            ground_audit,
            arm_id=arm.arm_id,
        )

    def audit(result: Any, **audit_kwargs: Any) -> dict[str, Any]:
        return _audit_d92_be_fit(result, arm=arm, **audit_kwargs)

    try:
        d81_probe.build_d81_fit = builder
        d81_eval.CANDIDATE_D81 = arm.candidate_id
        d81_eval.SCHEMA = SCHEMA_BY_ARM[arm.arm_id]
        d81_eval._audit_fit = audit
        result = d81_eval.run_d81_query_evaluation(**kwargs)
    finally:
        d81_probe.build_d81_fit = original_builder
        d81_eval.CANDIDATE_D81 = original_candidate
        d81_eval.SCHEMA = original_schema
        d81_eval._audit_fit = original_audit
    if (
        result.get("candidate") != arm.candidate_id
        or result.get("schema") != SCHEMA_BY_ARM[arm.arm_id]
    ):
        raise D92BEQueryEvaluationError("D92-BE result identity drift")
    return {**result, "arm_id": arm.arm_id}


__all__ = [
    "D92BEQueryEvaluationError",
    "SCHEMA_BY_ARM",
    "_audit_d92_be_fit",
    "run_d92_be_query_evaluation",
]
