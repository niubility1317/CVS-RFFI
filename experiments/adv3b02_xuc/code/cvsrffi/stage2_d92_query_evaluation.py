"""Locked full-query D92 evaluation built on the sealed D81 evaluator."""

from __future__ import annotations

from typing import Any


CANDIDATE_D92 = "d92_registration_balanced_covariance"
SCHEMA_D92 = "cvs.phase2.d92.full_query_evaluation.v1"


class D92QueryEvaluationError(ValueError):
    """Raised when a D92 full-query audit closure drifts."""


def run_d92_query_evaluation(**kwargs: Any) -> dict[str, Any]:
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe
    from scripts import probe_d92_registration_balanced_covariance as d92_probe
    from cvsrffi import stage2_d81_query_evaluation as d81_eval

    original_builder = d81_probe.build_d81_fit
    original_candidate = d81_eval.CANDIDATE_D81
    original_schema = d81_eval.SCHEMA
    original_audit = d81_eval._audit_fit

    def audit(result: Any, **audit_kwargs: Any) -> dict[str, Any]:
        row = original_audit(result, **audit_kwargs)
        geometry = result.geometry_audit
        before = geometry["before_covariance_audit"]
        after = geometry["final_covariance_audit"]
        k_shot = int(audit_kwargs["k_shot"])
        expected_after = k_shot > 2
        if (
            before.get("d92_status") != "before_exact_d81"
            or before.get("d92_registration_balanced_active") is not False
            or bool(after.get("d92_registration_balanced_active")) != expected_after
            or int(before.get("d92_query_rows_used", -1)) != 0
            or int(after.get("d92_query_rows_used", -1)) != 0
            or before.get("d92_query_role_oracle_access") is not False
            or after.get("d92_query_role_oracle_access") is not False
            or before.get("d92_registration_state_support_only") is not True
            or after.get("d92_registration_state_support_only") is not True
        ):
            raise D92QueryEvaluationError("D92 support-only fit closure drift")
        row.update(
            {
                "d92_before_status": before["d92_status"],
                "d92_after_status": after["d92_status"],
                "d92_registration_balanced_active": expected_after,
                "d92_old_covariance_weight": 0.5,
                "d92_new_covariance_weight": 0.5,
                "d92_weight_scan_count": 0,
                "d92_query_rows_used": 0,
            }
        )
        return row

    try:
        d81_probe.build_d81_fit = d92_probe.build_d92_fit
        d81_eval.CANDIDATE_D81 = CANDIDATE_D92
        d81_eval.SCHEMA = SCHEMA_D92
        d81_eval._audit_fit = audit
        result = d81_eval.run_d81_query_evaluation(**kwargs)
    finally:
        d81_probe.build_d81_fit = original_builder
        d81_eval.CANDIDATE_D81 = original_candidate
        d81_eval.SCHEMA = original_schema
        d81_eval._audit_fit = original_audit
    if result.get("candidate") != CANDIDATE_D92 or result.get("schema") != SCHEMA_D92:
        raise D92QueryEvaluationError("D92 result identity drift")
    return result


__all__ = ["CANDIDATE_D92", "D92QueryEvaluationError", "run_d92_query_evaluation"]
