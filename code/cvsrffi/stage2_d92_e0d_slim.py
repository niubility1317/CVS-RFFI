"""Frozen five-arm E0 D-geometry wrappers over the shared D92 fit path."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from cvsrffi.stage2_registration_resource_probe import measure_registration_call
from scripts import probe_d92_registration_balanced_covariance as d92_probe


@dataclass(frozen=True)
class D92E0DSlimArmSpec:
    """One frozen D92-E0D arm; no run-specific choice is permitted."""

    arm_id: str
    candidate_id: str
    registered_d_mode: str
    b_enabled: bool
    e_enabled: bool
    ocf_lambda: float | None = None
    floorboost_quantile: float | None = None
    floorboost_kappa: float | None = None


D92_E0D_ARMS: Mapping[str, D92E0DSlimArmSpec] = MappingProxyType(
    {
        "D92_FULL": D92E0DSlimArmSpec(
            "D92_FULL", "d92_e0d_d92_full", "fusion_loo", True, True
        ),
        "E0_FUSION": D92E0DSlimArmSpec(
            "E0_FUSION", "d92_e0d_e0_fusion", "fusion_loo", True, False
        ),
        "E0_FULL_ONLY": D92E0DSlimArmSpec(
            "E0_FULL_ONLY", "d92_e0d_e0_full_only", "full_only", True, False
        ),
        "E0_BLOCK_ONLY": D92E0DSlimArmSpec(
            "E0_BLOCK_ONLY", "d92_e0d_e0_block_only", "block_only", True, False
        ),
        "E0_FIXED50": D92E0DSlimArmSpec(
            "E0_FIXED50", "d92_e0d_e0_fixed50", "fixed50", True, False
        ),
        "E0_OCF25": D92E0DSlimArmSpec(
            "E0_OCF25", "d92_e0ocf_e0_ocf25", "ocf25", True, False, 0.25
        ),
        "E0_OCF50": D92E0DSlimArmSpec(
            "E0_OCF50", "d92_e0ocf_e0_ocf50", "ocf50", True, False, 0.50
        ),
        "E0_FULL_MAXMIN_FLOORBOOST": D92E0DSlimArmSpec(
            "E0_FULL_MAXMIN_FLOORBOOST",
            "d92_e0_full_maxmin_floorboost",
            "floorboost",
            True,
            False,
            0.25,
            0.20,
            0.35,
        ),
        "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN": D92E0DSlimArmSpec(
            "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN",
            "d92_e0_full_bidirectional_newguard_maxmin",
            "newguard_maxmin",
            True,
            False,
        ),
        "E0_FULL_BLOCK_PARETO_DISTILL": D92E0DSlimArmSpec(
            "E0_FULL_BLOCK_PARETO_DISTILL",
            "d92_e0_full_block_pareto_distill",
            "pareto_distill",
            True,
            False,
        ),
        "E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE": D92E0DSlimArmSpec(
            "E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE",
            "d92_e0_full_d42_tail_pair_code_exchange",
            "full_only",
            True,
            False,
        ),
        "E0_FULL_D42_TAIL_CLASS_ROW_ASCENT": D92E0DSlimArmSpec(
            "E0_FULL_D42_TAIL_CLASS_ROW_ASCENT",
            "d92_e0_full_d42_tail_class_row_ascent",
            "full_only",
            True,
            False,
        ),
    }
)


class D92E0DSlimError(RuntimeError):
    """Raised when an E0D arm drifts from the frozen D-only comparison."""


def _arm(arm_id: str) -> D92E0DSlimArmSpec:
    try:
        return D92_E0D_ARMS[str(arm_id)]
    except KeyError as error:
        raise D92E0DSlimError(f"unknown D92-E0D arm: {arm_id}") from error


def expected_total_component_fit_count(k_shot: int, *, arm_id: str) -> int:
    """Return the frozen two-state registered D-graph fit count for K>2."""

    k = int(k_shot)
    if k <= 2:
        raise ValueError("D92-E0D fit-count formula is defined only for K>2")
    arm = _arm(arm_id)
    if arm.e_enabled:
        return 8 * (k + 1)
    if arm.registered_d_mode == "fusion_loo":
        return 4 * (k + 1)
    if arm.registered_d_mode in ("full_only", "block_only"):
        return 2
    if arm.registered_d_mode == "newguard_maxmin":
        return 2
    if arm.registered_d_mode == "pareto_distill":
        return 4
    if arm.registered_d_mode in (
        "fixed50",
        "ocf25",
        "ocf50",
        "floorboost",
    ):
        return 4
    raise D92E0DSlimError("D92-E0D frozen D mode drift")


def _expected_actual_registered_component_fit_count(
    k_shot: int, *, arm: D92E0DSlimArmSpec
) -> int:
    """One state is half the frozen two-state receipt by construction."""

    return expected_total_component_fit_count(k_shot, arm_id=arm.arm_id) // 2


def build_d92_e0d_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
    *,
    arm_id: str,
    resource_measure: Callable[[Callable[[], Any]], tuple[Any, dict[str, Any]]] = (
        measure_registration_call
    ),
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one locked arm without altering D92's common fail-closed path."""

    arm = _arm(arm_id)
    base_fit, call_records, transform_records = d92_probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_ground_center=not arm.b_enabled,
        disable_registered_fisher=not arm.e_enabled,
        registered_d_mode=arm.registered_d_mode,
    )

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        measured, resource = resource_measure(
            lambda: base_fit(rows, labels, class_count, k_shot)
        )
        coefficient, intercept, base_audit = measured
        registered = int(class_count) > int(d92_probe.OLD_CLASS_COUNT)
        d_mode_active = bool(
            registered and int(k_shot) > 2 and not arm.e_enabled
        )
        actual_count = int(base_audit["d92_component_fit_count"])
        inventory = dict(base_audit["d92_component_fit_inventory"])
        if registered and int(k_shot) > 2:
            total_count = expected_total_component_fit_count(
                k_shot, arm_id=arm.arm_id
            )
            expected_actual = _expected_actual_registered_component_fit_count(
                k_shot, arm=arm
            )
            if actual_count != expected_actual:
                raise D92E0DSlimError(
                    "D92-E0D actual registered component-fit count drift: "
                    f"{actual_count} != {expected_actual}"
                )
            if int(inventory.get("actual_component_fit_count", -1)) != actual_count:
                raise D92E0DSlimError("D92-E0D actual component inventory drift")
        else:
            # The common before/K1/K2 D92_FULL alias is intentionally not
            # reinterpreted by the E0D two-state registered count convention.
            total_count = actual_count
        floorboost_mode = arm.registered_d_mode == "floorboost"
        floorboost_registered_state = bool(
            floorboost_mode and registered and int(k_shot) > 2
        )
        base_floorboost_active = base_audit.get("d92_floorboost_active", False)
        base_floorboost_fallback = base_audit.get(
            "d92_floorboost_fallback_active", False
        )
        base_floorboost_reason = base_audit.get("d92_floorboost_fallback_reason")
        if floorboost_mode:
            if not registered:
                if (
                    base_floorboost_active is not False
                    or base_floorboost_fallback is not False
                    or base_floorboost_reason != "NOT_REGISTERED_STATE"
                ):
                    raise D92E0DSlimError("D92-E0D floorboost before receipt drift")
            elif int(k_shot) <= 2:
                if (
                    base_floorboost_active is not False
                    or base_floorboost_fallback is not False
                    or base_floorboost_reason != "K1_K2_EXACT_D92_FULL_ALIAS"
                ):
                    raise D92E0DSlimError("D92-E0D floorboost K1/K2 receipt drift")
            elif base_floorboost_fallback is True:
                if (
                    base_floorboost_active is not False
                    or not isinstance(base_floorboost_reason, str)
                    or not base_floorboost_reason
                    or base_audit.get("d92_floorboost_new_rows_byte_exact") is not True
                    or base_audit.get("d92_floorboost_full_head_byte_exact")
                    is not True
                ):
                    raise D92E0DSlimError("D92-E0D floorboost fallback receipt drift")
            elif base_floorboost_fallback is False:
                if base_floorboost_active is not True or base_floorboost_reason is not None:
                    raise D92E0DSlimError("D92-E0D floorboost base receipt drift")
            else:
                raise D92E0DSlimError("D92-E0D floorboost fallback flag drift")
            if floorboost_registered_state and (
                base_audit.get("d92_floorboost_lambda") != arm.ocf_lambda
                or base_audit.get("d92_floorboost_quantile")
                != arm.floorboost_quantile
                or base_audit.get("d92_floorboost_quantile_method") != "lower"
                or base_audit.get("d92_floorboost_kappa") != arm.floorboost_kappa
            ):
                raise D92E0DSlimError("D92-E0D floorboost parameter receipt drift")
        elif (
            base_floorboost_active is not False
            or base_floorboost_fallback is not False
            or base_floorboost_reason is not None
        ):
            raise D92E0DSlimError("D92-E0D floorboost inactive receipt drift")
        newguard_mode = arm.registered_d_mode == "newguard_maxmin"
        newguard_registered_state = bool(
            newguard_mode and registered and int(k_shot) > 2
        )
        if newguard_mode:
            newguard_required = (
                "d92_newguard_active",
                "d92_newguard_fallback_active",
                "d92_newguard_fallback_reason",
                "d92_newguard_full_component_fit_count",
                "d92_newguard_new_rows_byte_exact",
                "d92_newguard_tau_old_envelope_shift",
                "d92_newguard_deployment_new_rows_byte_exact",
                "d92_newguard_deployment_protection_pass",
                "d92_newguard_protection_tolerance",
                "d92_newguard_new_support_min_margin_change",
                "d92_newguard_new_support_old_envelope_change_max",
                "d92_newguard_deployment_new_support_min_margin_change",
                "d92_newguard_deployment_new_support_old_envelope_change_max",
                "d92_newguard_deployment_tail_margin_change_by_old_class",
                "d92_newguard_deployment_backtrack_scale",
                "d92_newguard_deployment_attempt_count",
                "d92_newguard_deployment_full_head_byte_exact",
                "d92_newguard_deployment_codec_roundtrip_count",
                "d92_newguard_deployment_codec_macs_upper_bound",
                "d92_newguard_persistent_state_bytes_delta",
                "d92_newguard_query_rows_used",
                "d92_newguard_query_fit_access",
                "d92_newguard_query_update_access",
                "d92_newguard_query_selection_access",
                "d92_newguard_query_truth_access",
                "d92_newguard_query_role_oracle_access",
                "d92_newguard_query_class_quota_access",
                "d92_newguard_query_global_reassignment",
            )
            if any(key not in base_audit for key in newguard_required):
                raise D92E0DSlimError("D92-E0D NewGuard receipt missing")
            base_newguard_active = base_audit["d92_newguard_active"]
            base_newguard_fallback = base_audit["d92_newguard_fallback_active"]
            base_newguard_reason = base_audit["d92_newguard_fallback_reason"]
            if newguard_registered_state:
                if int(base_audit["d92_newguard_full_component_fit_count"]) != 1:
                    raise D92E0DSlimError("D92-E0D NewGuard FULL inventory drift")
                if base_newguard_fallback is True:
                    if (
                        base_newguard_active is not False
                        or not isinstance(base_newguard_reason, str)
                        or not base_newguard_reason
                        or base_audit.get("d92_newguard_full_head_byte_exact")
                        is not True
                        or base_audit.get(
                            "d92_newguard_deployment_backtrack_scale"
                        )
                        is not None
                        or int(
                            base_audit["d92_newguard_deployment_attempt_count"]
                        )
                        <= 0
                        or base_audit.get(
                            "d92_newguard_deployment_full_head_byte_exact"
                        )
                        is not True
                    ):
                        raise D92E0DSlimError(
                            "D92-E0D NewGuard fallback receipt drift"
                        )
                elif base_newguard_fallback is False:
                    protection_tolerance = float(
                        base_audit["d92_newguard_protection_tolerance"]
                    )
                    deployed_tail = np.asarray(
                        base_audit[
                            "d92_newguard_deployment_tail_margin_change_by_old_class"
                        ],
                        dtype=np.float64,
                    )
                    if (
                        base_newguard_active is not True
                        or base_newguard_reason is not None
                        or base_audit.get("d92_newguard_new_rows_byte_exact")
                        is not True
                        or base_audit.get(
                            "d92_newguard_deployment_new_rows_byte_exact"
                        )
                        is not True
                        or base_audit.get("d92_newguard_deployment_protection_pass")
                        is not True
                        or float(
                            base_audit["d92_newguard_deployment_backtrack_scale"]
                        )
                        <= 0.0
                        or int(
                            base_audit[
                                "d92_newguard_deployment_attempt_count"
                            ]
                        )
                        <= 0
                        or base_audit.get(
                            "d92_newguard_deployment_full_head_byte_exact"
                        )
                        is not False
                        or int(
                            base_audit[
                                "d92_newguard_deployment_codec_roundtrip_count"
                            ]
                        )
                        != int(
                            base_audit[
                                "d92_newguard_deployment_attempt_count"
                            ]
                        )
                        + 1
                        or int(
                            base_audit[
                                "d92_newguard_deployment_codec_macs_upper_bound"
                            ]
                        )
                        <= 0
                        or protection_tolerance
                        != float(1024.0 * np.finfo(np.float32).eps)
                        or float(
                            base_audit[
                                "d92_newguard_new_support_min_margin_change"
                            ]
                        )
                        < -protection_tolerance
                        or float(
                            base_audit[
                                "d92_newguard_deployment_new_support_min_margin_change"
                            ]
                        )
                        < -protection_tolerance
                        or float(
                            base_audit[
                                "d92_newguard_new_support_old_envelope_change_max"
                            ]
                        )
                        > protection_tolerance
                        or float(
                            base_audit[
                                "d92_newguard_deployment_new_support_old_envelope_change_max"
                            ]
                        )
                        > protection_tolerance
                        or deployed_tail.shape != (6,)
                        or not np.isfinite(deployed_tail).all()
                        or np.any(deployed_tail < -protection_tolerance)
                        or float(base_audit["d92_newguard_tau_old_envelope_shift"])
                        > 0.0
                    ):
                        raise D92E0DSlimError(
                            "D92-E0D NewGuard active receipt drift"
                        )
                else:
                    raise D92E0DSlimError(
                        "D92-E0D NewGuard fallback flag drift"
                    )
            else:
                expected_reason = (
                    "NOT_REGISTERED_STATE"
                    if not registered
                    else "K1_K2_EXACT_D92_FULL_ALIAS"
                )
                if (
                    base_newguard_active is not False
                    or base_newguard_fallback is not False
                    or base_newguard_reason != expected_reason
                    or base_audit.get("d92_newguard_deployment_backtrack_scale")
                    is not None
                    or int(base_audit["d92_newguard_deployment_attempt_count"]) != 0
                    or base_audit.get(
                        "d92_newguard_deployment_full_head_byte_exact"
                    )
                    is not True
                ):
                    raise D92E0DSlimError(
                        "D92-E0D NewGuard inactive receipt drift"
                    )
        pareto_mode = arm.registered_d_mode == "pareto_distill"
        pareto_registered_state = bool(
            pareto_mode and registered and int(k_shot) > 2
        )
        if pareto_mode:
            pareto_required = (
                "d92_pareto_distill_mode",
                "d92_pareto_distill_active",
                "d92_pareto_distill_fallback_active",
                "d92_pareto_distill_fallback_reason",
                "d92_pareto_distill_local_valid",
                "d92_pareto_distill_full_head_byte_exact",
                "d92_pareto_distill_deployed_support_constraints_pass",
                "d92_pareto_distill_deployed_full_head_byte_exact",
                "d92_pareto_distill_deployment_cross_group_margin_change_max_abs",
                "d92_pareto_distill_deployment_cross_group_margin_quantum",
                "d92_pareto_distill_deployment_cross_group_quantum_pass",
                "d92_pareto_distill_deployed_e0_affine_sha256",
                "d92_pareto_distill_deployed_candidate_affine_sha256",
                "d92_pareto_distill_full_solve_count",
                "d92_pareto_distill_block_solve_count",
                "d92_pareto_distill_loo_fit_count",
                "d92_pareto_distill_fisher_fit_count",
                "d92_pareto_distill_component_fit_count",
                "d92_pareto_distill_covariance_estimation_count",
                "d92_pareto_distill_robust_center_transform_count",
                "d92_pareto_distill_query_rows_used",
                "d92_pareto_distill_query_macs",
                "d92_pareto_distill_query_fit_access",
                "d92_pareto_distill_query_update_access",
                "d92_pareto_distill_query_selection_access",
                "d92_pareto_distill_query_truth_access",
                "d92_pareto_distill_query_role_oracle_access",
                "d92_pareto_distill_query_class_quota_access",
                "d92_pareto_distill_query_global_reassignment",
            )
            if any(key not in base_audit for key in pareto_required):
                raise D92E0DSlimError("D92-E0D Pareto Distill receipt missing")
            active = base_audit["d92_pareto_distill_active"]
            fallback = base_audit["d92_pareto_distill_fallback_active"]
            reason = base_audit["d92_pareto_distill_fallback_reason"]
            cross_group_change = base_audit[
                "d92_pareto_distill_deployment_cross_group_margin_change_max_abs"
            ]
            cross_group_quantum = base_audit[
                "d92_pareto_distill_deployment_cross_group_margin_quantum"
            ]
            cross_group_quantum_pass = base_audit[
                "d92_pareto_distill_deployment_cross_group_quantum_pass"
            ]
            if base_audit["d92_pareto_distill_mode"] != "pareto_distill":
                raise D92E0DSlimError("D92-E0D Pareto Distill mode drift")
            if pareto_registered_state:
                fixed_counts = (
                    int(base_audit["d92_pareto_distill_full_solve_count"]),
                    int(base_audit["d92_pareto_distill_block_solve_count"]),
                    int(base_audit["d92_pareto_distill_loo_fit_count"]),
                    int(base_audit["d92_pareto_distill_fisher_fit_count"]),
                    int(base_audit["d92_pareto_distill_component_fit_count"]),
                    int(
                        base_audit[
                            "d92_pareto_distill_covariance_estimation_count"
                        ]
                    ),
                    int(
                        base_audit[
                            "d92_pareto_distill_robust_center_transform_count"
                        ]
                    ),
                )
                if fixed_counts != (1, 1, 0, 0, 2, 1, 1):
                    raise D92E0DSlimError(
                        "D92-E0D Pareto Distill shared-count receipt drift"
                    )
                if fallback is True:
                    if (
                        active is not False
                        or base_audit["d92_pareto_distill_local_valid"] is not False
                        or not isinstance(reason, str)
                        or not reason
                        or base_audit["d92_pareto_distill_full_head_byte_exact"]
                        is not True
                        or base_audit[
                            "d92_pareto_distill_deployed_support_constraints_pass"
                        ]
                        is not False
                        or base_audit[
                            "d92_pareto_distill_deployed_full_head_byte_exact"
                        ]
                        is not True
                    ):
                        raise D92E0DSlimError(
                            "D92-E0D Pareto Distill fallback receipt drift"
                        )
                    if cross_group_quantum_pass is not False:
                        raise D92E0DSlimError(
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
                                raise D92E0DSlimError(
                                    "D92-E0D Pareto Distill fallback quantum receipt drift"
                                ) from error
                            if not np.isfinite(numeric) or numeric < 0.0 or (
                                positive and numeric <= 0.0
                            ):
                                raise D92E0DSlimError(
                                    "D92-E0D Pareto Distill fallback quantum receipt drift"
                                )
                elif fallback is False:
                    if (
                        active is not True
                        or base_audit["d92_pareto_distill_local_valid"] is not True
                        or reason is not None
                        or base_audit["d92_pareto_distill_full_head_byte_exact"]
                        is not False
                        or base_audit[
                            "d92_pareto_distill_deployed_support_constraints_pass"
                        ]
                        is not True
                        or base_audit[
                            "d92_pareto_distill_deployed_full_head_byte_exact"
                        ]
                        is not False
                    ):
                        raise D92E0DSlimError(
                            "D92-E0D Pareto Distill active receipt drift"
                        )
                    try:
                        quantum = float(cross_group_quantum)
                        change = float(cross_group_change)
                    except (TypeError, ValueError) as error:
                        raise D92E0DSlimError(
                            "D92-E0D Pareto Distill active quantum receipt drift"
                        ) from error
                    if (
                        cross_group_quantum_pass is not True
                        or not np.isfinite(quantum)
                        or not np.isfinite(change)
                        or quantum <= 0.0
                        or change < quantum
                    ):
                        raise D92E0DSlimError(
                            "D92-E0D Pareto Distill active quantum receipt drift"
                        )
                else:
                    raise D92E0DSlimError(
                        "D92-E0D Pareto Distill fallback flag drift"
                    )
            else:
                expected_reason = (
                    "NOT_REGISTERED_STATE"
                    if not registered
                    else "K1_K2_EXACT_D92_FULL_ALIAS"
                )
                if (
                    active is not False
                    or fallback is not False
                    or reason != expected_reason
                    or base_audit["d92_pareto_distill_local_valid"] is not False
                    or int(base_audit["d92_pareto_distill_component_fit_count"]) != 0
                    or int(base_audit["d92_pareto_distill_full_solve_count"]) != 0
                    or int(base_audit["d92_pareto_distill_block_solve_count"]) != 0
                    or base_audit["d92_pareto_distill_deployed_e0_affine_sha256"]
                    is not None
                    or base_audit[
                        "d92_pareto_distill_deployed_candidate_affine_sha256"
                    ]
                    is not None
                    or cross_group_change is not None
                    or cross_group_quantum is not None
                    or cross_group_quantum_pass is not None
                ):
                    raise D92E0DSlimError(
                        "D92-E0D Pareto Distill inactive receipt drift"
                    )
        ocf_expected_active = bool(
            arm.ocf_lambda is not None
            and registered
            and int(k_shot) > 2
            and not (floorboost_registered_state and base_floorboost_fallback is True)
        )
        base_ocf_active = base_audit.get("d92_ocf_active")
        base_ocf_lambda = base_audit.get("d92_ocf_lambda")
        if ocf_expected_active:
            if (
                base_ocf_active is not True
                or base_ocf_lambda != arm.ocf_lambda
            ):
                raise D92E0DSlimError(
                    "D92-E0D OCF base receipt active/lambda drift"
                )
        elif base_ocf_active is not False or base_ocf_lambda is not None:
            raise D92E0DSlimError(
                "D92-E0D OCF base receipt must be inactive outside registered K>2"
            )
        coefficient_array = np.asarray(coefficient, dtype=np.float32)
        intercept_array = np.asarray(intercept, dtype=np.float32)
        finite = bool(
            np.isfinite(coefficient_array).all()
            and np.isfinite(intercept_array).all()
        )
        if not finite:
            raise D92E0DSlimError("D92-E0D affine state became non-finite")
        ocf_receipt_masked_by_floorboost_fallback = bool(
            floorboost_registered_state and base_floorboost_fallback is True
        )
        audit = dict(base_audit)
        newguard_receipt = {
            key.replace("d92_newguard_", "d92_e0d_newguard_"): value
            for key, value in base_audit.items()
            if key.startswith("d92_newguard_")
        }
        pareto_receipt = {
            key.replace("d92_pareto_distill_", "d92_e0d_pareto_distill_"): value
            for key, value in base_audit.items()
            if key.startswith("d92_pareto_distill_")
        }
        audit.update(
            {
                "d92_e0d_arm_id": arm.arm_id,
                "d92_e0d_candidate_id": arm.candidate_id,
                "d92_e0d_A_lock": "joint288_z160_fft96_rf32",
                "d92_e0d_B_lock": "ground_spectrum_cauchy_robust_center_enabled",
                "d92_e0d_C_lock": "task_balanced_covariance_0.5_0.5",
                "d92_e0d_D_lock": "registered_state_frozen_mode_only",
                "d92_e0d_E_lock": "fisher_pareto_enabled_only_for_d92_full",
                "d92_e0d_F_lock": "f0_fp32_weight_fp32_bias",
                "d92_e0d_B_enabled": arm.b_enabled,
                "d92_e0d_E_enabled": arm.e_enabled,
                "d92_e0d_B_effective": bool(
                    base_audit["d92_ground_center_active"]
                ),
                "d92_e0d_E_effective": bool(
                    base_audit["d92_fisher_residual_pareto_active"]
                ),
                "d92_e0d_registered_state": registered,
                "d92_e0d_registered_d_mode": arm.registered_d_mode,
                "d92_e0d_registered_d_mode_active": d_mode_active,
                "d92_e0d_registered_d_mode_effective": base_audit[
                    "d92_registered_d_mode_effective"
                ],
                "d92_e0d_ocf_active": (
                    False
                    if ocf_receipt_masked_by_floorboost_fallback
                    else bool(base_audit.get("d92_ocf_active", False))
                ),
                "d92_e0d_ocf_lambda": (
                    None
                    if ocf_receipt_masked_by_floorboost_fallback
                    else base_audit.get("d92_ocf_lambda")
                ),
                "d92_e0d_ocf_same_after_joint_state": base_audit.get(
                    "d92_ocf_same_after_joint_state"
                ),
                "d92_e0d_ocf_new_rows_byte_exact": base_audit.get(
                    "d92_ocf_new_rows_byte_exact"
                ),
                "d92_e0d_ocf_support_alignment_affine_macs_upper_bound": (
                    None
                    if ocf_receipt_masked_by_floorboost_fallback
                    else base_audit.get(
                        "d92_ocf_support_alignment_affine_macs_upper_bound"
                    )
                ),
                "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound": (
                    None
                    if ocf_receipt_masked_by_floorboost_fallback
                    else base_audit.get(
                        "d92_ocf_support_alignment_contrast_mix_macs_upper_bound"
                    )
                ),
                "d92_e0d_ocf_support_alignment_macs_upper_bound": (
                    None
                    if ocf_receipt_masked_by_floorboost_fallback
                    else base_audit.get(
                        "d92_ocf_support_alignment_macs_upper_bound"
                    )
                ),
                "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": (
                    None
                    if ocf_receipt_masked_by_floorboost_fallback
                    else base_audit.get(
                        "d92_ocf_support_alignment_transient_bytes_upper_bound"
                    )
                ),
                "d92_e0d_floorboost_active": base_audit.get(
                    "d92_floorboost_active", False
                ),
                "d92_e0d_floorboost_lambda": base_audit.get(
                    "d92_floorboost_lambda"
                ),
                "d92_e0d_floorboost_quantile": base_audit.get(
                    "d92_floorboost_quantile"
                ),
                "d92_e0d_floorboost_quantile_method": base_audit.get(
                    "d92_floorboost_quantile_method"
                ),
                "d92_e0d_floorboost_kappa": base_audit.get(
                    "d92_floorboost_kappa"
                ),
                "d92_e0d_floorboost_fallback_active": base_audit.get(
                    "d92_floorboost_fallback_active", False
                ),
                "d92_e0d_floorboost_fallback_reason": base_audit.get(
                    "d92_floorboost_fallback_reason"
                ),
                "d92_e0d_floorboost_new_rows_byte_exact": base_audit.get(
                    "d92_floorboost_new_rows_byte_exact"
                ),
                "d92_e0d_floorboost_full_head_byte_exact": base_audit.get(
                    "d92_floorboost_full_head_byte_exact"
                ),
                "d92_e0d_floorboost_old_bias_zero_sum_residual_abs": base_audit.get(
                    "d92_floorboost_old_bias_zero_sum_residual_abs"
                ),
                "d92_e0d_floorboost_old_intercept_mean_residual_abs": base_audit.get(
                    "d92_floorboost_old_intercept_mean_residual_abs"
                ),
                "d92_e0d_floorboost_max_abs_delta_over_rms": base_audit.get(
                    "d92_floorboost_max_abs_delta_over_rms"
                ),
                "d92_e0d_floorboost_full_old_rms": base_audit.get(
                    "d92_floorboost_full_old_rms"
                ),
                "d92_e0d_floorboost_retention_score_by_old_class": base_audit.get(
                    "d92_floorboost_retention_score_by_old_class"
                ),
                "d92_e0d_floorboost_registration_drift_by_old_class": base_audit.get(
                    "d92_floorboost_registration_drift_by_old_class"
                ),
                "d92_e0d_floorboost_delta_bias_by_old_class": base_audit.get(
                    "d92_floorboost_delta_bias_by_old_class"
                ),
                "d92_e0d_floorboost_support_ocf_alignment_macs_upper_bound": base_audit.get(
                    "d92_floorboost_support_ocf_alignment_macs_upper_bound"
                ),
                "d92_e0d_floorboost_support_retention_affine_macs_upper_bound": base_audit.get(
                    "d92_floorboost_support_retention_affine_macs_upper_bound"
                ),
                "d92_e0d_floorboost_support_bias_calibration_macs_upper_bound": base_audit.get(
                    "d92_floorboost_support_bias_calibration_macs_upper_bound"
                ),
                "d92_e0d_floorboost_support_macs_upper_bound": base_audit.get(
                    "d92_floorboost_support_macs_upper_bound"
                ),
                "d92_e0d_floorboost_support_transient_bytes_upper_bound": base_audit.get(
                    "d92_floorboost_support_transient_bytes_upper_bound"
                ),
                "d92_e0d_floorboost_persistent_state_bytes_delta": base_audit.get(
                    "d92_floorboost_persistent_state_bytes_delta", 0
                ),
                "d92_e0d_k1_k2_exact_full_alias": bool(
                    base_audit["d92_k1_k2_exact_full_alias"]
                ),
                "d92_e0d_actual_component_fit_count": actual_count,
                "d92_e0d_actual_component_inventory": inventory,
                "d92_e0d_total_component_fit_count": int(total_count),
                "d92_e0d_two_state_registered_count_applies": bool(
                    registered and int(k_shot) > 2
                ),
                "d92_e0d_head_state_bytes": int(class_count) * (288 + 1) * 4,
                "d92_e0d_query_macs": int(class_count) * 288,
                "d92_e0d_query_fit_access": False,
                "d92_e0d_query_update_access": False,
                "d92_e0d_query_selection_access": False,
                "d92_e0d_query_truth_access": False,
                "d92_e0d_query_role_oracle_access": False,
                "d92_e0d_query_class_quota_access": False,
                "d92_e0d_query_global_reassignment": False,
                "d92_e0d_finite_output_pass": finite,
                **newguard_receipt,
                **pareto_receipt,
                **resource,
            }
        )
        return coefficient_array, intercept_array, audit

    return fit, call_records, transform_records


__all__ = [
    "D92E0DSlimArmSpec",
    "D92E0DSlimError",
    "D92_E0D_ARMS",
    "build_d92_e0d_fit",
    "expected_total_component_fit_count",
]
