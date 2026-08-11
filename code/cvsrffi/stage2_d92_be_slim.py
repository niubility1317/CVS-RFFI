"""Registered-state B/E switches for the shared D92 fit path."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from cvsrffi.stage2_registration_resource_probe import measure_registration_call
from scripts import probe_d92_registration_balanced_covariance as d92_probe


@dataclass(frozen=True)
class D92BESlimArmSpec:
    arm_id: str
    candidate_id: str
    b_enabled: bool
    e_enabled: bool


D92_BE_ARMS: Mapping[str, D92BESlimArmSpec] = MappingProxyType(
    {
        "FULL": D92BESlimArmSpec("FULL", "d92_be_full", True, True),
        "B0": D92BESlimArmSpec("B0", "d92_be_b0", False, True),
        "E0": D92BESlimArmSpec("E0", "d92_be_e0", True, False),
        "B0E0": D92BESlimArmSpec("B0E0", "d92_be_b0e0", False, False),
    }
)


class D92BESlimError(RuntimeError):
    """Raised when a frozen D92-BE arm or resource closure drifts."""


def expected_total_component_fit_count(k_shot: int, *, e_enabled: bool) -> int:
    """Return the registered K>2 closed-form component-fit count."""

    k = int(k_shot)
    if k <= 2:
        raise ValueError("D92-BE fit-count formula is defined only for K>2")
    return (8 if e_enabled else 4) * (k + 1)


def _arm(arm_id: str) -> D92BESlimArmSpec:
    try:
        return D92_BE_ARMS[str(arm_id)]
    except KeyError as error:
        raise D92BESlimError(f"unknown D92-BE arm: {arm_id}") from error


def build_d92_be_fit(
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
    """Build one arm while retaining one shared D92 implementation."""

    arm = _arm(arm_id)
    base_fit, call_records, transform_records = d92_probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_ground_center=not arm.b_enabled,
        disable_registered_fisher=not arm.e_enabled,
    )

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        call_start = len(call_records)
        measured, resource = resource_measure(
            lambda: base_fit(rows, labels, class_count, k_shot)
        )
        coefficient, intercept, base_audit = measured
        registered = int(class_count) > int(d92_probe.OLD_CLASS_COUNT)
        exact_alias = int(k_shot) <= 2
        raw_component_calls = int(base_audit["d92_component_fit_count"])
        fisher_evidence_calls = len(call_records) - call_start
        base_count = 2 * (raw_component_calls - fisher_evidence_calls)
        fisher_count = 2 * fisher_evidence_calls
        total_count = base_count + fisher_count
        if registered and not exact_alias:
            expected = expected_total_component_fit_count(
                k_shot, e_enabled=arm.e_enabled
            )
            if total_count != expected:
                raise D92BESlimError(
                    f"D92-BE total fit-count drift: {total_count} != {expected}"
                )
        coefficient_array = np.asarray(coefficient, dtype=np.float32)
        intercept_array = np.asarray(intercept, dtype=np.float32)
        finite = bool(
            np.isfinite(coefficient_array).all()
            and np.isfinite(intercept_array).all()
        )
        if not finite:
            raise D92BESlimError("D92-BE affine state became non-finite")
        audit = dict(base_audit)
        audit.update(
            {
                "d92_be_arm_id": arm.arm_id,
                "d92_be_candidate_id": arm.candidate_id,
                "d92_be_B_enabled": arm.b_enabled,
                "d92_be_E_enabled": arm.e_enabled,
                "d92_be_B_effective": bool(
                    base_audit["d92_ground_center_active"]
                ),
                "d92_be_E_effective": bool(
                    base_audit["d92_fisher_residual_pareto_active"]
                ),
                "d92_be_A_lock": "joint288_z160_fft96_rf32",
                "d92_be_C_lock": "task_balanced_covariance_0.5_0.5",
                "d92_be_D_lock": "full_block3_loo_reliability_fusion",
                "d92_be_F_lock": "f0_fp32_weight_fp32_bias",
                "d92_be_registered_state": registered,
                "d92_be_k1_k2_exact_full_alias": exact_alias,
                "d92_be_raw_component_call_count": raw_component_calls,
                "d92_be_base_component_fit_count": base_count,
                "d92_be_fisher_component_fit_count": fisher_count,
                "d92_be_total_component_fit_count": total_count,
                "d92_be_head_state_bytes": int(class_count) * (288 + 1) * 4,
                "d92_be_query_macs": int(class_count) * 288,
                "d92_be_query_fit_access": False,
                "d92_be_query_update_access": False,
                "d92_be_query_selection_access": False,
                "d92_be_query_role_oracle_access": False,
                "d92_be_query_class_quota_access": False,
                "d92_be_query_global_reassignment": False,
                "d92_be_finite_output_pass": finite,
                **resource,
            }
        )
        return coefficient_array, intercept_array, audit

    return fit, call_records, transform_records


__all__ = [
    "D92BESlimArmSpec",
    "D92BESlimError",
    "D92_BE_ARMS",
    "build_d92_be_fit",
    "expected_total_component_fit_count",
]
