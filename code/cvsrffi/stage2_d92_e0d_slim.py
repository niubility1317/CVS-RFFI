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
    if arm.registered_d_mode in ("fixed50", "ocf25", "ocf50"):
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
        ocf_expected_active = bool(
            arm.ocf_lambda is not None and registered and int(k_shot) > 2
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
        audit = dict(base_audit)
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
                "d92_e0d_ocf_active": bool(base_audit.get("d92_ocf_active", False)),
                "d92_e0d_ocf_lambda": base_audit.get("d92_ocf_lambda"),
                "d92_e0d_ocf_same_after_joint_state": base_audit.get(
                    "d92_ocf_same_after_joint_state"
                ),
                "d92_e0d_ocf_new_rows_byte_exact": base_audit.get(
                    "d92_ocf_new_rows_byte_exact"
                ),
                "d92_e0d_ocf_support_alignment_affine_macs_upper_bound": (
                    base_audit.get(
                        "d92_ocf_support_alignment_affine_macs_upper_bound"
                    )
                ),
                "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound": (
                    base_audit.get(
                        "d92_ocf_support_alignment_contrast_mix_macs_upper_bound"
                    )
                ),
                "d92_e0d_ocf_support_alignment_macs_upper_bound": base_audit.get(
                    "d92_ocf_support_alignment_macs_upper_bound"
                ),
                "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": (
                    base_audit.get(
                        "d92_ocf_support_alignment_transient_bytes_upper_bound"
                    )
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
                "d92_e0d_query_role_oracle_access": False,
                "d92_e0d_query_class_quota_access": False,
                "d92_e0d_query_global_reassignment": False,
                "d92_e0d_finite_output_pass": finite,
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
