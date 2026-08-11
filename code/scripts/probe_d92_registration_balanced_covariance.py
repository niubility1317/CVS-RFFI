#!/usr/bin/env python3
"""D92 D81-center plus registration-task-balanced covariance integration."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scripts import probe_d81_ground_nuisance_cauchy_center as d81
from cvsrffi.stage2_d92_registration_balanced_covariance import (
    OLD_CLASS_COUNT,
    build_registration_balanced_equal_lda,
)


d62, d43 = d81.d62, d81.d43
d44 = d62.d61.d46.d44
load_ground_basis = d81.load_ground_basis
ARM = "registration_balanced_covariance"
STRUCTURE = "d81_center_with_fixed_equal_old_new_auto_shrinkage_covariance"
FORMULA = (
    "apply the locked D81 classwise robust support-center translation; on every "
    "registered full/block outer and held fit estimate old-prefix and new-suffix "
    "auto-shrinkage covariance separately; use fixed Sigma=0.5*Sigma_old+0.5*Sigma_new; "
    "compile one equal-prior affine head over all registered classes"
)
REGISTERED_D_MODES = ("fusion_loo", "full_only", "block_only", "fixed50")
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D92ProbeError(RuntimeError):
    """Raised when D92 integration or audit evidence drifts."""


def _component_inventory(
    records: list[dict[str, Any]], *, requested_k_shot: int
) -> dict[str, Any]:
    """Report the component fits actually invoked for one D92 state build."""

    current = [dict(row) for row in records]
    requested_k = int(requested_k_shot)
    full_count = sum(row["arm"] == "full" for row in current)
    block_count = sum(row["arm"] == "block3_centered" for row in current)
    inner_count = sum(int(row["k_shot"]) < requested_k for row in current)
    return {
        "schema": "cvs.phase2.d92.actual_component_fit_inventory.v1",
        "actual_component_fit_count": len(current),
        "full_component_fit_count": int(full_count),
        "block3_component_fit_count": int(block_count),
        "outer_component_fit_count": int(
            sum(int(row["k_shot"]) == requested_k for row in current)
        ),
        "loo_component_fit_count": int(inner_count),
        "loo_fold_count": requested_k if inner_count else 0,
        "actual_component_calls": current,
    }


def _require_registered_d_mode(registered_d_mode: str) -> str:
    mode = str(registered_d_mode)
    if mode not in REGISTERED_D_MODES:
        raise D92ProbeError(f"unknown D92 registered D mode: {registered_d_mode}")
    return mode


def build_d92_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
    *,
    disable_registered_ground_center: bool = False,
    disable_registered_fisher: bool = False,
    registered_d_mode: str = "fusion_loo",
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    registered_d_mode = _require_registered_d_mode(registered_d_mode)
    aliases = (d62.d43, d62.d61.d43, d62.d61.d46.d43, d62.d61.d46.d45.d43)
    if any(alias is not d43 for alias in aliases):
        raise D92ProbeError("D92 D43 module alias identity drift")
    original_fit = d42._fit_equal_prior_lda
    original_builder = d43.build_structured_fit
    transform_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    basis_audit = {
        "basis_sha256": ground_audit["d81_basis_sha256"],
        "spectral_weight_sha256": ground_audit["d81_spectral_weight_sha256"],
        "participation_ratio_effective_rank": ground_audit[
            "d81_participation_ratio_effective_rank"
        ],
        "retained_rank": ground_audit["d81_retained_rank"],
        "rank_policy": ground_audit["d81_rank_policy"],
    }

    d92_full = build_registration_balanced_equal_lda(
        d42, original_fit, arm="full"
    )

    def collect(component_fit: Callable[..., Any], arm: str) -> Callable[..., Any]:
        def fit(rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int):
            coefficient, intercept, audit = component_fit(
                rows, labels, class_count, k_shot
            )
            component_records.append(
                {
                    "arm": arm,
                    "class_count": int(class_count),
                    "k_shot": int(k_shot),
                    "status": audit["d92_status"],
                    "active": bool(audit["d92_registration_balanced_active"]),
                }
            )
            return coefficient, intercept, audit

        return fit

    full_component = collect(d92_full, "full")
    centered_full_fit = d81.core.build_robust_center_component_fit(
        full_component,
        basis,
        spectral_weights,
        basis_audit,
        "full",
        transform_records,
    )

    def ground_center_active(class_count: int, k_shot: int) -> bool:
        return not (
            disable_registered_ground_center
            and int(class_count) > OLD_CLASS_COUNT
            and int(k_shot) > 2
        )

    def full_fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ):
        selected = (
            centered_full_fit
            if ground_center_active(class_count, k_shot)
            else full_component
        )
        return selected(rows, labels, class_count, k_shot)

    def structured_builder(d42_arg: Any, arm: str) -> Callable[..., Any]:
        if d42_arg is not d42 or arm != "block3_centered":
            raise D92ProbeError("D92 unexpected structured covariance request")
        # Preserve D81 exactly for the registration-before state and for the
        # K1/K2 fallback.  Only the active registered state replaces the
        # structured covariance with D92's task-balanced estimate.
        baseline_block = original_builder(d42_arg, arm)
        d92_block = build_registration_balanced_equal_lda(
            d42, baseline_block, arm="block3_centered"
        )
        block_component = collect(d92_block, arm)
        centered_block_fit = d81.core.build_robust_center_component_fit(
            block_component,
            basis,
            spectral_weights,
            basis_audit,
            arm,
            transform_records,
        )

        def block_fit(
            rows: np.ndarray,
            labels: np.ndarray,
            class_count: int,
            k_shot: int,
        ):
            selected = (
                centered_block_fit
                if ground_center_active(class_count, k_shot)
                else block_component
            )
            return selected(rows, labels, class_count, k_shot)

        return block_fit

    try:
        d42._fit_equal_prior_lda = full_fit
        d43.build_structured_fit = structured_builder
        fisher_fit, call_records = d62.build_d62_fit(d42)
        no_fisher_fit = (
            d62.d61.d46.build_classwise_loo_reliability_fit(d42)
            if disable_registered_fisher
            else None
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d43.build_structured_fit = original_builder

    direct_block_fit = structured_builder(d42, "block3_centered")

    def fixed50_fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        full_coefficient, full_intercept, full_audit = full_fit(
            rows, labels, class_count, k_shot
        )
        block_coefficient, block_intercept, block_audit = direct_block_fit(
            rows, labels, class_count, k_shot
        )
        full_scale = d44._class_centered_logit_rms(
            rows, full_coefficient, full_intercept
        )
        block_scale = d44._class_centered_logit_rms(
            rows, block_coefficient, block_intercept
        )
        fused_coefficient64 = 0.5 * (
            np.asarray(full_coefficient, dtype=np.float64) / full_scale
            + np.asarray(block_coefficient, dtype=np.float64) / block_scale
        )
        fused_intercept64 = 0.5 * (
            np.asarray(full_intercept, dtype=np.float64) / full_scale
            + np.asarray(block_intercept, dtype=np.float64) / block_scale
        )
        centered_coefficient, centered_intercept = d43._center_affine_scores(
            fused_coefficient64, fused_intercept64
        )
        coefficient = np.asarray(centered_coefficient, dtype=np.float32)
        intercept = np.asarray(centered_intercept, dtype=np.float32)
        if (
            coefficient.shape != (int(class_count), int(d42.FEATURE_DIM))
            or intercept.shape != (int(class_count),)
            or not np.isfinite(coefficient).all()
            or not np.isfinite(intercept).all()
        ):
            raise D92ProbeError("D92 fixed50 affine state drift")
        audit = dict(full_audit)
        audit.update(
            {
                "coefficient_source": (
                    "d92_registered_fixed50_support_logit_rms_"
                    "full_block_equal_affine_fusion"
                ),
                "covariance_equation_residual_max": float(
                    max(
                        float(full_audit["covariance_equation_residual_max"]),
                        float(block_audit["covariance_equation_residual_max"]),
                    )
                ),
                "d92_fixed50_component_arms": ["full", "block3_centered"],
                "d92_fixed50_full_support_logit_rms": float(full_scale),
                "d92_fixed50_block_support_logit_rms": float(block_scale),
                "d92_fixed50_full_weight": 0.5,
                "d92_fixed50_block_weight": 0.5,
                "d92_fixed50_weight_scan_count": 0,
                "d92_fixed50_uses_labels_or_roles": False,
                "d92_fixed50_uses_outer_held_or_query": False,
                "d92_fixed50_class_or_scenario_specific_branch": False,
            }
        )
        return coefficient, intercept, audit

    def fit(rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int):
        start = len(component_records)
        transform_start = len(transform_records)
        registered = int(class_count) > OLD_CLASS_COUNT
        exact_full_alias = int(k_shot) <= 2
        d_mode_active = bool(
            disable_registered_fisher and registered and not exact_full_alias
        )
        fisher_active = not d_mode_active
        if not d_mode_active:
            selected_fit = fisher_fit
            effective_d_mode = "d92_full_alias"
        elif registered_d_mode == "fusion_loo":
            selected_fit = no_fisher_fit
            effective_d_mode = registered_d_mode
        elif registered_d_mode == "full_only":
            selected_fit = full_fit
            effective_d_mode = registered_d_mode
        elif registered_d_mode == "block_only":
            selected_fit = direct_block_fit
            effective_d_mode = registered_d_mode
        else:
            selected_fit = fixed50_fit
            effective_d_mode = registered_d_mode
        if selected_fit is None:
            raise D92ProbeError("D92 selected registered D fit was not constructed")
        original_centering_policy = d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT
        if not ground_center_active(class_count, k_shot):
            d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT = True
        try:
            coefficient, intercept, base_audit = selected_fit(
                rows, labels, class_count, k_shot
            )
        finally:
            d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT = original_centering_policy
        current = component_records[start:]
        component_inventory = _component_inventory(
            current, requested_k_shot=int(k_shot)
        )
        expected_active = registered and int(k_shot) > 2
        if current and any(bool(row["active"]) != expected_active for row in current):
            raise D92ProbeError("D92 component activity drift")
        audit = dict(base_audit)
        center_active = ground_center_active(class_count, k_shot)
        d81_audit = (
            {
                "d81_probe_arm": ARM,
                "d81_structure": STRUCTURE,
                "d81_formula": d81.FORMULA,
                "d81_ground_int8_component_used": True,
                "d81_ground_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "d81_ground_component_update_access": False,
                "d81_ground_statistic_semantics": ground_audit[
                    "ground_statistic_semantics"
                ],
                "d81_ground_bundle_contains_sample_radius": False,
                "d81_ground_bundle_contains_sample_count": False,
                "d81_ground_effective_rank": ground_audit[
                    "d81_participation_ratio_effective_rank"
                ],
                "d81_ground_retained_rank": int(ground_audit["d81_retained_rank"]),
                "d81_ground_rank_policy": ground_audit["d81_rank_policy"],
                "d81_all_full_block_outer_held_fits_transformed": True,
                "d81_target_covariance_preserved_by_class_translation": True,
                "d81_query_metric_source": "target_registered_support_only_d92",
                "d81_old_new_role_specific_branch": False,
                "d81_class_id_specific_formula": False,
                "d81_scene_receiver_handle_specific_branch": False,
                "d81_uses_outer_held_or_query": False,
                "d81_query_rows_used": 0,
                "d81_hyperparameter_count": 0,
                "d81_rank_scan_count": 0,
                "d81_weight_scan_count": 0,
                "d81_optimizer_steps": 0,
                "d81_single_affine_state_only": True,
            }
            if center_active
            else {
                "d81_probe_arm": "registered_robust_center_disabled",
                "d81_structure": "registered_support_plain_mean_no_ground_spectrum",
                "d81_formula": "registered support uses the unshifted 288D rows",
                "d81_ground_int8_component_used": False,
                "d81_ground_component_input_count": 0,
                "d81_ground_component_update_access": False,
                "d81_ground_bundle_contains_sample_radius": False,
                "d81_ground_bundle_contains_sample_count": False,
                "d81_all_full_block_outer_held_fits_transformed": False,
                "d81_target_covariance_preserved_by_class_translation": True,
                "d81_query_metric_source": "target_registered_support_only_d92",
                "d81_old_new_role_specific_branch": False,
                "d81_class_id_specific_formula": False,
                "d81_scene_receiver_handle_specific_branch": False,
                "d81_uses_outer_held_or_query": False,
                "d81_query_rows_used": 0,
                "d81_hyperparameter_count": 0,
                "d81_rank_scan_count": 0,
                "d81_weight_scan_count": 0,
                "d81_optimizer_steps": 0,
                "d81_single_affine_state_only": True,
            }
        )
        audit.update(
            {
                **d81_audit,
                "d81_actual_coefficient_fp32": np.asarray(
                    coefficient, dtype=np.float32
                ).tolist(),
                "d81_actual_intercept_fp32": np.asarray(
                    intercept, dtype=np.float32
                ).tolist(),
                "d92_probe_arm": ARM,
                "d92_structure": STRUCTURE,
                "d92_formula": FORMULA,
                "d92_registration_balanced_active": expected_active,
                "d92_status": (
                    "registration_balanced_active"
                    if expected_active
                    else (
                        "before_exact_d81"
                        if int(class_count) == OLD_CLASS_COUNT
                        else "k1_k2_exact_d81_fallback"
                    )
                ),
                "d92_old_class_count": OLD_CLASS_COUNT,
                "d92_new_class_count": max(0, int(class_count) - OLD_CLASS_COUNT),
                "d92_old_covariance_weight": 0.5,
                "d92_new_covariance_weight": 0.5,
                "d92_weight_scan_count": 0,
                "d92_hyperparameter_scan_count": 0,
                "d92_query_rows_used": 0,
                "d92_query_role_oracle_access": False,
                "d92_scene_receiver_seed_specific_branch": False,
                "d92_class_id_specific_formula": False,
                "d92_registration_state_support_only": True,
                "d92_component_fit_count": len(current),
                "d92_ground_center_active": center_active,
                "d92_ground_transform_execution_count": (
                    len(transform_records) - transform_start
                ),
                "d92_fisher_residual_pareto_active": fisher_active,
                "d92_k1_k2_exact_full_alias": exact_full_alias,
                "d92_registered_d_mode_requested": registered_d_mode,
                "d92_registered_d_mode_active": d_mode_active,
                "d92_registered_d_mode_effective": effective_d_mode,
                "d92_component_fit_inventory": component_inventory,
            }
        )
        return coefficient, intercept, audit

    return fit, call_records, transform_records


__all__ = [
    "ARM",
    "FORMULA",
    "REGISTERED_D_MODES",
    "STRUCTURE",
    "build_d92_fit",
    "load_ground_basis",
]
