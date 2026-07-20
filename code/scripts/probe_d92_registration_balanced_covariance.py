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
load_ground_basis = d81.load_ground_basis
ARM = "registration_balanced_covariance"
STRUCTURE = "d81_center_with_fixed_equal_old_new_auto_shrinkage_covariance"
FORMULA = (
    "apply the locked D81 classwise robust support-center translation; on every "
    "registered full/block outer and held fit estimate old-prefix and new-suffix "
    "auto-shrinkage covariance separately; use fixed Sigma=0.5*Sigma_old+0.5*Sigma_new; "
    "compile one equal-prior affine head over all registered classes"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D92ProbeError(RuntimeError):
    """Raised when D92 integration or audit evidence drifts."""


def build_d92_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
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

    full_fit = d81.core.build_robust_center_component_fit(
        collect(d92_full, "full"),
        basis,
        spectral_weights,
        basis_audit,
        "full",
        transform_records,
    )

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
        return d81.core.build_robust_center_component_fit(
            collect(d92_block, arm),
            basis,
            spectral_weights,
            basis_audit,
            arm,
            transform_records,
        )

    try:
        d42._fit_equal_prior_lda = full_fit
        d43.build_structured_fit = structured_builder
        base_fit, call_records = d62.build_d62_fit(d42)
    finally:
        d42._fit_equal_prior_lda = original_fit
        d43.build_structured_fit = original_builder

    def fit(rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int):
        start = len(component_records)
        coefficient, intercept, base_audit = base_fit(
            rows, labels, class_count, k_shot
        )
        current = component_records[start:]
        expected_active = int(class_count) > OLD_CLASS_COUNT and int(k_shot) > 2
        if current and any(bool(row["active"]) != expected_active for row in current):
            raise D92ProbeError("D92 component activity drift")
        audit = dict(base_audit)
        audit.update(
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
            }
        )
        return coefficient, intercept, audit

    return fit, call_records, transform_records


__all__ = ["ARM", "FORMULA", "STRUCTURE", "build_d92_fit", "load_ground_basis"]
