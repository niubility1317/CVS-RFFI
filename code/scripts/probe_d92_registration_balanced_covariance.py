#!/usr/bin/env python3
"""D92 D81-center plus registration-task-balanced covariance integration."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from scripts import probe_d81_ground_nuisance_cauchy_center as d81
from cvsrffi.stage2_d92_registration_balanced_covariance import (
    OLD_CLASS_COUNT,
    build_registration_balanced_equal_lda,
)
from cvsrffi.stage2_td_htrc_target_transport import (
    build_td_htrc_component_fit,
)
from cvsrffi.stage2_td_htrc_m22 import (
    build_td_htrc_m22_component_fit,
)


d62, d43 = d81.d62, d81.d43
load_ground_basis = d81.load_ground_basis
load_ground_class_centers = d81.load_ground_class_centers
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
    *,
    apply_ground_center: bool = True,
    allow_fp32_centering_argmax_drift: bool = False,
    center_uncertainty_provider: Callable[
        [np.ndarray, np.ndarray, int, int], np.ndarray
    ]
    | None = None,
    covariance_mode: str = "auto",
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    aliases = (d62.d43, d62.d61.d43, d62.d61.d46.d43, d62.d61.d46.d45.d43)
    if any(alias is not d43 for alias in aliases):
        raise D92ProbeError("D92 D43 module alias identity drift")
    original_fit = d42._fit_equal_prior_lda
    original_builder = d43.build_structured_fit
    centering_kwargs = (
        {"allow_fp32_centering_argmax_drift": True}
        if allow_fp32_centering_argmax_drift
        else {}
    )
    transform_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    basis_audit = (
        {
            "basis_sha256": ground_audit["d81_basis_sha256"],
            "spectral_weight_sha256": ground_audit[
                "d81_spectral_weight_sha256"
            ],
            "participation_ratio_effective_rank": ground_audit[
                "d81_participation_ratio_effective_rank"
            ],
            "retained_rank": ground_audit["d81_retained_rank"],
            "rank_policy": ground_audit["d81_rank_policy"],
        }
        if apply_ground_center
        else {}
    )

    d92_full = build_registration_balanced_equal_lda(
        d42,
        original_fit,
        arm="full",
        center_uncertainty_provider=center_uncertainty_provider,
        covariance_mode=covariance_mode,
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
    full_fit = (
        d81.core.build_robust_center_component_fit(
            full_component,
            basis,
            spectral_weights,
            basis_audit,
            "full",
            transform_records,
        )
        if apply_ground_center
        else full_component
    )

    def structured_builder(
        d42_arg: Any,
        arm: str,
        *,
        allow_fp32_centering_argmax_drift: bool = False,
    ) -> Callable[..., Any]:
        if d42_arg is not d42 or arm != "block3_centered":
            raise D92ProbeError("D92 unexpected structured covariance request")
        baseline_block = original_builder(
            d42_arg,
            arm,
            **centering_kwargs,
        )
        d92_block = build_registration_balanced_equal_lda(
            d42,
            baseline_block,
            arm="block3_centered",
            center_uncertainty_provider=center_uncertainty_provider,
            covariance_mode=covariance_mode,
        )
        component = collect(d92_block, arm)
        return (
            d81.core.build_robust_center_component_fit(
                component,
                basis,
                spectral_weights,
                basis_audit,
                arm,
                transform_records,
            )
            if apply_ground_center
            else component
        )

    try:
        d42._fit_equal_prior_lda = full_fit
        d43.build_structured_fit = structured_builder
        base_fit, call_records = d62.build_d62_fit(
            d42, **centering_kwargs
        )
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
            if apply_ground_center
            else {
                "d81_probe_arm": "robust_center_disabled",
                "d81_structure": "no_ground_robust_center",
                "d81_ground_int8_component_used": False,
                "d81_ground_component_update_access": False,
                "d81_all_full_block_outer_held_fits_transformed": False,
                "d81_target_covariance_preserved_by_class_translation": True,
                "d81_query_rows_used": 0,
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
            }
        )
        return coefficient, intercept, audit

    return fit, call_records, transform_records


def build_td_htrc_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
    ground_class_centers: np.ndarray,
    *,
    ground_class_registry: Sequence[str] | None = None,
    target_old_class_registry: Sequence[str] | None = None,
    allow_fp32_centering_argmax_drift: bool = False,
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build D92's registration head with the opt-in TD-HTRC M2.1 module.

    The default ``build_d92_fit`` path remains unchanged.  This builder first
    constructs the same D92 full/block component fit without the legacy
    classwise wrapper, then applies TD-HTRC's shared transport followed by the
    locked Cauchy centre rule.  The returned intercept is in raw target-query
    coordinates, so callers retain the existing scoring interface.
    """

    base_fit, call_records, _ = build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        apply_ground_center=False,
        allow_fp32_centering_argmax_drift=allow_fp32_centering_argmax_drift,
    )
    transform_records: list[dict[str, Any]] = []
    fit = build_td_htrc_component_fit(
        base_fit,
        ground_class_centers=ground_class_centers,
        basis=basis,
        spectral_weights=spectral_weights,
        component_arm=ARM,
        collector=transform_records,
        old_class_count=OLD_CLASS_COUNT,
        ground_class_registry=ground_class_registry,
        target_old_class_registry=target_old_class_registry,
    )
    return fit, call_records, transform_records


def build_td_htrc_m22_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
    ground_class_centers: np.ndarray,
    *,
    ground_full_centers: np.ndarray | None = None,
    ground_class_registry: Sequence[str] | None = None,
    target_old_class_registry: Sequence[str] | None = None,
    allow_fp32_centering_argmax_drift: bool = False,
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build D92 with the explicit support-only TD-HTRC M2.2 wrapper.

    The mutable holder is registration-scoped only: M2.2 writes its diagonal
    posterior uncertainty immediately before fitting the D92 components, and
    the provider is cleared in the wrapper's ``finally`` block. It is never
    consulted by query scoring.
    """

    uncertainty_holder: dict[str, np.ndarray | None] = {"value": None}

    def center_uncertainty_provider(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> np.ndarray:
        value = uncertainty_holder["value"]
        if value is None:
            return np.zeros((int(class_count), int(d42.FEATURE_DIM)), dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (int(class_count), int(d42.FEATURE_DIM)):
            raise D92ProbeError("TD-HTRC M2.2 centre uncertainty shape drift")
        return array

    base_fit, call_records, _ = build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        apply_ground_center=False,
        allow_fp32_centering_argmax_drift=allow_fp32_centering_argmax_drift,
        center_uncertainty_provider=center_uncertainty_provider,
    )
    transform_records: list[dict[str, Any]] = []

    def set_uncertainty(value: np.ndarray | None) -> None:
        uncertainty_holder["value"] = None if value is None else np.asarray(
            value, dtype=np.float64
        )

    fit = build_td_htrc_m22_component_fit(
        base_fit,
        ground_class_centers=ground_class_centers,
        ground_full_centers=ground_full_centers,
        basis=basis,
        spectral_weights=spectral_weights,
        component_arm=ARM,
        collector=transform_records,
        center_uncertainty_setter=set_uncertainty,
        old_class_count=OLD_CLASS_COUNT,
        ground_class_registry=ground_class_registry,
        target_old_class_registry=target_old_class_registry,
    )
    return fit, call_records, transform_records


__all__ = [
    "ARM",
    "FORMULA",
    "STRUCTURE",
    "build_d92_fit",
    "build_td_htrc_fit",
    "build_td_htrc_m22_fit",
    "load_ground_basis",
    "load_ground_class_centers",
]
