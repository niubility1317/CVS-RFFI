#!/usr/bin/env python3
"""D46 probe: classwise support-LOO likelihood fusion of full and block LDA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D45_HELPER_PATH = SCRIPT_DIR / "probe_d45_inner_loo_reliability_fusion.py"
D45_SPEC = importlib.util.spec_from_file_location("d46_d45_probe_helper", D45_HELPER_PATH)
if D45_SPEC is None or D45_SPEC.loader is None:
    raise RuntimeError("D46 could not load the D45 probe helper")
d45 = importlib.util.module_from_spec(D45_SPEC)
D45_SPEC.loader.exec_module(d45)
d44 = d45.d44
d43 = d45.d43


ARM = "classwise_inner_loo_likelihood"
STRUCTURE = "full_block_support_inner_loo_classwise_likelihood_fusion"
INNER_SCOPE = d45.INNER_SCOPE
WEIGHT_FORMULA = "w_g_c=softmax_g(-k_shot*inner_loo_ce_g_c)"
LOG_EVIDENCE_FORMULA = "log_evidence_g_c=-k_shot*inner_loo_ce_g_c"
CANONICAL_GAUGE = "class_mean_zero_coefficient_and_intercept_before_rms_ce_fusion"
CANONICAL_GAUGE_TOLERANCE = 1.0e-7
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D46ProbeError(RuntimeError):
    pass


def _canonical_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    collector: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficients, intercept, audit = component_fit(
            transformed, targets, class_count, k_shot
        )
        canonical_coef, canonical_intercept = d43._center_affine_scores(
            np.asarray(coefficients, dtype=np.float64),
            np.asarray(intercept, dtype=np.float64),
        )
        # Keep the canonical component in FP64 through RMS calibration, inner-LOO
        # evidence scoring, and fusion.  Casting each class row independently to
        # FP32 can reintroduce a class-common affine term and makes classwise
        # weights depend on an arbitrary score gauge.  The single fused state is
        # still cast to FP32 at the normal D42 boundary below.
        canonical_coef = np.asarray(canonical_coef, dtype=np.float64)
        canonical_intercept = np.asarray(canonical_intercept, dtype=np.float64)
        coef_mean_max = float(
            np.max(np.abs(canonical_coef.mean(axis=0)))
        )
        intercept_mean = float(abs(canonical_intercept.mean()))
        if (
            canonical_coef.ndim != 2
            or canonical_coef.shape[0] != class_count
            or canonical_intercept.shape != (class_count,)
            or not np.isfinite(canonical_coef).all()
            or not np.isfinite(canonical_intercept).all()
            or coef_mean_max > CANONICAL_GAUGE_TOLERANCE
            or intercept_mean > CANONICAL_GAUGE_TOLERANCE
        ):
            raise D46ProbeError("D46 canonical component gauge drift")
        collector.append(
            {
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "coefficient_class_mean_max_abs": coef_mean_max,
                "intercept_class_mean_abs": intercept_mean,
            }
        )
        result_audit = dict(audit)
        result_audit.update(
            {
                "d46_canonical_gauge": CANONICAL_GAUGE,
                "d46_canonical_gauge_tolerance": CANONICAL_GAUGE_TOLERANCE,
                "d46_canonical_coefficient_class_mean_max_abs": coef_mean_max,
                "d46_canonical_intercept_class_mean_abs": intercept_mean,
            }
        )
        return canonical_coef, canonical_intercept, result_audit

    return fit


def _classwise_likelihood_weights(
    full_per_class_ce: list[float] | np.ndarray,
    block_per_class_ce: list[float] | np.ndarray,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(full_per_class_ce, dtype=np.float64)
    block = np.asarray(block_per_class_ce, dtype=np.float64)
    if (
        full.ndim != 1
        or full.shape != block.shape
        or len(full) <= 0
        or int(k_shot) <= 1
        or not np.isfinite(full).all()
        or not np.isfinite(block).all()
        or np.any(full < 0.0)
        or np.any(block < 0.0)
    ):
        raise D46ProbeError("D46 classwise reliability CE drift")
    raw_log_evidence = -float(k_shot) * np.stack([full, block], axis=1)
    shifted = raw_log_evidence - np.max(raw_log_evidence, axis=1, keepdims=True)
    reliability = np.exp(shifted)
    weights = reliability / np.sum(reliability, axis=1, keepdims=True)
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-15)
    ):
        raise D46ProbeError("D46 classwise reliability weight drift")
    return weights, raw_log_evidence


def _enrich_partition_evidence(
    partition: dict[str, Any],
    targets: np.ndarray,
    class_count: int,
) -> dict[str, Any]:
    labels = np.asarray(targets, dtype=np.int64)
    held_by_fold = partition.get("held_support_row_indices_by_fold")
    if not isinstance(held_by_fold, list):
        raise D46ProbeError("D46 partition evidence missing")
    all_indices = set(range(len(labels)))
    held_classes_by_fold: list[list[int]] = []
    train_indices_by_fold: list[list[int]] = []
    for held_indices_raw in held_by_fold:
        held_indices = [int(value) for value in held_indices_raw]
        if any(value < 0 or value >= len(labels) for value in held_indices):
            raise D46ProbeError("D46 held partition index drift")
        held_classes = [int(labels[value]) for value in held_indices]
        if sorted(held_classes) != list(range(class_count)):
            raise D46ProbeError("D46 per-fold per-class held partition drift")
        held_set = set(held_indices)
        train_indices = sorted(all_indices - held_set)
        if held_set.intersection(train_indices):
            raise D46ProbeError("D46 train-held partition overlap")
        held_classes_by_fold.append(held_classes)
        train_indices_by_fold.append(train_indices)
    enriched = dict(partition)
    enriched.update(
        {
            "d46_held_class_indices_by_fold": held_classes_by_fold,
            "d46_train_support_row_indices_by_fold": train_indices_by_fold,
            "d46_train_indices_are_exact_held_complements": True,
        }
    )
    return enriched


def build_classwise_loo_reliability_fit(
    d42: Any,
) -> Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    full_base_fit = d45._build_locked_d42_full_component_fit(d42)
    block_base_fit = d43.build_structured_fit(d42, "block3_centered")

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        full_gauge_checks: list[dict[str, Any]] = []
        block_gauge_checks: list[dict[str, Any]] = []
        full_fit = _canonical_component_fit(full_base_fit, full_gauge_checks)
        block_fit = _canonical_component_fit(block_base_fit, block_gauge_checks)
        full_coef, full_intercept, full_audit = full_fit(
            transformed, targets, class_count, k_shot
        )
        block_coef, block_intercept, block_audit = block_fit(
            transformed, targets, class_count, k_shot
        )
        full_scale = d44._class_centered_logit_rms(
            transformed, full_coef, full_intercept
        )
        block_scale = d44._class_centered_logit_rms(
            transformed, block_coef, block_intercept
        )
        if int(k_shot) == 1:
            weights = np.full((class_count, 2), 0.5, dtype=np.float64)
            log_evidence = None
            full_macro_ce = None
            block_macro_ce = None
            full_per_class_ce = None
            block_per_class_ce = None
            full_partition_audit = None
            block_partition_audit = None
            inner_fold_count = 0
            k1_fallback = True
        else:
            (
                full_macro_ce,
                full_per_class_ce,
                full_partition_audit,
            ) = d45._inner_loo_component_ce(
                full_fit, transformed, targets, class_count, k_shot
            )
            (
                block_macro_ce,
                block_per_class_ce,
                block_partition_audit,
            ) = d45._inner_loo_component_ce(
                block_fit, transformed, targets, class_count, k_shot
            )
            full_partition_audit = _enrich_partition_evidence(
                full_partition_audit, targets, class_count
            )
            block_partition_audit = _enrich_partition_evidence(
                block_partition_audit, targets, class_count
            )
            weights, log_evidence_array = _classwise_likelihood_weights(
                full_per_class_ce, block_per_class_ce, k_shot
            )
            if int(k_shot) == 2 and not (
                np.allclose(
                    full_per_class_ce,
                    block_per_class_ce,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(weights, 0.5, rtol=0.0, atol=1.0e-12)
            ):
                raise D46ProbeError("D46 K2 unit-covariance classwise closure drift")
            log_evidence = log_evidence_array.tolist()
            inner_fold_count = int(k_shot)
            k1_fallback = False
        full_weight = weights[:, 0]
        block_weight = weights[:, 1]
        fused_coef64 = (
            full_weight[:, None]
            * np.asarray(full_coef, dtype=np.float64)
            / full_scale
            + block_weight[:, None]
            * np.asarray(block_coef, dtype=np.float64)
            / block_scale
        )
        fused_intercept64 = (
            full_weight * np.asarray(full_intercept, dtype=np.float64) / full_scale
            + block_weight
            * np.asarray(block_intercept, dtype=np.float64)
            / block_scale
        )
        centered_coef, centered_intercept = d43._center_affine_scores(
            fused_coef64, fused_intercept64
        )
        coef32 = centered_coef.astype(np.float32)
        intercept32 = centered_intercept.astype(np.float32)
        if (
            coef32.shape != (class_count, d42.FEATURE_DIM)
            or intercept32.shape != (class_count,)
            or not np.isfinite(coef32).all()
            or not np.isfinite(intercept32).all()
        ):
            raise D46ProbeError("D46 fused affine state drift")
        component_condition_numbers = [
            float(value)
            for value in (
                full_audit.get("d43_covariance_condition_number"),
                block_audit.get("d43_covariance_condition_number"),
            )
            if value is not None
        ]
        audit = dict(full_audit)
        audit.update(
            {
                "coefficient_source": (
                    "d46_support_inner_loo_classwise_likelihood_"
                    "full_block_rms_affine_fusion"
                ),
                "covariance_equation_residual_max": float(
                    max(
                        float(full_audit["covariance_equation_residual_max"]),
                        float(block_audit["covariance_equation_residual_max"]),
                    )
                ),
                "sklearn_prediction_equivalent": None,
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d43_covariance_condition_number": (
                    max(component_condition_numbers)
                    if component_condition_numbers
                    else None
                ),
                "d46_probe_arm": ARM,
                "d46_component_arms": [
                    "full_centered_control",
                    "block3_centered",
                ],
                "d46_scale_formula": d44.SCALE_FORMULA,
                "d46_weight_formula": WEIGHT_FORMULA,
                "d46_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d46_canonical_gauge": CANONICAL_GAUGE,
                "d46_canonical_gauge_tolerance": CANONICAL_GAUGE_TOLERANCE,
                "d46_inner_scope": INNER_SCOPE,
                "d46_outer_b20_frozen_across_inner_folds": True,
                "d46_outer_b20_refit_per_inner_fold": False,
                "d46_inner_loo_generalization_claim_allowed": False,
                "d46_full_support_logit_rms": full_scale,
                "d46_block_support_logit_rms": block_scale,
                "d46_full_inner_loo_macro_class_ce": full_macro_ce,
                "d46_block_inner_loo_macro_class_ce": block_macro_ce,
                "d46_full_inner_loo_ce_by_class": full_per_class_ce,
                "d46_block_inner_loo_ce_by_class": block_per_class_ce,
                "d46_log_evidence_by_class_and_component": log_evidence,
                "d46_full_inner_partition_audit": full_partition_audit,
                "d46_block_inner_partition_audit": block_partition_audit,
                "d46_full_weight_by_class": full_weight.tolist(),
                "d46_block_weight_by_class": block_weight.tolist(),
                "d46_inner_loo_fold_count": inner_fold_count,
                "d46_k1_equivalent_unit_covariance_fallback": k1_fallback,
                "d46_reliability_uses_support_labels": int(k_shot) > 1,
                "d46_reliability_uses_outer_held_or_query": False,
                "d46_class_id_specific_formula": False,
                "d46_old_new_role_specific_branch": False,
                "d46_scene_handle_specific_branch": False,
                "d46_weight_scan_count": 0,
                "d46_full_component_canonical_gauge_checks": full_gauge_checks,
                "d46_block_component_canonical_gauge_checks": block_gauge_checks,
                "d46_actual_inner_fold_count_used_as_likelihood_exponent": int(
                    k_shot
                ) if int(k_shot) > 1 else None,
            }
        )
        return coef32, intercept32, audit

    return fit


def _install_d46_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d45._install_d45_core_resource_accounting(d42)
    d45_installed_top = d42.fit_d42_unified_shrinkage_lda

    def fit_with_d46_resource_audit(*args: Any, **kwargs: Any) -> Any:
        result = d45_installed_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_class_count = len(result.before_state.classes)
        all_class_count = len(result.state.classes)
        k_shot = int(resource["old_k_shot"])
        dimension = int(d42.FEATURE_DIM)
        if (
            result.before_state.coef_fp32.shape[1] != dimension
            or result.state.coef_fp32.shape[1] != dimension
        ):
            raise D46ProbeError("D46 coefficient dimension drift")
        class_square_sum = old_class_count**2 + all_class_count**2
        reliability_macs = int(
            2
            * k_shot
            * (1 if k_shot <= 1 else k_shot + 1)
            * dimension
            * class_square_sum
        )
        fusion_macs = int(
            2 * (dimension + 1) * (old_class_count + all_class_count)
        )
        resource.update(
            {
                "coefficient_dimension": dimension,
                "d46_classwise_component_weight_count": int(
                    2 * resource["registered_class_count"]
                ),
                "d46_fused_query_state_count": 1,
                "d46_outer_b20_training_count": 1,
                "d46_inner_scope": INNER_SCOPE,
                "d46_resource_inventory_helper": "d45_exact_4k_plus4_inventory",
                "d46_estimated_reliability_scoring_macs": reliability_macs,
                "d46_estimated_classwise_affine_fusion_macs": fusion_macs,
                "d46_reliability_mac_formula": (
                    "2*K*D*(C_old^2+C_all^2)_if_K1_else_"
                    "2*K*(K+1)*D*(C_old^2+C_all^2)"
                ),
                "d46_affine_fusion_mac_formula": "2*(D+1)*(C_old+C_all)",
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"]
            + resource["estimated_lda_fit_macs"]
            + reliability_macs
            + fusion_macs
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = fit_with_d46_resource_audit
    return original_macs, original_top


def _verify_d46_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d46_rows = [
        row
        for row in training_rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d46_rows) != 30:
        raise D46ProbeError("D46 training-row closure drift")
    for row in d46_rows:
        resource = row.get("resource")
        geometry = row.get("geometry_summary")
        if not isinstance(resource, dict) or not isinstance(geometry, dict):
            raise D46ProbeError("D46 row audit missing")
        k_shot = int(resource.get("old_k_shot", -1))
        if int(resource.get("new_k_shot", -2)) != k_shot:
            raise D46ProbeError("D46 K closure drift")
        class_counts: list[int] = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field)
            if not isinstance(audit, dict):
                raise D46ProbeError(f"D46 fit audit missing from {field}")
            required = {
                "d46_probe_arm": ARM,
                "d46_component_arms": [
                    "full_centered_control",
                    "block3_centered",
                ],
                "d46_scale_formula": d44.SCALE_FORMULA,
                "d46_weight_formula": WEIGHT_FORMULA,
                "d46_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d46_canonical_gauge": CANONICAL_GAUGE,
                "d46_canonical_gauge_tolerance": CANONICAL_GAUGE_TOLERANCE,
                "d46_inner_scope": INNER_SCOPE,
                "d46_outer_b20_frozen_across_inner_folds": True,
                "d46_outer_b20_refit_per_inner_fold": False,
                "d46_inner_loo_generalization_claim_allowed": False,
                "d46_reliability_uses_outer_held_or_query": False,
                "d46_class_id_specific_formula": False,
                "d46_old_new_role_specific_branch": False,
                "d46_scene_handle_specific_branch": False,
                "d46_weight_scan_count": 0,
            }
            if any(audit.get(name) != value for name, value in required.items()):
                raise D46ProbeError(f"D46 fit audit drift in {field}")
            full_weights = np.asarray(
                audit.get("d46_full_weight_by_class"), dtype=np.float64
            )
            block_weights = np.asarray(
                audit.get("d46_block_weight_by_class"), dtype=np.float64
            )
            if (
                full_weights.ndim != 1
                or block_weights.shape != full_weights.shape
                or len(full_weights) <= 0
                or not np.isfinite(full_weights).all()
                or not np.isfinite(block_weights).all()
                or np.any(full_weights <= 0.0)
                or np.any(block_weights <= 0.0)
                or not np.allclose(
                    full_weights + block_weights, 1.0, rtol=0.0, atol=1.0e-12
                )
            ):
                raise D46ProbeError("D46 classwise weight closure drift")
            class_count = len(full_weights)
            class_counts.append(class_count)
            expected_gauge_check_count = 1 if k_shot <= 1 else k_shot + 1
            for name in (
                "d46_full_component_canonical_gauge_checks",
                "d46_block_component_canonical_gauge_checks",
            ):
                checks = audit.get(name)
                if not isinstance(checks, list) or len(checks) != expected_gauge_check_count:
                    raise D46ProbeError("D46 canonical gauge check-count drift")
                expected_k = [k_shot] + ([] if k_shot <= 1 else [k_shot - 1] * k_shot)
                for check, expected_fit_k in zip(checks, expected_k):
                    if (
                        not isinstance(check, dict)
                        or check.get("class_count") != class_count
                        or check.get("k_shot") != expected_fit_k
                        or not np.isfinite(
                            float(check.get("coefficient_class_mean_max_abs", np.nan))
                        )
                        or not np.isfinite(
                            float(check.get("intercept_class_mean_abs", np.nan))
                        )
                        or float(check["coefficient_class_mean_max_abs"])
                        > CANONICAL_GAUGE_TOLERANCE
                        or float(check["intercept_class_mean_abs"])
                        > CANONICAL_GAUGE_TOLERANCE
                    ):
                        raise D46ProbeError("D46 canonical gauge evidence drift")
            expected_likelihood_exponent = k_shot if k_shot > 1 else None
            if audit.get(
                "d46_actual_inner_fold_count_used_as_likelihood_exponent"
            ) != expected_likelihood_exponent:
                raise D46ProbeError("D46 likelihood exponent drift")
            if k_shot > 1:
                full_ce = np.asarray(
                    audit.get("d46_full_inner_loo_ce_by_class"), dtype=np.float64
                )
                block_ce = np.asarray(
                    audit.get("d46_block_inner_loo_ce_by_class"), dtype=np.float64
                )
                if full_ce.shape != (class_count,) or block_ce.shape != (class_count,):
                    raise D46ProbeError("D46 classwise CE shape drift")
                d45._verify_partition_evidence(
                    audit.get("d46_full_inner_partition_audit"),
                    class_count=class_count,
                    k_shot=k_shot,
                    expected_per_class_ce=full_ce,
                )
                d45._verify_partition_evidence(
                    audit.get("d46_block_inner_partition_audit"),
                    class_count=class_count,
                    k_shot=k_shot,
                    expected_per_class_ce=block_ce,
                )
                for partition_name in (
                    "d46_full_inner_partition_audit",
                    "d46_block_inner_partition_audit",
                ):
                    partition = audit[partition_name]
                    held_by_fold = partition[
                        "held_support_row_indices_by_fold"
                    ]
                    held_classes_by_fold = partition.get(
                        "d46_held_class_indices_by_fold"
                    )
                    train_by_fold = partition.get(
                        "d46_train_support_row_indices_by_fold"
                    )
                    if (
                        not isinstance(held_classes_by_fold, list)
                        or len(held_classes_by_fold) != k_shot
                        or not isinstance(train_by_fold, list)
                        or len(train_by_fold) != k_shot
                        or partition.get(
                            "d46_train_indices_are_exact_held_complements"
                        ) is not True
                    ):
                        raise D46ProbeError("D46 enriched partition evidence drift")
                    all_indices = set(range(class_count * k_shot))
                    for held_indices, held_classes, train_indices in zip(
                        held_by_fold, held_classes_by_fold, train_by_fold
                    ):
                        held_set = set(held_indices)
                        if (
                            sorted(held_classes) != list(range(class_count))
                            or train_indices != sorted(all_indices - held_set)
                            or held_set.intersection(train_indices)
                        ):
                            raise D46ProbeError(
                                "D46 per-fold per-class held partition drift"
                            )
                expected_weights, expected_log_evidence = (
                    _classwise_likelihood_weights(full_ce, block_ce, k_shot)
                )
                if (
                    audit.get("d46_inner_loo_fold_count") != k_shot
                    or audit.get("d46_reliability_uses_support_labels") is not True
                    or audit.get("d46_k1_equivalent_unit_covariance_fallback")
                    is not False
                    or not np.allclose(
                        np.stack([full_weights, block_weights], axis=1),
                        expected_weights,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        audit.get("d46_log_evidence_by_class_and_component"),
                        expected_log_evidence,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise D46ProbeError("D46 CE/log-evidence/weight closure drift")
                if k_shot == 2 and (
                    not np.allclose(full_ce, block_ce, rtol=0.0, atol=1.0e-12)
                    or not np.allclose(expected_weights, 0.5, rtol=0.0, atol=1.0e-12)
                ):
                    raise D46ProbeError("D46 K2 classwise verifier drift")
                for name, values in (
                    ("d46_full_inner_loo_macro_class_ce", full_ce),
                    ("d46_block_inner_loo_macro_class_ce", block_ce),
                ):
                    if not np.isclose(
                        float(audit.get(name)),
                        float(np.mean(values)),
                        rtol=0.0,
                        atol=1.0e-12,
                    ):
                        raise D46ProbeError("D46 macro CE closure drift")
            elif (
                audit.get("d46_inner_loo_fold_count") != 0
                or audit.get("d46_reliability_uses_support_labels") is not False
                or audit.get("d46_k1_equivalent_unit_covariance_fallback") is not True
                or audit.get("d46_full_inner_loo_ce_by_class") is not None
                or audit.get("d46_block_inner_loo_ce_by_class") is not None
                or audit.get("d46_log_evidence_by_class_and_component") is not None
                or audit.get("d46_full_inner_partition_audit") is not None
                or audit.get("d46_block_inner_partition_audit") is not None
                or not np.allclose(full_weights, 0.5, rtol=0.0, atol=0.0)
                or not np.allclose(block_weights, 0.5, rtol=0.0, atol=0.0)
            ):
                raise D46ProbeError("D46 K1 fallback closure drift")
        old_count, all_count = class_counts
        if all_count != int(resource.get("registered_class_count", -1)):
            raise D46ProbeError("D46 registered class-count drift")
        d45._verify_fit_inventory(
            resource,
            old_class_count=old_count,
            all_class_count=all_count,
            k_shot=k_shot,
        )
        expected_resource = {
            "d46_classwise_component_weight_count": 2 * all_count,
            "d46_fused_query_state_count": 1,
            "d46_outer_b20_training_count": 1,
            "d46_inner_scope": INNER_SCOPE,
            "d46_resource_inventory_helper": "d45_exact_4k_plus4_inventory",
        }
        if any(resource.get(name) != value for name, value in expected_resource.items()):
            raise D46ProbeError("D46 resource closure drift")
        dimension = int(resource.get("coefficient_dimension", -1))
        expected_reliability_macs = int(
            2
            * k_shot
            * (1 if k_shot <= 1 else k_shot + 1)
            * dimension
            * (old_count**2 + all_count**2)
        )
        expected_fusion_macs = int(
            2 * (dimension + 1) * (old_count + all_count)
        )
        if (
            resource.get("d46_estimated_reliability_scoring_macs")
            != expected_reliability_macs
            or resource.get("d46_estimated_classwise_affine_fusion_macs")
            != expected_fusion_macs
            or resource.get("estimated_adaptation_macs")
            != resource.get("estimated_metric_adaptation_macs")
            + resource.get("estimated_lda_fit_macs")
            + expected_reliability_macs
            + expected_fusion_macs
        ):
            raise D46ProbeError("D46 additional MAC closure drift")
        if (
            geometry.get("before_materialized_pre_stage2c") is not True
            or geometry.get("before_state_immutable_during_stage2c") is not True
            or int(geometry.get("old_only_metric_new_support_argument_count", -1))
            != 0
        ):
            raise D46ProbeError("D46 before/new-support lifecycle drift")
    return len(d46_rows)


def _verify_d46_output(
    output: Path,
    probe_script_sha256: str,
    d45_helper_sha256: str,
    d44_helper_sha256: str,
    d43_helper_sha256: str,
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    source_closure = support.get("candidate_lock", {}).get("source_closure", {})
    expected = {
        "d46_d45_helper_sha256": d45_helper_sha256,
        "d46_d44_helper_sha256": d44_helper_sha256,
        "d46_d43_helper_sha256": d43_helper_sha256,
    }
    if any(source_closure.get(name) != value for name, value in expected.items()):
        raise D46ProbeError("D46 helper source closure drift")
    training_rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    count = _verify_d46_fit_audits(training_rows)
    return {**evidence, "verified_d46_fit_row_count": count, **expected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d46-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D46ProbeError(f"D46 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_macs = None
    original_top = None
    runner_module_name = "d46_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d45_helper_sha256 = d43._sha256(D45_HELPER_PATH)
    d44_helper_sha256 = d43._sha256(d45.D44_HELPER_PATH)
    d43_helper_sha256 = d43._sha256(d44.D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_classwise_loo_reliability_fit(d42)
        original_macs, original_top = _install_d46_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D46ProbeError("D46 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d46_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d46_d45_helper_sha256": d45_helper_sha256,
                "d46_d44_helper_sha256": d44_helper_sha256,
                "d46_d43_helper_sha256": d43_helper_sha256,
            },
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_macs is not None:
            d42._lda_fit_macs = original_macs
        if d42 is not None and original_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_d46_output(
        output,
        probe_script_sha256,
        d45_helper_sha256,
        d44_helper_sha256,
        d43_helper_sha256,
    )
    metadata = {
        "schema": "cvs.phase2.d46.classwise_loo_reliability_fusion_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d46_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "d45_helper_sha256": d45_helper_sha256,
        "d44_helper_sha256": d44_helper_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "weight_formula": WEIGHT_FORMULA,
        "log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "canonical_gauge": CANONICAL_GAUGE,
        "inner_scope": INNER_SCOPE,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D46_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
