#!/usr/bin/env python3
"""D45 probe: frozen-outer-B20 head-only LOO fusion of full and block LDA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D44_HELPER_PATH = SCRIPT_DIR / "probe_d44_full_block_rms_fusion.py"
D44_SPEC = importlib.util.spec_from_file_location("d45_d44_probe_helper", D44_HELPER_PATH)
if D44_SPEC is None or D44_SPEC.loader is None:
    raise RuntimeError("D45 could not load the D44 probe helper")
d44 = importlib.util.module_from_spec(D44_SPEC)
D44_SPEC.loader.exec_module(d44)
d43 = d44.d43


ARM = "inner_loo_class_likelihood"
STRUCTURE = "full_block_support_inner_loo_class_likelihood_fusion"
WEIGHT_FORMULA = "w_g=softmax_g(-registered_class_count*macro_class_inner_loo_ce_g)"
INNER_SCOPE = "frozen_outer_b20_head_only_loo"
LOG_EVIDENCE_FORMULA = "log_evidence_g=-registered_class_count*macro_class_inner_loo_ce_g"
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D45ProbeError(RuntimeError):
    pass


def _class_balanced_cross_entropy(
    scores: np.ndarray,
    targets: np.ndarray,
    class_count: int,
) -> tuple[float, list[float]]:
    logits = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if (
        logits.ndim != 2
        or logits.shape[1] != class_count
        or labels.shape != (logits.shape[0],)
        or not np.isfinite(logits).all()
        or np.any(labels < 0)
        or np.any(labels >= class_count)
    ):
        raise D45ProbeError("D45 inner-LOO score/target shape drift")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_norm = np.log(np.sum(np.exp(shifted), axis=1))
    nll = log_norm - shifted[np.arange(len(labels)), labels]
    per_class = []
    for class_index in range(class_count):
        values = nll[labels == class_index]
        if len(values) == 0:
            raise D45ProbeError("D45 inner-LOO class coverage drift")
        per_class.append(float(np.mean(values)))
    macro = float(np.mean(per_class))
    if not np.isfinite(macro) or macro < 0.0:
        raise D45ProbeError("D45 inner-LOO CE became invalid")
    return macro, per_class


def _inner_loo_component_ce(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[float, list[float], dict[str, Any]]:
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if int(k_shot) <= 1:
        raise D45ProbeError("D45 inner-LOO CE is undefined for K1")
    indices_by_class = [np.flatnonzero(labels == index) for index in range(class_count)]
    if any(len(indices) != int(k_shot) for indices in indices_by_class):
        raise D45ProbeError("D45 inner-LOO requires exact equal K per class")
    held_scores: list[np.ndarray] = []
    held_labels: list[np.ndarray] = []
    held_indices_by_fold: list[list[int]] = []
    held_ce_by_fold_and_class: list[list[float]] = []
    train_held_overlap_count = 0
    for rank in range(int(k_shot)):
        held_indices = np.asarray(
            [indices_by_class[index][rank] for index in range(class_count)],
            dtype=np.int64,
        )
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[held_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_held_overlap_count += int(
            len(np.intersect1d(train_indices, held_indices, assume_unique=True))
        )
        train_rows = rows[train_mask]
        train_labels = labels[train_mask]
        coefficients, intercept, _audit = component_fit(
            train_rows,
            train_labels,
            class_count,
            int(k_shot) - 1,
        )
        scale = d44._class_centered_logit_rms(
            train_rows,
            coefficients,
            intercept,
        )
        fold_scores = (
            rows[held_indices] @ np.asarray(coefficients, dtype=np.float64).T
            + np.asarray(intercept, dtype=np.float64)[None, :]
        ) / scale
        held_scores.append(fold_scores)
        held_labels.append(labels[held_indices])
        held_indices_by_fold.append([int(value) for value in held_indices])
        _fold_macro, fold_per_class = _class_balanced_cross_entropy(
            fold_scores,
            labels[held_indices],
            class_count,
        )
        held_ce_by_fold_and_class.append(fold_per_class)
    macro, per_class = _class_balanced_cross_entropy(
        np.concatenate(held_scores, axis=0),
        np.concatenate(held_labels, axis=0),
        class_count,
    )
    held_flat = [value for fold in held_indices_by_fold for value in fold]
    expected_indices = list(range(len(rows)))
    partition_audit = {
        "partition_unit": "per_class_support_row_rank",
        "held_support_row_indices_by_fold": held_indices_by_fold,
        "held_ce_by_fold_and_class": held_ce_by_fold_and_class,
        "train_held_overlap_count": train_held_overlap_count,
        "held_support_row_count": len(held_flat),
        "held_support_row_unique_count": len(set(held_flat)),
        "held_support_row_exact_once_coverage": sorted(held_flat) == expected_indices,
        "train_rows_per_fold": len(rows) - class_count,
        "held_rows_per_fold": class_count,
    }
    if (
        train_held_overlap_count != 0
        or not partition_audit["held_support_row_exact_once_coverage"]
        or partition_audit["held_support_row_unique_count"] != len(rows)
    ):
        raise D45ProbeError("D45 head-only LOO partition closure drift")
    return macro, per_class, partition_audit


def _likelihood_weights(
    full_macro_ce: float,
    block_macro_ce: float,
    class_count: int,
) -> tuple[float, float, list[float]]:
    if (
        not np.isfinite(full_macro_ce)
        or not np.isfinite(block_macro_ce)
        or int(class_count) <= 0
    ):
        raise D45ProbeError("D45 reliability CE became invalid")
    log_reliability = -float(class_count) * np.asarray(
        [full_macro_ce, block_macro_ce], dtype=np.float64
    )
    raw_log_reliability = [float(value) for value in log_reliability]
    log_reliability -= np.max(log_reliability)
    reliability = np.exp(log_reliability)
    weights = reliability / np.sum(reliability)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise D45ProbeError("D45 reliability weights became invalid")
    return float(weights[0]), float(weights[1]), raw_log_reliability


def _build_locked_d42_full_component_fit(
    d42: Any,
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Reuse D42's locked sklearn solution; only remove a class-common affine term."""

    original_fit = d42._fit_equal_prior_lda

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficients, intercept, audit = original_fit(
            transformed, targets, class_count, k_shot
        )
        centered_coef, centered_intercept = d43._center_affine_scores(
            coefficients, intercept
        )
        rows32 = np.asarray(transformed, dtype=np.float32)
        original_scores = (
            rows32 @ np.asarray(coefficients, dtype=np.float32).T
            + np.asarray(intercept, dtype=np.float32)[None, :]
        )
        centered_coef32 = centered_coef.astype(np.float32)
        centered_intercept32 = centered_intercept.astype(np.float32)
        centered_scores = (
            rows32 @ centered_coef32.T + centered_intercept32[None, :]
        )
        if not np.array_equal(
            np.argmax(original_scores, axis=1), np.argmax(centered_scores, axis=1)
        ):
            raise D45ProbeError("D45 locked D42 full-component centering drift")
        result_audit = dict(audit)
        result_audit.update(
            {
                "d43_probe_arm": "full_centered_control",
                "d43_covariance_structure": "locked_d42_full_covariance",
                "d43_class_common_affine_omitted": True,
                "d45_full_component_uses_locked_d42_fit": True,
                "d45_full_component_refits_covariance": False,
            }
        )
        return centered_coef32, centered_intercept32, result_audit

    return fit


def build_inner_loo_reliability_fit(
    d42: Any,
) -> Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    full_fit = _build_locked_d42_full_component_fit(d42)
    block_fit = d43.build_structured_fit(d42, "block3_centered")

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
            full_macro_ce = None
            block_macro_ce = None
            full_per_class_ce = None
            block_per_class_ce = None
            full_weight = 0.5
            block_weight = 0.5
            log_evidence = None
            full_partition_audit = None
            block_partition_audit = None
            inner_fold_count = 0
            k1_equivalent_fallback = True
        else:
            full_macro_ce, full_per_class_ce, full_partition_audit = _inner_loo_component_ce(
                full_fit, transformed, targets, class_count, k_shot
            )
            block_macro_ce, block_per_class_ce, block_partition_audit = _inner_loo_component_ce(
                block_fit, transformed, targets, class_count, k_shot
            )
            full_weight, block_weight, log_evidence = _likelihood_weights(
                full_macro_ce,
                block_macro_ce,
                class_count,
            )
            if int(k_shot) == 2 and not (
                np.isclose(full_macro_ce, block_macro_ce, rtol=0.0, atol=1.0e-12)
                and np.isclose(full_weight, 0.5, rtol=0.0, atol=1.0e-12)
                and np.isclose(block_weight, 0.5, rtol=0.0, atol=1.0e-12)
            ):
                raise D45ProbeError("D45 K2 unit-covariance equal-weight closure drift")
            inner_fold_count = int(k_shot)
            k1_equivalent_fallback = False
        fused_coef64 = (
            full_weight * np.asarray(full_coef, dtype=np.float64) / full_scale
            + block_weight * np.asarray(block_coef, dtype=np.float64) / block_scale
        )
        fused_intercept64 = (
            full_weight
            * np.asarray(full_intercept, dtype=np.float64)
            / full_scale
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
            raise D45ProbeError("D45 fused affine state drift")
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
                    "d45_support_inner_loo_class_likelihood_"
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
                    float(max(component_condition_numbers))
                    if component_condition_numbers
                    else None
                ),
                "d45_probe_arm": ARM,
                "d45_component_arms": [
                    "full_centered_control",
                    "block3_centered",
                ],
                "d45_scale_formula": d44.SCALE_FORMULA,
                "d45_weight_formula": WEIGHT_FORMULA,
                "d45_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d45_inner_scope": INNER_SCOPE,
                "d45_outer_b20_frozen_across_inner_folds": True,
                "d45_outer_b20_refit_per_inner_fold": False,
                "d45_inner_loo_generalization_claim_allowed": False,
                "d45_full_support_logit_rms": full_scale,
                "d45_block_support_logit_rms": block_scale,
                "d45_full_inner_loo_macro_class_ce": full_macro_ce,
                "d45_block_inner_loo_macro_class_ce": block_macro_ce,
                "d45_full_inner_loo_ce_by_class": full_per_class_ce,
                "d45_block_inner_loo_ce_by_class": block_per_class_ce,
                "d45_log_evidence_by_component": log_evidence,
                "d45_full_inner_partition_audit": full_partition_audit,
                "d45_block_inner_partition_audit": block_partition_audit,
                "d45_full_weight": full_weight,
                "d45_block_weight": block_weight,
                "d45_inner_loo_fold_count": inner_fold_count,
                "d45_k1_equivalent_unit_covariance_fallback": k1_equivalent_fallback,
                "d45_k2_unit_covariance_equal_weight_verified": (
                    int(k_shot) == 2
                ),
                "d45_reliability_uses_support_labels": int(k_shot) > 1,
                "d45_reliability_uses_outer_held_or_query": False,
                "d45_role_handle_scene_specific_branch": False,
                "d45_weight_scan_count": 0,
            }
        )
        return coef32, intercept32, audit

    return fit


def _install_d45_core_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_lda_fit_macs = d42._lda_fit_macs
    original_top_level_fit = d42.fit_d42_unified_shrinkage_lda

    def d45_lda_fit_macs(row_count: int, class_count: int) -> int:
        if class_count <= 0 or row_count % class_count != 0:
            raise D45ProbeError("D45 resource audit requires exact equal K")
        k_shot = row_count // class_count
        main = 2 * int(original_lda_fit_macs(row_count, class_count))
        if k_shot <= 1:
            return main
        inner_rows = (k_shot - 1) * class_count
        inner = 2 * k_shot * int(original_lda_fit_macs(inner_rows, class_count))
        return main + inner

    def fit_with_d45_resource_audit(*args: Any, **kwargs: Any) -> Any:
        result = original_top_level_fit(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_k = int(resource["old_k_shot"])
        new_k = int(resource["new_k_shot"])
        if old_k != new_k:
            raise D45ProbeError("D45 resource audit K mismatch")
        inner_fit_count = 0 if old_k <= 1 else 4 * old_k
        old_class_count = len(result.before_state.classes)
        all_class_count = len(result.state.classes)
        fit_inventory = [
            {
                "fit_group": "before_main_components",
                "fit_count": 2,
                "row_count_per_fit": old_class_count * old_k,
                "class_count": old_class_count,
                "macs_per_fit": int(
                    original_lda_fit_macs(old_class_count * old_k, old_class_count)
                ),
            },
            {
                "fit_group": "final_main_components",
                "fit_count": 2,
                "row_count_per_fit": all_class_count * old_k,
                "class_count": all_class_count,
                "macs_per_fit": int(
                    original_lda_fit_macs(all_class_count * old_k, all_class_count)
                ),
            },
        ]
        if old_k > 1:
            fit_inventory.extend(
                [
                    {
                        "fit_group": "before_inner_head_only_components",
                        "fit_count": 2 * old_k,
                        "row_count_per_fit": old_class_count * (old_k - 1),
                        "class_count": old_class_count,
                        "macs_per_fit": int(
                            original_lda_fit_macs(
                                old_class_count * (old_k - 1), old_class_count
                            )
                        ),
                    },
                    {
                        "fit_group": "final_inner_head_only_components",
                        "fit_count": 2 * old_k,
                        "row_count_per_fit": all_class_count * (old_k - 1),
                        "class_count": all_class_count,
                        "macs_per_fit": int(
                            original_lda_fit_macs(
                                all_class_count * (old_k - 1), all_class_count
                            )
                        ),
                    },
                ]
            )
        inventory_macs = sum(
            int(item["fit_count"]) * int(item["macs_per_fit"])
            for item in fit_inventory
        )
        if inventory_macs != int(resource["estimated_lda_fit_macs"]):
            raise D45ProbeError("D45 LDA fit inventory MAC closure drift")
        resource.update(
            {
                "lda_closed_form_fit_count": 4 + inner_fit_count,
                "d45_component_main_fit_count": 4,
                "d45_inner_loo_component_fit_count": inner_fit_count,
                "d45_fused_query_state_count": 1,
                "d45_inner_scope": INNER_SCOPE,
                "d45_outer_b20_training_count": 1,
                "d45_lda_fit_inventory": fit_inventory,
                "d45_lda_fit_inventory_macs": inventory_macs,
            }
        )
        return replace(result, resource_audit=resource)

    d42._lda_fit_macs = d45_lda_fit_macs
    d42.fit_d42_unified_shrinkage_lda = fit_with_d45_resource_audit
    return original_lda_fit_macs, original_top_level_fit


def _verify_partition_evidence(
    partition: Any,
    *,
    class_count: int,
    k_shot: int,
    expected_per_class_ce: np.ndarray,
) -> None:
    if not isinstance(partition, dict):
        raise D45ProbeError("D45 head-only partition missing")
    held_indices = partition.get("held_support_row_indices_by_fold")
    fold_ce = partition.get("held_ce_by_fold_and_class")
    if (
        partition.get("partition_unit") != "per_class_support_row_rank"
        or partition.get("train_held_overlap_count") != 0
        or partition.get("held_support_row_exact_once_coverage") is not True
        or partition.get("train_rows_per_fold") != class_count * (k_shot - 1)
        or partition.get("held_rows_per_fold") != class_count
        or not isinstance(held_indices, list)
        or len(held_indices) != k_shot
        or not isinstance(fold_ce, list)
        or len(fold_ce) != k_shot
    ):
        raise D45ProbeError("D45 head-only partition structural drift")
    held_flat: list[int] = []
    for indices in held_indices:
        if (
            not isinstance(indices, list)
            or len(indices) != class_count
            or any(not isinstance(value, int) or isinstance(value, bool) for value in indices)
        ):
            raise D45ProbeError("D45 held support-row index drift")
        held_flat.extend(indices)
    expected_rows = class_count * k_shot
    if (
        sorted(held_flat) != list(range(expected_rows))
        or partition.get("held_support_row_count") != expected_rows
        or partition.get("held_support_row_unique_count") != expected_rows
    ):
        raise D45ProbeError("D45 held support-row exact-once closure drift")
    fold_ce_array = np.asarray(fold_ce, dtype=np.float64)
    if (
        fold_ce_array.shape != (k_shot, class_count)
        or not np.isfinite(fold_ce_array).all()
        or np.any(fold_ce_array < 0.0)
        or not np.allclose(
            np.mean(fold_ce_array, axis=0),
            expected_per_class_ce,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise D45ProbeError("D45 fold/class CE closure drift")


def _expected_lda_fit_macs(row_count: int, class_count: int, dimension: int) -> int:
    return int(
        row_count * dimension * dimension
        + dimension * dimension * dimension
        + class_count * dimension * dimension
    )


def _verify_fit_inventory(
    resource: dict[str, Any],
    *,
    old_class_count: int,
    all_class_count: int,
    k_shot: int,
) -> None:
    dimension = resource.get("coefficient_dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise D45ProbeError("D45 coefficient dimension drift")
    expected_specs = [
        ("before_main_components", 2, old_class_count * k_shot, old_class_count),
        ("final_main_components", 2, all_class_count * k_shot, all_class_count),
    ]
    if k_shot > 1:
        expected_specs.extend(
            [
                (
                    "before_inner_head_only_components",
                    2 * k_shot,
                    old_class_count * (k_shot - 1),
                    old_class_count,
                ),
                (
                    "final_inner_head_only_components",
                    2 * k_shot,
                    all_class_count * (k_shot - 1),
                    all_class_count,
                ),
            ]
        )
    inventory = resource.get("d45_lda_fit_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(expected_specs):
        raise D45ProbeError("D45 LDA fit inventory group drift")
    inventory_by_group = {
        item.get("fit_group"): item for item in inventory if isinstance(item, dict)
    }
    if set(inventory_by_group) != {spec[0] for spec in expected_specs}:
        raise D45ProbeError("D45 LDA fit inventory name drift")
    total_macs = 0
    total_fits = 0
    for group, fit_count, row_count, class_count in expected_specs:
        item = inventory_by_group[group]
        expected_macs = _expected_lda_fit_macs(row_count, class_count, dimension)
        required = {
            "fit_group": group,
            "fit_count": fit_count,
            "row_count_per_fit": row_count,
            "class_count": class_count,
            "macs_per_fit": expected_macs,
        }
        if any(item.get(name) != value for name, value in required.items()):
            raise D45ProbeError(f"D45 LDA fit inventory drift for {group}")
        total_fits += fit_count
        total_macs += fit_count * expected_macs
    if (
        total_fits != resource.get("lda_closed_form_fit_count")
        or total_macs != resource.get("estimated_lda_fit_macs")
        or total_macs != resource.get("d45_lda_fit_inventory_macs")
    ):
        raise D45ProbeError("D45 LDA fit inventory total drift")


def _verify_d45_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d45_rows = [
        row
        for row in training_rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d45_rows) != 30:
        raise D45ProbeError("D45 training-row closure drift")
    for row in d45_rows:
        resource = row.get("resource")
        if not isinstance(resource, dict):
            raise D45ProbeError("D45 row resource audit missing")
        k_shot = int(resource.get("old_k_shot", -1))
        if int(resource.get("new_k_shot", -2)) != k_shot:
            raise D45ProbeError("D45 row K audit mismatch")
        expected_inner = 0 if k_shot <= 1 else 4 * k_shot
        expected_total = 4 + expected_inner
        required_resource = {
            "lda_closed_form_fit_count": expected_total,
            "d45_component_main_fit_count": 4,
            "d45_inner_loo_component_fit_count": expected_inner,
            "d45_fused_query_state_count": 1,
            "d45_inner_scope": INNER_SCOPE,
            "d45_outer_b20_training_count": 1,
        }
        for name, expected in required_resource.items():
            if resource.get(name) != expected:
                raise D45ProbeError(f"D45 resource audit drift for {name}")
        if resource.get("estimated_adaptation_macs") != (
            resource.get("estimated_metric_adaptation_macs", -1)
            + resource.get("estimated_lda_fit_macs", -2)
        ):
            raise D45ProbeError("D45 adaptation MAC closure drift")
        geometry = row.get("geometry_summary")
        if not isinstance(geometry, dict):
            raise D45ProbeError("D45 geometry summary missing")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field)
            if not isinstance(audit, dict):
                raise D45ProbeError(f"D45 fit audit missing from {field}")
            required = {
                "d45_probe_arm": ARM,
                "d45_component_arms": [
                    "full_centered_control",
                    "block3_centered",
                ],
                "d45_scale_formula": d44.SCALE_FORMULA,
                "d45_weight_formula": WEIGHT_FORMULA,
                "d45_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d45_inner_scope": INNER_SCOPE,
                "d45_outer_b20_frozen_across_inner_folds": True,
                "d45_outer_b20_refit_per_inner_fold": False,
                "d45_inner_loo_generalization_claim_allowed": False,
                "d45_reliability_uses_outer_held_or_query": False,
                "d45_role_handle_scene_specific_branch": False,
                "d45_weight_scan_count": 0,
            }
            for name, expected in required.items():
                if audit.get(name) != expected:
                    raise D45ProbeError(f"D45 fit audit drift for {field}.{name}")
            weights = [audit.get("d45_full_weight"), audit.get("d45_block_weight")]
            if (
                not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and np.isfinite(float(value))
                    and float(value) > 0.0
                    for value in weights
                )
                or not np.isclose(sum(float(value) for value in weights), 1.0)
            ):
                raise D45ProbeError("D45 reliability weight closure drift")
            if k_shot > 1:
                if (
                    audit.get("d45_inner_loo_fold_count") != k_shot
                    or audit.get("d45_reliability_uses_support_labels") is not True
                    or audit.get("d45_k1_equivalent_unit_covariance_fallback")
                    is not False
                ):
                    raise D45ProbeError("D45 inner-LOO audit drift")
                full_per_class = np.asarray(
                    audit.get("d45_full_inner_loo_ce_by_class"), dtype=np.float64
                )
                block_per_class = np.asarray(
                    audit.get("d45_block_inner_loo_ce_by_class"), dtype=np.float64
                )
                if (
                    full_per_class.ndim != 1
                    or block_per_class.shape != full_per_class.shape
                    or len(full_per_class) <= 0
                    or not np.isfinite(full_per_class).all()
                    or not np.isfinite(block_per_class).all()
                    or np.any(full_per_class < 0.0)
                    or np.any(block_per_class < 0.0)
                ):
                    raise D45ProbeError("D45 per-class CE drift")
                class_count = len(full_per_class)
                macro_values = np.asarray(
                    [
                        audit.get("d45_full_inner_loo_macro_class_ce"),
                        audit.get("d45_block_inner_loo_macro_class_ce"),
                    ],
                    dtype=np.float64,
                )
                if (
                    not np.isfinite(macro_values).all()
                    or np.any(macro_values < 0.0)
                    or not np.allclose(
                        macro_values,
                        [np.mean(full_per_class), np.mean(block_per_class)],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise D45ProbeError("D45 macro/per-class CE closure drift")
                _verify_partition_evidence(
                    audit.get("d45_full_inner_partition_audit"),
                    class_count=class_count,
                    k_shot=k_shot,
                    expected_per_class_ce=full_per_class,
                )
                _verify_partition_evidence(
                    audit.get("d45_block_inner_partition_audit"),
                    class_count=class_count,
                    k_shot=k_shot,
                    expected_per_class_ce=block_per_class,
                )
                expected_full_weight, expected_block_weight, expected_log_evidence = (
                    _likelihood_weights(
                        float(macro_values[0]),
                        float(macro_values[1]),
                        class_count,
                    )
                )
                if (
                    not np.allclose(
                        audit.get("d45_log_evidence_by_component"),
                        expected_log_evidence,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        weights,
                        [expected_full_weight, expected_block_weight],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise D45ProbeError("D45 CE/log-evidence/weight closure drift")
                if k_shot == 2 and (
                    audit.get("d45_k2_unit_covariance_equal_weight_verified")
                    is not True
                    or not np.isclose(
                        float(macro_values[0]),
                        float(macro_values[1]),
                        atol=1.0e-12,
                        rtol=0.0,
                    )
                    or not np.isclose(float(weights[0]), 0.5, atol=1.0e-12, rtol=0.0)
                    or not np.isclose(float(weights[1]), 0.5, atol=1.0e-12, rtol=0.0)
                ):
                    raise D45ProbeError("D45 K2 equal-weight verifier drift")
            elif (
                audit.get("d45_inner_loo_fold_count") != 0
                or audit.get("d45_reliability_uses_support_labels") is not False
                or audit.get("d45_k1_equivalent_unit_covariance_fallback") is not True
                or audit.get("d45_full_inner_loo_macro_class_ce") is not None
                or audit.get("d45_block_inner_loo_macro_class_ce") is not None
                or audit.get("d45_full_inner_loo_ce_by_class") is not None
                or audit.get("d45_block_inner_loo_ce_by_class") is not None
                or audit.get("d45_log_evidence_by_component") is not None
                or audit.get("d45_full_inner_partition_audit") is not None
                or audit.get("d45_block_inner_partition_audit") is not None
                or not np.allclose(weights, [0.5, 0.5], rtol=0.0, atol=0.0)
            ):
                raise D45ProbeError("D45 K1 fallback evidence drift")
        before_audit = geometry["before_covariance_audit"]
        final_audit = geometry["final_covariance_audit"]
        old_class_count = (
            len(before_audit["d45_full_inner_loo_ce_by_class"])
            if k_shot > 1
            else int(row.get("before_old", {}).get("per_class_accuracy", {}).__len__())
        )
        all_class_count = (
            len(final_audit["d45_full_inner_loo_ce_by_class"])
            if k_shot > 1
            else old_class_count
            + int(row.get("after_new", {}).get("per_class_accuracy", {}).__len__())
        )
        if (
            old_class_count <= 0
            or all_class_count != resource.get("registered_class_count")
        ):
            raise D45ProbeError("D45 registry class-count closure drift")
        _verify_fit_inventory(
            resource,
            old_class_count=old_class_count,
            all_class_count=all_class_count,
            k_shot=k_shot,
        )
        if (
            geometry.get("before_materialized_pre_stage2c") is not True
            or geometry.get("before_state_immutable_during_stage2c") is not True
            or int(geometry.get("old_only_metric_new_support_argument_count", -1))
            != 0
        ):
            raise D45ProbeError("D45 before/new-support lifecycle drift")
    return len(d45_rows)


def _verify_d45_output(
    output: Path,
    probe_script_sha256: str,
    d44_helper_sha256: str,
    d43_helper_sha256: str,
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    source_closure = support.get("candidate_lock", {}).get("source_closure", {})
    if (
        source_closure.get("d45_d44_helper_sha256") != d44_helper_sha256
        or source_closure.get("d45_d43_helper_sha256") != d43_helper_sha256
    ):
        raise D45ProbeError("D45 helper source closure drift")
    training_rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    count = _verify_d45_fit_audits(training_rows)
    return {
        **evidence,
        "verified_d45_fit_row_count": count,
        "verified_d44_helper_sha256": d44_helper_sha256,
        "verified_d43_helper_sha256": d43_helper_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d45-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D45ProbeError(f"D45 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_lda_fit_macs = None
    original_top_level_fit = None
    runner_module_name = "d45_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d44_helper_sha256 = d43._sha256(D44_HELPER_PATH)
    d43_helper_sha256 = d43._sha256(d44.D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_inner_loo_reliability_fit(d42)
        original_lda_fit_macs, original_top_level_fit = (
            _install_d45_core_resource_accounting(d42)
        )
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D45ProbeError("D45 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d45_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d45_d44_helper_sha256": d44_helper_sha256,
                "d45_d43_helper_sha256": d43_helper_sha256,
            },
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_lda_fit_macs is not None:
            d42._lda_fit_macs = original_lda_fit_macs
        if d42 is not None and original_top_level_fit is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top_level_fit
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_d45_output(
        output,
        probe_script_sha256,
        d44_helper_sha256,
        d43_helper_sha256,
    )
    metadata = {
        "schema": "cvs.phase2.d45.inner_loo_reliability_fusion_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d45_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "d44_helper_sha256": d44_helper_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "weight_formula": WEIGHT_FORMULA,
        "log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "inner_scope": INNER_SCOPE,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D45_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
