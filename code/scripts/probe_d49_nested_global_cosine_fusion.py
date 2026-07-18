#!/usr/bin/env python3
"""D49 probe: strict-nested D45/global unit-sphere cosine affine fusion."""

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
D45_SPEC = importlib.util.spec_from_file_location("d49_d45_probe_helper", D45_HELPER_PATH)
if D45_SPEC is None or D45_SPEC.loader is None:
    raise RuntimeError("D49 could not load the D45 probe helper")
d45 = importlib.util.module_from_spec(D45_SPEC)
D45_SPEC.loader.exec_module(d45)
d43 = d45.d43
d44 = d45.d44


ARM = "nested_global_cosine_fusion"
STRUCTURE = "strict_nested_d45_global_unit_sphere_cosine_prototype_affine_fusion"
WEIGHT_FORMULA = "w=softmax(-registered_class_count*[macro_ce_d45,macro_ce_cosine])"
QUERY_VIEW = "full_288d_only"
COEFFICIENT_BLOCK_BOUNDS = ((0, 160), (160, 256), (256, 288))
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D49ProbeError(RuntimeError):
    pass


def _audit_quantize_coefficients(
    coefficients: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(coefficients, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != COEFFICIENT_BLOCK_BOUNDS[-1][1]
        or not np.isfinite(rows).all()
    ):
        raise D49ProbeError("D49 coefficient quantization input drift")
    code1 = np.zeros(rows.shape, dtype=np.int8)
    code2 = np.zeros(rows.shape, dtype=np.int8)
    scale1 = np.empty((len(rows), len(COEFFICIENT_BLOCK_BOUNDS)), dtype=np.float16)
    scale2 = np.empty_like(scale1)
    smallest = np.nextafter(np.float16(0), np.float16(1))
    for row_index, row in enumerate(rows):
        for block_index, (start, stop) in enumerate(COEFFICIENT_BLOCK_BOUNDS):
            values = row[start:stop]
            first_scale = np.float16(
                max(float(np.max(np.abs(values))) / 127.0, float(smallest))
            )
            first_code = np.clip(
                np.rint(values / np.float32(first_scale)), -127, 127
            ).astype(np.int8)
            residual = values - np.float32(first_scale) * first_code.astype(np.float32)
            second_scale = np.float16(
                max(float(np.max(np.abs(residual))) / 127.0, float(smallest))
            )
            if (
                not np.isfinite(first_scale)
                or first_scale <= 0
                or not np.isfinite(second_scale)
                or second_scale <= 0
            ):
                raise D49ProbeError("D49 coefficient quantization scale drift")
            second_code = np.clip(
                np.rint(residual / np.float32(second_scale)), -127, 127
            ).astype(np.int8)
            code1[row_index, start:stop] = first_code
            code2[row_index, start:stop] = second_code
            scale1[row_index, block_index] = first_scale
            scale2[row_index, block_index] = second_scale
    return code1, code2, scale1, scale2


def _exact_json_array(value: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise D49ProbeError(f"D49 {name} JSON array drift") from exc
    target = np.asarray(raw, dtype=dtype)
    if (
        raw.shape != shape
        or not np.isfinite(raw).all()
        or not np.array_equal(raw, target.astype(np.float64))
    ):
        raise D49ProbeError(f"D49 {name} JSON exact-dtype drift")
    return target


def _cosine_component_fit(
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    energy_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if (
        rows.ndim != 2
        or len(rows) != int(class_count) * int(k_shot)
        or labels.shape != (len(rows),)
        or int(class_count) < 2
        or int(k_shot) < 1
        or not np.isfinite(rows).all()
        or np.any(labels < 0)
        or np.any(labels >= int(class_count))
    ):
        raise D49ProbeError("D49 cosine support shape/coverage drift")
    row_norms = np.linalg.norm(rows, axis=1)
    if not np.allclose(row_norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise D49ProbeError("D49 requires D42 global unit-sphere features")
    means = np.stack(
        [rows[labels == index].mean(axis=0) for index in range(int(class_count))]
    )
    resultant_norms = np.linalg.norm(means, axis=1)
    if (
        not np.isfinite(resultant_norms).all()
        or np.any(resultant_norms <= float(energy_epsilon))
    ):
        raise D49ProbeError("D49 cosine prototype resultant norm degenerated")
    prototypes = means / resultant_norms[:, None]
    intercept = np.zeros(int(class_count), dtype=np.float64)
    return (
        prototypes.astype(np.float32),
        intercept.astype(np.float32),
        {
            "d49_cosine_geometry": "D42_global_unit_sphere_cosine_prototype",
            "d49_cosine_query_reference": "x_dot_normalized_class_mean",
            "d49_cosine_affine_intercept_zero": True,
            "d49_cosine_prototype_resultant_norm_by_class": resultant_norms.tolist(),
            "d49_cosine_prototype_fp32": prototypes.astype(np.float32).astype(np.float64).tolist(),
            "d49_global_unit_sphere_row_norm_max_abs_error": float(
                np.max(np.abs(row_norms - 1.0))
            ),
        },
    )


def _strict_weights(
    d45_macro_ce: float, cosine_macro_ce: float, class_count: int
) -> tuple[float, float, list[float]]:
    values = np.asarray([d45_macro_ce, cosine_macro_ce], dtype=np.float64)
    if int(class_count) < 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise D49ProbeError("D49 nested CE became invalid")
    with np.errstate(over="ignore", invalid="ignore"):
        evidence = -float(class_count) * values
    if not np.isfinite(evidence).all():
        raise D49ProbeError("D49 global fusion weight endpoint/sum drift")
    shifted = evidence - np.max(evidence)
    probability = np.exp(shifted)
    weights = probability / np.sum(probability)
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or np.any(weights >= 1.0)
        or abs(float(np.sum(weights)) - 1.0) > 1.0e-12
    ):
        raise D49ProbeError("D49 global fusion weight endpoint/sum drift")
    if d45_macro_ce == cosine_macro_ce and not np.array_equal(
        weights, np.asarray([0.5, 0.5], dtype=np.float64)
    ):
        raise D49ProbeError("D49 exact CE tie must produce exact half weights")
    return float(weights[0]), float(weights[1]), evidence.tolist()


def _nested_head_evidence(
    d45_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    energy_epsilon: float,
) -> tuple[float, float, dict[str, Any]]:
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    indices_by_class = [np.flatnonzero(labels == index) for index in range(class_count)]
    if int(k_shot) <= 1 or any(len(value) != int(k_shot) for value in indices_by_class):
        raise D49ProbeError("D49 nested evidence requires exact K>1 per class")
    d45_scores: list[np.ndarray] = []
    cosine_scores: list[np.ndarray] = []
    held_labels: list[np.ndarray] = []
    held_indices_by_fold: list[list[int]] = []
    train_indices_by_fold: list[list[int]] = []
    d45_train_rms: list[float] = []
    cosine_train_rms: list[float] = []
    nested_d45_audits: list[dict[str, Any]] = []
    cosine_audits: list[dict[str, Any]] = []
    nested_d45_coefficients: list[list[list[float]]] = []
    nested_d45_intercepts: list[list[float]] = []
    for rank in range(int(k_shot)):
        held_indices = np.asarray(
            [indices_by_class[index][rank] for index in range(class_count)],
            dtype=np.int64,
        )
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[held_indices] = False
        train_indices = np.flatnonzero(train_mask)
        if len(np.intersect1d(train_indices, held_indices)) != 0:
            raise D49ProbeError("D49 nested train/held overlap")
        d_coef, d_intercept, d_audit = d45_fit(
            rows[train_mask], labels[train_mask], class_count, int(k_shot) - 1
        )
        c_coef, c_intercept, c_audit = _cosine_component_fit(
            rows[train_mask],
            labels[train_mask],
            class_count,
            int(k_shot) - 1,
            energy_epsilon=energy_epsilon,
        )
        d_scale = d44._class_centered_logit_rms(
            rows[train_mask], d_coef, d_intercept
        )
        c_scale = d44._class_centered_logit_rms(
            rows[train_mask], c_coef, c_intercept
        )
        if not np.isfinite([d_scale, c_scale]).all() or d_scale <= 0.0 or c_scale <= 0.0:
            raise D49ProbeError("D49 inner-train head RMS degenerated")
        d45_scores.append(
            (rows[held_indices] @ np.asarray(d_coef, dtype=np.float64).T
             + np.asarray(d_intercept, dtype=np.float64)[None, :]) / d_scale
        )
        cosine_scores.append(
            (rows[held_indices] @ np.asarray(c_coef, dtype=np.float64).T) / c_scale
        )
        held_labels.append(labels[held_indices])
        held_indices_by_fold.append(held_indices.tolist())
        train_indices_by_fold.append(train_indices.tolist())
        d45_train_rms.append(float(d_scale))
        cosine_train_rms.append(float(c_scale))
        nested_d45_audits.append(dict(d_audit))
        cosine_audits.append(dict(c_audit))
        nested_d45_coefficients.append(
            np.asarray(d_coef, dtype=np.float32).astype(np.float64).tolist()
        )
        nested_d45_intercepts.append(
            np.asarray(d_intercept, dtype=np.float32).astype(np.float64).tolist()
        )
    d45_macro, d45_by_class = d45._class_balanced_cross_entropy(
        np.concatenate(d45_scores), np.concatenate(held_labels), class_count
    )
    cosine_macro, cosine_by_class = d45._class_balanced_cross_entropy(
        np.concatenate(cosine_scores), np.concatenate(held_labels), class_count
    )
    held_flat = [value for fold in held_indices_by_fold for value in fold]
    exact_once = sorted(held_flat) == list(range(len(rows)))
    if not exact_once or len(set(held_flat)) != len(rows):
        raise D49ProbeError("D49 nested held exact-once coverage drift")
    return d45_macro, cosine_macro, {
        "partition_unit": "per_class_support_row_rank",
        "held_support_row_indices_by_fold": held_indices_by_fold,
        "train_support_row_indices_by_fold": train_indices_by_fold,
        "train_held_overlap_count": 0,
        "held_support_row_count": len(held_flat),
        "held_support_row_unique_count": len(set(held_flat)),
        "held_support_row_exact_once_coverage": exact_once,
        "train_rows_per_fold": class_count * (int(k_shot) - 1),
        "held_rows_per_fold": class_count,
        "d49_d45_inner_train_logit_rms_by_fold": d45_train_rms,
        "d49_cosine_inner_train_logit_rms_by_fold": cosine_train_rms,
        "d49_d45_outer_held_macro_class_ce": d45_macro,
        "d49_cosine_outer_held_macro_class_ce": cosine_macro,
        "d49_d45_outer_held_ce_by_class": d45_by_class,
        "d49_cosine_outer_held_ce_by_class": cosine_by_class,
        "d49_d45_held_scores_by_fold": np.stack(d45_scores).tolist(),
        "d49_cosine_held_scores_by_fold": np.stack(cosine_scores).tolist(),
        "d49_nested_d45_fit_audit_by_fold": nested_d45_audits,
        "d49_nested_cosine_fit_audit_by_fold": cosine_audits,
        "d49_nested_d45_coefficient_fp32_by_fold": nested_d45_coefficients,
        "d49_nested_d45_intercept_fp32_by_fold": nested_d45_intercepts,
    }


def build_nested_global_cosine_fit(d42: Any) -> Any:
    d45_fit = d45.build_inner_loo_reliability_fit(d42)

    def fit(transformed: np.ndarray, targets: np.ndarray, class_count: int, k_shot: int):
        d_coef, d_intercept, d_audit = d45_fit(
            transformed, targets, class_count, k_shot
        )
        if int(k_shot) == 1:
            return d_coef, d_intercept, d_audit
        c_coef, c_intercept, c_audit = _cosine_component_fit(
            transformed,
            targets,
            class_count,
            k_shot,
            energy_epsilon=d42.ENERGY_EPSILON,
        )
        d_scale = d44._class_centered_logit_rms(
            transformed, d_coef, d_intercept
        )
        c_scale = d44._class_centered_logit_rms(
            transformed, c_coef, c_intercept
        )
        if not np.isfinite([d_scale, c_scale]).all() or d_scale <= 0.0 or c_scale <= 0.0:
            raise D49ProbeError("D49 full-support head RMS degenerated")
        d_ce, c_ce, partition = _nested_head_evidence(
            d45_fit,
            transformed,
            targets,
            class_count,
            k_shot,
            energy_epsilon=d42.ENERGY_EPSILON,
        )
        d_weight, c_weight, evidence = _strict_weights(d_ce, c_ce, class_count)
        fused_coef = (
            d_weight * np.asarray(d_coef, dtype=np.float64) / d_scale
            + c_weight * np.asarray(c_coef, dtype=np.float64) / c_scale
        )
        fused_intercept = (
            d_weight * np.asarray(d_intercept, dtype=np.float64) / d_scale
        )
        centered_coef, centered_intercept = d43._center_affine_scores(
            fused_coef, fused_intercept
        )
        coef32 = centered_coef.astype(np.float32)
        intercept32 = centered_intercept.astype(np.float32)
        if not np.isfinite(coef32).all() or not np.isfinite(intercept32).all():
            raise D49ProbeError("D49 fused affine state became non-finite")
        audit = dict(d_audit)
        audit.update(c_audit)
        audit.update(
            {
                "coefficient_source": "d49_single_fp32_d45_global_cosine_affine_fusion",
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d49_probe_arm": ARM,
                "d49_weight_formula": WEIGHT_FORMULA,
                "d49_query_view": QUERY_VIEW,
                "d49_d45_full_support_logit_rms": float(d_scale),
                "d49_cosine_full_support_logit_rms": float(c_scale),
                "d49_d45_nested_macro_class_ce": float(d_ce),
                "d49_cosine_nested_macro_class_ce": float(c_ce),
                "d49_log_evidence_by_head": evidence,
                "d49_d45_weight": d_weight,
                "d49_cosine_weight": c_weight,
                "d49_support_transformed_fp32": (
                    np.asarray(transformed, dtype=np.float32).astype(np.float64).tolist()
                ),
                "d49_support_targets": (
                    np.asarray(targets, dtype=np.int64).tolist()
                ),
                "d49_d45_full_support_coefficient_fp32": (
                    np.asarray(d_coef, dtype=np.float32).astype(np.float64).tolist()
                ),
                "d49_d45_full_support_intercept_fp32": (
                    np.asarray(d_intercept, dtype=np.float32).astype(np.float64).tolist()
                ),
                "d49_final_fused_coefficient_fp32": coef32.astype(np.float64).tolist(),
                "d49_final_fused_intercept_fp32": intercept32.astype(np.float64).tolist(),
                "d49_nested_partition_audit": partition,
                "d49_outer_fold_count": int(k_shot),
                "d49_complete_d45_refit_per_outer_fold": True,
                "d49_inner_train_rms_used_for_held": True,
                "d49_full_support_rms_used_for_final_affine": True,
                "d49_full_support_fused_once_pre_quantization": True,
                "d49_quantized_component_fused_or_decoded": False,
                "d49_role_handle_scene_specific_branch": False,
                "d49_scan_temperature_threshold_count": 0,
            }
        )
        return coef32, intercept32, audit

    return fit


def _d45_fit_specs(prefix: str, class_count: int, k_shot: int) -> list[tuple[str, int, int, int]]:
    c = int(class_count)
    k = int(k_shot)
    specs = [(f"{prefix}_outer_main_d45_main_components", 2, c * k, c)]
    if k > 1:
        specs.extend(
            [
                (f"{prefix}_outer_main_d45_inner_components", 2 * k, c * (k - 1), c),
                (f"{prefix}_nested_inner_d45_main_components", 2 * k, c * (k - 1), c),
            ]
        )
    if k > 2:
        specs.append(
            (
                f"{prefix}_nested_inner_d45_inner_components",
                2 * k * (k - 1),
                c * (k - 2),
                c,
            )
        )
    return specs


def _extra_head_macs(class_count: int, k_shot: int, dimension: int) -> dict[str, int]:
    c, k, d = int(class_count), int(k_shot), int(dimension)
    if k <= 1:
        return {"prototype": 0, "rms": 0, "held_score": 0, "fusion": 0}
    full_rows = c * k
    inner_rows = c * (k - 1)
    prototype = full_rows * d + 3 * c * d + k * (inner_rows * d + 3 * c * d)
    rms = 4 * full_rows * c * d + 6 * full_rows * c
    rms += k * (4 * inner_rows * c * d + 6 * inner_rows * c)
    held_score = k * 4 * c * c * d
    fusion = 4 * c * d + 2 * c
    return {
        "prototype": int(prototype),
        "rms": int(rms),
        "held_score": int(held_score),
        "fusion": int(fusion),
    }


def _assert_no_exact_top_tie(scores: np.ndarray, name: str) -> None:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise D49ProbeError(f"D49 {name} score shape/nonfinite drift")
    ordered = np.sort(values, axis=1)
    if np.any(ordered[:, -1] == ordered[:, -2]):
        raise D49ProbeError(f"D49 exact top tie fail-close: {name}")


def _install_runner_score_tie_guard(runner_module: Any) -> tuple[Any, dict[str, int]]:
    original_score = runner_module.score_d42_unified_shrinkage_lda
    checked = {"fp32_rows": 0, "int8_rows": 0, "call_count": 0}

    def guarded_score(state: Any, features: np.ndarray) -> np.ndarray:
        scores = original_score(state, features)
        fp32 = np.asarray(getattr(state, "coef_fp32", np.empty(0))).size > 0
        precision = "fp32" if fp32 else "int8"
        _assert_no_exact_top_tie(scores, f"runner_{precision}")
        checked[f"{precision}_rows"] += int(np.asarray(scores).shape[0])
        checked["call_count"] += 1
        return scores

    runner_module.score_d42_unified_shrinkage_lda = guarded_score
    return original_score, checked


def _verify_complete_nested_d45_audit(
    audit: Any, *, class_count: int, k_shot: int
) -> None:
    if not isinstance(audit, dict):
        raise D49ProbeError("D49 nested D45 audit missing")
    required = {
        "d45_probe_arm": d45.ARM,
        "d45_component_arms": ["full_centered_control", "block3_centered"],
        "d45_scale_formula": d45.d44.SCALE_FORMULA,
        "d45_weight_formula": d45.WEIGHT_FORMULA,
        "d45_log_evidence_formula": d45.LOG_EVIDENCE_FORMULA,
        "d45_inner_scope": d45.INNER_SCOPE,
        "d45_outer_b20_frozen_across_inner_folds": True,
        "d45_outer_b20_refit_per_inner_fold": False,
        "d45_inner_loo_generalization_claim_allowed": False,
        "d45_reliability_uses_outer_held_or_query": False,
        "d45_role_handle_scene_specific_branch": False,
        "d45_weight_scan_count": 0,
    }
    if any(audit.get(name) != value for name, value in required.items()):
        raise D49ProbeError("D49 nested D45 locked audit drift")
    component_rms = np.asarray(
        [
            audit.get("d45_full_support_logit_rms"),
            audit.get("d45_block_support_logit_rms"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(component_rms).all() or np.any(component_rms <= 0.0):
        raise D49ProbeError("D49 nested D45 component RMS drift")
    weights = np.asarray(
        [audit.get("d45_full_weight"), audit.get("d45_block_weight")],
        dtype=np.float64,
    )
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise D49ProbeError("D49 nested D45 weight closure drift")
    if int(k_shot) == 1:
        if (
            audit.get("d45_inner_loo_fold_count") != 0
            or audit.get("d45_reliability_uses_support_labels") is not False
            or audit.get("d45_k1_equivalent_unit_covariance_fallback") is not True
            or audit.get("d45_full_inner_loo_macro_class_ce") is not None
            or audit.get("d45_block_inner_loo_macro_class_ce") is not None
            or not np.array_equal(weights, np.asarray([0.5, 0.5]))
        ):
            raise D49ProbeError("D49 nested D45 K1 fallback drift")
        return
    full_by_class = np.asarray(
        audit.get("d45_full_inner_loo_ce_by_class"), dtype=np.float64
    )
    block_by_class = np.asarray(
        audit.get("d45_block_inner_loo_ce_by_class"), dtype=np.float64
    )
    macros = np.asarray(
        [
            audit.get("d45_full_inner_loo_macro_class_ce"),
            audit.get("d45_block_inner_loo_macro_class_ce"),
        ],
        dtype=np.float64,
    )
    if (
        full_by_class.shape != (class_count,)
        or block_by_class.shape != (class_count,)
        or not np.isfinite(full_by_class).all()
        or not np.isfinite(block_by_class).all()
        or np.any(full_by_class < 0.0)
        or np.any(block_by_class < 0.0)
        or not np.allclose(
            macros,
            [full_by_class.mean(), block_by_class.mean()],
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise D49ProbeError("D49 nested D45 CE closure drift")
    d45._verify_partition_evidence(
        audit.get("d45_full_inner_partition_audit"),
        class_count=class_count,
        k_shot=k_shot,
        expected_per_class_ce=full_by_class,
    )
    d45._verify_partition_evidence(
        audit.get("d45_block_inner_partition_audit"),
        class_count=class_count,
        k_shot=k_shot,
        expected_per_class_ce=block_by_class,
    )
    expected_full, expected_block, expected_evidence = d45._likelihood_weights(
        float(macros[0]), float(macros[1]), class_count
    )
    if (
        audit.get("d45_inner_loo_fold_count") != k_shot
        or audit.get("d45_reliability_uses_support_labels") is not True
        or audit.get("d45_k1_equivalent_unit_covariance_fallback") is not False
        or not np.allclose(
            weights, [expected_full, expected_block], rtol=0.0, atol=1.0e-12
        )
        or not np.allclose(
            audit.get("d45_log_evidence_by_component"),
            expected_evidence,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise D49ProbeError("D49 nested D45 likelihood evidence drift")


def _install_d49_resource_and_tie_audit(d42: Any) -> tuple[Any, Any]:
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda

    def d49_lda_fit_macs(row_count: int, class_count: int) -> int:
        if class_count < 2 or row_count % class_count != 0:
            raise D49ProbeError("D49 resource audit requires exact equal K")
        k = row_count // class_count
        return int(
            sum(
                count * int(original_macs(rows, classes))
                for _group, count, rows, classes in _d45_fit_specs(
                    "state", class_count, k
                )
            )
        )

    def fit_with_audit(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        if int(resource["new_k_shot"]) != k:
            raise D49ProbeError("D49 before/final K mismatch")
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        inventory = []
        for prefix, count in (("before", old_count), ("final", all_count)):
            for group, fit_count, row_count, class_count in _d45_fit_specs(prefix, count, k):
                inventory.append(
                    {
                        "fit_group": group,
                        "fit_count": fit_count,
                        "row_count_per_fit": row_count,
                        "class_count": class_count,
                        "macs_per_fit": int(original_macs(row_count, class_count)),
                    }
                )
        lda_count = sum(int(item["fit_count"]) for item in inventory)
        lda_macs = sum(
            int(item["fit_count"]) * int(item["macs_per_fit"])
            for item in inventory
        )
        if lda_macs != int(resource["estimated_lda_fit_macs"]):
            raise D49ProbeError("D49 LDA MAC inventory closure drift")
        before_extra = _extra_head_macs(old_count, k, d42.FEATURE_DIM)
        final_extra = _extra_head_macs(all_count, k, d42.FEATURE_DIM)
        extra = {name: before_extra[name] + final_extra[name] for name in before_extra}
        extra_total = sum(extra.values())
        old_features = np.asarray(
            args[0] if args else kwargs["old_support_features"]
        )
        old_labels = list(
            args[1] if len(args) > 1 else kwargs["old_support_labels"]
        )
        old_classes = list(args[2] if len(args) > 2 else kwargs["old_classes"])
        new_features = np.asarray(
            args[3] if len(args) > 3 else kwargs["new_support_features"]
        )
        new_labels = list(
            args[4] if len(args) > 4 else kwargs["new_support_labels"]
        )
        new_classes = list(args[5] if len(args) > 5 else kwargs["new_classes"])
        before_states = (
            ("before_fp32", result.matched_fp32_before_state, old_features),
            ("before_int8", result.before_state, old_features),
        )
        final_features = np.concatenate([old_features, new_features], axis=0)
        before_transformed = d42._transform(
            old_features, result.before_state.log_diag_fp32
        )
        final_transformed = d42._transform(
            final_features, result.state.log_diag_fp32
        )
        before_fit_evidence = result.geometry_audit["before_covariance_audit"]
        final_fit_evidence = result.geometry_audit["final_covariance_audit"]
        old_index = {label: index for index, label in enumerate(old_classes)}
        new_index = {
            label: len(old_classes) + index for index, label in enumerate(new_classes)
        }
        before_targets = np.asarray([old_index[label] for label in old_labels], dtype=np.int64)
        final_targets = np.concatenate(
            [before_targets, np.asarray([new_index[label] for label in new_labels], dtype=np.int64)]
        )
        if (
            not np.array_equal(
                before_transformed,
                np.asarray(
                    before_fit_evidence["d49_support_transformed_fp32"],
                    dtype=np.float32,
                ),
            )
            or not np.array_equal(
                final_transformed,
                np.asarray(
                    final_fit_evidence["d49_support_transformed_fp32"],
                    dtype=np.float32,
                ),
            )
            or not np.array_equal(
                before_targets,
                np.asarray(before_fit_evidence["d49_support_targets"], dtype=np.int64),
            )
            or not np.array_equal(
                final_targets,
                np.asarray(final_fit_evidence["d49_support_targets"], dtype=np.int64),
            )
        ):
            raise D49ProbeError("D49 support transform/runtime input binding drift")
        final_states = (
            ("final_fp32", result.matched_fp32_state, final_features),
            ("final_int8", result.state, final_features),
        )
        for name, state, features in (*before_states, *final_states):
            _assert_no_exact_top_tie(
                d42.score_d42_unified_shrinkage_lda(state, features), name
            )
        resource.update(
            {
                "lda_closed_form_fit_count": lda_count,
                "d49_lda_fit_inventory": inventory,
                "d49_lda_fit_inventory_macs": lda_macs,
                "d49_k8_exact_292_lda_fit_count_pass": k != 8 or lda_count == 292,
                "d49_outer_main_and_nested_inner_d45_refit": True,
                "d49_fused_query_state_count": 1,
                "d49_additional_query_state_count": 0,
                "d49_query_view": QUERY_VIEW,
                "d49_cosine_prototype_adaptation_macs": extra["prototype"],
                "d49_head_rms_adaptation_macs": extra["rms"],
                "d49_nested_held_scoring_macs": extra["held_score"],
                "d49_fp32_affine_fusion_macs": extra["fusion"],
                "d49_extra_adaptation_macs": extra_total,
                "d49_fp32_exact_top_tie_count": 0,
                "d49_int8_exact_top_tie_count": 0,
                "d49_cuda_peak_memory_measured": str(
                    resource.get("runtime_device", "")
                ).startswith("cuda"),
                "d49_host_fp64_peak_memory_measured": False,
                "d49_host_fp64_peak_memory_bytes": None,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"] + lda_macs + extra_total
        )
        geometry = dict(result.geometry_audit)
        geometry.update(
            {
                "d49_fp32_exact_top_tie_fail_close_checked": True,
                "d49_int8_exact_top_tie_fail_close_checked": True,
                "d49_query_view": QUERY_VIEW,
                "d49_single_affine_state_only": True,
                "d49_before_support_transform_bound_to_runtime_input": True,
                "d49_final_support_transform_bound_to_runtime_input": True,
                "d49_before_support_targets_bound_to_runtime_labels": True,
                "d49_final_support_targets_bound_to_runtime_labels": True,
                "d49_before_actual_matched_fp32_coefficient": (
                    np.asarray(result.matched_fp32_before_state.coef_fp32)
                    .astype(np.float64)
                    .tolist()
                ),
                "d49_before_actual_matched_fp32_intercept": (
                    np.asarray(result.matched_fp32_before_state.intercept_fp32)
                    .astype(np.float64)
                    .tolist()
                ),
                "d49_final_actual_matched_fp32_coefficient": (
                    np.asarray(result.matched_fp32_state.coef_fp32)
                    .astype(np.float64)
                    .tolist()
                ),
                "d49_final_actual_matched_fp32_intercept": (
                    np.asarray(result.matched_fp32_state.intercept_fp32)
                    .astype(np.float64)
                    .tolist()
                ),
                "d49_before_actual_coef1_qint8": (
                    np.asarray(result.before_state.coef1_qint8).astype(np.int64).tolist()
                ),
                "d49_before_actual_coef2_qint8": (
                    np.asarray(result.before_state.coef2_qint8).astype(np.int64).tolist()
                ),
                "d49_before_actual_scale1_fp16": (
                    np.asarray(result.before_state.scale1_fp16).astype(np.float64).tolist()
                ),
                "d49_before_actual_scale2_fp16": (
                    np.asarray(result.before_state.scale2_fp16).astype(np.float64).tolist()
                ),
                "d49_before_actual_intercept_fp16": (
                    np.asarray(result.before_state.intercept_fp16).astype(np.float64).tolist()
                ),
                "d49_final_actual_coef1_qint8": (
                    np.asarray(result.state.coef1_qint8).astype(np.int64).tolist()
                ),
                "d49_final_actual_coef2_qint8": (
                    np.asarray(result.state.coef2_qint8).astype(np.int64).tolist()
                ),
                "d49_final_actual_scale1_fp16": (
                    np.asarray(result.state.scale1_fp16).astype(np.float64).tolist()
                ),
                "d49_final_actual_scale2_fp16": (
                    np.asarray(result.state.scale2_fp16).astype(np.float64).tolist()
                ),
                "d49_final_actual_intercept_fp16": (
                    np.asarray(result.state.intercept_fp16).astype(np.float64).tolist()
                ),
            }
        )
        return replace(result, resource_audit=resource, geometry_audit=geometry)

    d42._lda_fit_macs = d49_lda_fit_macs
    d42.fit_d42_unified_shrinkage_lda = fit_with_audit
    return original_macs, original_top


def _verify_d49_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    rows = [
        row
        for row in training_rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 30:
        raise D49ProbeError("D49 training-row closure drift")
    for row in rows:
        resource = row.get("resource")
        geometry = row.get("geometry_summary")
        if not isinstance(resource, dict) or not isinstance(geometry, dict):
            raise D49ProbeError("D49 resource/geometry evidence missing")
        k = int(resource.get("old_k_shot", -1))
        if int(resource.get("new_k_shot", -2)) != k or k <= 1:
            raise D49ProbeError("D49 outer probe requires matched K>1")
        dimension = int(resource.get("coefficient_dimension", -1))
        if dimension <= 0:
            raise D49ProbeError("D49 coefficient dimension drift")
        class_counts: dict[str, int] = {}
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field)
            if not isinstance(audit, dict):
                raise D49ProbeError("D49 fit audit missing")
            required = {
                "d49_probe_arm": ARM,
                "d49_weight_formula": WEIGHT_FORMULA,
                "d49_query_view": QUERY_VIEW,
                "d49_outer_fold_count": k,
                "d49_complete_d45_refit_per_outer_fold": True,
                "d49_inner_train_rms_used_for_held": True,
                "d49_full_support_rms_used_for_final_affine": True,
                "d49_full_support_fused_once_pre_quantization": True,
                "d49_quantized_component_fused_or_decoded": False,
                "d49_role_handle_scene_specific_branch": False,
                "d49_scan_temperature_threshold_count": 0,
            }
            if any(audit.get(name) != value for name, value in required.items()):
                raise D49ProbeError("D49 exact fit audit drift")
            class_count = len(audit.get("d49_cosine_prototype_resultant_norm_by_class", []))
            if class_count < 2:
                raise D49ProbeError("D49 cosine class-count audit drift")
            class_counts[field] = class_count
            support_rows = _exact_json_array(
                audit.get("d49_support_transformed_fp32"),
                np.float32,
                (class_count * k, dimension),
                f"{field} support transformed FP32",
            )
            support_labels = _exact_json_array(
                audit.get("d49_support_targets"),
                np.int64,
                (class_count * k,),
                f"{field} support targets",
            )
            if (
                support_rows.shape != (class_count * k, dimension)
                or not np.isfinite(support_rows).all()
                or np.any(support_labels < 0)
                or np.any(support_labels >= class_count)
                or any(
                    np.count_nonzero(support_labels == index) != k
                    for index in range(class_count)
                )
                or not np.allclose(
                    np.linalg.norm(support_rows.astype(np.float64), axis=1),
                    1.0,
                    rtol=0.0,
                    atol=2.0e-6,
                )
            ):
                raise D49ProbeError("D49 persisted support transform drift")
            resultant_norms = np.asarray(
                audit.get("d49_cosine_prototype_resultant_norm_by_class"),
                dtype=np.float64,
            )
            prototypes = _exact_json_array(
                audit.get("d49_cosine_prototype_fp32"),
                np.float32,
                (class_count, dimension),
                f"{field} cosine prototype FP32",
            )
            if (
                resultant_norms.shape != (class_count,)
                or not np.isfinite(resultant_norms).all()
                or np.any(resultant_norms <= 0.0)
                or np.any(resultant_norms > 1.0 + 2.0e-6)
                or prototypes.shape != (class_count, dimension)
                or not np.isfinite(prototypes).all()
                or not np.allclose(
                    np.linalg.norm(prototypes.astype(np.float64), axis=1),
                    1.0,
                    rtol=0.0,
                    atol=2.0e-6,
                )
            ):
                raise D49ProbeError("D49 cosine prototype evidence drift")
            reference_proto, reference_zero, reference_cosine_audit = (
                _cosine_component_fit(
                    support_rows,
                    support_labels,
                    class_count,
                    k,
                    energy_epsilon=1.0e-12,
                )
            )
            if (
                not np.array_equal(prototypes, reference_proto)
                or not np.array_equal(
                    reference_zero, np.zeros(class_count, dtype=np.float32)
                )
                or not np.allclose(
                    resultant_norms,
                    reference_cosine_audit[
                        "d49_cosine_prototype_resultant_norm_by_class"
                    ],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise D49ProbeError("D49 cosine support provenance drift")
            full_rms = np.asarray(
                [
                    audit.get("d49_d45_full_support_logit_rms"),
                    audit.get("d49_cosine_full_support_logit_rms"),
                ],
                dtype=np.float64,
            )
            if not np.isfinite(full_rms).all() or np.any(full_rms <= 0.0):
                raise D49ProbeError("D49 full-support RMS evidence drift")
            d_ce = float(audit.get("d49_d45_nested_macro_class_ce", np.nan))
            c_ce = float(audit.get("d49_cosine_nested_macro_class_ce", np.nan))
            expected_d, expected_c, expected_evidence = _strict_weights(
                d_ce, c_ce, class_count
            )
            if not (
                np.allclose(
                    [audit.get("d49_d45_weight"), audit.get("d49_cosine_weight")],
                    [expected_d, expected_c],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    audit.get("d49_log_evidence_by_head"),
                    expected_evidence,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise D49ProbeError("D49 CE/weight evidence closure drift")
            partition = audit.get("d49_nested_partition_audit")
            if not isinstance(partition, dict):
                raise D49ProbeError("D49 nested partition missing")
            held = partition.get("held_support_row_indices_by_fold")
            train = partition.get("train_support_row_indices_by_fold")
            if (
                partition.get("train_held_overlap_count") != 0
                or partition.get("held_support_row_exact_once_coverage") is not True
                or not isinstance(held, list)
                or len(held) != k
                or not isinstance(train, list)
                or len(train) != k
                or len(partition.get("d49_nested_d45_fit_audit_by_fold", [])) != k
                or len(partition.get("d49_nested_cosine_fit_audit_by_fold", [])) != k
                or len(partition.get("d49_nested_d45_coefficient_fp32_by_fold", [])) != k
                or len(partition.get("d49_nested_d45_intercept_fp32_by_fold", [])) != k
                or len(partition.get("d49_d45_inner_train_logit_rms_by_fold", [])) != k
                or len(partition.get("d49_cosine_inner_train_logit_rms_by_fold", [])) != k
            ):
                raise D49ProbeError("D49 nested partition/RMS structural drift")
            rms_evidence = np.asarray(
                [
                    partition["d49_d45_inner_train_logit_rms_by_fold"],
                    partition["d49_cosine_inner_train_logit_rms_by_fold"],
                ],
                dtype=np.float64,
            )
            if not np.isfinite(rms_evidence).all() or np.any(rms_evidence <= 0.0):
                raise D49ProbeError("D49 nested RMS evidence drift")
            d45_held = np.asarray(
                partition.get("d49_d45_held_scores_by_fold"), dtype=np.float64
            )
            cosine_held = np.asarray(
                partition.get("d49_cosine_held_scores_by_fold"), dtype=np.float64
            )
            if (
                d45_held.shape != (k, class_count, class_count)
                or cosine_held.shape != d45_held.shape
                or not np.isfinite(d45_held).all()
                or not np.isfinite(cosine_held).all()
            ):
                raise D49ProbeError("D49 held-score evidence drift")
            held_labels = np.concatenate(
                [support_labels[np.asarray(fold, dtype=np.int64)] for fold in held]
            )
            recomputed_d_ce, recomputed_d_by_class = d45._class_balanced_cross_entropy(
                d45_held.reshape(k * class_count, class_count),
                held_labels,
                class_count,
            )
            recomputed_c_ce, recomputed_c_by_class = d45._class_balanced_cross_entropy(
                cosine_held.reshape(k * class_count, class_count),
                held_labels,
                class_count,
            )
            if (
                not np.isclose(recomputed_d_ce, d_ce, rtol=0.0, atol=1.0e-12)
                or not np.isclose(recomputed_c_ce, c_ce, rtol=0.0, atol=1.0e-12)
                or not np.allclose(
                    partition.get("d49_d45_outer_held_ce_by_class"),
                    recomputed_d_by_class,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.allclose(
                    partition.get("d49_cosine_outer_held_ce_by_class"),
                    recomputed_c_by_class,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    partition.get("d49_d45_outer_held_macro_class_ce"),
                    recomputed_d_ce,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    partition.get("d49_cosine_outer_held_macro_class_ce"),
                    recomputed_c_ce,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise D49ProbeError("D49 held-score CE closure drift")
            held_flat = [int(value) for fold in held for value in fold]
            expected_rows = class_count * k
            if sorted(held_flat) != list(range(expected_rows)):
                raise D49ProbeError("D49 held exact-once verifier drift")
            indices_by_class = [
                np.flatnonzero(support_labels == index).tolist()
                for index in range(class_count)
            ]
            for rank, (
                held_fold,
                train_fold,
                nested_audit,
                cosine_audit,
                nested_d45_coef_value,
                nested_d45_intercept_value,
            ) in enumerate(zip(
                held,
                train,
                partition["d49_nested_d45_fit_audit_by_fold"],
                partition["d49_nested_cosine_fit_audit_by_fold"],
                partition["d49_nested_d45_coefficient_fp32_by_fold"],
                partition["d49_nested_d45_intercept_fp32_by_fold"],
                strict=True,
            )):
                expected_held = [indices_by_class[index][rank] for index in range(class_count)]
                expected_train = sorted(set(range(expected_rows)) - set(expected_held))
                if (
                    held_fold != expected_held
                    or train_fold != expected_train
                    or set(held_fold).intersection(train_fold)
                ):
                    raise D49ProbeError("D49 nested rank partition drift")
                _verify_complete_nested_d45_audit(
                    nested_audit,
                    class_count=class_count,
                    k_shot=k - 1,
                )
                nested_d45_coef = _exact_json_array(
                    nested_d45_coef_value,
                    np.float32,
                    (class_count, dimension),
                    f"{field} nested D45 coefficient FP32",
                )
                nested_d45_intercept = _exact_json_array(
                    nested_d45_intercept_value,
                    np.float32,
                    (class_count,),
                    f"{field} nested D45 intercept FP32",
                )
                if (
                    nested_d45_coef.shape != (class_count, dimension)
                    or nested_d45_intercept.shape != (class_count,)
                    or not np.isfinite(nested_d45_coef).all()
                    or not np.isfinite(nested_d45_intercept).all()
                ):
                    raise D49ProbeError("D49 nested D45 state evidence drift")
                nested_norms = np.asarray(
                    cosine_audit.get("d49_cosine_prototype_resultant_norm_by_class"),
                    dtype=np.float64,
                )
                nested_prototypes = _exact_json_array(
                    cosine_audit.get("d49_cosine_prototype_fp32"),
                    np.float32,
                    (class_count, dimension),
                    f"{field} nested cosine prototype FP32",
                )
                if (
                    cosine_audit.get("d49_cosine_geometry")
                    != "D42_global_unit_sphere_cosine_prototype"
                    or nested_norms.shape != (class_count,)
                    or not np.isfinite(nested_norms).all()
                    or np.any(nested_norms <= 0.0)
                    or nested_prototypes.shape != (class_count, dimension)
                    or not np.allclose(
                        np.linalg.norm(nested_prototypes.astype(np.float64), axis=1),
                        1.0,
                        rtol=0.0,
                        atol=2.0e-6,
                    )
                ):
                    raise D49ProbeError("D49 nested cosine evidence drift")
                train_rows = support_rows[np.asarray(train_fold, dtype=np.int64)]
                train_labels = support_labels[np.asarray(train_fold, dtype=np.int64)]
                held_rows = support_rows[np.asarray(held_fold, dtype=np.int64)]
                reference_nested_proto, _zero, reference_nested_audit = (
                    _cosine_component_fit(
                        train_rows,
                        train_labels,
                        class_count,
                        k - 1,
                        energy_epsilon=1.0e-12,
                    )
                )
                recomputed_d_scale = d44._class_centered_logit_rms(
                    train_rows, nested_d45_coef, nested_d45_intercept
                )
                recomputed_c_scale = d44._class_centered_logit_rms(
                    train_rows,
                    reference_nested_proto,
                    np.zeros(class_count, dtype=np.float32),
                )
                recomputed_d_scores = (
                    held_rows.astype(np.float64)
                    @ nested_d45_coef.astype(np.float64).T
                    + nested_d45_intercept.astype(np.float64)[None, :]
                ) / recomputed_d_scale
                recomputed_c_scores = (
                    held_rows.astype(np.float64)
                    @ reference_nested_proto.astype(np.float64).T
                ) / recomputed_c_scale
                if (
                    not np.array_equal(nested_prototypes, reference_nested_proto)
                    or not np.allclose(
                        nested_norms,
                        reference_nested_audit[
                            "d49_cosine_prototype_resultant_norm_by_class"
                        ],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.isclose(
                        rms_evidence[0, rank],
                        recomputed_d_scale,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.isclose(
                        rms_evidence[1, rank],
                        recomputed_c_scale,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        d45_held[rank],
                        recomputed_d_scores,
                        rtol=0.0,
                        atol=2.0e-7,
                    )
                    or not np.allclose(
                        cosine_held[rank],
                        recomputed_c_scores,
                        rtol=0.0,
                        atol=2.0e-7,
                    )
                ):
                    raise D49ProbeError("D49 nested support provenance drift")
            _verify_complete_nested_d45_audit(
                audit, class_count=class_count, k_shot=k
            )
            d45_coef = _exact_json_array(
                audit.get("d49_d45_full_support_coefficient_fp32"),
                np.float32,
                (class_count, dimension),
                f"{field} D45 full coefficient FP32",
            )
            d45_intercept = _exact_json_array(
                audit.get("d49_d45_full_support_intercept_fp32"),
                np.float32,
                (class_count,),
                f"{field} D45 full intercept FP32",
            )
            final_coef = _exact_json_array(
                audit.get("d49_final_fused_coefficient_fp32"),
                np.float32,
                (class_count, dimension),
                f"{field} final fused coefficient FP32",
            )
            final_intercept = _exact_json_array(
                audit.get("d49_final_fused_intercept_fp32"),
                np.float32,
                (class_count,),
                f"{field} final fused intercept FP32",
            )
            if (
                d45_coef.shape != (class_count, dimension)
                or d45_intercept.shape != (class_count,)
                or final_coef.shape != (class_count, dimension)
                or final_intercept.shape != (class_count,)
            ):
                raise D49ProbeError("D49 fusion state evidence shape drift")
            recomputed_full_d_scale = d44._class_centered_logit_rms(
                support_rows, d45_coef, d45_intercept
            )
            recomputed_full_c_scale = d44._class_centered_logit_rms(
                support_rows,
                reference_proto,
                np.zeros(class_count, dtype=np.float32),
            )
            if (
                not np.isclose(
                    full_rms[0], recomputed_full_d_scale, rtol=0.0, atol=1.0e-12
                )
                or not np.isclose(
                    full_rms[1], recomputed_full_c_scale, rtol=0.0, atol=1.0e-12
                )
            ):
                raise D49ProbeError("D49 full-support RMS provenance drift")
            expected_coef, expected_intercept = d43._center_affine_scores(
                expected_d * d45_coef.astype(np.float64) / full_rms[0]
                + expected_c * prototypes.astype(np.float64) / full_rms[1],
                expected_d * d45_intercept.astype(np.float64) / full_rms[0],
            )
            if (
                not np.array_equal(final_coef, expected_coef.astype(np.float32))
                or not np.array_equal(
                    final_intercept, expected_intercept.astype(np.float32)
                )
            ):
                raise D49ProbeError("D49 FP32 fusion formula closure drift")
            state_prefix = "before" if field.startswith("before") else "final"
            actual_fp32_coef = _exact_json_array(
                geometry.get(
                    f"d49_{state_prefix}_actual_matched_fp32_coefficient"
                ),
                np.float32,
                (class_count, dimension),
                f"{state_prefix} matched FP32 coefficient",
            )
            actual_fp32_intercept = _exact_json_array(
                geometry.get(
                    f"d49_{state_prefix}_actual_matched_fp32_intercept"
                ),
                np.float32,
                (class_count,),
                f"{state_prefix} matched FP32 intercept",
            )
            expected_q1, expected_q2, expected_s1, expected_s2 = (
                _audit_quantize_coefficients(final_coef)
            )
            actual_q1 = _exact_json_array(
                geometry.get(f"d49_{state_prefix}_actual_coef1_qint8"),
                np.int8,
                (class_count, dimension),
                f"{state_prefix} coef1 qint8",
            )
            actual_q2 = _exact_json_array(
                geometry.get(f"d49_{state_prefix}_actual_coef2_qint8"),
                np.int8,
                (class_count, dimension),
                f"{state_prefix} coef2 qint8",
            )
            actual_s1 = _exact_json_array(
                geometry.get(f"d49_{state_prefix}_actual_scale1_fp16"),
                np.float16,
                (class_count, len(COEFFICIENT_BLOCK_BOUNDS)),
                f"{state_prefix} scale1 FP16",
            )
            actual_s2 = _exact_json_array(
                geometry.get(f"d49_{state_prefix}_actual_scale2_fp16"),
                np.float16,
                (class_count, len(COEFFICIENT_BLOCK_BOUNDS)),
                f"{state_prefix} scale2 FP16",
            )
            actual_i16 = _exact_json_array(
                geometry.get(f"d49_{state_prefix}_actual_intercept_fp16"),
                np.float16,
                (class_count,),
                f"{state_prefix} intercept FP16",
            )
            if (
                not np.array_equal(actual_fp32_coef, final_coef)
                or not np.array_equal(actual_fp32_intercept, final_intercept)
                or not np.array_equal(actual_q1, expected_q1)
                or not np.array_equal(actual_q2, expected_q2)
                or not np.array_equal(actual_s1, expected_s1)
                or not np.array_equal(actual_s2, expected_s2)
                or not np.array_equal(actual_i16, final_intercept.astype(np.float16))
            ):
                raise D49ProbeError("D49 compiled state binding drift")
        if (
            geometry.get("before_materialized_pre_stage2c") is not True
            or geometry.get("before_state_immutable_during_stage2c") is not True
            or int(geometry.get("old_only_metric_new_support_argument_count", -1)) != 0
        ):
            raise D49ProbeError("D49 before/new-support lifecycle drift")
        inventory = resource.get("d49_lda_fit_inventory")
        expected_specs = _d45_fit_specs(
            "before", class_counts["before_covariance_audit"], k
        ) + _d45_fit_specs("final", class_counts["final_covariance_audit"], k)
        if not isinstance(inventory, list) or len(inventory) != len(expected_specs):
            raise D49ProbeError("D49 LDA inventory missing")
        total_count = 0
        total_macs = 0
        for item, (group, expected_fit_count, expected_rows, expected_classes) in zip(
            inventory, expected_specs, strict=True
        ):
            if not isinstance(item, dict):
                raise D49ProbeError("D49 LDA inventory row drift")
            fit_count = int(item.get("fit_count", -1))
            row_count = int(item.get("row_count_per_fit", -1))
            class_count = int(item.get("class_count", -1))
            expected_macs = d45._expected_lda_fit_macs(
                expected_rows, expected_classes, dimension
            )
            if (
                item.get("fit_group") != group
                or fit_count != expected_fit_count
                or row_count != expected_rows
                or class_count != expected_classes
                or item.get("macs_per_fit") != expected_macs
            ):
                raise D49ProbeError("D49 LDA inventory MAC row drift")
            total_count += fit_count
            total_macs += fit_count * expected_macs
        if (
            total_count != resource.get("lda_closed_form_fit_count")
            or total_macs != resource.get("estimated_lda_fit_macs")
            or total_macs != resource.get("d49_lda_fit_inventory_macs")
            or (k == 8 and total_count != 292)
            or resource.get("d49_k8_exact_292_lda_fit_count_pass") is not True
            or resource.get("d49_fused_query_state_count") != 1
            or resource.get("d49_additional_query_state_count") != 0
            or resource.get("d49_query_view") != QUERY_VIEW
            or resource.get("d49_fp32_exact_top_tie_count") != 0
            or resource.get("d49_int8_exact_top_tie_count") != 0
        ):
            raise D49ProbeError("D49 resource total/state/tie closure drift")
        before_extra = _extra_head_macs(
            class_counts["before_covariance_audit"], k, dimension
        )
        final_extra = _extra_head_macs(
            class_counts["final_covariance_audit"], k, dimension
        )
        expected_extra_fields = {
            "d49_cosine_prototype_adaptation_macs": before_extra["prototype"]
            + final_extra["prototype"],
            "d49_head_rms_adaptation_macs": before_extra["rms"] + final_extra["rms"],
            "d49_nested_held_scoring_macs": before_extra["held_score"]
            + final_extra["held_score"],
            "d49_fp32_affine_fusion_macs": before_extra["fusion"]
            + final_extra["fusion"],
        }
        if any(
            resource.get(name) != value
            for name, value in expected_extra_fields.items()
        ):
            raise D49ProbeError("D49 extra MAC formula drift")
        extra = sum(expected_extra_fields.values())
        cuda_measured = str(resource.get("runtime_device", "")).startswith("cuda")
        if (
            extra != resource.get("d49_extra_adaptation_macs")
            or resource.get("estimated_adaptation_macs")
            != resource.get("estimated_metric_adaptation_macs") + total_macs + extra
            or resource.get("d49_host_fp64_peak_memory_measured") is not False
            or resource.get("d49_host_fp64_peak_memory_bytes") is not None
            or resource.get("d49_cuda_peak_memory_measured") is not cuda_measured
            or geometry.get("d49_fp32_exact_top_tie_fail_close_checked") is not True
            or geometry.get("d49_int8_exact_top_tie_fail_close_checked") is not True
            or geometry.get("d49_single_affine_state_only") is not True
            or geometry.get("d49_before_support_transform_bound_to_runtime_input")
            is not True
            or geometry.get("d49_final_support_transform_bound_to_runtime_input")
            is not True
            or geometry.get("d49_before_support_targets_bound_to_runtime_labels")
            is not True
            or geometry.get("d49_final_support_targets_bound_to_runtime_labels")
            is not True
        ):
            raise D49ProbeError("D49 adaptation/state memory audit drift")
    return len(rows)


def _verify_d49_output(
    output: Path,
    probe_script_sha256: str,
    d45_helper_sha256: str,
    d44_helper_sha256: str,
    d43_helper_sha256: str,
    runner_score_guard_counts: dict[str, int],
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    expected = {
        "d49_d45_helper_sha256": d45_helper_sha256,
        "d49_d44_helper_sha256": d44_helper_sha256,
        "d49_d43_helper_sha256": d43_helper_sha256,
    }
    if any(closure.get(name) != value for name, value in expected.items()):
        raise D49ProbeError("D49 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = _verify_d49_fit_audits(rows)
    if (
        runner_score_guard_counts.get("call_count", 0) <= 0
        or runner_score_guard_counts.get("fp32_rows", 0) <= 0
        or runner_score_guard_counts.get("int8_rows", 0) <= 0
    ):
        raise D49ProbeError("D49 runner outer-score tie guard coverage drift")
    return {
        **evidence,
        "verified_d49_fit_row_count": count,
        "d49_runner_score_tie_guard_counts": dict(runner_score_guard_counts),
        **expected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d49-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D49ProbeError(f"D49 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_macs = None
    original_top = None
    runner_module = None
    original_runner_score = None
    runner_score_guard_counts = {"fp32_rows": 0, "int8_rows": 0, "call_count": 0}
    runner_module_name = "d49_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d45_helper_sha256 = d43._sha256(D45_HELPER_PATH)
    d44_helper_sha256 = d43._sha256(d45.D44_HELPER_PATH)
    d43_helper_sha256 = d43._sha256(d44.D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_nested_global_cosine_fit(d42)
        original_macs, original_top = _install_d49_resource_and_tie_audit(d42)
        runner_path = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner_path)
        if spec is None or spec.loader is None:
            raise D49ProbeError("D49 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        original_runner_score, runner_score_guard_counts = (
            _install_runner_score_tie_guard(runner_module)
        )
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d49_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d49_d45_helper_sha256": d45_helper_sha256,
                "d49_d44_helper_sha256": d44_helper_sha256,
                "d49_d43_helper_sha256": d43_helper_sha256,
            },
        )
        sys.argv = [str(runner_path), *runner_arguments]
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
        if runner_module is not None and original_runner_score is not None:
            runner_module.score_d42_unified_shrinkage_lda = original_runner_score
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_d49_output(
        output,
        probe_script_sha256,
        d45_helper_sha256,
        d44_helper_sha256,
        d43_helper_sha256,
        runner_score_guard_counts,
    )
    metadata = {
        "schema": "cvs.phase2.d49.nested_global_cosine_fusion_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d49_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "query_view": QUERY_VIEW,
        "probe_script_sha256": probe_script_sha256,
        "d45_helper_sha256": d45_helper_sha256,
        "d44_helper_sha256": d44_helper_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "weight_formula": WEIGHT_FORMULA,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D49_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
