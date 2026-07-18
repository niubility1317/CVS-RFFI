#!/usr/bin/env python3
"""D48 probe: one-shot support-OOF class-margin intercept residual on D45."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D45_HELPER_PATH = SCRIPT_DIR / "probe_d45_inner_loo_reliability_fusion.py"
D45_SPEC = importlib.util.spec_from_file_location("d48_d45_probe_helper", D45_HELPER_PATH)
if D45_SPEC is None or D45_SPEC.loader is None:
    raise RuntimeError("D48 could not load the D45 probe helper")
d45 = importlib.util.module_from_spec(D45_SPEC)
D45_SPEC.loader.exec_module(d45)
d44 = d45.d44
d43 = d45.d43


ARM = "one_shot_oof_head_margin_residual"
STRUCTURE = "d45_global_fusion_one_shot_support_oof_head_margin_intercept_residual"
MARGIN_FORMULA = "margin_c_r=q_c_r_c-max_j_ne_c(q_c_r_j)"
BIAS_FORMULA = "m_c=mean_r(margin_c_r);beta_c=center_c(mean_c(m_c)-m_c)"
EVIDENCE_FORMULA = "q=w_full*d_full_inner_rms+w_block*d_block_inner_rms"
STATISTICAL_CLAIM = "support_supervised_one_shot_oof_head_margin_residual_not_independent_calibration"
COEFFICIENT_BLOCK_BOUNDS = ((0, 160), (160, 256), (256, 288))
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D48ProbeError(RuntimeError):
    pass


def _validate_partition(partition: Any, k_shot: int, class_count: int) -> None:
    if not isinstance(partition, dict):
        raise D48ProbeError("D48 partition evidence missing")
    held = partition.get("held_support_row_indices_by_fold")
    held_classes = partition.get("private_collector_held_class_indices_by_fold")
    train = partition.get("private_collector_train_support_row_indices_by_fold")
    if (
        not isinstance(held, list)
        or len(held) != k_shot
        or not isinstance(held_classes, list)
        or len(held_classes) != k_shot
        or not isinstance(train, list)
        or len(train) != k_shot
        or partition.get("private_collector_train_indices_are_exact_held_complements")
        is not True
    ):
        raise D48ProbeError("D48 private partition structure drift")
    all_indices = set(range(k_shot * class_count))
    held_flat: list[int] = []
    for held_fold, class_fold, train_fold in zip(held, held_classes, train):
        held_set = set(held_fold)
        if (
            len(held_fold) != class_count
            or sorted(class_fold) != list(range(class_count))
            or train_fold != sorted(all_indices - held_set)
            or held_set.intersection(train_fold)
        ):
            raise D48ProbeError("D48 per-fold anonymous-class partition drift")
        held_flat.extend(int(value) for value in held_fold)
    if sorted(held_flat) != list(range(k_shot * class_count)):
        raise D48ProbeError("D48 held support exact-once drift")


def _one_shot_margin_residual(
    *,
    full_held_scores: Any,
    block_held_scores: Any,
    full_weight: float,
    block_weight: float,
    full_partition: Any,
    block_partition: Any,
    k_shot: int,
    class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(k_shot)
    c = int(class_count)
    if k < 1 or c < 2:
        raise D48ProbeError("D48 requires K>=1 and C>=2")
    weights = np.asarray([full_weight, block_weight], dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise D48ProbeError("D48 D45 global-weight closure drift")
    common = {
        "d43_probe_arm": ARM,
        "d43_covariance_structure": STRUCTURE,
        "d45_probe_arm": ARM,
        "d48_probe_arm": ARM,
        "d48_evidence_formula": EVIDENCE_FORMULA,
        "d48_margin_formula": MARGIN_FORMULA,
        "d48_bias_formula": BIAS_FORMULA,
        "d48_statistical_claim": STATISTICAL_CLAIM,
        "d48_actual_k": k,
        "d48_class_count": c,
        "d48_d45_full_weight": float(full_weight),
        "d48_d45_block_weight": float(block_weight),
        "d48_support_supervised": k > 1,
        "d48_uses_outer_held_or_query": False,
        "d48_class_id_specific_formula": False,
        "d48_old_new_role_specific_branch": False,
        "d48_scene_receiver_handle_specific_branch": False,
        "d48_scan_clip_threshold_count": 0,
        "d48_beta_applied_once": True,
        "d48_beta_recomputed_or_iterated": False,
        "d48_beta_recomputes_rms_or_weight": False,
        "d48_independent_calibration_claim_allowed": False,
    }
    if k == 1:
        if full_held_scores is not None or block_held_scores is not None:
            raise D48ProbeError("D48 K1 held-score evidence must be absent")
        if full_partition is not None or block_partition is not None:
            raise D48ProbeError("D48 K1 partition evidence must be absent")
        if not np.array_equal(weights, np.asarray([0.5, 0.5])):
            raise D48ProbeError("D48 K1 D45 unit fallback drift")
        zero = [0.0] * c
        return np.zeros(c, dtype=np.float64), {
            **common,
            "d48_boundary_status": "k1_exact_d45_zero_bias_fallback",
            "d48_full_held_logits_by_fold_class_logit": None,
            "d48_block_held_logits_by_fold_class_logit": None,
            "d48_fused_held_logits_by_fold_class_logit": None,
            "d48_true_logit_by_fold_class": None,
            "d48_max_other_logit_by_fold_class": None,
            "d48_max_other_tie_count_by_fold_class": None,
            "d48_total_max_other_tie_count": 0,
            "d48_margin_by_fold_class": None,
            "d48_mean_margin_by_class": None,
            "d48_global_mean_margin": None,
            "d48_beta_raw_by_class": zero,
            "d48_beta_centered_by_class": zero,
            "d48_beta_sum_residual": 0.0,
            "d48_beta_nonzero_class_count": 0,
            "d48_beta_max_abs": 0.0,
            "d48_margin_min": None,
            "d48_margin_max": None,
        }

    _validate_partition(full_partition, k, c)
    _validate_partition(block_partition, k, c)
    if full_partition["held_support_row_indices_by_fold"] != block_partition[
        "held_support_row_indices_by_fold"
    ]:
        raise D48ProbeError("D48 component held partition mismatch")
    full = np.asarray(full_held_scores, dtype=np.float64)
    block = np.asarray(block_held_scores, dtype=np.float64)
    if (
        full.shape != (k, c, c)
        or block.shape != full.shape
        or not np.isfinite(full).all()
        or not np.isfinite(block).all()
    ):
        raise D48ProbeError("D48 held-logit evidence drift")
    if k == 2 and not (
        np.isclose(full_weight, 0.5, rtol=0.0, atol=1.0e-12)
        and np.isclose(block_weight, 0.5, rtol=0.0, atol=1.0e-12)
    ):
        raise D48ProbeError("D48 K2 D45 unit-component weight drift")
    fused = full_weight * full + block_weight * block
    true_logits = np.empty((k, c), dtype=np.float64)
    max_other = np.empty((k, c), dtype=np.float64)
    tie_counts = np.empty((k, c), dtype=np.int64)
    for rank in range(k):
        for class_index in range(c):
            values = fused[rank, class_index]
            true_logits[rank, class_index] = values[class_index]
            other = np.delete(values, class_index)
            maximum = float(np.max(other))
            max_other[rank, class_index] = maximum
            tie_counts[rank, class_index] = int(np.sum(other == maximum))
    margins = true_logits - max_other
    mean_margin = np.mean(margins, axis=0)
    global_mean = float(np.mean(mean_margin))
    beta_raw = global_mean - mean_margin
    beta = beta_raw - float(np.mean(beta_raw))
    if (
        not np.isfinite(fused).all()
        or not np.isfinite(margins).all()
        or not np.isfinite(mean_margin).all()
        or not np.isfinite(beta).all()
        or abs(float(np.mean(beta))) > 1.0e-12
        or np.any(tie_counts < 1)
    ):
        raise D48ProbeError("D48 one-shot margin residual became invalid")
    return beta, {
        **common,
        "d48_boundary_status": "one_shot_oof_head_mean_margin_residual",
        "d48_full_held_logits_by_fold_class_logit": full.tolist(),
        "d48_block_held_logits_by_fold_class_logit": block.tolist(),
        "d48_fused_held_logits_by_fold_class_logit": fused.tolist(),
        "d48_true_logit_by_fold_class": true_logits.tolist(),
        "d48_max_other_logit_by_fold_class": max_other.tolist(),
        "d48_max_other_tie_count_by_fold_class": tie_counts.tolist(),
        "d48_total_max_other_tie_count": int(np.sum(tie_counts > 1)),
        "d48_margin_by_fold_class": margins.tolist(),
        "d48_mean_margin_by_class": mean_margin.tolist(),
        "d48_global_mean_margin": global_mean,
        "d48_beta_raw_by_class": beta_raw.tolist(),
        "d48_beta_centered_by_class": beta.tolist(),
        "d48_beta_sum_residual": float(np.sum(beta)),
        "d48_beta_nonzero_class_count": int(np.count_nonzero(beta)),
        "d48_beta_max_abs": float(np.max(np.abs(beta))),
        "d48_margin_min": float(np.min(margins)),
        "d48_margin_max": float(np.max(margins)),
    }


def build_one_shot_margin_residual_fit(d42: Any) -> Any:
    return d45.build_inner_loo_reliability_fit(
        d42, post_fusion_calibration=_one_shot_margin_residual
    )


def _operation_upper_bound(k_shot: int, old_count: int, all_count: int) -> int:
    k = int(k_shot)
    if k < 1 or old_count < 2 or all_count < old_count:
        raise D48ProbeError("D48 operation inventory input drift")
    if k == 1:
        return 0
    square_sum = int(old_count) ** 2 + int(all_count) ** 2
    class_sum = int(old_count) + int(all_count)
    return int(4 * k * square_sum + 8 * k * class_sum + 16 * class_sum + 32)


def _evidence_peak_numeric_bytes(
    k_shot: int, old_count: int, all_count: int
) -> int:
    k = int(k_shot)
    if k < 1 or old_count < 2 or all_count < old_count:
        raise D48ProbeError("D48 evidence-byte inventory input drift")

    def per_state(class_count: int) -> int:
        if k == 1:
            return int(24 * class_count + 64)
        return int(
            24 * k * class_count**2
            + 32 * k * class_count
            + 24 * class_count
            + 64
        )

    return max(per_state(int(old_count)), per_state(int(all_count)))


def _formal_coefficient_payload_sha256(
    coef1_qint8: np.ndarray,
    coef2_qint8: np.ndarray,
    scale1_fp16: np.ndarray,
    scale2_fp16: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for values in (coef1_qint8, coef2_qint8, scale1_fp16, scale2_fp16):
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _formal_coefficient_sha256(state: Any) -> str:
    return _formal_coefficient_payload_sha256(
        state.coef1_qint8,
        state.coef2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
    )


def _audit_quantize_coefficients(
    coefficients: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(coefficients, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != COEFFICIENT_BLOCK_BOUNDS[-1][1]
        or not np.isfinite(rows).all()
    ):
        raise D48ProbeError("D48 coefficient quantization input drift")
    code1 = np.zeros(rows.shape, dtype=np.int8)
    code2 = np.zeros(rows.shape, dtype=np.int8)
    scale1 = np.empty((len(rows), len(COEFFICIENT_BLOCK_BOUNDS)), dtype=np.float16)
    scale2 = np.empty((len(rows), len(COEFFICIENT_BLOCK_BOUNDS)), dtype=np.float16)
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
                raise D48ProbeError("D48 coefficient quantization scale drift")
            second_code = np.clip(
                np.rint(residual / np.float32(second_scale)), -127, 127
            ).astype(np.int8)
            code1[row_index, start:stop] = first_code
            code2[row_index, start:stop] = second_code
            scale1[row_index, block_index] = first_scale
            scale2[row_index, block_index] = second_scale
    return code1, code2, scale1, scale2


def _fit_audit_evidence_payload(geometry: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for state_name in ("before_covariance_audit", "final_covariance_audit"):
        state = geometry.get(state_name)
        if not isinstance(state, dict):
            raise D48ProbeError("D48 fit-audit geometry missing")
        payload[state_name] = {
            name: value
            for name, value in state.items()
            if name.startswith("d48_")
            or name.startswith("d45_post_fusion_calibration_")
        }
    payload["formal_state_binding"] = {
        name: value for name, value in geometry.items() if name.startswith("d48_")
    }
    return payload


def _fit_audit_json_utf8_bytes(geometry: dict[str, Any]) -> int:
    encoded = json.dumps(
        _fit_audit_evidence_payload(geometry),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return len(encoded)


def _install_d48_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d45._install_d45_core_resource_accounting(d42)
    d45_top = d42.fit_d42_unified_shrinkage_lda

    def fit_with_d48_resource_audit(*args: Any, **kwargs: Any) -> Any:
        result = d45_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = int(d42.FEATURE_DIM)
        square_sum = old_count**2 + all_count**2
        scoring_macs = int(
            2 * k * (1 if k <= 1 else k + 1) * dimension * square_sum
        )
        affine_fusion_macs = int(2 * (dimension + 1) * (old_count + all_count))
        margin_upper = _operation_upper_bound(k, old_count, all_count)
        evidence_peak_bytes = _evidence_peak_numeric_bytes(k, old_count, all_count)
        resource.update(
            {
                "coefficient_dimension": dimension,
                "d48_outer_b20_training_count": 1,
                "d48_fused_query_state_count": 1,
                "d48_additional_lda_fit_count": 0,
                "d48_additional_optimizer_steps": 0,
                "d48_additional_query_state_count": 0,
                "d48_query_sidecar_bytes": 0,
                "d48_estimated_component_scoring_macs": scoring_macs,
                "d48_estimated_affine_fusion_macs": affine_fusion_macs,
                "d48_margin_operation_upper_bound": margin_upper,
                "d48_margin_operation_upper_bound_formula": (
                    "0_if_K1_else_4*K*(C_old^2+C_all^2)+"
                    "8*K*(C_old+C_all)+16*(C_old+C_all)+32"
                ),
                "d48_margin_operation_complexity": "O(K*C^2)",
                "d48_adaptation_evidence_peak_numeric_bytes": evidence_peak_bytes,
                "d48_adaptation_evidence_memory_scope": (
                    "support_only_full_block_fused_held_logits_and_margin_audit"
                ),
                "d48_final_state_dimension_unchanged_from_d45": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"]
            + resource["estimated_lda_fit_macs"]
            + scoring_macs
            + affine_fusion_macs
            + margin_upper
        )
        geometry = dict(result.geometry_audit)
        before_fit_audit = geometry["before_covariance_audit"]
        final_fit_audit = geometry["final_covariance_audit"]
        before_intercept = np.asarray(result.before_state.intercept_fp16)
        final_intercept = np.asarray(result.state.intercept_fp16)
        before_formal_coefficient = (
            np.asarray(result.before_state.coef1_qint8),
            np.asarray(result.before_state.coef2_qint8),
            np.asarray(result.before_state.scale1_fp16),
            np.asarray(result.before_state.scale2_fp16),
        )
        final_formal_coefficient = (
            np.asarray(result.state.coef1_qint8),
            np.asarray(result.state.coef2_qint8),
            np.asarray(result.state.scale1_fp16),
            np.asarray(result.state.scale2_fp16),
        )
        before_requantized = _audit_quantize_coefficients(
            before_fit_audit["d45_post_fusion_calibration_coefficient_fp32"]
        )
        final_requantized = _audit_quantize_coefficients(
            final_fit_audit["d45_post_fusion_calibration_coefficient_fp32"]
        )
        geometry.update(
            {
                "d48_before_formal_intercept_fp16_values": (
                    before_intercept.astype(np.float64).tolist()
                ),
                "d48_final_formal_intercept_fp16_values": (
                    final_intercept.astype(np.float64).tolist()
                ),
                "d48_before_formal_intercept_fp16_sha256": d45._array_sha256(
                    before_intercept
                ),
                "d48_final_formal_intercept_fp16_sha256": d45._array_sha256(
                    final_intercept
                ),
                "d48_before_formal_coefficient_int8_sha256": (
                    _formal_coefficient_sha256(result.before_state)
                ),
                "d48_final_formal_coefficient_int8_sha256": (
                    _formal_coefficient_sha256(result.state)
                ),
                "d48_before_formal_coef1_qint8_values": (
                    before_formal_coefficient[0].astype(np.int64).tolist()
                ),
                "d48_before_formal_coef2_qint8_values": (
                    before_formal_coefficient[1].astype(np.int64).tolist()
                ),
                "d48_before_formal_scale1_fp16_values": (
                    before_formal_coefficient[2].astype(np.float64).tolist()
                ),
                "d48_before_formal_scale2_fp16_values": (
                    before_formal_coefficient[3].astype(np.float64).tolist()
                ),
                "d48_final_formal_coef1_qint8_values": (
                    final_formal_coefficient[0].astype(np.int64).tolist()
                ),
                "d48_final_formal_coef2_qint8_values": (
                    final_formal_coefficient[1].astype(np.int64).tolist()
                ),
                "d48_final_formal_scale1_fp16_values": (
                    final_formal_coefficient[2].astype(np.float64).tolist()
                ),
                "d48_final_formal_scale2_fp16_values": (
                    final_formal_coefficient[3].astype(np.float64).tolist()
                ),
                "d48_before_fit_fp32_coefficient_compiles_to_formal_int8": all(
                    np.array_equal(actual, expected)
                    for actual, expected in zip(
                        before_formal_coefficient, before_requantized
                    )
                ),
                "d48_final_fit_fp32_coefficient_compiles_to_formal_int8": all(
                    np.array_equal(actual, expected)
                    for actual, expected in zip(final_formal_coefficient, final_requantized)
                ),
                "d48_before_fit_fp32_intercept_compiles_to_formal_fp16": bool(
                    np.array_equal(
                        before_intercept,
                        np.asarray(
                            before_fit_audit[
                                "d45_post_fusion_calibration_final_intercept_fp32"
                            ],
                            dtype=np.float16,
                        ),
                    )
                ),
                "d48_final_fit_fp32_intercept_compiles_to_formal_fp16": bool(
                    np.array_equal(
                        final_intercept,
                        np.asarray(
                            final_fit_audit[
                                "d45_post_fusion_calibration_final_intercept_fp32"
                            ],
                            dtype=np.float16,
                        ),
                    )
                ),
            }
        )
        resource["d48_persisted_fit_audit_json_utf8_bytes"] = (
            _fit_audit_json_utf8_bytes(geometry)
        )
        resource["d48_persisted_fit_audit_serialization"] = (
            "canonical_json_sort_keys_compact_separators_allow_nan_false_utf8"
        )
        return replace(result, resource_audit=resource, geometry_audit=geometry)

    d42.fit_d42_unified_shrinkage_lda = fit_with_d48_resource_audit
    return original_macs, original_top


def _close(actual: Any, expected: Any, atol: float = 1.0e-12) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        left = np.asarray(actual, dtype=np.float64)
        right = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return actual == expected
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=atol)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fp32_compiled_bias_rounding_bound(
    base_intercept: Any,
    final_intercept: Any,
    requested_bias: Any,
) -> np.ndarray:
    base = np.asarray(base_intercept, dtype=np.float32)
    final = np.asarray(final_intercept, dtype=np.float32)
    bias = np.asarray(requested_bias, dtype=np.float64)
    if (
        base.shape != final.shape
        or base.shape != bias.shape
        or not np.isfinite(base).all()
        or not np.isfinite(final).all()
        or not np.isfinite(bias).all()
    ):
        raise D48ProbeError("D48 FP32 compiled-bias bound input drift")
    compiled_delta = (final - base).astype(np.float32)
    ulp_sum = (
        np.abs(np.spacing(base).astype(np.float64))
        + np.abs(np.spacing(final).astype(np.float64))
        + np.abs(np.spacing(compiled_delta).astype(np.float64))
    )
    magnitude = np.maximum.reduce(
        (
            np.ones(bias.shape, dtype=np.float64),
            np.abs(base.astype(np.float64)),
            np.abs(final.astype(np.float64)),
            np.abs(bias),
        )
    )
    centering_bound = abs(float(np.mean(bias)))
    return (
        0.5 * ulp_sum
        + centering_bound
        + 64.0 * np.finfo(np.float64).eps * magnitude
    )


def _verify_d48_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d48_rows = [
        row
        for row in training_rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d48_rows) != 30:
        raise D48ProbeError("D48 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in d48_rows:
        resource = row.get("resource", {})
        geometry = row.get("geometry_summary", {})
        k = int(resource.get("old_k_shot", -1))
        class_counts: list[int] = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field)
            if not isinstance(audit, dict):
                raise D48ProbeError("D48 fit audit missing")
            c = int(audit.get("d48_class_count", -1))
            class_counts.append(c)
            expected_bias, expected = _one_shot_margin_residual(
                full_held_scores=audit.get("d48_full_held_logits_by_fold_class_logit"),
                block_held_scores=audit.get("d48_block_held_logits_by_fold_class_logit"),
                full_weight=float(audit.get("d48_d45_full_weight", np.nan)),
                block_weight=float(audit.get("d48_d45_block_weight", np.nan)),
                full_partition=audit.get("d45_full_inner_partition_audit"),
                block_partition=audit.get("d45_block_inner_partition_audit"),
                k_shot=k,
                class_count=c,
            )
            required = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d45_probe_arm": ARM,
                "d48_probe_arm": ARM,
                "d48_evidence_formula": EVIDENCE_FORMULA,
                "d48_margin_formula": MARGIN_FORMULA,
                "d48_bias_formula": BIAS_FORMULA,
                "d48_statistical_claim": STATISTICAL_CLAIM,
                "d48_support_supervised": k > 1,
                "d48_uses_outer_held_or_query": False,
                "d48_class_id_specific_formula": False,
                "d48_old_new_role_specific_branch": False,
                "d48_scene_receiver_handle_specific_branch": False,
                "d48_scan_clip_threshold_count": 0,
                "d48_beta_applied_once": True,
                "d48_beta_recomputed_or_iterated": False,
                "d48_beta_recomputes_rms_or_weight": False,
                "d48_independent_calibration_claim_allowed": False,
                "d48_boundary_status": expected["d48_boundary_status"],
                "d45_post_fusion_calibration_applied_once": True,
                "d45_post_fusion_calibration_recomputed_rms_weight_or_margin": False,
                "d45_post_fusion_calibration_coefficient_bitwise_unchanged": True,
            }
            if any(audit.get(name) != value for name, value in required.items()):
                raise D48ProbeError("D48 exact audit drift")
            if not (
                np.isclose(
                    float(audit.get("d48_d45_full_weight", np.nan)),
                    float(audit.get("d45_full_weight", np.nan)),
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(audit.get("d48_d45_block_weight", np.nan)),
                    float(audit.get("d45_block_weight", np.nan)),
                    rtol=0.0,
                    atol=1.0e-15,
                )
            ):
                raise D48ProbeError("D48/D45 global-weight binding drift")
            fields = [name for name in expected if name.startswith("d48_")]
            if any(not _close(audit.get(name), expected.get(name)) for name in fields):
                raise D48ProbeError("D48 OOF-margin evidence closure drift")
            if k > 1:
                labels = np.arange(c, dtype=np.int64)
                for logits_name, partition_name in (
                    (
                        "d48_full_held_logits_by_fold_class_logit",
                        "d45_full_inner_partition_audit",
                    ),
                    (
                        "d48_block_held_logits_by_fold_class_logit",
                        "d45_block_inner_partition_audit",
                    ),
                ):
                    logits = np.asarray(audit.get(logits_name), dtype=np.float64)
                    partition = audit.get(partition_name, {})
                    fold_ce = np.asarray(
                        partition.get("held_ce_by_fold_and_class"), dtype=np.float64
                    )
                    recomputed = []
                    for rank in range(k):
                        _macro, per_class = d45._class_balanced_cross_entropy(
                            logits[rank], labels, c
                        )
                        recomputed.append(per_class)
                    if not np.allclose(
                        recomputed, fold_ce, rtol=0.0, atol=1.0e-12
                    ):
                        raise D48ProbeError("D48 held-logit/D45 CE binding drift")
            hash_fields = (
                "d45_post_fusion_calibration_base_coefficient_sha256",
                "d45_post_fusion_calibration_final_coefficient_sha256",
                "d45_post_fusion_calibration_base_intercept_sha256",
                "d45_post_fusion_calibration_final_intercept_sha256",
            )
            if any(not _valid_sha256(audit.get(name)) for name in hash_fields):
                raise D48ProbeError("D48 calibration lifecycle SHA drift")
            base_intercept = np.asarray(
                audit.get("d45_post_fusion_calibration_base_intercept_fp32"),
                dtype=np.float32,
            )
            final_intercept = np.asarray(
                audit.get("d45_post_fusion_calibration_final_intercept_fp32"),
                dtype=np.float32,
            )
            coefficient = np.asarray(
                audit.get("d45_post_fusion_calibration_coefficient_fp32"),
                dtype=np.float32,
            )
            coefficient_hash = d45._array_sha256(coefficient)
            delta = np.asarray(
                audit.get("d45_post_fusion_calibration_intercept_delta_fp32"),
                dtype=np.float64,
            )
            compiled_bias_error = np.abs(delta - expected_bias)
            compiled_bias_bound = _fp32_compiled_bias_rounding_bound(
                base_intercept, final_intercept, expected_bias
            )
            if (
                base_intercept.shape != (c,)
                or final_intercept.shape != (c,)
                or coefficient.shape
                != (c, int(resource.get("coefficient_dimension", -1)))
                or not np.isfinite(base_intercept).all()
                or not np.isfinite(final_intercept).all()
                or not np.isfinite(coefficient).all()
                or audit[
                    "d45_post_fusion_calibration_base_coefficient_sha256"
                ]
                != coefficient_hash
                or audit[
                    "d45_post_fusion_calibration_final_coefficient_sha256"
                ]
                != coefficient_hash
                or audit["d45_post_fusion_calibration_base_intercept_sha256"]
                != d45._array_sha256(base_intercept)
                or audit["d45_post_fusion_calibration_final_intercept_sha256"]
                != d45._array_sha256(final_intercept)
                or delta.shape != (c,)
                or not np.array_equal(
                    delta.astype(np.float32), final_intercept - base_intercept
                )
                or np.any(compiled_bias_error > compiled_bias_bound)
            ):
                raise D48ProbeError("D48 coefficient/intercept residual drift")
            beta_nonzero = bool(np.any(np.asarray(expected_bias) != 0.0))
            intercept_hash_equal = (
                audit["d45_post_fusion_calibration_base_intercept_sha256"]
                == audit["d45_post_fusion_calibration_final_intercept_sha256"]
            )
            if (k == 1 or not beta_nonzero) != intercept_hash_equal:
                raise D48ProbeError("D48 one-shot intercept lifecycle drift")
        old_count, all_count = class_counts
        scoring = int(
            2
            * k
            * (1 if k <= 1 else k + 1)
            * int(resource.get("coefficient_dimension", -1))
            * (old_count**2 + all_count**2)
        )
        fusion = int(
            2
            * (int(resource.get("coefficient_dimension", -1)) + 1)
            * (old_count + all_count)
        )
        margin = _operation_upper_bound(k, old_count, all_count)
        evidence_peak = _evidence_peak_numeric_bytes(k, old_count, all_count)
        serialized_audit_bytes = _fit_audit_json_utf8_bytes(geometry)
        expected_resource = {
            "d48_outer_b20_training_count": 1,
            "d48_fused_query_state_count": 1,
            "d48_additional_lda_fit_count": 0,
            "d48_additional_optimizer_steps": 0,
            "d48_additional_query_state_count": 0,
            "d48_query_sidecar_bytes": 0,
            "d48_estimated_component_scoring_macs": scoring,
            "d48_estimated_affine_fusion_macs": fusion,
            "d48_margin_operation_upper_bound": margin,
            "d48_margin_operation_upper_bound_formula": (
                "0_if_K1_else_4*K*(C_old^2+C_all^2)+"
                "8*K*(C_old+C_all)+16*(C_old+C_all)+32"
            ),
            "d48_margin_operation_complexity": "O(K*C^2)",
            "d48_adaptation_evidence_peak_numeric_bytes": evidence_peak,
            "d48_persisted_fit_audit_json_utf8_bytes": serialized_audit_bytes,
            "d48_persisted_fit_audit_serialization": (
                "canonical_json_sort_keys_compact_separators_allow_nan_false_utf8"
            ),
            "d48_adaptation_evidence_memory_scope": (
                "support_only_full_block_fused_held_logits_and_margin_audit"
            ),
            "d48_final_state_dimension_unchanged_from_d45": True,
        }
        if any(resource.get(name) != value for name, value in expected_resource.items()):
            raise D48ProbeError("D48 resource audit drift")
        total = int(
            resource.get("estimated_metric_adaptation_macs", -1)
            + resource.get("estimated_lda_fit_macs", -1)
            + scoring
            + fusion
            + margin
        )
        if resource.get("estimated_adaptation_macs") != total:
            raise D48ProbeError("D48 total adaptation MAC-equivalent drift")
        formal_specs = (
            (
                "before_covariance_audit",
                "before",
                "d48_before_formal_intercept_fp16_values",
                "d48_before_formal_intercept_fp16_sha256",
                "d48_before_formal_coefficient_int8_sha256",
                "d48_before_fit_fp32_coefficient_compiles_to_formal_int8",
                "d48_before_fit_fp32_intercept_compiles_to_formal_fp16",
            ),
            (
                "final_covariance_audit",
                "final",
                "d48_final_formal_intercept_fp16_values",
                "d48_final_formal_intercept_fp16_sha256",
                "d48_final_formal_coefficient_int8_sha256",
                "d48_final_fit_fp32_coefficient_compiles_to_formal_int8",
                "d48_final_fit_fp32_intercept_compiles_to_formal_fp16",
            ),
        )
        for (
            fit_field,
            prefix,
            values_name,
            sha_name,
            coef_sha_name,
            coef_match_name,
            intercept_match_name,
        ) in formal_specs:
            fit_audit = geometry[fit_field]
            values = np.asarray(geometry.get(values_name), dtype=np.float16)
            expected_count = int(fit_audit["d48_class_count"])
            dimension = int(resource.get("coefficient_dimension", -1))
            expected_fp16 = np.asarray(
                fit_audit["d45_post_fusion_calibration_final_intercept_fp32"],
                dtype=np.float16,
            )
            raw_coef1 = np.asarray(
                geometry.get(f"d48_{prefix}_formal_coef1_qint8_values"),
                dtype=np.float64,
            )
            raw_coef2 = np.asarray(
                geometry.get(f"d48_{prefix}_formal_coef2_qint8_values"),
                dtype=np.float64,
            )
            raw_scale1 = np.asarray(
                geometry.get(f"d48_{prefix}_formal_scale1_fp16_values"),
                dtype=np.float64,
            )
            raw_scale2 = np.asarray(
                geometry.get(f"d48_{prefix}_formal_scale2_fp16_values"),
                dtype=np.float64,
            )
            coef1 = raw_coef1.astype(np.int8)
            coef2 = raw_coef2.astype(np.int8)
            scale1 = raw_scale1.astype(np.float16)
            scale2 = raw_scale2.astype(np.float16)
            fit_coefficient = np.asarray(
                fit_audit["d45_post_fusion_calibration_coefficient_fp32"],
                dtype=np.float32,
            )
            requantized = _audit_quantize_coefficients(fit_coefficient)
            coefficient_payload = (coef1, coef2, scale1, scale2)
            if (
                values.shape != (expected_count,)
                or not np.isfinite(values).all()
                or not _valid_sha256(geometry.get(sha_name))
                or geometry[sha_name] != d45._array_sha256(values)
                or not _valid_sha256(geometry.get(coef_sha_name))
                or raw_coef1.shape != (expected_count, dimension)
                or raw_coef2.shape != (expected_count, dimension)
                or raw_scale1.shape
                != (expected_count, len(COEFFICIENT_BLOCK_BOUNDS))
                or raw_scale2.shape
                != (expected_count, len(COEFFICIENT_BLOCK_BOUNDS))
                or not np.isfinite(raw_coef1).all()
                or not np.isfinite(raw_coef2).all()
                or not np.isfinite(raw_scale1).all()
                or not np.isfinite(raw_scale2).all()
                or not np.array_equal(raw_coef1, coef1.astype(np.float64))
                or not np.array_equal(raw_coef2, coef2.astype(np.float64))
                or not np.array_equal(raw_scale1, scale1.astype(np.float64))
                or not np.array_equal(raw_scale2, scale2.astype(np.float64))
                or np.any(scale1 <= 0)
                or np.any(scale2 <= 0)
                or not all(
                    np.array_equal(actual, expected)
                    for actual, expected in zip(coefficient_payload, requantized)
                )
                or geometry[coef_sha_name]
                != _formal_coefficient_payload_sha256(*coefficient_payload)
                or geometry.get(coef_match_name) is not True
                or geometry.get(intercept_match_name) is not True
                or not np.array_equal(values, expected_fp16)
            ):
                raise D48ProbeError("D48 formal FP16/state binding drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit.update(
                {
                    "d43_probe_arm": d45.ARM,
                    "d43_covariance_structure": d45.STRUCTURE,
                    "d45_probe_arm": d45.ARM,
                }
            )
        resource = row["resource"]
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"]
            + resource["estimated_lda_fit_macs"]
        )
    d45._verify_d45_fit_audits(sanitized)
    return len(d48_rows)


def _verify_d48_output(
    output: Path,
    probe_script_sha256: str,
    d45_helper_sha256: str,
    d44_helper_sha256: str,
    d43_helper_sha256: str,
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    expected = {
        "d48_d45_helper_sha256": d45_helper_sha256,
        "d48_d44_helper_sha256": d44_helper_sha256,
        "d48_d43_helper_sha256": d43_helper_sha256,
    }
    if any(closure.get(name) != value for name, value in expected.items()):
        raise D48ProbeError("D48 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = _verify_d48_fit_audits(rows)
    return {**evidence, "verified_d48_fit_row_count": count, **expected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d48-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D48ProbeError(f"D48 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_macs = None
    original_top = None
    runner_module_name = "d48_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d45_helper_sha256 = d43._sha256(D45_HELPER_PATH)
    d44_helper_sha256 = d43._sha256(d45.D44_HELPER_PATH)
    d43_helper_sha256 = d43._sha256(d44.D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_one_shot_margin_residual_fit(d42)
        original_macs, original_top = _install_d48_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D48ProbeError("D48 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d48_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d48_d45_helper_sha256": d45_helper_sha256,
                "d48_d44_helper_sha256": d44_helper_sha256,
                "d48_d43_helper_sha256": d43_helper_sha256,
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
    evidence = _verify_d48_output(
        output,
        probe_script_sha256,
        d45_helper_sha256,
        d44_helper_sha256,
        d43_helper_sha256,
    )
    metadata = {
        "schema": "cvs.phase2.d48.one_shot_oof_head_margin_residual_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d48_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "d45_helper_sha256": d45_helper_sha256,
        "d44_helper_sha256": d44_helper_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "evidence_formula": EVIDENCE_FORMULA,
        "margin_formula": MARGIN_FORMULA,
        "bias_formula": BIAS_FORMULA,
        "statistical_claim": STATISTICAL_CLAIM,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D48_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
