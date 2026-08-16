"""Support-only D42 quantization-intercept closure for the E0 FULL head.

QIC preserves the deployed D42 coefficient codec and replaces only the FP16
intercept with the equal-prior intercept implied by the *decoded* coefficient
rows and the same transformed support that produced the E0 state.  It has no
query input, no second fit, and no candidate sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


STATE_POSTPROCESS_MODE = "d42_quantization_intercept_closure"
FORMULA_REVISION = "d42_quantization_intercept_self_consistency_v1"
_ALIAS_REASON = "K1_K2_EXACT_D92_FULL_ALIAS"


class D92D42QICError(ValueError):
    """Raised for structural QIC input or lifecycle drift."""


def _state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    digest = hashlib.sha256()
    metadata = {
        "classes": list(state.classes),
        "covariance_policy": state.covariance_policy,
        "old_class_count": int(state.old_class_count),
        "schema": state.schema,
    }
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "coef_fp32",
        "intercept_fp32",
    ):
        value = np.ascontiguousarray(getattr(state, name))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def d42_qic_state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    """Return the canonical deployment identity used by QIC receipts."""

    _validate_state(state)
    return _state_sha256(state)


def _validate_state(state: d42.D42UnifiedShrinkageLDAState) -> None:
    if not isinstance(state, d42.D42UnifiedShrinkageLDAState) or not state.is_int8:
        raise D92D42QICError("QIC requires a compiled D42 int8 state")
    if (
        len(state.classes) < 2
        or len(set(state.classes)) != len(state.classes)
        or any(not isinstance(value, str) or not value for value in state.classes)
        or not 2 <= int(state.old_class_count) <= len(state.classes)
        or state.intercept_fp16.shape != (len(state.classes),)
    ):
        raise D92D42QICError("QIC state registry closure drift")


def _support_closure(
    state: d42.D42UnifiedShrinkageLDAState,
    transformed_support_rows: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.asarray(transformed_support_rows, dtype=np.float32)
    raw_targets = np.asarray(targets)
    if not np.issubdtype(raw_targets.dtype, np.integer):
        raise D92D42QICError("QIC support/state closure drift")
    target_array = np.asarray(raw_targets, dtype=np.int64)
    class_count = len(state.classes)
    if (
        rows.ndim != 2
        or rows.shape[1] != d42.FEATURE_DIM
        or target_array.ndim != 1
        or len(rows) != len(target_array)
        or len(rows) == 0
        or int(np.min(target_array)) < 0
        or int(np.max(target_array)) >= class_count
    ):
        raise D92D42QICError("QIC support/state closure drift")
    counts = np.bincount(target_array, minlength=class_count)
    if len(counts) != class_count or int(np.min(counts)) < 1:
        raise D92D42QICError("QIC support/state closure drift")
    if not bool(np.all(counts == counts[0])):
        raise D92D42QICError("QIC support/state closure drift")
    return np.ascontiguousarray(rows), target_array, int(counts[0])


def _row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")


def _canonical_indices(rows: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise D92D42QICError("QIC canonical support closure drift")
    ordered = sorted(selected.tolist(), key=lambda index: _row_bytes(rows[index]))
    return np.asarray(ordered, dtype=np.int64)


def _canonical_mean(rows: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return a row-order-invariant float64 support mean."""

    ordered = _canonical_indices(rows, indices)
    return np.sum(
        np.asarray(rows[ordered], dtype=np.float64), axis=0, dtype=np.float64
    ) / float(len(ordered))


def _canonical_class_order(state: d42.D42UnifiedShrinkageLDAState) -> list[int]:
    return sorted(
        range(len(state.classes)),
        key=lambda index: state.classes[index].encode("utf-8"),
    )


def _center(values: np.ndarray, order: list[int]) -> tuple[np.ndarray, float]:
    common = math.fsum(float(values[index]) for index in order) / len(order)
    centered = np.asarray(
        [float(value) - common for value in values], dtype=np.float64
    )
    return centered, common


def _centered_residual(
    deployed_intercept: np.ndarray,
    target_centered: np.ndarray,
    order: list[int],
) -> tuple[float, float]:
    centered, common = _center(
        np.asarray(deployed_intercept, dtype=np.float64), order
    )
    residual = math.fsum(
        abs(float(centered[index]) - float(target_centered[index]))
        for index in order
    )
    return float(residual), float(common)


def _resource_values(
    *, row_count: int, class_count: int, k_shot: int
) -> tuple[int, int, int, int, int]:
    rows_bytes = int(row_count * d42.FEATURE_DIM * np.dtype(np.float32).itemsize)
    decoded_bytes = int(
        class_count * d42.FEATURE_DIM * np.dtype(np.float32).itemsize
    )
    mean_bytes = int(class_count * d42.FEATURE_DIM * np.dtype(np.float64).itemsize)
    vector_bytes = int(class_count * (8 * 6 + 2))
    transient_bytes = int(rows_bytes + decoded_bytes + mean_bytes + vector_bytes)
    support_macs = int(class_count * (int(k_shot) + 1) * d42.FEATURE_DIM)
    return support_macs, transient_bytes, rows_bytes, decoded_bytes, mean_bytes


def _base_audit(
    state: d42.D42UnifiedShrinkageLDAState,
    *,
    active: bool,
    fallback_active: bool,
    fallback_reason: str | None,
    final_state_sha256: str,
    modified_state_field_names: list[str],
    changed_intercept_count: int,
    candidate_intercept_count: int | None,
    k_shot: int | None,
    coefficient_decode_count: int,
    e0_residual_l1: float | None,
    candidate_residual_l1: float | None,
    e0_intercept_common_shift: float | None,
    candidate_intercept_common_shift: float | None,
    target_center_sum_abs: float | None,
    support_macs: int,
    transient_bytes: int,
    rows_bytes: int,
    decoded_bytes: int,
    mean_bytes: int,
) -> dict[str, Any]:
    e0_sha = _state_sha256(state)
    reduction = (
        None
        if e0_residual_l1 is None or candidate_residual_l1 is None
        else float(e0_residual_l1 - candidate_residual_l1)
    )
    return {
        "d92_qic_active": bool(active),
        "d92_qic_fallback_active": bool(fallback_active),
        "d92_qic_fallback_reason": fallback_reason,
        "d92_qic_formula_revision": FORMULA_REVISION,
        "d92_qic_state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "d92_qic_direct_state_publish": True,
        "d92_qic_deployed_d42_coordinate": True,
        "d92_qic_common_shift_invariant_residual": True,
        "d92_qic_equal_prior_policy": "log_1_over_registered_class_count",
        "d92_qic_all_class_shared_formula": True,
        "d92_qic_class_permutation_equivariant": True,
        "d92_qic_row_permutation_invariant": True,
        "d92_qic_support_row_canonicalization": "float32_row_bytes_lexicographic",
        "d92_qic_centering_reduction": "math.fsum_canonical_class_registry_order",
        "d92_qic_e0_state_sha256": e0_sha,
        "d92_qic_final_state_sha256": str(final_state_sha256),
        "d92_qic_modified_state_field_names": list(modified_state_field_names),
        "d92_qic_intercept_fp16_bit_change_count": int(changed_intercept_count),
        "d92_qic_state_delta_intercept_l1": int(changed_intercept_count),
        "d92_qic_candidate_intercept_fp16_bit_change_count": int(
            changed_intercept_count
            if candidate_intercept_count is None
            else candidate_intercept_count
        ),
        "d92_qic_coef1_byte_exact": True,
        "d92_qic_coef2_byte_exact": True,
        "d92_qic_scale1_byte_exact": True,
        "d92_qic_scale2_byte_exact": True,
        "d92_qic_log_diag_byte_exact": True,
        "d92_qic_coef_fp32_byte_exact": True,
        "d92_qic_intercept_fp32_byte_exact": True,
        "d92_qic_class_registry_byte_exact": True,
        "d92_qic_state_shape_byte_exact": True,
        "d92_qic_intercept_byte_exact": not bool(active),
        "d92_qic_class_count": int(len(state.classes)),
        "d92_qic_k_shot": None if k_shot is None else int(k_shot),
        "d92_qic_e0_residual_l1": e0_residual_l1,
        "d92_qic_candidate_residual_l1": candidate_residual_l1,
        "d92_qic_residual_reduction_l1": reduction,
        "d92_qic_target_center_sum_abs": target_center_sum_abs,
        "d92_qic_e0_intercept_common_shift": e0_intercept_common_shift,
        "d92_qic_candidate_intercept_common_shift": candidate_intercept_common_shift,
        "d92_qic_requantize_call_count": 0,
        "d92_qic_coefficient_decode_count": int(coefficient_decode_count),
        "d92_qic_additional_full_fit_count": 0,
        "d92_qic_block_fit_count": 0,
        "d92_qic_loo_fit_count": 0,
        "d92_qic_fisher_scan_count": 0,
        "d92_qic_candidate_scan_count": 0,
        "d92_qic_support_only": True,
        "d92_qic_support_macs_upper_bound": int(support_macs),
        "d92_qic_support_transient_bytes_upper_bound": int(transient_bytes),
        "d92_qic_support_rows_bytes_upper_bound": int(rows_bytes),
        "d92_qic_support_decoded_coefficient_bytes_upper_bound": int(decoded_bytes),
        "d92_qic_support_mean_bytes_upper_bound": int(mean_bytes),
        "d92_qic_support_candidate_matrix_bytes": 0,
        "d92_qic_support_288_square_matrix_bytes": 0,
        "d92_qic_support_work_complexity": "O(C*K*288)+O(C*288)",
        "d92_qic_persistent_state_bytes_delta": 0,
        "d92_qic_query_macs_delta": 0,
        "d92_qic_query_rows_used": 0,
        "d92_qic_query_fit_access": False,
        "d92_qic_query_update_access": False,
        "d92_qic_query_selection_access": False,
        "d92_qic_query_truth_access": False,
        "d92_qic_query_role_oracle_access": False,
        "d92_qic_query_class_quota_access": False,
        "d92_qic_query_global_reassignment": False,
        "d92_qic_clean_sample_access": False,
        "d92_qic_source_sample_access": False,
    }


def d42_qic_inactive_receipt(
    state: d42.D42UnifiedShrinkageLDAState, *, k_shot: int | None = None
) -> dict[str, Any]:
    """Return the exact-D92-FULL receipt used for K1/K2 aliases."""

    _validate_state(state)
    if k_shot is not None and int(k_shot) not in (1, 2):
        raise D92D42QICError("QIC inactive receipt is reserved for K1/K2")
    class_count = len(state.classes)
    shot = None if k_shot is None else int(k_shot)
    row_count = 0 if shot is None else int(class_count * shot)
    resources = _resource_values(
        row_count=row_count, class_count=class_count, k_shot=0 if shot is None else shot
    )
    sha = _state_sha256(state)
    return _base_audit(
        state,
        active=False,
        fallback_active=False,
        fallback_reason=_ALIAS_REASON,
        final_state_sha256=sha,
        modified_state_field_names=[],
        changed_intercept_count=0,
        candidate_intercept_count=0,
        k_shot=shot,
        coefficient_decode_count=0,
        e0_residual_l1=None,
        candidate_residual_l1=None,
        e0_intercept_common_shift=None,
        candidate_intercept_common_shift=None,
        target_center_sum_abs=None,
        support_macs=resources[0],
        transient_bytes=resources[1],
        rows_bytes=resources[2],
        decoded_bytes=resources[3],
        mean_bytes=resources[4],
    )


def apply_d42_quantization_intercept_closure(
    state: d42.D42UnifiedShrinkageLDAState,
    transformed_support_rows: np.ndarray,
    targets: np.ndarray,
) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
    """Publish a stricter self-consistent FP16 intercept or exact E0.

    ``transformed_support_rows`` must be the support-only rows in the deployed
    D42 coordinate from the same already-completed E0 FULL registration.
    """

    _validate_state(state)
    rows, target_array, k_shot = _support_closure(
        state, transformed_support_rows, targets
    )
    if k_shot <= 2:
        return state, d42_qic_inactive_receipt(state, k_shot=k_shot)

    class_count = len(state.classes)
    resources = _resource_values(
        row_count=len(rows), class_count=class_count, k_shot=k_shot
    )
    e0_sha = _state_sha256(state)
    decode_count = 0
    e0_residual: float | None = None
    candidate_residual: float | None = None
    e0_common: float | None = None
    candidate_common: float | None = None
    target_center_sum_abs: float | None = None

    def fallback(
        reason: str, *, candidate_intercept_count: int = 0
    ) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
        return state, _base_audit(
            state,
            active=False,
            fallback_active=True,
            fallback_reason=str(reason),
            final_state_sha256=e0_sha,
            modified_state_field_names=[],
            changed_intercept_count=0,
            candidate_intercept_count=candidate_intercept_count,
            k_shot=k_shot,
            coefficient_decode_count=decode_count,
            e0_residual_l1=e0_residual,
            candidate_residual_l1=candidate_residual,
            e0_intercept_common_shift=e0_common,
            candidate_intercept_common_shift=candidate_common,
            target_center_sum_abs=target_center_sum_abs,
            support_macs=resources[0],
            transient_bytes=resources[1],
            rows_bytes=resources[2],
            decoded_bytes=resources[3],
            mean_bytes=resources[4],
        )

    if not np.isfinite(rows).all():
        return fallback("support_nonfinite")
    try:
        coefficient = np.asarray(d42.decode_d42_coefficients(state), dtype=np.float32)
        decode_count = 1
    except (d42.D42UnifiedShrinkageLDAError, FloatingPointError):
        return fallback("coefficient_decode_nonfinite")
    if coefficient.shape != (class_count, d42.FEATURE_DIM) or not np.isfinite(
        coefficient
    ).all():
        return fallback("coefficient_decode_nonfinite")

    means = np.empty((class_count, d42.FEATURE_DIM), dtype=np.float64)
    for class_index in range(class_count):
        means[class_index] = _canonical_mean(
            rows, np.flatnonzero(target_array == class_index)
        )
    if not np.isfinite(means).all():
        return fallback("support_mean_nonfinite")

    raw_target = np.empty(class_count, dtype=np.float64)
    log_prior = math.log(1.0 / class_count)
    for class_index in range(class_count):
        dot = math.fsum(
            float(means[class_index, coordinate])
            * float(coefficient[class_index, coordinate])
            for coordinate in range(d42.FEATURE_DIM)
        )
        raw_target[class_index] = -0.5 * dot + log_prior
    if not np.isfinite(raw_target).all():
        return fallback("intercept_target_nonfinite")

    class_order = _canonical_class_order(state)
    target_centered, _ = _center(raw_target, class_order)
    target_center_sum_abs = abs(
        math.fsum(float(target_centered[index]) for index in class_order)
    )
    if not np.isfinite(target_centered).all() or not math.isfinite(
        target_center_sum_abs
    ):
        return fallback("intercept_target_nonfinite")
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_intercept = np.asarray(target_centered, dtype=np.float16)
    if not np.isfinite(candidate_intercept).all():
        return fallback("candidate_intercept_nonfinite")

    e0_residual, e0_common = _centered_residual(
        state.intercept_fp16, target_centered, class_order
    )
    candidate_residual, candidate_common = _centered_residual(
        candidate_intercept, target_centered, class_order
    )
    if not (
        math.isfinite(e0_residual)
        and math.isfinite(candidate_residual)
        and math.isfinite(e0_common)
        and math.isfinite(candidate_common)
    ):
        return fallback("residual_nonfinite")
    changed = int(
        np.count_nonzero(
            candidate_intercept.view(np.uint16) != state.intercept_fp16.view(np.uint16)
        )
    )
    if changed <= 0:
        return fallback("intercept_fp16_byte_exact")
    if not candidate_residual < e0_residual:
        return fallback(
            "self_consistency_not_strictly_improved",
            candidate_intercept_count=changed,
        )

    candidate_state = replace(state, intercept_fp16=candidate_intercept)
    final_sha = _state_sha256(candidate_state)
    if final_sha == e0_sha:
        return fallback("state_sha_unchanged", candidate_intercept_count=changed)
    return candidate_state, _base_audit(
        state,
        active=True,
        fallback_active=False,
        fallback_reason=None,
        final_state_sha256=final_sha,
        modified_state_field_names=["intercept_fp16"],
        changed_intercept_count=changed,
        candidate_intercept_count=None,
        k_shot=k_shot,
        coefficient_decode_count=decode_count,
        e0_residual_l1=e0_residual,
        candidate_residual_l1=candidate_residual,
        e0_intercept_common_shift=e0_common,
        candidate_intercept_common_shift=candidate_common,
        target_center_sum_abs=target_center_sum_abs,
        support_macs=resources[0],
        transient_bytes=resources[1],
        rows_bytes=resources[2],
        decoded_bytes=resources[3],
        mean_bytes=resources[4],
    )


__all__ = [
    "D92D42QICError",
    "FORMULA_REVISION",
    "STATE_POSTPROCESS_MODE",
    "apply_d42_quantization_intercept_closure",
    "d42_qic_inactive_receipt",
    "d42_qic_state_sha256",
]
