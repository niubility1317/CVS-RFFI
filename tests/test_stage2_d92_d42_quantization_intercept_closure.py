from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_d42_quantization_intercept_closure as qic


def _state(class_count: int = 8) -> d42.D42UnifiedShrinkageLDAState:
    code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
    for class_index in range(class_count):
        code1[class_index, class_index] = 1
    return d42.D42UnifiedShrinkageLDAState(
        schema=d42.SCHEMA_INT8,
        classes=tuple(f"tx_{index}" for index in range(class_count)),
        old_class_count=6,
        log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
        coef1_qint8=code1,
        coef2_qint8=np.zeros_like(code1),
        scale1_fp16=np.ones(
            (class_count, len(d42.BLOCK_SLICES)), dtype=np.float16
        ),
        scale2_fp16=np.full(
            (class_count, len(d42.BLOCK_SLICES)), 0.25, dtype=np.float16
        ),
        intercept_fp16=np.zeros(class_count, dtype=np.float16),
        coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
    )


def _support(
    class_count: int = 8, k_shot: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for class_index in range(class_count):
        for sample_index in range(k_shot):
            row = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
            row[class_index] = np.float32(
                1.0 + 0.2 * class_index + 0.01 * sample_index
            )
            row[160 + class_index] = np.float32(0.25 + 0.01 * sample_index)
            row[256 + class_index] = np.float32(0.125 + 0.01 * sample_index)
            rows.append(row)
            targets.append(class_index)
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


def _hand_qic_target(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Derive the frozen formula independently from the QIC implementation."""

    coefficient = np.asarray(d42.decode_d42_coefficients(state), dtype=np.float32)
    values: list[float] = []
    for class_index in range(len(state.classes)):
        class_rows = rows[targets == class_index]
        mean = np.asarray(
            [
                math.fsum(float(value) for value in class_rows[:, coordinate])
                / len(class_rows)
                for coordinate in range(d42.FEATURE_DIM)
            ],
            dtype=np.float64,
        )
        dot = math.fsum(
            float(mean[coordinate]) * float(coefficient[class_index, coordinate])
            for coordinate in range(d42.FEATURE_DIM)
        )
        values.append(-0.5 * dot + math.log(1.0 / len(state.classes)))
    common = math.fsum(values) / len(values)
    return np.asarray([value - common for value in values], dtype=np.float64)


def _centered_residual(deployed: np.ndarray, target: np.ndarray) -> float:
    common = math.fsum(float(value) for value in deployed) / len(deployed)
    return math.fsum(
        abs((float(deployed[index]) - common) - float(target[index]))
        for index in range(len(deployed))
    )


def _permuted_state(
    state: d42.D42UnifiedShrinkageLDAState, permutation: np.ndarray
) -> d42.D42UnifiedShrinkageLDAState:
    return replace(
        state,
        classes=tuple(state.classes[index] for index in permutation),
        coef1_qint8=state.coef1_qint8[permutation].copy(),
        coef2_qint8=state.coef2_qint8[permutation].copy(),
        scale1_fp16=state.scale1_fp16[permutation].copy(),
        scale2_fp16=state.scale2_fp16[permutation].copy(),
        intercept_fp16=state.intercept_fp16[permutation].copy(),
    )


def _assert_only_intercept_can_change(
    original: d42.D42UnifiedShrinkageLDAState,
    candidate: d42.D42UnifiedShrinkageLDAState,
) -> None:
    assert candidate.schema == original.schema
    assert candidate.classes == original.classes
    assert candidate.old_class_count == original.old_class_count
    assert candidate.covariance_policy == original.covariance_policy
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "coef_fp32",
        "intercept_fp32",
    ):
        assert getattr(candidate, name).tobytes() == getattr(original, name).tobytes()


def test_qic_publishes_the_strictly_closer_fp16_intercept_and_nothing_else() -> None:
    """A broken deployed-coordinate closure must not leave the E0 intercept live."""

    state = _state()
    rows, targets = _support()
    expected = _hand_qic_target(state, rows, targets)

    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        state, rows, targets
    )

    assert audit["d92_qic_active"] is True
    assert audit["d92_qic_fallback_active"] is False
    assert audit["d92_qic_modified_state_field_names"] == ["intercept_fp16"]
    assert candidate is not state
    assert qic.d42_qic_state_sha256(candidate) == audit["d92_qic_final_state_sha256"]
    _assert_only_intercept_can_change(state, candidate)
    assert candidate.intercept_fp16.tobytes() == expected.astype(np.float16).tobytes()
    assert audit["d92_qic_intercept_fp16_bit_change_count"] == int(
        np.count_nonzero(
            candidate.intercept_fp16.view(np.uint16)
            != state.intercept_fp16.view(np.uint16)
        )
    )
    assert audit["d92_qic_candidate_residual_l1"] < audit["d92_qic_e0_residual_l1"]
    assert audit["d92_qic_e0_residual_l1"] == pytest.approx(
        _centered_residual(state.intercept_fp16, expected)
    )
    assert audit["d92_qic_candidate_residual_l1"] == pytest.approx(
        _centered_residual(candidate.intercept_fp16, expected)
    )
    assert audit["d92_qic_coefficient_decode_count"] == 1
    assert audit["d92_qic_additional_full_fit_count"] == 0
    assert audit["d92_qic_block_fit_count"] == 0
    assert audit["d92_qic_loo_fit_count"] == 0
    assert audit["d92_qic_fisher_scan_count"] == 0
    assert audit["d92_qic_candidate_scan_count"] == 0
    assert audit["d92_qic_persistent_state_bytes_delta"] == 0
    assert audit["d92_qic_query_macs_delta"] == 0
    assert audit["d92_qic_query_fit_access"] is False
    assert audit["d92_qic_query_update_access"] is False
    assert audit["d92_qic_query_selection_access"] is False
    assert audit["d92_qic_query_truth_access"] is False
    assert audit["d92_qic_query_role_oracle_access"] is False
    assert audit["d92_qic_query_class_quota_access"] is False
    assert audit["d92_qic_query_global_reassignment"] is False
    assert audit["d92_qic_support_candidate_matrix_bytes"] == 0
    assert audit["d92_qic_support_288_square_matrix_bytes"] == 0
    assert audit["d92_qic_support_macs_upper_bound"] == len(state.classes) * (
        5 + 1
    ) * d42.FEATURE_DIM


def test_qic_falls_back_to_the_exact_e0_state_when_fp16_cannot_improve() -> None:
    """Removing the strict-improvement gate would incorrectly republish E0 as QIC."""

    state = _state()
    rows, targets = _support()
    aligned = replace(state, intercept_fp16=_hand_qic_target(state, rows, targets).astype(np.float16))

    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        aligned, rows, targets
    )

    assert candidate is aligned
    assert audit["d92_qic_active"] is False
    assert audit["d92_qic_fallback_active"] is True
    assert audit["d92_qic_final_state_sha256"] == audit["d92_qic_e0_state_sha256"]
    assert audit["d92_qic_modified_state_field_names"] == []
    assert audit["d92_qic_intercept_fp16_bit_change_count"] == 0
    assert audit["d92_qic_candidate_residual_l1"] >= audit["d92_qic_e0_residual_l1"]


def test_qic_reports_a_hypothetical_bit_change_separately_from_an_e0_fallback() -> None:
    """A rejected FP16 proposal must not be recorded as a published state delta."""

    state = _state()
    rows = np.zeros((len(state.classes) * 3, d42.FEATURE_DIM), dtype=np.float32)
    targets = np.repeat(np.arange(len(state.classes), dtype=np.int64), 3)
    for class_index in range(len(state.classes)):
        rows[targets == class_index, class_index] = np.float32(1.0 + class_index)
    target = _hand_qic_target(state, rows, targets)
    shifted = replace(state, intercept_fp16=(target + 4.0).astype(np.float16))

    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        shifted, rows, targets
    )

    assert candidate is shifted
    assert audit["d92_qic_active"] is False
    assert audit["d92_qic_fallback_active"] is True
    assert audit["d92_qic_intercept_fp16_bit_change_count"] == 0
    assert audit["d92_qic_candidate_intercept_fp16_bit_change_count"] == len(
        state.classes
    )


@pytest.mark.parametrize("k_shot", [1, 2])
def test_qic_k1_k2_are_exact_d92_full_aliases(k_shot: int) -> None:
    """Applying QIC to a frozen small-K alias must never write a new state."""

    state = _state()
    rows, targets = _support(k_shot=k_shot)

    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        state, rows, targets
    )

    assert candidate is state
    assert audit["d92_qic_active"] is False
    assert audit["d92_qic_fallback_active"] is False
    assert audit["d92_qic_fallback_reason"] == "K1_K2_EXACT_D92_FULL_ALIAS"
    assert audit["d92_qic_k_shot"] == k_shot
    assert audit["d92_qic_coefficient_decode_count"] == 0
    inactive = qic.d42_qic_inactive_receipt(state, k_shot=k_shot)
    assert inactive["d92_qic_fallback_reason"] == "K1_K2_EXACT_D92_FULL_ALIAS"
    assert inactive["d92_qic_final_state_sha256"] == inactive[
        "d92_qic_e0_state_sha256"
    ]


def test_qic_is_byte_exact_under_support_row_permutation_and_class_equivariant() -> None:
    """A row-order or class-order branch would violate the shared all-class rule."""

    state = _state()
    rows, targets = _support()
    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        state, rows, targets
    )
    row_permuted, row_audit = qic.apply_d42_quantization_intercept_closure(
        state, rows[::-1], targets[::-1]
    )

    permutation = np.asarray([2, 0, 1, 5, 3, 4, 7, 6], dtype=np.int64)
    inverse = np.argsort(permutation)
    old_to_new = np.empty(len(permutation), dtype=np.int64)
    old_to_new[permutation] = np.arange(len(permutation))
    permuted_state = _permuted_state(state, permutation)
    class_permuted, class_audit = qic.apply_d42_quantization_intercept_closure(
        permuted_state, rows, old_to_new[targets]
    )

    assert row_permuted.intercept_fp16.tobytes() == candidate.intercept_fp16.tobytes()
    assert row_audit == audit
    assert class_permuted.intercept_fp16[inverse].tobytes() == candidate.intercept_fp16.tobytes()
    assert class_audit["d92_qic_class_permutation_equivariant"] is True
    assert class_audit["d92_qic_row_permutation_invariant"] is True


def test_qic_nonfinite_support_falls_back_but_registry_closure_drift_raises() -> None:
    """A numerical bad row is recoverable; a missing registered class is not."""

    state = _state()
    rows, targets = _support()
    bad_rows = rows.copy()
    bad_rows[0, 0] = np.inf

    candidate, audit = qic.apply_d42_quantization_intercept_closure(
        state, bad_rows, targets
    )

    assert candidate is state
    assert audit["d92_qic_active"] is False
    assert audit["d92_qic_fallback_active"] is True
    assert audit["d92_qic_fallback_reason"] == "support_nonfinite"
    missing = targets.copy()
    missing[missing == len(state.classes) - 1] = 0
    with pytest.raises(qic.D92D42QICError, match="support/state closure"):
        qic.apply_d42_quantization_intercept_closure(state, rows, missing)
