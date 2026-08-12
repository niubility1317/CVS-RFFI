from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_d42_tail_pair_code_exchange as tpce


def _state(class_count: int = 8) -> d42.D42UnifiedShrinkageLDAState:
    code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
    for index in range(class_count):
        code1[index, index] = 1
    return d42.D42UnifiedShrinkageLDAState(
        schema=d42.SCHEMA_INT8,
        classes=tuple(f"tx_{index}" for index in range(class_count)),
        old_class_count=6,
        log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
        coef1_qint8=code1,
        coef2_qint8=np.zeros_like(code1),
        scale1_fp16=np.ones((class_count, len(d42.BLOCK_SLICES)), dtype=np.float16),
        scale2_fp16=np.full(
            (class_count, len(d42.BLOCK_SLICES)), 0.25, dtype=np.float16
        ),
        intercept_fp16=np.zeros(class_count, dtype=np.float16),
        coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
    )


def _support(class_count: int = 8, k_shot: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for class_index in range(class_count):
        for sample_index in range(k_shot):
            row = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.0 + 0.01 * sample_index)
            # Give every D42 block a nonzero, deterministic coordinate.
            row[160 + class_index] = np.float32(0.5 + 0.01 * sample_index)
            row[256 + class_index] = np.float32(0.25 + 0.01 * sample_index)
            rows.append(row)
            targets.append(class_index)
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


def test_tpce_directly_publishes_only_code2_and_improves_frozen_tails() -> None:
    state = _state()
    rows, targets = _support()

    candidate, audit = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=6
    )

    assert audit["d92_tpce_active"] is True
    assert audit["d92_tpce_fallback_active"] is False
    assert audit["d92_tpce_requantize_call_count"] == 0
    assert audit["d92_tpce_changed_code2_count"] > 0
    assert audit["d92_tpce_requested_atomic_exchange_count"] == audit[
        "d92_tpce_applied_atomic_exchange_count"
    ]
    assert audit["d92_tpce_aggregate_saturation_count"] == 0
    assert all(
        audit[name] is True
        for name in (
            "d92_tpce_code1_byte_exact",
            "d92_tpce_scale1_byte_exact",
            "d92_tpce_scale2_byte_exact",
            "d92_tpce_intercept_byte_exact",
            "d92_tpce_log_diag_byte_exact",
            "d92_tpce_support_guard_pass",
            "d92_tpce_class_permutation_equivariant",
        )
    )
    tolerance = audit["d92_tpce_guard_tolerance"]
    assert min(audit["d92_tpce_old_tail_gain_by_class"]) > tolerance
    assert audit["d92_tpce_pooled_new_cross_tail_gain"] > tolerance
    assert audit["d92_tpce_pooled_new_allclass_tail_gain"] >= -tolerance
    assert audit["d92_tpce_old_to_new_hinge_delta"] <= tolerance
    assert audit["d92_tpce_new_to_old_hinge_delta"] <= tolerance
    assert candidate.coef2_qint8.tobytes() != state.coef2_qint8.tobytes()
    for name in (
        "coef1_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "log_diag_fp32",
    ):
        assert getattr(candidate, name).tobytes() == getattr(state, name).tobytes()
    assert candidate.persistent_state_bytes == state.persistent_state_bytes


def test_tpce_is_order_independent_and_label_permutation_equivariant() -> None:
    state = _state()
    rows, targets = _support()
    reverse = np.arange(len(rows))[::-1]
    candidate_a, audit_a = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=6
    )
    candidate_b, audit_b = tpce.apply_d42_tail_pair_code_exchange(
        state, rows[reverse], targets[reverse], old_class_count=6
    )
    assert candidate_a.coef2_qint8.tobytes() == candidate_b.coef2_qint8.tobytes()
    assert audit_a["d92_tpce_requested_atomic_exchange_count"] == audit_b[
        "d92_tpce_requested_atomic_exchange_count"
    ]

    permutation = np.asarray([2, 0, 1, 5, 3, 4, 7, 6], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted_state = d42.D42UnifiedShrinkageLDAState(
        schema=state.schema,
        classes=tuple(state.classes[index] for index in permutation),
        old_class_count=6,
        log_diag_fp32=state.log_diag_fp32.copy(),
        coef1_qint8=state.coef1_qint8[permutation].copy(),
        coef2_qint8=state.coef2_qint8[permutation].copy(),
        scale1_fp16=state.scale1_fp16[permutation].copy(),
        scale2_fp16=state.scale2_fp16[permutation].copy(),
        intercept_fp16=state.intercept_fp16[permutation].copy(),
        coef_fp32=state.coef_fp32.copy(),
        intercept_fp32=state.intercept_fp32.copy(),
        covariance_policy=state.covariance_policy,
    )
    old_to_new = np.empty(len(permutation), dtype=np.int64)
    old_to_new[permutation] = np.arange(len(permutation))
    permuted_targets = old_to_new[targets]
    candidate_p, audit_p = tpce.apply_d42_tail_pair_code_exchange(
        permuted_state, rows, permuted_targets, old_class_count=6
    )
    assert candidate_p.coef2_qint8[inverse].tobytes() == candidate_a.coef2_qint8.tobytes()
    assert audit_p["d92_tpce_class_permutation_equivariant"] is True


def test_tpce_aggregate_saturation_falls_back_exactly_to_e0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    rows, targets = _support(k_shot=5)
    forced = replace(
        state,
        coef2_qint8=np.full_like(state.coef2_qint8, 127, dtype=np.int8),
    )

    # Force the atomic builder to return an aggregate that cannot be published.
    monkeypatch.setattr(
        tpce,
        "_aggregate_atomic_exchanges",
        lambda *args, **kwargs: (
            np.full(forced.coef2_qint8.shape, 1, dtype=np.int32),
            1,
            1,
        ),
    )
    candidate, audit = tpce.apply_d42_tail_pair_code_exchange(
        forced, rows, targets, old_class_count=6
    )
    assert audit["d92_tpce_active"] is False
    assert audit["d92_tpce_fallback_active"] is True
    assert audit["d92_tpce_fallback_reason"] == "aggregate_saturation"
    assert audit["d92_tpce_aggregate_saturation_count"] > 0
    assert audit["d92_tpce_applied_atomic_exchange_count"] == 0
    assert audit["d92_tpce_changed_code2_count"] == 0
    assert audit["d92_tpce_final_state_sha256"] == audit["d92_tpce_e0_state_sha256"]
    assert candidate.coef2_qint8.tobytes() == forced.coef2_qint8.tobytes()

