from __future__ import annotations

from dataclasses import replace
import time

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


def _interference_state_and_support() -> tuple[
    d42.D42UnifiedShrinkageLDAState, np.ndarray, np.ndarray
]:
    """Return a fixed support-only case where synchronous atoms harm old class 5.

    The fixture is generated from a fixed stream rather than a class-specific
    branch.  It keeps all six old and two new classes under the same formula.
    """

    class_count = 8
    k_shot = 5
    rng = np.random.default_rng(200)
    state: d42.D42UnifiedShrinkageLDAState | None = None
    rows: np.ndarray | None = None
    targets: np.ndarray | None = None
    for _ in range(443):
        code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
        for class_index in range(class_count):
            for block_start in (0, 160, 256):
                code1[class_index, block_start + class_index] = rng.integers(1, 5)
        for class_index in range(class_count):
            for other_index in range(class_count):
                if class_index == other_index:
                    continue
                for block_start in (0, 160, 256):
                    code1[class_index, block_start + other_index] = rng.integers(
                        -2, 3
                    )
        state = d42.D42UnifiedShrinkageLDAState(
            schema=d42.SCHEMA_INT8,
            classes=tuple(f"tx_{index}" for index in range(class_count)),
            old_class_count=6,
            log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
            coef1_qint8=code1,
            coef2_qint8=np.zeros_like(code1),
            scale1_fp16=np.full((class_count, 3), 0.1, dtype=np.float16),
            scale2_fp16=np.full((class_count, 3), 0.05, dtype=np.float16),
            intercept_fp16=np.zeros(class_count, dtype=np.float16),
            coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
            intercept_fp32=np.zeros(0, dtype=np.float32),
            covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
        )
        generated_rows: list[np.ndarray] = []
        generated_targets: list[int] = []
        for class_index in range(class_count):
            for _sample_index in range(k_shot):
                row = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
                for block_start, amplitude in ((0, 1.0), (160, 0.7), (256, 0.45)):
                    row[block_start : block_start + class_count] = rng.normal(
                        0.0, 0.20, class_count
                    )
                    row[block_start + class_index] += amplitude + rng.normal(
                        0.0, 0.1
                    )
                generated_rows.append(row)
                generated_targets.append(class_index)
        rows = np.stack(generated_rows)
        targets = np.asarray(generated_targets, dtype=np.int64)
    assert state is not None and rows is not None and targets is not None
    return state, rows, targets


def test_tpce_selects_a_pareto_safe_subset_when_full_sync_harms_an_old_tail() -> None:
    """Would fail if TPCE continues to publish every frozen-tail atom at once."""

    state, rows, targets = _interference_state_and_support()
    base_scores = tpce._score(state, rows)
    old_tails, new_tail, relations, _ = tpce._fixed_tail_and_relations(
        base_scores, targets, old_class_count=6
    )
    full_delta, full_requested, _ = tpce._aggregate_atomic_exchanges(
        state, rows, relations
    )
    full_state = replace(
        state,
        coef2_qint8=(state.coef2_qint8.astype(np.int32) + full_delta).astype(
            np.int8
        ),
    )
    full_scores = tpce._score(full_state, rows)
    base_margin = tpce._true_vs_all_margin(base_scores, targets)
    full_margin = tpce._true_vs_all_margin(full_scores, targets)
    scale = max(1.0, float(np.max(np.abs(base_scores))), float(np.max(np.abs(full_scores))))
    tolerance = tpce.GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * scale
    assert float(np.mean(full_margin[old_tails[5]] - base_margin[old_tails[5]])) < -tolerance

    candidate, audit = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=6
    )

    assert audit["d92_tpce_active"] is True
    assert audit["d92_tpce_fallback_active"] is False
    assert audit["d92_tpce_generated_atomic_exchange_count"] == full_requested
    assert 0 < audit["d92_tpce_selected_atomic_exchange_count"] < full_requested
    assert audit["d92_tpce_generated_atomic_exchange_count"] == audit[
        "d92_tpce_selected_atomic_exchange_count"
    ] + audit["d92_tpce_rejected_atomic_exchange_count"]
    assert audit["d92_tpce_requested_atomic_exchange_count"] == audit[
        "d92_tpce_applied_atomic_exchange_count"
    ]
    assert min(audit["d92_tpce_old_tail_gain_by_class"]) > audit[
        "d92_tpce_guard_tolerance"
    ]
    assert audit["d92_tpce_pooled_new_cross_tail_gain"] > audit[
        "d92_tpce_guard_tolerance"
    ]
    assert audit["d92_tpce_old_to_new_hinge_delta"] <= audit[
        "d92_tpce_guard_tolerance"
    ]
    assert audit["d92_tpce_new_to_old_hinge_delta"] <= audit[
        "d92_tpce_guard_tolerance"
    ]
    assert candidate.coef2_qint8.tobytes() != state.coef2_qint8.tobytes()

    reverse = np.arange(len(rows))[::-1]
    reordered, reordered_audit = tpce.apply_d42_tail_pair_code_exchange(
        state, rows[reverse], targets[reverse], old_class_count=6
    )
    assert reordered.coef2_qint8.tobytes() == candidate.coef2_qint8.tobytes()
    assert reordered_audit["d92_tpce_selected_atomic_exchange_count"] == audit[
        "d92_tpce_selected_atomic_exchange_count"
    ]

    permutation = np.asarray([2, 0, 1, 5, 3, 4, 7, 6], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted_state = replace(
        state,
        classes=tuple(state.classes[index] for index in permutation),
        coef1_qint8=state.coef1_qint8[permutation].copy(),
        coef2_qint8=state.coef2_qint8[permutation].copy(),
        scale1_fp16=state.scale1_fp16[permutation].copy(),
        scale2_fp16=state.scale2_fp16[permutation].copy(),
        intercept_fp16=state.intercept_fp16[permutation].copy(),
    )
    old_to_new = np.empty(len(permutation), dtype=np.int64)
    old_to_new[permutation] = np.arange(len(permutation))
    permuted, permuted_audit = tpce.apply_d42_tail_pair_code_exchange(
        permuted_state, rows, old_to_new[targets], old_class_count=6
    )
    assert permuted.coef2_qint8[inverse].tobytes() == candidate.coef2_qint8.tobytes()
    assert permuted_audit["d92_tpce_class_permutation_equivariant"] is True


def test_tpce_fast_greedy_matches_reference_c26_k10_state_and_receipt() -> None:
    """Would fail if the fast scorer changes frozen greedy selection semantics."""

    state = _state(class_count=26)
    rows, targets = _support(class_count=26, k_shot=10)
    base_scores = tpce._score(state, rows)
    old_tails, new_tail, relations, _ = tpce._fixed_tail_and_relations(
        base_scores, targets, old_class_count=6
    )
    atoms = tpce._build_atomic_exchange_candidates(state, rows, relations)
    tolerance = tpce.GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * max(
        1.0, float(np.max(np.abs(base_scores)))
    )

    reference = tpce._pareto_safe_greedy_subset_reference(
        state,
        rows,
        targets,
        6,
        base_scores,
        old_tails,
        new_tail,
        atoms,
        tolerance,
    )
    started = time.perf_counter_ns()
    fast = tpce._pareto_safe_greedy_subset(
        state,
        rows,
        targets,
        6,
        base_scores,
        old_tails,
        new_tail,
        atoms,
        tolerance,
    )
    elapsed_ns = time.perf_counter_ns() - started

    assert fast[0].tobytes() == reference[0].tobytes()
    assert [atom.stable_handle for atom in fast[1]] == [
        atom.stable_handle for atom in reference[1]
    ]
    assert fast[2] == reference[2]
    assert np.array_equal(fast[3], reference[3])
    assert fast[4:8] == reference[4:8]
    assert np.array_equal(fast[8], reference[8])

    candidate, audit = tpce.apply_d42_tail_pair_code_exchange(
        state, rows, targets, old_class_count=6
    )
    expected_codes = (state.coef2_qint8.astype(np.int32) + fast[0]).astype(np.int8)
    assert candidate.coef2_qint8.tobytes() == expected_codes.tobytes()
    assert audit["d92_tpce_selected_atomic_exchange_count"] == len(fast[1])
    assert elapsed_ns > 0


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

    # Force the only available atom to exceed the true-class int8 boundary.
    monkeypatch.setattr(
        tpce,
        "_build_atomic_exchange_candidates",
        lambda *args, **kwargs: [
            tpce._AtomicExchange(
                true_class=0,
                competitor=1,
                coordinate=0,
                direction=1,
                block_index=0,
                stable_handle=("tx_0", "tx_1", 0, 0, 1, "forced"),
            )
        ],
    )
    candidate, audit = tpce.apply_d42_tail_pair_code_exchange(
        forced, rows, targets, old_class_count=6
    )
    assert audit["d92_tpce_active"] is False
    assert audit["d92_tpce_fallback_active"] is True
    assert audit["d92_tpce_fallback_reason"] == "aggregate_saturation"
    assert audit["d92_tpce_aggregate_saturation_count"] > 0
    assert audit["d92_tpce_generated_atomic_exchange_count"] == 1
    assert audit["d92_tpce_selected_atomic_exchange_count"] == 0
    assert audit["d92_tpce_generated_atomic_exchange_count"] == audit[
        "d92_tpce_selected_atomic_exchange_count"
    ] + audit["d92_tpce_rejected_atomic_exchange_count"]
    assert audit["d92_tpce_applied_atomic_exchange_count"] == 0
    assert audit["d92_tpce_changed_code2_count"] == 0
    assert audit["d92_tpce_final_state_sha256"] == audit["d92_tpce_e0_state_sha256"]
    assert candidate.coef2_qint8.tobytes() == forced.coef2_qint8.tobytes()
    inactive = tpce.d42_tpce_inactive_receipt(state)
    assert inactive["d92_tpce_generated_atomic_exchange_count"] == 0
    assert inactive["d92_tpce_selected_atomic_exchange_count"] == 0
    assert inactive["d92_tpce_rejected_atomic_exchange_count"] == 0
    assert inactive["d92_tpce_greedy_step_count"] == 0
