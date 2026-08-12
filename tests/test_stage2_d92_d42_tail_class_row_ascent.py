from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_d42_tail_class_row_ascent as tcra


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
            row[class_index] = np.float32(1.0 + 0.01 * sample_index)
            row[160 + class_index] = np.float32(0.5 + 0.01 * sample_index)
            row[256 + class_index] = np.float32(0.25 + 0.01 * sample_index)
            rows.append(row)
            targets.append(class_index)
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


def _interference_state_and_support() -> tuple[
    d42.D42UnifiedShrinkageLDAState, np.ndarray, np.ndarray
]:
    """The fixed TPCE interference case must remain active under TCRA."""

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


def _permute_state(
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


def test_tcra_is_active_on_the_fixed_pair_interference_fixture() -> None:
    state, rows, targets = _interference_state_and_support()

    candidate, audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    tolerance = audit["d92_tcra_guard_tolerance"]
    assert audit["d92_tcra_active"] is True
    assert audit["d92_tcra_fallback_active"] is False
    assert audit["d92_tcra_state_postprocess_mode"] == "d42_tcra"
    assert audit["d92_tcra_generated_atomic_ascent_count"] <= 8 * 3
    assert audit["d92_tcra_generated_atomic_ascent_count"] == audit[
        "d92_tcra_selected_atomic_ascent_count"
    ] + audit["d92_tcra_rejected_atomic_ascent_count"]
    assert audit["d92_tcra_changed_code2_count"] == audit[
        "d92_tcra_selected_atomic_ascent_count"
    ]
    assert min(audit["d92_tcra_old_tail_gain_by_class"]) > tolerance
    assert audit["d92_tcra_pooled_new_cross_tail_gain"] > tolerance
    assert audit["d92_tcra_pooled_new_allclass_tail_gain"] >= -tolerance
    assert audit["d92_tcra_old_to_new_hinge_delta"] <= tolerance
    assert audit["d92_tcra_new_to_old_hinge_delta"] <= tolerance
    assert audit["d92_tcra_final_gate_revision"] == "safe_directional_v2"
    assert audit["d92_tcra_old_tail_gain_sum"] == float(
        np.sum(
            np.asarray(audit["d92_tcra_old_tail_gain_by_class"]),
            dtype=np.float64,
        )
    )
    assert audit["d92_tcra_old_tail_strict_positive_count"] == sum(
        value > tolerance
        for value in audit["d92_tcra_old_tail_gain_by_class"]
    )
    assert audit["d92_tcra_safe_directional_pass"] is True
    assert audit["d92_tcra_modified_state_field_names"] == ["coef2_qint8"]
    difference = candidate.coef2_qint8.astype(np.int16) - state.coef2_qint8
    assert np.all(np.abs(difference[difference != 0]) == 1)
    assert np.all(np.count_nonzero(difference, axis=1) <= 3)
    for name in (
        "coef1_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "log_diag_fp32",
    ):
        assert getattr(candidate, name).tobytes() == getattr(state, name).tobytes()


def test_tcra_safe_directional_v2_activates_the_real_clear_g0_values() -> None:
    """The v1 all-strict gate rejected clear's within-tolerance new tail."""

    tolerance = 0.0008059885585680604
    old_gains = np.asarray(
        [
            -0.0006160736083984375,
            -0.00021839141845703125,
            0.005859375,
            -0.0000858306884765625,
            0.00125885009765625,
            0.006592750549316406,
        ],
        dtype=np.float64,
    )
    new_cross_gain = -0.00023097991943359374
    gains = np.concatenate([old_gains, [new_cross_gain]])

    assert new_cross_gain < 0.0
    assert new_cross_gain >= -tolerance
    assert not np.all(gains > tolerance)  # frozen v1 strict gate
    assert tcra._final_guard_pass(gains, new_cross_gain, 0.0, 0.0, tolerance)


@pytest.mark.parametrize(
    ("gains", "all_gain", "old_to_new", "new_to_old"),
    [
        ([-0.1000001, 0.2, 0.05, 0.05, 0.05, 0.05, 0.0], 0.0, 0.0, 0.0),
        ([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, -0.1000001], 0.0, 0.0, 0.0),
        ([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0], -0.1000001, 0.0, 0.0),
        ([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0], 0.0, 0.1000001, 0.0),
        ([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0], 0.0, 0.0, 0.1000001),
        ([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0], 0.0, 0.0, 0.0),
        ([0.2, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0),
    ],
    ids=(
        "old_below_negative_tolerance",
        "new_cross_below_negative_tolerance",
        "new_all_below_negative_tolerance",
        "old_to_new_hinge_above_tolerance",
        "new_to_old_hinge_above_tolerance",
        "no_old_strictly_positive",
        "old_gain_sum_not_strictly_positive",
    ),
)
def test_tcra_safe_directional_v2_fails_closed_when_any_condition_breaks(
    gains, all_gain, old_to_new, new_to_old
) -> None:
    assert not tcra._final_guard_pass(
        np.asarray(gains, dtype=np.float64),
        all_gain,
        old_to_new,
        new_to_old,
        0.1,
    )


def test_tcra_fast_batch_is_byte_exact_to_the_frozen_reference_c26_k10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(class_count=26)
    rows, targets = _support(class_count=26, k_shot=10)
    fast_state, fast_audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    def reference(
        state_arg,
        rows_arg,
        targets_arg,
        old_count_arg,
        base_scores,
        current_scores,
        old_tails,
        new_tail,
        atoms,
    ):
        values = []
        for atom in atoms:
            candidate_scores = current_scores.copy()
            candidate_scores[:, atom.true_class] += tcra._atom_score_delta(
                state_arg, rows_arg, atom
            )
            values.append(
                tcra._support_group_values(
                    rows_arg,
                    base_scores,
                    candidate_scores,
                    targets_arg,
                    old_count_arg,
                    old_tails,
                    new_tail,
                )
            )
        return (
            np.stack([value[0] for value in values]),
            np.asarray([value[1] for value in values], dtype=np.float64),
            np.asarray([value[2] for value in values], dtype=np.float64),
            np.asarray([value[3] for value in values], dtype=np.float64),
        )

    monkeypatch.setattr(tcra, "_analytic_candidate_group_values_batch", reference)
    reference_state, reference_audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    assert fast_state.coef2_qint8.tobytes() == reference_state.coef2_qint8.tobytes()
    assert fast_audit == reference_audit


def test_tcra_canonical_float64_sum_and_row_permutation_are_byte_exact() -> None:
    state = _state()
    rows, targets = _support()
    order = np.asarray([3, 0, 4, 1, 2], dtype=np.int64)
    class_rows = np.flatnonzero(targets == 0)
    aggregate_a = tcra._canonical_float64_sum(rows, class_rows)
    aggregate_b = tcra._canonical_float64_sum(rows, class_rows[order])
    assert aggregate_a.dtype == np.float64
    assert aggregate_a.tobytes() == aggregate_b.tobytes()

    candidate_a, audit_a = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )
    reverse = np.arange(len(rows))[::-1]
    candidate_b, audit_b = tcra.apply_d42_tail_class_row_ascent(
        state, rows[reverse], targets[reverse], old_class_count=6
    )
    assert tcra.d42_tcra_state_sha256(candidate_a) == tcra.d42_tcra_state_sha256(
        candidate_b
    )
    assert candidate_a.coef2_qint8.tobytes() == candidate_b.coef2_qint8.tobytes()
    assert audit_a == audit_b


def test_tcra_is_equivariant_to_old_and_new_group_label_permutations() -> None:
    state = _state()
    rows, targets = _support()
    candidate, audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    permutation = np.asarray([2, 0, 1, 5, 3, 4, 7, 6], dtype=np.int64)
    inverse = np.argsort(permutation)
    old_to_new = np.empty(len(permutation), dtype=np.int64)
    old_to_new[permutation] = np.arange(len(permutation))
    permuted_state = _permute_state(state, permutation)
    permuted, permuted_audit = tcra.apply_d42_tail_class_row_ascent(
        permuted_state, rows, old_to_new[targets], old_class_count=6
    )

    assert permuted.coef2_qint8[inverse].tobytes() == candidate.coef2_qint8.tobytes()
    assert np.asarray(permuted_audit["d92_tcra_old_tail_gain_by_class"])[
        inverse[:6]
    ].tobytes() == np.asarray(
        audit["d92_tcra_old_tail_gain_by_class"]
    ).tobytes()
    assert permuted_audit["d92_tcra_class_permutation_equivariant"] is True


def test_tcra_rejects_an_unsafe_prefix_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    rows, targets = _support()
    original = tcra._prefix_guard_pass
    calls = {"count": 0}

    def reject_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return False
        return original(*args, **kwargs)

    monkeypatch.setattr(tcra, "_prefix_guard_pass", reject_first)
    candidate, audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    assert audit["d92_tcra_active"] is True
    assert audit["d92_tcra_prefix_guard_rejected_count"] == 1
    assert audit["d92_tcra_greedy_step_count"] > audit[
        "d92_tcra_selected_atomic_ascent_count"
    ]
    assert candidate.coef2_qint8.tobytes() != state.coef2_qint8.tobytes()


def test_tcra_all_prefixes_fail_with_byte_exact_e0_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    rows, targets = _support()
    monkeypatch.setattr(tcra, "_prefix_guard_pass", lambda *args, **kwargs: False)

    candidate, audit = tcra.apply_d42_tail_class_row_ascent(
        state, rows, targets, old_class_count=6
    )

    assert candidate is state
    assert audit["d92_tcra_active"] is False
    assert audit["d92_tcra_fallback_active"] is True
    assert audit["d92_tcra_fallback_reason"] == "no_pareto_safe_subset"
    assert audit["d92_tcra_selected_atomic_ascent_count"] == 0
    assert audit["d92_tcra_prefix_guard_rejected_count"] == audit[
        "d92_tcra_generated_atomic_ascent_count"
    ]
    assert audit["d92_tcra_final_state_sha256"] == audit[
        "d92_tcra_e0_state_sha256"
    ]
    assert audit["d92_tcra_modified_state_field_names"] == []
    assert audit["d92_tcra_final_gate_revision"] == "safe_directional_v2"
    assert audit["d92_tcra_old_tail_gain_sum"] == 0.0
    assert audit["d92_tcra_old_tail_strict_positive_count"] == 0
    assert audit["d92_tcra_safe_directional_pass"] is False


def test_tcra_int8_boundary_falls_back_exactly_to_e0() -> None:
    state = _state()
    rows, targets = _support()
    boundary = replace(
        state,
        coef2_qint8=np.full_like(state.coef2_qint8, 127, dtype=np.int8),
    )

    candidate, audit = tcra.apply_d42_tail_class_row_ascent(
        boundary, rows, targets, old_class_count=6
    )

    assert candidate is boundary
    assert audit["d92_tcra_fallback_reason"] == "aggregate_saturation"
    assert audit["d92_tcra_aggregate_saturation_count"] > 0
    assert audit["d92_tcra_selected_atomic_ascent_count"] == 0
    assert audit["d92_tcra_generated_atomic_ascent_count"] == audit[
        "d92_tcra_rejected_atomic_ascent_count"
    ]
    assert audit["d92_tcra_final_state_sha256"] == audit[
        "d92_tcra_e0_state_sha256"
    ]


@pytest.mark.parametrize("k_shot", [1, 2])
def test_tcra_k1_k2_inactive_receipt_is_an_exact_full_alias(k_shot: int) -> None:
    state = _state()
    receipt = tcra.d42_tcra_inactive_receipt(state)

    assert k_shot <= 2
    assert receipt["d92_tcra_active"] is False
    assert receipt["d92_tcra_fallback_active"] is False
    assert receipt["d92_tcra_fallback_reason"] == "K1_K2_EXACT_D92_FULL_ALIAS"
    assert receipt["d92_tcra_e0_state_sha256"] == receipt[
        "d92_tcra_final_state_sha256"
    ]
    assert receipt["d92_tcra_component_fit_count"] == 0
    assert receipt["d92_tcra_generated_atomic_ascent_count"] == 0
    assert receipt["d92_tcra_selected_atomic_ascent_count"] == 0
    assert receipt["d92_tcra_rejected_atomic_ascent_count"] == 0
    assert receipt["d92_tcra_prefix_guard_rejected_count"] == 0
    assert receipt["d92_tcra_modified_state_field_names"] == []
    assert receipt["d92_tcra_final_gate_revision"] == "safe_directional_v2"
    assert receipt["d92_tcra_old_tail_gain_sum"] is None
    assert receipt["d92_tcra_old_tail_strict_positive_count"] is None
    assert receipt["d92_tcra_safe_directional_pass"] is False
    assert all(
        receipt[f"d92_tcra_query_{name}"] is False
        for name in (
            "fit_access",
            "update_access",
            "selection_access",
            "truth_access",
            "role_oracle_access",
            "class_quota_access",
            "global_reassignment",
        )
    )
