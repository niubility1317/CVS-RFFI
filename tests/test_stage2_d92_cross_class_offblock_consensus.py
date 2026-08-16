from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_cross_class_offblock_consensus as ccoc
from cvsrffi.stage2_d92_registration_balanced_covariance import (
    build_registration_balanced_statistics,
)


def _blockdiag(covariance: np.ndarray) -> np.ndarray:
    result = np.zeros_like(covariance)
    for block in d42.BLOCK_SLICES:
        result[block, block] = covariance[block, block]
    return result


def _manual_consensus_support() -> tuple[np.ndarray, np.ndarray]:
    """Hand-built K3 registry with known normalized off-block directions."""

    dimension = d42.FEATURE_DIM
    old_direction = np.zeros(dimension, dtype=np.float32)
    old_direction[[0, 160, 256]] = 1.0
    new_directions: list[np.ndarray] = []
    for values in ((1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (0.0, 1.0, 1.0)):
        direction = np.zeros(dimension, dtype=np.float32)
        direction[[0, 160, 256]] = values
        new_directions.append(direction)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    directions = [old_direction] * 6 + new_directions
    for class_index, direction in enumerate(directions):
        center = np.full(dimension, float(class_index * 10), dtype=np.float32)
        # Deliberately avoid canonical order: the implementation must sort
        # each class by its float32 row bytes before mean/scatter arithmetic.
        rows.extend((center + direction, center - direction, center))
        labels.extend((class_index, class_index, class_index))
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def _random_support(classes: int, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes, dtype=np.int64), shots)
    means = rng.normal(size=(classes, d42.FEATURE_DIM))
    rows = means[labels] + 0.2 * rng.normal(size=(classes * shots, d42.FEATURE_DIM))
    return rows.astype(np.float32), labels


def _near_endpoint_consensus_group() -> tuple[np.ndarray, np.ndarray]:
    """Two K3 directions whose cosine is below but very near one."""

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, middle in enumerate((1.0, 1.0 + 2.0e-7)):
        direction = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
        direction[[0, 160, 256]] = (1.0, middle, 1.0)
        rows.extend((-direction, np.zeros_like(direction), direction))
        labels.extend((class_index, class_index, class_index))
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def _float64_boundary_consensus_group() -> tuple[np.ndarray, np.ndarray]:
    """K3 support retains a sub-float32 direction magnitude difference."""

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, first in enumerate((1.0, 1.0 + 2.0e-8)):
        direction = np.zeros(d42.FEATURE_DIM, dtype=np.float64)
        direction[[0, 160, 256]] = (first, 1.0, 1.0)
        # Reverse canonical order to require byte-key sorting before FP64 work.
        rows.extend((direction, -direction, np.zeros_like(direction)))
        labels.extend((class_index, class_index, class_index))
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def _float32_collision_support() -> tuple[np.ndarray, np.ndarray]:
    """11 K3 classes with colliding float32 keys but distinct float64 rows."""

    rng = np.random.default_rng(53)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index in range(11):
        center = np.zeros(d42.FEATURE_DIM, dtype=np.float64)
        center[[0, 160, 256]] = 1.0e12
        center[10] = float(class_index) * 1.0e7
        offsets = rng.normal(size=(3, 3))
        offsets += np.asarray(
            (class_index * 0.01, -class_index * 0.02, class_index * 0.03)
        )
        for offset in offsets:
            row = center.copy()
            row[[0, 160, 256]] += offset
            rows.append(row)
            labels.append(class_index)
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def _float32_class_handle_collision_support() -> tuple[np.ndarray, np.ndarray]:
    """11 K3 classes whose complete float32 sequences collide across classes."""

    rng = np.random.default_rng(991)
    scales = (
        1.0,
        10.0,
        100.0,
        1000.0,
        5000.0,
        10000.0,
        2.0,
        20.0,
        200.0,
        2000.0,
        8000.0,
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, scale in enumerate(scales):
        offsets = rng.normal(size=(3, 3)) * scale
        offsets -= offsets.mean(axis=0, keepdims=True)
        for offset in offsets:
            row = np.zeros(d42.FEATURE_DIM, dtype=np.float64)
            row[[0, 160, 256]] = 1.0e12 + offset
            rows.append(row)
            labels.append(class_index)
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def test_ccoc_pairwise_consensus_has_literal_full_and_block_endpoints():
    """Would fail if CCOC averaged raw off-blocks instead of pairwise cosine."""

    rows, labels = _manual_consensus_support()
    statistics = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )

    assert statistics.old_rho == 1.0
    assert statistics.new_rho == 0.0
    np.testing.assert_array_equal(
        statistics.covariance,
        0.5 * statistics.base.old_covariance
        + 0.5 * _blockdiag(statistics.base.new_covariance),
    )


def test_ccoc_rejects_a_class_with_zero_offblock_direction():
    """Would fail if a zero Q were silently treated as a consensus vote."""

    rows, labels = _manual_consensus_support()
    class_zero = np.flatnonzero(labels == 0)
    rows[class_zero] = 0.0
    rows[class_zero[0], 0] = 1.0
    rows[class_zero[1], 0] = -1.0
    with pytest.raises(ccoc.D92CCOCNumericalError, match="zero"):
        ccoc.build_cross_class_offblock_consensus_statistics(
            d42, rows, labels, class_count=11, k_shot=3
        )


def test_ccoc_near_endpoint_rho_is_not_absorbed_by_a_tolerance():
    """Would fail if a non-endpoint pairwise cosine were snapped to one."""

    rows, labels = _near_endpoint_consensus_group()
    rho, audit = ccoc._stream_group_consensus(rows, labels, range(2), 3)

    assert 1.0 - 64.0 * np.finfo(np.float64).eps < audit["pairwise_cosine_raw"] < 1.0
    assert rho == audit["pairwise_cosine_raw"]
    assert rho < 1.0


def test_ccoc_float32_byte_sorting_keeps_original_float64_reduction_values():
    """Would fail if canonical sorting quantized the rows used for reduction."""

    rows, labels = _float64_boundary_consensus_group()
    expected_order = tuple(
        sorted(
            np.flatnonzero(labels == 0).tolist(),
            key=lambda index: np.ascontiguousarray(
                np.asarray(rows[index], dtype=np.float32)
            ).tobytes(order="C"),
        )
    )
    class_key, actual_order = ccoc._canonical_class_row_order(rows, labels, 0, 3)
    _, audit = ccoc._stream_group_consensus(rows, labels, range(2), 3)

    assert actual_order == expected_order
    assert class_key == tuple(
        (
            np.ascontiguousarray(np.asarray(rows[index], dtype=np.float32)).tobytes(
                order="C"
            ),
            np.ascontiguousarray(np.asarray(rows[index], dtype=np.float64)).tobytes(
                order="C"
            ),
        )
        for index in expected_order
    )
    assert audit["canonicalization"] == "lexicographic_float32_row_bytes_then_float64_reduce"
    assert audit["offblock_norm_max"] > audit["offblock_norm_min"]


def test_ccoc_rejects_nonfinite_q_and_rho_boundaries(monkeypatch):
    """Would fail if nonfinite streaming math reached a covariance mix."""

    rows, labels = _manual_consensus_support()
    rows[0, 0] = np.nan
    with pytest.raises(ccoc.D92CCOCNumericalError, match="nonfinite"):
        ccoc._stream_group_consensus(rows, labels, range(6), 3)

    finite_rows, finite_labels = _manual_consensus_support()
    base = build_registration_balanced_statistics(
        d42, finite_rows, finite_labels, 11, 3
    )
    with pytest.raises(ccoc.D92CCOCNumericalError, match="rho"):
        ccoc._mix_full_and_blockdiag(base.old_covariance, float("nan"))

    non_spd_base = replace(base, old_covariance=np.zeros_like(base.old_covariance))
    monkeypatch.setattr(
        ccoc, "build_registration_balanced_statistics", lambda *_args, **_kwargs: non_spd_base
    )
    with pytest.raises(ccoc.D92CCOCNumericalError, match="positive_definite"):
        ccoc.build_cross_class_offblock_consensus_statistics(
            d42, finite_rows, finite_labels, class_count=11, k_shot=3
        )


def test_ccoc_is_bitwise_invariant_to_support_row_permutation():
    """Would fail if float64 reductions followed the caller's row order."""

    rows, labels = _manual_consensus_support()
    shuffled = np.random.default_rng(221).permutation(len(rows))
    first = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    second = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows[shuffled], labels[shuffled], class_count=11, k_shot=3
    )
    first_coefficient, first_intercept, first_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, first)
    )
    second_coefficient, second_intercept, second_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, second)
    )

    np.testing.assert_array_equal(second.covariance, first.covariance)
    np.testing.assert_array_equal(second_coefficient, first_coefficient)
    np.testing.assert_array_equal(second_intercept, first_intercept)
    assert second.audit == first.audit
    assert second_audit == first_audit


def test_ccoc_float32_key_collisions_are_bitwise_row_permutation_invariant():
    """Would fail if equal float32 keys preserved caller order for FP64 rows."""

    rows, labels = _float32_collision_support()
    assert len({ccoc._float32_row_bytes(row) for row in rows[:3]}) == 1
    shuffled = np.concatenate(
        [np.flatnonzero(labels == class_index)[::-1] for class_index in range(11)]
    )
    first = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    second = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows[shuffled], labels[shuffled], class_count=11, k_shot=3
    )
    first_coefficient, first_intercept, first_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, first)
    )
    second_coefficient, second_intercept, second_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, second)
    )

    np.testing.assert_array_equal(second.covariance, first.covariance)
    np.testing.assert_array_equal(second_coefficient, first_coefficient)
    np.testing.assert_array_equal(second_intercept, first_intercept)
    assert second.audit == first.audit
    assert second_audit == first_audit


def test_ccoc_is_bitwise_equivariant_to_within_group_label_permutation():
    """Would fail if group aggregation depended on arbitrary class ID order."""

    rows, labels = _manual_consensus_support()
    old_new_preserving_permutation = np.asarray(
        [2, 5, 0, 4, 1, 3, 9, 6, 8, 7, 10], dtype=np.int64
    )
    inverse = np.empty_like(old_new_preserving_permutation)
    inverse[old_new_preserving_permutation] = np.arange(len(inverse))
    first = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    second = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, inverse[labels], class_count=11, k_shot=3
    )
    first_coefficient, first_intercept, first_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, first)
    )
    second_coefficient, second_intercept, second_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, second)
    )

    np.testing.assert_array_equal(second.covariance, first.covariance)
    np.testing.assert_array_equal(second_coefficient[inverse], first_coefficient)
    np.testing.assert_array_equal(second_intercept[inverse], first_intercept)
    assert second.audit == first.audit
    assert second_audit == first_audit


def test_ccoc_float32_key_collisions_are_bitwise_label_equivariant():
    """Would fail if class-handle ties used registration class IDs as ordering."""

    rows, labels = _float32_class_handle_collision_support()
    handles = [
        ccoc._canonical_class_row_order(rows, labels, class_index, 3)[0]
        for class_index in range(11)
    ]
    assert all(
        isinstance(item, tuple) and len(item) == 2
        for handle in handles
        for item in handle
    )
    primary_sequences = [
        tuple(primary for primary, _ in handle) for handle in handles
    ]
    secondary_sequences = [
        tuple(secondary for _, secondary in handle) for handle in handles
    ]
    assert all(sequence == primary_sequences[0] for sequence in primary_sequences)
    assert len(set(secondary_sequences)) == 11
    permutation = np.asarray(
        [2, 5, 0, 4, 1, 3, 9, 6, 8, 7, 10], dtype=np.int64
    )
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(inverse))
    first = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    second = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, inverse[labels], class_count=11, k_shot=3
    )
    first_coefficient, first_intercept, first_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, first)
    )
    second_coefficient, second_intercept, second_audit = (
        ccoc.compile_cross_class_offblock_consensus_affine(d42, second)
    )

    np.testing.assert_array_equal(second.covariance, first.covariance)
    np.testing.assert_array_equal(second_coefficient[inverse], first_coefficient)
    np.testing.assert_array_equal(second_intercept[inverse], first_intercept)
    assert second.audit == first.audit
    assert second_audit == first_audit


def test_ccoc_rejects_an_exact_duplicate_canonical_class_handle():
    """Would fail if an identical class-handle tie fell back to a class ID."""

    rows, labels = _manual_consensus_support()
    rows[np.flatnonzero(labels == 1)] = rows[np.flatnonzero(labels == 0)]

    with pytest.raises(ccoc.D92CCOCError, match="identical_class_handle"):
        ccoc._stream_group_consensus(rows, labels, range(6), 3)


def test_ccoc_task_balanced_mix_is_exactly_invariant_to_swapping_tasks():
    """Would fail if equal task weights were implemented asymmetrically."""

    rows, labels = _manual_consensus_support()
    base = build_registration_balanced_statistics(d42, rows, labels, 11, 3)
    old_endpoint = ccoc._mix_full_and_blockdiag(base.old_covariance, 1.0)
    new_endpoint = ccoc._mix_full_and_blockdiag(base.new_covariance, 0.0)
    forward = ccoc._combine_task_covariances(old_endpoint, new_endpoint)
    reverse = ccoc._combine_task_covariances(new_endpoint, old_endpoint)

    np.testing.assert_array_equal(forward, reverse)


def test_ccoc_compile_performs_one_dense_solve_without_extra_fit_families(monkeypatch):
    """Would fail if compile introduced BLOCK, LOO, Fisher, or another solve."""

    rows, labels = _manual_consensus_support()
    statistics = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    solve = np.linalg.solve
    calls = 0

    def counted_solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(ccoc.np.linalg, "solve", counted_solve)
    coefficient, intercept, audit = ccoc.compile_cross_class_offblock_consensus_affine(
        d42, statistics
    )

    assert calls == 1
    assert coefficient.shape == (11, 288)
    assert intercept.shape == (11,)
    assert audit["d92_ccoc_dense_solve_count"] == 1
    assert audit["d92_ccoc_additional_fit_count"] == 0
    assert audit["d92_ccoc_additional_block_fit_count"] == 0
    assert audit["d92_ccoc_additional_loo_fit_count"] == 0
    assert audit["d92_ccoc_additional_fisher_fit_count"] == 0
    assert audit["d92_ccoc_additional_scan_count"] == 0


def test_ccoc_k10_receipt_uses_the_frozen_streaming_bound():
    """Would fail if the K10 resource receipt guessed from ndarray.nbytes."""

    rows, labels = _random_support(11, 10, seed=731)
    statistics = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=10
    )

    assert statistics.audit["d92_ccoc_support_transient_bytes_upper_bound"] == 334336
    assert statistics.audit["d92_ccoc_workspace_upper_accumulators_bytes"] == 188416
    assert statistics.audit["d92_ccoc_workspace_cross_block_buffer_bytes"] == 122880
    assert statistics.audit["d92_ccoc_workspace_residual_buffer_bytes"] == 23040
    assert statistics.audit["d92_ccoc_workspace_numeric_bytes_upper_bound"] == 334336
    assert statistics.audit["d92_ccoc_query_fit_access"] is False
    assert statistics.audit["d92_ccoc_query_update_access"] is False
    assert statistics.audit["d92_ccoc_query_selection_access"] is False
    assert statistics.audit["d92_ccoc_query_truth_access"] is False
    assert statistics.audit["d92_ccoc_query_role_oracle_access"] is False
    assert statistics.audit["d92_ccoc_query_class_quota_access"] is False
    assert statistics.audit["d92_ccoc_query_global_reassignment"] is False


def test_ccoc_active_audit_keeps_the_frozen_no_extra_work_contract():
    """Would fail if the CCOC receipt hid an extra fit, scan, or query path."""

    rows, labels = _manual_consensus_support()
    statistics = ccoc.build_cross_class_offblock_consensus_statistics(
        d42, rows, labels, class_count=11, k_shot=3
    )
    _, _, compiled_audit = ccoc.compile_cross_class_offblock_consensus_affine(
        d42, statistics
    )

    assert statistics.audit["d92_ccoc_formula_revision"] == "pairwise_cosine_v1"
    assert statistics.audit["d92_ccoc_old_group_class_count"] == 6
    assert statistics.audit["d92_ccoc_new_group_class_count"] == 5
    assert statistics.audit["d92_ccoc_covariance_symmetric"] is True
    assert statistics.audit["d92_ccoc_full_endpoint_reused"] is True
    assert statistics.audit["d92_ccoc_additional_fit_count"] == 0
    assert statistics.audit["d92_ccoc_additional_block_fit_count"] == 0
    assert statistics.audit["d92_ccoc_additional_loo_fit_count"] == 0
    assert statistics.audit["d92_ccoc_additional_fisher_fit_count"] == 0
    assert statistics.audit["d92_ccoc_hyperparameter_scan_count"] == 0
    assert statistics.audit["d92_ccoc_weight_scan_count"] == 0
    assert statistics.audit["d92_ccoc_cholesky_pass"] is True
    assert statistics.audit["d92_ccoc_cholesky_check_count"] == 3
    assert statistics.audit["d92_ccoc_persistent_bytes_delta"] == 0
    assert statistics.audit["d92_ccoc_query_bytes_delta"] == 0
    assert statistics.audit["d92_ccoc_query_macs_delta"] == 0
    assert compiled_audit["d92_ccoc_dense_solve_count"] == 1
    assert compiled_audit["d92_ccoc_full_solve_count"] == 1


def test_ccoc_inactive_receipt_rejects_an_active_registration_state():
    """Would fail if an active registered CCOC row silently claimed K1/K2 fallback."""

    with pytest.raises(ccoc.D92CCOCError, match="active_registration"):
        ccoc.ccoc_inactive_receipt(11, 3)


def test_ccoc_inactive_receipt_rejects_old_class_count_override_drift():
    """Would fail if the frozen old-class boundary could be caller-overridden."""

    with pytest.raises(ccoc.D92CCOCError, match="old_class_count"):
        ccoc.ccoc_inactive_receipt(11, 3, old_class_count=11)


@pytest.mark.parametrize(
    ("class_count", "k_shot", "old_class_count"),
    (
        (11, 2.9, 6),
        (6.9, 2, 6),
        (11, 2, 6.9),
        (11, 2, 6.0),
        (True, 2, 6),
        (11, "2", 6),
    ),
)
def test_ccoc_inactive_receipt_rejects_non_exact_count_types(
    class_count, k_shot, old_class_count
):
    """Would fail if int() could truncate or coerce public count arguments."""

    with pytest.raises(ccoc.D92CCOCError, match="exact_integer"):
        ccoc.ccoc_inactive_receipt(
            class_count, k_shot, old_class_count=old_class_count
        )


def test_ccoc_inactive_receipt_accepts_exact_numpy_integer_counts():
    """The public receipt accepts exact NumPy integer counts without coercion."""

    receipt = ccoc.ccoc_inactive_receipt(
        np.int64(11), np.int64(2), old_class_count=np.int64(6)
    )

    assert receipt["d92_ccoc_status"] == "k1_k2_exact_d81_fallback"


def test_ccoc_inactive_receipt_keeps_only_frozen_pre_and_low_k_reasons():
    """Would fail if inactive receipts drifted from the D81 lifecycle reasons."""

    assert ccoc.ccoc_inactive_receipt(6, 5)["d92_ccoc_status"] == "before_exact_d81"
    assert (
        ccoc.ccoc_inactive_receipt(11, 1)["d92_ccoc_status"]
        == "k1_k2_exact_d81_fallback"
    )
    assert (
        ccoc.ccoc_inactive_receipt(11, 2)["d92_ccoc_status"]
        == "k1_k2_exact_d81_fallback"
    )
