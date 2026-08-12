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
