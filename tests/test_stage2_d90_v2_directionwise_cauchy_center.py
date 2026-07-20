from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d90_v2_directionwise_cauchy_center import (
    TRANSLATION_EXTRA_MACS_PER_ROW_RANK,
    translate_to_directionwise_cauchy_centers,
)


def _basis() -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(160, 3, dtype=np.float64)
    weights = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)
    return basis, weights


def test_directionwise_center_preserves_residuals_fft_and_class_symmetry() -> None:
    assert TRANSLATION_EXTRA_MACS_PER_ROW_RANK == 7
    rng = np.random.default_rng(9001)
    rows = rng.normal(size=(24, 288))
    rows[0, 0] += 8.0
    labels = np.repeat(np.arange(3), 8)
    basis, weights = _basis()
    transformed, audit = translate_to_directionwise_cauchy_centers(
        rows, labels, 3, 8, basis, weights
    )
    assert audit["directionwise_subspace_center_replaced"] is True
    assert audit["d81_orthogonal_center_preserved"] is True
    assert audit["old_new_role_specific_branch"] is False
    assert audit["within_class_residual_max_abs_error"] <= 2.0e-12
    np.testing.assert_array_equal(transformed[:, 160:], rows[:, 160:])
    assert not np.array_equal(transformed[:, :160], rows[:, :160])


def test_k2_is_bitwise_identity_and_row_permutation_equivariant() -> None:
    rng = np.random.default_rng(9002)
    basis, weights = _basis()
    rows2 = rng.normal(size=(6, 288))
    labels2 = np.repeat(np.arange(3), 2)
    transformed2, audit2 = translate_to_directionwise_cauchy_centers(
        rows2, labels2, 3, 2, basis, weights
    )
    np.testing.assert_array_equal(transformed2, rows2)
    assert audit2["k1_k2_exact_identity"] is True

    rows = rng.normal(size=(24, 288))
    labels = np.repeat(np.arange(3), 8)
    transformed, _ = translate_to_directionwise_cauchy_centers(
        rows, labels, 3, 8, basis, weights
    )
    order = rng.permutation(len(rows))
    inverse = np.argsort(order)
    permuted, _ = translate_to_directionwise_cauchy_centers(
        rows[order], labels[order], 3, 8, basis, weights
    )
    np.testing.assert_allclose(permuted[inverse], transformed, atol=2.0e-12)
