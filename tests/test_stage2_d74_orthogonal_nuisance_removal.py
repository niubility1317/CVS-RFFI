from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d74_orthogonal_nuisance_removal import (
    D74ProjectionError,
    fit_orthogonal_nuisance_direction,
)


def _support(k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(74)
    centers = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    nuisance = np.asarray([0.0, 0.0, 0.0, 1.0, -0.5, 0.25])
    nuisance /= np.linalg.norm(nuisance)
    rows = []
    labels = []
    for class_index, center in enumerate(centers):
        for rank in range(k):
            amplitude = (rank - (k - 1) / 2.0) * 0.4
            rows.append(
                center
                + amplitude * nuisance
                + 0.01 * rng.normal(size=center.shape)
            )
            labels.append(class_index)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def test_rank1_projection_is_active_noninvertible_and_centroid_safe() -> None:
    rows, labels = _support()
    direction, projected, audit = fit_orthogonal_nuisance_direction(
        rows, labels, 4, 5
    )
    assert direction.flags.writeable is False
    assert projected.flags.writeable is False
    assert audit["status"] == "rank1_orthogonal_nuisance_removal_active"
    assert audit["projection_rank"] == rows.shape[1] - 1
    assert audit["projection_removed_rank"] == 1
    assert audit["direction_l2"] == pytest.approx(1.0)
    assert audit["centroid_direction_max_abs"] < 1e-9
    assert audit["centroid_pairwise_squared_distance_drift_max"] < 1e-8
    assert audit["projector_idempotence_max_abs_error"] < 1e-10
    assert audit["within_residual_energy_after"] < audit[
        "within_residual_energy_before"
    ]


def test_class_permutation_preserves_projector_and_projection() -> None:
    rows, labels = _support()
    direction, projected, _ = fit_orthogonal_nuisance_direction(
        rows, labels, 4, 5
    )
    mapping = np.asarray([2, 0, 3, 1])
    other_direction, other_projected, _ = fit_orthogonal_nuisance_direction(
        rows, mapping[labels], 4, 5
    )
    assert np.allclose(
        np.outer(direction, direction),
        np.outer(other_direction, other_direction),
        atol=1e-7,
    )
    assert np.allclose(projected, other_projected, atol=1e-7)


def test_k1_is_exact_fallback() -> None:
    rows, labels = _support(k=1)
    direction, projected, audit = fit_orthogonal_nuisance_direction(
        rows, labels, 4, 1
    )
    assert np.count_nonzero(direction) == 0
    assert np.array_equal(projected, rows)
    assert audit["status"] == "k1_exact_d62_fallback"
    assert audit["projection_active"] is False


def test_invalid_asymmetric_support_fails_closed() -> None:
    rows, labels = _support()
    with pytest.raises(D74ProjectionError, match="exact symmetric"):
        fit_orthogonal_nuisance_direction(rows[:-1], labels[:-1], 4, 5)
