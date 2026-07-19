from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi.stage2_d75_crossfitted_margin_safe_projection import (
    D75MarginSafetyError,
    fit_crossfitted_margin_safe_projection,
)


def _rows() -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(
        [[2.0, -0.2], [2.2, 0.0], [1.8, 0.2], [-2.0, -0.2], [-2.2, 0.0], [-1.8, 0.2]],
        dtype=np.float32,
    )
    return rows, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)


def _direction_fit(axis: int):
    def fit(rows, labels, class_count, k_shot):
        del labels
        direction = np.zeros(rows.shape[1], dtype=np.float32)
        direction[axis] = 1.0
        projected = rows - np.outer(rows @ direction, direction)
        before = float(np.sum(np.asarray(rows, dtype=np.float64) ** 2))
        after = float(np.sum(np.asarray(projected, dtype=np.float64) ** 2))
        return direction, projected, {
            "schema": "stub",
            "status": "rank1_orthogonal_nuisance_removal_active",
            "class_count": class_count,
            "k_shot": k_shot,
            "dimension": rows.shape[1],
            "projection_active": True,
            "projection_removed_rank": 1,
            "projection_rank": rows.shape[1] - 1,
            "direction_l2": 1.0,
            "direction_sha256": hashlib.sha256(direction.tobytes()).hexdigest(),
            "centroid_direction_max_abs": 0.0,
            "centroid_pairwise_squared_distance_drift_max": 0.0,
            "removed_residual_energy_fraction": 0.5,
            "within_residual_energy_before": before,
            "within_residual_energy_after": after,
            "within_residual_energy_removed_fraction": ((before - after) / before),
            "projector_symmetry_max_abs_error": 0.0,
            "projector_idempotence_max_abs_error": 0.0,
            "projector_annihilation_l2": 0.0,
            "ground_component_input_count": 0,
            "query_rows_used": 0,
        }

    return fit


def _fixed_lda(rows, labels, class_count, k_shot):
    del rows, labels, k_shot
    assert class_count == 2
    return (
        np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        {},
    )


def test_rejects_projection_that_destroys_held_margin() -> None:
    rows, labels = _rows()
    direction, projected, audit = fit_crossfitted_margin_safe_projection(
        rows,
        labels,
        2,
        3,
        direction_fit=_direction_fit(0),
        lda_fit=_fixed_lda,
    )
    assert audit["crossfit_gate_pass"] is False
    assert audit["status"] == "crossfitted_margin_rejected_exact_d62_fallback"
    assert audit["crossfit_margin_delta_min_class_mean"] < 0.0
    assert audit["projection_removed_rank"] == 0
    assert np.array_equal(direction, np.zeros(2, dtype=np.float32))
    assert np.array_equal(projected, rows)


def test_accepts_projection_that_is_margin_neutral() -> None:
    rows, labels = _rows()
    direction, projected, audit = fit_crossfitted_margin_safe_projection(
        rows,
        labels,
        2,
        3,
        direction_fit=_direction_fit(1),
        lda_fit=_fixed_lda,
    )
    assert audit["crossfit_gate_pass"] is True
    assert audit["status"] == "crossfitted_margin_safe_projection_active"
    assert audit["crossfit_margin_delta_min_class_mean"] == pytest.approx(0.0)
    assert audit["crossfit_correct_delta"] == 0
    assert audit["crossfit_fold_count"] == 3
    assert np.array_equal(direction, np.asarray([0.0, 1.0], dtype=np.float32))
    assert not np.array_equal(projected, rows)


def test_class_interleaving_preserves_physical_rank_holdouts() -> None:
    rows, labels = _rows()
    order = np.asarray([0, 3, 1, 4, 2, 5])
    _, _, audit = fit_crossfitted_margin_safe_projection(
        rows[order],
        labels[order],
        2,
        3,
        direction_fit=_direction_fit(1),
        lda_fit=_fixed_lda,
    )
    assert audit["crossfit_held_row_count"] == 6
    assert audit["crossfit_per_class_margin_delta_mean"] == pytest.approx([0.0, 0.0])


def test_rejects_asymmetric_support() -> None:
    rows, labels = _rows()
    with pytest.raises(D75MarginSafetyError, match="exact symmetric support"):
        fit_crossfitted_margin_safe_projection(
            rows[:-1],
            labels[:-1],
            2,
            3,
            direction_fit=_direction_fit(1),
            lda_fit=_fixed_lda,
        )
