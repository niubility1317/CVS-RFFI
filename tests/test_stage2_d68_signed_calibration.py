from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d68_signed_calibration import (
    D68SignedCalibrationError,
    class_balanced_squared_risk,
    compile_signed_affine,
    leave_one_rank_partitions,
    solve_orientations,
    standardize_affine_rows,
    standardized_scores,
)


def _support(class_count: int = 3, k_shot: int = 5) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(class_count), k_shot)
    rows = np.stack(
        [
            np.array([float(label), float(rank), float(label - rank) / 10.0])
            for label in range(class_count)
            for rank in range(k_shot)
        ]
    )
    return rows, labels


def test_leave_one_rank_partitions_are_exact_once() -> None:
    rows, labels = _support()
    partitions = leave_one_rank_partitions(labels, 3, 5)
    assert len(partitions) == 5
    held = []
    for train, test in partitions:
        assert len(train) == 12
        assert len(test) == 3
        assert all(np.sum(labels[train] == index) == 4 for index in range(3))
        assert all(np.sum(labels[test] == index) == 1 for index in range(3))
        assert len(np.intersect1d(train, test)) == 0
        held.extend(test.tolist())
    assert sorted(held) == list(range(len(rows)))


def test_leave_one_rank_rejects_k1() -> None:
    _, labels = _support(k_shot=1)
    with pytest.raises(D68SignedCalibrationError):
        leave_one_rank_partitions(labels, 3, 1)


def test_orientation_solver_flips_inverted_rows_and_reduces_risk() -> None:
    labels = np.repeat(np.arange(2), 3)
    good = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])
    scores = np.stack([good, good], axis=1)
    scores[:, 1] = -np.roll(good, 3)
    orientation, audit = solve_orientations(scores, labels, 2)
    assert np.array_equal(orientation, np.array([1.0, -1.0]))
    assert audit["crossfit_delta"][0] > 0.0
    assert audit["crossfit_delta"][1] < 0.0
    assert audit["risk_signed"][1] < audit["risk_raw"][1]


def test_standardization_and_signed_compile_preserve_centered_scores() -> None:
    rows, labels = _support()
    coefficient = np.array([[1.0, 0.1, 0.0], [0.0, 0.1, -1.0], [-1.0, 0.1, 0.5]])
    intercept = np.array([0.2, -0.1, 0.3])
    state = standardize_affine_rows(coefficient, intercept, rows, labels, 3)
    orientation = np.array([1.0, -1.0, 1.0])
    compiled_coefficient, compiled_intercept = compile_signed_affine(
        state, orientation
    )
    signed = standardized_scores(rows, state) * orientation[None, :]
    compiled = rows.astype(np.float32) @ compiled_coefficient.T + compiled_intercept
    centered_error = np.max(
        np.abs(
            (signed - signed.mean(axis=1, keepdims=True))
            - (compiled - compiled.mean(axis=1, keepdims=True))
        )
    )
    assert centered_error < 2.0e-6
    sorted_scores = np.sort(signed, axis=1)
    non_tied = (sorted_scores[:, -1] - sorted_scores[:, -2]) > 1.0e-5
    assert np.array_equal(
        np.argmax(signed[non_tied], axis=1),
        np.argmax(compiled[non_tied], axis=1),
    )


def test_orientation_solver_is_class_permutation_equivariant() -> None:
    _, labels = _support()
    rng = np.random.default_rng(68)
    scores = rng.normal(size=(len(labels), 3))
    orientation, _ = solve_orientations(scores, labels, 3)
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    permuted, _ = solve_orientations(
        scores[:, permutation], inverse[labels], 3
    )
    assert np.array_equal(permuted, orientation[permutation])


def test_class_balanced_risk_is_finite() -> None:
    _, labels = _support()
    scores = np.eye(3)[labels] * 2.0 - 1.0
    risk = class_balanced_squared_risk(scores, labels, 3)
    assert np.array_equal(risk, np.zeros(3))
