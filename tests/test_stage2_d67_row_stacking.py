from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d67_row_stacking import (
    D67RowStackingError,
    compile_stacked_affine,
    four_rank_partitions,
    solve_class_balanced_convex_weights,
    standardize_affine_rows,
    standardized_scores,
)


def _support(class_count: int = 3, k_shot: int = 8) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(class_count), k_shot)
    rows = np.stack(
        [
            np.array([float(label), float(rank), float(label - rank) / 10.0])
            for label in range(class_count)
            for rank in range(k_shot)
        ]
    )
    return rows, labels


def test_four_rank_partitions_are_exact_once_and_class_balanced() -> None:
    rows, labels = _support()
    partitions = four_rank_partitions(labels, 3, 8)
    assert len(partitions) == 4
    held = []
    for train, test in partitions:
        assert len(train) == 18
        assert len(test) == 6
        assert all(np.sum(labels[train] == index) == 6 for index in range(3))
        assert all(np.sum(labels[test] == index) == 2 for index in range(3))
        held.extend(test.tolist())
    assert sorted(held) == list(range(len(rows)))


def test_four_rank_partitions_reject_small_or_nondivisible_k() -> None:
    _, labels = _support(k_shot=8)
    with pytest.raises(D67RowStackingError):
        four_rank_partitions(labels[:18], 3, 6)


def test_standardization_compiles_exact_scores() -> None:
    rows, labels = _support()
    coefficient = np.array([[1.0, 0.1, 0.0], [0.0, 0.1, 1.0], [-1.0, 0.1, 0.5]])
    intercept = np.array([0.2, -0.1, 0.3])
    state = standardize_affine_rows(coefficient, intercept, rows, labels, 3)
    original = rows @ coefficient.T + intercept[None, :]
    center = 0.5 * (state.positive_mean + state.negative_mean)
    expected = (original - center[None, :]) / state.scale[None, :]
    assert np.allclose(standardized_scores(rows, state), expected, atol=1.0e-12)
    assert np.all(state.scale > 0.0)


def test_closed_form_convex_weight_hits_known_solution() -> None:
    labels = np.array([0, 0, 1, 1])
    target0 = np.array([1.0, 1.0, -1.0, -1.0])
    target1 = -target0
    target = np.stack([target0, target1], axis=1)
    d62 = target - 0.5
    d65 = target + 0.5
    alpha, audit = solve_class_balanced_convex_weights(d62, d65, labels, 2)
    assert np.allclose(alpha, 0.5, atol=1.0e-12)
    assert np.all(audit["risk_stacked"] <= audit["risk_d62"] + 1.0e-12)


def test_weight_solver_is_class_permutation_equivariant() -> None:
    rows, labels = _support()
    rng = np.random.default_rng(19)
    first = rng.normal(size=(len(rows), 3))
    second = rng.normal(size=(len(rows), 3))
    alpha, _ = solve_class_balanced_convex_weights(first, second, labels, 3)
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    permuted_labels = inverse[labels]
    permuted_alpha, _ = solve_class_balanced_convex_weights(
        first[:, permutation], second[:, permutation], permuted_labels, 3
    )
    assert np.allclose(permuted_alpha, alpha[permutation], atol=1.0e-12)


def test_compile_stacked_affine_preserves_argmax_and_float32_equivalence() -> None:
    rows, labels = _support()
    first = standardize_affine_rows(
        np.eye(3), np.zeros(3), rows, labels, 3
    )
    second = standardize_affine_rows(
        np.array([[0.8, 0.1, 0.0], [0.0, 0.2, 0.9], [-0.7, 0.1, 0.4]]),
        np.array([0.2, -0.3, 0.1]),
        rows,
        labels,
        3,
    )
    alpha = np.array([0.0, 0.5, 1.0])
    coefficient, intercept, error = compile_stacked_affine(first, second, alpha)
    normalized = (
        (1.0 - alpha[None, :]) * standardized_scores(rows, first)
        + alpha[None, :] * standardized_scores(rows, second)
    )
    center = 0.5 * (first.positive_mean + first.negative_mean)
    raw = normalized * first.scale[None, :] + center[None, :]
    compiled = rows.astype(np.float32) @ coefficient.T + intercept[None, :]
    assert np.array_equal(np.argmax(raw, axis=1), np.argmax(compiled, axis=1))
    assert np.max(np.abs((raw - raw.mean(axis=1, keepdims=True)) - compiled)) < 1.0e-5
    assert error < 1.0e-6
