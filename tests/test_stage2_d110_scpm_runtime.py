"""Narrow D110-SCPM runtime tests.

Design traceability (D110 theory §4.3, §4.4, §9.1):

| Requirement | Runtime acceptance check |
| --- | --- |
| K=1 uses only sealed prior variances | K=1 state retains ``alpha=1`` and prior values |
| K>1 uses class-block Ledoit-Wolf-style shrinkage | 160D/rank3 pooled t, alpha, and final variance match the frozen equation |
| Relative variance condition cap is 20 | the smallest safe relative variance is 1/20 |
| Zero group variance falls back to Euclidean distance | all-zero variances produce exact squared Euclidean scores |
| Query is per-sample, all-class, and read-only | a query returns every class score without changing state |
| K=1 prior is genuinely anisotropic | SCPM prediction differs from the Euclidean nearest center on a fixed example |
"""

from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d110_scpm_runtime import (
    D110SCPMRuntimeError,
    MIN_RELATIVE_VARIANCE,
    fit_d110_scpm_runtime,
    predict_d110_scpm_query,
    score_d110_scpm_query,
)


def _basis() -> np.ndarray:
    return np.eye(3, 160, dtype=np.float64)


def _vector(*coordinates: tuple[int, float]) -> np.ndarray:
    row = np.zeros(160, dtype=np.float64)
    for index, value in coordinates:
        row[index] = value
    return row


def _three_class_k1_support() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack((_vector((0, 1.0)), _vector((1, 1.0)), _vector((2, 1.0)))),
        np.asarray(["old", "new", "other"], dtype="<U8"),
    )


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_k1_uses_only_prior_and_scores_explicit_predictive_mahalanobis():
    support, labels = _three_class_k1_support()
    prior = np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    state = fit_d110_scpm_runtime(support, labels, _basis(), prior)

    assert state.active_k == 1
    assert state.alpha == 1.0
    assert state.target_variances is None
    assert state.euclidean_fallback is False
    np.testing.assert_array_equal(state.variances, prior)

    query = _vector((0, 2.0), (1, 1.0))
    normalized_query = query / np.linalg.norm(query)
    delta = normalized_query[None, :] - state.centers
    projected = delta @ state.closed_u.T
    expected = (
        np.sum(np.square(projected) / state.predictive_variances[None, :-1], axis=1)
        + (
            np.sum(np.square(delta), axis=1) - np.sum(np.square(projected), axis=1)
        )
        / state.predictive_variances[-1]
    )
    scores = score_d110_scpm_query(state, query)
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1.0e-12)
    assert predict_d110_scpm_query(state, query) == "old"


def test_k_greater_than_one_matches_frozen_class_block_lw_equation():
    offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    group_a = np.zeros((5, 160), dtype=np.float64)
    group_b = np.zeros((5, 160), dtype=np.float64)
    group_a[:, 0] = 4.0 + 0.40 * offsets
    group_a[:, 1] = 0.20 * offsets
    group_a[:, 2] = 0.10 * offsets
    group_a[:, 3] = 0.05 * offsets
    group_a[:, 159] = 0.07 * offsets
    group_b[:, 0] = 0.15 * offsets
    group_b[:, 1] = 4.0 + 0.10 * offsets
    group_b[:, 2] = 0.30 * offsets
    group_b[:, 3] = 0.08 * offsets
    group_b[:, 159] = 0.04 * offsets
    support = np.concatenate((group_a, group_b), axis=0)
    labels = np.asarray([7] * 5 + [3] * 5, dtype=np.int64)
    prior = np.asarray([5.0, 4.0, 3.0, 2.0], dtype=np.float64)
    state = fit_d110_scpm_runtime(support, labels, _basis(), prior)

    normalized = _normalize(support)
    grouped = np.stack((normalized[labels == 3], normalized[labels == 7]), axis=0)
    residual = grouped - grouped.mean(axis=1, keepdims=True)
    projected = residual @ _basis().T
    t_parallel = np.sum(np.square(projected), axis=1) / 4.0
    perpendicular_dimensions = 160 - _basis().shape[0]
    assert perpendicular_dimensions == 157
    t_perp = (
        np.sum(np.square(residual), axis=(1, 2))
        - np.sum(np.square(projected), axis=(1, 2))
    ) / float(4 * perpendicular_dimensions)
    class_t = np.concatenate((t_parallel, t_perp[:, None]), axis=1)
    target = class_t.mean(axis=0)
    variation = np.sum(np.square(class_t - target[None, :]), axis=0) / 2.0
    group_dimensions = np.asarray([1.0, 1.0, 1.0, 157.0], dtype=np.float64)
    expected_alpha = np.clip(
        np.dot(group_dimensions, variation)
        / np.dot(group_dimensions, np.square(target - prior)),
        0.0,
        1.0,
    )
    expected_variances = expected_alpha * prior + (1.0 - expected_alpha) * target

    assert state.active_k == 5
    assert 0.0 <= state.alpha <= 1.0
    np.testing.assert_allclose(state.target_variances, target, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(state.alpha, expected_alpha, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(state.variances, expected_variances, rtol=0.0, atol=1.0e-12)


def test_all_zero_variances_use_exact_squared_euclidean_fallback():
    support, labels = _three_class_k1_support()
    state = fit_d110_scpm_runtime(
        support, labels, _basis(), np.zeros(4, dtype=np.float64)
    )
    query = _vector((0, 1.0), (1, 1.0))
    normalized_query = query / np.linalg.norm(query)
    expected = np.sum(np.square(normalized_query[None, :] - state.centers), axis=1)

    assert state.euclidean_fallback is True
    np.testing.assert_allclose(
        score_d110_scpm_query(state, query), expected, rtol=0.0, atol=1.0e-12
    )


def test_relative_condition_cap_is_fixed_at_twenty():
    support, labels = _three_class_k1_support()
    state = fit_d110_scpm_runtime(
        support,
        labels,
        _basis(),
        np.asarray([100.0, 1.0, 100.0, 100.0], dtype=np.float64),
    )

    np.testing.assert_allclose(
        state.safe_relative_variances,
        np.asarray([1.0, MIN_RELATIVE_VARIANCE, 1.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(state.safe_relative_variances) / np.min(
        state.safe_relative_variances
    ) == 20.0


def test_k1_anisotropic_prior_changes_the_euclidean_nearest_center_prediction():
    support = np.stack((_vector((0, 1.0)), _vector((1, 1.0))))
    labels = np.asarray(["u1", "u2"], dtype="<U2")
    state = fit_d110_scpm_runtime(
        support,
        labels,
        _basis(),
        np.asarray([1.0, 100.0, 100.0, 100.0], dtype=np.float64),
    )
    query = _vector((0, 0.6), (1, 0.8))
    normalized_query = query / np.linalg.norm(query)
    euclidean = np.sum(
        np.square(normalized_query[None, :] - state.centers), axis=1
    )

    assert state.class_labels[int(np.argmin(euclidean))] == "u2"
    assert predict_d110_scpm_query(state, query) == "u1"
    assert int(np.argmin(score_d110_scpm_query(state, query))) == 0


def test_runtime_rejects_k_outside_the_frozen_matrix():
    support, labels = _three_class_k1_support()
    with pytest.raises(D110SCPMRuntimeError, match="supports only frozen K"):
        fit_d110_scpm_runtime(
            np.repeat(support, 2, axis=0),
            np.repeat(labels, 2),
            _basis(),
            np.ones(4, dtype=np.float64),
        )


def test_query_is_single_sample_all_class_and_read_only():
    support, labels = _three_class_k1_support()
    state = fit_d110_scpm_runtime(
        support, labels, _basis(), np.asarray([1.0, 2.0, 3.0, 4.0])
    )
    before = (
        state.centers.copy(),
        state.variances.copy(),
        state.query_rows_used_for_fit,
        state.query_state_updates,
    )
    query = _vector((0, 1.0), (1, 0.1))
    scores = score_d110_scpm_query(state, query)

    assert scores.shape == (3,)
    assert scores.flags.writeable is False
    np.testing.assert_array_equal(state.centers, before[0])
    np.testing.assert_array_equal(state.variances, before[1])
    assert state.query_rows_used_for_fit == before[2] == 0
    assert state.query_state_updates == before[3] == 0
    assert predict_d110_scpm_query(state, query) == "old"
    with pytest.raises(D110SCPMRuntimeError, match="1D"):
        score_d110_scpm_query(state, query[None, :])


def test_formula_is_class_permutation_equivariant_and_rejects_open_basis():
    support, labels = _three_class_k1_support()
    prior = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    first = fit_d110_scpm_runtime(support, np.asarray([0, 1, 2]), _basis(), prior)
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    second = fit_d110_scpm_runtime(support, permutation, _basis(), prior)
    query = _vector((0, 0.1), (1, 1.0), (2, 0.2))
    np.testing.assert_allclose(
        score_d110_scpm_query(second, query)[permutation],
        score_d110_scpm_query(first, query),
        rtol=0.0,
        atol=1.0e-12,
    )
    with pytest.raises(D110SCPMRuntimeError, match="row-orthonormal"):
        fit_d110_scpm_runtime(
            support,
            labels,
            np.ones((3, 160), dtype=np.float64),
            prior,
        )
