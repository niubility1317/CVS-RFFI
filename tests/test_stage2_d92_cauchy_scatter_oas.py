from __future__ import annotations

import time

import numpy as np
import pytest

from cvsrffi.stage2_d92_cauchy_scatter_oas import (
    build_cauchy_scatter_oas_statistics,
    compile_cauchy_scatter_oas_affine,
)


def _hand_fixture() -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Seven balanced 2-D classes; class zero is checked by hand below."""

    base = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, 4.0], [4.0, 1.0]],
        dtype=np.float64,
    )
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    rows = []
    labels = []
    for class_index in range(7):
        rows.append(base + np.asarray([10.0 * class_index, -class_index]))
        labels.extend([class_index] * len(base))
    return np.concatenate(rows, axis=0), np.asarray(labels), [weights] * 7


def test_csoas_uses_nonweighted_class_mean_but_hand_weighted_scatter_and_oas():
    """Would fail if the classifier mean, weighted center, scatter, or OAS equation drifted."""

    rows, labels, weights = _hand_fixture()

    statistics = build_cauchy_scatter_oas_statistics(
        rows,
        labels,
        weights,
        class_count=7,
        k_shot=5,
        old_class_count=6,
    )

    # Independently hand-calculated from class 0, not with the implementation.
    np.testing.assert_allclose(statistics.classification_means[0], [1.4, 1.6])
    np.testing.assert_allclose(
        statistics.audit["d92_csoas_weighted_center_by_class"][0], [1.05, 1.3]
    )
    np.testing.assert_allclose(statistics.audit["d92_csoas_effective_sample_size_by_class"][0], 4.444444444444445, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_scatter_trace_by_class"][0], 5.493548387096774, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_tau_by_class"][0], 2.746774193548387, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_alpha_by_class"][0], 4.076090010405827, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_rho_by_class"][0], 1.0, rtol=0.0, atol=1.0e-14)
    # Shrinkage preserves every class trace even when rho is clipped to one.
    np.testing.assert_allclose(statistics.audit["d92_csoas_shrunk_trace_by_class"][0], 5.493548387096774, rtol=0.0, atol=1.0e-14)


def test_csoas_uses_the_closed_form_effective_dof_oas_not_a_fixed_shrinkage():
    """Would fail if rho ignored the Cauchy effective sample size or OAS denominator."""

    class_zero = np.asarray(
        [
            [-0.6675185718255245, 0.9275238003546156, -0.47579417185340583],
            [-0.19787719091639938, -1.3119836405529348, -0.23587252505641282],
            [-0.6131352721869353, 1.7216715988127644, -1.206724214726855],
        ]
    )
    weight = [0.2875880730269743, 0.29435680652096086, 0.41805512045206494]
    rows = np.concatenate(
        [class_zero + np.asarray([float(index), 0.0, 0.0]) for index in range(7)]
    )
    labels = np.repeat(np.arange(7), 3)

    statistics = build_cauchy_scatter_oas_statistics(
        rows,
        labels,
        [weight] * 7,
        class_count=7,
        k_shot=3,
        old_class_count=6,
    )

    np.testing.assert_allclose(statistics.audit["d92_csoas_effective_sample_size_by_class"][0], 2.905938436933943, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_tau_by_class"][0], 0.9444046741383859, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_alpha_by_class"][0], 0.8468573809608981, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_oas_rho_by_class"][0], 0.8100291433458454, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_scatter_trace_by_class"][0], 2.8332140224151576, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(statistics.audit["d92_csoas_shrunk_trace_by_class"][0], 2.833214022415158, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        statistics.covariance,
        [
            [0.7766525922442328, -0.06918572280643014, 0.01574808579377231],
            [-0.06918572280643015, 1.2379548626333043, -0.13836443127303402],
            [0.01574808579377231, -0.13836443127303402, 0.8186065675376208],
        ],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_csoas_is_row_and_label_permutation_equivariant_without_changing_group_balance():
    """Would fail if support order or label IDs affected the common covariance."""

    rows, labels, weights = _hand_fixture()
    baseline = build_cauchy_scatter_oas_statistics(
        rows, labels, weights, class_count=7, k_shot=5, old_class_count=6
    )
    order = np.asarray(
        [28, 4, 16, 9, 1, 33, 7, 25, 14, 0, 22, 31, 12, 6, 18, 34, 10, 2, 20, 30,
         8, 15, 26, 3, 13, 23, 5, 17, 24, 32, 11, 19, 29, 21, 27],
        dtype=np.int64,
    )
    mapping = np.asarray([2, 5, 1, 0, 4, 3, 6], dtype=np.int64)
    original_weight_per_row = np.asarray(weights, dtype=np.float64)[
        labels, np.arange(len(labels), dtype=np.int64) % 5
    ]
    permuted_labels = mapping[labels[order]]
    permuted_weight_per_row = original_weight_per_row[order]
    permuted_weights = np.stack(
        [
            permuted_weight_per_row[permuted_labels == class_index]
            for class_index in range(7)
        ]
    )
    permuted = build_cauchy_scatter_oas_statistics(
        rows[order],
        permuted_labels,
        permuted_weights,
        class_count=7,
        k_shot=5,
        old_class_count=6,
    )

    np.testing.assert_allclose(
        permuted.classification_means[mapping], baseline.classification_means
    )
    np.testing.assert_allclose(permuted.covariance, baseline.covariance)


def test_csoas_rejects_effective_dof_degeneracy_instead_of_using_an_unsafe_head():
    """Would fail if one-hot Cauchy weights silently produced a usable covariance."""

    rows, labels, _ = _hand_fixture()
    weights = np.zeros((7, 5), dtype=np.float64)
    weights[:, 0] = 1.0

    with pytest.raises(Exception, match="weight|effective_dof|sample_size"):
        build_cauchy_scatter_oas_statistics(
            rows, labels, weights, class_count=7, k_shot=5, old_class_count=6
        )


def test_csoas_compiles_one_centered_equal_prior_full_affine_head():
    """Would fail if the FULL solve used weighted classifier means or a non-common affine term."""

    rows, labels, weights = _hand_fixture()
    statistics = build_cauchy_scatter_oas_statistics(
        rows, labels, weights, class_count=7, k_shot=5, old_class_count=6
    )
    coefficient, intercept, audit = compile_cauchy_scatter_oas_affine(statistics)

    assert coefficient.shape == (7, 2)
    assert intercept.shape == (7,)
    assert audit["d92_csoas_centered_coefficient_mean_max_abs"] <= 1.0e-12
    assert audit["d92_csoas_centered_intercept_mean_abs"] <= 1.0e-12
    assert audit["d92_csoas_full_solve_count"] == 1
    assert audit["d92_csoas_covariance_policy"] == "sklearn_lsqr_auto_shrinkage_equal_prior"


def test_csoas_fast_scatter_matches_a_hand_streaming_reference():
    """Would fail if a vectorized scatter changed the weighted-center formula or OAS output."""

    rng = np.random.default_rng(92_819)
    class_count, k_shot, dimension = 7, 4, 5
    rows = rng.normal(size=(class_count * k_shot, dimension))
    labels = np.repeat(np.arange(class_count), k_shot)
    weights = rng.uniform(0.1, 1.0, size=(class_count, k_shot))
    weights /= weights.sum(axis=1, keepdims=True)
    statistics = build_cauchy_scatter_oas_statistics(
        rows, labels, weights, class_count=class_count, k_shot=k_shot, old_class_count=6
    )

    reference_group = np.zeros((dimension, dimension), dtype=np.float64)
    for class_index in range(class_count):
        x = rows[labels == class_index]
        a = weights[class_index]
        center = np.sum(a[:, None] * x, axis=0)
        scatter = np.zeros((dimension, dimension), dtype=np.float64)
        for row, weight in zip(x, a):
            residual = row - center
            scatter += weight * np.outer(residual, residual)
        scatter /= 1.0 - np.sum(a * a)
        effective_size = 1.0 / np.sum(a * a)
        tau = np.trace(scatter) / dimension
        alpha = np.mean(scatter * scatter)
        denominator = (effective_size + 1.0) * (alpha - tau * tau / dimension)
        rho = 1.0 if denominator <= 0.0 else np.clip((alpha + tau * tau) / denominator, 0.0, 1.0)
        shrunk = (1.0 - rho) * scatter + rho * tau * np.eye(dimension)
        reference_group += 0.5 * shrunk / (6 if class_index < 6 else 1)

    np.testing.assert_allclose(statistics.covariance, reference_group, rtol=0.0, atol=1.0e-12)


def test_csoas_c11_k10_p288_stats_complete_within_support_side_budget():
    """Would fail if one live-scatter implementation regressed beyond the G0 resource budget."""

    rng = np.random.default_rng(92_820)
    class_count, k_shot, dimension = 11, 10, 288
    labels = np.repeat(np.arange(class_count), k_shot)
    means = rng.normal(size=(class_count, dimension))
    rows = means[labels] + 0.08 * rng.normal(size=(class_count * k_shot, dimension))
    weights = rng.uniform(0.1, 1.0, size=(class_count, k_shot))
    weights /= weights.sum(axis=1, keepdims=True)
    # Warm numerical libraries once, then measure only the support statistic.
    build_cauchy_scatter_oas_statistics(
        rows, labels, weights, class_count=class_count, k_shot=k_shot, old_class_count=6
    )
    started = time.perf_counter()
    statistics = build_cauchy_scatter_oas_statistics(
        rows, labels, weights, class_count=class_count, k_shot=k_shot, old_class_count=6
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - started)

    assert statistics.audit["d92_csoas_live_class_scatter_buffers"] == 1
    assert statistics.audit["d92_csoas_class_matrix_stack"] is False
    assert elapsed_ms <= 150.0
