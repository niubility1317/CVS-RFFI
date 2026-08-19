from __future__ import annotations

import numpy as np

from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m24_quality import (
    effective_sample_size,
    ess_safe_weights,
    if_residual_reliability,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_physical_if256_matches_zero_padded_historical_f1_geometry() -> None:
    rng = np.random.default_rng(2401)
    legacy = rng.normal(size=(7, 288))
    actual = physical_if256(legacy)
    expected = _unit(
        np.concatenate(
            [_unit(legacy[:, :160]), 4.0 * _unit(legacy[:, 160:256])], axis=1
        )
    ).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=2.0e-7)


def test_ess_safe_weights_blend_extreme_quality_to_required_floor() -> None:
    raw = np.array([1.0, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9])
    weights, audit = ess_safe_weights(raw, k_shot=5)
    assert np.isclose(weights.sum(), 1.0)
    assert effective_sample_size(weights) >= 3.0 - 1.0e-9
    assert audit["required_ess"] == 3.0
    assert 0.0 < audit["uniform_blend"] <= 1.0


def test_k1_quality_is_uniform_and_if_residual_is_independent_of_rf_quality() -> None:
    weights, audit = ess_safe_weights(np.array([0.03]), k_shot=1)
    np.testing.assert_array_equal(weights, np.array([1.0]))
    assert audit["uniform_blend"] == 1.0

    rows = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    labels = np.array(["a", "a", "b", "b"])
    first = if_residual_reliability(rows, labels, ("a", "b"))
    second = if_residual_reliability(rows, labels, ("a", "b"))
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0.0)
