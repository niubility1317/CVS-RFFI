"""Regression for FP32 loss of a class-common LDA score component."""
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


def rows(scale):
    rng = np.random.default_rng(392005)
    labels = np.repeat(np.arange(6), 5)
    means = rng.normal(size=(6, 288))
    noise = rng.normal(size=(30, 288))
    common = rng.normal(size=(1, 288))
    return common + scale * (means[labels] + noise), labels


def original_solution(x, y):
    estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
        priors=np.full(6, 1 / 6), store_covariance=True).fit(x, y)
    c = np.linalg.lstsq(estimator.covariance_, estimator.means_.T, rcond=None)[0].T
    b = -.5 * np.diag(estimator.means_ @ c.T) + np.log(np.full(6, 1 / 6))
    return c.astype(np.float32), b.astype(np.float32), estimator


def test_large_common_term_fails_old_guard_and_is_repaired():
    x, y = rows(1e-4)
    old_c, old_b, estimator = original_solution(x, y)
    reference = estimator.predict(x)
    assert np.any((x @ old_c.astype(np.float64).T + old_b).argmax(1) != reference)
    c, b, audit = d42._fit_equal_prior_lda(x, y, 6, 5)
    assert audit["deployment_common_affine_centered_before_fp32"] is True
    assert audit["uncentered_deployment_mismatch_count"] > 0
    assert audit["sklearn_prediction_equivalent"] is True
    np.testing.assert_array_equal((x @ c.astype(np.float64).T + b).argmax(1), reference)


def test_existing_success_path_is_bitwise_unchanged():
    x, y = rows(1e-2)
    old_c, old_b, _ = original_solution(x, y)
    c, b, audit = d42._fit_equal_prior_lda(x, y, 6, 5)
    np.testing.assert_array_equal(c, old_c)
    np.testing.assert_array_equal(b, old_b)
    assert audit["deployment_common_affine_centered_before_fp32"] is False


def test_guard_still_rejects_nonrounding_reference_drift(monkeypatch):
    x, y = rows(1e-4)
    original = d42.LinearDiscriminantAnalysis.predict
    monkeypatch.setattr(d42.LinearDiscriminantAnalysis, "predict",
        lambda self, value: (original(self, value) + 1) % 6)
    with pytest.raises(d42.D42UnifiedShrinkageLDAError, match="float64 coefficient prediction drift"):
        d42._fit_equal_prior_lda(x, y, 6, 5)
