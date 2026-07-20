from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d89_v2_radius_cauchy_center import (
    radius_reliability_ground_spectrum,
    translate_to_robust_centers,
)


def _ground(seed: int = 8901):
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(6, 160))
    domain = rng.normal(scale=0.08, size=(14, 160))
    interaction = rng.normal(scale=0.01, size=(14, 6, 160))
    prototypes = identity[None] + domain[:, None] + interaction
    radius = rng.uniform(0.001, 0.03, size=(14, 6))
    return prototypes, radius


def test_spectrum_is_domain_and_ground_class_permutation_invariant() -> None:
    prototypes, radius = _ground()
    basis, weights, audit = radius_reliability_ground_spectrum(
        prototypes, radius, 0.0015
    )
    domain_order = np.asarray([9, 0, 4, 13, 1, 11, 2, 8, 3, 12, 5, 10, 6, 7])
    class_order = np.asarray([4, 0, 5, 2, 1, 3])
    basis2, weights2, audit2 = radius_reliability_ground_spectrum(
        prototypes[domain_order][:, class_order],
        radius[domain_order][:, class_order],
        0.0015,
    )
    covariance1 = (basis * weights[None, :]) @ basis.T
    covariance2 = (basis2 * weights2[None, :]) @ basis2.T
    np.testing.assert_allclose(covariance2, covariance1, atol=2.0e-12)
    assert audit["ground_component_input_count"] == 84
    assert audit["equal_ground_class_contribution"] is True
    assert audit["domain_weight_sum_max_abs_error"] <= 1.0e-14
    assert audit2["ground_target_identity_mapping_access"] is False


def test_d81_center_formula_preserves_residuals_and_k2_identity() -> None:
    prototypes, radius = _ground()
    basis, weights, _ = radius_reliability_ground_spectrum(
        prototypes, radius, 0.0015
    )
    rng = np.random.default_rng(8902)
    labels = np.repeat(np.arange(3), 5)
    rows = rng.normal(size=(15, 288))
    transformed, audit = translate_to_robust_centers(
        rows, labels, 3, 5, basis, weights
    )
    assert audit["old_new_role_specific_branch"] is False
    assert audit["within_class_residual_max_abs_error"] <= 2.0e-12
    np.testing.assert_array_equal(transformed[:, 160:], rows[:, 160:])
    labels2 = np.repeat(np.arange(3), 2)
    rows2 = rng.normal(size=(6, 288))
    transformed2, audit2 = translate_to_robust_centers(
        rows2, labels2, 3, 2, basis, weights
    )
    np.testing.assert_array_equal(transformed2, rows2)
    assert audit2["k1_k2_exact_identity"] is True
