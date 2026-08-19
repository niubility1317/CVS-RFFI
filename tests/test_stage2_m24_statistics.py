from __future__ import annotations

import numpy as np

from cvsrffi.stage2_m24_center import estimate_centers
from cvsrffi.stage2_m24_covariance import relative_psd_jitter
from cvsrffi.stage2_m24_prior_transport import gated_old_prior
from cvsrffi.stage2_m24_uncertainty import normalized_capped_penalty


def test_support_decision_and_covariance_centers_are_separate() -> None:
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    labels = np.array(["a", "a", "b", "b"])
    support, decision, covariance = estimate_centers(
        rows,
        labels,
        ("a", "b"),
        center_weights=np.array([0.9, 0.1, 0.5, 0.5]),
        covariance_weights=np.array([0.5, 0.5, 0.9, 0.1]),
    )
    assert not np.allclose(support, decision)
    assert not np.allclose(decision, covariance)
    np.testing.assert_allclose(np.linalg.norm(decision, axis=1), 1.0)


def test_relative_jitter_is_psd_and_scale_equivariant() -> None:
    covariance = np.array([[2.0, 3.0], [3.0, 2.0]])
    repaired, audit = relative_psd_jitter(covariance, relative_floor=1.0e-3)
    scaled, scaled_audit = relative_psd_jitter(10.0 * covariance, relative_floor=1.0e-3)
    assert np.min(np.linalg.eigvalsh(repaired)) > 0.0
    np.testing.assert_allclose(scaled, 10.0 * repaired, rtol=1.0e-10, atol=1.0e-10)
    assert np.isclose(scaled_audit["jitter"], 10.0 * audit["jitter"])


def test_k1_prior_is_forced_off_and_harmful_prior_falls_back() -> None:
    support = np.array([[1.0, 0.0], [0.0, 1.0]])
    prior = np.array([[-1.0, 0.0], [0.0, -1.0]])
    rows = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    targets = np.array([0, 0, 1, 1])
    k1, k1_gate, k1_audit = gated_old_prior(
        support, prior, k_shot=1, support_rows=rows, support_targets=targets
    )
    np.testing.assert_array_equal(k1, support)
    np.testing.assert_array_equal(k1_gate, np.zeros(2))
    assert k1_audit["mode"] == "forced_off_k1"
    safe, gate, audit = gated_old_prior(
        support, prior, k_shot=10, support_rows=rows, support_targets=targets
    )
    np.testing.assert_array_equal(safe, support)
    np.testing.assert_array_equal(gate, np.zeros(2))
    assert audit["fallback_count"] == 2
    assert audit["mode"] == "support_loo_no_harm"


def test_uncertainty_penalty_is_scale_normalized_and_capped() -> None:
    uncertainty = np.array([[1000.0, 1000.0], [1.0, 1.0]])
    precision = np.eye(2)
    penalty = normalized_capped_penalty(uncertainty, precision, cap=0.2)
    assert penalty.shape == (2,)
    assert np.all(penalty >= 0.0)
    assert np.max(penalty) <= 0.2 + 1.0e-12
