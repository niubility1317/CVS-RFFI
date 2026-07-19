import numpy as np
import pytest

from scripts.probe_d85_ground_radius_calibrated_consensus import (
    D85ProbeError,
    radius_calibrated_consensus_templates,
)


def _fixture():
    rng = np.random.default_rng(8501)
    classes = rng.normal(size=(4, 160))
    classes /= np.linalg.norm(classes, axis=1, keepdims=True)
    drift = rng.normal(scale=0.02, size=(5, 160))
    prototypes = classes[None, :, :] + drift[:, None, :]
    radius = np.full((5, 4), 0.01, dtype=np.float64)
    radius[4] = 0.20
    return prototypes, radius


def test_radius_calibration_downweights_broad_ground_domain() -> None:
    prototypes, radius = _fixture()
    _templates, weights, audit = radius_calibrated_consensus_templates(
        prototypes, radius
    )

    assert weights.shape == (5,)
    assert weights[4] < min(weights[:4])
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-14)
    assert audit["radius_scan_count"] == 0
    assert audit["radius_hyperparameter_count"] == 0
    assert audit["ground_class_centers_discarded"] is True


def test_radius_calibration_is_ground_class_permutation_invariant() -> None:
    prototypes, radius = _fixture()
    templates_a, weights_a, audit_a = radius_calibrated_consensus_templates(
        prototypes, radius
    )
    order = np.asarray([2, 0, 3, 1])
    templates_b, weights_b, audit_b = radius_calibrated_consensus_templates(
        prototypes[:, order], radius[:, order]
    )

    assert np.array_equal(templates_a, templates_b)
    assert np.array_equal(weights_a, weights_b)
    assert audit_a["weight_sha256"] == audit_b["weight_sha256"]


def test_radius_calibration_rejects_missing_or_zero_radius() -> None:
    prototypes, radius = _fixture()
    radius[0, 0] = 0.0
    with pytest.raises(D85ProbeError, match="tensor drift"):
        radius_calibrated_consensus_templates(prototypes, radius)
