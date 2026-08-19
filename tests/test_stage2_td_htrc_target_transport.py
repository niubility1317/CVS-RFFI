from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_td_htrc_target_transport import (
    TDHTRCError,
    apply_shared_target_transport,
    build_td_htrc_component_fit,
    estimate_shared_target_transport,
)


def _episode(classes: int, shots: int, offset: np.ndarray, seed: int = 1201):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    ground = rng.normal(size=(6, 160))
    target_centers = np.concatenate(
        [ground + offset[None, :], rng.normal(size=(classes - 6, 160))], axis=0
    )
    rows = np.zeros((classes * shots, 288), dtype=np.float64)
    rows[:, :160] = target_centers[labels] + 0.01 * rng.normal(
        size=(len(labels), 160)
    )
    rows[:, 160:] = rng.normal(size=(len(labels), 128))
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    spectral = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    return rows.astype(np.float32), labels, ground, basis, spectral


def test_shared_offset_uses_old_class_pairs_and_keeps_spectral_outside_component():
    offset = np.zeros(160, dtype=np.float64)
    offset[0] = 0.35
    offset[17] = -0.22
    rows, labels, ground, basis, spectral = _episode(11, 5, offset)
    estimate = estimate_shared_target_transport(
        rows, labels, 11, 5, ground, basis, spectral
    )
    np.testing.assert_allclose(
        estimate.shared_offset, offset, rtol=0.0, atol=0.04
    )
    assert estimate.audit["query_rows_used"] == 0
    assert estimate.audit["query_truth_used"] is False
    assert estimate.audit["ground_anchor_update_access"] is False
    assert estimate.audit["spectral_perpendicular_ratio"] > 0.0
    assert estimate.offset_covariance.shape == (160, 160)
    assert np.all(np.linalg.eigvalsh(estimate.offset_covariance) > 0.0)


def test_k1_shared_offset_is_identifiable_and_transport_preserves_auxiliary_blocks():
    offset = np.zeros(160, dtype=np.float64)
    offset[3] = 0.4
    rows, labels, ground, basis, spectral = _episode(11, 1, offset, seed=1202)
    estimate = estimate_shared_target_transport(
        rows, labels, 11, 1, ground, basis, spectral
    )
    canonical = apply_shared_target_transport(rows, estimate)
    np.testing.assert_array_equal(canonical[:, 160:], rows[:, 160:])
    np.testing.assert_allclose(estimate.shared_offset, offset, atol=0.04, rtol=0.0)
    assert estimate.audit["k_shot"] == 1


def test_builder_compiles_shared_transport_into_raw_query_intercept():
    offset = np.zeros(160, dtype=np.float64)
    offset[5] = 0.25
    rows, labels, ground, basis, spectral = _episode(11, 2, offset, seed=1203)
    coefficient = np.arange(11 * 288, dtype=np.float64).reshape(11, 288) / 1000.0
    intercept = np.linspace(-0.5, 0.5, 11, dtype=np.float64)

    def component_fit(x, y, class_count, k_shot):
        assert x.shape == rows.shape
        assert np.array_equal(y, labels)
        return coefficient.astype(np.float32), intercept.astype(np.float32), {
            "base_component": True
        }

    records: list[dict[str, object]] = []
    fit = build_td_htrc_component_fit(
        component_fit,
        ground_class_centers=ground,
        basis=basis,
        spectral_weights=spectral,
        component_arm="synthetic",
        collector=records,
    )
    actual_coefficient, actual_intercept, audit = fit(rows, labels, 11, 2)
    estimate = estimate_shared_target_transport(
        rows, labels, 11, 2, ground, basis, spectral
    )
    expected_intercept = intercept - coefficient[:, :160] @ estimate.shared_offset
    np.testing.assert_allclose(actual_coefficient, coefficient, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(actual_intercept, expected_intercept, rtol=0.0, atol=2e-6)
    assert audit["td_htrc_query_rows_used"] == 0
    assert audit["td_htrc_query_transform_compiled_into_intercept"] is True
    assert records[0]["component_arm"] == "synthetic"


def test_builder_rejects_mismatched_old_class_registry():
    rows, labels, ground, basis, spectral = _episode(11, 1, np.zeros(160))

    with pytest.raises(TDHTRCError, match="registry mismatch"):
        build_td_htrc_component_fit(
            lambda x, y, c, k: (
                np.zeros((c, 288), dtype=np.float32),
                np.zeros(c, dtype=np.float32),
                {},
            ),
            ground_class_centers=ground,
            basis=basis,
            spectral_weights=spectral,
            component_arm="synthetic",
            collector=[],
            ground_class_registry=("a", "b"),
            target_old_class_registry=("b", "a"),
        )
