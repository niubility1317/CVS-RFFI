from __future__ import annotations

import numpy as np

from cvsrffi.stage2_td_htrc_m22 import (
    build_td_htrc_m22_component_fit,
    estimate_m22_transport,
)


def _episode(
    classes: int = 11,
    shots: int = 5,
    *,
    complete_ground: bool,
    seed: int = 2210,
):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    ground_full = rng.normal(size=(6, 288))
    offset = np.zeros(288, dtype=np.float64)
    offset[:160] = 0.05 * rng.normal(size=160)
    offset[160:] = 0.05 * rng.normal(size=128)
    target_centers = np.concatenate(
        [ground_full + offset[None, :], rng.normal(size=(classes - 6, 288))],
        axis=0,
    )
    rows = target_centers[labels] + 0.02 * rng.normal(
        size=(len(labels), 288)
    )
    basis, _ = np.linalg.qr(rng.normal(size=(160, 4)))
    spectral = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    return (
        rows.astype(np.float32),
        labels,
        ground_full[:, :160],
        basis,
        spectral,
        ground_full if complete_ground else None,
    )


def test_m22_builds_low_rank_transport_adaptive_spectrum_and_posteriors():
    rows, labels, ground, basis, spectral, full_ground = _episode(
        complete_ground=True
    )
    estimate, canonical, audit = estimate_m22_transport(
        rows,
        labels,
        11,
        5,
        ground,
        basis,
        spectral,
        ground_full_centers=full_ground,
    )
    assert canonical.shape == rows.shape
    assert estimate.transport_matrix.shape == (288, 288)
    np.testing.assert_allclose(
        estimate.transport_matrix @ estimate.inverse_transport_matrix,
        np.eye(288),
        rtol=0.0,
        atol=2e-8,
    )
    assert audit["transport_low_rank_enabled"] is True
    assert audit["transport_complete_ground_centres_available"] is True
    assert audit["adaptive_spectrum_enabled"] is True
    assert 1 <= audit["adaptive_spectrum_audit"]["adaptive_spectrum_retained_rank"] <= 5
    assert estimate.posterior_centers.shape == (11, 288)
    assert estimate.posterior_variance.shape == (11, 288)
    assert np.all(estimate.posterior_variance > 0.0)
    assert audit["posterior_audit"]["posterior_prior_enabled_by_class"][:6] == [
        288
    ] * 6
    assert audit["posterior_audit"]["posterior_prior_enabled_by_class"][6:] == [
        0
    ] * 5
    assert audit["query_rows_used"] == 0
    assert audit["query_truth_used"] is False


def test_m22_without_complete_ground_centres_keeps_auxiliary_transport_fixed():
    rows, labels, ground, basis, spectral, _ = _episode(
        complete_ground=False, seed=2211
    )
    estimate, _, audit = estimate_m22_transport(
        rows, labels, 11, 5, ground, basis, spectral
    )
    assert audit["transport_complete_ground_centres_available"] is False
    assert audit["transport_audit"]["transport_block_scale_estimated"] == [
        True,
        False,
        False,
    ]
    np.testing.assert_allclose(
        estimate.transport_matrix[160:, 160:], np.eye(128), rtol=0.0, atol=0.0
    )


def test_m22_compiles_affine_head_to_raw_query_coordinates():
    rows, labels, ground, basis, spectral, full_ground = _episode(
        complete_ground=True, seed=2212
    )
    rng = np.random.default_rng(2213)
    coefficient = rng.normal(size=(11, 288))
    intercept = rng.normal(size=11)
    observed = {}

    def component_fit(x, y, class_count, k_shot):
        observed["support"] = np.asarray(x).copy()
        return coefficient.astype(np.float32), intercept.astype(np.float32), {
            "component": "synthetic"
        }

    records: list[dict[str, object]] = []
    fit = build_td_htrc_m22_component_fit(
        component_fit,
        ground_class_centers=ground,
        ground_full_centers=full_ground,
        basis=basis,
        spectral_weights=spectral,
        component_arm="synthetic",
        collector=records,
    )
    actual_coefficient, actual_intercept, audit = fit(rows, labels, 11, 5)
    estimate, canonical, _ = estimate_m22_transport(
        rows, labels, 11, 5, ground, basis, spectral,
        ground_full_centers=full_ground,
    )
    expected_coefficient = coefficient @ estimate.inverse_transport_matrix
    expected_intercept = intercept - expected_coefficient @ estimate.shared_offset
    np.testing.assert_allclose(
        actual_coefficient, expected_coefficient, rtol=0.0, atol=3e-6
    )
    np.testing.assert_allclose(
        actual_intercept, expected_intercept, rtol=0.0, atol=3e-6
    )
    assert observed["support"].shape == rows.shape
    assert audit["td_htrc_m22_query_rows_used"] == 0
    assert audit["td_htrc_m22_query_transform_compiled_into_affine"] is True
    assert records[0]["component_arm"] == "synthetic"
