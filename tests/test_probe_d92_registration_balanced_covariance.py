from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe
from scripts import probe_d92_registration_balanced_covariance as probe


def _ground_audit(basis_hash: str, weight_hash: str) -> dict[str, object]:
    return {
        "d81_basis_sha256": basis_hash,
        "d81_spectral_weight_sha256": weight_hash,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }


def test_synthetic_d62_stack_uses_d92_in_all_registered_components():
    rng = np.random.default_rng(920)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    fit, call_records, transform_records = probe.build_d92_fit(
        d42, basis, weights, _ground_audit("a" * 64, "b" * 64)
    )
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert audit["d92_registration_balanced_active"] is True
    assert audit["d92_component_fit_count"] > 0
    assert len(call_records) > 0
    assert len(transform_records) == audit["d92_component_fit_count"]


def test_lock_has_no_scene_receiver_seed_or_query_tuning():
    assert "fixed Sigma=0.5*Sigma_old+0.5*Sigma_new" in probe.FORMULA
    assert "query truth" not in probe.FORMULA.lower()
    assert "receiver" not in probe.FORMULA.lower()
    assert "scene" not in probe.FORMULA.lower()


def test_td_htrc_builder_is_explicit_opt_in_and_reaches_d92_components():
    rng = np.random.default_rng(922)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_centers = rng.normal(size=(6, 160))
    fit, call_records, transform_records = probe.build_td_htrc_fit(
        d42,
        basis,
        weights,
        _ground_audit("e" * 64, "f" * 64),
        ground_centers,
    )
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert audit["td_htrc_method"] == "TD-HTRC-M2.1"
    assert audit["td_htrc_query_rows_used"] == 0
    assert audit["td_htrc_query_transform_compiled_into_intercept"] is True
    assert audit["d92_registration_balanced_active"] is True
    assert len(call_records) > 0
    assert len(transform_records) == 1


def test_td_htrc_m22_builder_passes_posterior_uncertainty_into_d92():
    rng = np.random.default_rng(923)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = _ground_audit("1" * 64, "2" * 64)
    ground_full = rng.normal(size=(6, 288))
    fit, call_records, transform_records = probe.build_td_htrc_m22_fit(
        d42,
        basis,
        weights,
        ground_audit,
        ground_full[:, :160],
        ground_full_centers=ground_full,
    )
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert audit["td_htrc_method"] == "TD-HTRC-M2.2"
    assert audit["td_htrc_m22_posterior_uncertainty_enabled"] is True
    assert audit["d92_center_uncertainty_enabled"] is True
    assert audit["d92_center_uncertainty_trace"] > 0.0
    assert len(call_records) > 0
    assert len(transform_records) == 1


def test_registration_before_head_is_exact_d81_for_k5():
    rng = np.random.default_rng(921)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = _ground_audit("c" * 64, "d" * 64)
    d81_fit, _, _ = d81_probe.build_d81_fit(
        d42, basis, weights, ground_audit
    )
    d92_fit, _, _ = probe.build_d92_fit(d42, basis, weights, ground_audit)
    classes, shots = 6, 5
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    d81_coefficient, d81_intercept, _ = d81_fit(rows, labels, classes, shots)
    d92_coefficient, d92_intercept, audit = d92_fit(
        rows, labels, classes, shots
    )
    np.testing.assert_array_equal(d92_coefficient, d81_coefficient)
    np.testing.assert_array_equal(d92_intercept, d81_intercept)
    assert audit["d92_status"] == "before_exact_d81"
