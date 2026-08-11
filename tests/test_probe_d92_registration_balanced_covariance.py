from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe
from scripts import probe_d92_registration_balanced_covariance as probe


def test_synthetic_d62_stack_uses_d92_in_all_registered_components():
    rng = np.random.default_rng(920)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "a" * 64,
        "d81_spectral_weight_sha256": "b" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    fit, call_records, transform_records = probe.build_d92_fit(
        d42, basis, weights, ground_audit
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


def test_registration_before_head_is_exact_d81_for_k5():
    rng = np.random.default_rng(921)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "c" * 64,
        "d81_spectral_weight_sha256": "d" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
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


def test_registered_e0_d_modes_use_only_the_frozen_component_graphs():
    """Would fail if a D mode still invoked LOO or the wrong primary arm."""

    rng = np.random.default_rng(922)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "e" * 64,
        "d81_spectral_weight_sha256": "f" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    expected = {
        "fusion_loo": (12, 6, 6),
        "full_only": (1, 1, 0),
        "block_only": (1, 0, 1),
        "fixed50": (2, 1, 1),
    }
    for mode, (total, full_count, block_count) in expected.items():
        fit, _, _ = probe.build_d92_fit(
            d42,
            basis,
            weights,
            ground_audit,
            disable_registered_fisher=True,
            registered_d_mode=mode,
        )
        coefficient, intercept, audit = fit(rows, labels, classes, shots)
        inventory = audit["d92_component_fit_inventory"]
        assert np.isfinite(coefficient).all()
        assert np.isfinite(intercept).all()
        assert audit["d92_registered_d_mode_effective"] == mode
        assert audit["d92_fisher_residual_pareto_active"] is False
        assert audit["d92_component_fit_count"] == total
        assert inventory["actual_component_fit_count"] == total
        assert inventory["full_component_fit_count"] == full_count
        assert inventory["block3_component_fit_count"] == block_count


def test_registered_e0_d_mode_keeps_k1_and_k2_as_exact_d92_full_aliases():
    """Would fail if the E0 D switch leaked into the frozen low-K fallback."""

    rng = np.random.default_rng(923)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "1" * 64,
        "d81_spectral_weight_sha256": "2" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    for shots in (1, 2):
        classes = 11
        labels = np.repeat(np.arange(classes), shots)
        means = rng.normal(size=(classes, 288))
        rows = means[labels].astype(np.float32)
        full_fit, _, _ = probe.build_d92_fit(
            d42, basis, weights, ground_audit
        )
        d_fit, _, _ = probe.build_d92_fit(
            d42,
            basis,
            weights,
            ground_audit,
            disable_registered_fisher=True,
            registered_d_mode="fixed50",
        )
        full_coefficient, full_intercept, _ = full_fit(
            rows, labels, classes, shots
        )
        coefficient, intercept, audit = d_fit(rows, labels, classes, shots)
        np.testing.assert_array_equal(coefficient, full_coefficient)
        np.testing.assert_array_equal(intercept, full_intercept)
        assert audit["d92_registered_d_mode_effective"] == "d92_full_alias"
        assert audit["d92_k1_k2_exact_full_alias"] is True
