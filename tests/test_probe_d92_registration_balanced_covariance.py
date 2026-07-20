from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
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
