from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d92_cauchy_scatter_oas import (
    D92CauchyScatterOASNumericalError,
)
from cvsrffi.stage2_d92_cross_class_offblock_consensus import (
    D92CCOCError,
    D92CCOCNumericalError,
)
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe
from scripts import probe_d92_registration_balanced_covariance as probe


def _ocf_hand_fixture():
    """Return independently hand-checked full/block affine components."""

    classes, shots = 8, 3
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    after_rows = np.ones((classes * shots, 1), dtype=np.float32)
    full_coefficient = np.asarray(
        [[8.0], [12.0], [8.0], [12.0], [8.0], [12.0], [31.0], [32.0]],
        dtype=np.float32,
    )
    full_intercept = np.asarray(
        [6.0, 2.0, 6.0, 4.0, 8.0, 4.0, 40.0, 41.0], dtype=np.float32
    )
    block_coefficient = np.asarray(
        [[96.0], [96.0], [104.0], [96.0], [104.0], [104.0], [500.0], [600.0]],
        dtype=np.float32,
    )
    block_intercept = np.asarray(
        [-5.0, -5.0, -13.0, -1.0, -9.0, -9.0, -50.0, -60.0],
        dtype=np.float32,
    )
    return (
        after_rows,
        labels,
        full_coefficient,
        full_intercept,
        block_coefficient,
        block_intercept,
        classes,
        shots,
    )


def test_ocf_math_uses_old_class_centered_rms_without_second_canonical_centering(
    monkeypatch,
):
    """Would fail if OCF changed new rows, old means, RMS alignment, or re-centered."""

    (
        after_rows,
        labels,
        full_coefficient,
        full_intercept,
        block_coefficient,
        block_intercept,
        classes,
        shots,
    ) = _ocf_hand_fixture()

    def forbidden_centering(*_args, **_kwargs):
        raise AssertionError("OCF must not invoke all-class canonical centering")

    monkeypatch.setattr(probe.d43, "_center_affine_scores", forbidden_centering)
    coefficient25, intercept25, audit25 = probe._build_ocf_affine_state(
        full_rows=after_rows,
        full_labels=labels,
        block_rows=after_rows,
        block_labels=labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=classes,
        k_shot=shots,
        lambda_value=0.25,
    )
    coefficient50, intercept50, audit50 = probe._build_ocf_affine_state(
        full_rows=after_rows,
        full_labels=labels,
        block_rows=after_rows,
        block_labels=labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=classes,
        k_shot=shots,
        lambda_value=0.50,
    )
    np.testing.assert_array_equal(
        coefficient25,
        np.asarray([[8.0], [11.0], [9.0], [11.0], [9.0], [12.0], [31.0], [32.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        intercept25,
        np.asarray([6.0, 3.0, 5.0, 5.0, 7.0, 4.0, 40.0, 41.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        coefficient50,
        np.asarray([[8.0], [10.0], [10.0], [10.0], [10.0], [12.0], [31.0], [32.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        intercept50,
        np.asarray([6.0, 4.0, 4.0, 6.0, 6.0, 4.0, 40.0, 41.0], dtype=np.float32),
    )
    for coefficient, intercept, audit, expected_lambda in (
        (coefficient25, intercept25, audit25, 0.25),
        (coefficient50, intercept50, audit50, 0.50),
    ):
        assert coefficient.dtype == np.float32
        assert intercept.dtype == np.float32
        assert coefficient[6:].tobytes() == full_coefficient[6:].tobytes()
        assert intercept[6:].tobytes() == full_intercept[6:].tobytes()
        np.testing.assert_allclose(
            coefficient[:6].mean(axis=0), full_coefficient[:6].mean(axis=0), atol=1.0e-5
        )
        np.testing.assert_allclose(intercept[:6].mean(), full_intercept[:6].mean(), atol=1.0e-5)
        np.testing.assert_allclose(coefficient[:6].sum(axis=0), 6.0 * full_coefficient[:6].mean(axis=0), atol=1.0e-5)
        np.testing.assert_allclose(intercept[:6].sum(), 6.0 * full_intercept[:6].mean(), atol=1.0e-5)
        assert audit["d92_ocf_lambda"] == expected_lambda
        assert audit["d92_ocf_full_old_rms"] == pytest.approx(1.0)
        assert audit["d92_ocf_block_old_rms"] == pytest.approx(2.0)
        assert audit["d92_ocf_unclipped_block_to_full_ratio"] == pytest.approx(0.5)
        assert audit["d92_ocf_new_rows_byte_exact"] is True
        assert audit["d92_ocf_no_second_all_class_centering"] is True


def test_ocf_math_rejects_invalid_lambda_label_nonfinite_or_degenerate_inputs():
    """Would fail if OCF accepted an unsafe fixed-arm support state."""

    fixture = _ocf_hand_fixture()
    kwargs = dict(
        full_rows=fixture[0],
        full_labels=fixture[1],
        block_rows=fixture[0],
        block_labels=fixture[1],
        full_coefficient=fixture[2],
        full_intercept=fixture[3],
        block_coefficient=fixture[4],
        block_intercept=fixture[5],
        class_count=fixture[6],
        k_shot=fixture[7],
    )
    with pytest.raises(probe.D92ProbeError, match="lambda"):
        probe._build_ocf_affine_state(lambda_value=0.30, **kwargs)
    missing_labels = fixture[1].copy()
    missing_labels[-1] = 6
    with pytest.raises(probe.D92ProbeError, match="registry"):
        probe._build_ocf_affine_state(
            lambda_value=0.25, full_labels=missing_labels, **{key: value for key, value in kwargs.items() if key != "full_labels"}
        )
    nonfinite_rows = fixture[0].copy()
    nonfinite_rows[0, 0] = np.nan
    with pytest.raises(probe.D92ProbeError, match="finite"):
        probe._build_ocf_affine_state(
            lambda_value=0.25, full_rows=nonfinite_rows, **{key: value for key, value in kwargs.items() if key != "full_rows"}
        )
    zero_full = np.zeros_like(fixture[2])
    zero_block = np.zeros_like(fixture[4])
    with pytest.raises(probe.D92ProbeError, match="RMS"):
        probe._build_ocf_affine_state(
            lambda_value=0.25,
            full_coefficient=zero_full,
            block_coefficient=zero_block,
            full_intercept=np.zeros_like(fixture[3]),
            block_intercept=np.zeros_like(fixture[5]),
            **{key: value for key, value in kwargs.items() if key not in {"full_coefficient", "block_coefficient", "full_intercept", "block_intercept"}},
        )


def test_ocf_cn20_transient_bound_covers_explicit_full_class_workspace():
    """Would fail if a Cn20 OCF receipt omitted live full-class FP64/FP32 arrays."""

    classes, shots, dimension = 26, 5, 288
    rng = np.random.default_rng(92_026)
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    after_rows = rng.normal(size=(classes * shots, dimension)).astype(np.float32)
    full_coefficient = rng.normal(size=(classes, dimension)).astype(np.float32)
    full_intercept = rng.normal(size=classes).astype(np.float32)
    block_coefficient = rng.normal(size=(classes, dimension)).astype(np.float32)
    block_intercept = rng.normal(size=classes).astype(np.float32)
    _, _, audit = probe._build_ocf_affine_state(
        full_rows=after_rows,
        full_labels=labels,
        block_rows=after_rows,
        block_labels=labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=classes,
        k_shot=shots,
        lambda_value=0.25,
    )
    old_rows = probe.OLD_CLASS_COUNT * shots
    explicit_live_array_bytes = (
        2 * classes * (dimension + 1) * 8
        + 3 * classes * (dimension + 1) * 4
        + classes * shots * dimension * 8
        + old_rows * dimension * 8
        + 3 * old_rows * probe.OLD_CLASS_COUNT * 8
        + 5 * probe.OLD_CLASS_COUNT * (dimension + 1) * 8
        + probe.OLD_CLASS_COUNT * (dimension + 1) * 4
    )
    assert audit["d92_ocf_support_alignment_transient_bytes_upper_bound"] >= (
        explicit_live_array_bytes
    )


@pytest.mark.parametrize(
    ("shots", "expected_affine_macs", "expected_mix_macs", "expected_total_macs"),
    (
        (5, 103_680, 8_670, 112_350),
        (10, 207_360, 8_670, 216_030),
    ),
)
def test_ocf_support_alignment_macs_include_affine_and_contrast_mix(
    shots,
    expected_affine_macs,
    expected_mix_macs,
    expected_total_macs,
):
    """Would fail if the frozen total omitted either old-support work term."""

    classes, dimension = 11, 288
    rng = np.random.default_rng(92_100 + shots)
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    after_rows = rng.normal(size=(classes * shots, dimension)).astype(np.float32)
    full_coefficient = rng.normal(size=(classes, dimension)).astype(np.float32)
    full_intercept = rng.normal(size=classes).astype(np.float32)
    block_coefficient = rng.normal(size=(classes, dimension)).astype(np.float32)
    block_intercept = rng.normal(size=classes).astype(np.float32)
    _, _, audit = probe._build_ocf_affine_state(
        full_rows=after_rows,
        full_labels=labels,
        block_rows=after_rows,
        block_labels=labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=classes,
        k_shot=shots,
        lambda_value=0.25,
    )
    affine_formula = (
        2
        * (probe.OLD_CLASS_COUNT * shots)
        * probe.OLD_CLASS_COUNT
        * dimension
    )
    mix_formula = 5 * probe.OLD_CLASS_COUNT * (dimension + 1)
    assert probe.OLD_CLASS_COUNT == 6
    assert affine_formula == expected_affine_macs
    assert mix_formula == expected_mix_macs
    assert affine_formula + mix_formula == expected_total_macs
    assert (
        audit["d92_ocf_support_alignment_affine_macs_upper_bound"]
        == affine_formula
    )
    assert (
        audit["d92_ocf_support_alignment_contrast_mix_macs_upper_bound"]
        == mix_formula
    )
    assert (
        audit["d92_ocf_support_alignment_macs_upper_bound"]
        == affine_formula + mix_formula
    )


def test_ocf_is_equivariant_under_independent_old_and_new_label_permutations():
    """Would fail if an OCF formula used an individual old or new class identity."""

    fixture = _ocf_hand_fixture()
    rows = fixture[0].copy()
    rows[:, 0] = np.linspace(-1.5, 2.5, num=len(rows), dtype=np.float32)
    labels = fixture[1]
    full_coefficient, full_intercept = fixture[2], fixture[3]
    block_coefficient, block_intercept = fixture[4], fixture[5]
    classes, shots = fixture[6], fixture[7]
    coefficient, intercept, audit = probe._build_ocf_affine_state(
        full_rows=rows,
        full_labels=labels,
        block_rows=rows,
        block_labels=labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=classes,
        k_shot=shots,
        lambda_value=0.50,
    )
    mapping = np.asarray([2, 0, 5, 1, 4, 3, 7, 6], dtype=np.int64)
    inverse = np.argsort(mapping)
    row_order = np.asarray(
        [7, 2, 18, 4, 20, 1, 11, 0, 22, 5, 16, 3, 10, 8, 6, 9, 12, 13, 14, 15, 17, 19, 21, 23],
        dtype=np.int64,
    )
    permuted_rows = rows[row_order]
    permuted_labels = mapping[labels[row_order]]
    p_coefficient, p_intercept, p_audit = probe._build_ocf_affine_state(
        full_rows=permuted_rows,
        full_labels=permuted_labels,
        block_rows=permuted_rows,
        block_labels=permuted_labels,
        full_coefficient=full_coefficient[inverse],
        full_intercept=full_intercept[inverse],
        block_coefficient=block_coefficient[inverse],
        block_intercept=block_intercept[inverse],
        class_count=classes,
        k_shot=shots,
        lambda_value=0.50,
    )
    tolerance = max(
        audit["d92_ocf_affine_invariant_tolerance"],
        p_audit["d92_ocf_affine_invariant_tolerance"],
    )
    np.testing.assert_allclose(p_coefficient[mapping], coefficient, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(p_intercept[mapping], intercept, rtol=0.0, atol=tolerance)
    assert p_coefficient[6:].tobytes() == full_coefficient[inverse][6:].tobytes()
    assert p_intercept[6:].tobytes() == full_intercept[inverse][6:].tobytes()
    assert p_audit["d92_ocf_new_rows_byte_exact"] is True


@pytest.mark.parametrize(
    ("run_arm", "disable_fisher", "registered_d_mode", "expected_fit_count"),
    (
        ("D92_FULL", False, "fusion_loo", 24),
        ("E0_FULL_ONLY", True, "full_only", 1),
        ("E0_FIXED50", True, "fixed50", 2),
        ("E0_OCF25", True, "ocf25", 2),
        ("E0_OCF50", True, "ocf50", 2),
    ),
)
def test_component_support_capture_retains_nothing_across_three_scenarios(
    run_arm,
    disable_fisher,
    registered_d_mode,
    expected_fit_count,
):
    """Would fail if any run arm kept support arrays after a completed fit."""

    rng = np.random.default_rng(92_200)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "7" * 64,
        "d81_spectral_weight_sha256": "8" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=disable_fisher,
        registered_d_mode=registered_d_mode,
    )
    classes, shots = 11, 5
    for scenario_index in range(3):
        labels = np.repeat(np.arange(classes), shots).astype(np.int64)
        means = rng.normal(size=(classes, 288)) + float(scenario_index)
        rows = (
            means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
        ).astype(np.float32)
        rows_ref = weakref.ref(rows)
        labels_ref = weakref.ref(labels)
        coefficient, intercept, audit = fit(rows, labels, classes, shots)
        retained_count = audit.get("d92_component_support_retained_count", -1)
        actual_fit_count = audit["d92_component_fit_inventory"][
            "actual_component_fit_count"
        ]
        del rows, labels, means, coefficient, intercept, audit
        gc.collect()
        assert rows_ref() is None, f"{run_arm} retained scenario {scenario_index} rows"
        assert labels_ref() is None, (
            f"{run_arm} retained scenario {scenario_index} labels"
        )
        assert retained_count == 0
        assert actual_fit_count == expected_fit_count


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


def test_pareto_distill_probe_uses_one_shared_statistic_and_two_components():
    """Would fail if Pareto recomputed a centre/covariance or reintroduced LOO."""

    rng = np.random.default_rng(92_913)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "1" * 64,
        "d81_spectral_weight_sha256": "2" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_eigenspectrum",
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))).astype(np.float32)
    fit, _call_records, transform_records = probe.build_d92_fit(
        d42, basis, weights, ground_audit, disable_registered_fisher=True,
        registered_d_mode="pareto_distill"
    )
    _, _, audit = fit(rows, labels, classes, shots)

    inventory = audit["d92_component_fit_inventory"]
    assert audit["d92_registered_d_mode_effective"] == "pareto_distill"
    assert inventory["actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 1
    assert inventory["loo_component_fit_count"] == 0
    assert audit["d92_pareto_distill_covariance_estimation_count"] == 1
    assert audit["d92_pareto_distill_robust_center_transform_count"] == 1
    assert len(transform_records) == 1
    assert audit["d92_pareto_distill_stage1_constraint_count"] == 7
    if audit["d92_pareto_distill_active"]:
        assert audit["d92_pareto_distill_deployment_codec_roundtrip_count"] == 2
        assert audit[
            "d92_pareto_distill_deployment_candidate_codec_roundtrip_count"
        ] == 1
        assert audit[
            "d92_pareto_distill_deployment_cross_group_margin_quantum"
        ] > 0.0
        assert (
            audit[
                "d92_pareto_distill_deployment_cross_group_margin_change_max_abs"
            ]
            >= audit[
                "d92_pareto_distill_deployment_cross_group_margin_quantum"
            ]
        )
        assert audit[
            "d92_pareto_distill_deployment_cross_group_quantum_pass"
        ] is True
    else:
        assert audit["d92_pareto_distill_fallback_active"] is True
        assert audit["d92_pareto_distill_deployment_codec_roundtrip_count"] in (1, 2)
        assert audit[
            "d92_pareto_distill_deployment_cross_group_quantum_pass"
        ] is False
    assert audit["d92_pareto_distill_query_rows_used"] == 0
    assert audit["d92_pareto_distill_query_fit_access"] is False
    assert audit["d92_pareto_distill_support_optimization_macs_upper_bound"] >= 0


def test_pareto_probe_numeric_codec_failure_is_exact_e0_and_structural_input_throws(monkeypatch):
    """Numerics fail closed; malformed support registry remains a hard error."""

    rng = np.random.default_rng(92_914)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "3" * 64,
        "d81_spectral_weight_sha256": "4" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_eigenspectrum",
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    rows = rng.normal(size=(classes * shots, 288)).astype(np.float32)
    full_fit, _, _ = probe.build_d92_fit(
        d42, basis, weights, ground_audit, disable_registered_fisher=True,
        registered_d_mode="full_only"
    )
    expected_w, expected_b, _ = full_fit(rows, labels, classes, shots)
    fit, _, _ = probe.build_d92_fit(
        d42, basis, weights, ground_audit, disable_registered_fisher=True,
        registered_d_mode="pareto_distill"
    )

    monkeypatch.setattr(d42, "_quantize_coefficients", lambda _value: (_ for _ in ()).throw(ValueError("injected codec")))
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    assert coefficient.tobytes() == expected_w.tobytes()
    assert intercept.tobytes() == expected_b.tobytes()
    assert audit["d92_pareto_distill_fallback_active"] is True
    assert audit["d92_pareto_distill_full_head_byte_exact"] is True
    assert audit[
        "d92_pareto_distill_deployment_cross_group_quantum_pass"
    ] is False

    broken_labels = labels.copy()
    broken_labels[-1] = 0
    with pytest.raises(probe.D92ProbeError, match="shared-statistics|registry"):
        fit(rows, broken_labels, classes, shots)


def test_pareto_block_numeric_fallback_records_decoded_e0_reference(monkeypatch):
    """Would fail if a legal BLOCK numeric fallback lacked its D42 E0 preview."""

    rng = np.random.default_rng(92_915)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "5" * 64,
        "d81_spectral_weight_sha256": "6" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_eigenspectrum",
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    rows = rng.normal(size=(classes * shots, 288)).astype(np.float32)
    original_compile = probe.compile_registration_balanced_affine

    def block_numeric_failure(d42_arg, statistics, *, arm):
        if arm == "block3_centered":
            raise probe.D92RegistrationBalancedCovarianceError(
                "positive definite injected block failure"
            )
        return original_compile(d42_arg, statistics, arm=arm)

    monkeypatch.setattr(
        probe, "compile_registration_balanced_affine", block_numeric_failure
    )
    fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="pareto_distill",
    )
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    decoded = d42._quantize_coefficients(coefficient)[4]
    decoded_intercept = np.asarray(intercept, dtype=np.float16).astype(np.float32)
    from cvsrffi import stage2_d92_full_block_pareto_distill as pareto

    assert audit["d92_pareto_distill_active"] is False
    assert audit["d92_pareto_distill_fallback_active"] is True
    assert audit["d92_pareto_distill_deployment_codec_roundtrip_count"] == 1
    assert audit["d92_pareto_distill_deployment_candidate_codec_roundtrip_count"] == 0
    assert audit["d92_pareto_distill_deployed_candidate_affine_sha256"] is None
    assert audit[
        "d92_pareto_distill_deployment_cross_group_quantum_pass"
    ] is False
    assert audit["d92_pareto_distill_deployed_e0_affine_sha256"] == (
        pareto.affine_preview_sha256(decoded, decoded_intercept)
    )


def _floorboost_hand_fixture():
    """Return a support set whose Q20 lower statistic is hand-checkable."""

    classes, shots, dimension = 8, 5, 8
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    rows = np.zeros((classes * shots, dimension), dtype=np.float32)
    all_margin_patterns = (
        (0.0, 4.0, 4.0, 4.0, 4.0),
        (1.0, 5.0, 5.0, 5.0, 5.0),
        (2.0, 4.0, 4.0, 4.0, 4.0),
        (3.0, 5.0, 5.0, 5.0, 5.0),
        (4.0, 5.0, 5.0, 5.0, 5.0),
        (5.0, 5.0, 5.0, 5.0, 5.0),
    )
    for old_class, margins in enumerate(all_margin_patterns):
        for shot, margin in enumerate(margins):
            row = rows[old_class * shots + shot]
            row[old_class] = 10.0
            row[(old_class + 1) % 6] = 5.0
            row[6] = 10.0 - margin
    for new_class in (6, 7):
        rows[new_class * shots : (new_class + 1) * shots, new_class] = 10.0
    coefficient = np.eye(classes, dimension, dtype=np.float32)
    intercept = np.zeros(classes, dtype=np.float32)
    return rows, labels, coefficient, intercept, classes, shots


def test_floorboost_uses_lower_q20_and_preserves_new_rows_and_old_bias_mean():
    """Would fail if floorboost interpolated Q20, moved new rows, or broke zero-sum bias."""

    rows, labels, coefficient, intercept, classes, shots = _floorboost_hand_fixture()
    output_coefficient, output_intercept, audit = probe._build_floorboost_affine_state(
        full_rows=rows,
        full_labels=labels,
        block_rows=rows,
        block_labels=labels,
        full_coefficient=coefficient,
        full_intercept=intercept,
        block_coefficient=coefficient,
        block_intercept=intercept,
        class_count=classes,
        k_shot=shots,
    )

    np.testing.assert_allclose(
        audit["d92_floorboost_retention_score_by_old_class"],
        [-1.8, 0.2, 0.6, 2.6, 3.8, 5.0],
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        audit["d92_floorboost_registration_drift_by_old_class"],
        [1.8, 0.8, 1.4, 0.4, 0.2, 0.0],
        rtol=0.0,
        atol=1.0e-6,
    )
    assert audit["d92_floorboost_quantile"] == pytest.approx(0.20)
    assert audit["d92_floorboost_quantile_method"] == "lower"
    assert audit["d92_floorboost_kappa"] == pytest.approx(0.35)
    assert audit["d92_floorboost_fallback_active"] is False
    assert output_coefficient[6:].tobytes() == coefficient[6:].tobytes()
    assert output_intercept[6:].tobytes() == intercept[6:].tobytes()
    np.testing.assert_allclose(output_coefficient[:6], coefficient[:6], rtol=0.0, atol=0.0)
    assert abs(float(np.sum(audit["d92_floorboost_delta_bias_by_old_class"]))) <= 1.0e-6
    assert max(abs(value) for value in audit["d92_floorboost_delta_bias_by_old_class"]) <= (
        0.35 * audit["d92_floorboost_full_old_rms"] + 1.0e-6
    )
    assert abs(float(output_intercept[:6].mean() - intercept[:6].mean())) <= 1.0e-6


def test_floorboost_ocf_numeric_degeneracy_fails_closed_to_full_head():
    """Would fail if an OCF RMS degeneration escaped or returned a hybrid head."""

    rows, labels, coefficient, intercept, classes, shots = _floorboost_hand_fixture()
    output_coefficient, output_intercept, audit = probe._build_floorboost_affine_state(
        full_rows=rows,
        full_labels=labels,
        block_rows=rows,
        block_labels=labels,
        full_coefficient=coefficient,
        full_intercept=intercept,
        block_coefficient=np.zeros_like(coefficient),
        block_intercept=np.zeros_like(intercept),
        class_count=classes,
        k_shot=shots,
    )

    assert output_coefficient.tobytes() == coefficient.tobytes()
    assert output_intercept.tobytes() == intercept.tobytes()
    assert audit["d92_floorboost_active"] is False
    assert audit["d92_floorboost_fallback_active"] is True
    assert audit["d92_floorboost_fallback_reason"] == (
        "ocf_old_class_centered_rms_degenerate"
    )
    assert audit["d92_floorboost_full_head_byte_exact"] is True
    assert audit["d92_floorboost_new_rows_byte_exact"] is True
    assert audit["d92_floorboost_delta_bias_by_old_class"] is None


def test_newguard_mode_calls_one_centered_full_component_and_strictly_falls_back():
    """The real D42 probe must keep one FULL fit and reject a failed closure."""

    rng = np.random.default_rng(92_711)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "9" * 64,
        "d81_spectral_weight_sha256": "a" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="newguard_maxmin",
    )
    _, _, audit = fit(rows, labels, classes, shots)

    inventory = audit["d92_component_fit_inventory"]
    assert inventory["actual_component_fit_count"] == 1
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert audit["d92_newguard_full_component_fit_count"] == 1
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_fallback_reason"] == "deployment_protection_failed"
    assert audit["d92_newguard_full_head_byte_exact"] is True
    assert audit["d92_newguard_deployment_strength_scale"] is None
    assert audit["d92_newguard_deployment_candidate_count"] == 1
    assert audit["d92_newguard_deployment_full_head_byte_exact"] is True
    assert audit["d92_newguard_deployment_codec_roundtrip_count"] == 2
    assert audit["d92_newguard_deployment_codec_macs_upper_bound"] > 0
    assert audit["d92_newguard_deployment_protection_pass"] is False
    assert (
        audit["d92_newguard_deployment_max_abs_Xnew_internal_residual"]
        > audit["d92_newguard_closure_tolerance"]
        or audit[
            "d92_newguard_deployment_new_support_old_envelope_change_max_abs_error"
        ]
        > audit["d92_newguard_closure_tolerance"]
    )
    assert audit["d92_newguard_query_rows_used"] == 0


def test_newguard_d42_codec_value_error_routes_to_exact_full_fallback(monkeypatch):
    """Would fail if a numerical D42 codec error escaped instead of restoring E0."""

    rng = np.random.default_rng(92_712)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "9" * 64,
        "d81_spectral_weight_sha256": "a" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    full_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    expected_coefficient, expected_intercept, _ = full_fit(
        rows, labels, classes, shots
    )
    newguard_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="newguard_maxmin",
    )

    def injected_codec_value_error(_coefficient: np.ndarray):
        raise ValueError("injected D42 coefficient codec failure")

    monkeypatch.setattr(d42, "_quantize_coefficients", injected_codec_value_error)
    coefficient, intercept, audit = newguard_fit(rows, labels, classes, shots)

    assert coefficient.tobytes() == expected_coefficient.tobytes()
    assert intercept.tobytes() == expected_intercept.tobytes()
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_full_head_byte_exact"] is True


def test_csoas_registered_path_consumes_the_same_d81_weights_once_for_one_full_fit():
    """Would fail if CSOAS rebuilt Cauchy weights, added a component fit, or skipped its receipt."""

    rng = np.random.default_rng(92_813)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    spectral_weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "b" * 64,
        "d81_spectral_weight_sha256": "c" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    fit, _, transform_records = probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="csoas_full",
    )

    coefficient, intercept, audit = fit(rows, labels, classes, shots)

    inventory = audit["d92_component_fit_inventory"]
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert audit["d92_registered_d_mode_effective"] == "csoas_full"
    assert audit["d92_csoas_active"] is True
    assert audit["d92_csoas_fallback_active"] is False
    assert audit["d92_csoas_candidate_attempt_fit_count"] == 1
    assert audit["d92_csoas_fallback_reference_fit_count"] == 0
    assert audit["d92_component_fit_count"] == 1
    assert inventory["actual_component_fit_count"] == 1
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert len(transform_records) == 1
    assert audit["d92_csoas_normalized_cauchy_weight_by_class"] == audit[
        "d81_transform_audit"
    ]["normalized_cauchy_weight_by_class"]


def test_csoas_numeric_failure_records_both_candidate_and_exact_e0_reference_full_fits(
    monkeypatch,
):
    """Would fail if a numeric CSOAS attempt disappeared from the fallback inventory."""

    rng = np.random.default_rng(92_814)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    spectral_weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "d" * 64,
        "d81_spectral_weight_sha256": "e" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    e0_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    expected_coefficient, expected_intercept, _ = e0_fit(
        rows, labels, classes, shots
    )
    csoas_fit, _, transform_records = probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="csoas_full",
    )

    def injected_numeric_failure(_statistics):
        raise D92CauchyScatterOASNumericalError("injected_csoas_numeric_failure")

    monkeypatch.setattr(
        probe, "compile_cauchy_scatter_oas_affine", injected_numeric_failure
    )
    coefficient, intercept, audit = csoas_fit(rows, labels, classes, shots)

    inventory = audit["d92_component_fit_inventory"]
    assert coefficient.tobytes() == expected_coefficient.tobytes()
    assert intercept.tobytes() == expected_intercept.tobytes()
    assert audit["d92_csoas_active"] is False
    assert audit["d92_csoas_fallback_active"] is True
    assert audit["d92_csoas_fallback_reason"] == "injected_csoas_numeric_failure"
    assert audit["d92_csoas_candidate_attempt_fit_count"] == 1
    assert audit["d92_csoas_fallback_reference_fit_count"] == 1
    assert audit["d92_csoas_fallback_reference_full_head_byte_exact"] is True
    assert audit["d92_csoas_paired_e0_codec_state_equal"] is None
    assert audit["d92_component_fit_count"] == 2
    assert inventory["actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 2
    assert inventory["block3_component_fit_count"] == 0
    assert len(transform_records) == 2


def test_ccoc_registered_path_uses_one_d81_transform_and_one_full_fit():
    """Would fail if CCOC were not a single support-only FULL route."""

    rng = np.random.default_rng(92_815)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    spectral_weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "f" * 64,
        "d81_spectral_weight_sha256": "1" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    fit, _, transform_records = probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="ccoc_full",
    )

    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    e0_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        spectral_weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    e0_coefficient, e0_intercept, _ = e0_fit(rows, labels, classes, shots)
    ccoc_state, _ = d42._compile_state(
        tuple(f"tx_{index}" for index in range(classes)),
        6,
        np.zeros(288, dtype=np.float32),
        coefficient,
        intercept,
        "sklearn_lsqr_auto_shrinkage_equal_prior",
        precision="int8",
    )
    e0_state, _ = d42._compile_state(
        tuple(f"tx_{index}" for index in range(classes)),
        6,
        np.zeros(288, dtype=np.float32),
        e0_coefficient,
        e0_intercept,
        "sklearn_lsqr_auto_shrinkage_equal_prior",
        precision="int8",
    )

    inventory = audit["d92_component_fit_inventory"]
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert audit["d92_registered_d_mode_effective"] == "ccoc_full"
    assert audit["d92_ccoc_active"] is True
    assert audit["d92_ccoc_fallback_active"] is False
    assert audit["d92_ccoc_full_endpoint_reused"] is True
    assert audit["d92_ccoc_additional_fit_count"] == 0
    assert audit["d92_ccoc_dense_solve_count"] == 1
    assert 0.0 <= audit["d92_ccoc_old_rho"] <= 1.0
    assert 0.0 <= audit["d92_ccoc_new_rho"] <= 1.0
    assert any(
        not np.array_equal(getattr(ccoc_state, field), getattr(e0_state, field))
        for field in ("coef1_qint8", "coef2_qint8", "scale1_fp16", "scale2_fp16")
    )
    assert audit["d92_component_fit_count"] == 1
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert len(transform_records) == 1
    for field in (
        "d92_ccoc_query_fit_access",
        "d92_ccoc_query_update_access",
        "d92_ccoc_query_selection_access",
        "d92_ccoc_query_truth_access",
        "d92_ccoc_query_role_oracle_access",
        "d92_ccoc_query_class_quota_access",
        "d92_ccoc_query_global_reassignment",
    ):
        assert audit[field] is False


@pytest.mark.parametrize("shots", (1, 2))
def test_ccoc_k1_k2_are_byte_exact_d92_full_aliases(shots):
    """Would fail if CCOC changed either frozen low-K D92 FULL alias."""

    rng = np.random.default_rng(92_816 + shots)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "2" * 64,
        "d81_spectral_weight_sha256": "3" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes = 11
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = means[labels].astype(np.float32)
    e0_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    ccoc_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="ccoc_full",
    )

    expected_coefficient, expected_intercept, _ = e0_fit(
        rows, labels, classes, shots
    )
    coefficient, intercept, audit = ccoc_fit(rows, labels, classes, shots)
    e0_state, _ = d42._compile_state(
        tuple(f"tx_{index}" for index in range(classes)),
        6,
        np.zeros(288, dtype=np.float32),
        expected_coefficient,
        expected_intercept,
        "sklearn_lsqr_auto_shrinkage_equal_prior",
        precision="int8",
    )
    ccoc_state, _ = d42._compile_state(
        tuple(f"tx_{index}" for index in range(classes)),
        6,
        np.zeros(288, dtype=np.float32),
        coefficient,
        intercept,
        "sklearn_lsqr_auto_shrinkage_equal_prior",
        precision="int8",
    )

    assert coefficient.tobytes() == expected_coefficient.tobytes()
    assert intercept.tobytes() == expected_intercept.tobytes()
    for field in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    ):
        assert getattr(ccoc_state, field).tobytes() == getattr(e0_state, field).tobytes()
    assert audit["d92_registered_d_mode_effective"] == "d92_full_alias"
    assert audit["d92_ccoc_active"] is False
    assert audit["d92_ccoc_fallback_active"] is False
    assert audit["d92_ccoc_fallback_reason"] == "k1_k2_exact_d81_fallback"
    assert audit["d92_ccoc_old_rho"] is None
    assert audit["d92_ccoc_new_rho"] is None


def test_ccoc_numeric_failure_records_candidate_and_exact_e0_reference_fit(monkeypatch):
    """Would fail if a real CCOC numerical attempt vanished from inventory."""

    rng = np.random.default_rng(92_817)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "4" * 64,
        "d81_spectral_weight_sha256": "5" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 5
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    means = rng.normal(size=(classes, 288))
    rows = (
        means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    ).astype(np.float32)
    e0_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    ccoc_fit, _, transform_records = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="ccoc_full",
    )
    expected_coefficient, expected_intercept, _ = e0_fit(
        rows, labels, classes, shots
    )

    def injected_numeric_failure(_d42, _statistics):
        raise D92CCOCNumericalError("injected_ccoc_numeric_failure")

    monkeypatch.setattr(
        probe,
        "compile_cross_class_offblock_consensus_affine",
        injected_numeric_failure,
        raising=False,
    )
    coefficient, intercept, audit = ccoc_fit(rows, labels, classes, shots)

    inventory = audit["d92_component_fit_inventory"]
    assert coefficient.tobytes() == expected_coefficient.tobytes()
    assert intercept.tobytes() == expected_intercept.tobytes()
    assert audit["d92_ccoc_active"] is False
    assert audit["d92_ccoc_fallback_active"] is True
    assert audit["d92_ccoc_fallback_reason"] == "injected_ccoc_numeric_failure"
    assert audit["d92_ccoc_candidate_attempt_fit_count"] == 1
    assert audit["d92_ccoc_fallback_reference_fit_count"] == 1
    assert audit["d92_ccoc_fallback_reference_full_head_byte_exact"] is True
    assert audit["d92_component_fit_count"] == 2
    assert inventory["actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 2
    assert len(transform_records) == 2


def test_ccoc_zero_offblock_core_error_falls_to_exact_e0_full_reference():
    """A real zero-Q core failure must retain the exact E0 fallback route."""

    rng = np.random.default_rng(92_818)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "6" * 64,
        "d81_spectral_weight_sha256": "7" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    classes, shots = 11, 3
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    rows = np.zeros((classes * shots, 288), dtype=np.float32)
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    for class_index in range(classes):
        selected = np.flatnonzero(labels == class_index)
        rows[selected, 3] = np.float32(10.0 * class_index)
        rows[selected, 0] = offsets
    e0_fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="full_only",
    )
    ccoc_fit, _, transform_records = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="ccoc_full",
    )

    expected_coefficient, expected_intercept, _ = e0_fit(
        rows, labels, classes, shots
    )
    coefficient, intercept, audit = ccoc_fit(rows, labels, classes, shots)

    assert coefficient.tobytes() == expected_coefficient.tobytes()
    assert intercept.tobytes() == expected_intercept.tobytes()
    assert audit["d92_ccoc_active"] is False
    assert audit["d92_ccoc_fallback_active"] is True
    assert audit["d92_ccoc_fallback_reason"] == "ccoc_q_zero_frobenius_norm"
    assert audit["d92_ccoc_candidate_statistic_receipt_available"] is False
    assert audit["d92_component_fit_count"] == 2
    assert len(transform_records) == 2


def test_ccoc_structural_core_error_is_not_relabelled_as_numeric_fallback(
    monkeypatch,
):
    """Class-registry drift must stay fail-closed instead of selecting E0."""

    rng = np.random.default_rng(92_819)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    ground_audit = {
        "d81_basis_sha256": "8" * 64,
        "d81_spectral_weight_sha256": "9" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }
    labels = np.repeat(np.arange(11), 3).astype(np.int64)
    rows = rng.normal(size=(len(labels), 288)).astype(np.float32)
    fit, _, _ = probe.build_d92_fit(
        d42,
        basis,
        weights,
        ground_audit,
        disable_registered_fisher=True,
        registered_d_mode="ccoc_full",
    )

    def structural_drift(*_args, **_kwargs):
        raise D92CCOCError("ccoc_unbalanced_group_registry")

    monkeypatch.setattr(
        probe,
        "build_cross_class_offblock_consensus_statistics",
        structural_drift,
    )
    with pytest.raises(probe.D92ProbeError, match="CCOC registry/receipt drift"):
        fit(rows, labels, 11, 3)
