from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "probe_d47_anchored_reliability_shrinkage.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d47", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _partition(values: np.ndarray) -> dict:
    return {"held_ce_by_fold_and_class": np.asarray(values, dtype=float).tolist()}


def _strategy(full_fold: np.ndarray, block_fold: np.ndarray):
    k_shot, class_count = full_fold.shape
    return probe._anchored_reliability_strategy(
        full_per_class_ce=full_fold.mean(axis=0),
        block_per_class_ce=block_fold.mean(axis=0),
        full_partition=_partition(full_fold),
        block_partition=_partition(block_fold),
        k_shot=k_shot,
        class_count=class_count,
    )


def test_complete_pooling_uses_d45_c_mu_not_incorrect_k_mu() -> None:
    k_shot, class_count = 5, 3
    full = np.full((k_shot, class_count), 0.4)
    block = full + 0.2
    weights, _log_evidence, audit = _strategy(full, block)
    expected = probe._stable_sigmoid(np.asarray([class_count * 0.2]))[0]
    incorrect = probe._stable_sigmoid(np.asarray([k_shot * 0.2]))[0]
    np.testing.assert_allclose(weights[:, 0], expected, rtol=0.0, atol=1e-15)
    assert not np.isclose(expected, incorrect, rtol=0.0, atol=1e-6)
    assert audit["d47_tau_squared"] == 0.0
    assert audit["d47_d45_anchor_z0"] == pytest.approx(class_count * 0.2)
    assert audit["d47_zbar"] == pytest.approx(k_shot * 0.2)
    assert audit["d47_complete_pooling_d45_formula_weight_error"] == 0.0


def test_zero_within_variance_and_positive_heterogeneity_reaches_d46() -> None:
    advantage = np.asarray([0.1, -0.3, 0.6, 0.2])
    full = np.full((5, 4), 0.5)
    block = full + advantage[None, :]
    weights, _log_evidence, audit = _strategy(full, block)
    expected = probe._stable_sigmoid(5.0 * advantage)
    np.testing.assert_allclose(weights[:, 0], expected, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(audit["d47_shrink_factor_by_class"], [1.0] * 4)
    assert audit["d47_tau_squared"] > 0.0
    assert audit["d47_no_shrinkage_d46_max_abs_error"] <= 1e-15


def test_equal_nonzero_class_effects_do_not_create_false_heterogeneity() -> None:
    pattern = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])[:, None]
    full = np.full((5, 4), 0.6)
    block = full + 0.25 + pattern
    _weights, _log_evidence, audit = _strategy(full, block)
    assert audit["d47_between_observation_variance"] == pytest.approx(0.0, abs=1e-15)
    assert audit["d47_tau_squared"] == 0.0
    assert audit["d47_d45_anchor_z0"] != audit["d47_zbar"]


def test_partial_shrinkage_matches_hand_calculated_moments() -> None:
    full = np.zeros((3, 3), dtype=np.float64)
    block = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 4.0],
            [0.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    weights, _log_evidence, audit = _strategy(full, block)
    np.testing.assert_allclose(
        audit["d47_sample_variance_by_class"], [0.0, 1.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        audit["d47_within_log_odds_variance_proxy_by_class"],
        [0.0, 3.0, 0.0],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        audit["d47_observation_log_odds_by_class"], [0.0, 3.0, 12.0], atol=1e-15
    )
    assert audit["d47_between_observation_variance"] == pytest.approx(39.0)
    assert audit["d47_mean_within_log_odds_variance_proxy"] == pytest.approx(1.0)
    assert audit["d47_tau_squared"] == pytest.approx(38.0)
    np.testing.assert_allclose(
        audit["d47_shrink_factor_by_class"], [1.0, 38.0 / 41.0, 1.0], atol=1e-15
    )
    expected_post = np.asarray([0.0, 129.0 / 41.0, 12.0])
    np.testing.assert_allclose(
        audit["d47_post_log_odds_by_class"], expected_post, atol=1e-15
    )
    np.testing.assert_allclose(
        weights[:, 0], 1.0 / (1.0 + np.exp(-expected_post)), atol=1e-15
    )


def test_strategy_is_label_permutation_equivariant() -> None:
    rng = np.random.default_rng(4701)
    full = rng.uniform(0.1, 1.0, size=(5, 4))
    block = rng.uniform(0.1, 1.0, size=(5, 4))
    weights, _log, audit = _strategy(full, block)
    permutation = np.asarray([2, 0, 3, 1])
    permuted, _permuted_log, permuted_audit = _strategy(
        full[:, permutation], block[:, permutation]
    )
    np.testing.assert_allclose(permuted[:, 0], weights[permutation, 0], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        permuted_audit["d47_post_log_odds_by_class"],
        np.asarray(audit["d47_post_log_odds_by_class"])[permutation],
        rtol=0.0,
        atol=1e-12,
    )


def test_k1_k2_and_stable_sigmoid_boundaries() -> None:
    weights, log_evidence, audit = probe._anchored_reliability_strategy(
        full_per_class_ce=None,
        block_per_class_ce=None,
        full_partition=None,
        block_partition=None,
        k_shot=1,
        class_count=4,
    )
    np.testing.assert_array_equal(weights, np.full((4, 2), 0.5))
    assert log_evidence is None
    assert audit["d47_boundary_status"] == "k1_equal_unit_fallback"
    zero = np.zeros((2, 4))
    weights, _log_evidence, audit = _strategy(zero, zero)
    np.testing.assert_array_equal(weights, np.full((4, 2), 0.5))
    assert audit["d47_boundary_status"] == "k2_equal_component_fallback"
    with pytest.raises(probe.D47ProbeError, match="K2 component-equivalence"):
        _strategy(zero, zero + 0.1)
    stable = probe._stable_sigmoid(np.asarray([-30.0, 0.0, 30.0]))
    assert np.isfinite(stable).all() and np.all(stable > 0.0) and np.all(stable < 1.0)
    with pytest.raises(probe.D47ProbeError, match="underflow/overflow"):
        probe._stable_sigmoid(np.asarray([-1000.0, 1000.0]))
    with pytest.raises(probe.D47ProbeError, match="non-finite"):
        probe._stable_sigmoid(np.asarray([np.nan]))


@pytest.mark.parametrize(
    ("k_shot", "expected"),
    [(1, 0), (2, 644), (5, 950), (8, 1256), (10, 1460), (20, 2480)],
)
def test_scalar_operation_upper_bound_locked_values(k_shot: int, expected: int) -> None:
    assert probe._scalar_operation_upper_bound(k_shot, 6, 11) == expected
    assert probe.SCALAR_OPERATION_UPPER_BOUND_DERIVATION == {
        "counting_unit": "conservative_scalar_MAC_equivalent",
        "fold_evidence_and_first_second_moments_per_state": "6*K*C",
        "cross_class_moments_and_positive_part_shrinkage_per_state": "16*C+8",
        "post_logit_sigmoid_and_endpoint_checks_per_state": "8*C+8",
        "two_state_total": "6*K*(C_old+C_all)+24*(C_old+C_all)+32",
        "valid_integer_k_domain": "K>=2;K1_executes_no_moment_algebra_and_is_zero",
    }


def _support(seed: int, class_count: int, k_shot: int):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    for index in range(class_count):
        centers[index, (23 * index + 7) % d42.FEATURE_DIM] += np.float32(6.0)
    rows = []
    for index in range(class_count):
        for _ in range(k_shot):
            row = centers[index] + np.float32(0.2) * rng.normal(
                size=d42.FEATURE_DIM
            ).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
    return np.stack(rows), labels


def _run_fit(old_rows, old_labels, new_rows, new_labels):
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    d42._fit_equal_prior_lda = probe.build_anchored_reliability_fit(d42)
    probe._install_d47_resource_accounting(d42)
    try:
        return d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            old_labels,
            ["old-a", "old-b"],
            new_rows,
            new_labels,
            ["new-a", "new-b"],
            seed=4704,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top


def _fit_result(k_shot: int = 5):
    old_rows, _ = _support(4702, 2, k_shot)
    new_rows, _ = _support(4703, 2, k_shot)
    old_labels = [value for value in ("old-a", "old-b") for _ in range(k_shot)]
    new_labels = [value for value in ("new-a", "new-b") for _ in range(k_shot)]
    return _run_fit(old_rows, old_labels, new_rows, new_labels)


def _rows_for_result(result):
    rows = []
    for candidate_id in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
        for _ in range(15):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "geometry_summary": copy.deepcopy(result.geometry_audit),
                    "resource": copy.deepcopy(result.resource_audit),
                }
            )
    return rows


def _valid_rows(k_shot: int = 5):
    return _rows_for_result(_fit_result(k_shot))


def test_integrated_fit_resource_and_verifier_close() -> None:
    rows = _valid_rows()
    assert probe._verify_d47_fit_audits(rows) == 30
    audit = rows[0]["geometry_summary"]["final_covariance_audit"]
    assert audit["d47_probe_arm"] == probe.ARM
    assert audit["d47_statistical_claim"] == probe.STATISTICAL_CLAIM
    resource = rows[0]["resource"]
    assert resource["lda_closed_form_fit_count"] == 24
    assert resource["d47_additional_lda_fit_count"] == 0
    expected_upper = 356
    assert probe._scalar_operation_upper_bound(5, 2, 4) == expected_upper
    assert resource["d47_estimated_scalar_operation_upper_bound"] == expected_upper
    assert resource["d47_additional_adaptation_mac_equivalents"] == expected_upper
    assert resource["estimated_adaptation_macs"] == (
        resource["estimated_metric_adaptation_macs"]
        + resource["estimated_lda_fit_macs"]
        + resource["d46_estimated_reliability_scoring_macs"]
        + resource["d46_estimated_classwise_affine_fusion_macs"]
        + expected_upper
    )


def test_k1_integrated_fit_resource_and_verifier_close() -> None:
    rows = _valid_rows(k_shot=1)
    assert probe._verify_d47_fit_audits(rows) == 30
    resource = rows[0]["resource"]
    assert resource["lda_closed_form_fit_count"] == 4
    assert resource["d47_estimated_scalar_operation_upper_bound"] == 0
    for field in ("before_covariance_audit", "final_covariance_audit"):
        audit = rows[0]["geometry_summary"][field]
        assert audit["d47_boundary_status"] == "k1_equal_unit_fallback"


def test_real_k2_duplicate_support_closes_full_chain() -> None:
    old_rows, _ = _support(4705, 2, 1)
    new_rows, _ = _support(4706, 2, 1)
    result = _run_fit(
        np.repeat(old_rows, 2, axis=0),
        ["old-a", "old-a", "old-b", "old-b"],
        np.repeat(new_rows, 2, axis=0),
        ["new-a", "new-a", "new-b", "new-b"],
    )
    rows = _rows_for_result(result)
    assert probe._verify_d47_fit_audits(rows) == 30
    for field in ("before_covariance_audit", "final_covariance_audit"):
        audit = rows[0]["geometry_summary"][field]
        assert audit["d47_boundary_status"] == "k2_equal_component_fallback"
        np.testing.assert_array_equal(
            audit["d47_full_weight_by_class"],
            [0.5] * len(audit["d47_full_weight_by_class"]),
        )


def test_verifier_rejects_anchored_evidence_tampering() -> None:
    rows = _valid_rows()
    rows[0]["geometry_summary"]["final_covariance_audit"][
        "d47_d45_anchor_z0"
    ] += 1.0
    with pytest.raises(probe.D47ProbeError, match="anchored evidence closure"):
        probe._verify_d47_fit_audits(rows)


@pytest.mark.parametrize(
    "field",
    [
        "d43_probe_arm",
        "d43_covariance_structure",
        "d46_probe_arm",
        "d46_weight_formula",
        "d46_log_evidence_formula",
    ],
)
def test_verifier_rejects_pre_sanitize_core_field_tampering(field: str) -> None:
    rows = _valid_rows()
    rows[0]["geometry_summary"]["before_covariance_audit"][field] = "tampered"
    with pytest.raises(probe.D47ProbeError, match="exact audit drift"):
        probe._verify_d47_fit_audits(rows)
