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
    / "probe_d46_classwise_loo_reliability_fusion.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d46", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(seed: int = 4601, *, class_count: int = 4, k_shot: int = 5):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    for index in range(class_count):
        centers[index, (19 * index + 5) % d42.FEATURE_DIM] += np.float32(6.0)
    rows = []
    for index in range(class_count):
        for _ in range(k_shot):
            row = centers[index] + np.float32(0.2) * rng.normal(
                size=d42.FEATURE_DIM
            ).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
    return np.stack(rows).astype(np.float32), labels, class_count, k_shot


def test_classwise_likelihood_weights_use_locked_per_class_product_formula() -> None:
    full_ce = np.asarray([0.2, 0.7, 0.4])
    block_ce = np.asarray([0.5, 0.3, 0.4])
    weights, log_evidence = probe._classwise_likelihood_weights(
        full_ce, block_ce, 5
    )
    expected_log = -5.0 * np.stack([full_ce, block_ce], axis=1)
    shifted = expected_log - expected_log.max(axis=1, keepdims=True)
    expected_weights = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(log_evidence, expected_log, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights, expected_weights, rtol=0.0, atol=1.0e-15)
    assert weights[0, 0] > 0.5
    assert weights[1, 0] < 0.5
    assert weights[2, 0] == 0.5


def test_classwise_likelihood_weights_are_stable_and_fail_closed() -> None:
    weights, log_evidence = probe._classwise_likelihood_weights(
        np.asarray([0.0, 100.0]), np.asarray([100.0, 0.0]), 5
    )
    assert np.isfinite(weights).all() and np.isfinite(log_evidence).all()
    assert weights[0, 0] > 1.0 - 1.0e-12
    assert weights[1, 1] > 1.0 - 1.0e-12
    with pytest.raises(probe.D46ProbeError, match="CE drift"):
        probe._classwise_likelihood_weights(
            np.asarray([0.2, np.nan]), np.asarray([0.3, 0.4]), 5
        )


def test_canonical_component_gauge_erases_arbitrary_common_affine() -> None:
    rows, labels, class_count, k_shot = _support(4606)
    rng = np.random.default_rng(4607)
    base_coef = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    base_intercept = rng.normal(size=class_count).astype(np.float32)
    common_coef = rng.normal(size=d42.FEATURE_DIM).astype(np.float32)
    common_intercept = np.float32(7.0)

    def base_fit(_rows, _targets, _class_count, _k_shot):
        return base_coef, base_intercept, {}

    def shifted_fit(_rows, _targets, _class_count, _k_shot):
        return (
            base_coef + common_coef[None, :],
            base_intercept + common_intercept,
            {},
        )

    base_checks = []
    shifted_checks = []
    base_state = probe._canonical_component_fit(base_fit, base_checks)(
        rows, labels, class_count, k_shot
    )
    shifted_state = probe._canonical_component_fit(shifted_fit, shifted_checks)(
        rows, labels, class_count, k_shot
    )
    np.testing.assert_allclose(base_state[0], shifted_state[0], rtol=0.0, atol=5e-7)
    np.testing.assert_allclose(base_state[1], shifted_state[1], rtol=0.0, atol=5e-7)
    assert base_checks[0]["coefficient_class_mean_max_abs"] <= 1.0e-7
    assert shifted_checks[0]["intercept_class_mean_abs"] <= 1.0e-7


def test_classwise_fused_fit_is_exact_and_label_permutation_equivariant() -> None:
    rows, labels, class_count, k_shot = _support()
    fit = probe.build_classwise_loo_reliability_fit(d42)
    coef, intercept, audit = fit(rows, labels, class_count, k_shot)
    full_fit = probe.d45._build_locked_d42_full_component_fit(d42)
    block_fit = probe.d43.build_structured_fit(d42, "block3_centered")
    full_coef, full_intercept, _ = full_fit(rows, labels, class_count, k_shot)
    block_coef, block_intercept, _ = block_fit(rows, labels, class_count, k_shot)
    full_coef, full_intercept = probe.d43._center_affine_scores(
        full_coef, full_intercept
    )
    block_coef, block_intercept = probe.d43._center_affine_scores(
        block_coef, block_intercept
    )
    full_scale = probe.d44._class_centered_logit_rms(
        rows, full_coef, full_intercept
    )
    block_scale = probe.d44._class_centered_logit_rms(
        rows, block_coef, block_intercept
    )
    full_weight = np.asarray(audit["d46_full_weight_by_class"])
    block_weight = np.asarray(audit["d46_block_weight_by_class"])
    expected_coef, expected_intercept = probe.d43._center_affine_scores(
        full_weight[:, None] * full_coef.astype(np.float64) / full_scale
        + block_weight[:, None] * block_coef.astype(np.float64) / block_scale,
        full_weight * full_intercept.astype(np.float64) / full_scale
        + block_weight * block_intercept.astype(np.float64) / block_scale,
    )
    np.testing.assert_array_equal(coef, expected_coef.astype(np.float32))
    np.testing.assert_array_equal(intercept, expected_intercept.astype(np.float32))
    assert audit["d46_class_id_specific_formula"] is False
    assert audit["d46_old_new_role_specific_branch"] is False
    assert audit["d46_scene_handle_specific_branch"] is False
    assert audit["d46_canonical_gauge"] == probe.CANONICAL_GAUGE
    assert len(audit["d46_full_component_canonical_gauge_checks"]) == k_shot + 1
    assert audit["d46_actual_inner_fold_count_used_as_likelihood_exponent"] == k_shot
    for partition_name in (
        "d46_full_inner_partition_audit",
        "d46_block_inner_partition_audit",
    ):
        partition = audit[partition_name]
        assert all(
            sorted(values) == list(range(class_count))
            for values in partition["d46_held_class_indices_by_fold"]
        )
        assert partition["d46_train_indices_are_exact_held_complements"] is True

    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted_coef, permuted_intercept, permuted_audit = fit(
        rows, permutation[labels], class_count, k_shot
    )
    np.testing.assert_array_equal(permuted_coef[permutation], coef)
    np.testing.assert_array_equal(permuted_intercept[permutation], intercept)
    np.testing.assert_allclose(
        np.asarray(permuted_audit["d46_full_weight_by_class"])[permutation],
        full_weight,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_k1_uses_exact_equal_classwise_fallback() -> None:
    rows, labels, class_count, k_shot = _support(4602, k_shot=1)
    coef, intercept, audit = probe.build_classwise_loo_reliability_fit(d42)(
        rows, labels, class_count, k_shot
    )
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    np.testing.assert_array_equal(
        audit["d46_full_weight_by_class"], [0.5] * class_count
    )
    np.testing.assert_array_equal(
        audit["d46_block_weight_by_class"], [0.5] * class_count
    )
    assert audit["d46_inner_loo_fold_count"] == 0
    assert audit["d46_full_inner_loo_ce_by_class"] is None
    assert audit["d46_k1_equivalent_unit_covariance_fallback"] is True
    assert audit["d46_actual_inner_fold_count_used_as_likelihood_exponent"] is None


def test_k2_equal_component_evidence_closes_to_exact_equal_weights() -> None:
    per_class_ce = np.asarray([0.2, 0.4, 0.8, 1.2], dtype=np.float64)
    weights, log_evidence = probe._classwise_likelihood_weights(
        per_class_ce, per_class_ce.copy(), 2
    )
    np.testing.assert_array_equal(weights, np.full((4, 2), 0.5))
    np.testing.assert_array_equal(
        log_evidence, -2.0 * np.stack([per_class_ce, per_class_ce], axis=1)
    )


def test_real_k2_full_and_block_components_close_to_equal_weights() -> None:
    rows, labels, class_count, k_shot = _support(4608, k_shot=1)
    duplicated_rows = np.repeat(rows, 2, axis=0)
    duplicated_labels = np.repeat(labels, 2)
    coef, intercept, audit = probe.build_classwise_loo_reliability_fit(d42)(
        duplicated_rows, duplicated_labels, class_count, 2
    )
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    np.testing.assert_allclose(
        audit["d46_full_inner_loo_ce_by_class"],
        audit["d46_block_inner_loo_ce_by_class"],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        audit["d46_full_weight_by_class"], [0.5] * class_count
    )


def _fit_result(k_shot: int = 5):
    old_rows, _old_targets, _old_count, _ = _support(
        4603, class_count=2, k_shot=k_shot
    )
    new_rows, _new_targets, _new_count, _ = _support(
        4604, class_count=2, k_shot=k_shot
    )
    old_labels = [value for value in ("old-a", "old-b") for _ in range(k_shot)]
    new_labels = [value for value in ("new-a", "new-b") for _ in range(k_shot)]
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    d42._fit_equal_prior_lda = probe.build_classwise_loo_reliability_fit(d42)
    probe._install_d46_resource_accounting(d42)
    try:
        return d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            old_labels,
            ["old-a", "old-b"],
            new_rows,
            new_labels,
            ["new-a", "new-b"],
            seed=4605,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top


def _valid_rows():
    result = _fit_result()
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


def test_resource_and_output_verifier_recompute_classwise_evidence() -> None:
    rows = _valid_rows()
    assert probe._verify_d46_fit_audits(rows) == 30
    resource = rows[0]["resource"]
    assert resource["lda_closed_form_fit_count"] == 24
    assert resource["d46_classwise_component_weight_count"] == 8
    assert resource["d46_fused_query_state_count"] == 1


def test_k1_resource_accounting_excludes_nonexistent_inner_scoring() -> None:
    result = _fit_result(k_shot=1)
    resource = result.resource_audit
    expected = 2 * d42.FEATURE_DIM * (2**2 + 4**2)
    assert resource["d46_estimated_reliability_scoring_macs"] == expected
    assert resource["lda_closed_form_fit_count"] == 4


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d46_full_weight_by_class"
            ].__setitem__(0, 0.99),
            "weight closure|CE/log-evidence/weight",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["final_covariance_audit"][
                "d46_log_evidence_by_class_and_component"
            ][0].__setitem__(0, -99.0),
            "CE/log-evidence/weight",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d46_full_inner_partition_audit"
            ]["held_support_row_indices_by_fold"][0].__setitem__(0, 1),
            "exact-once",
        ),
        (
            lambda rows: rows[0]["resource"]["d45_lda_fit_inventory"][0].__setitem__(
                "row_count_per_fit", 1
            ),
            "inventory drift",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d46_full_component_canonical_gauge_checks"
            ][0].__setitem__("coefficient_class_mean_max_abs", 1.0),
            "canonical gauge evidence drift",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["final_covariance_audit"].__setitem__(
                "d46_actual_inner_fold_count_used_as_likelihood_exponent", 4
            ),
            "likelihood exponent drift",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d46_full_inner_partition_audit"
            ]["d46_held_class_indices_by_fold"][0].__setitem__(0, 1),
            "per-fold per-class held partition drift",
        ),
    ],
)
def test_d46_verifier_rejects_tampering(mutator, message) -> None:
    rows = _valid_rows()
    mutator(rows)
    with pytest.raises((probe.D46ProbeError, probe.d45.D45ProbeError), match=message):
        probe._verify_d46_fit_audits(rows)
