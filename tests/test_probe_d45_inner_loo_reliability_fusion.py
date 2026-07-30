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
    / "probe_d45_inner_loo_reliability_fusion.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d45", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(seed: int = 4501, *, class_count: int = 4, k_shot: int = 4):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    for index in range(class_count):
        centers[index, (17 * index + 3) % d42.FEATURE_DIM] += np.float32(6.0)
    rows = []
    for index in range(class_count):
        for _ in range(k_shot):
            row = centers[index] + np.float32(0.2) * rng.normal(
                size=d42.FEATURE_DIM
            ).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
    rows = np.stack(rows).astype(np.float32)
    return rows, labels, class_count, k_shot


def test_class_balanced_ce_and_likelihood_weights_are_label_permutation_invariant() -> None:
    rng = np.random.default_rng(4502)
    scores = rng.normal(size=(20, 4))
    labels = np.repeat(np.arange(4), 5)
    ce, per_class = probe._class_balanced_cross_entropy(scores, labels, 4)
    permutation = np.asarray([2, 0, 3, 1])
    permuted_scores = scores[:, permutation]
    inverse = np.argsort(permutation)
    permuted_labels = inverse[labels]
    permuted_ce, permuted_per_class = probe._class_balanced_cross_entropy(
        permuted_scores, permuted_labels, 4
    )
    np.testing.assert_allclose(permuted_ce, ce, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.asarray(permuted_per_class)[inverse],
        per_class,
        rtol=0.0,
        atol=1.0e-15,
    )
    full_weight, block_weight, log_evidence = probe._likelihood_weights(
        ce, ce + 0.1, 4
    )
    assert full_weight > block_weight > 0.0
    np.testing.assert_allclose(full_weight + block_weight, 1.0)
    np.testing.assert_allclose(log_evidence, [-4 * ce, -4 * (ce + 0.1)])


def test_head_only_loo_partition_excludes_each_held_support_row_exactly_once() -> None:
    rows, labels, class_count, k_shot = _support(4505, class_count=3, k_shot=5)
    fit = probe._build_locked_d42_full_component_fit(d42)
    macro, per_class, audit = probe._inner_loo_component_ce(
        fit, rows, labels, class_count, k_shot
    )
    assert np.isfinite(macro)
    assert len(per_class) == class_count
    assert audit["partition_unit"] == "per_class_support_row_rank"
    assert audit["train_held_overlap_count"] == 0
    assert audit["held_support_row_exact_once_coverage"] is True
    assert audit["held_support_row_unique_count"] == len(rows)
    assert len(audit["held_ce_by_fold_and_class"]) == k_shot
    assert all(
        len(fold_values) == class_count
        for fold_values in audit["held_ce_by_fold_and_class"]
    )


def test_inner_loo_fused_fit_is_exact_weighted_normalized_affine_combination() -> None:
    rows, labels, class_count, k_shot = _support()
    full_fit = probe._build_locked_d42_full_component_fit(d42)
    block_fit = probe.d43.build_structured_fit(d42, "block3_centered")
    full_coef, full_intercept, _ = full_fit(rows, labels, class_count, k_shot)
    block_coef, block_intercept, _ = block_fit(rows, labels, class_count, k_shot)
    full_scale = probe.d44._class_centered_logit_rms(rows, full_coef, full_intercept)
    block_scale = probe.d44._class_centered_logit_rms(
        rows, block_coef, block_intercept
    )
    coef, intercept, audit = probe.build_inner_loo_reliability_fit(d42)(
        rows, labels, class_count, k_shot
    )
    expected_coef, expected_intercept = probe.d43._center_affine_scores(
        audit["d45_full_weight"] * full_coef.astype(np.float64) / full_scale
        + audit["d45_block_weight"] * block_coef.astype(np.float64) / block_scale,
        audit["d45_full_weight"]
        * full_intercept.astype(np.float64)
        / full_scale
        + audit["d45_block_weight"]
        * block_intercept.astype(np.float64)
        / block_scale,
    )
    np.testing.assert_array_equal(coef, expected_coef.astype(np.float32))
    np.testing.assert_array_equal(intercept, expected_intercept.astype(np.float32))
    np.testing.assert_allclose(
        audit["d45_full_weight"] + audit["d45_block_weight"], 1.0
    )
    assert audit["d45_inner_loo_fold_count"] == k_shot
    assert audit["d45_reliability_uses_outer_held_or_query"] is False
    assert audit["d45_role_handle_scene_specific_branch"] is False
    assert audit["d45_inner_scope"] == probe.INNER_SCOPE
    assert audit["d45_outer_b20_frozen_across_inner_folds"] is True
    assert audit["d45_inner_loo_generalization_claim_allowed"] is False


def test_k1_uses_equal_equivalent_unit_covariance_fallback() -> None:
    rows, labels, class_count, k_shot = _support(4503, k_shot=1)
    coef, intercept, audit = probe.build_inner_loo_reliability_fit(d42)(
        rows, labels, class_count, k_shot
    )
    assert np.isfinite(coef).all()
    assert np.isfinite(intercept).all()
    assert audit["d45_full_weight"] == 0.5
    assert audit["d45_block_weight"] == 0.5
    assert audit["d45_inner_loo_fold_count"] == 0
    assert audit["d45_k1_equivalent_unit_covariance_fallback"] is True
    assert audit["d45_full_inner_loo_macro_class_ce"] is None


def test_k2_inner_unit_covariance_evidence_is_exactly_equal_weight() -> None:
    rows, labels, class_count, k_shot = _support(4506, k_shot=2)
    def unit_component_fit(transformed, targets, class_count, k_shot):
        assert k_shot == 1
        means = np.stack(
            [transformed[targets == index].mean(axis=0) for index in range(class_count)]
        ).astype(np.float32)
        intercept = (
            -0.5 * np.sum(means.astype(np.float64) ** 2, axis=1)
            - np.log(class_count)
        ).astype(np.float32)
        return means, intercept, {}

    full_ce, _full_by_class, _full_audit = probe._inner_loo_component_ce(
        unit_component_fit, rows, labels, class_count, k_shot
    )
    block_ce, _block_by_class, _block_audit = probe._inner_loo_component_ce(
        unit_component_fit, rows, labels, class_count, k_shot
    )
    full_weight, block_weight, _log_evidence = probe._likelihood_weights(
        full_ce, block_ce, class_count
    )
    np.testing.assert_allclose(
        [full_weight, block_weight],
        [0.5, 0.5],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_d45_resource_accounting_counts_inner_and_main_component_fits() -> None:
    k_shot = 5
    old_rows, _old_targets, _old_count, _ = _support(
        4504, class_count=2, k_shot=k_shot
    )
    new_rows, _new_targets, _new_count, _ = _support(
        4507, class_count=2, k_shot=k_shot
    )
    old_labels = [value for value in ("old-a", "old-b") for _ in range(k_shot)]
    new_labels = [value for value in ("new-a", "new-b") for _ in range(k_shot)]
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    d42._fit_equal_prior_lda = probe.build_inner_loo_reliability_fit(d42)
    probe._install_d45_core_resource_accounting(d42)
    try:
        result = d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            old_labels,
            ["old-a", "old-b"],
            new_rows,
            new_labels,
            ["new-a", "new-b"],
            seed=4504,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top
    expected_fit_count = 4 + 4 * k_shot
    expected_macs = (
        2 * original_macs(2 * k_shot, 2)
        + 2 * k_shot * original_macs(2 * (k_shot - 1), 2)
        + 2 * original_macs(4 * k_shot, 4)
        + 2 * k_shot * original_macs(4 * (k_shot - 1), 4)
    )
    resource = result.resource_audit
    assert resource["lda_closed_form_fit_count"] == expected_fit_count
    assert resource["d45_inner_loo_component_fit_count"] == 4 * k_shot
    assert resource["estimated_lda_fit_macs"] == expected_macs
    assert resource["d45_lda_fit_inventory_macs"] == expected_macs
    assert sum(
        item["fit_count"] * item["macs_per_fit"]
        for item in resource["d45_lda_fit_inventory"]
    ) == expected_macs
    assert resource["estimated_adaptation_macs"] == (
        resource["estimated_metric_adaptation_macs"] + expected_macs
    )


def _partition_evidence(class_count: int, k_shot: int, ce: float):
    held = [
        [class_index * k_shot + rank for class_index in range(class_count)]
        for rank in range(k_shot)
    ]
    return {
        "partition_unit": "per_class_support_row_rank",
        "held_support_row_indices_by_fold": held,
        "held_ce_by_fold_and_class": [
            [ce for _ in range(class_count)] for _ in range(k_shot)
        ],
        "train_held_overlap_count": 0,
        "held_support_row_count": class_count * k_shot,
        "held_support_row_unique_count": class_count * k_shot,
        "held_support_row_exact_once_coverage": True,
        "train_rows_per_fold": class_count * (k_shot - 1),
        "held_rows_per_fold": class_count,
    }


def test_d45_records_fp32_centering_roundoff_when_ablation_policy_allows_it() -> None:
    rows = np.asarray([[0.6512653827667236]], dtype=np.float32)
    coefficients = np.asarray(
        [[85.9393081665039], [85.93941497802734], [85.93926239013672]],
        dtype=np.float32,
    )
    intercept = np.asarray(
        [-163.86297607421875, -163.863037109375, -163.8631134033203],
        dtype=np.float32,
    )

    class FakeD42:
        @staticmethod
        def _fit_equal_prior_lda(*_args):
            return coefficients, intercept, {}

    strict_fit = probe._build_locked_d42_full_component_fit(FakeD42)
    with pytest.raises(probe.D45ProbeError, match="centering drift"):
        strict_fit(rows, np.asarray([0]), 3, 1)
    allowed_fit = probe._build_locked_d42_full_component_fit(
        FakeD42,
        allow_fp32_centering_argmax_drift=True,
    )
    _coef, _intercept, audit = allowed_fit(
        rows, np.asarray([0]), 3, 1
    )
    assert (
        audit["d45_full_component_centered_support_fp64_argmax_equivalent"]
        is True
    )
    assert (
        audit["d45_full_component_centered_support_fp32_argmax_equivalent"]
        is False
    )
    assert (
        audit["d45_full_component_centered_support_fp32_argmax_changed_count"]
        == 1
    )
    assert (
        audit["d45_full_component_centered_support_fp32_argmax_drift_allowed"]
        is True
    )


def test_d45_rejects_fp64_algebraic_centering_drift_even_in_ablation_scope(
    monkeypatch,
) -> None:
    coefficients = np.asarray([[1.0], [0.0]], dtype=np.float32)
    intercept = np.zeros(2, dtype=np.float32)

    class FakeD42:
        @staticmethod
        def _fit_equal_prior_lda(*_args):
            return coefficients, intercept, {}

    monkeypatch.setattr(
        probe.d43,
        "_center_affine_scores",
        lambda *_args: (
            np.zeros((2, 1), dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        ),
    )
    fit = probe._build_locked_d42_full_component_fit(
        FakeD42,
        allow_fp32_centering_argmax_drift=True,
    )
    with pytest.raises(probe.D45ProbeError, match="algebraic centering drift"):
        fit(np.asarray([[1.0]], dtype=np.float32), np.asarray([0]), 2, 1)


def _valid_audit(class_count: int, k_shot: int):
    full_ce = 0.4
    block_ce = 0.5
    full_weight, block_weight, log_evidence = probe._likelihood_weights(
        full_ce, block_ce, class_count
    )
    return {
        "d45_probe_arm": probe.ARM,
        "d45_component_arms": ["full_centered_control", "block3_centered"],
        "d45_scale_formula": probe.d44.SCALE_FORMULA,
        "d45_weight_formula": probe.WEIGHT_FORMULA,
        "d45_log_evidence_formula": probe.LOG_EVIDENCE_FORMULA,
        "d45_inner_scope": probe.INNER_SCOPE,
        "d45_outer_b20_frozen_across_inner_folds": True,
        "d45_outer_b20_refit_per_inner_fold": False,
        "d45_inner_loo_generalization_claim_allowed": False,
        "d45_full_weight": full_weight,
        "d45_block_weight": block_weight,
        "d45_inner_loo_fold_count": k_shot,
        "d45_reliability_uses_support_labels": True,
        "d45_k1_equivalent_unit_covariance_fallback": False,
        "d45_reliability_uses_outer_held_or_query": False,
        "d45_role_handle_scene_specific_branch": False,
        "d45_weight_scan_count": 0,
        "d45_full_inner_loo_macro_class_ce": 0.4,
        "d45_block_inner_loo_macro_class_ce": 0.5,
        "d45_full_inner_loo_ce_by_class": [full_ce] * class_count,
        "d45_block_inner_loo_ce_by_class": [block_ce] * class_count,
        "d45_log_evidence_by_component": log_evidence,
        "d45_full_inner_partition_audit": _partition_evidence(
            class_count, k_shot, full_ce
        ),
        "d45_block_inner_partition_audit": _partition_evidence(
            class_count, k_shot, block_ce
        ),
    }


def _valid_rows() -> list[dict[str, object]]:
    k_shot = 8
    old_class_count = 3
    all_class_count = 5
    dimension = d42.FEATURE_DIM
    before_audit = _valid_audit(old_class_count, k_shot)
    final_audit = _valid_audit(all_class_count, k_shot)
    specs = [
        ("before_main_components", 2, old_class_count * k_shot, old_class_count),
        ("final_main_components", 2, all_class_count * k_shot, all_class_count),
        (
            "before_inner_head_only_components",
            2 * k_shot,
            old_class_count * (k_shot - 1),
            old_class_count,
        ),
        (
            "final_inner_head_only_components",
            2 * k_shot,
            all_class_count * (k_shot - 1),
            all_class_count,
        ),
    ]
    inventory = [
        {
            "fit_group": group,
            "fit_count": count,
            "row_count_per_fit": row_count,
            "class_count": class_count,
            "macs_per_fit": d42._lda_fit_macs(row_count, class_count),
        }
        for group, count, row_count, class_count in specs
    ]
    lda_macs = sum(item["fit_count"] * item["macs_per_fit"] for item in inventory)
    resource = {
        "old_k_shot": k_shot,
        "new_k_shot": k_shot,
        "registered_class_count": all_class_count,
        "coefficient_dimension": dimension,
        "lda_closed_form_fit_count": 36,
        "d45_component_main_fit_count": 4,
        "d45_inner_loo_component_fit_count": 32,
        "d45_fused_query_state_count": 1,
        "d45_inner_scope": probe.INNER_SCOPE,
        "d45_outer_b20_training_count": 1,
        "estimated_metric_adaptation_macs": 10,
        "estimated_lda_fit_macs": lda_macs,
        "estimated_adaptation_macs": 10 + lda_macs,
        "d45_lda_fit_inventory": inventory,
        "d45_lda_fit_inventory_macs": lda_macs,
    }
    rows = []
    for candidate_id in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
        for _ in range(15):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "geometry_summary": {
                        "before_covariance_audit": copy.deepcopy(before_audit),
                        "final_covariance_audit": copy.deepcopy(final_audit),
                        "before_materialized_pre_stage2c": True,
                        "before_state_immutable_during_stage2c": True,
                        "old_only_metric_new_support_argument_count": 0,
                    },
                    "resource": copy.deepcopy(resource),
                }
            )
    return rows


def test_d45_verifier_rejects_weight_or_resource_tampering() -> None:
    rows = _valid_rows()
    assert probe._verify_d45_fit_audits(rows) == 30
    rows[0]["geometry_summary"]["before_covariance_audit"]["d45_full_weight"] = 0.7
    with pytest.raises(probe.D45ProbeError, match="weight closure"):
        probe._verify_d45_fit_audits(rows)
    rows = _valid_rows()
    rows[-1]["resource"]["d45_inner_loo_component_fit_count"] = 0
    with pytest.raises(probe.D45ProbeError, match="inner_loo_component_fit_count"):
        probe._verify_d45_fit_audits(rows)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d45_log_evidence_by_component"
            ].__setitem__(0, -99.0),
            "log-evidence",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d45_full_inner_partition_audit"
            ]["held_ce_by_fold_and_class"][0].__setitem__(0, 0.9),
            "fold/class CE",
        ),
        (
            lambda rows: rows[0]["geometry_summary"]["before_covariance_audit"][
                "d45_full_inner_partition_audit"
            ]["held_support_row_indices_by_fold"][0].__setitem__(0, 1),
            "exact-once",
        ),
        (
            lambda rows: rows[0]["resource"]["d45_lda_fit_inventory"][0].__setitem__(
                "row_count_per_fit", 1
            ),
            "inventory drift",
        ),
    ],
)
def test_d45_verifier_recomputes_persisted_evidence(mutator, message) -> None:
    rows = _valid_rows()
    mutator(rows)
    with pytest.raises(probe.D45ProbeError, match=message):
        probe._verify_d45_fit_audits(rows)
