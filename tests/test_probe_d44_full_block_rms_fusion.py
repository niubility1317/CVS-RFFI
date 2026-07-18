from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "probe_d44_full_block_rms_fusion.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d44", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(seed: int = 4401):
    rng = np.random.default_rng(seed)
    class_count = 4
    k_shot = 10
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(scale=0.3, size=(class_count, d42.FEATURE_DIM))
    rows = np.vstack(
        [centers[index] + rng.normal(scale=0.2, size=(k_shot, d42.FEATURE_DIM)) for index in range(class_count)]
    ).astype(np.float32)
    return rows, labels, class_count, k_shot


def test_class_centered_logit_rms_is_positive_and_common_score_invariant() -> None:
    rows, _labels, class_count, _k = _support()
    rng = np.random.default_rng(44)
    coef = rng.normal(size=(class_count, d42.FEATURE_DIM))
    intercept = rng.normal(size=class_count)
    scale = probe._class_centered_logit_rms(rows, coef, intercept)
    common_coef = rng.normal(size=d42.FEATURE_DIM)
    common_intercept = 17.0
    shifted = probe._class_centered_logit_rms(
        rows,
        coef + common_coef[None, :],
        intercept + common_intercept,
    )
    assert scale > 0.0
    np.testing.assert_allclose(shifted, scale, rtol=1.0e-12, atol=1.0e-12)


def test_fused_fit_is_exact_fixed_equal_normalized_affine_combination() -> None:
    rows, labels, class_count, k_shot = _support()
    full_fit = probe.d43.build_structured_fit(d42, "full_centered_control")
    block_fit = probe.d43.build_structured_fit(d42, "block3_centered")
    full_coef, full_intercept, _ = full_fit(rows, labels, class_count, k_shot)
    block_coef, block_intercept, _ = block_fit(rows, labels, class_count, k_shot)
    full_scale = probe._class_centered_logit_rms(rows, full_coef, full_intercept)
    block_scale = probe._class_centered_logit_rms(rows, block_coef, block_intercept)
    expected_coef, expected_intercept = probe.d43._center_affine_scores(
        0.5 * (full_coef.astype(np.float64) / full_scale + block_coef.astype(np.float64) / block_scale),
        0.5 * (
            full_intercept.astype(np.float64) / full_scale
            + block_intercept.astype(np.float64) / block_scale
        ),
    )
    coef, intercept, audit = probe.build_full_block_rms_fit(d42)(
        rows, labels, class_count, k_shot
    )
    np.testing.assert_array_equal(coef, expected_coef.astype(np.float32))
    np.testing.assert_array_equal(intercept, expected_intercept.astype(np.float32))
    assert audit["d44_full_weight"] == 0.5
    assert audit["d44_block_weight"] == 0.5
    assert audit["d44_weight_scan_count"] == 0
    assert audit["d44_scale_uses_labels_or_roles"] is False
    assert audit["d44_class_or_scenario_specific_branch"] is False
    assert audit["d43_covariance_structure"] == probe.STRUCTURE


def test_fused_fit_is_label_permutation_equivariant() -> None:
    rows, labels, class_count, k_shot = _support(4402)
    fit = probe.build_full_block_rms_fit(d42)
    coef, intercept, _ = fit(rows, labels, class_count, k_shot)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted_labels = permutation[labels]
    permuted_coef, permuted_intercept, _ = fit(
        rows, permuted_labels, class_count, k_shot
    )
    np.testing.assert_allclose(permuted_coef[permutation], coef, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        permuted_intercept[permutation], intercept, rtol=0.0, atol=0.0
    )


def test_d44_resource_accounting_counts_four_covariance_fits() -> None:
    rng = np.random.default_rng(4403)
    old_rows = rng.normal(size=(2, d42.FEATURE_DIM)).astype(np.float32)
    new_rows = rng.normal(size=(2, d42.FEATURE_DIM)).astype(np.float32)
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    d42._fit_equal_prior_lda = probe.build_full_block_rms_fit(d42)
    probe._install_d44_core_resource_accounting(d42)
    try:
        result = d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            ["old-a", "old-b"],
            ["old-a", "old-b"],
            new_rows,
            ["new-a", "new-b"],
            ["new-a", "new-b"],
            seed=4403,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top
    expected = 2 * (original_macs(2, 2) + original_macs(4, 4))
    assert result.resource_audit["lda_closed_form_fit_count"] == 4
    assert result.resource_audit["estimated_lda_fit_macs"] == expected
    assert result.resource_audit["d44_fused_query_state_count"] == 1


def _valid_d44_training_rows() -> list[dict[str, object]]:
    fit_audit = {
        "d44_probe_arm": probe.ARM,
        "d44_component_arms": [
            "full_centered_control",
            "block3_centered",
        ],
        "d44_scale_formula": probe.SCALE_FORMULA,
        "d44_scale_uses_labels_or_roles": False,
        "d44_full_support_logit_rms": 1.25,
        "d44_block_support_logit_rms": 0.75,
        "d44_full_weight": 0.5,
        "d44_block_weight": 0.5,
        "d44_weight_scan_count": 0,
        "d44_class_or_scenario_specific_branch": False,
    }
    resource = {
        "lda_closed_form_fit_count": 4,
        "d44_component_lda_fit_count_per_stage": 2,
        "d44_component_geometry_count": 2,
        "d44_fused_query_state_count": 1,
        "estimated_metric_adaptation_macs": 10,
        "estimated_lda_fit_macs": 20,
        "estimated_adaptation_macs": 30,
    }
    rows: list[dict[str, object]] = []
    for candidate_id in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
        for _ in range(15):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "geometry_summary": {
                        "before_covariance_audit": dict(fit_audit),
                        "final_covariance_audit": dict(fit_audit),
                    },
                    "resource": dict(resource),
                }
            )
    return rows


def test_d44_fit_audit_verifier_rejects_fusion_or_resource_tampering() -> None:
    rows = _valid_d44_training_rows()
    assert probe._verify_d44_fit_audits(rows) == 30
    rows[0]["geometry_summary"]["final_covariance_audit"][
        "d44_weight_scan_count"
    ] = 1
    with pytest.raises(probe.D44ProbeError, match="weight_scan_count"):
        probe._verify_d44_fit_audits(rows)
    rows = _valid_d44_training_rows()
    rows[-1]["resource"]["lda_closed_form_fit_count"] = 2
    with pytest.raises(probe.D44ProbeError, match="lda_closed_form_fit_count"):
        probe._verify_d44_fit_audits(rows)
