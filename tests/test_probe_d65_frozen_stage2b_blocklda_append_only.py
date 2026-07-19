from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d65_frozen_stage2b_blocklda_append_only.py"
SPEC = importlib.util.spec_from_file_location("probe_d65_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d65 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d65)


def _stub() -> SimpleNamespace:
    return SimpleNamespace(
        FEATURE_DIM=9,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=(slice(0, 3), slice(3, 6), slice(6, 9)),
    )


def _support(classes: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=2.0, size=(classes, 9))
    rows = np.concatenate(
        [means[index] + rng.normal(scale=0.25, size=(k, 9)) for index in range(classes)]
    )
    return rows, np.repeat(np.arange(classes), k)


def _before_final(k: int = 4):
    old_rows, old_labels = _support(3, k, 65)
    new_rows, new_labels = _support(2, k, 650)
    all_rows = np.concatenate([old_rows, new_rows])
    all_labels = np.concatenate([old_labels, new_labels + 3])
    return old_rows, old_labels, all_rows, all_labels


def test_append_only_rows_are_bitwise_unchanged() -> None:
    old_rows, old_labels, all_rows, all_labels = _before_final()
    fit, records, lifecycle = d65.build_d65_fit(_stub())
    before_coef, before_bias, before_audit = fit(old_rows, old_labels, 3, 4)
    final_coef, final_bias, final_audit = fit(all_rows, all_labels, 5, 4)
    assert np.array_equal(before_coef, final_coef[:3])
    assert np.array_equal(before_bias, final_bias[:3])
    assert before_audit["d65_phase"] == "stage2b_fit_and_freeze"
    assert final_audit["d65_phase"] == "stage2c_append_only"
    assert final_audit["d65_old_row_fp32_bitwise_unchanged"] is True
    assert before_audit["d65_stage2b_covariance_sha256"] == final_audit["d65_stage2b_covariance_sha256"]
    assert len(records) == 2 and lifecycle["completed_pairs"] == 1
    assert lifecycle["pending"] is None


def test_new_support_changes_only_appended_rows() -> None:
    old_rows, old_labels, all_rows, all_labels = _before_final()
    first, _, _ = d65.build_d65_fit(_stub())
    before1, bias1, _ = first(old_rows, old_labels, 3, 4)
    final1, final_bias1, _ = first(all_rows, all_labels, 5, 4)
    altered = all_rows.copy()
    altered[12:] += 4.0
    second, _, _ = d65.build_d65_fit(_stub())
    before2, bias2, _ = second(old_rows, old_labels, 3, 4)
    final2, final_bias2, _ = second(altered, all_labels, 5, 4)
    assert np.array_equal(before1, before2) and np.array_equal(bias1, bias2)
    assert np.array_equal(final1[:3], final2[:3])
    assert np.array_equal(final_bias1[:3], final_bias2[:3])
    assert not np.array_equal(final1[3:], final2[3:])


def test_support_predictions_and_class_permutation_equivariance() -> None:
    old_rows, old_labels, all_rows, all_labels = _before_final()
    fit1, _, _ = d65.build_d65_fit(_stub())
    _, _, _ = fit1(old_rows, old_labels, 3, 4)
    coef1, bias1, audit1 = fit1(all_rows, all_labels, 5, 4)
    old_perm = np.asarray([2, 0, 1])
    new_perm = np.asarray([1, 0])
    permuted = np.concatenate([old_perm, new_perm + 3])
    labels2 = permuted[all_labels]
    fit2, _, _ = d65.build_d65_fit(_stub())
    coef_before2, bias_before2, _ = fit2(old_rows, old_perm[old_labels], 3, 4)
    coef2, bias2, audit2 = fit2(all_rows, labels2, 5, 4)
    np.testing.assert_allclose(coef2[permuted], coef1, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(bias2[permuted], bias1, rtol=2e-5, atol=2e-5)
    assert np.array_equal(coef_before2[old_perm], coef1[:3])
    assert audit1["d65_compiled_support_accuracy"] >= 0.9
    assert audit2["d65_compiled_support_accuracy"] >= 0.9


def test_k1_uses_frozen_unit_covariance() -> None:
    old_rows, old_labels = _support(3, 1, 651)
    new_rows, new_labels = _support(2, 1, 652)
    fit, _, lifecycle = d65.build_d65_fit(_stub())
    before, _, before_audit = fit(old_rows, old_labels, 3, 1)
    final, _, final_audit = fit(
        np.concatenate([old_rows, new_rows]),
        np.concatenate([old_labels, new_labels + 3]),
        5,
        1,
    )
    assert before_audit["unit_covariance_fallback"] is True
    assert final_audit["unit_covariance_fallback"] is True
    assert np.array_equal(before, final[:3])
    assert lifecycle["pending"] is None


def test_rejects_non_symmetric_nonfinite_and_lifecycle_drift() -> None:
    rows, labels = _support(3, 3, 653)
    with pytest.raises(d65.D65ProbeError, match="symmetric"):
        d65._validate_symmetric_support(rows[:-1], labels[:-1], 3, 3)
    rows[0, 0] = np.nan
    with pytest.raises(d65.D65ProbeError, match="finite"):
        d65._validate_symmetric_support(rows, labels, 3, 3)
    fit, _, _ = d65.build_d65_fit(_stub())
    clean, clean_labels = _support(3, 3, 654)
    fit(clean, clean_labels, 3, 3)
    wrong, wrong_labels = _support(4, 2, 655)
    with pytest.raises(d65.D65ProbeError, match="lifecycle"):
        fit(wrong, wrong_labels, 4, 2)


def test_formula_and_audit_forbid_query_role_and_tunable_freeze() -> None:
    lowered = d65.FORMULA.lower()
    assert "freeze" in lowered and "append" in lowered
    for forbidden in ("role", "scene", "threshold", "alpha", "temperature"):
        assert forbidden not in lowered
    old_rows, old_labels, all_rows, all_labels = _before_final()
    fit, _, _ = d65.build_d65_fit(_stub())
    _, _, before = fit(old_rows, old_labels, 3, 4)
    _, _, final = fit(all_rows, all_labels, 5, 4)
    for audit in (before, final):
        assert audit["d65_class_id_specific_formula"] is False
        assert audit["d65_old_new_role_specific_query_branch"] is False
        assert audit["d65_uses_outer_held_or_query"] is False
        assert audit["d65_hyperparameter_count"] == 0


def test_state_old_row_comparison_covers_int8_and_fp32() -> None:
    before_int8 = SimpleNamespace(
        classes=("a", "b"),
        log_diag_fp32=np.ones(2, dtype=np.float32),
        coef1_qint8=np.ones((2, 2), dtype=np.int8),
        coef2_qint8=np.ones((2, 2), dtype=np.int8),
        scale1_fp16=np.ones((2, 1), dtype=np.float16),
        scale2_fp16=np.ones((2, 1), dtype=np.float16),
        intercept_fp16=np.ones(2, dtype=np.float16),
        coef_fp32=np.zeros((0, 2), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
    )
    final_int8 = SimpleNamespace(
        **{
            **before_int8.__dict__,
            "classes": ("a", "b", "c"),
            "coef1_qint8": np.ones((3, 2), dtype=np.int8),
            "coef2_qint8": np.ones((3, 2), dtype=np.int8),
            "scale1_fp16": np.ones((3, 1), dtype=np.float16),
            "scale2_fp16": np.ones((3, 1), dtype=np.float16),
            "intercept_fp16": np.ones(3, dtype=np.float16),
        }
    )
    assert d65._state_old_rows_equal(before_int8, final_int8)
    final_int8.coef1_qint8[0, 0] = 2
    assert not d65._state_old_rows_equal(before_int8, final_int8)


def test_real_d42_compiles_bitwise_append_only_states() -> None:
    import cvsrffi.stage2_d42_unified_shrinkage_lda as d42

    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    def real_support(seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(2, 288)).astype(np.float32)
        centers[0, 0] += 5.0
        centers[1, 17] += 5.0
        rows = np.concatenate(
            [
                centers[index]
                    + np.float32(0.2) * rng.normal(size=(k, 288)).astype(np.float32)
                for index in range(2)
            ]
        ).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        return rows, np.repeat(np.arange(2), k)

    old_rows, old_targets = real_support(656)
    new_rows, new_targets = real_support(657)
    old_labels = np.asarray([old_classes[index] for index in old_targets])
    new_labels = np.asarray([new_classes[index] for index in new_targets])
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    try:
        fit, records, lifecycle = d65.build_d65_fit(d42)
        d42._fit_equal_prior_lda = fit
        d65._install_resource_accounting(d42)
        result = d42.fit_d42_unified_shrinkage_lda(
            old_rows.astype(np.float32),
            old_labels,
            old_classes,
            new_rows.astype(np.float32),
            new_labels,
            new_classes,
            seed=658,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top
    assert d65._state_old_rows_equal(result.before_state, result.state)
    assert d65._state_old_rows_equal(
        result.matched_fp32_before_state, result.matched_fp32_state
    )
    assert result.resource_audit["d65_int8_old_rows_bitwise_unchanged"] is True
    assert result.resource_audit["d65_fp32_old_rows_bitwise_unchanged"] is True
    assert len(records) == 2 and lifecycle["completed_pairs"] == 1
