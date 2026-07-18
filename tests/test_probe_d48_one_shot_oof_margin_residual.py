from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "probe_d48_one_shot_oof_margin_residual.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d48", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _partition(k_shot: int, class_count: int) -> dict:
    held = [
        [class_index * k_shot + rank for class_index in range(class_count)]
        for rank in range(k_shot)
    ]
    all_indices = set(range(k_shot * class_count))
    return {
        "held_support_row_indices_by_fold": held,
        "private_collector_held_class_indices_by_fold": [
            list(range(class_count)) for _ in range(k_shot)
        ],
        "private_collector_train_support_row_indices_by_fold": [
            sorted(all_indices - set(fold)) for fold in held
        ],
        "private_collector_train_indices_are_exact_held_complements": True,
    }


def _strategy(full: np.ndarray, block: np.ndarray, full_weight: float = 0.5):
    k_shot, class_count, logits = full.shape
    assert logits == class_count
    partition = _partition(k_shot, class_count)
    return probe._one_shot_margin_residual(
        full_held_scores=full,
        block_held_scores=block,
        full_weight=full_weight,
        block_weight=1.0 - full_weight,
        full_partition=partition,
        block_partition=copy.deepcopy(partition),
        k_shot=k_shot,
        class_count=class_count,
    )


def _diagonal_margin_logits(margins: np.ndarray, k_shot: int) -> np.ndarray:
    values = np.zeros((k_shot, len(margins), len(margins)), dtype=np.float64)
    for class_index, margin in enumerate(margins):
        values[:, class_index, class_index] = margin
    return values


def test_hand_calculated_one_shot_mean_margin_residual() -> None:
    logits = _diagonal_margin_logits(np.asarray([1.0, 0.0, 2.0]), 3)
    beta, audit = _strategy(logits, logits)
    np.testing.assert_array_equal(audit["d48_mean_margin_by_class"], [1.0, 0.0, 2.0])
    assert audit["d48_global_mean_margin"] == 1.0
    np.testing.assert_array_equal(beta, [0.0, 1.0, -1.0])
    assert audit["d48_beta_sum_residual"] == 0.0
    assert audit["d48_beta_nonzero_class_count"] == 2


def test_heterogeneous_components_and_nonhalf_weight_match_hand_calculation() -> None:
    full = _diagonal_margin_logits(np.asarray([1.0, 0.0, 2.0]), 3)
    block = _diagonal_margin_logits(np.asarray([3.0, 2.0, 0.0]), 3)
    beta, audit = _strategy(full, block, full_weight=0.25)
    expected_fused = 0.25 * full + 0.75 * block
    np.testing.assert_allclose(
        audit["d48_fused_held_logits_by_fold_class_logit"],
        expected_fused,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        audit["d48_mean_margin_by_class"], [2.5, 1.5, 0.5], atol=1e-15
    )
    np.testing.assert_allclose(beta, [-1.0, 0.0, 1.0], atol=1e-15)


def test_equal_margins_are_exact_zero_bias() -> None:
    logits = _diagonal_margin_logits(np.asarray([0.7, 0.7, 0.7, 0.7]), 5)
    beta, audit = _strategy(logits, logits, full_weight=0.63)
    np.testing.assert_array_equal(beta, np.zeros(4))
    assert audit["d48_beta_max_abs"] == 0.0


def test_label_and_rank_permutation_equivariance() -> None:
    rng = np.random.default_rng(4801)
    full = rng.normal(size=(5, 4, 4))
    block = rng.normal(size=(5, 4, 4))
    beta, audit = _strategy(full, block, full_weight=0.61)
    permutation = np.asarray([2, 0, 3, 1])
    permuted_beta, permuted_audit = _strategy(
        full[:, permutation][:, :, permutation],
        block[:, permutation][:, :, permutation],
        full_weight=0.61,
    )
    np.testing.assert_allclose(permuted_beta, beta[permutation], atol=1e-15)
    np.testing.assert_allclose(
        permuted_audit["d48_mean_margin_by_class"],
        np.asarray(audit["d48_mean_margin_by_class"])[permutation],
        atol=1e-15,
    )
    rank_permutation = np.asarray([4, 1, 3, 0, 2])
    rank_beta, _ = _strategy(
        full[rank_permutation], block[rank_permutation], full_weight=0.61
    )
    np.testing.assert_allclose(rank_beta, beta, atol=1e-15)


def test_common_affine_held_logit_shift_does_not_change_margin_or_beta() -> None:
    rng = np.random.default_rng(4802)
    full = rng.normal(size=(5, 4, 4))
    block = rng.normal(size=(5, 4, 4))
    beta, audit = _strategy(full, block, full_weight=0.44)
    shift = rng.normal(size=(5, 4, 1))
    shifted_beta, shifted_audit = _strategy(
        full + shift, block + shift, full_weight=0.44
    )
    np.testing.assert_allclose(shifted_beta, beta, atol=1e-15)
    np.testing.assert_allclose(
        shifted_audit["d48_margin_by_fold_class"],
        audit["d48_margin_by_fold_class"],
        atol=1e-15,
    )


def test_k1_and_invalid_evidence_fail_close() -> None:
    beta, audit = probe._one_shot_margin_residual(
        full_held_scores=None,
        block_held_scores=None,
        full_weight=0.5,
        block_weight=0.5,
        full_partition=None,
        block_partition=None,
        k_shot=1,
        class_count=3,
    )
    np.testing.assert_array_equal(beta, np.zeros(3))
    assert audit["d48_boundary_status"] == "k1_exact_d45_zero_bias_fallback"
    with pytest.raises(probe.D48ProbeError, match="C>=2"):
        probe._one_shot_margin_residual(
            full_held_scores=None,
            block_held_scores=None,
            full_weight=0.5,
            block_weight=0.5,
            full_partition=None,
            block_partition=None,
            k_shot=1,
            class_count=1,
        )
    bad = _diagonal_margin_logits(np.asarray([1.0, 2.0, 3.0]), 3)
    bad[0, 0, 0] = np.nan
    with pytest.raises(probe.D48ProbeError, match="held-logit"):
        _strategy(bad, np.nan_to_num(bad))


@pytest.mark.parametrize(
    ("k_shot", "expected"),
    [(1, 0), (2, 1832), (5, 4124), (8, 6416), (10, 7944), (20, 15584)],
)
def test_operation_upper_bound_locked_values(k_shot: int, expected: int) -> None:
    assert probe._operation_upper_bound(k_shot, 6, 11) == expected


def _support(seed: int, class_count: int, k_shot: int):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    for index in range(class_count):
        centers[index, (29 * index + 11) % d42.FEATURE_DIM] += np.float32(5.0)
    rows = []
    for index in range(class_count):
        for rank in range(k_shot):
            scale = np.float32(0.12 + 0.03 * index + 0.01 * rank)
            row = centers[index] + scale * rng.normal(size=d42.FEATURE_DIM).astype(
                np.float32
            )
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
    return np.stack(rows), labels


def test_real_collector_label_and_rank_permutation_chain() -> None:
    rows, labels = _support(4806, 4, 5)
    fit = probe.build_one_shot_margin_residual_fit(d42)
    coef, intercept, audit = fit(rows, labels, 4, 5)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted_coef, permuted_intercept, permuted_audit = fit(
        rows, permutation[labels], 4, 5
    )
    np.testing.assert_allclose(permuted_coef[permutation], coef, atol=2e-6)
    np.testing.assert_allclose(permuted_intercept[permutation], intercept, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(permuted_audit["d48_beta_centered_by_class"])[permutation],
        audit["d48_beta_centered_by_class"],
        atol=2e-7,
    )
    rank_permutation = np.asarray([4, 1, 3, 0, 2], dtype=np.int64)
    order = np.concatenate(
        [5 * class_index + rank_permutation for class_index in range(4)]
    )
    rank_coef, rank_intercept, rank_audit = fit(rows[order], labels[order], 4, 5)
    np.testing.assert_allclose(rank_coef, coef, atol=2e-6)
    np.testing.assert_allclose(rank_intercept, intercept, atol=2e-6)
    np.testing.assert_allclose(
        rank_audit["d48_beta_centered_by_class"],
        audit["d48_beta_centered_by_class"],
        atol=2e-7,
    )


def _run_fit(old_rows, old_labels, new_rows, new_labels, *, d48: bool):
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    if d48:
        d42._fit_equal_prior_lda = probe.build_one_shot_margin_residual_fit(d42)
        probe._install_d48_resource_accounting(d42)
    else:
        d42._fit_equal_prior_lda = probe.d45.build_inner_loo_reliability_fit(d42)
        probe.d45._install_d45_core_resource_accounting(d42)
    try:
        return d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            old_labels,
            ["old-a", "old-b"],
            new_rows,
            new_labels,
            ["new-a", "new-b"],
            seed=4803,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top


def _fit_pair(k_shot: int = 5, duplicate_k2: bool = False):
    source_k = 1 if duplicate_k2 else k_shot
    old_rows, _ = _support(4804, 2, source_k)
    new_rows, _ = _support(4805, 2, source_k)
    if duplicate_k2:
        old_rows = np.repeat(old_rows, 2, axis=0)
        new_rows = np.repeat(new_rows, 2, axis=0)
        k_shot = 2
    old_labels = [value for value in ("old-a", "old-b") for _ in range(k_shot)]
    new_labels = [value for value in ("new-a", "new-b") for _ in range(k_shot)]
    return (
        _run_fit(old_rows, old_labels, new_rows, new_labels, d48=False),
        _run_fit(old_rows, old_labels, new_rows, new_labels, d48=True),
    )


def _rows_for_result(result):
    before_count = int(
        result.geometry_audit["before_covariance_audit"]["d48_class_count"]
    )
    all_count = int(result.geometry_audit["final_covariance_audit"]["d48_class_count"])
    rows = []
    for candidate_id in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
        for _ in range(15):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "geometry_summary": copy.deepcopy(result.geometry_audit),
                    "resource": copy.deepcopy(result.resource_audit),
                    "before_old": {
                        "per_class_accuracy": {
                            f"old-{index}": 1.0 for index in range(before_count)
                        }
                    },
                    "after_new": {
                        "per_class_accuracy": {
                            f"new-{index}": 1.0
                            for index in range(all_count - before_count)
                        }
                    },
                }
            )
    return rows


def test_integrated_state_resource_and_verifier_close() -> None:
    d45_result, d48_result = _fit_pair()
    rows = _rows_for_result(d48_result)
    assert probe._verify_d48_fit_audits(rows) == 30
    np.testing.assert_array_equal(d48_result.before_state.coef_fp32, d45_result.before_state.coef_fp32)
    np.testing.assert_array_equal(d48_result.state.coef_fp32, d45_result.state.coef_fp32)
    assert not np.array_equal(d48_result.state.intercept_fp16, d45_result.state.intercept_fp16)
    final_audit = rows[0]["geometry_summary"]["final_covariance_audit"]
    assert final_audit["d48_beta_nonzero_class_count"] > 0
    assert final_audit["d45_post_fusion_calibration_coefficient_bitwise_unchanged"] is True
    resource = rows[0]["resource"]
    expected_margin = 768
    assert resource["lda_closed_form_fit_count"] == 24
    assert resource["d48_margin_operation_upper_bound"] == expected_margin
    assert resource["d48_adaptation_evidence_peak_numeric_bytes"] == 2720
    geometry = rows[0]["geometry_summary"]
    independently_selected = {
        field: {
            name: value
            for name, value in geometry[field].items()
            if name.startswith("d48_")
            or name.startswith("d45_post_fusion_calibration_")
        }
        for field in ("before_covariance_audit", "final_covariance_audit")
    }
    independently_selected["formal_state_binding"] = {
        name: value for name, value in geometry.items() if name.startswith("d48_")
    }
    expected_json_bytes = len(
        json.dumps(
            independently_selected,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    assert resource["d48_persisted_fit_audit_json_utf8_bytes"] == expected_json_bytes
    assert resource["d48_persisted_fit_audit_serialization"].startswith(
        "canonical_json"
    )
    assert resource["d48_additional_lda_fit_count"] == 0
    assert resource["d48_query_sidecar_bytes"] == 0


def test_compiled_fp32_bias_uses_elementwise_rounding_envelope() -> None:
    base = np.asarray([5.942129135131836], dtype=np.float32)
    final = np.asarray([5.25760555267334], dtype=np.float32)
    requested = np.asarray([-0.684523816254983], dtype=np.float64)
    compiled = (final - base).astype(np.float32).astype(np.float64)
    error = np.abs(compiled - requested)
    bound = probe._fp32_compiled_bias_rounding_bound(base, final, requested)
    assert error[0] > 2.0e-7
    assert error[0] <= bound[0]

    with pytest.raises(probe.D48ProbeError, match="bound input drift"):
        probe._fp32_compiled_bias_rounding_bound(base, final, [np.nan])


def test_k1_is_bitwise_d45_fallback_full_chain() -> None:
    d45_result, d48_result = _fit_pair(k_shot=1)
    np.testing.assert_array_equal(d48_result.before_state.coef_fp32, d45_result.before_state.coef_fp32)
    np.testing.assert_array_equal(
        d48_result.before_state.intercept_fp16, d45_result.before_state.intercept_fp16
    )
    np.testing.assert_array_equal(d48_result.state.coef_fp32, d45_result.state.coef_fp32)
    np.testing.assert_array_equal(d48_result.state.intercept_fp16, d45_result.state.intercept_fp16)
    rows = _rows_for_result(d48_result)
    assert probe._verify_d48_fit_audits(rows) == 30
    assert rows[0]["resource"]["d48_margin_operation_upper_bound"] == 0


def test_real_k2_unit_components_close_full_chain() -> None:
    _d45_result, d48_result = _fit_pair(duplicate_k2=True)
    rows = _rows_for_result(d48_result)
    assert probe._verify_d48_fit_audits(rows) == 30
    for field in ("before_covariance_audit", "final_covariance_audit"):
        audit = rows[0]["geometry_summary"][field]
        assert audit["d48_actual_k"] == 2
        assert audit["d48_d45_full_weight"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "field",
    [
        "d43_probe_arm",
        "d43_covariance_structure",
        "d45_probe_arm",
        "d48_bias_formula",
        "d48_statistical_claim",
    ],
)
def test_verifier_rejects_core_field_tampering(field: str) -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    rows[0]["geometry_summary"]["before_covariance_audit"][field] = "tampered"
    with pytest.raises(probe.D48ProbeError, match="exact audit drift"):
        probe._verify_d48_fit_audits(rows)


def test_verifier_rejects_margin_and_resource_tampering() -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    rows[0]["geometry_summary"]["final_covariance_audit"][
        "d48_beta_centered_by_class"
    ][0] += 0.5
    with pytest.raises(probe.D48ProbeError, match="evidence closure"):
        probe._verify_d48_fit_audits(rows)
    rows = _rows_for_result(result)
    rows[0]["resource"]["d48_margin_operation_upper_bound"] += 1
    with pytest.raises(probe.D48ProbeError, match="resource audit drift"):
        probe._verify_d48_fit_audits(rows)


def test_verifier_binds_d48_weight_and_logits_to_d45_evidence() -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    audit = rows[0]["geometry_summary"]["final_covariance_audit"]
    audit["d48_d45_full_weight"] += 0.01
    audit["d48_d45_block_weight"] -= 0.01
    with pytest.raises(probe.D48ProbeError, match="global-weight binding"):
        probe._verify_d48_fit_audits(rows)

    rows = _rows_for_result(result)
    audit = rows[0]["geometry_summary"]["final_covariance_audit"]
    full = np.asarray(audit["d48_full_held_logits_by_fold_class_logit"])
    block = np.asarray(audit["d48_block_held_logits_by_fold_class_logit"])
    full_weight = float(audit["d48_d45_full_weight"])
    block_weight = float(audit["d48_d45_block_weight"])
    full[0, 0, 0] += 0.1
    block[0, 0, 0] -= 0.1 * full_weight / block_weight
    audit["d48_full_held_logits_by_fold_class_logit"] = full.tolist()
    audit["d48_block_held_logits_by_fold_class_logit"] = block.tolist()
    with pytest.raises(probe.D48ProbeError, match="held-logit/D45 CE binding"):
        probe._verify_d48_fit_audits(rows)


def test_verifier_rejects_missing_or_consistently_forged_lifecycle_hashes() -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    audit = rows[0]["geometry_summary"]["before_covariance_audit"]
    audit.pop("d45_post_fusion_calibration_base_intercept_sha256")
    with pytest.raises(probe.D48ProbeError, match="lifecycle SHA"):
        probe._verify_d48_fit_audits(rows)

    rows = _rows_for_result(result)
    audit = rows[0]["geometry_summary"]["before_covariance_audit"]
    forged = "0" * 64
    audit["d45_post_fusion_calibration_base_coefficient_sha256"] = forged
    audit["d45_post_fusion_calibration_final_coefficient_sha256"] = forged
    with pytest.raises(probe.D48ProbeError, match="coefficient/intercept residual"):
        probe._verify_d48_fit_audits(rows)


def test_verifier_rejects_formal_coefficient_state_or_sha_forgery() -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    geometry = rows[0]["geometry_summary"]
    geometry["d48_before_formal_coefficient_int8_sha256"] = "0" * 64
    with pytest.raises(probe.D48ProbeError, match="formal FP16/state binding"):
        probe._verify_d48_fit_audits(rows)

    rows = _rows_for_result(result)
    geometry = rows[0]["geometry_summary"]
    geometry["d48_final_formal_coef1_qint8_values"][0][0] += 1
    with pytest.raises(probe.D48ProbeError, match="formal FP16/state binding"):
        probe._verify_d48_fit_audits(rows)


def test_verifier_rejects_persisted_json_byte_count_forgery() -> None:
    _base, result = _fit_pair()
    rows = _rows_for_result(result)
    rows[0]["resource"]["d48_persisted_fit_audit_json_utf8_bytes"] += 1
    with pytest.raises(probe.D48ProbeError, match="resource audit drift"):
        probe._verify_d48_fit_audits(rows)
