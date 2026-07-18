from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "probe_d49_nested_global_cosine_fusion.py"
SPEC = importlib.util.spec_from_file_location("probe_d49", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(seed: int = 4901, *, class_count: int = 3, k_shot: int = 3):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), k_shot)
    centers = rng.normal(size=(class_count, d42.FEATURE_DIM)).astype(np.float32)
    for index in range(class_count):
        centers[index, index * 13 + 1] += np.float32(7.0)
    rows = []
    for index in range(class_count):
        for _rank in range(k_shot):
            row = centers[index] + np.float32(0.25) * rng.normal(size=d42.FEATURE_DIM).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
    return np.stack(rows), labels, class_count, k_shot


def test_global_cosine_reference_is_exact_single_affine_score() -> None:
    rows, labels, class_count, k_shot = _support()
    coef, intercept, audit = probe._cosine_component_fit(
        rows, labels, class_count, k_shot, energy_epsilon=d42.ENERGY_EPSILON
    )
    reference = rows.astype(np.float64) @ coef.astype(np.float64).T
    affine = rows.astype(np.float64) @ coef.astype(np.float64).T + intercept[None, :]
    np.testing.assert_array_equal(reference, affine)
    np.testing.assert_allclose(np.linalg.norm(coef, axis=1), 1.0, rtol=0.0, atol=2e-7)
    rows64 = rows.astype(np.float64)
    independent = np.stack(
        [rows64[labels == index].mean(axis=0) for index in range(class_count)]
    )
    independent /= np.linalg.norm(independent, axis=1, keepdims=True)
    np.testing.assert_array_equal(coef, independent.astype(np.float32))
    assert audit["d49_cosine_geometry"] == "D42_global_unit_sphere_cosine_prototype"
    assert audit["d49_cosine_affine_intercept_zero"] is True


def test_cosine_fail_closes_invalid_class_norm_and_non_unit_rows() -> None:
    rows, labels, class_count, k_shot = _support(class_count=2, k_shot=2)
    with pytest.raises(probe.D49ProbeError, match="unit-sphere"):
        probe._cosine_component_fit(rows * 2, labels, class_count, k_shot, energy_epsilon=1e-12)
    zero = np.stack([rows[0], -rows[0], rows[2], -rows[2]])
    zero_labels = np.repeat(np.arange(2), 2)
    with pytest.raises(probe.D49ProbeError, match="resultant norm"):
        probe._cosine_component_fit(zero, zero_labels, 2, 2, energy_epsilon=1e-12)
    with pytest.raises(probe.D49ProbeError, match="coverage"):
        probe._cosine_component_fit(rows[:2], np.zeros(2, dtype=int), 1, 2, energy_epsilon=1e-12)


def test_weight_tie_is_exact_half_and_unequal_ce_is_not_forced_half() -> None:
    left, right, evidence = probe._strict_weights(0.4, 0.4, 5)
    assert (left, right) == (0.5, 0.5)
    assert evidence == [-2.0, -2.0]
    left, right, _ = probe._strict_weights(0.2, 0.6, 5)
    assert 1.0 > left > 0.5 > right > 0.0
    assert abs(left + right - 1.0) <= 1e-12
    with pytest.raises(probe.D49ProbeError):
        probe._strict_weights(np.nan, 0.2, 5)
    with pytest.raises(probe.D49ProbeError, match="endpoint"):
        probe._strict_weights(0.0, 1.0e308, 5)


def test_nested_fit_uses_complete_d45_refits_inner_rms_and_one_fp32_fusion() -> None:
    rows, labels, class_count, k_shot = _support(4902, class_count=4, k_shot=5)
    d45_fit = probe.d45.build_inner_loo_reliability_fit(d42)
    d_coef, d_intercept, _ = d45_fit(rows, labels, class_count, k_shot)
    c_coef, c_intercept, _ = probe._cosine_component_fit(
        rows, labels, class_count, k_shot, energy_epsilon=d42.ENERGY_EPSILON
    )
    coef, intercept, audit = probe.build_nested_global_cosine_fit(d42)(
        rows, labels, class_count, k_shot
    )
    d_scale = probe.d44._class_centered_logit_rms(rows, d_coef, d_intercept)
    c_scale = probe.d44._class_centered_logit_rms(rows, c_coef, c_intercept)
    expected = probe.d43._center_affine_scores(
        audit["d49_d45_weight"] * d_coef.astype(np.float64) / d_scale
        + audit["d49_cosine_weight"] * c_coef.astype(np.float64) / c_scale,
        audit["d49_d45_weight"] * d_intercept.astype(np.float64) / d_scale,
    )
    np.testing.assert_array_equal(coef, expected[0].astype(np.float32))
    np.testing.assert_array_equal(intercept, expected[1].astype(np.float32))
    partition = audit["d49_nested_partition_audit"]
    assert partition["held_support_row_exact_once_coverage"] is True
    assert partition["train_held_overlap_count"] == 0
    assert len(partition["d49_nested_d45_fit_audit_by_fold"]) == k_shot
    assert all(item["d45_probe_arm"] == probe.d45.ARM for item in partition["d49_nested_d45_fit_audit_by_fold"])
    assert audit["d49_inner_train_rms_used_for_held"] is True
    assert audit["d49_full_support_rms_used_for_final_affine"] is True
    assert audit["d49_full_support_fused_once_pre_quantization"] is True
    assert audit["d49_quantized_component_fused_or_decoded"] is False


def test_k1_is_bitwise_and_audit_exact_d45_fallback() -> None:
    rows, labels, class_count, k_shot = _support(4903, class_count=3, k_shot=1)
    expected = probe.d45.build_inner_loo_reliability_fit(d42)(rows, labels, class_count, k_shot)
    actual = probe.build_nested_global_cosine_fit(d42)(rows, labels, class_count, k_shot)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert json.dumps(actual[2], sort_keys=True) == json.dumps(expected[2], sort_keys=True)
    assert not any(name.startswith("d49_") for name in actual[2])


def test_k2_nested_weight_is_learned_and_not_forced_to_half(monkeypatch) -> None:
    rows, labels, class_count, k_shot = _support(4910, class_count=3, k_shot=2)
    observed_k: list[int] = []

    def fake_builder(_d42):
        def fake_fit(transformed, targets, class_count, k_shot):
            observed_k.append(int(k_shot))
            means = np.stack(
                [transformed[targets == index].mean(axis=0) for index in range(class_count)]
            ).astype(np.float32)
            coef = means.copy()
            coef[:, :32] *= np.float32(3.0)
            intercept = np.linspace(-0.3, 0.4, class_count, dtype=np.float32)
            return coef, intercept, {"d45_probe_arm": probe.d45.ARM}

        return fake_fit

    monkeypatch.setattr(probe.d45, "build_inner_loo_reliability_fit", fake_builder)
    _coef, _intercept, audit = probe.build_nested_global_cosine_fit(d42)(
        rows, labels, class_count, k_shot
    )
    expected = probe._strict_weights(
        audit["d49_d45_nested_macro_class_ce"],
        audit["d49_cosine_nested_macro_class_ce"],
        class_count,
    )
    np.testing.assert_allclose(
        [audit["d49_d45_weight"], audit["d49_cosine_weight"]],
        expected[:2],
        rtol=0.0,
        atol=1e-15,
    )
    assert not np.allclose(expected[:2], [0.5, 0.5], rtol=0.0, atol=1e-12)
    assert observed_k == [2, 1, 1]


def test_real_locked_k2_path_uses_nested_k1_d45_without_forced_half() -> None:
    base, base_labels, class_count, _ = _support(4915, class_count=3, k_shot=1)
    rows = np.repeat(base, 2, axis=0)
    labels = np.repeat(base_labels, 2)
    _coef, _intercept, audit = probe.build_nested_global_cosine_fit(d42)(
        rows, labels, class_count, 2
    )
    assert audit["d49_outer_fold_count"] == 2
    assert audit["d49_d45_weight"] > 0.0
    assert audit["d49_cosine_weight"] > 0.0
    np.testing.assert_allclose(
        audit["d49_d45_weight"] + audit["d49_cosine_weight"], 1.0
    )
    assert all(
        item["d45_k1_equivalent_unit_covariance_fallback"] is True
        for item in audit["d49_nested_partition_audit"][
            "d49_nested_d45_fit_audit_by_fold"
        ]
    )


def test_label_and_common_rank_permutation_are_equivariant() -> None:
    rows, labels, class_count, k_shot = _support(4904, class_count=4, k_shot=5)
    fit = probe.build_nested_global_cosine_fit(d42)
    coef, intercept, audit = fit(rows, labels, class_count, k_shot)
    rank_permutation = np.asarray([2, 0, 4, 1, 3])
    row_order = np.concatenate(
        [index * k_shot + rank_permutation for index in range(class_count)]
    )
    rank_coef, rank_intercept, rank_audit = fit(
        rows[row_order], labels[row_order], class_count, k_shot
    )
    np.testing.assert_allclose(rank_coef, coef, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(rank_intercept, intercept, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(
        [rank_audit["d49_d45_weight"], rank_audit["d49_cosine_weight"]],
        [audit["d49_d45_weight"], audit["d49_cosine_weight"]],
        rtol=0.0,
        atol=2e-12,
    )
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_labels = inverse[labels]
    pcoef, pintercept, _ = fit(rows, permuted_labels, class_count, k_shot)
    np.testing.assert_allclose(pcoef[inverse], coef, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(pintercept[inverse], intercept, rtol=0.0, atol=2e-6)


def test_k8_inventory_is_exactly_292_lda_fits() -> None:
    before = probe._d45_fit_specs("before", 6, 8)
    final = probe._d45_fit_specs("final", 11, 8)
    assert sum(item[1] for item in before + final) == 292
    assert [item[1] for item in before] == [2, 16, 16, 112]
    assert [item[1] for item in final] == [2, 16, 16, 112]


def test_real_compilation_resource_wrapper_keeps_one_affine_state() -> None:
    k_shot = 5
    old_rows, _labels, _count, _k = _support(4911, class_count=2, k_shot=k_shot)
    new_rows, _labels, _count, _k = _support(4912, class_count=2, k_shot=k_shot)
    old_labels = [value for value in ("old-a", "old-b") for _ in range(k_shot)]
    new_labels = [value for value in ("new-a", "new-b") for _ in range(k_shot)]
    original_fit = d42._fit_equal_prior_lda
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda
    d42._fit_equal_prior_lda = probe.build_nested_global_cosine_fit(d42)
    probe._install_d49_resource_and_tie_audit(d42)
    try:
        result = d42.fit_d42_unified_shrinkage_lda(
            old_support_features=old_rows,
            old_support_labels=old_labels,
            old_classes=("old-a", "old-b"),
            new_support_features=new_rows,
            new_support_labels=new_labels,
            new_classes=("new-a", "new-b"),
            seed=4911,
            device="cpu",
        )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42._lda_fit_macs = original_macs
        d42.fit_d42_unified_shrinkage_lda = original_top
    resource = result.resource_audit
    assert resource["lda_closed_form_fit_count"] == 4 + 4 * k_shot + 4 * k_shot**2
    assert resource["d49_fused_query_state_count"] == 1
    assert resource["d49_additional_query_state_count"] == 0
    assert resource["d49_query_view"] == "full_288d_only"
    assert resource["d49_fp32_exact_top_tie_count"] == 0
    assert resource["d49_int8_exact_top_tie_count"] == 0
    assert resource["d49_cuda_peak_memory_measured"] is False
    assert result.state.coef1_qint8.shape == (4, d42.FEATURE_DIM)
    assert result.state.intercept_fp16.shape == (4,)


def test_fp32_and_int8_exact_top_ties_fail_closed() -> None:
    probe._assert_no_exact_top_tie(np.asarray([[1.0, 0.0], [0.0, 1.0]]), "ok")
    with pytest.raises(probe.D49ProbeError, match="fp32"):
        probe._assert_no_exact_top_tie(np.asarray([[1.0, 1.0]]), "fp32")
    with pytest.raises(probe.D49ProbeError, match="int8"):
        probe._assert_no_exact_top_tie(np.asarray([[2, 2]], dtype=np.int8), "int8")


def test_runner_outer_score_guard_checks_both_precisions_and_rejects_tie() -> None:
    module = SimpleNamespace()
    module.score_d42_unified_shrinkage_lda = lambda state, _features: state.scores
    _original, counts = probe._install_runner_score_tie_guard(module)
    fp32_state = SimpleNamespace(
        coef_fp32=np.ones((2, 2), dtype=np.float32),
        scores=np.asarray([[2.0, 1.0]], dtype=np.float32),
    )
    int8_state = SimpleNamespace(
        coef_fp32=np.empty((0, 2), dtype=np.float32),
        scores=np.asarray([[3.0, 1.0]], dtype=np.float32),
    )
    module.score_d42_unified_shrinkage_lda(fp32_state, np.zeros((1, 2)))
    module.score_d42_unified_shrinkage_lda(int8_state, np.zeros((1, 2)))
    assert counts == {"fp32_rows": 1, "int8_rows": 1, "call_count": 2}
    int8_state.scores = np.asarray([[1.0, 1.0]], dtype=np.float32)
    with pytest.raises(probe.D49ProbeError, match="runner_int8"):
        module.score_d42_unified_shrinkage_lda(int8_state, np.zeros((1, 2)))


def _partition(class_count: int, k_shot: int) -> dict:
    held = [[index * k_shot + rank for index in range(class_count)] for rank in range(k_shot)]
    all_rows = set(range(class_count * k_shot))
    return {
        "held_support_row_indices_by_fold": held,
        "train_support_row_indices_by_fold": [sorted(all_rows - set(fold)) for fold in held],
        "train_held_overlap_count": 0,
        "held_support_row_exact_once_coverage": True,
        "d49_nested_d45_fit_audit_by_fold": [{"d45_probe_arm": probe.d45.ARM} for _ in range(k_shot)],
        "d49_d45_inner_train_logit_rms_by_fold": [1.0] * k_shot,
        "d49_cosine_inner_train_logit_rms_by_fold": [1.0] * k_shot,
    }


def _fit_audit(class_count: int, k_shot: int) -> dict:
    d_weight, c_weight, evidence = probe._strict_weights(0.4, 0.5, class_count)
    return {
        "d49_probe_arm": probe.ARM, "d49_weight_formula": probe.WEIGHT_FORMULA,
        "d49_query_view": probe.QUERY_VIEW, "d49_outer_fold_count": k_shot,
        "d49_complete_d45_refit_per_outer_fold": True,
        "d49_inner_train_rms_used_for_held": True,
        "d49_full_support_rms_used_for_final_affine": True,
        "d49_full_support_fused_once_pre_quantization": True,
        "d49_quantized_component_fused_or_decoded": False,
        "d49_role_handle_scene_specific_branch": False,
        "d49_scan_temperature_threshold_count": 0,
        "d49_cosine_prototype_resultant_norm_by_class": [1.0] * class_count,
        "d49_d45_nested_macro_class_ce": 0.4,
        "d49_cosine_nested_macro_class_ce": 0.5,
        "d49_log_evidence_by_head": evidence,
        "d49_d45_weight": d_weight, "d49_cosine_weight": c_weight,
        "d49_nested_partition_audit": _partition(class_count, k_shot),
    }


def _valid_rows() -> list[dict]:
    k, old_count, all_count, dimension = 5, 4, 5, d42.FEATURE_DIM
    old_rows, old_labels, _, _ = _support(4902, class_count=old_count, k_shot=k)
    all_rows, all_labels, _, _ = _support(4904, class_count=all_count, k_shot=k)
    fit = probe.build_nested_global_cosine_fit(d42)
    _old_coef, _old_intercept, before_audit = fit(
        old_rows, old_labels, old_count, k
    )
    _all_coef, _all_intercept, final_audit = fit(
        all_rows, all_labels, all_count, k
    )
    inventory = []
    for prefix, count in (("before", old_count), ("final", all_count)):
        for group, fit_count, row_count, class_count in probe._d45_fit_specs(prefix, count, k):
            inventory.append({
                "fit_group": group, "fit_count": fit_count,
                "row_count_per_fit": row_count, "class_count": class_count,
                "macs_per_fit": probe.d45._expected_lda_fit_macs(row_count, class_count, dimension),
            })
    lda_macs = sum(item["fit_count"] * item["macs_per_fit"] for item in inventory)
    before_extra = probe._extra_head_macs(old_count, k, dimension)
    final_extra = probe._extra_head_macs(all_count, k, dimension)
    extra_fields = {
        "d49_cosine_prototype_adaptation_macs": before_extra["prototype"] + final_extra["prototype"],
        "d49_head_rms_adaptation_macs": before_extra["rms"] + final_extra["rms"],
        "d49_nested_held_scoring_macs": before_extra["held_score"] + final_extra["held_score"],
        "d49_fp32_affine_fusion_macs": before_extra["fusion"] + final_extra["fusion"],
    }
    extra_total = sum(extra_fields.values())
    fit_count = sum(item["fit_count"] for item in inventory)
    resource = {
        "old_k_shot": k, "new_k_shot": k, "coefficient_dimension": dimension,
        "lda_closed_form_fit_count": fit_count, "estimated_lda_fit_macs": lda_macs,
        "d49_lda_fit_inventory": inventory, "d49_lda_fit_inventory_macs": lda_macs,
        "d49_k8_exact_292_lda_fit_count_pass": True, "d49_fused_query_state_count": 1,
        "d49_additional_query_state_count": 0, "d49_query_view": probe.QUERY_VIEW,
        "d49_fp32_exact_top_tie_count": 0, "d49_int8_exact_top_tie_count": 0,
        "d49_extra_adaptation_macs": extra_total, "estimated_metric_adaptation_macs": 5,
        "estimated_adaptation_macs": 5 + lda_macs + extra_total,
        "d49_host_fp64_peak_memory_measured": False, "d49_host_fp64_peak_memory_bytes": None,
        "runtime_device": "cpu", "d49_cuda_peak_memory_measured": False,
        **extra_fields,
    }
    def compiled_binding(prefix: str, audit: dict) -> dict:
        coef = np.asarray(audit["d49_final_fused_coefficient_fp32"], dtype=np.float32)
        intercept = np.asarray(
            audit["d49_final_fused_intercept_fp32"], dtype=np.float32
        )
        q1, q2, s1, s2 = probe._audit_quantize_coefficients(coef)
        return {
            f"d49_{prefix}_actual_matched_fp32_coefficient": coef.astype(np.float64).tolist(),
            f"d49_{prefix}_actual_matched_fp32_intercept": intercept.astype(np.float64).tolist(),
            f"d49_{prefix}_actual_coef1_qint8": q1.astype(np.int64).tolist(),
            f"d49_{prefix}_actual_coef2_qint8": q2.astype(np.int64).tolist(),
            f"d49_{prefix}_actual_scale1_fp16": s1.astype(np.float64).tolist(),
            f"d49_{prefix}_actual_scale2_fp16": s2.astype(np.float64).tolist(),
            f"d49_{prefix}_actual_intercept_fp16": intercept.astype(np.float16).astype(np.float64).tolist(),
        }
    geometry = {
        "before_covariance_audit": before_audit,
        "final_covariance_audit": final_audit,
        "before_materialized_pre_stage2c": True,
        "before_state_immutable_during_stage2c": True,
        "old_only_metric_new_support_argument_count": 0,
        "d49_fp32_exact_top_tie_fail_close_checked": True,
        "d49_int8_exact_top_tie_fail_close_checked": True,
        "d49_single_affine_state_only": True,
        "d49_before_support_transform_bound_to_runtime_input": True,
        "d49_final_support_transform_bound_to_runtime_input": True,
        "d49_before_support_targets_bound_to_runtime_labels": True,
        "d49_final_support_targets_bound_to_runtime_labels": True,
        **compiled_binding("before", before_audit),
        **compiled_binding("final", final_audit),
    }
    return [
        {"candidate_id": candidate, "resource": copy.deepcopy(resource), "geometry_summary": copy.deepcopy(geometry)}
        for candidate in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
        for _ in range(15)
    ]


def test_verifier_recomputes_nested_and_resource_evidence() -> None:
    rows = _valid_rows()
    assert probe._verify_d49_fit_audits(rows) == 30
    broken = copy.deepcopy(rows)
    broken[0]["resource"]["lda_closed_form_fit_count"] = 291
    with pytest.raises(probe.D49ProbeError, match="resource total"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"]["d49_nested_partition_audit"]["train_support_row_indices_by_fold"][0].append(0)
    with pytest.raises(probe.D49ProbeError, match="nested rank"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_d45_held_scores_by_fold"][0][0][0] += 1.0
    with pytest.raises(probe.D49ProbeError, match="held-score CE"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_cosine_prototype_resultant_norm_by_class"
    ][0] = -1.0
    with pytest.raises(probe.D49ProbeError, match="prototype evidence"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    support_targets = broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_support_targets"
    ]
    support_targets[0], support_targets[5] = support_targets[5], support_targets[0]
    with pytest.raises(probe.D49ProbeError, match="support provenance"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_cosine_inner_train_logit_rms_by_fold"][0] = 0.0
    with pytest.raises(probe.D49ProbeError, match="nested RMS"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_cosine_inner_train_logit_rms_by_fold"][0] *= 2.0
    with pytest.raises(probe.D49ProbeError, match="nested support provenance"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_nested_cosine_fit_audit_by_fold"][0][
        "d49_cosine_prototype_resultant_norm_by_class"
    ][0] = 2.0
    with pytest.raises(probe.D49ProbeError, match="nested support provenance"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    nested_proto = broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_nested_cosine_fit_audit_by_fold"][0]["d49_cosine_prototype_fp32"][0]
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_nested_cosine_fit_audit_by_fold"][0]["d49_cosine_prototype_fp32"][0] = [
        -value for value in nested_proto
    ]
    with pytest.raises(probe.D49ProbeError, match="nested support provenance"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_nested_d45_fit_audit_by_fold"][0]["d45_full_support_logit_rms"] = -1.0
    with pytest.raises(probe.D49ProbeError, match="component RMS"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    del broken[0]["geometry_summary"]["before_covariance_audit"][
        "d49_nested_partition_audit"
    ]["d49_nested_d45_fit_audit_by_fold"][0]["d45_weight_formula"]
    with pytest.raises(probe.D49ProbeError, match="locked audit"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["resource"]["d49_lda_fit_inventory"][0]["fit_group"] = "forged"
    with pytest.raises(probe.D49ProbeError, match="inventory MAC row"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["resource"]["d49_head_rms_adaptation_macs"] = 0
    with pytest.raises(probe.D49ProbeError, match="extra MAC formula"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["before_state_immutable_during_stage2c"] = False
    with pytest.raises(probe.D49ProbeError, match="lifecycle"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["final_covariance_audit"][
        "d49_final_fused_coefficient_fp32"
    ][0][0] += 0.25
    with pytest.raises(probe.D49ProbeError, match="exact-dtype|fusion formula"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    current_code = broken[0]["geometry_summary"]["d49_final_actual_coef1_qint8"][0][0]
    broken[0]["geometry_summary"]["d49_final_actual_coef1_qint8"][0][0] = (
        -127 if current_code != -127 else 127
    )
    with pytest.raises(probe.D49ProbeError, match="compiled state binding"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"]["d49_final_actual_coef1_qint8"][0][0] += 256
    with pytest.raises(probe.D49ProbeError, match="exact-dtype"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"][
        "d49_final_actual_matched_fp32_coefficient"
    ][0][0] += 1.0e-12
    with pytest.raises(probe.D49ProbeError, match="exact-dtype"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["resource"]["d49_cuda_peak_memory_measured"] = True
    with pytest.raises(probe.D49ProbeError, match="memory audit"):
        probe._verify_d49_fit_audits(broken)
    broken = copy.deepcopy(rows)
    broken[0]["geometry_summary"][
        "d49_before_support_transform_bound_to_runtime_input"
    ] = False
    with pytest.raises(probe.D49ProbeError, match="memory audit"):
        probe._verify_d49_fit_audits(broken)
