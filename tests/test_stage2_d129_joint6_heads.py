from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from cvsrffi import stage2_d129_joint6_heads as d129
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


def _lock(k_shot: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _case(
    *,
    class_count: int = 4,
    old_class_count: int = 2,
    k_shot: int = 5,
    seed: int = 19,
    partition_semantics: str = "formal_stage2c_old_new_registration",
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    classes = tuple(f"c{index}" for index in range(class_count))
    centers = d129.normalize_zid160_rows(
        rng.normal(size=(class_count, d129.Z_DIM)).astype(np.float32), name="centers"
    )
    support_rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(classes):
        support_rows.append(
            centers[index][None, :]
            + 0.035 * rng.normal(size=(k_shot, d129.Z_DIM)).astype(np.float32)
        )
        labels.extend([label] * k_shot)
    base_support = np.concatenate(support_rows, axis=0).astype(np.float32)
    base_query = (
        np.repeat(centers, 2, axis=0)
        + 0.025 * rng.normal(size=(2 * class_count, d129.Z_DIM)).astype(np.float32)
    )
    direction = d129.normalize_zid160_rows(
        rng.normal(size=(1, d129.Z_DIM)).astype(np.float32), name="adaptation_direction"
    )[0]
    adapted_support = base_support + 0.0125 * direction[None, :]
    adapted_query = base_query + 0.0125 * direction[None, :]
    return {
        "base_support_zid": base_support,
        "adapted_support_zid": adapted_support,
        "base_query_zid": base_query,
        "adapted_query_zid": adapted_query,
        "support_labels": tuple(labels),
        "registered_classes": classes,
        "old_class_count": old_class_count,
        "partition_semantics": partition_semantics,
        "opaque_query_ids": tuple(f"opaque-{index}" for index in range(len(base_query))),
        "qknn_lock": _lock(k_shot),
    }


@pytest.mark.parametrize("k_shot", [5, 10])
def test_active_k_emits_six_same_row_arms_and_split_resource_receipts(
    k_shot: int,
) -> None:
    result = d129.run_d129_joint6_heads(
        **_case(k_shot=k_shot), da_numeric_state_bytes=48
    )

    assert tuple(arm.arm_id for arm in result.arms) == d129.ARM_IDS
    assert result.r0_cache.support_zid160.shape[1] == d129.Z_DIM
    assert result.r1_cache.query_zid160.shape == result.r0_cache.query_zid160.shape
    assert result.row_receipt["backbone_forward_calls_in_joint6_interface"] == 0
    assert result.row_receipt["active_k"] == k_shot
    assert result.row_receipt["query_rows_used_for_fit"] == 0
    assert result.row_receipt["query_state_updates"] == 0

    for representation in (d129.R0, d129.R1):
        resources = result.head_causal_resource_receipt["representations"][representation]
        full = resources["full160"]
        lite = resources["lite160"]
        assert resources["same_160d_cache"] is True
        assert resources["same_affine_wire_for_full160_lite160"] is True
        assert full["deployed_numeric_state_bytes"] == lite["deployed_numeric_state_bytes"]
        assert full["query_head_macs_per_sample"] == lite["query_head_macs_per_sample"]
        assert full["explicit_dense_matrix_elements_constructed"] > 0
        assert full["explicit_linear_system_solve_count"] == 1
        assert lite["explicit_dense_matrix_elements_constructed"] == 0
        assert lite["explicit_linear_system_solve_count"] == 0
        assert full["deployed_numeric_state_bytes"] == 164 * 4
        assert full["timing_threshold_claim_permitted"] is False
        assert lite["actual_peak_workspace_measured"] is False

    system = result.system_formal_replacement_resource_receipt
    assert system["formal_d92_numeric_state_bytes"] == 1152 + 590 * 4
    assert system["lite160_numeric_state_bytes"] == 164 * 4
    assert system["joint_lite160_da_numeric_state_bytes"] == 164 * 4 + 48
    assert system["representation_pipeline_changed"] is True
    assert system["not_head_causal_comparator"] is True
    assert system["performance_causal_claim_permitted"] is False
    assert system["formal_d92_affine_query_macs_per_sample"] == 288 * 4
    assert system["lite160_affine_query_macs_per_sample"] == 160 * 4
    assert system["formal_efficiency_thresholds_evaluated"] is False


def test_k1_is_an_exact_qknn_logit_alias_not_a_historical_equivalence() -> None:
    result = d129.run_d129_joint6_heads(**_case(k_shot=1))

    for q_arm, full_arm, lite_arm in (
        (result.r0q, result.r0f, result.r0l),
        (result.r1q, result.r1f, result.r1l),
    ):
        assert full_arm.logits is q_arm.logits
        assert lite_arm.logits is q_arm.logits
        assert full_arm.predictions == q_arm.predictions
        assert lite_arm.predictions == q_arm.predictions
        for arm in (full_arm, lite_arm):
            assert isinstance(arm.state, d129.D129K1QKNNAliasState)
            assert arm.qknn_alias_receipt is not None
            assert arm.qknn_alias_receipt.historical_k1_equivalence_claim is False
            assert arm.receipt["historical_k1_equivalence_claim"] is False

    k1_r0 = result.head_causal_resource_receipt["representations"][d129.R0]
    assert k1_r0["k1_alias"] is True
    assert k1_r0["full160"]["incremental_deployed_numeric_state_bytes"] == 0
    assert k1_r0["lite160"]["incremental_query_head_macs_per_sample"] == 0


def test_loco_five_retained_one_held_k5_is_explicit_proxy_extension() -> None:
    result = d129.run_d129_joint6_heads(
        **_case(
            class_count=6,
            old_class_count=5,
            k_shot=5,
            seed=47,
            partition_semantics="phase1_seen_class_loco_directional_proxy",
        )
    )

    assert result.row_receipt["formal_new_registration_claim"] is False
    assert result.row_receipt["full160_single_class_proxy_extension"] is True
    assert (
        result.row_receipt["full160_strict_historical_d92_group_covariance_path"]
        is False
    )

    for arm in (result.r0f, result.r1f):
        assert isinstance(arm.state, d129.D129AffineHeadState)
        assert arm.receipt["fit_mode"] == "old_new_task_balanced_auto_shrinkage_full_covariance"
        assert arm.receipt["shared_logit_scale_audit"][
            "argmax_invariant_in_exact_arithmetic"
        ] is True
        assert arm.state.active_k == 5
    for representation in (d129.R0, d129.R1):
        head = result.arm(f"{representation}F")
        # The protected fit receipt is reachable through a direct re-fit below;
        # here we assert the actual six-arm route completes for the 5+1 LOCO row.
        assert head.logits.shape == (12, 6)
        assert np.isfinite(head.logits).all()

    direct = d129.fit_d92_full160(
        result.r0_cache.support_zid160,
        result.row_receipt["support_labels"],
        result.row_receipt["registered_classes"],
        old_class_count=5,
    )
    assert direct.fit_receipt["old_covariance_estimator"] == (
        "sklearn_LDA_lsqr_auto_shrinkage_equal_prior"
    )
    assert direct.fit_receipt["new_covariance_estimator"] == (
        "sklearn_LedoitWolf_single_class_centered_residuals"
    )
    assert direct.fit_receipt["old_covariance_weight"] == 0.5
    assert direct.fit_receipt["new_covariance_weight"] == 0.5


def test_shared_power_of_two_logit_scale_closes_fp16_without_changing_argmax() -> None:
    rng = np.random.default_rng(129649)
    classes = tuple(f"c{index}" for index in range(6))
    weights = rng.normal(size=(6, 160)).astype(np.float64)
    weights -= weights.mean(axis=0, keepdims=True)
    intercepts = rng.normal(size=6).astype(np.float64)
    intercepts -= intercepts.mean()
    query = d129.normalize_zid160_rows(
        rng.normal(size=(41, 160)).astype(np.float32), name="scale-test query"
    )

    base, base_audit = d129._quantize_shared_affine(
        head=d129.FULL_HEAD,
        classes=classes,
        active_k=5,
        weights=weights,
        intercepts=intercepts,
    )
    multiplier = math.ldexp(1.0, 40)
    huge, huge_audit = d129._quantize_shared_affine(
        head=d129.FULL_HEAD,
        classes=classes,
        active_k=5,
        weights=weights * multiplier,
        intercepts=intercepts * multiplier,
    )

    assert base_audit["shared_logit_scale"] == 1.0
    assert huge_audit["shared_logit_scale_exponent_base2"] < 0
    assert huge_audit["class_specific_clipping"] is False
    assert huge_audit["argmax_equivalence_scope"] == (
        "prequantized_common_positive_scaling_only"
    )
    assert huge_audit["quantized_any_query_argmax_equivalence_claim"] is False
    assert np.isfinite(huge.intercept_fp16).all()
    assert np.isfinite(huge.scale_fp16).all()
    base_predictions = np.argmax(d129.score_d129_affine_head(base, query), axis=1)
    huge_predictions = np.argmax(d129.score_d129_affine_head(huge, query), axis=1)
    np.testing.assert_array_equal(huge_predictions, base_predictions)


def test_shared_logit_scale_fails_closed_when_fp16_dynamic_range_is_impossible() -> None:
    weights = np.ones((6, 160), dtype=np.float64)
    weights[0] = np.float64(1.0e-200)
    intercepts = np.asarray([1.0e100, -1.0e100, 3.0, 2.0, 1.0, 0.0])

    with pytest.raises(d129.D129Joint6HeadsError, match="dynamic range"):
        d129._quantize_shared_affine(
            head=d129.LITE_HEAD,
            classes=tuple(f"c{index}" for index in range(6)),
            active_k=5,
            weights=weights,
            intercepts=intercepts,
        )


def test_class_label_renaming_preserves_all_six_logit_columns() -> None:
    row = _case()
    baseline = d129.run_d129_joint6_heads(**row)
    renaming = {f"c{index}": f"renamed_{index}" for index in range(4)}
    renamed = dict(row)
    renamed["registered_classes"] = tuple(
        renaming[value] for value in row["registered_classes"]
    )
    renamed["support_labels"] = tuple(renaming[value] for value in row["support_labels"])
    replay = d129.run_d129_joint6_heads(**renamed)

    for left, right in zip(baseline.arms, replay.arms, strict=True):
        np.testing.assert_allclose(left.logits, right.logits, rtol=0.0, atol=0.0)


def test_two_candidate_calls_reuse_one_common_r0_without_refit() -> None:
    row = _case(class_count=6, old_class_count=5, seed=71)
    common = d129.build_d129_common_r0(
        base_support_zid=row["base_support_zid"],
        base_query_zid=row["base_query_zid"],
        support_labels=row["support_labels"],
        registered_classes=row["registered_classes"],
        old_class_count=row["old_class_count"],
        partition_semantics=row["partition_semantics"],
        opaque_query_ids=row["opaque_query_ids"],
        qknn_lock=row["qknn_lock"],
    )
    first = d129.run_d129_joint6_heads(**row, common_r0=common)
    second_row = dict(row)
    second_row["adapted_support_zid"] = (
        np.asarray(row["adapted_support_zid"]) * np.float32(1.001)
    )
    second_row["adapted_query_zid"] = (
        np.asarray(row["adapted_query_zid"]) * np.float32(1.001)
    )
    second = d129.run_d129_joint6_heads(**second_row, common_r0=common)
    assert first.row_receipt["common_r0_sha256"] == second.row_receipt[
        "common_r0_sha256"
    ]
    assert first.row_receipt["common_r0_head_fit_calls_in_this_candidate_call"] == 0
    assert second.row_receipt["common_r0_head_fit_calls_in_this_candidate_call"] == 0
    assert first.r0q is second.r0q
    assert first.r0f is second.r0f
    assert first.r0l is second.r0l


def test_joint6_api_excludes_query_truth_roles_quotas_and_source_clean_inputs() -> None:
    parameters = set(inspect.signature(d129.run_d129_joint6_heads).parameters)
    forbidden = {"query_truth", "query_roles", "class_quota", "source", "clean"}
    assert forbidden.isdisjoint(parameters)

    bad = _case()
    bad["adapted_query_zid"] = np.asarray(bad["adapted_query_zid"])[:, :-1]
    with pytest.raises(d129.D129Joint6HeadsError, match="N,160"):
        d129.run_d129_joint6_heads(**bad)
