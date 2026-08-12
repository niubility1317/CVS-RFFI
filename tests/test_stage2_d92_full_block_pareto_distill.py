from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_full_block_pareto_distill as pareto


def _test_d42_scales(coefficient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a small, positive three-block codec receipt for unit fixtures."""

    shape = (int(np.asarray(coefficient).shape[0]), 3)
    scale = np.full(shape, np.float16(1.0e-4), dtype=np.float16)
    return scale, scale.copy()


def _roundtrip_identity(
    coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A controlled deployed-codec boundary for support-only solver tests."""

    scale1, scale2 = _test_d42_scales(coefficient)
    return (
        np.asarray(coefficient, dtype=np.float32).copy(),
        np.asarray(intercept, dtype=np.float32).copy(),
        scale1,
        scale2,
    )


def _d42_preview_codec(
    coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A non-identity stand-in for the deployed D42 int8/FP16 roundtrip."""

    scale1, scale2 = _test_d42_scales(coefficient)
    return (
        np.asarray(np.round(np.asarray(coefficient) * 8.0) / 8.0, dtype=np.float32),
        np.asarray(np.asarray(intercept, dtype=np.float16), dtype=np.float32),
        scale1,
        scale2,
    )


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A hand-built registered support fixture with tied lower-tail margins."""

    classes, shots, dimension = 8, 5, 8
    labels = np.repeat(np.arange(classes, dtype=np.int64), shots)
    rows = np.zeros((classes * shots, dimension), dtype=np.float32)
    for index in range(classes):
        segment = slice(index * shots, (index + 1) * shots)
        rows[segment, index] = 1.0
        rows[segment, (index + 1) % classes] = np.asarray(
            [0.00, 0.10, 0.10, 0.20, 0.20], dtype=np.float32
        )
    full_coefficient = np.eye(classes, dimension, dtype=np.float32)
    full_intercept = np.zeros(classes, dtype=np.float32)
    block_coefficient = full_coefficient.copy()
    block_coefficient[np.arange(classes), np.arange(classes)] = np.asarray(
        [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17],
        dtype=np.float32,
    )
    block_intercept = np.zeros(classes, dtype=np.float32)
    return rows, labels, full_coefficient, full_intercept, block_coefficient, block_intercept


def _active_block_fixture() -> np.ndarray:
    """Frozen support-only BLOCK complement that admits a non-E0 solution."""

    return np.asarray(
        [
            [0.97848094, 0.48070797, 0.0979209, 0.00508255, 0.27998611, 0.10022877, -0.08027538, 0.22639142],
            [-0.1170349, 1.06702042, 0.50020283, 0.16017501, -0.16764891, 0.07791618, 0.31156144, -0.51210082],
            [0.01660064, -0.08541534, 1.24508691, -0.0472964, -0.17401099, 0.09495639, -0.55274493, -0.56306684],
            [0.1728625, -0.5567221, -0.51956642, 2.29499531, 0.0262693, 0.29867524, -0.23570281, 0.02083864],
            [0.28181857, -0.07912765, 0.38143042, -0.11676901, 1.48988986, 0.05129727, -0.24886879, -0.07890654],
            [-0.00305309, 0.07176821, 0.22503625, -0.10449605, -0.56098098, 1.11511493, 0.10463309, 0.20924233],
            [-0.18612935, 0.0307765, 0.19872902, 0.20158328, -0.04390346, 0.03154386, 1.6789434, 0.03158128],
            [0.01252883, -0.04975862, 0.31452912, -0.0434458, -0.07471423, 0.23764914, 0.03162961, 2.402843],
        ],
        dtype=np.float32,
    )


def test_fixed_lower_tail_keeps_all_threshold_ties():
    """Would fail if a Q20 tail dropped support samples tied at its lower threshold."""

    margins = np.asarray(
        [0.10, 0.20, 0.10, 0.50, 0.10, 0.30, 0.20, 0.20, 0.40, 0.20],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(2, dtype=np.int64), 5)

    selected, thresholds = pareto.fixed_lower_tail_indices(
        margins, labels, class_count=2, k_shot=5
    )

    assert thresholds == [0.10, 0.20]
    np.testing.assert_array_equal(selected[0], np.asarray([0, 2, 4]))
    np.testing.assert_array_equal(selected[1], np.asarray([6, 7, 9]))


def test_fixed_pooled_new_tail_keeps_cross_class_threshold_ties():
    """Would fail if the seventh constraint split or dropped tied new support."""

    margins = np.asarray(
        [0.9] * 30 + [0.10, 0.20, 0.10, 0.40, 0.10, 0.20, 0.30, 0.10, 0.50, 0.10],
        dtype=np.float64,
    )
    labels = np.concatenate(
        [np.repeat(np.arange(6, dtype=np.int64), 5), np.repeat(np.asarray([6, 7]), 5)]
    )

    selected, threshold = pareto.fixed_pooled_new_tail_indices(margins, labels)

    assert threshold == 0.10
    np.testing.assert_array_equal(selected, np.asarray([30, 32, 34, 37, 39]))


def test_pooled_new_tail_ignores_other_new_logits_and_uses_only_new_to_old_margin():
    """A stronger competing new class must not alter the seventh constraint."""

    classes = 8
    labels = np.repeat(np.arange(classes, dtype=np.int64), 5)
    scores = np.zeros((len(labels), classes), dtype=np.float64)
    scores[np.arange(len(labels)), labels] = 4.0
    # The deployed old logits define a known pooled new->old sequence.
    new_indices = np.flatnonzero(labels >= 6)
    scores[new_indices, 0] = np.asarray(
        [3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0]
    )
    baseline = pareto._new_to_old_margins(scores, labels)
    selected, threshold = pareto.fixed_pooled_new_tail_indices(baseline, labels)
    perturbed = scores.copy()
    # New-to-new competition changes dramatically, but old logits do not.
    perturbed[new_indices, 7] += 100.0
    perturbed[labels == 7, 6] += 100.0
    changed = pareto._new_to_old_margins(perturbed, labels)
    selected_changed, threshold_changed = pareto.fixed_pooled_new_tail_indices(
        changed, labels
    )

    np.testing.assert_array_equal(selected_changed, selected)
    assert threshold_changed == threshold


def test_deployment_equal_to_e0_falls_back_byte_exactly():
    """Would fail if a no-op deployed head was published instead of LOCAL_INVALID."""

    rows, labels, full_w, full_b, _, _ = _fixture()
    coefficient, intercept, audit = pareto.build_full_block_pareto_distill_affine_state(
    full_rows=rows,
    full_labels=labels,
    full_coefficient=full_w,
    full_intercept=full_b,
        deployed_full_coefficient=full_w,
        deployed_full_intercept=full_b,
        deployed_full_scale1=_test_d42_scales(full_w)[0],
        deployed_full_scale2=_test_d42_scales(full_w)[1],
        block_coefficient=full_w,
        block_intercept=full_b,
        class_count=8,
        k_shot=5,
        quantize_decode=_roundtrip_identity,
    )

    assert coefficient.tobytes() == full_w.tobytes()
    assert intercept.tobytes() == full_b.tobytes()
    assert audit["d92_pareto_distill_active"] is False
    assert audit["d92_pareto_distill_fallback_active"] is True
    assert audit["d92_pareto_distill_local_valid"] is False
    assert audit["d92_pareto_distill_full_head_byte_exact"] is True
    assert audit["d92_pareto_distill_deployed_e0_affine_sha256"] == (
        pareto.affine_preview_sha256(full_w, full_b)
    )
    assert audit["d92_pareto_distill_query_rows_used"] == 0


def test_registered_pareto_receipt_has_fixed_query_and_fit_inventory():
    """Would fail if active registration omitted its fixed two-solve/no-LOO receipt."""

    rows, labels, full_w, full_b, block_w, block_b = _fixture()
    _, _, audit = pareto.build_full_block_pareto_distill_affine_state(
    full_rows=rows,
    full_labels=labels,
    full_coefficient=full_w,
    full_intercept=full_b,
        deployed_full_coefficient=full_w,
        deployed_full_intercept=full_b,
        deployed_full_scale1=_test_d42_scales(full_w)[0],
        deployed_full_scale2=_test_d42_scales(full_w)[1],
        block_coefficient=block_w,
        block_intercept=block_b,
        class_count=8,
        k_shot=5,
        quantize_decode=_roundtrip_identity,
    )

    assert audit["d92_pareto_distill_mode"] == "pareto_distill"
    assert audit["d92_pareto_distill_full_solve_count"] == 1
    assert audit["d92_pareto_distill_block_solve_count"] == 1
    assert audit["d92_pareto_distill_loo_fit_count"] == 0
    assert audit["d92_pareto_distill_fisher_fit_count"] == 0
    assert audit["d92_pareto_distill_query_macs"] == 8 * 8
    assert audit["d92_pareto_distill_query_fit_access"] is False
    assert audit["d92_pareto_distill_query_update_access"] is False
    assert audit["d92_pareto_distill_query_selection_access"] is False
    assert audit["d92_pareto_distill_query_truth_access"] is False
    assert audit["d92_pareto_distill_query_role_oracle_access"] is False
    assert audit["d92_pareto_distill_query_class_quota_access"] is False
    assert audit["d92_pareto_distill_query_global_reassignment"] is False
    assert audit["d92_pareto_distill_stage1_constraint_count"] == 7
    assert audit["d92_pareto_distill_support_optimization_macs_upper_bound"] == 0


def test_active_receipt_locks_exactly_six_old_and_one_pooled_new_constraints():
    """Would fail if K>2 accidentally reinstated one gate per new class."""

    rows, labels, full_w, full_b, _, block_b = _fixture()
    _, _, audit = pareto.build_full_block_pareto_distill_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=full_w,
        full_intercept=full_b,
        deployed_full_coefficient=full_w,
        deployed_full_intercept=full_b,
        deployed_full_scale1=_test_d42_scales(full_w)[0],
        deployed_full_scale2=_test_d42_scales(full_w)[1],
        block_coefficient=_active_block_fixture(),
        block_intercept=block_b,
        class_count=8,
        k_shot=5,
        quantize_decode=_roundtrip_identity,
    )

    assert audit["d92_pareto_distill_active"] is True
    assert audit["d92_pareto_distill_stage1_constraint_count"] == 7
    gains = audit["d92_pareto_distill_old_tail_gain_by_class"]
    assert len(gains) == 6
    pooled_gain = audit["d92_pareto_distill_pooled_new_tail_gain"]
    assert audit["d92_pareto_distill_common_tail_gain"] == min([*gains, pooled_gain])
    assert audit["d92_pareto_distill_support_optimization_macs_upper_bound"] > 0
    assert audit["d92_pareto_distill_support_optimization_macs_scope"].startswith(
        "matrix_construction_plus_highs"
    )
    components = audit["d92_pareto_distill_support_optimization_macs_components"]
    assert sum(components.values()) == audit["d92_pareto_distill_support_optimization_macs_upper_bound"]
    assert components["highs_stage1"] > 0
    assert components["highs_stage2"] > 0
    assert components["slsqp_stage3"] > 0


def test_old_and_new_group_label_permutations_are_equivariant():
    """The new method itself—not just E0D—must respect both group relabelings."""

    rows, labels, full_w, full_b, _, block_b = _fixture()
    block_w = _active_block_fixture()
    coefficient, intercept, audit = pareto.build_full_block_pareto_distill_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=full_w,
        full_intercept=full_b,
        deployed_full_coefficient=full_w,
        deployed_full_intercept=full_b,
        deployed_full_scale1=_test_d42_scales(full_w)[0],
        deployed_full_scale2=_test_d42_scales(full_w)[1],
        block_coefficient=block_w,
        block_intercept=block_b,
        class_count=8,
        k_shot=5,
        quantize_decode=_roundtrip_identity,
    )
    permutation = np.asarray([2, 0, 5, 1, 4, 3, 7, 6], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted_coefficient, permuted_intercept, permuted_audit = (
        pareto.build_full_block_pareto_distill_affine_state(
            full_rows=rows,
            full_labels=permutation[labels],
            full_coefficient=full_w[inverse],
            full_intercept=full_b[inverse],
            deployed_full_coefficient=full_w[inverse],
            deployed_full_intercept=full_b[inverse],
            deployed_full_scale1=_test_d42_scales(full_w[inverse])[0],
            deployed_full_scale2=_test_d42_scales(full_w[inverse])[1],
            block_coefficient=block_w[inverse],
            block_intercept=block_b[inverse],
            class_count=8,
            k_shot=5,
            quantize_decode=_roundtrip_identity,
        )
    )

    assert audit["d92_pareto_distill_active"] is True
    assert permuted_audit["d92_pareto_distill_active"] is True
    np.testing.assert_allclose(permuted_coefficient[permutation], coefficient, atol=2e-6)
    np.testing.assert_allclose(permuted_intercept[permutation], intercept, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(permuted_audit["d92_pareto_distill_beta_by_class"])[permutation],
        np.asarray(audit["d92_pareto_distill_beta_by_class"]),
        atol=2e-6,
    )


def test_deployed_non_e0_check_uses_decoded_e0_reference_not_continuous_full():
    """The codec boundary must compare against the real decoded E0 D42 head."""

    rows, labels, full_w, full_b, _, block_b = _fixture()
    full_w = full_w.copy()
    full_w[np.arange(8), np.arange(8)] += np.float32(0.06)
    (
        deployed_full_w,
        deployed_full_b,
        deployed_full_scale1,
        deployed_full_scale2,
    ) = _d42_preview_codec(full_w, full_b)
    _, _, audit = pareto.build_full_block_pareto_distill_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=full_w,
        full_intercept=full_b,
        deployed_full_coefficient=deployed_full_w,
        deployed_full_intercept=deployed_full_b,
        deployed_full_scale1=deployed_full_scale1,
        deployed_full_scale2=deployed_full_scale2,
        block_coefficient=_active_block_fixture(),
        block_intercept=block_b,
        class_count=8,
        k_shot=5,
        quantize_decode=_d42_preview_codec,
    )

    assert full_w.tobytes() != deployed_full_w.tobytes()
    assert audit["d92_pareto_distill_deployed_e0_reference"] == "d42_decoded_full_head"


def test_real_d42_cross_group_change_below_quantum_falls_back_to_e0():
    """A sub-step D42 cross-group change must not publish due to other margins."""

    classes, shots, dimension = 8, 5, 288
    labels = np.repeat(np.arange(classes, dtype=np.int64), shots)
    rows = np.zeros((classes * shots, dimension), dtype=np.float32)
    for class_index in range(classes):
        local = slice(class_index * shots, (class_index + 1) * shots)
        rows[local, class_index] = 1.0
        rows[local, (class_index + 1) % classes] = np.asarray(
            [0.0, 0.1, 0.1, 0.2, 0.2], dtype=np.float32
        )
    full_w = np.zeros((classes, dimension), dtype=np.float32)
    full_w[np.arange(classes), np.arange(classes)] = 1.0
    full_b = np.zeros(classes, dtype=np.float32)
    block_w = full_w.copy()
    block_w[np.arange(classes), np.arange(classes)] += np.float32(0.1)
    _, _, deployed_full_scale1, deployed_full_scale2, deployed_full_w = (
        d42._quantize_coefficients(full_w)
    )
    deployed_full_b = np.asarray(full_b, dtype=np.float16).astype(np.float32)
    substep_w = full_w.copy()
    substep_w[np.arange(classes), np.arange(classes)] += np.float32(0.001)
    _, _, scale1, scale2, expected_w = d42._quantize_coefficients(substep_w)
    expected_b = np.asarray(full_b, dtype=np.float16).astype(np.float32)
    actual_quantum = float(
        max(
            np.max(deployed_full_scale1),
            np.max(deployed_full_scale2),
            np.max(scale1),
            np.max(scale2),
        )
    )

    def real_d42_substep_preview(_coefficient, intercept):
        return (
            expected_w,
            np.asarray(intercept, dtype=np.float16).astype(np.float32),
            scale1,
            scale2,
        )

    _, _, audit = pareto.build_full_block_pareto_distill_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=full_w,
        full_intercept=full_b,
        deployed_full_coefficient=deployed_full_w,
        deployed_full_intercept=deployed_full_b,
        deployed_full_scale1=deployed_full_scale1,
        deployed_full_scale2=deployed_full_scale2,
        block_coefficient=block_w,
        block_intercept=full_b,
        class_count=classes,
        k_shot=shots,
        quantize_decode=real_d42_substep_preview,
    )

    assert actual_quantum > 0.001
    assert audit["d92_pareto_distill_active"] is False
    assert audit["d92_pareto_distill_fallback_active"] is True
    assert audit["d92_pareto_distill_fallback_reason"] == (
        "deployment_cross_group_margin_change_below_quantum"
    )
    assert audit[
        "d92_pareto_distill_deployment_cross_group_margin_quantum"
    ] == actual_quantum
    assert (
        audit["d92_pareto_distill_deployment_cross_group_margin_change_max_abs"]
        < actual_quantum
    )
    assert audit["d92_pareto_distill_deployment_cross_group_quantum_pass"] is False
