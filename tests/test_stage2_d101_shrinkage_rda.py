from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_d100_ra_cgspr_lgf as d100
import cvsrffi.stage2_d101_shrinkage_rda as d101
import cvsrffi.stage2_d99_ra_cgtmk_d81 as d99
from test_stage2_d100_ra_cgspr_lgf import (
    CLASSES,
    OLD_CLASSES,
    _bank,
)


def _lock(
    bank: d99.TypedINT8MetricKernelBank,
    *,
    alpha: float = 0.35,
    target_rank: int | None = None,
) -> d101.Phase1D101Lock:
    k_shot = int(bank.metric.k_shot)
    return d101.Phase1D101Lock(
        k_shot=k_shot,
        block_variance_prior=(0.85, 1.15, 0.65),
        prior_degrees_of_freedom=8.0,
        target_residual_rank=(0 if k_shot == 1 else 2)
        if target_rank is None
        else target_rank,
        lambda_relative=0.08,
        temperature=0.9,
        d99_temperature=1.0,
        alpha=alpha,
        d99_phase1_lock_digest=bank.config.lock_digest,
        ground_geometry_receipt_sha256=bank.metric.ground_geometry_receipt_sha256,
        phase1_lodo_receipt_sha256="8" * 64,
    )


def _state(
    k_shot: int = 5,
    *,
    alpha: float = 0.35,
    degenerate_ground: bool = False,
    support_order: np.ndarray | None = None,
):
    bundle, config, ground, metric, bank, support = _bank(
        k_shot,
        degenerate_ground=degenerate_ground,
        support_order=support_order,
    )
    lock = _lock(bank, alpha=alpha)
    state = d101.build_shrinkage_rda_state(bank, ground, config=lock)
    return bundle, config, ground, metric, bank, support, lock, state


def _query(rows: int = 9, seed: int = 21201) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    query = rng.normal(size=(rows, d99.FEATURE_DIM)).astype(np.float32)
    logits = rng.normal(size=(rows, len(CLASSES))).astype(np.float32)
    return query, logits


def _typed_d81(query: np.ndarray, logits: np.ndarray, k_shot: int):
    return d100.bind_typed_d81_logits(
        logits,
        query,
        CLASSES,
        k_shot,
        source_schema="cvs.test.d81_logits.v1",
        source_receipt_sha256="9" * 64,
    )


def test_build_is_typed_quantized_closed_and_resource_claim_is_partial():
    _bundle, _config, ground, _metric, bank, _support_rows, _lock_value, state = _state(5)

    assert state.classes == bank.classes
    assert state.weight_codes_qint8.shape == (len(CLASSES), d99.FEATURE_DIM)
    assert state.weight_codes_qint8.dtype == np.int8
    assert state.weight_scales_fp16.shape == (len(CLASSES), 3)
    assert state.weight_scales_fp16.dtype == np.float16
    assert state.bias_fp16.dtype == np.float16
    assert not state.weight_codes_qint8.flags.writeable
    assert state.fit_audit["support_source"] == "exact_typed_D99_INT8_bank_decode_only"
    assert state.fit_audit["all_registered_class_means_from_support_only"] is True
    assert state.fit_audit["ground_class_means_accessed"] is False
    assert state.fit_audit["ground_basis_exact_normalized_push_forward_claimed"] is False
    assert "first_order_proxy" in state.fit_audit["ground_basis_push_forward_model"]
    assert state.fit_audit["optimizer_steps"] == 0
    assert state.fit_audit["query_rows_used"] == 0
    assert state.fit_audit["target_rank"] <= 2
    assert state.fit_audit["woodbury_rank"] <= 6
    assert state.quantization_audit["scope"] == "support_fit_diagnostic_not_held_lodo_margin_authority"
    assert state.quantization_audit["held_lodo_margin_audit_present"] is False

    wire = d101.serialize_shrinkage_rda_state(state)
    assert len(wire) == state.resource_audit["actual_serialized_state_bytes"]
    assert wire == d101.serialize_shrinkage_rda_state(state)
    assert state.resource_audit["persistent_parameter_equivalent"] <= 80_000
    partial = d101.audit_known_partial_combined_resources(state, bank, ground)
    assert partial["all_reported_wire_sizes_recomputed_from_exact_serializers"] is True
    assert partial["complete_combined_resource_claim"] is False
    assert partial["typed_d81_logit_batch_not_counted_as_persistent_head"] is True
    assert "byte" not in str(inspect.signature(d101.audit_known_partial_combined_resources))


def test_support_permutation_is_exactly_invariant():
    permutation = np.random.default_rng(44).permutation(len(CLASSES) * 5)
    first = _state(5)[-1]
    second = _state(5, support_order=permutation)[-1]
    assert first.state_receipt_sha256 == second.state_receipt_sha256
    assert np.array_equal(first.weight_codes_qint8, second.weight_codes_qint8)
    assert np.array_equal(first.weight_scales_fp16, second.weight_scales_fp16)
    assert np.array_equal(first.bias_fp16, second.bias_fp16)


def test_external_class_order_is_equivariant():
    _bundle, config, ground, _metric, bank, support = _bank(5)
    features, labels, physical = support
    permutation = np.asarray([2, 0, 3, 1])
    permuted_classes = tuple(CLASSES[index] for index in permutation)
    metric_permuted = d99.fit_support_metric(
        ground,
        features,
        labels,
        physical,
        permuted_classes,
        OLD_CLASSES,
        config=config,
    )
    bank_permuted = d99.build_typed_support_bank(
        metric_permuted,
        features,
        labels,
        physical,
        permuted_classes,
        config=config,
    )
    state = d101.build_shrinkage_rda_state(bank, ground, config=_lock(bank))
    state_permuted = d101.build_shrinkage_rda_state(
        bank_permuted, ground, config=_lock(bank_permuted)
    )
    weights = d101._decode_weight_rows(state.weight_codes_qint8, state.weight_scales_fp16)
    weights_permuted = d101._decode_weight_rows(
        state_permuted.weight_codes_qint8, state_permuted.weight_scales_fp16
    )
    assert state_permuted.classes == permuted_classes
    assert np.allclose(weights_permuted, weights[permutation], atol=2e-6)
    assert np.allclose(
        state_permuted.bias_fp16.astype(np.float32),
        state.bias_fp16.astype(np.float32)[permutation],
        atol=2e-6,
    )


def test_old_new_like_names_cannot_change_numeric_head():
    role_swapped_classes = ("new-a", "new-b", "new-c", "old-x")
    role_swapped_old = role_swapped_classes[:3]
    original = _state(5)[-1]
    _bundle, _config, ground, _metric, bank, _support_rows = _bank(
        5,
        classes=role_swapped_classes,
        old_classes=role_swapped_old,
    )
    swapped = d101.build_shrinkage_rda_state(bank, ground, config=_lock(bank))
    assert np.array_equal(original.weight_codes_qint8, swapped.weight_codes_qint8)
    assert np.array_equal(original.weight_scales_fp16, swapped.weight_scales_fp16)
    assert np.array_equal(original.bias_fp16, swapped.bias_fp16)
    signature = str(inspect.signature(d101.build_shrinkage_rda_state))
    assert "old" not in signature and "new" not in signature


def test_ground_class_means_are_neither_read_nor_dependency():
    _bundle, _config, ground, _metric, bank, _support_rows, lock, first = _state(5)
    alternative = np.roll(np.asarray(ground.class_means_fp32), shift=17, axis=1).copy()
    alternative /= np.linalg.norm(alternative, axis=1, keepdims=True)
    altered_ground = replace(ground, class_means_fp32=alternative.astype(np.float32))
    second = d101.build_shrinkage_rda_state(bank, altered_ground, config=lock)
    assert first.state_receipt_sha256 == second.state_receipt_sha256
    assert np.array_equal(first.weight_codes_qint8, second.weight_codes_qint8)
    assert "class_means_fp32" not in inspect.getsource(d101._shared_ground_covariance_view)


def test_k1_and_zero_coverage_are_true_rank_zero_block_isotropic_fallback():
    _bundle, _config, _ground, _metric, _bank_value, _support_rows, _lock_value, state = _state(
        1, degenerate_ground=True
    )
    assert state.fit_audit["residual_degrees_of_freedom"] == 0
    assert state.fit_audit["support_shrinkage_a"] == 0.0
    assert state.fit_audit["target_rank"] == 0
    assert state.fit_audit["ground_rank"] == 0
    assert state.fit_audit["woodbury_rank"] == 0
    assert state.fit_audit["k1_target_covariance_exact_zero"] is True
    assert state.fit_audit["low_coverage_fallback"] == "three_block_isotropic"
    assert np.allclose(
        state.fit_audit["posterior_block_variance_before_ridge"],
        state.fit_audit["block_variance_prior"],
        atol=0.0,
    )


@pytest.mark.parametrize(
    "diagonal,factor",
    [
        (np.linspace(0.3, 1.7, d99.FEATURE_DIM), np.zeros((d99.FEATURE_DIM, 0))),
        (np.linspace(0.2, 1.4, d99.FEATURE_DIM), np.random.default_rng(1).normal(size=(d99.FEATURE_DIM, 4)) * 0.03),
        (np.linspace(0.4, 1.1, d99.FEATURE_DIM), np.random.default_rng(2).normal(size=(d99.FEATURE_DIM, 2)) * 0.05),
        (np.full(d99.FEATURE_DIM, 1e-5), np.tile(np.linspace(-0.1, 0.1, d99.FEATURE_DIM)[:, None], (1, 6))),
    ],
)
def test_woodbury_matches_dense_inverse_for_rank_cases(diagonal: np.ndarray, factor: np.ndarray):
    rows = np.random.default_rng(700).normal(size=(7, d99.FEATURE_DIM))
    actual = d101._woodbury_precision_apply(rows, diagonal, factor)
    dense = np.diag(diagonal) + factor @ factor.T
    expected = np.linalg.solve(dense, rows.T).T
    assert np.allclose(actual, expected, rtol=2e-7, atol=2e-5)
    assert np.allclose(
        d101._woodbury_precision_dense(diagonal, factor),
        np.linalg.inv(dense),
        rtol=2e-7,
        atol=2e-5,
    )


def test_full_invertible_lda_transform_is_score_invariant():
    rng = np.random.default_rng(811)
    dimension = 17
    classes = 4
    query = rng.normal(size=(11, dimension))
    means = rng.normal(size=(classes, dimension))
    covariance_seed = rng.normal(size=(dimension, dimension))
    covariance = covariance_seed @ covariance_seed.T + 0.7 * np.eye(dimension)
    transform = rng.normal(size=(dimension, dimension)) + 2.5 * np.eye(dimension)

    precision_means = np.linalg.solve(covariance, means.T).T
    score = query @ precision_means.T - 0.5 * np.sum(means * precision_means, axis=1)
    transformed_query = query @ transform
    transformed_means = means @ transform
    transformed_covariance = transform.T @ covariance @ transform
    transformed_precision_means = np.linalg.solve(
        transformed_covariance, transformed_means.T
    ).T
    transformed_score = transformed_query @ transformed_precision_means.T - 0.5 * np.sum(
        transformed_means * transformed_precision_means, axis=1
    )
    assert np.allclose(score, transformed_score, rtol=2e-9, atol=2e-9)


def test_canonical_formula_batch_split_and_query_immutability():
    _bundle, _config, _ground, _metric, bank, _support_rows, _lock_value, state = _state(5)
    query, logits = _query()
    full = d101.canonical_fuse_typed_d81_d99_d101(
        state, bank, _typed_d81(query, logits, 5), query
    )
    parts = []
    for selection in (slice(0, 4), slice(4, None)):
        part_query = np.ascontiguousarray(query[selection])
        part_logits = np.ascontiguousarray(logits[selection])
        parts.append(
            d101.canonical_fuse_typed_d81_d99_d101(
                state,
                bank,
                _typed_d81(part_query, part_logits, 5),
                part_query,
            )
        )
    assert np.allclose(
        full.fused_probability_fp32,
        np.concatenate([item.fused_probability_fp32 for item in parts]),
        atol=2e-7,
    )
    assert full.audit["formula"] == "p99=(1-eta)*p81+eta*pStudentT;p101=(1-alpha)*p99+alpha*pRDA"
    assert full.audit["d101_replaces_d100_not_third_alpha"] is True
    assert full.audit["query_state_updates"] == 0
    assert not any(name.startswith(("update", "partial_fit", "adapt_query")) for name in d101.__all__)

    tampered = query.copy()
    tampered[0, 0] += 0.25
    with pytest.raises(d101.D101ShrinkageRDAError, match="query receipt"):
        d101.canonical_fuse_typed_d81_d99_d101(
            state, bank, _typed_d81(query, logits, 5), tampered
        )


def test_alpha_zero_skips_rda_branch_but_keeps_compiled_state(monkeypatch):
    _bundle, _config, _ground, _metric, bank, _support_rows, _lock_value, state = _state(
        5, alpha=0.0
    )
    query, logits = _query(4)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("RDA branch must not run for the exact alpha-zero control")

    monkeypatch.setattr(d101, "_score_compiled_rda_logits", forbidden)
    result = d101.canonical_fuse_typed_d81_d99_d101(
        state, bank, _typed_d81(query, logits, 5), query
    )
    assert result.rda_probability_fp32 is None
    assert np.array_equal(result.fused_probability_fp32, result.d99_probability_fp32)
    assert state.resource_audit["actual_serialized_state_bytes"] > 0


def test_typed_d81_logit_mutation_fails_closed_on_full_batch_receipt():
    _bundle, _config, _ground, _metric, bank, _support_rows, _lock_value, state = _state(5)
    query, logits = _query(3)
    typed = _typed_d81(query, logits, 5)
    typed.logits_fp32.setflags(write=True)
    typed.logits_fp32[0, 0] += np.float32(0.5)
    with pytest.raises(d101.D101ShrinkageRDAError, match="D81 batch receipt drift"):
        d101.canonical_fuse_typed_d81_d99_d101(state, bank, typed, query)


def test_public_fusion_result_rejects_out_of_range_probability():
    invalid = np.asarray([[1.2, -0.2, 0.0, 0.0]], dtype=np.float32)
    valid = np.full((1, len(CLASSES)), 1.0 / len(CLASSES), dtype=np.float32)
    with pytest.raises(d101.D101ShrinkageRDAError, match="fusion result drift"):
        d101.D101CanonicalFusionResult(
            d81_probability_fp32=invalid,
            student_t_probability_fp32=valid,
            d99_probability_fp32=valid,
            rda_probability_fp32=valid,
            fused_probability_fp32=valid,
            prediction=np.asarray([CLASSES[0]]),
            audit={},
        )


def test_formal_target_path_is_explicitly_blocked():
    assert "_score_compiled_rda_logits" not in d101.__all__
    with pytest.raises(d101.D101ShrinkageRDAError, match="blocked"):
        d101.predict_formal()
