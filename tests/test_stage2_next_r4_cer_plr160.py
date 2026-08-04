from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi import stage2_next_r4_cer_plr160 as cer
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


def _lock(k_shot: int, *, kernel_volume_gamma: float = 1.0) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=kernel_volume_gamma,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _case(
    *, class_count: int = 4, k_shot: int = 5, seed: int = 4084
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = tuple(f"tx-{index}" for index in range(class_count))
    centers = rng.normal(0.0, 0.24, size=(class_count, cer.Z_DIM)).astype(np.float32)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(classes):
        # Deliberately retain signed coordinates and non-unit norms: the CER
        # core must consume R0/R1 inputs directly rather than normalize/ReLU.
        rows.append(
            centers[index][None, :]
            + np.float32(0.03)
            * rng.normal(size=(k_shot, cer.Z_DIM)).astype(np.float32)
        )
        labels.extend([label] * k_shot)
    support = np.concatenate(rows, axis=0).astype(np.float32)
    query = (
        centers[[0, 1, 2, 3][:class_count]]
        + np.float32(0.02) * rng.normal(size=(class_count, cer.Z_DIM)).astype(np.float32)
    )
    return support, tuple(labels), classes, query.astype(np.float32)


def _distinct_qknn_logits(query_rows: int, class_count: int) -> np.ndarray:
    base = np.arange(class_count, dtype=np.float32)[None, :] * np.float32(10.0)
    return np.ascontiguousarray(base + np.arange(query_rows, dtype=np.float32)[:, None] * np.float32(0.01))


def test_k1_is_an_exact_qknn_logit_object_alias_and_tie_is_unresolved() -> None:
    support, labels, classes, query = _case(k_shot=1)
    fit = cer.fit_cer_plr160(
        support,
        labels,
        classes,
        qknn_lock=_lock(1),
        representation="R1",
    )
    logits = _distinct_qknn_logits(len(query), len(classes))

    assert type(fit.state) is cer.CERPLR160K1QKNNAliasState
    assert fit.fit_receipt["head_status"] == "K1_EXACT_QKNN_ALIAS"
    assert fit.fit_receipt["r1_post_signed_unit_relu_applied"] is False
    assert fit.fit_receipt["r1_post_signed_unit_l2_normalization_applied"] is False
    assert cer.alias_k1_qknn_logits(fit, logits) is logits
    assert cer.score_cer_plr160(fit, logits, query) is logits
    assert fit.resource_receipt["incremental_deployed_numeric_state_bytes"] == 0
    assert fit.resource_receipt["incremental_query_head_macs_per_sample"] == 0

    tied = logits.copy()
    tied[0, -1] = tied[0, -2]
    with pytest.raises(cer.NextR4CERPLR160TieError, match="TIE_UNRESOLVED"):
        cer.alias_k1_qknn_logits(fit, tied)


def test_k5_state_is_164c_int8_fp16_wire_with_bounded_support_only_residual() -> None:
    support, labels, classes, query = _case()
    fit = cer.fit_cer_plr160(
        support, labels, classes, qknn_lock=_lock(5), representation="R0"
    )
    assert type(fit.state) is cer.CERPLR160State
    state = fit.state
    assert state.numeric_state_bytes == 164 * len(classes)
    assert len(state.to_wire()) == 164 * len(classes)
    assert state.weight_qint8.dtype == np.int8
    assert state.scale_fp16.dtype == np.float16
    assert state.intercept_fp16.dtype == np.float16
    assert np.all(state.weight_qint8 >= -127)
    restored = cer.CERPLR160State.from_wire(classes, _lock(5).lock_digest, state.to_wire())
    assert restored.to_wire() == state.to_wire()
    assert restored.state_sha256 == state.state_sha256

    assert fit.fit_receipt["shrinkage_lambda"] == pytest.approx(
        len(classes) * 4 / (len(classes) * 4 + cer.Z_DIM)
    )
    # Freeze the author-defined Sr exactly: each prototype's own-class
    # residual is compared with every other-class residual at that prototype.
    class_indices = np.asarray([classes.index(label) for label in labels], dtype=np.int16)
    support64 = support.astype(np.float64)
    means = np.stack(
        [np.mean(support64[class_indices == index], axis=0) for index in range(len(classes))]
    )
    variance = np.mean(
        np.stack(
            [
                np.mean(np.square(support64[class_indices == index] - means[index]), axis=0)
                for index in range(len(classes))
            ]
        ),
        axis=0,
    )
    v_bar = float(np.mean(variance))
    lam = len(classes) * 4 / (len(classes) * 4 + cer.Z_DIM)
    d = 1.0 / (lam * variance + (1.0 - lam) * v_bar + cer.EPS32)
    d0 = 1.0 / (v_bar + cer.EPS32)
    raw_weight = means * (d - d0)[None, :]
    raw_intercept = -0.5 * np.sum(means * raw_weight, axis=1)
    raw_weight -= np.mean(raw_weight, axis=0, keepdims=True)
    raw_intercept -= np.mean(raw_intercept)
    prototype_residual = means @ raw_weight.T + raw_intercept[None, :]
    own = np.diag(prototype_residual)
    gaps = own[:, None] - prototype_residual
    expected_sr = float(
        np.sqrt(
            np.sum(np.square(gaps[~np.eye(len(classes), dtype=bool)]))
            / (len(classes) * (len(classes) - 1))
        )
    )
    assert fit.fit_receipt["sr"] == pytest.approx(expected_sr, rel=0.0, abs=1.0e-14)
    assert fit.fit_receipt["sr_definition"] == (
        "sqrt(sum_c_sum_a_ne_c[r_c(mu_c)-r_a(mu_c)]^2/(C(C-1)))"
    )
    assert fit.fit_receipt["gamma_sr"] <= fit.fit_receipt["gamma_sr_upper_bound"]
    assert fit.fit_receipt["sq_qknn_lock_fields_only"] == (
        "student_nu",
        "kernel_effective_dim",
        "shared_h0",
    )
    assert fit.resource_receipt["fit_analytic_mac_formula"] == "4Nd+8d+2Cd"
    assert fit.resource_receipt["incremental_query_head_macs_per_sample"] == cer.Z_DIM * len(classes)
    assert fit.resource_receipt["deployed_numeric_state_bytes"] == 164 * len(classes)

    q_logits = _distinct_qknn_logits(len(query), len(classes))
    actual = cer.score_cer_plr160(fit, q_logits, query)
    manual = np.asarray(
        q_logits.astype(np.float64)
        + query.astype(np.float64) @ state.decoded_weight().astype(np.float64).T
        + state.intercept_fp16.astype(np.float64)[None, :],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, manual)
    assert actual is not q_logits


def test_k5_is_class_permutation_equivariant_without_role_inputs() -> None:
    support, labels, classes, query = _case(class_count=4, seed=8124)
    original = cer.fit_cer_plr160(
        support, labels, classes, qknn_lock=_lock(5), representation="R1"
    )
    assert type(original.state) is cer.CERPLR160State
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted_classes = tuple(classes[index] for index in permutation)
    permuted = cer.fit_cer_plr160(
        support,
        labels,
        permuted_classes,
        qknn_lock=_lock(5),
        representation="R1",
    )
    assert type(permuted.state) is cer.CERPLR160State
    q_logits = _distinct_qknn_logits(len(query), len(classes))
    original_logits = cer.score_cer_plr160(original, q_logits, query)
    permuted_logits = cer.score_cer_plr160(permuted, q_logits[:, permutation], query)
    np.testing.assert_allclose(
        permuted_logits,
        original_logits[:, permutation],
        rtol=0.0,
        atol=2.0e-5,
    )
    assert original.fit_receipt["old_new_role_access"] is False
    assert original.fit_receipt["leave_one_out_access"] is False
    assert original.fit_receipt["query_accuracy_access"] is False


def test_sr_zero_and_quantized_zero_are_no_head_function_exact_aliases() -> None:
    classes = ("a", "b", "c")
    centers = np.zeros((3, cer.Z_DIM), dtype=np.float32)
    centers[0, 0] = 0.2
    centers[1, 1] = -0.2
    centers[2, 2] = 0.2
    exact_zero_support = np.repeat(centers, 5, axis=0)
    labels = tuple(label for label in classes for _ in range(5))
    q_logits = _distinct_qknn_logits(2, len(classes))
    query = np.zeros((2, cer.Z_DIM), dtype=np.float32)
    zero_fit = cer.fit_cer_plr160(
        exact_zero_support, labels, classes, qknn_lock=_lock(5), representation="R0"
    )
    assert type(zero_fit.state) is cer.CERPLR160NoFunctionAliasState
    assert zero_fit.fit_receipt["head_status"] == "NO_HEAD_FUNCTION"
    assert zero_fit.fit_receipt["no_head_function_reason"] == "Sr_ZERO"
    assert cer.score_cer_plr160(zero_fit, q_logits, query) is q_logits

    rng = np.random.default_rng(1003)
    tiny_support = (
        rng.normal(0.0, 1.0e-8, size=(len(classes) * 5, cer.Z_DIM)).astype(np.float32)
    )
    tiny_fit = cer.fit_cer_plr160(
        tiny_support, labels, classes, qknn_lock=_lock(5), representation="R1"
    )
    assert type(tiny_fit.state) is cer.CERPLR160NoFunctionAliasState
    assert tiny_fit.fit_receipt["no_head_function_reason"] == "QUANTIZED_RESIDUAL_ZERO"
    assert cer.alias_qknn_logits(tiny_fit, q_logits) is q_logits


def test_sq_is_lock_only_and_r1_query_is_consumed_without_relu_or_renorm() -> None:
    lock = _lock(5)
    assert cer.qknn_score_scale_from_lock(lock) == cer.qknn_score_scale_from_lock(
        replace(lock, kernel_volume_gamma=2.5, temperature=1.1)
    )
    support, labels, classes, query = _case(seed=619)
    query[:, :4] = np.asarray(
        [[-0.8, 0.2, -0.1, 0.4], [-0.7, -0.3, 0.2, 0.5], [-0.4, 0.1, -0.5, 0.3], [-0.2, -0.7, 0.4, 0.1]],
        dtype=np.float32,
    )
    fit = cer.fit_cer_plr160(
        support, labels, classes, qknn_lock=lock, representation="R1"
    )
    assert type(fit.state) is cer.CERPLR160State
    q_logits = _distinct_qknn_logits(len(query), len(classes))
    actual = cer.score_cer_plr160(fit, q_logits, query)
    expected = np.asarray(
        q_logits.astype(np.float64)
        + query.astype(np.float64) @ fit.state.decoded_weight().astype(np.float64).T
        + fit.state.intercept_fp16.astype(np.float64)[None, :],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_public_api_is_support_only_and_fails_closed_for_nonfinite_input() -> None:
    support, labels, classes, query = _case()
    for function in (cer.fit_cer_plr160, cer.score_cer_plr160):
        parameters = set(inspect.signature(function).parameters)
        forbidden = {"query_truth", "query_roles", "old_class_count", "new_class_count", "top_k", "loo"}
        assert forbidden.isdisjoint(parameters)

    bad = support.copy()
    bad[0, 0] = np.nan
    with pytest.raises(cer.NextR4CERPLR160Error, match="finite numpy float32"):
        cer.fit_cer_plr160(bad, labels, classes, qknn_lock=_lock(5))

    fit = cer.fit_cer_plr160(support, labels, classes, qknn_lock=_lock(5))
    assert type(fit.state) is cer.CERPLR160State
    bad_query = query.copy()
    bad_query[0, 0] = np.inf
    with pytest.raises(cer.NextR4CERPLR160Error, match="query representation"):
        cer.score_cer_plr160(fit, _distinct_qknn_logits(len(query), len(classes)), bad_query)
