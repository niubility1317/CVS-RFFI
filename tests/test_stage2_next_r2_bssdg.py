"""Focused scientific and wire-contract tests for BSSDG-160.

These tests are intentionally local and deterministic.  They do not launch a
runner, read query truth, or exercise CVFR; the main agent runs them serially
after the implementation review.
"""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi.stage2_next_r2_bssdg import (
    BSSDGBinding,
    BSSDGDuplicateFunctionError,
    BSSDGExactTieError,
    BSSDGError,
    BSSDGWireError,
    FP16_MAX,
    FP16_MIN_NORMAL,
    bssdg_resource_receipt,
    deserialize_bssdg_state,
    fit_bssdg,
    predict_bssdg_unique,
    score_bssdg,
    serialize_bssdg_state,
    validate_positive_fp16,
    validate_signed_fp16,
    verify_bssdg_binding,
)


def _support(k: int = 1) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    classes = ("old", "new")
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_id in enumerate(classes):
        for shot in range(k):
            row = np.zeros(160, dtype=np.float32)
            row[0] = np.float32(class_index + 0.02 * shot)
            row[1] = np.float32(0.01 * shot)
            rows.append(row)
            labels.append(class_id)
    return np.asarray(rows, dtype=np.float32), tuple(labels), classes


def test_pooled_prior_is_class_blind_and_class_permutation_equivariant() -> None:
    rows, labels, classes = _support(5)
    first = fit_bssdg(rows, labels, classes, k_shot=5)
    permutation = ("new", "old")
    second = fit_bssdg(rows, labels, permutation, k_shot=5)
    assert np.array_equal(first.v0_fp16, second.v0_fp16)
    query = np.zeros((3, 160), dtype=np.float32)
    query[:, 0] = np.asarray((0.1, 0.5, 0.9), dtype=np.float32)
    first_scores = score_bssdg(first, query)
    second_scores = score_bssdg(second, query)
    assert np.array_equal(first_scores[:, 0], second_scores[:, 1])
    assert np.array_equal(first_scores[:, 1], second_scores[:, 0])


def test_k1_is_native_continuous_student_t_not_an_alias() -> None:
    rows, labels, classes = _support(1)
    state = fit_bssdg(rows, labels, classes, k_shot=1)
    query = np.zeros((3, 160), dtype=np.float32)
    query[:, 0] = np.asarray((0.1, 0.5, 0.9), dtype=np.float32)
    scores = score_bssdg(state, query)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert not np.array_equal(scores[0], scores[1])
    assert state.active_k == 1


def test_posterior_mean_is_shrunk_toward_pooled_prior_before_wire_quantization() -> None:
    rows, labels, classes = _support(1)
    state = fit_bssdg(rows, labels, classes, k_shot=1)
    # zbar=(0,1) on feature zero and m0=(0.5); K1 therefore yields
    # m_c=(m0+zbar)/2=(0.25,0.75), not the raw support means (0,1).
    decoded = state.decoded_class_means[:, 0]
    assert np.isclose(decoded[0], 0.25, atol=2.0e-3, rtol=0.0)
    assert np.isclose(decoded[1], 0.75, atol=2.0e-3, rtol=0.0)
    assert not np.isclose(decoded[0], 0.0, atol=2.0e-2, rtol=0.0)
    assert not np.isclose(decoded[1], 1.0, atol=2.0e-2, rtol=0.0)


def test_k5_accepts_rho_below_one_and_negative_logrho() -> None:
    # Three repeated class centers (-0.125, 0, +0.125) keep every diagonal
    # prior representable as positive-normal FP16.  The center class equals
    # the pooled mean and has A=0, hence rho=(4/9)*(6/5)=0.533... and
    # log(rho)<0 by construction.
    rows = np.zeros((15, 160), dtype=np.float32)
    rows[0:5, :] = np.float32(-0.125)
    rows[10:15, :] = np.float32(0.125)
    labels = ("left",) * 5 + ("center",) * 5 + ("right",) * 5
    classes = ("left", "center", "right")
    state = fit_bssdg(rows, labels, classes, k_shot=5)
    assert np.any(state.logrho_fp16.astype(np.float32) < 0.0)
    assert np.isfinite(score_bssdg(state, rows[:2])).all()


def test_finite_zero_support_and_query_are_legal_when_pooled_trace_is_positive() -> None:
    rows, labels, classes = _support(1)
    rows[0] = 0.0
    state = fit_bssdg(rows, labels, classes, k_shot=1)
    scores = score_bssdg(state, np.zeros((1, 160), dtype=np.float32))
    assert scores.shape == (1, 2)
    assert np.isfinite(scores).all()


def test_all_zero_support_fails_only_at_zero_pooled_trace() -> None:
    rows = np.zeros((2, 160), dtype=np.float32)
    with pytest.raises(BSSDGError, match="trace"):
        fit_bssdg(rows, ("old", "new"), ("old", "new"), k_shot=1)


def test_duplicate_canonical_function_tuple_fails_closed() -> None:
    rows = np.zeros((10, 160), dtype=np.float32)
    rows[:, 0] = np.asarray((0, 1, 2, 3, 4, 0, 1, 2, 3, 4), dtype=np.float32)
    labels = ("old",) * 5 + ("new",) * 5
    with pytest.raises(BSSDGDuplicateFunctionError):
        fit_bssdg(rows, labels, ("old", "new"), k_shot=5)


def test_exact_top_tie_is_rejected_without_a_secondary_key() -> None:
    rows = np.zeros((2, 160), dtype=np.float32)
    rows[0, 0] = 1.0
    rows[1, 0] = -1.0
    state = fit_bssdg(rows, ("left", "right"), ("left", "right"), k_shot=1)
    raw_scores = score_bssdg(state, np.zeros((1, 160), dtype=np.float32))
    assert np.isfinite(raw_scores).all()
    with pytest.raises(BSSDGExactTieError):
        predict_bssdg_unique(state, np.zeros((1, 160), dtype=np.float32))


def test_signed_fp16_zero_and_negative_are_legal_but_subnormal_overflow_are_not() -> None:
    signed = validate_signed_fp16(np.asarray((0.0, -1.0, 1.0), dtype=np.float32))
    assert signed.dtype == np.float16
    with pytest.raises(BSSDGWireError):
        validate_signed_fp16(np.asarray((FP16_MIN_NORMAL / 2,), dtype=np.float32))
    with pytest.raises(BSSDGWireError):
        validate_signed_fp16(np.asarray((FP16_MAX * 2,), dtype=np.float32))
    with pytest.raises(BSSDGWireError):
        validate_positive_fp16(np.asarray((0.0,), dtype=np.float32))


def test_state_roundtrip_sha_and_binding_are_canonical() -> None:
    rows, labels, classes = _support(1)
    binding = BSSDGBinding(
        state_name="DA1_REG1",
        registration_name="REG1",
        canonical_sha256="a" * 64,
    )
    state = fit_bssdg(rows, labels, classes, k_shot=1, binding=binding)
    wire = serialize_bssdg_state(state)
    restored = deserialize_bssdg_state(wire)
    assert serialize_bssdg_state(restored) == wire
    assert restored.state_sha256 == state.state_sha256
    verify_bssdg_binding(restored, binding)


def test_score_uses_wire_intercept_and_tampered_intercept_is_rejected() -> None:
    rows, labels, classes = _support(1)
    state = fit_bssdg(rows, labels, classes, k_shot=1)
    query = np.zeros((2, 160), dtype=np.float32)
    query[:, 0] = np.asarray((0.2, 0.8), dtype=np.float32)
    scores = score_bssdg(state, query)
    means = state.decoded_class_means
    inv_v0 = np.float32(1.0) / state.decoded_v0
    rho = state.decoded_rho
    nu = np.float32(4.0 + state.active_k)
    delta = query[:, None, :] - means[None, :, :]
    d2 = np.sum(delta * delta * inv_v0[None, None, :], axis=2, dtype=np.float32)
    expected = (
        state.intercept_fp16.astype(np.float32)[None, :]
        - np.float32(0.5 * (160 + float(nu)))
        * np.log1p(d2 / (nu * rho[None, :])).astype(np.float32)
    )
    assert np.array_equal(scores, np.asarray(expected, dtype=np.float32))
    tampered = np.array(state.intercept_fp16, copy=True)
    tampered[0] = np.float16(float(tampered[0]) + 1.0)
    with pytest.raises(BSSDGWireError, match="intercept"):
        replace(state, intercept_fp16=tampered)


def test_query_scoring_does_not_mutate_state_or_use_extra_metadata() -> None:
    rows, labels, classes = _support(5)
    state = fit_bssdg(rows, labels, classes, k_shot=5)
    before = serialize_bssdg_state(state)
    scores = score_bssdg(state, rows[:1])
    after = serialize_bssdg_state(state)
    assert np.isfinite(scores).all()
    assert before == after
    signature = inspect.signature(fit_bssdg)
    assert "query_truth" not in signature.parameters
    assert "role" not in signature.parameters
    assert "quota" not in signature.parameters


def test_resource_receipt_has_exact_fit_query_and_latency_fields() -> None:
    rows, labels, classes = _support(5)
    state = fit_bssdg(rows, labels, classes, k_shot=5)
    receipt = bssdg_resource_receipt(state)
    assert receipt["fit_analytic_ops"] > 0
    assert receipt["query_analytic_ops_per_row"] > 0
    assert receipt["fit_latency_ns"] == 0
    assert receipt["fit_latency_observed"] is False
    assert receipt["query_latency_ns"] == 0
    assert receipt["query_rows_used_for_fit"] == 0
    assert receipt["query_state_updates"] == 0
    assert receipt["query_selection_count"] == 0
    assert receipt["state_bytes"] <= receipt["state_limit_bytes"]
    assert receipt["deploy_state_bytes"] == receipt["state_bytes"]
    assert receipt["wire_bytes"] == receipt["actual_wire_bytes"]
    assert receipt["wire_bytes"] == len(serialize_bssdg_state(state))
    assert receipt["wire_bytes"] <= receipt["state_limit_bytes"]


def _wire_registry_support(class_count: int) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    classes = tuple(f"c{index:02d}" for index in range(class_count))
    values = np.arange(1, class_count + 1, dtype=np.float32) * np.float32(0.125)
    rows = np.repeat(values[:, None], 160, axis=1).astype(np.float32)
    return rows, classes, classes


@pytest.mark.parametrize("class_count", (5, 6))
def test_c5_c6_canonical_wire_states_fit_under_limit(class_count: int) -> None:
    rows, labels, classes = _wire_registry_support(class_count)
    state = fit_bssdg(rows, labels, classes, k_shot=1)
    receipt = bssdg_resource_receipt(state)
    assert receipt["wire_bytes"] <= receipt["state_limit_bytes"]


def test_c20_canonical_wire_state_is_rejected_even_when_deploy_bytes_fit() -> None:
    rows, labels, classes = _wire_registry_support(20)
    with pytest.raises(BSSDGWireError, match="canonical wire state"):
        fit_bssdg(rows, labels, classes, k_shot=1)


def test_repeated_fit_has_deterministic_state_sha_and_wire() -> None:
    rows, labels, classes = _support(5)
    first = fit_bssdg(rows, labels, classes, k_shot=5)
    second = fit_bssdg(rows, labels, classes, k_shot=5)
    assert first.state_sha256 == second.state_sha256
    assert serialize_bssdg_state(first) == serialize_bssdg_state(second)
