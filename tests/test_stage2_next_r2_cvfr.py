from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_next_r2_cvfr as cvfr


def _case(
    *,
    k: int = 1,
    seed: int = 4084,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], cvfr.CVFRSupportBinding]:
    rng = np.random.default_rng(seed)
    classes = tuple(f"class-{index}" for index in range(6))
    labels = tuple(label for label in classes for _ in range(k))
    rows = len(labels)
    canonical = rng.normal(size=(rows, cvfr.Z_DIM)).astype(np.float32)
    # Independent, bounded view perturbations make the raw 319-column
    # Jacobian identifiable without treating either view as an extra shot.
    plus = (canonical + 0.35 * rng.normal(size=canonical.shape)).astype(np.float32)
    minus = (canonical + 0.35 * rng.normal(size=canonical.shape)).astype(np.float32)
    physical_ids = tuple(f"physical-{index}" for index in range(rows))
    binding = cvfr.CVFRSupportBinding(
        capsule_id="capsule-fixed",
        split_id="split-fixed",
        outer_key=f"rx-source-only|held-class|k={k}",
        state_id="DA1_REG1",
        k=k,
        registered_classes=classes,
        support_physical_ids=physical_ids,
    )
    return (
        np.ascontiguousarray(canonical),
        np.ascontiguousarray(plus),
        np.ascontiguousarray(minus),
        labels,
        binding,
    )


def _fit(k: int = 1, seed: int = 4084) -> cvfr.CVFRState:
    canonical, plus, minus, labels, binding = _case(k=k, seed=seed)
    return cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)


def test_helmert_basis_is_orthonormal_and_zero_sum() -> None:
    basis = cvfr.helmert_basis()
    assert basis.shape == (cvfr.Z_DIM, cvfr.SCALE_DIM)
    assert basis.dtype == np.float64
    assert not basis.flags.writeable
    assert np.allclose(basis.T @ basis, np.eye(cvfr.SCALE_DIM), atol=2.0e-15)
    assert float(np.max(np.abs(np.ones(cvfr.Z_DIM) @ basis))) <= (
        cvfr.ZERO_SUM_AUDIT_TOLERANCE
    )


def test_uniform_log_scale_gauge_is_absent() -> None:
    basis = cvfr.helmert_basis()
    uniform = np.ones(cvfr.Z_DIM, dtype=np.float64)
    assert float(np.max(np.abs(basis.T @ uniform))) <= (
        cvfr.ZERO_SUM_AUDIT_TOLERANCE
    )
    coefficients, *_ = np.linalg.lstsq(basis, uniform, rcond=None)
    assert np.linalg.norm(basis @ coefficients - uniform) == pytest.approx(
        np.sqrt(cvfr.Z_DIM)
    )


def test_exact_zero_totalizes_to_zero_and_nonzero_rows_to_unit_norm() -> None:
    values = np.zeros((3, cvfr.Z_DIM), dtype=np.float32)
    values[1, 0] = 3.0
    values[1, 1] = 4.0
    values[2, 7] = -2.0
    observed = cvfr.totalize_rows(np.ascontiguousarray(values))
    assert np.array_equal(observed[0], np.zeros(cvfr.Z_DIM))
    assert np.linalg.norm(observed[1]) == pytest.approx(1.0)
    assert np.linalg.norm(observed[2]) == pytest.approx(1.0)


def test_rank_insufficient_support_returns_legal_identity_and_still_transforms() -> None:
    canonical, _plus, _minus, labels, binding = _case(k=1)
    state = cvfr.fit_cvfr_support(
        canonical,
        canonical.copy(),
        canonical.copy(),
        labels,
        binding,
    )
    assert state.status == cvfr.STATUS_IDENTITY_UNIDENTIFIABLE
    assert state.receipt["raw_jacobian_rank"] < cvfr.PARAM_DIM
    assert np.array_equal(state.coefficients_fp16, np.zeros(cvfr.PARAM_DIM, np.float16))
    observed = cvfr.transform_cvfr(
        canonical,
        state,
        expected_binding_digest=binding.digest,
    )
    expected = cvfr.totalize_rows(canonical).astype(np.float32)
    assert np.array_equal(observed, expected)


def test_zero_support_and_zero_query_are_legal_and_audited() -> None:
    canonical, plus, minus, labels, binding = _case(k=1)
    canonical[0] = 0.0
    plus[0] = 0.0
    minus[1] = 0.0
    state = cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)
    assert state.receipt["zero_canonical_rows"] == 1
    assert state.receipt["zero_plus_rows"] == 1
    assert state.receipt["zero_minus_rows"] == 1
    assert state.receipt["zero_view_residual_count"] >= 2
    query = np.zeros((2, cvfr.Z_DIM), dtype=np.float32)
    observed = cvfr.transform_cvfr(
        query,
        state,
        expected_binding_digest=binding.digest,
    )
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    # A learned shift may totalise a zero input to a nonzero vector.  The rule
    # is nevertheless deterministic and never dispatches an auxiliary scorer.
    assert np.all((np.linalg.norm(observed, axis=1) == 0.0) | np.isclose(
        np.linalg.norm(observed, axis=1), 1.0
    ))


@pytest.mark.parametrize("k", [1, 5])
def test_identifiable_synthetic_fit_produces_nonzero_bounded_state(k: int) -> None:
    state = _fit(k=k, seed=5000 + k)
    assert state.status == cvfr.STATUS_APPLIED
    assert state.receipt["raw_jacobian_rank"] == cvfr.PARAM_DIM
    assert float(state.receipt["raw_jacobian_condition"]) <= cvfr.MAX_CONDITION
    assert abs(float(state.receipt["sum_u"])) <= float(
        state.receipt["sum_u_audit_tolerance"]
    )
    assert np.any(state.coefficients_fp16 != np.float16(0.0))
    assert float(np.linalg.norm(state.coefficients_fp16.astype(np.float64))) <= cvfr.TRUST_RADIUS
    assert state.receipt["view_rows_count_as_additional_k"] is False
    assert state.receipt["query_rows_used_for_fit"] == 0
    assert state.receipt["query_state_updates"] == 0
    assert state.receipt["truth_role_quota_inputs"] == 0
    assert state.receipt["fallback_calls"] == 0


def test_fixed_trust_solver_hits_boundary_without_changing_radius() -> None:
    design = np.ascontiguousarray(np.eye(cvfr.PARAM_DIM), dtype=np.float64)
    residual = np.ascontiguousarray(np.full(cvfr.PARAM_DIM, 20.0), dtype=np.float64)
    delta, active = cvfr.solve_trust_region(design, residual)
    assert active
    assert np.linalg.norm(delta) == pytest.approx(cvfr.TRUST_RADIUS, rel=1.0e-12)


def test_class_token_permutation_does_not_change_numeric_state() -> None:
    canonical, plus, minus, labels, binding = _case(k=1, seed=7091)
    first = cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)
    rename = {label: f"renamed-{index}" for index, label in enumerate(reversed(binding.registered_classes))}
    renamed_labels = tuple(rename[label] for label in labels)
    renamed_classes = tuple(rename[label] for label in binding.registered_classes)
    renamed_binding = cvfr.CVFRSupportBinding(
        capsule_id=binding.capsule_id,
        split_id=binding.split_id,
        outer_key=binding.outer_key,
        state_id=binding.state_id,
        k=binding.k,
        registered_classes=tuple(reversed(renamed_classes)),
        support_physical_ids=binding.support_physical_ids,
    )
    second = cvfr.fit_cvfr_support(
        canonical,
        plus,
        minus,
        renamed_labels,
        renamed_binding,
    )
    assert first.status == second.status
    assert np.array_equal(first.coefficients_fp16, second.coefficients_fp16)
    assert np.array_equal(first.rms_fp32, second.rms_fp32)


def test_state_wire_roundtrip_is_canonical_and_preserves_transform() -> None:
    canonical, plus, minus, labels, binding = _case(k=1, seed=9017)
    state = cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)
    wire = state.to_wire()
    loaded = cvfr.CVFRState.from_wire(wire)
    assert loaded.to_wire() == wire
    assert loaded.status == state.status
    assert np.array_equal(loaded.coefficients_fp16, state.coefficients_fp16)
    assert np.array_equal(loaded.rms_fp32, state.rms_fp32)
    assert loaded.receipt["resource"]["numeric_wire_bytes"] == 642
    expected = cvfr.transform_cvfr(
        canonical,
        state,
        expected_binding_digest=binding.digest,
    )
    observed = cvfr.transform_cvfr(
        canonical,
        loaded,
        expected_binding_digest=binding.digest,
    )
    assert np.array_equal(observed, expected)


def test_query_transform_is_repeatable_and_does_not_mutate_state() -> None:
    canonical, plus, minus, labels, binding = _case(k=1, seed=117)
    state = cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)
    before_coefficients = state.coefficients_fp16.copy()
    before_rms = state.rms_fp32.copy()
    before_wire = state.to_wire()
    first = cvfr.transform_cvfr(
        plus,
        state,
        expected_binding_digest=binding.digest,
    )
    second = cvfr.transform_cvfr(
        plus,
        state,
        expected_binding_digest=binding.digest,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(state.coefficients_fp16, before_coefficients)
    assert np.array_equal(state.rms_fp32, before_rms)
    assert state.to_wire() == before_wire
    assert not state.coefficients_fp16.flags.writeable
    assert not state.rms_fp32.flags.writeable


def test_fit_and_transform_api_exclude_forbidden_runtime_inputs() -> None:
    fit_parameters = set(inspect.signature(cvfr.fit_cvfr_support).parameters)
    assert fit_parameters == {
        "canonical_support",
        "phase_plus_support",
        "phase_minus_support",
        "support_labels",
        "binding",
    }
    transform_parameters = set(inspect.signature(cvfr.transform_cvfr).parameters)
    assert transform_parameters == {"features", "state", "expected_binding_digest"}
    forbidden = {
        "query_truth",
        "query_label",
        "old_role",
        "new_role",
        "class_quota",
        "source_sample",
        "clean_sample",
    }
    assert fit_parameters.isdisjoint(forbidden)
    assert transform_parameters.isdisjoint(forbidden)


def test_nonfinite_shape_binding_and_state_overflow_fail_closed() -> None:
    canonical, plus, minus, labels, binding = _case(k=1)
    nonfinite = canonical.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(cvfr.CVFRError, match="finite"):
        cvfr.fit_cvfr_support(nonfinite, plus, minus, labels, binding)
    with pytest.raises(cvfr.CVFRError, match=r"\[6,160\]"):
        cvfr.fit_cvfr_support(canonical[:, :-1].copy(), plus, minus, labels, binding)
    with pytest.raises(cvfr.CVFRError, match="binding mismatch"):
        state = cvfr.fit_cvfr_support(canonical, plus, minus, labels, binding)
        cvfr.transform_cvfr(
            canonical,
            state,
            expected_binding_digest="0" * 64,
        )
    with pytest.raises(cvfr.CVFRError, match="trust cap"):
        cvfr.CVFRState(
            status=cvfr.STATUS_APPLIED,
            coefficients_fp16=np.ascontiguousarray(
                np.full(cvfr.PARAM_DIM, np.float16(1.0)), dtype=np.float16
            ),
            rms_fp32=np.ascontiguousarray([1.0], dtype=np.float32),
            binding_digest="1" * 64,
            receipt={
                "schema": "cvs.phase2.next_r2.cvfr_receipt.v1",
                "status": cvfr.STATUS_APPLIED,
                "binding_digest": "1" * 64,
            },
        )


def test_wire_hash_tamper_is_rejected() -> None:
    state = _fit(k=1, seed=1239)
    wire = bytearray(state.to_wire())
    wire[-1] ^= 1
    with pytest.raises(cvfr.CVFRError, match="hash mismatch"):
        cvfr.CVFRState.from_wire(bytes(wire))


def test_binding_rejects_wrong_protocol_duplicate_ids_and_bad_k() -> None:
    base = dict(
        capsule_id="capsule",
        split_id="split",
        outer_key="outer",
        state_id="DA1_REG1",
        registered_classes=("a", "b"),
    )
    with pytest.raises(cvfr.CVFRError, match="p2_min_v1"):
        cvfr.CVFRSupportBinding(
            **base,
            k=1,
            support_physical_ids=("p0", "p1"),
            protocol_schema="wrong",
        )
    with pytest.raises(cvfr.CVFRError, match="unique"):
        cvfr.CVFRSupportBinding(
            **base,
            k=1,
            support_physical_ids=("p0", "p0"),
        )
    with pytest.raises(cvfr.CVFRError, match="exactly 1 or 5"):
        cvfr.CVFRSupportBinding(
            **base,
            k=2,
            support_physical_ids=("p0", "p1", "p2", "p3"),
        )
