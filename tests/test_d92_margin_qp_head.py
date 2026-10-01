"""Synthetic component checks; independent finite-feature primal KKT oracle.

No production artifacts or feature caches are read. The imported certificate's
enumeration is intentionally tiny and is not a candidate implementation.
"""
from pathlib import Path
import sys

if __name__ == "__main__":
    import ast
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    print("AST_UTF8_OK test_d92_margin_qp_head.py")
    raise SystemExit(0)

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
import cvsrffi.d92_margin_qp_head as candidate
from test_d92_margin_joint_math_certificate import (
    _active_fixture, _centered_labels, _constraints, _feature_primal,
    _multi_active_fixture, _regular_vjp, _three_class_fixture, _vec,
)


def fit(K, M, labels, old, **limits):
    options = dict(max_transitions=100, max_factor_buffer_bytes=1_000_000)
    options.update(limits)
    return candidate.fit_margin_qp_head(K=K, M=M, labels=np.asarray(labels),
                                        old_indices=np.asarray(old), **options)


def data(kind):
    if kind in ("active", "multi"):
        original, _ = (_active_fixture() if kind == "active" else _multi_active_fixture())
        phi = np.linalg.cholesky(original.K)
        psi = np.linalg.solve(phi, original.L.T).T
        return phi, psi, original.M, np.array([0, original.M.shape[1] - 1]), np.array([0]), original.M_H
    if kind == "three":
        phi, psi, M, _, _, _, M_H, old = _three_class_fixture()
        return phi, psi, M, np.array([0, 1, 2, 2]), old, M_H
    phi = np.array([[.5], [.5], [-.2]]) if kind == "duplicate" else np.zeros((3, 0))
    psi = np.array([[.8], [-.4]]) if kind == "duplicate" else np.zeros((2, 0))
    return phi, psi, np.zeros((3, 2)), np.array([0, 1, 1]), np.array([0, 1]), np.zeros((2, 2))


def feature_solution(K, L, M, labels, old, M_H):
    """Independent finite feature problem; no candidate response matrix/solve."""
    phi = np.linalg.cholesky(K)
    psi = np.linalg.solve(phi, L.T).T
    R = _centered_labels(labels, M.shape[1]) - M
    D, delta, _, _ = _constraints(M, labels, old)
    primal = _feature_primal(phi, R, M, D, delta)
    return M_H + psi @ primal["theta"][:-1] + primal["theta"][-1]


@pytest.mark.parametrize("kind", ["active", "multi", "three", "duplicate", "zero"])
def test_independent_finite_feature_primal_oracle_and_all_kkt_equations(kind):
    phi, psi, M, labels, old, M_H = data(kind)
    K, L = phi @ phi.T, psi @ phi.T
    state = fit(K, M, labels, old)
    D, delta, pairs, margins = _constraints(M, labels, old)
    R = _centered_labels(labels, M.shape[1]) - M
    primal = _feature_primal(phi, R, M, D, delta)
    np.testing.assert_allclose(state.train_scores, M + primal["residual"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(phi.T @ state.alpha, primal["theta"][:-1], atol=2e-9, rtol=0)
    np.testing.assert_allclose(state.b, primal["theta"][-1], atol=2e-9, rtol=0)
    predicted = candidate.predict_margin_qp_head(state, L=L, M=M_H)
    expected = M_H + psi @ primal["theta"][:-1] + primal["theta"][-1]
    np.testing.assert_allclose(predicted, expected, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(state.arrays["margins"], margins)
    assert state.audit["constraint_count"] == len(old) * (M.shape[1] - 1) == len(pairs)
    mu = state.arrays["multipliers"]
    V = (D.T @ mu).reshape(M.shape, order="F")
    np.testing.assert_allclose((K + np.eye(len(K))) @ state.alpha + state.b - R - V,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose(np.sum(state.alpha, axis=0), 0, atol=2e-9, rtol=0)
    slack = D @ _vec(state.train_scores) - delta
    assert np.min(slack) >= -state.audit["final_residuals"]["residual_tolerance"]
    assert np.min(mu) >= -state.audit["final_residuals"]["dual_tolerance"]
    np.testing.assert_allclose(mu * slack, 0, atol=2e-9, rtol=0)
    report = state.audit["final_residuals"]
    assert report["primal_objective"] == pytest.approx(primal["objective"], abs=2e-9)
    assert report["dual_objective"] == pytest.approx(primal["objective"], abs=2e-9)
    assert abs(report["primal_dual_gap"]) <= report["gap_tolerance"]
    assert state.audit["status"] == "NUMERIC_KKT_CERTIFIED"
    assert state.audit["training_method_complete"] is False


def test_zero_kernel_keeps_free_intercept_and_all_physical_label_constraints():
    K = np.zeros((3, 3))
    M = np.zeros((3, 2))
    state = fit(K, M, [0, 0, 0], [0, 1])
    np.testing.assert_allclose(state.b, [.5, -.5], atol=2e-12)
    np.testing.assert_allclose(state.train_scores, np.tile([.5, -.5], (3, 1)), atol=2e-12)
    assert state.audit["n"] == 3 and state.audit["m"] == 2
    assert state.arrays["old_indices"].tolist() == [0, 1]
    opposite = fit(K, M, [0, 1, 1], [0, 1])
    np.testing.assert_allclose(opposite.b, 0, atol=2e-12)
    assert len(opposite.arrays["slack"]) == 2
    with pytest.raises(candidate.UnsupportedJacobian, match="UNSUPPORTED_ACTIVE_JACOBIAN"):
        candidate.margin_qp_head_vjp(opposite, L=np.zeros((1, 3)), G=np.array([[.5, -.5]]))


def test_minimum_prior_margin_is_not_clamped_and_every_column_competes():
    M = np.array([[0., 2., 1.], [.1, -.2, .3]])
    labels = np.array([0, 2])
    state = fit(np.eye(2), M, labels, [0])
    assert state.arrays["margins"].tolist() == [-2.]
    assert state.arrays["delta"].tolist() == [-2., -2.]
    assert state.arrays["pair_other"].tolist() == [1, 2]
    D, delta, _, _ = _constraints(M, labels, [0])
    primal = _feature_primal(np.eye(2), _centered_labels(labels, 3) - M, M, D, delta)
    np.testing.assert_allclose(state.train_scores, M + primal["residual"], atol=2e-9)


def test_zero_start_full_blocking_scan_and_deterministic_add_remove_transitions():
    # With K=I, P_00=.75. Initial unconstrained pair slacks are -.1,-1.
    # Constraint 0 blocks at zero, but becomes redundant after constraint 1.
    K = np.eye(2)
    M = np.array([[0., 0., -.1], [0., -3.4, -6.7]])
    state = fit(K, M, [0, 2], [0])
    events = state.audit["events"]
    assert events[0]["rho"] == 0 and events[0]["working"] == []
    assert events[0]["step"] == 0 and events[0]["blocker"] == 0
    assert any(event["action"] == "REMOVE_NEGATIVE_MULTIPLIER" for event in events)
    assert state.audit["full_constraint_scans"] == len(events) + 1
    assert all(event["minimum_slack"] >= -event["slack_tolerance"] for event in events)
    assert all(event["predicted_objective_change"] <= 0 for event in events if event["action"] == "MOVE")
    assert all(event["blocker_schur"] > event["blocker_rank_tolerance"] for event in events if event.get("blocker") is not None)
    repeat = fit(K, M, [0, 2], [0])
    assert events == repeat.audit["events"]
    D, delta, _, _ = _constraints(M, np.array([0, 2]), [0])
    primal = _feature_primal(np.eye(2), _centered_labels([0, 2], 3) - M, M, D, delta)
    np.testing.assert_allclose(state.train_scores, M + primal["residual"], atol=2e-9)


def test_actual_rhs_factor_spectrum_and_numeric_buffer_accounting():
    original, _ = _multi_active_fixture()
    state = fit(original.K, original.M, [0, 2], [0])
    audit = state.audit
    solves = audit["solves"]
    assert audit["triangular_calls"] == len(solves)
    assert audit["triangular_rhs_columns"] == sum(item["rhs_columns"] for item in solves)
    assert audit["triangular_rhs_elements"] == sum(item["dimension"] * item["rhs_columns"] for item in solves)
    assert audit["triangular_dense_work_units"] == sum(item["dimension"] ** 2 * item["rhs_columns"] for item in solves)
    factors = audit["factorizations"]
    assert audit["factorization_attempts"] == audit["factorizations_completed"] == len(factors)
    assert audit["condition_estimation_calls"] == len(factors)
    assert all(item["status"] == "COMPLETED" for item in factors + solves)
    assert audit["spectral_checks"] == 1 and audit["spectral_cubic_dimension_units"] == 2 ** 3
    max_working = max((item["dimension"] for item in factors if item["system"] == "working"), default=0)
    assert audit["peak_factor_buffer_bytes"] == 16 * (2 ** 2 + max_working ** 2)
    assert audit["A_input_numeric_bytes"] == audit["A_factor_numeric_bytes"] == 8 * 2 ** 2
    assert audit["P_old_numeric_bytes"] == 8
    assert audit["returned_numeric_state_bytes"] == sum(value.nbytes for value in state.arrays.values())
    assert audit["head_expansion_numeric_bytes"] == state.alpha.nbytes + state.b.nbytes
    assert audit["process_peak_memory_bytes"] is None and audit["wall_seconds"] is None
    assert audit["deployment_bytes"] is None and audit["no_full_dual_hessian_preallocation"]


def test_input_isolation_and_returned_arrays_cannot_be_made_writeable():
    K = np.eye(2)
    M = np.zeros((2, 2))
    labels = np.array([0, 0])
    old = np.array([0])
    state = fit(K, M, labels, old)
    K[:] = 99
    M[:] = 88
    labels[:] = 1
    old[:] = 1
    np.testing.assert_array_equal(state.arrays["K"], np.eye(2))
    np.testing.assert_array_equal(state.arrays["M"], np.zeros((2, 2)))
    assert state.arrays["labels"].tolist() == [0, 0]
    for value in state.arrays.values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        state.arrays["alpha"] = np.zeros((2, 2))
    audit = state.audit
    audit["events"][0]["rho"] = 99
    assert state.audit["events"][0]["rho"] == 0


def test_explicit_finite_resource_limits_preserve_failure_state():
    original, _ = _active_fixture()
    args = (original.K, original.M, [0, 1], [0])
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(*args, max_transitions=1)
    error = failure.value
    assert error.code == "TRANSITION_LIMIT"
    assert error.audit["transitions"] == 1
    assert error.arrays["working_set"].tolist() == [0]
    assert float(error.arrays["rho"]) == 0
    assert error.audit["status"] == "TECHNICAL_FAILURE"
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(*args, max_factor_buffer_bytes=1)
    assert failure.value.code == "FACTOR_BUFFER_LIMIT"
    assert failure.value.audit["factorization_attempts"] == 0
    assert failure.value.audit["peak_factor_buffer_bytes"] == 0
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(*args, max_factor_buffer_bytes=64)
    assert failure.value.code == "FACTOR_BUFFER_LIMIT"
    assert failure.value.audit["factorizations_completed"] == 1
    assert failure.value.audit["peak_factor_buffer_bytes"] == 64
    assert failure.value.arrays["working_set"].tolist() == [0]
    for limits in ({"max_transitions": 0}, {"max_transitions": True},
                   {"max_factor_buffer_bytes": float("inf")}, {"max_factor_buffer_bytes": -1}):
        with pytest.raises(ValueError, match="finite positive integer"):
            fit(*args, **limits)
    with pytest.raises(TypeError):
        candidate.fit_margin_qp_head(K=original.K, M=original.M, labels=[0, 1], old_indices=[0])


def test_partial_cholesky_failure_keeps_actual_work_without_retry(monkeypatch):
    original, _ = _active_fixture()
    real = candidate.cholesky
    calls = []
    def fail_working(matrix, **kwargs):
        calls.append(matrix.copy())
        if len(calls) == 2:
            raise np.linalg.LinAlgError("synthetic working factor failure")
        return real(matrix, **kwargs)
    monkeypatch.setattr(candidate, "cholesky", fail_working)
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(original.K, original.M, [0, 1], [0])
    audit = failure.value.audit
    assert failure.value.code == "WORKING_RANK_UNRESOLVED"
    assert len(calls) == audit["factorization_attempts"] == 2
    assert audit["factorizations_completed"] == audit["condition_estimation_calls"] == 1
    assert audit["factorizations"][-1]["status"] == "FAILED"
    assert audit["peak_factor_buffer_bytes"] == 64 + 8
    assert failure.value.arrays["working_set"].tolist() == [0]


def assert_current_compact_failure_state(error):
    """Reconstruct saved rho/V in independent finite-feature coordinates."""
    values = error.arrays
    K, M, labels, old = (values[name] for name in ("K", "M", "labels", "old_indices"))
    phi = np.linalg.cholesky(K)
    X = np.column_stack((phi, np.ones(len(K))))
    H = X.T @ X + np.diag(np.r_[np.ones(phi.shape[1]), 0.])
    R = _centered_labels(labels, M.shape[1]) - M
    theta0 = np.linalg.solve(H, X.T @ R)
    theta_response = np.linalg.solve(H, X[old].T @ values["V"])
    scores = M + X @ (float(values["rho"]) * theta0 + theta_response)
    D, delta, _, _ = _constraints(M, labels, old)
    actual_slack = D @ _vec(scores) - delta
    np.testing.assert_allclose(values["slack"], actual_slack, atol=2e-12, rtol=0)
    np.testing.assert_allclose(values["slack"], [0., 0.], atol=2e-12, rtol=0)
    assert float(values["rho"]) == pytest.approx(1 / 12, abs=2e-12)
    np.testing.assert_allclose(values["V"], [[1 / 180, -1 / 180, 0.]], atol=2e-12, rtol=0)
    assert values["working_set"].tolist() == [0, 1]
    # Directions/multipliers from the previous working set must not appear as
    # current-state values when the new working solve never completed.
    assert "last_working_multipliers" not in values and "direction_energy" not in values
    assert error.audit["compact_snapshot_rebuilds"] == 3
    assert error.audit["full_constraint_scans"] == 3
    assert error.audit["compact_snapshot_dense_work_units"] == 3 * 1 * 1 * 3
    assert len(error.audit["events"]) == 2


def test_transition_limit_snapshot_slack_matches_post_move_rho_v_working_set():
    K = np.eye(2)
    M = np.array([[0., 0., -.1], [0., -3.4, -6.7]])
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(K, M, [0, 2], [0], max_transitions=2)
    error = failure.value
    assert error.code == "TRANSITION_LIMIT"
    assert error.audit["transitions"] == 2
    assert error.audit["factorization_attempts"] == error.audit["factorizations_completed"] == 2
    assert_current_compact_failure_state(error)


def test_next_factor_failure_snapshot_slack_matches_post_move_rho_v_working_set(monkeypatch):
    K = np.eye(2)
    M = np.array([[0., 0., -.1], [0., -3.4, -6.7]])
    real = candidate.cholesky
    calls = []
    def fail_new_working_set(matrix, **kwargs):
        calls.append(matrix.copy())
        if len(calls) == 3:
            raise np.linalg.LinAlgError("synthetic failure after nonzero compact MOVE")
        return real(matrix, **kwargs)
    monkeypatch.setattr(candidate, "cholesky", fail_new_working_set)
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(K, M, [0, 2], [0])
    error = failure.value
    assert error.code == "WORKING_RANK_UNRESOLVED"
    assert len(calls) == error.audit["factorization_attempts"] == 3
    assert error.audit["factorizations_completed"] == error.audit["condition_estimation_calls"] == 2
    assert error.audit["transitions"] == 3
    assert error.audit["factorizations"][-1]["status"] == "FAILED"
    assert_current_compact_failure_state(error)


def test_original_equation_readback_rejects_corrupted_final_solve(monkeypatch):
    original, _ = _active_fixture()
    real = candidate._affine
    def corrupt(ledger, factor, z, s, rhs, system):
        alpha, b = real(ledger, factor, z, s, rhs, system)
        if system == "affine_final_readback":
            alpha = alpha.copy()
            alpha[0, 0] += .1
        return alpha, b
    monkeypatch.setattr(candidate, "_affine", corrupt)
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(original.K, original.M, [0, 1], [0])
    assert failure.value.code == "FINAL_KKT_READBACK_FAILED"
    assert failure.value.audit["final_residuals"]["stationarity"] > .1
    assert "alpha" in failure.value.arrays and "train_scores" in failure.value.arrays


def test_non_psd_and_unresolved_affine_condition_fail_without_jitter():
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(np.diag([-1., 1.]), np.zeros((2, 2)), [0, 1], [0])
    assert failure.value.code == "NON_PSD_KERNEL"
    assert failure.value.audit["spectral_checks"] == 1
    assert failure.value.audit["factorizations_completed"] == 0
    with pytest.raises(candidate.MarginQPFailure) as failure:
        fit(np.diag([1e20, 0.]), np.zeros((2, 2)), [0, 1], [0])
    assert failure.value.code == "AFFINE_CONDITION_UNRESOLVED"
    assert failure.value.audit["factorizations_completed"] == 1


@pytest.mark.parametrize("change", ["asymmetric", "nan", "label_float", "label_range", "old_duplicate", "old_empty", "shape"])
def test_illegal_input_contracts_are_rejected(change):
    K, M, labels, old = np.eye(2), np.zeros((2, 2)), np.array([0, 1]), np.array([0])
    if change == "asymmetric":
        K[0, 1] = .2
    elif change == "nan":
        M[0, 0] = np.nan
    elif change == "label_float":
        labels = labels.astype(float)
    elif change == "label_range":
        labels[1] = 2
    elif change == "old_duplicate":
        old = np.array([0, 0])
    elif change == "old_empty":
        old = np.array([], dtype=int)
    else:
        K = np.eye(3)
    with pytest.raises(ValueError):
        fit(K, M, labels, old)


@pytest.mark.parametrize("fixture", [_active_fixture, _multi_active_fixture])
def test_regular_vjp_against_independent_primal_directional_difference(fixture):
    original, G = fixture()
    labels = np.array([0, original.M.shape[1] - 1])
    state = fit(original.K, original.M, labels, [0])
    adjoint = candidate.margin_qp_head_vjp(state, L=original.L, G=G)
    dK = np.array([[.17, -.11], [-.11, .23]])
    dL = np.array([[.13, -.07], [-.09, .16]])
    epsilon = 2e-6
    predictions = []
    for sign in (-1, 1):
        K = original.K + sign * epsilon * dK
        L = original.L + sign * epsilon * dL
        predictions.append(feature_solution(K, L, original.M, labels, [0], original.M_H))
        perturbed = fit(K, original.M, labels, [0])
        assert np.array_equal(perturbed.arrays["working_set"], state.arrays["working_set"])
        candidate.margin_qp_head_vjp(perturbed, L=L, G=G)  # Both endpoints remain regular.
    finite = float(np.sum(G * (predictions[1] - predictions[0])) / (2 * epsilon))
    analytic = float(np.sum(adjoint.arrays["K"] * dK) + np.sum(adjoint.arrays["L"] * dL))
    assert analytic == pytest.approx(finite, rel=3e-6, abs=2e-8)
    np.testing.assert_allclose(adjoint.arrays["K"], adjoint.arrays["K"].T, atol=1e-14)
    np.testing.assert_allclose(adjoint.arrays["g_b"], np.sum(G, axis=0), atol=1e-14)
    assert np.linalg.norm(adjoint.arrays["g_b"]) > .1
    assert adjoint.audit["factorization_attempts"] == adjoint.audit["spectral_checks"] == 0
    assert adjoint.audit["triangular_calls"] == 6
    assert adjoint.audit["reused_factor_bytes"] == state.arrays["chol_A"].nbytes + state.arrays["chol_working"].nbytes
    residuals = adjoint.audit["adjoint_residuals"]
    assert all(value <= residuals["tolerance"] for key, value in residuals.items() if key != "tolerance")


@pytest.mark.parametrize("omission", ["intercept", "multiplier"])
def test_missing_constant_or_multiplier_response_is_a_detectable_wrong_gradient(omission):
    original, G = _active_fixture()
    state = fit(original.K, original.M, [0, 1], [0])
    good = candidate.margin_qp_head_vjp(state, L=original.L, G=G).arrays
    bad = _regular_vjp(original, G, include_intercept=omission != "intercept", include_multiplier=omission != "multiplier")
    direction = good["K"] - bad["K"]
    norm = np.linalg.norm(direction)
    assert norm > 1e-4
    direction /= norm
    epsilon = 2e-6
    plus = feature_solution(original.K + epsilon * direction, original.L, original.M, [0, 1], [0], original.M_H)
    minus = feature_solution(original.K - epsilon * direction, original.L, original.M, [0, 1], [0], original.M_H)
    finite = np.sum(G * (plus - minus)) / (2 * epsilon)
    assert np.sum(good["K"] * direction) == pytest.approx(finite, rel=3e-6, abs=2e-8)
    assert abs(finite - np.sum(bad["K"] * direction)) > .9 * norm


def test_no_active_adjoint_has_nonzero_constant_rhs_and_switch_is_rejected():
    K = np.array([[1., .2], [.2, .8]])
    L, G = np.array([[.3, .6]]), np.array([[.4, -.4]])
    state = fit(K, np.zeros((2, 2)), [0, 0], [0])
    assert len(state.arrays["working_set"]) == 0
    adjoint = candidate.margin_qp_head_vjp(state, L=L, G=G)
    saddle = np.block([[K + np.eye(2), np.ones((2, 1))], [np.ones((1, 2)), np.zeros((1, 1))]])
    expected = np.linalg.solve(saddle, np.vstack((L.T @ G, G.sum(axis=0))))
    np.testing.assert_allclose(adjoint.arrays["T_G"], expected[:-1], atol=2e-12)
    np.testing.assert_allclose(adjoint.arrays["t_G"], expected[-1], atol=2e-12)
    assert adjoint.audit["triangular_calls"] == 2 and adjoint.arrays["eta"].size == 0
    Y = _centered_labels([0, 1], 2)
    switch = fit(np.eye(2), Y, [0, 1], [0])
    np.testing.assert_allclose(switch.arrays["multipliers"], 0, atol=2e-12)
    with pytest.raises(candidate.UnsupportedJacobian) as failure:
        candidate.margin_qp_head_vjp(switch, L=L, G=G)
    assert failure.value.audit["status"] == "UNSUPPORTED_JACOBIAN"


def test_full_column_prediction_single_record_and_class_row_permutations():
    phi, psi, M, labels, old, M_H = data("three")
    K, L = phi @ phi.T, psi @ phi.T
    state = fit(K, M, labels, old)
    predictions = candidate.predict_margin_qp_head(state, L=L, M=M_H)
    singles = np.vstack([candidate.predict_margin_qp_head(state, L=L[i:i+1], M=M_H[i:i+1]) for i in range(len(L))])
    np.testing.assert_allclose(predictions, singles, atol=2e-12)
    row_order, class_order = np.array([3, 1, 0, 2]), np.array([2, 0, 1])
    inverse = np.argsort(class_order)
    permuted = fit(K[np.ix_(row_order, row_order)], M[row_order][:, class_order],
                   inverse[labels[row_order]], np.flatnonzero(np.isin(row_order, old)))
    np.testing.assert_allclose(permuted.train_scores, state.train_scores[row_order][:, class_order], atol=2e-9)
    other = candidate.predict_margin_qp_head(permuted, L=L[:, row_order], M=M_H[:, class_order])
    np.testing.assert_allclose(other, predictions[:, class_order], atol=2e-9)
    with pytest.raises(ValueError):
        candidate.predict_margin_qp_head(state, L=L[:, :1], M=M_H)
    with pytest.raises(TypeError):
        candidate.predict_margin_qp_head(state, L=L, M=M_H, labels=[0, 1])
    with pytest.raises(ValueError):
        predictions.setflags(write=True)


def test_centered_prior_preserves_class_sum_without_score_freezing():
    M = np.array([[.5, 0., -.5], [-.5, .5, 0.], [0., -.5, .5]])
    state = fit(np.eye(3), M, [0, 1, 2], [0, 1, 2])
    np.testing.assert_allclose(state.train_scores.sum(axis=1), 0, atol=2e-12)
    np.testing.assert_allclose(state.alpha.sum(axis=1), 0, atol=2e-12)
    assert np.linalg.norm(state.train_scores - M) > .1
    assert np.min(state.arrays["slack"]) >= -state.audit["final_residuals"]["residual_tolerance"]
