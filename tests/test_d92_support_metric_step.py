"""Source-only synthetic oracles for the independent support metric subproblem.

Root owns numerical execution.  --static-check reads only these two UTF-8 source
files and parses their AST before importing NumPy or pytest.
"""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:] != ['--static-check']:
        raise SystemExit('Only --static-check is available as a script')
    root = Path(__file__).resolve().parents[1]
    for relative in ('code/cvsrffi/d92_support_metric_step.py',
                     'tests/test_d92_support_metric_step.py'):
        path = root / relative
        raw = path.read_bytes()
        ast.parse(raw.decode('utf-8', errors='strict'), filename=str(path))
        print('STATIC_UTF8_AST_OK', relative, len(raw))
    raise SystemExit(0)

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi import d92_support_metric_step as metric

EPS = np.finfo(np.float64).eps


def _no_fisher(r):
    return dict(score_jvp=np.zeros((2, 2, r)),
        probabilities=np.full((2, 2), .5), labels=np.array([0, 1]))


def _solve(g, G, B, **kwargs):
    support = _no_fisher(len(g))
    support.update(kwargs)
    return metric.solve_support_metric_step(
        gradient=np.asarray(g), ggn=np.asarray(G), physical_gram=np.asarray(B), **support)


@pytest.mark.parametrize('g,G,B,d,lam,active', [
    (.2, 0., 4., -.05, 0., False),
    (6., 4., 4., -.25, 4., True),
])
def test_one_dimensional_independent_analytic_solution(g, G, B, d, lam, active):
    result = _solve([g], [[G]], [[B]])
    np.testing.assert_allclose(result.direction, [d], rtol=512 * EPS, atol=512 * EPS)
    np.testing.assert_allclose(result.multiplier, lam, rtol=1024 * EPS, atol=1024 * EPS)
    assert result.audit['active'] is active
    assert result.audit['metric_ball_value'] <= .25
    assert result.audit['stationarity_relative_residual'] <= result.audit[
        'numerical_diagnostic_relative_tolerance']
    # Independent physical-coordinate complementarity, not a norm-gap proxy.
    physical_slack = .25 - B * float(result.direction[0]) ** 2
    relative_complementarity = abs(result.multiplier * physical_slack) / max(
        1., result.multiplier * .25)
    assert relative_complementarity <= result.audit['numerical_diagnostic_relative_tolerance']
    assert result.audit['evidence_level'] == metric.EVIDENCE_LEVEL
    assert result.audit['direction_error_bound'] is None
    assert result.audit['precise_original_problem_certificate'] is False


def test_fisher_independent_dense_variance_oracle_and_class_balance():
    J = np.array([[[1., 0.], [0., 2.]], [[2., 1.], [-1., 3.]],
        [[0., -1.], [4., 1.]], [[-2., 5.], [1., -1.]]])
    p = np.array([[.25, .75], [.5, .5], [.75, .25], [.125, .875]])
    y = np.array([0, 0, 0, 1])
    result = metric.build_class_balanced_prediction_fisher(
        score_jvp=J, probabilities=p, labels=y)
    counts = np.array([3, 1])
    oracle = np.zeros((2, 2))
    observed_label_outer = np.zeros((2, 2))
    for i in range(4):
        weight = 1. / (2 * counts[y[i]])
        oracle += weight * J[i].T @ (np.diag(p[i]) - np.outer(p[i], p[i])) @ J[i]
        observed = (p[i] - np.eye(2)[y[i]]) @ J[i]
        observed_label_outer += weight * np.outer(observed, observed)
    np.testing.assert_allclose(result.fisher, oracle, rtol=64 * EPS, atol=64 * EPS)
    np.testing.assert_allclose(result.sample_weights, [1/6, 1/6, 1/6, .5],
        rtol=0, atol=EPS)
    assert not np.allclose(result.fisher, observed_label_outer, rtol=1e-8, atol=1e-8)
    assert result.factors.shape == J.shape
    assert result.audit['dense_class_fisher_materialized'] is False
    assert result.audit['returned_numeric_state_bytes'] == sum(
        a.nbytes for a in result.arrays.values())


def test_fisher_ignores_common_log_score_jvp_and_preserves_inputs():
    J = np.arange(24, dtype=np.float64).reshape(4, 3, 2) / 8
    p = np.array([[.25, .25, .5], [.5, .25, .25],
        [.25, .5, .25], [.125, .375, .5]])
    y = np.array([0, 1, 2, 2])
    offset = np.array([[1., 3.], [-2., 4.], [0., -3.], [2., .5]])
    originals = [a.copy() for a in (J, p, y)]
    first = metric.build_class_balanced_prediction_fisher(
        score_jvp=J, probabilities=p, labels=y)
    shifted = metric.build_class_balanced_prediction_fisher(
        score_jvp=J + offset[:, None, :], probabilities=p, labels=y)
    np.testing.assert_allclose(first.fisher, shifted.fisher, rtol=128 * EPS, atol=128 * EPS)
    for actual, expected in zip((J, p, y), originals):
        np.testing.assert_array_equal(actual, expected)


def test_nonorthogonal_coordinate_change_preserves_physical_direction():
    W = np.array([[1.25, .25], [-.25, 1.], [.5, .75]])
    B = W.T @ W
    G = np.array([[.75, .125], [.125, .5]])
    g = np.array([1.75, -.875])
    J = np.array([[[1., -.5], [.25, .75]], [[-.25, 1.], [.5, -.25]],
        [[.75, .25], [-.5, 1.25]], [[-.75, -.25], [.5, 1.]]])
    p = np.array([[.25, .75], [.5, .5], [.75, .25], [.125, .875]])
    y = np.array([0, 0, 1, 1])
    T = np.array([[2., .5], [-.25, .75]])
    original = metric.solve_support_metric_step(
        gradient=g, ggn=G, physical_gram=B, score_jvp=J, probabilities=p, labels=y)
    G2, B2 = T.T @ G @ T, T.T @ B @ T
    # A quadratic uses its symmetric part.  Establish the caller's required
    # exact symmetry after this independent coordinate-oracle computation.
    G2, B2 = .5 * (G2 + G2.T), .5 * (B2 + B2.T)
    changed = metric.solve_support_metric_step(
        gradient=T.T @ g, ggn=G2, physical_gram=B2,
        score_jvp=J @ T, probabilities=p, labels=y)
    np.testing.assert_allclose(W @ original.direction, W @ T @ changed.direction,
        rtol=4096 * EPS, atol=4096 * EPS)
    np.testing.assert_allclose(changed.fisher, T.T @ original.fisher @ T,
        rtol=256 * EPS, atol=256 * EPS)
    assert original.audit['metric_ball_value'] <= .25
    assert changed.audit['metric_ball_value'] <= .25


def test_physical_gram_is_used_instead_of_identity():
    actual = _solve([.4, -.3], [[0., 0.], [0., 0.]], [[4., 0.], [0., 9.]])
    np.testing.assert_allclose(actual.direction, [-.1, 1/30],
        rtol=128 * EPS, atol=128 * EPS)
    np.testing.assert_array_equal(actual.M, [[4., 0.], [0., 9.]])
    assert actual.audit['physical_gram_replaced_by_identity'] is False
    assert actual.audit['active'] is False


def test_full_five_dimensional_input_state_and_actual_rhs_ledger():
    r = 5
    J = np.arange(4 * 2 * r, dtype=np.float64).reshape(4, 2, r) / 16
    p, y = np.full((4, 2), .5), np.array([0, 0, 1, 1])
    g = np.array([.01, -.02, .03, -.01, .02])
    G = np.diag(np.arange(1, 6) / 4) + np.ones((5, 5)) / 16
    B = np.diag(np.arange(1, 6, dtype=np.float64))
    originals = [v.copy() for v in (g, G, B, J, p, y)]
    result = metric.solve_support_metric_step(gradient=g, ggn=G,
        physical_gram=B, score_jvp=J, probabilities=p, labels=y)
    assert result.audit['active'] is False
    audit = result.audit
    assert audit['factorization_attempts'] == audit['factorizations_completed'] == 2
    assert audit['triangular_calls'] == audit['triangular_calls_completed'] == 3
    # M: one stacked gradient+5 basis substitution.  Quadratic: two one-RHS
    # substitutions.  These are actual RHS, not five independent factors.
    assert audit['triangular_rhs_columns'] == 6 + 2
    assert audit['triangular_rhs_elements'] == 5 * (6 + 2)
    assert audit['triangular_dense_work_units'] == 25 * (6 + 2)
    assert audit['spectral_check_attempts'] == audit['spectral_checks_completed'] == 2
    assert audit['secular_evaluation_count'] == 1
    assert audit['secular_iteration_count'] == 0
    assert audit['metric_factor_reused_for_stacked_rhs'] is True
    assert audit['returned_numeric_state_bytes'] == sum(
        a.nbytes for a in result.arrays.values())
    assert audit['peak_single_factor_input_output_bytes'] == 2 * 8 * r * r
    for actual, expected in zip((g, G, B, J, p, y), originals):
        np.testing.assert_array_equal(actual, expected)
    for array in result.arrays.values():
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)
    changed_audit = result.audit
    changed_audit['status'] = 'MODIFIED_COPY'
    assert result.audit['status'] == 'COMPLETED'


@pytest.mark.parametrize('r', [0, 3])
def test_zero_direction_does_not_certify_complete_head_gradient(r):
    result = _solve(np.zeros(r), np.zeros((r, r)), np.eye(r))
    np.testing.assert_array_equal(result.direction, np.zeros(r))
    assert result.audit['gradient_zero_certified'] is False
    assert result.audit['complete_head_error_bound_available'] is False
    assert result.audit['zero_dimensional_physical_space'] is (r == 0)
    assert result.audit['factorization_attempts'] == (0 if r == 0 else 2)
    assert result.audit['fisher_constructions_completed'] == 1
    assert result.audit['metric_ball_slack'] == .25


@pytest.mark.parametrize('change', [
    {'score_jvp': np.zeros((2, 2, 6))},
    {'probabilities': np.ones((2, 3))},
    {'probabilities': np.array([[.1, .1], [.5, .5]])},
    {'probabilities': np.array([[1.1, -.1], [.5, .5]])},
    {'labels': np.array([0, 0])},
    {'labels': np.array([0., 1.])},
    {'labels': np.array([False, True])},
    {'labels': np.array([0, 2])},
    {'gradient': np.array([np.nan])},
    {'ggn': np.zeros((2, 2))},
])
def test_invalid_input_shapes_probabilities_labels_and_nonfinite_are_rejected(change):
    inputs = dict(gradient=np.zeros(1), ggn=np.zeros((1, 1)),
        physical_gram=np.eye(1), **_no_fisher(1))
    inputs.update(change)
    with pytest.raises(ValueError):
        metric.solve_support_metric_step(**inputs)


def test_zero_dimension_still_rejects_invalid_probabilities_and_labels():
    with pytest.raises(ValueError):
        _solve([], np.zeros((0, 0)), np.zeros((0, 0)), probabilities=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        _solve([], np.zeros((0, 0)), np.zeros((0, 0)), labels=np.array([0, 0]))


@pytest.mark.parametrize('budget', [0, 129, True, 1.5])
def test_illegal_secular_budget_is_rejected_without_a_scientific_parameter(budget):
    with pytest.raises(ValueError):
        _solve([1.], [[0.]], [[1.]], max_secular_iterations=budget)


@pytest.mark.parametrize('G,B,code,spectra', [
    ([[0.]], [[0.]], 'NON_SPD_FACTOR', 2),
    ([[-1.]], [[1.]], 'NON_PSD_INPUT', 2),
    ([[0.]], [[-1.]], 'NON_PSD_INPUT', 1),
])
def test_non_spd_or_non_psd_fails_without_jitter_or_clipping(G, B, code, spectra):
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([1.], G, B)
    failure = caught.value
    assert failure.code == code
    assert failure.audit['spectral_checks_completed'] == spectra
    assert failure.audit['jitter_added'] is False
    assert failure.audit['curvature_clipped'] is False
    assert failure.audit['pseudoinverse_used'] is False
    np.testing.assert_array_equal(failure.arrays['ggn'], G)
    np.testing.assert_array_equal(failure.arrays['physical_gram'], B)


def test_nonsymmetric_input_is_not_silently_repaired():
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([1., 0.], [[1., .1], [0., 1.]], np.eye(2))
    assert caught.value.code == 'NONSYMMETRIC_MATRIX'
    assert caught.value.audit['symmetry_check_count'] == 1
    assert caught.value.audit['fisher_construction_attempts'] == 0


def test_bounded_secular_failure_retains_actual_factor_rhs_and_arrays():
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([10.], [[0.]], [[1.]], max_secular_iterations=1)
    failure = caught.value
    audit = failure.audit
    assert failure.code == 'SECULAR_BUDGET_EXHAUSTED'
    assert audit['secular_iteration_count'] == 1
    assert audit['secular_evaluation_count'] == 3
    assert audit['factorization_attempts'] == audit['factorizations_completed'] == 4
    assert audit['triangular_calls'] == audit['triangular_calls_completed'] == 7
    assert audit['triangular_rhs_columns'] == 8
    assert {'metric', 'hessian', 'whitening_rhs', 'secular_direction'} <= set(failure.arrays)
    assert audit['partial_numeric_state_bytes'] == sum(a.nbytes for a in failure.arrays.values())
    assert audit['seconds'] >= audit['factor_seconds'] >= 0
    assert audit['seconds'] >= audit['triangular_seconds'] >= 0


def test_failed_second_factor_keeps_completed_metric_work(monkeypatch):
    original = metric.np.linalg.cholesky
    calls = []

    def second_failure(matrix):
        calls.append(matrix.copy())
        if len(calls) == 2:
            raise np.linalg.LinAlgError('literal synthetic second-factor failure')
        return original(matrix)

    monkeypatch.setattr(metric.np.linalg, 'cholesky', second_failure)
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([.1, -.1], np.eye(2), np.eye(2))
    failure = caught.value
    assert len(calls) == 2
    assert failure.audit['factorization_attempts'] == 2
    assert failure.audit['factorizations_completed'] == 1
    assert failure.audit['triangular_calls'] == failure.audit['triangular_calls_completed'] == 1
    assert failure.audit['triangular_rhs_columns'] == 3
    assert 'metric_chol' in failure.arrays
    assert failure.audit['seconds'] >= failure.audit['factor_seconds'] >= 0
    assert failure.audit['status'] == 'TECHNICAL_FAILURE'


def test_positive_probability_weight_underflow_is_a_technical_failure():
    p = np.array([[np.nextafter(0., 1.), 1.], [.5, .5]])
    with pytest.raises(metric.SupportMetricFailure) as caught:
        metric.build_class_balanced_prediction_fisher(
            score_jvp=np.ones((2, 2, 1)), probabilities=p, labels=np.array([0, 1]))
    assert caught.value.code == 'FISHER_WEIGHT_UNDERFLOW'
    assert caught.value.audit['fisher_construction_attempts'] == 1
    assert caught.value.audit['fisher_constructions_completed'] == 0
    assert 'weighted_probabilities' in caught.value.arrays


@pytest.mark.parametrize('failed_factor_number', [3, 4])
def test_later_secular_factor_failure_has_only_current_attempt_arrays(
        monkeypatch, failed_factor_number):
    original = metric.np.linalg.cholesky
    calls = []

    def later_failure(matrix):
        calls.append(matrix.copy())
        if len(calls) == failed_factor_number:
            raise np.linalg.LinAlgError('literal later secular factor failure')
        return original(matrix)

    monkeypatch.setattr(metric.np.linalg, 'cholesky', later_failure)
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([10.], [[0.]], [[1.]])
    failure = caught.value
    audit = failure.audit
    assert audit['factorization_attempts'] == failed_factor_number
    assert audit['factorizations_completed'] == failed_factor_number - 1
    assert audit['triangular_calls'] == audit['triangular_calls_completed'] == (
        1 + 2 * (failed_factor_number - 2))
    assert audit['triangular_rhs_columns'] == 2 + 2 * (failed_factor_number - 2)
    assert audit['secular_evaluation_count'] == failed_factor_number - 1
    assert audit['secular_iteration_count'] == failed_factor_number - 3
    assert audit['secular_attempt_phase'] == 'FACTOR_ATTEMPT'
    arrays = failure.arrays
    assert 'metric_chol' in arrays
    assert 'secular_chol' not in arrays
    assert 'secular_direction' not in arrays
    assert 'secular_forward' not in arrays
    np.testing.assert_array_equal(arrays['secular_rhs'], [[-10.]])
    # The actual new multiplier/input belong together, never to the prior
    # completed factor.  Upper bracket is 20; first bisection multiplier is 10.
    multiplier = 20. if failed_factor_number == 3 else 10.
    assert float(arrays['secular_multiplier']) == multiplier
    np.testing.assert_array_equal(arrays['secular_factor_input'], [[1. + multiplier]])


def test_later_backward_triangular_failure_preserves_current_rhs_and_forward(monkeypatch):
    original_triangular = metric._triangular

    def fail_third_secular_backward(chol, rhs, ledger, *, transpose=False):
        if transpose and ledger.data['secular_evaluation_count'] == 3:
            def synthetic_finite_failure(*values):
                raise FloatingPointError('literal current backward validation failure')
            # Inject at the true triangular helper's output validation, after
            # its attempted RHS work and before completion is recorded.
            with monkeypatch.context() as scoped:
                scoped.setattr(metric, '_finite', synthetic_finite_failure)
                return original_triangular(chol, rhs, ledger, transpose=transpose)
        return original_triangular(chol, rhs, ledger, transpose=transpose)

    monkeypatch.setattr(metric, '_triangular', fail_third_secular_backward)
    with pytest.raises(metric.SupportMetricFailure) as caught:
        _solve([10.], [[0.]], [[1.]])
    failure = caught.value
    assert failure.code == 'NUMERICAL_FAILURE'
    audit = failure.audit
    assert audit['factorization_attempts'] == audit['factorizations_completed'] == 4
    assert audit['triangular_calls'] == 7
    assert audit['triangular_calls_completed'] == 6
    assert audit['triangular_rhs_columns'] == 8
    assert audit['secular_attempt_phase'] == 'BACKWARD_TRIANGULAR_ATTEMPT'
    arrays = failure.arrays
    assert 'secular_direction' not in arrays
    assert float(arrays['secular_multiplier']) == 10.
    np.testing.assert_array_equal(arrays['secular_factor_input'], [[11.]])
    np.testing.assert_array_equal(arrays['secular_rhs'], [[-10.]])
    np.testing.assert_allclose(arrays['secular_chol'], [[np.sqrt(11.)]],
        rtol=4 * EPS, atol=4 * EPS)
    np.testing.assert_allclose(arrays['secular_forward'], [[-10. / np.sqrt(11.)]],
        rtol=4 * EPS, atol=4 * EPS)
    assert audit['seconds'] >= audit['triangular_seconds'] >= 0
