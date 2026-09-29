"""Independent synthetic numerical/protocol checks; no experimental inputs."""
import itertools
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint, Bounds
from cvsrffi import d92_branch_local_margin as margin
from cvsrffi import d92_branch_local_ridge as local
from cvsrffi import d92_branch_interaction as interaction


def episode(c=3, k=3, seed=91):
    rng = np.random.default_rng(seed)
    raw = {name: rng.normal(size=(c*k, dim)) for name, dim in zip(local._NAMES, (160, 96, 160, 160, 160))}
    return dict(**raw, support_labels=np.repeat(np.arange(c), k),
        support_ids=[f'p{i:04d}' for i in range(c*k)], classes=[f'c{i:02d}' for i in range(c)], old_classes=['c00'])


def features(args):
    return {name: args[name] for name in local._NAMES}


def assert_certificate(audit):
    assert audit['certified']
    assert audit['relative_duality_gap'] <= np.sqrt(np.finfo(float).eps)
    assert audit['relative_kkt_residual'] <= np.sqrt(np.finfo(float).eps)
    assert audit['loss_total'] == audit['loss_data']+audit['loss_ridge']
    assert audit['factorization_calls'] == 0
    assert audit['optimizer_steps'] == audit['sweeps']*audit['train_physical_count']


@pytest.mark.parametrize('diagonal', [.01, 1., 100.])
def test_exact_row_block_matches_exhaustive_active_set_oracle(diagonal):
    rng = np.random.default_rng(7)
    for a in [rng.normal(size=5), np.ones(5), -np.ones(5), np.array([3., 3., 1., -2., 0.])]:
        q = diagonal*np.eye(5)+(diagonal+1)*np.ones((5, 5))
        valid = []
        for bits in itertools.product([False, True], repeat=5):
            active = np.asarray(bits)
            candidate = np.zeros(5)
            candidate[active] = np.linalg.solve(q[np.ix_(active, active)], a[active])
            gradient = q@candidate-a
            if np.all(candidate >= -1e-12) and np.all(gradient[~active] >= -1e-12):
                valid.append(candidate)
        assert len(valid)
        actual = margin._row_solution(a, diagonal)
        np.testing.assert_allclose(actual, valid[0], atol=1e-12, rtol=1e-12)
        perm = np.array([4, 2, 0, 3, 1])
        np.testing.assert_allclose(margin._row_solution(a[perm], diagonal), actual[perm], atol=1e-12)


def test_tiny_diagonal_equal_rivals_does_not_cancel_mass():
    for d in [1e-14, 1e-100, np.nextafter(0., 1.)]:
        actual = margin._row_solution(np.ones(25), d)
        np.testing.assert_allclose(actual, np.full(25, 1/(d+25*(d+1))), rtol=1e-14)


def test_dual_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 3)); x -= x.mean(axis=0)
    gram = x@x.T
    labels = np.array([0, 1, 2, 0])
    rivals = np.array([[j for j in range(3) if j != y] for y in labels])
    beta = rng.uniform(.1, .5, (4, 2))
    def objective(value):
        alpha, s = margin._alpha(value, labels, rivals, 3)
        return .5*np.sum(alpha*(gram@alpha))+.5*np.sum(s*s)-s.sum()
    alpha, s = margin._alpha(beta, labels, rivals, 3)
    f = gram@alpha
    analytic = s[:, None]+f[np.arange(4), labels, None]-f[np.arange(4)[:, None], rivals]-1
    for i, c in np.ndindex(beta.shape):
        delta = np.zeros_like(beta); delta[i, c] = 1e-6
        finite = (objective(beta+delta)-objective(beta-delta))/2e-6
        np.testing.assert_allclose(finite, analytic[i, c], rtol=1e-8, atol=1e-8)


def test_dual_solver_matches_independent_primal_constrained_qp():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(6, 4)); x -= x.mean(axis=0)
    labels = np.array([0, 1, 2, 0, 1, 2]); n, d, c = 6, 4, 3
    alpha, audit = margin._optimize(x@x.T, labels, c)
    # Independently optimize W and slack using explicit linear constraints.
    constraint = np.zeros((n*(c-1), d*c+n))
    row = 0
    for i, y in enumerate(labels):
        for other in range(c):
            if other == y:
                continue
            constraint[row, np.arange(d)*c+y] = x[i]
            constraint[row, np.arange(d)*c+other] = -x[i]
            constraint[row, d*c+i] = 1
            row += 1
    initial = np.r_[np.zeros(d*c), np.ones(n)]
    result = minimize(lambda value: .5*value@value, initial, jac=lambda value: value,
        method='SLSQP', constraints=LinearConstraint(constraint, 1., np.inf),
        bounds=Bounds(np.r_[np.full(d*c, -np.inf), np.zeros(n)], np.inf),
        options=dict(ftol=1e-12, maxiter=2000))
    assert result.success, result.message
    np.testing.assert_allclose(audit['primal_objective'], result.fun, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(x.T@alpha, result.x[:d*c].reshape(d, c), atol=2e-6, rtol=2e-6)
    assert audit['certified']


def test_binary_one_shot_analytic_solution_and_same_boundary_as_local_ridge():
    args = episode(c=2, k=1)
    state = margin.fit_branch_local_margin(**args)
    audit = state.audit_dict()['final_fit']; assert_certificate(audit)
    # Centered two-record Gram is d[[1,-1],[-1,1]]; beta=1/(1+4d).
    d = audit['interaction_centered_trace']/2
    expected = np.array([[1., -1.], [-1., 1.]])/(1+4*d)
    np.testing.assert_allclose(state.alpha, expected, atol=2e-8)
    control = local.fit_branch_local_ridge(**args)
    held = features(episode(c=2, k=10, seed=6))
    np.testing.assert_array_equal(state.predict(**held), control.predict(**held))


def test_geometry_is_frozen_local_ridge_and_candidate_never_cholesky():
    args = episode()
    baseline = local.fit_branch_local_ridge(**args).audit_dict()['final_fit']
    events = []
    with patch.object(np.linalg, 'cholesky', side_effect=AssertionError('candidate does not factor')):
        state = margin.fit_branch_local_margin(**args, log_callback=events.append)
    audit = state.audit_dict()['final_fit']; assert_certificate(audit)
    for key in ('bandwidth_tau', 'interaction_centered_trace', 'radial_centered_trace', 'trace_scale', 'trace_relative_error'):
        assert audit[key] == baseline[key]
    assert events == audit['optimization_trace']
    assert events[-1]['optimizer_steps'] == audit['optimizer_steps']
    assert all(row['scope'] == 'final' and row['parent_k'] == 3 for row in events)
    for event in events:
        for key in ('bandwidth_tau', 'trace_scale', 'interaction_centered_trace', 'radial_centered_trace'):
            assert event[key] == audit[key]
        assert event['train_physical_count'] == 9 and event['class_count'] == 3
        assert event['max_sweeps'] == 1000
    assert state.audit_dict()['source_rows_used'] == 0
    assert state.audit_dict()['new_source_payload_bytes'] == 0


@pytest.mark.parametrize('c,zero', [(1, False), (3, False), (3, True)])
def test_analytic_degeneracies_and_tie_rule(c, zero):
    args = episode(c=c, k=2)
    args['classes'] = ['z', 'a', 'm'][:c]; args['old_classes'] = ['z']
    if c > 1:
        for value in features(args).values():
            value[:] = 0 if zero else value[0]
    with patch.object(margin, '_optimize', side_effect=AssertionError('analytic case')):
        state = margin.fit_branch_local_margin(**args)
    audit = state.audit_dict()['final_fit']; assert_certificate(audit)
    assert audit['primal_objective'] == audit['dual_objective'] == (len(args['support_ids'])/2 if c > 1 else 0)
    assert audit['optimizer_steps'] == 0 and audit['optimization_trace'] == []
    np.testing.assert_array_equal(state.score(**features(args)), np.zeros((c*2, c)))
    assert (state.predict(**features(args)) == min(args['classes'])).all()


def test_zero_bandwidth_conflicting_labels_has_certified_finite_soft_margin():
    args = episode(c=4, k=1)
    for value in features(args).values():
        value[1] = value[0]; value[3] = value[2]
    state = margin.fit_branch_local_margin(**args)
    audit = state.audit_dict()['final_fit']; assert_certificate(audit)
    assert audit['bandwidth_tau'] == 0 and audit['trace_scale'] > 0
    held = features(episode(c=4, k=1, seed=7))
    scores = state.score(**held)
    np.testing.assert_array_equal(scores, np.broadcast_to(scores[0], scores.shape))


def test_physical_class_permutation_old_role_and_batch_independence_immutable():
    args = episode(); state = margin.fit_branch_local_margin(**args)
    perm = np.array([4, 2, 7, 1, 6, 5, 0, 8, 3])
    other = {**args, **{name: args[name][perm] for name in local._NAMES},
        'support_labels': (2-args['support_labels'])[perm], 'support_ids': [args['support_ids'][i] for i in perm],
        'classes': args['classes'][::-1], 'old_classes': args['classes']}
    state2 = margin.fit_branch_local_margin(**other)
    held = features(episode(c=3, k=2, seed=7))
    scores = state.score(**held)
    np.testing.assert_array_equal(scores, state2.score(**held)[:, ::-1])
    singles = np.concatenate([state.score(**{name: value[i:i+1] for name, value in held.items()}) for i in range(6)])
    np.testing.assert_array_equal(scores, singles)
    with pytest.raises(ValueError):
        state.alpha.flags.writeable = True
    with pytest.raises(TypeError):
        state.audit['selected'] = 'bad'


@pytest.mark.parametrize('arm', local._ARMS)
def test_all_three_controls_call_original_frozen_fits(arm):
    args = episode(); held = features(episode(seed=16))
    expected = local.fit_branch_local_ridge(**args, arm=arm)
    actual = margin.fit_branch_local_margin(**args, arm=arm)
    assert type(expected) is type(actual)
    np.testing.assert_array_equal(expected.score(**held), actual.score(**held))


def test_probe_four_fresh_arms_shared_folds_all_anchors_six_pairs_and_live_context():
    args = episode(c=3, k=3); events = []
    result = margin.probe_branch_local_margin(**args, log_callback=events.append)
    assert result['factorization_count'] == 18
    assert len(result['paired']) == 6
    assert set(result['oof']) == set(margin._ARMS)
    stages = []
    for entry in result['folds']+result['oneshot_proxy']['trials']:
        assert set(entry['training_ids']).isdisjoint(entry['held_ids'])
        assert set(entry['training_ids']) | set(entry['held_ids']) == set(args['support_ids'])
        assert [stage['arm'] for stage in entry['stages']] == list(margin._ARMS)
        for stage in entry['stages']:
            assert stage['training_physical_ids'] == entry['training_ids']
        candidate = entry['stages'][-1]; assert_certificate(candidate)
        assert candidate['bandwidth_tau'] == entry['stages'][-2]['bandwidth_tau']
        stages.extend(entry['stages'])
    assert sorted(pid for trial in result['oneshot_proxy']['trials'] for pid in trial['training_ids']) == sorted(args['support_ids'])
    assert result['optimizer_steps'] == sum(s['optimizer_steps'] for s in stages)
    assert len(events) == result['sweep_count']
    assert {event['scope'] for event in events} == {'support_oof', 'support_oneshot_proxy'}
    assert all((e['fold'] is not None) != (e['trial'] is not None) for e in events)
    json.dumps(result, allow_nan=False)


def test_true_k1_probe_is_numerical_only():
    with patch.object(margin, '_fit_state', side_effect=AssertionError('no fit')):
        result = margin.probe_branch_local_margin(**episode(k=1))
    assert result['optimizer_steps'] == result['factorization_count'] == 0
    assert result['oof'] is None and result['oneshot_proxy'] is None


def test_iteration_cap_failure_retains_iterates_and_context_without_fallback():
    labels = np.array([0, 1, 2])
    gram = np.eye(3)-np.ones((3, 3))/3
    with pytest.raises(margin.NumericalFailure) as caught:
        margin._optimize(gram, labels, 3, max_sweeps=1, context=dict(scope='test', fold=7))
    audit = caught.value.audit_dict()
    assert audit['failure_reason'] == 'MAX_SWEEPS_WITHOUT_CERTIFICATE'
    assert audit['optimizer_steps'] == 3 and audit['completed_sweeps'] == 1
    assert audit['scope'] == 'test' and audit['fold'] == 7
    assert set(audit['solver_state']) == {'beta', 'alpha', 'F', 'rivals', 'labels'}
    json.dumps(audit, allow_nan=False)


def test_failure_keeps_completed_stages_and_strict_json_nonfinite_state():
    original = margin._optimize; count = 0
    def second_failure(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise margin.NumericalFailure('injected', dict(optimizer_steps=1, factorization_calls=0,
                solver_state=dict(beta=[[float('nan')]], alpha=[[float('inf')]], F=[[-float('inf')]])))
        return original(*args, **kwargs)
    with patch.object(margin, '_optimize', side_effect=second_failure):
        with pytest.raises(margin.NumericalFailure) as caught:
            margin.probe_branch_local_margin(**episode())
    audit = caught.value.audit_dict()
    assert audit['scope'] == 'support_oof' and audit['fold'] == 1
    assert audit['arm'] == 'local_margin' and len(audit['completed_stages']) == 7
    assert audit['factorization_count'] == 6
    assert len(audit['training_physical_ids']) == 6 and len(audit['held_ids']) == 3
    assert audit['solver_state'] == dict(beta=[['NaN']], alpha=[['Infinity']], F=[['-Infinity']])
    json.dumps(audit, allow_nan=False)


def test_frozen_json_matches_code():
    path = Path(__file__).resolve().parents[1]/'configs/d92_branch_local_margin_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8'))['algorithm'] == margin.FROZEN_CONFIG


@pytest.mark.parametrize('k', [1, 14, 20])
def test_c26_synthetic_sizes_26_364_520_memory_and_certificate(k):
    state = margin.fit_branch_local_margin(**episode(c=26, k=k))
    audit = state.audit_dict(); fit = audit['final_fit']; assert_certificate(fit)
    assert fit['gram_bytes'] == (26*k)**2*8
    assert not fit['dual_hessian_materialized']
    assert fit['maximum_dual_variable_count'] == 26*k*25
    assert audit['persistent_state_bytes'] == 8*26*k*(738+26)+32
    assert audit['current_support_feature_bytes'] == 26*k*736*8
    assert 0 < fit['sweeps'] <= 1000
    print(json.dumps(dict(n=26*k, fit_seconds=fit['fit_seconds'], sweeps=fit['sweeps'],
        optimizer_steps=fit['optimizer_steps'], persistent_state_bytes=audit['persistent_state_bytes'],
        relative_duality_gap=fit['relative_duality_gap'], relative_kkt_residual=fit['relative_kkt_residual'])))
