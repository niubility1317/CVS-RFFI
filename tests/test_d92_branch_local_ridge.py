"""Synthetic algebra/protocol checks; never reads experimental data."""
import json
import tracemalloc
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from cvsrffi import d92_branch_local_ridge as local
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_ridge import fit_branch_ridge


def episode(c=3, k=3, seed=91):
    rng = np.random.default_rng(seed)
    raw = {name: rng.normal(size=(c*k, dim)) for name, dim in zip(local._NAMES, (160, 96, 160, 160, 160))}
    return dict(**raw, support_labels=np.repeat(np.arange(c), k),
        support_ids=[f'p{i:04d}' for i in range(c*k)], classes=[f'c{i:02d}' for i in range(c)], old_classes=['c00'])


def features(args):
    return {name: args[name] for name in local._NAMES}


def explicit(b, a):
    return np.concatenate((b, a, (b[:, :, None]*a[:, None, :]).reshape(len(b), -1)), axis=1)


def test_rank2_distance_matches_explicit_near_duplicate_zero_and_rank_deficient():
    rng = np.random.default_rng(4)
    b, a = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
    b[1], a[1] = b[0], a[0]
    b[1, 0] += 1e-14
    a[1, 1] -= 1e-14
    b[2] = 0
    a[3] = 0
    b[4] = 2*b[0]
    phi = explicit(b, a)
    expected = np.sum((phi[:, None]-phi[None])**2, axis=2)
    actual = local._distances(b, a)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-14)
    assert actual[0, 1] > 0
    assert np.all(actual >= 0)
    assert np.array_equal(actual, actual.T)
    assert np.array_equal(np.diag(actual), np.zeros(5))
    # Windows longdouble is float64. Use exact float-to-Decimal conversion and
    # an independent 80-digit explicit outer-product oracle for the tiny pair.
    with localcontext() as context:
        context.prec = 80
        high = []
        for row in (0, 1):
            db = [Decimal.from_float(float(value)) for value in b[row]]
            da = [Decimal.from_float(float(value)) for value in a[row]]
            high.append(db+da+[left*right for left in db for right in da])
        expected_tiny = float(sum((left-right)**2 for left, right in zip(*high)))
    np.testing.assert_allclose(actual[0, 1], expected_tiny, rtol=1e-12, atol=0)


def test_kernel_psd_trace_and_independent_solution():
    args = episode()
    state = local.fit_branch_local_ridge(**args)
    b, a = interaction._blocks(**features(args))
    distance = local._distances(b, a)
    audit = state.audit_dict()['final_fit']
    r = np.exp(-distance/audit['bandwidth_tau'])
    h = np.eye(len(b))-np.ones((len(b), len(b)))/len(b)
    g = audit['trace_scale']*(h@r@h)
    assert np.linalg.eigvalsh(r).min() > -1e-12
    np.testing.assert_allclose(np.trace(g), audit['interaction_centered_trace'], atol=1e-12)
    target = np.eye(3)[args['support_labels']]-1/3
    expected = g@np.linalg.solve(g+np.eye(len(b)), target)
    np.testing.assert_allclose(state.score(**features(args)), expected, atol=1e-13)
    # Independent explicit finite-dimensional primal feature realization.
    vals, vec = np.linalg.eigh(g)
    x = vec*np.sqrt(np.maximum(vals, 0))
    w = np.linalg.solve(x.T@x+np.eye(len(b)), x.T@target)
    np.testing.assert_allclose(x@w, expected, atol=2e-13)
    assert audit['normal_equation_residual'] <= audit['numerical_tolerance']


def test_zero_bandwidth_equivalence_and_unmatched_scores():
    args = episode(c=4, k=1)
    for value in features(args).values():
        value[1] = value[0]
        value[3] = value[2]
    state = local.fit_branch_local_ridge(**args)
    audit = state.audit_dict()['final_fit']
    assert audit['bandwidth_tau'] == 0
    assert audit['interaction_centered_trace'] > 0
    assert audit['trace_scale'] > 0
    b, a = interaction._blocks(**features(args))
    r = (local._distances(b, a) == 0).astype(float)
    assert np.linalg.eigvalsh(r).min() >= -1e-14
    unseen = features(episode(c=4, k=1, seed=7))
    scores = state.score(**unseen)
    np.testing.assert_array_equal(scores, np.broadcast_to(scores[0], scores.shape))


@pytest.mark.parametrize('zero', [False, True])
def test_identical_features_exact_zero_and_lexicographic_ties(zero):
    args = episode(c=3, k=2)
    args['classes'] = ['z', 'a', 'm']
    args['old_classes'] = ['z']
    for value in features(args).values():
        value[:] = 0 if zero else value[0]
    state = local.fit_branch_local_ridge(**args)
    audit = state.audit_dict()['final_fit']
    assert audit['bandwidth_tau'] == 0 and audit['trace_scale'] is None
    assert audit['factorization_calls'] == 0
    np.testing.assert_array_equal(state.score(**features(episode(c=3, k=2, seed=7))), np.zeros((6, 3)))
    assert (state.predict(**features(args)) == 'a').all()


def test_single_class_no_bandwidth_or_factorization():
    args = episode(c=1, k=3)
    with patch.object(np.linalg, 'cholesky', side_effect=AssertionError('not needed')):
        state = local.fit_branch_local_ridge(**args)
    audit = state.audit_dict()['final_fit']
    assert audit['bandwidth_tau'] is None
    assert audit['degeneracy_reason'] == 'SINGLE_REGISTERED_CLASS'
    assert audit['factorization_calls'] == 0
    np.testing.assert_array_equal(state.score(**features(args)), np.zeros((3, 1)))


def test_c2k1_trace_matching_train_but_not_unseen_extension():
    args = episode(c=2, k=1)
    state = local.fit_branch_local_ridge(**args)
    control = interaction.fit_branch_interaction(**args)
    np.testing.assert_allclose(state.score(**features(args)), control.score(**features(args)), atol=2e-14)
    held = features(episode(c=2, k=2, seed=20))
    assert np.max(np.abs(state.score(**held)-control.score(**held))) > 1e-6


def test_first_order_radial_recovers_interaction_train_and_unseen_extension():
    args = episode()
    b, a = interaction._blocks(**features(args))
    phi = explicit(b, a)
    d = local._distances(b, a)
    tau = 1.7
    lin = 1-d/tau
    gc, ref, refself, mean, grand = interaction._center_kernel(lin)
    k = phi@phi.T
    g0, _, _, _, _ = interaction._center_kernel(k)
    scale = np.trace(g0)/np.trace(gc)
    np.testing.assert_allclose(scale*gc, g0, atol=3e-14)
    hb, ha = interaction._blocks(**features(episode(c=3, k=1, seed=6)))
    r = 1-local._distances(hb, ha, b, a)/tau
    difference = r-r[:, :1]-ref[None, :]+refself
    centered = scale*(difference-difference.mean(axis=1, keepdims=True)-mean+grand)
    expected = (explicit(hb, ha)-phi.mean(axis=0))@(phi-phi.mean(axis=0)).T
    np.testing.assert_allclose(centered, expected, atol=3e-14)


def test_query_batch_permutation_held_independence_and_immutable_state():
    args = episode()
    state = local.fit_branch_local_ridge(**args)
    before = state.audit_dict()
    held = features(episode(c=3, k=2, seed=123))
    all_scores = state.score(**held)
    single = np.concatenate([state.score(**{key: val[i:i+1] for key, val in held.items()}) for i in range(6)])
    np.testing.assert_array_equal(all_scores, single)
    np.testing.assert_array_equal(state.score(**{key: val[::-1] for key, val in held.items()}), all_scores[::-1])
    assert state.audit_dict() == before
    with pytest.raises(ValueError):
        state.alpha[0, 0] = 10
    with pytest.raises(ValueError):
        state.alpha.flags.writeable = True
    with pytest.raises(TypeError):
        state.audit['selected'] = 'bad'
    assert state.score(**{key: val[:0] for key, val in held.items()}).shape == (0, 3)


def test_physical_and_class_order_equivariance_and_old_role_irrelevance():
    args = episode()
    state = local.fit_branch_local_ridge(**args)
    perm = np.array([4, 2, 7, 1, 6, 5, 0, 8, 3])
    other = {**args, **{name: args[name][perm] for name in local._NAMES},
        'support_labels': (2-args['support_labels'])[perm],
        'support_ids': [args['support_ids'][i] for i in perm], 'classes': args['classes'][::-1],
        'old_classes': args['classes']}
    state2 = local.fit_branch_local_ridge(**other)
    np.testing.assert_array_equal(state.score(**features(args)), state2.score(**features(args))[:, ::-1])
    assert state.audit_dict()['final_fit']['bandwidth_tau'] == state2.audit_dict()['final_fit']['bandwidth_tau']


def test_controls_are_exact_original_public_methods():
    args = episode()
    held = features(episode(c=3, k=2, seed=20))
    for arm, fn in [('branch_ridge', fit_branch_ridge), ('interaction_ridge', interaction.fit_branch_interaction)]:
        expected = fn(**args)
        actual = local.fit_branch_local_ridge(**args, arm=arm)
        assert type(expected) is type(actual)
        np.testing.assert_array_equal(actual.score(**held), expected.score(**held))


def test_full_physical_oof_and_all_anchor_proxy_and_true_k1_no_fit():
    args = episode(c=3, k=4)
    result = local.probe_branch_local_ridge(**args)
    assert result['factorization_count'] == 3*(3+4)
    assert set(result['paired']) == {left+'_minus_'+right for left, right in local._PAIRS}
    for entry in result['folds']+result['oneshot_proxy']['trials']:
        assert not set(entry['training_ids']) & set(entry['held_ids'])
        assert set(entry['training_ids']) | set(entry['held_ids']) == set(args['support_ids'])
        assert [stage['arm'] for stage in entry['stages']] == list(local._ARMS)
    trials = result['oneshot_proxy']['trials']
    assert len(trials) == 4
    assert sorted(pid for trial in trials for pid in trial['training_ids']) == sorted(args['support_ids'])
    assert all('ncm_equivalence' not in trial for trial in trials)
    assert all(trial['oof']['local_ridge']['record_count'] == 9 for trial in trials)
    with patch.object(local, 'fit_branch_local_ridge', side_effect=AssertionError('true K1 must not fit')):
        result = local.probe_branch_local_ridge(**episode(c=3, k=1))
    assert result['factorization_count'] == 0 and result['oof'] is None and result['oneshot_proxy'] is None


def test_failure_preserves_all_prior_stages_and_physical_context():
    original = local._solve
    calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise local.NumericalFailure('injected', dict(factorization_calls=1))
        return original(*args)
    with patch.object(local, '_solve', side_effect=fail_second):
        with pytest.raises(local.NumericalFailure) as caught:
            local.probe_branch_local_ridge(**episode(c=3, k=3))
    audit = caught.value.audit_dict()
    assert audit['scope'] == 'support_oof' and audit['fold'] == 1
    assert audit['arm'] == 'local_ridge' and audit['parent_k'] == 3 and audit['train_k'] == 2
    assert len(audit['training_physical_ids']) == 6 and len(audit['held_ids']) == 3
    assert len(audit['completed_stages']) == 5 and audit['factorization_count'] == 6


def test_cholesky_failure_no_jitter_or_fallback():
    with patch.object(np.linalg, 'cholesky', side_effect=np.linalg.LinAlgError('injected')) as call:
        with pytest.raises(local.NumericalFailure) as caught:
            local.fit_branch_local_ridge(**episode())
    assert call.call_count == 1
    assert caught.value.audit['factorization_calls'] == 1
    assert caught.value.audit['training_physical_ids']


def test_tiny_positive_bandwidth_and_distance_underflow_are_explicit():
    distance = np.array([0., 1e-300, 1.])
    actual = local._radial_minus_one(distance, 1e-300)
    np.testing.assert_array_equal(actual[[0, 2]], [0., -1.])
    np.testing.assert_allclose(actual[1], np.expm1(-1.))
    with pytest.raises(FloatingPointError, match='DISTANCE_UNDERFLOW'):
        local._pair_distances(np.array([[1e-200, 0.]]), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))


def test_frozen_config_artifact_matches_code():
    path = Path(__file__).resolve().parents[1]/'configs/d92_branch_local_ridge_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8'))['algorithm'] == local.FROZEN_CONFIG


@pytest.mark.parametrize('k', [1, 20])
def test_c26_synthetic_resource_contract(k):
    args = episode(c=26, k=k)
    tracemalloc.start()
    state = local.fit_branch_local_ridge(**args)
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audit = state.audit_dict()
    fit = audit['final_fit']
    assert fit['factorization_dim'] == 26*k
    assert fit['gram_bytes'] == (26*k)**2*8
    assert audit['persistent_state_bytes'] == sum(audit['state_array_bytes'].values())+audit['state_scalar_bytes']
    assert audit['current_support_feature_bytes'] == 26*k*736*8
    assert fit['effective_degrees_of_freedom_extra_triangular_solves'] == 2
    print(json.dumps(dict(k=k, fit_seconds=fit['fit_seconds'], gram_bytes=fit['gram_bytes'],
        persistent_state_bytes=audit['persistent_state_bytes'], pair_chunk=local.FROZEN_CONFIG['distance_pair_chunk'],
        peak_traced_bytes=peak_traced_bytes, peak_scope='tracemalloc_Python_and_NumPy_tracked_allocations_not_process_RSS')))
