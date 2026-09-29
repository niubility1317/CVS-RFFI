import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_branch_interaction as core
from cvsrffi import d92_branch_ridge as baseline
from cvsrffi.d92_branch_support_probe import _ridge_fit


KEYS = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')


def data(k=3, c=3, seed=929):
    rng = np.random.default_rng(seed)
    values = {key: rng.normal(size=(k*c, 96 if key == 'fft' else 160)).astype(np.float32)
              for key in KEYS}
    values.update(support_labels=np.repeat(np.arange(c), k),
                  support_ids=['physical%03d'%i for i in range(k*c)],
                  classes=['class%02d'%i for i in range(c)], old_classes=['class00'])
    return values


def features(value): return {key: value[key] for key in KEYS}


def subset(value, mask):
    result = {key: value[key][mask] for key in KEYS}
    result.update(support_labels=value['support_labels'][mask],
                  support_ids=[pid for i, pid in enumerate(value['support_ids']) if mask[i]],
                  classes=value['classes'], old_classes=value['old_classes'])
    return result


@pytest.mark.parametrize('arm', core.FROZEN_CONFIG['arms'])
def test_kernel_is_exact_explicit_feature_ridge_with_intercept(arm):
    rng = np.random.default_rng(20260929)
    b, a = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))
    b, a = b/np.linalg.norm(b, axis=1)[:, None], a/np.linalg.norm(a, axis=1)[:, None]
    labels = np.repeat(np.arange(3), 2)
    y = np.eye(3)[labels]-1/3
    x = np.concatenate((b, a), axis=1)
    if arm == 'energy_control': x = np.sqrt(1.5)*x
    if arm == 'interaction': x = np.concatenate((x, np.einsum('ni,nj->nij', b, a).reshape(6, -1)), axis=1)
    parts, audit = core._fit(b, a, labels, tuple(str(i) for i in range(6)), 3, arm)
    w, intercept, _ = _ridge_fit(x, y)
    alpha, reference, reference_self, mean, grand, target_mean = parts
    np.testing.assert_allclose(core._combine(b@b.T, a@a.T, arm), x@x.T, atol=2e-15)
    got = core._score_rows(b, a, b, a, alpha, reference, reference_self, mean, grand, target_mean, arm)
    np.testing.assert_allclose(got, x@w+intercept, atol=2e-15)
    residual = x@w+intercept-y
    assert audit['loss_data'] == pytest.approx(.5*np.sum(residual**2))
    assert audit['loss_ridge'] == pytest.approx(.5*np.sum(w*w))
    assert audit['normal_equation_residual'] < 1e-13
    assert audit['intercept_gradient_norm'] < 1e-13
    assert audit['condition_bound'] <= 1+3*len(x)+1e-12
    # Unseen rows use only train-support centering, including the tensor cross term.
    qb, qa = rng.normal(size=(2, 4)), rng.normal(size=(2, 3))
    qx = np.concatenate((qb, qa), axis=1)
    if arm == 'energy_control': qx *= np.sqrt(1.5)
    if arm == 'interaction': qx = np.concatenate((qx, np.einsum('ni,nj->nij', qb, qa).reshape(2, -1)), axis=1)
    qgot = core._score_rows(qb, qa, b, a, alpha, reference, reference_self, mean, grand, target_mean, arm)
    np.testing.assert_allclose(qgot, qx@w+intercept, atol=3e-15)


@pytest.mark.parametrize('k,c', [(1, 1), (1, 26), (5, 4), (10, 3), (20, 2)])
def test_linear_control_matches_original_branch_ridge(k, c):
    inputs = data(k, c)
    old = baseline.fit_branch_ridge(**inputs)
    state = core.fit_branch_interaction(**inputs, arm='linear')
    query = features(data(2, c, 7))
    np.testing.assert_allclose(state.score(**query), old.score(**query), atol=2e-14)
    assert state.audit_dict()['final_fit']['loss_total'] == pytest.approx(old.audit_dict()['final_fit']['loss_total'])


def test_energy_control_is_exact_regularization_control():
    inputs = data(3, 4)
    b, a = core._blocks(**features(inputs))
    x = np.concatenate((b, a), axis=1)
    y = np.eye(4)[inputs['support_labels']]-1/4
    w, bias, _ = _ridge_fit(x, y, ridge=2/3)
    state = core.fit_branch_interaction(**inputs, arm='energy_control')
    np.testing.assert_allclose(state.score(**features(inputs)), x@w+bias, atol=2e-14)


def test_interaction_adds_a_non_linear_cross_block_mechanism():
    # XOR in distinct feature blocks: every additive linear head has zero signal.
    b = np.asarray([[-1.], [-1.], [1.], [1.]])
    a = np.asarray([[-1.], [1.], [-1.], [1.]])
    labels = np.asarray([0, 1, 1, 0])
    outcomes = {}
    for arm in core.FROZEN_CONFIG['arms']:
        parts, _ = core._fit(b, a, labels, ('a', 'b', 'c', 'd'), 2, arm)
        outcomes[arm] = core._score_rows(b, a, b, a, *parts, arm)
    np.testing.assert_allclose(outcomes['linear'], 0, atol=1e-15)
    np.testing.assert_allclose(outcomes['energy_control'], 0, atol=1e-15)
    np.testing.assert_array_equal(outcomes['interaction'].argmax(axis=1), labels)


def test_query_batch_order_immutability_and_state_bytes():
    inputs = data(5, 4)
    state = core.fit_branch_interaction(**inputs)
    query = features(data(4, 4, 921))
    scores = state.score(**query)
    chunked = np.concatenate([state.score(**{key: v[i:i+3] for key, v in query.items()}) for i in range(0, 16, 3)])
    np.testing.assert_array_equal(scores, chunked)
    order = np.random.default_rng(11).permutation(16)
    np.testing.assert_array_equal(scores[order], state.score(**{key: v[order] for key, v in query.items()}))
    numeric = ('support_background', 'support_auxiliary', 'alpha', 'reference_kernel', 'center_mean', 'target_mean')
    for name in numeric:
        with pytest.raises(ValueError): getattr(state, name).setflags(write=True)
    with pytest.raises(TypeError): state.audit['k'] = 999
    original = state.score(**query)
    for key in KEYS: inputs[key].fill(0)
    np.testing.assert_array_equal(original, state.score(**query))
    audit = state.audit_dict()
    assert audit['persistent_state_bytes'] == sum(getattr(state, name).nbytes for name in numeric)+16
    assert audit['current_support_feature_bytes'] == 20*736*8
    assert audit['source_rows_used'] == audit['query_rows_used'] == 0
    assert audit['fold_count'] == 0 and audit['oof'] is None
    assert audit['factorization_count'] == 1
    json.dumps(audit, allow_nan=False)
    assert state.score(**{key: v[:0] for key, v in query.items()}).shape == (0, 4)
    assert state.predict(**{key: v[:0] for key, v in query.items()}).shape == (0,)


def test_physical_and_class_permutation_role_independence():
    inputs = data(5, 3)
    first = core.fit_branch_interaction(**inputs)
    changed = copy.deepcopy(inputs)
    order = np.random.default_rng(31).permutation(15)
    for key in KEYS+('support_labels',): changed[key] = changed[key][order]
    changed['support_ids'] = [changed['support_ids'][i] for i in order]
    changed['classes'] = changed['classes'][::-1]
    changed['support_labels'] = 2-changed['support_labels']
    changed['old_classes'] = ['class02', 'class01']
    second = core.fit_branch_interaction(**changed)
    np.testing.assert_array_equal(first.alpha[:, ::-1], second.alpha)
    np.testing.assert_array_equal(first.predict(**features(inputs)), second.predict(**features(inputs)))


@pytest.mark.parametrize('constant', [0., 1e-30, 1., 1e30])
def test_constant_support_and_stable_ties(constant):
    inputs = data(1, 3)
    inputs['classes'], inputs['old_classes'] = ['z', 'a', 'm'], ['z']
    for key in KEYS: inputs[key].fill(constant)
    state = core.fit_branch_interaction(**inputs)
    query = features(data(3, 3, 123))
    np.testing.assert_array_equal(state.score(**query), np.zeros((9, 3)))
    np.testing.assert_array_equal(state.predict(**query), np.full(9, 'a'))


def test_energy_scope_discloses_zero_blocks():
    inputs = data(1, 3)
    b, a = core._blocks(**features(inputs))
    diagnostic = core._diagonal_stats(b, a)
    assert diagnostic['interaction_minus_energy_control_diagonal']['max_abs'] < 2e-15
    for key in ('t_emb', 'f_emb', 'pa_local'): inputs[key].fill(0)
    result = core.probe_branch_interaction(**inputs)
    assert result['numerical']['interaction_minus_energy_control_diagonal']['mean'] == pytest.approx(-.5)
    assert result['numerical']['energy_match_scope'] == core.FROZEN_CONFIG['energy_match_scope']


def test_k1_probe_has_no_fit_or_invented_holdout(monkeypatch):
    def fail(*args, **kwargs): raise AssertionError('K1 diagnostics must not fit')
    monkeypatch.setattr(core, '_fit', fail)
    result = core.probe_branch_interaction(**data(1, 5))
    assert result['oof'] is None and result['paired'] is None
    assert result['fold_count'] == result['factorization_count'] == 0
    assert result['folds'] == result['physical_fold_assignment'] == []
    assert result['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT'


@pytest.mark.parametrize('k', [2, 5, 10, 20])
def test_complete_oof_refits_only_physical_training_support(k):
    inputs = data(k, 3)
    result = core.probe_branch_interaction(**inputs)
    assert result['factorization_count'] == 3*min(k, 3)
    assignments = {row['physical_id']: row['fold'] for row in result['physical_fold_assignment']}
    assert len(assignments) == k*3
    for fold in result['folds']:
        train_ids, held_ids = set(fold['training_ids']), set(fold['held_ids'])
        assert train_ids.isdisjoint(held_ids) and train_ids|held_ids == set(inputs['support_ids'])
        keep = np.asarray([pid in train_ids for pid in inputs['support_ids']])
        for stage in fold['stages']:
            arm = stage['arm']
            assert set(stage['training_physical_ids']) == train_ids
            independent = core.fit_branch_interaction(**subset(inputs, keep), arm=arm)
            scores = independent.score(**{key: inputs[key][~keep] for key in KEYS})
            preds = independent.predict(**{key: inputs[key][~keep] for key in KEYS})
            held_order = [pid for pid in inputs['support_ids'] if pid in held_ids]
            rows = {row['physical_id']: row for row in result['oof'][arm]['rows']}
            for i, pid in enumerate(held_order):
                assert rows[pid]['predicted_class'] == preds[i]
                label = int(inputs['support_labels'][inputs['support_ids'].index(pid)])
                maximum = scores[i].max()
                nll = maximum+np.log(np.exp(scores[i]-maximum).sum())-scores[i, label]
                assert rows[pid]['nll'] == pytest.approx(nll, abs=1e-14)
    assert set(result['paired']) == {'interaction_minus_linear', 'energy_control_minus_linear', 'interaction_minus_energy_control'}
    for pair in result['paired'].values(): assert len(pair['rows']) == k*3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('error', ['nonfinite', 'wrong_dimension', 'duplicate_id', 'unequal_k', 'bad_label', 'unknown_arm'])
def test_reject_invalid_inputs(error):
    inputs = data(3, 3)
    if error == 'nonfinite': inputs['fft'][0, 0] = np.nan
    if error == 'wrong_dimension': inputs['z_id'] = inputs['z_id'][:, :-1]
    if error == 'duplicate_id': inputs['support_ids'][0] = inputs['support_ids'][1]
    if error == 'unequal_k': inputs['support_labels'][0] = 1
    if error == 'bad_label': inputs['support_labels'] = inputs['support_labels'].astype(float)
    if error == 'unknown_arm': inputs['arm'] = 'selected_by_query'
    with pytest.raises((ValueError, FloatingPointError)): core.fit_branch_interaction(**inputs)


def test_frozen_config_and_input_boundary():
    path = Path(__file__).resolve().parents[1]/'configs'/'d92_branch_interaction_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8')) == {'algorithm': core.FROZEN_CONFIG}
    assert 'query' not in inspect.signature(core.fit_branch_interaction).parameters
    assert 'labels' not in inspect.signature(core.BranchInteractionState.score).parameters
    assert 'support_labels' not in inspect.signature(core.BranchInteractionState.score).parameters
    changed = core.fit_branch_interaction(**data()).audit_dict()
    changed['config']['arms'].append('unfrozen')
    assert core.fit_branch_interaction(**data()).audit_dict()['config']['arms'] == ['linear', 'energy_control', 'interaction']
