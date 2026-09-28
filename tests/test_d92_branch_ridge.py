import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_branch_ridge as core
from cvsrffi import d92_branch_support_probe as probe


KEYS = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')


def data(k=2, c=3):
    rng = np.random.default_rng(926)
    x = {key: rng.normal(size=(k*c, 96 if key == 'fft' else 160)).astype(np.float32) for key in KEYS}
    x.update(support_labels=np.repeat(np.arange(c), k), support_ids=['id%03d'%i for i in range(k*c)],
             classes=['class%02d'%i for i in range(c)], old_classes=['class00'])
    return x


def features(value): return {key: value[key] for key in KEYS}


@pytest.mark.parametrize('k,c', [(1, 1), (1, 26), (2, 3), (5, 6)])
def test_exact_probe_arm_equivalence_and_real_objective(k, c):
    inputs = data(k, c)
    state = core.fit_branch_ridge(**inputs)
    _, base, aux = probe._feature_blocks(**features(inputs))
    x = np.concatenate((base['zfft'], aux), axis=1)
    y = np.eye(c)[inputs['support_labels']]-1./c
    w, b, audit = probe._fit_stage(x, y, aux, background='zfft', arm='zfft_aux', train_k=k, classification_dim=c)
    np.testing.assert_array_equal(state.W, w)
    np.testing.assert_array_equal(state.b, b)
    result = state.audit_dict()
    assert result['fold_count'] == 0 and result['folds'] == [] and result['oof'] is None
    assert result['heldout_unavailable_reason'] == ('K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG')
    fit = result['final_fit']
    assert fit['gradient_norm'] < 1e-11
    assert fit['normal_equation_residual'] < 1e-11
    assert fit['loss_total'] == pytest.approx(audit['loss_total'])
    assert fit['target_norm_squared'] == pytest.approx(c*k*(1-1/c))
    assert fit['physical_loss_mass'] == c*k
    assert fit['factorization_dim'] == min(736, c*k)
    assert fit['condition_bound'] <= 1+2*c*k+1e-12
    assert result['persistent_state_bytes'] == state.W.nbytes+state.b.nbytes == 5896*c
    assert result['optimizer_steps'] == 0 and result['factorization_count'] == 1
    json.dumps(result, allow_nan=False)


def test_immutability_registry_and_batch_score_invariance():
    inputs = data(5, 4)
    state = core.fit_branch_ridge(**inputs)
    query = features(data(4, 4))
    before_w, before_b = state.W.copy(), state.b.copy()
    scores = state.score(**query)
    chunked = np.concatenate([state.score(**{key: value[begin:begin+3] for key, value in query.items()}) for begin in range(0, 16, 3)])
    np.testing.assert_array_equal(scores, chunked)
    order = np.random.default_rng(9).permutation(16)
    np.testing.assert_array_equal(scores[order], state.score(**{key: value[order] for key, value in query.items()}))
    np.testing.assert_array_equal(before_w, state.W); np.testing.assert_array_equal(before_b, state.b)
    with pytest.raises(ValueError): state.W[0, 0] = 1
    with pytest.raises(ValueError): state.W.setflags(write=True)
    with pytest.raises(TypeError): state.audit['k'] = 9
    plain = state.audit_dict(); plain['config']['branches'].append('bad')
    assert state.audit_dict()['config']['branches'] == ['t_emb', 'f_emb', 'pa_local']
    empty = {key: value[:0] for key, value in query.items()}
    assert state.score(**empty).shape == (0, 4)
    assert state.predict(**empty).shape == (0,)


def test_class_and_support_permutation_equivariance():
    inputs = data(5, 3); first = core.fit_branch_ridge(**inputs)
    changed = copy.deepcopy(inputs)
    permutation = np.random.default_rng(78).permutation(15)
    for key in KEYS+('support_labels',): changed[key] = changed[key][permutation]
    changed['support_ids'] = [changed['support_ids'][i] for i in permutation]
    changed['classes'] = changed['classes'][::-1]
    changed['support_labels'] = 2-changed['support_labels']
    second = core.fit_branch_ridge(**changed)
    assert second.classes == tuple(changed['classes'])
    np.testing.assert_array_equal(first.W[:, ::-1], second.W)
    np.testing.assert_array_equal(first.b[::-1], second.b)
    np.testing.assert_array_equal(first.predict(**features(inputs)), second.predict(**features(inputs)))
    assert first.audit_dict()['final_fit']['training_physical_ids'] == second.audit_dict()['final_fit']['training_physical_ids']


def test_all_zero_tiny_and_stable_physical_tie():
    inputs = data(1, 3); inputs['classes'] = ['z', 'a', 'm']; inputs['old_classes'] = ['z']
    for key in KEYS: inputs[key].fill(0)
    state = core.fit_branch_ridge(**inputs)
    np.testing.assert_array_equal(state.W, 0.)
    assert set(state.predict(**features(inputs))) == {'a'}
    for key in KEYS: inputs[key].fill(1e-20)
    state = core.fit_branch_ridge(**inputs)
    np.testing.assert_array_equal(state.W, 0.)
    assert state.audit_dict()['final_fit']['gradient_norm'] < 1e-12


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(z_id=np.zeros((6, 159))),
    lambda x: x.update(z_id=np.zeros((6, 160), dtype=object)),
    lambda x: x['fft'].__setitem__((0, 0), np.nan),
    lambda x: x.update(support_labels=np.zeros(6, dtype=int)),
    lambda x: x.update(support_labels=np.arange(6, dtype=float)),
    lambda x: x.update(support_ids=['dup']*6),
    lambda x: x.update(classes=['a', 'a', 'b']),
    lambda x: x.update(old_classes=['absent']),
    lambda x: x.update(support_labels=np.array([0, 0, 0, 1, 2, 2])),
    lambda x: x.update(pa_local=np.zeros((7, 160))),
])
def test_bad_inputs(mutation):
    inputs = data(); mutation(inputs)
    with pytest.raises(ValueError): core.fit_branch_ridge(**inputs)


def test_frozen_config_and_public_input_boundaries():
    path = Path(__file__).resolve().parents[1]/'configs/d92_branch_ridge_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8')) == {'algorithm': core.FROZEN_CONFIG}
    assert set(inspect.signature(core.fit_branch_ridge).parameters) == set(KEYS)|{'support_labels', 'support_ids', 'classes', 'old_classes'}
    assert set(inspect.signature(core.BranchRidgeState.score).parameters) == set(KEYS)|{'self'}


def test_current_row_states_do_not_depend_on_previous_fit():
    inputs = data()
    state = core.fit_branch_ridge(**inputs)
    core.fit_branch_ridge(**data(5, 6))
    repeat = core.fit_branch_ridge(**inputs)
    np.testing.assert_array_equal(state.W, repeat.W)
    np.testing.assert_array_equal(state.b, repeat.b)
