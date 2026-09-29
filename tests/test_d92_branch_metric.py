import copy
import json
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_branch_metric as core
from cvsrffi import d92_branch_interaction as interaction

KEYS = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')


def data(k=3, c=3, seed=930):
    rng = np.random.default_rng(seed)
    value = {key: rng.normal(size=(k*c, 96 if key == 'fft' else 160)).astype(np.float32) for key in KEYS}
    value.update(support_labels=np.repeat(np.arange(c), k), support_ids=['physical%03d'%i for i in range(k*c)],
                 classes=['class%02d'%i for i in range(c)], old_classes=['class00'])
    return value


def features(value): return {key: value[key] for key in KEYS}


def subset(value, keep):
    output = {key: value[key][keep] for key in KEYS}
    output.update(support_labels=value['support_labels'][keep],
                  support_ids=[pid for i, pid in enumerate(value['support_ids']) if keep[i]],
                  classes=value['classes'], old_classes=value['old_classes'])
    return output


def explicit(b, a): return np.concatenate((b, a, np.einsum('ni,nj->nij', b, a).reshape(len(b), -1)), axis=1)


@pytest.mark.parametrize('arm', ['kernel_ncm', 'within_metric'])
def test_exact_dual_matches_explicit_sum_within_scatter_and_distance(arm):
    rng = np.random.default_rng(930)
    b, a = rng.normal(size=(9, 3)), rng.normal(size=(9, 2))
    qb, qa = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
    labels = np.repeat(np.arange(3), 3)
    xraw, qraw = explicit(b, a), explicit(qb, qa)
    x, qx = xraw-xraw.mean(axis=0), qraw-xraw.mean(axis=0)
    p = np.eye(3)[labels]/3
    r = np.eye(9)-np.eye(3)[labels]@np.eye(3)[labels].T/3
    np.testing.assert_allclose(core._class_center(x, labels), r@x, atol=1e-14)
    sw = x.T@r@x
    mu = x.T@p
    np.testing.assert_allclose(x.T@x, sw+3*mu@mu.T, atol=1e-13)
    metric = np.linalg.inv(np.eye(x.shape[1])+sw) if arm == 'within_metric' else np.eye(x.shape[1])
    expected = qx@metric@mu-.5*np.diag(mu.T@metric@mu)
    parts, audit = core._solve(b, a, labels, tuple(map(str, range(9))), 3, arm)
    got = interaction._score_rows(qb, qa, b, a, *parts, 'interaction')
    np.testing.assert_allclose(got, expected, atol=2e-13)
    distance = np.asarray([[-.5*(row-mu[:, c])@metric@(row-mu[:, c]) for c in range(3)] for row in qx])
    np.testing.assert_array_equal(got.argmax(axis=1), distance.argmax(axis=1))
    assert audit['factorization_calls'] == (arm == 'within_metric')
    assert audit['normal_equation_residual'] < 1e-12
    assert audit['loss_data'] is audit['loss_ridge'] is None
    if arm == 'within_metric':
        assert audit['within_scatter_trace'] == pytest.approx(np.trace(sw))
        mean_metric = np.linalg.inv(np.eye(x.shape[1])+sw/len(x))
        assert not np.allclose(got, qx@mean_metric@mu-.5*np.diag(mu.T@mean_metric@mu))
        rhs = r@(x@x.T)@p
        q = np.linalg.solve(np.eye(9)+r@(x@x.T)@r, rhs)
        assert audit['solve_objective'] == pytest.approx(.5*np.sum(q*((np.eye(9)+r@(x@x.T)@r)@q))-np.sum(q*rhs))


def test_ridge_class_coupling_identity_and_metric_changes_scores():
    rng = np.random.default_rng(332)
    x = rng.normal(size=(12, 7)); x -= x.mean(axis=0)
    labels = np.repeat(np.arange(3), 4)
    z = np.eye(3)[labels]; mu = x.T@z/4
    r = np.eye(12)-z@z.T/4
    m = np.linalg.inv(np.eye(7)+x.T@r@x)
    ridge = np.linalg.solve(x.T@x+np.eye(7), x.T@(z-1/3))
    coupled = 4*m@mu@np.linalg.inv(np.eye(3)+4*mu.T@m@mu)
    np.testing.assert_allclose(ridge, coupled, atol=1e-15)
    assert not np.allclose(x@ridge, x@m@mu-.5*np.diag(mu.T@m@mu))


@pytest.mark.parametrize('c', [1, 3, 26])
def test_k1_exact_ncm_without_variance_or_factorization(c):
    inputs = data(1, c)
    metric, ncm = core.fit_branch_metric(**inputs), core.fit_branch_metric(**inputs, arm='kernel_ncm')
    query = features(data(2, c, 771))
    np.testing.assert_array_equal(metric.V, ncm.V)
    np.testing.assert_array_equal(metric.b, ncm.b)
    np.testing.assert_array_equal(metric.score(**query), ncm.score(**query))
    assert metric.audit_dict()['factorization_count'] == 0
    assert metric.audit_dict()['final_fit']['k1_within_variance_estimated'] is False
    train_b, train_a = interaction._blocks(**features(inputs))
    qb, qa = interaction._blocks(**query)
    raw = interaction._combine(qb@train_b.T, qa@train_a.T, 'interaction')
    norms = np.diag(interaction._combine(train_b@train_b.T, train_a@train_a.T, 'interaction'))
    np.testing.assert_array_equal(metric.score(**query).argmax(axis=1), (raw-.5*norms).argmax(axis=1))


@pytest.mark.parametrize('k', [1, 5, 10, 20])
def test_public_scores_invariant_to_query_batch_and_state_bytes(k):
    inputs = data(k, 3)
    state = core.fit_branch_metric(**inputs)
    query = features(data(4, 3, 817))
    score = state.score(**query)
    chunks = np.concatenate([state.score(**{key: v[i:i+2] for key, v in query.items()}) for i in range(0, 12, 2)])
    np.testing.assert_array_equal(score, chunks)
    order = np.random.default_rng(21).permutation(12)
    np.testing.assert_array_equal(score[order], state.score(**{key: v[order] for key, v in query.items()}))
    numeric = ['support_background', 'support_auxiliary', 'V', 'reference_kernel', 'center_mean', 'b']
    assert sum(getattr(state, key).nbytes for key in numeric)+16 == state.audit_dict()['persistent_state_bytes']
    assert state.audit_dict()['persistent_state_bytes'] == 8*((3*k)*(736+3+2)+3+2)
    for key in numeric:
        with pytest.raises(ValueError): getattr(state, key).setflags(write=True)
    with pytest.raises(TypeError): state.audit['selected'] = 'changed'
    for key in KEYS: inputs[key].fill(0)
    np.testing.assert_array_equal(score, state.score(**query))
    assert state.score(**{key: v[:0] for key, v in query.items()}).shape == (0, 3)
    assert state.predict(**{key: v[:0] for key, v in query.items()}).shape == (0,)
    json.dumps(state.audit_dict(), allow_nan=False)


def test_class_physical_order_and_old_roles_do_not_change_fit():
    inputs = data(5, 3); original = core.fit_branch_metric(**inputs)
    changed = copy.deepcopy(inputs)
    order = np.random.default_rng(2).permutation(15)
    for key in KEYS+('support_labels',): changed[key] = changed[key][order]
    changed['support_ids'] = [changed['support_ids'][i] for i in order]
    changed['classes'] = changed['classes'][::-1]
    changed['support_labels'] = 2-changed['support_labels']
    changed['old_classes'] = ['class02']
    second = core.fit_branch_metric(**changed)
    np.testing.assert_array_equal(original.V[:, ::-1], second.V)
    np.testing.assert_array_equal(original.b[::-1], second.b)
    np.testing.assert_array_equal(original.predict(**features(inputs)), second.predict(**features(inputs)))


@pytest.mark.parametrize('value', [0., 1e-30, 1., 1e30])
def test_constant_support_has_exact_tie_for_unseen_query(value):
    inputs = data(5, 3); inputs['classes'], inputs['old_classes'] = ['z', 'a', 'm'], ['a']
    for key in KEYS: inputs[key].fill(value)
    state = core.fit_branch_metric(**inputs)
    np.testing.assert_array_equal(state.score(**features(data(2, 3))), np.zeros((6, 3)))
    np.testing.assert_array_equal(state.predict(**features(data(2, 3))), np.full(6, 'a'))


def test_k1_probe_has_neither_oof_nor_proxy_nor_fit(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No K1 diagnostic fit')
    monkeypatch.setattr(core, '_solve', forbidden)
    monkeypatch.setattr(interaction, '_fit', forbidden)
    result = core.probe_branch_metric(**data(1, 6))
    assert result['oof'] is result['paired'] is result['oneshot_proxy'] is None
    assert result['fold_count'] == result['factorization_count'] == 0


@pytest.mark.parametrize('k', [2, 5])
def test_complete_physical_oof_and_exhaustive_proxy_are_fresh_support_fits(k):
    inputs = data(k, 3)
    result = core.probe_branch_metric(**inputs)
    all_ids = set(inputs['support_ids'])
    trials = result['oneshot_proxy']['trials']
    assert len(trials) == k
    anchors = []
    for entry in result['folds']+trials:
        train, held = set(entry['training_ids']), set(entry['held_ids'])
        assert train.isdisjoint(held) and train|held == all_ids
        keep = np.asarray([pid in train for pid in inputs['support_ids']])
        if 'trial' in entry:
            assert entry['parent_k'] == k and entry['proxy_train_k'] == entry['train_k'] == 1
            assert entry['held_k'] == k-1
            anchors.extend(entry['training_ids'])
        for stage in entry['stages']:
            arm = stage['arm']
            assert set(stage['training_physical_ids']) == train and stage['parent_k'] == k
            assert stage['train_k'] == int(keep.sum())//3
            independent = (interaction.fit_branch_interaction(**subset(inputs, keep)) if arm == 'interaction_ridge'
                           else core.fit_branch_metric(**subset(inputs, keep), arm=arm))
            scores = independent.score(**{key: inputs[key][~keep] for key in KEYS})
            preds = independent.predict(**{key: inputs[key][~keep] for key in KEYS})
            oof = entry['oof'] if 'trial' in entry else result['oof']
            held_ids = [pid for pid in inputs['support_ids'] if pid in held]
            records = {r['physical_id']: r for r in oof[arm]['rows']} if 'trial' not in entry else None
            confusion = np.zeros((3, 3), dtype=int)
            nll_sums = np.zeros(3)
            for i, pid in enumerate(held_ids):
                label = inputs['support_labels'][inputs['support_ids'].index(pid)]
                maximum = scores[i].max()
                nll = maximum+np.log(np.exp(scores[i]-maximum).sum())-scores[i, label]
                if records is not None:
                    assert records[pid]['predicted_class'] == preds[i]
                    assert records[pid]['nll'] == pytest.approx(nll, abs=1e-14)
                confusion[label, inputs['classes'].index(preds[i])] += 1
                nll_sums[label] += nll
            if 'trial' in entry:
                assert 'rows' not in oof[arm]
                np.testing.assert_array_equal(confusion, oof[arm]['confusion'])
                np.testing.assert_allclose(nll_sums, oof[arm]['classwise_nll_sum'], atol=1e-14)
                assert oof[arm]['classwise_count'] == [k-1]*3
                assert oof[arm]['record_count'] == 3*(k-1)
        if 'trial' in entry:
            assert entry['oof']['kernel_ncm'] == entry['oof']['within_metric']
            assert entry['paired']['within_metric_minus_kernel_ncm']['mean_correct_delta'] == 0
            assert entry['ncm_equivalence'] == dict(max_abs_score_difference=0., tolerance=0., equivalent=True, predicted_classes_equal=True)
            assert sum(entry['paired']['within_metric_minus_kernel_ncm']['counts'].values()) == 3*(k-1)
    assert sorted(anchors) == sorted(all_ids)
    proxy = result['oneshot_proxy']
    assert proxy['coverage'] == dict(parent_physical_count=3*k, training_occurrences=3*k,
        held_occurrences=3*k*(k-1), unique_held_physical_count=3*k)
    assert proxy['factorization_count'] == k
    standard = sum(s['factorization_calls'] for f in result['folds'] for s in f['stages'])
    assert result['factorization_count'] == standard+k
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('fault', ['nan', 'shape', 'duplicate_id', 'unbalanced', 'labels', 'arm'])
def test_invalid_input_rejected(fault):
    inputs = data()
    if fault == 'nan': inputs['fft'][0, 0] = np.nan
    if fault == 'shape': inputs['z_id'] = inputs['z_id'][:, :-1]
    if fault == 'duplicate_id': inputs['support_ids'][1] = inputs['support_ids'][0]
    if fault == 'unbalanced': inputs['support_labels'][0] = 1
    if fault == 'labels': inputs['support_labels'] = inputs['support_labels'].astype(float)
    if fault == 'arm': inputs['arm'] = 'interaction_ridge'
    with pytest.raises((ValueError, FloatingPointError)): core.fit_branch_metric(**inputs)


def test_config_exact_and_audit_copy_is_independent():
    path = Path(__file__).resolve().parents[1]/'configs/d92_branch_metric_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8')) == {'algorithm': core.FROZEN_CONFIG}
    state = core.fit_branch_metric(**data())
    value = state.audit_dict(); value['config']['arms'].append('bad')
    assert state.audit_dict()['config']['arms'] == ['interaction_ridge', 'kernel_ncm', 'within_metric']
