import copy
import inspect
import json
import numpy as np
import pytest

from cvsrffi import d92_branch_support_probe as core


def fixture(k=2, c=3):
    rng = np.random.default_rng(1901)
    n = c*k
    return dict(z_id=rng.normal(size=(n, 160)), fft=rng.normal(size=(n, 96)),
                t_emb=rng.normal(size=(n, 160)), f_emb=rng.normal(size=(n, 160)),
                pa_local=rng.normal(size=(n, 160)), support_labels=np.repeat(np.arange(c), k),
                support_ids=['p%04d' % i for i in range(n)],
                classes=['class%02d' % i for i in range(c)], old_classes=['class00'])


def test_primal_dual_solution_objective_and_joint_rhs():
    rng = np.random.default_rng(9)
    for n, d in ((11, 4), (4, 11)):
        x, target = rng.normal(size=(n, d)), rng.normal(size=(n, 5))
        w, b, audit = core._ridge_fit(x, target)
        xc, tc = x-x.mean(axis=0), target-target.mean(axis=0)
        expected = np.linalg.solve(xc.T@xc+np.eye(d), xc.T@tc)
        np.testing.assert_allclose(w, expected, atol=1e-13)
        np.testing.assert_allclose(b, target.mean(axis=0)-x.mean(axis=0)@expected, atol=1e-13)
        one_w, one_b, _ = core._ridge_fit(x, target[:, :2])
        np.testing.assert_allclose(w[:, :2], one_w, atol=1e-14)
        np.testing.assert_allclose(b[:2], one_b, atol=1e-14)
        obj = core._objective(x, target, w, b)
        assert obj['gradient_norm'] < 1e-12
        assert obj['normal_equation_residual'] < 1e-12
        residual = x@w+b-target
        assert obj['loss_total'] == pytest.approx(.5*np.sum(residual**2)+.5*np.sum(w**2))
        assert audit['solver'] == ('primal' if d <= n else 'dual')
        assert audit['factorization_dim'] == min(n, d)
        assert audit['temporary_state_bytes'] == w.nbytes+b.nbytes


def test_duplicate_is_fixed_regularization_control_and_no_new_information_claim():
    rng = np.random.default_rng(20)
    x, target = rng.normal(size=(9, 5)), rng.normal(size=(9, 3))
    w, b, _ = core._ridge_fit(np.concatenate((x, x), axis=1), target)
    refw, refb, _ = core._ridge_fit(x, target, ridge=.5)
    np.testing.assert_allclose(w[:5]+w[5:], refw, atol=1e-13)
    np.testing.assert_allclose(b, refb, atol=1e-13)
    zero_w, zero_b, _ = core._ridge_fit(np.concatenate((x, np.zeros_like(x)), axis=1), target)
    baseline_w, baseline_b, _ = core._ridge_fit(x, target)
    np.testing.assert_allclose(zero_w[:5], baseline_w, atol=1e-13)
    np.testing.assert_array_equal(zero_w[5:], 0.)
    np.testing.assert_allclose(zero_b, baseline_b, atol=1e-13)


def test_k1_has_no_fit_no_holdout_and_26_classes_json():
    result = core.probe_branch_support(**fixture(k=1, c=26))
    assert result['folds'] == [] and result['fold_count'] == 0
    assert result['factorization_count'] == 0
    assert result['oof'] is result['reconstruction'] is result['paired'] is None
    assert result['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT'
    assert result['persistent_state_bytes'] == result['optimizer_steps'] == 0
    assert result['numerical']['spectrum_decomposition_count'] == 5
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('k', [2, 5])
def test_fold_coverage_metrics_physical_sum_and_bytes(k):
    inputs = fixture(k=k)
    result = core.probe_branch_support(**inputs)
    n, c = 3*k, 3
    assert result['fold_count'] == min(k, 3)
    assert result['factorization_count'] == 6*min(k, 3)
    assert len(result['physical_fold_assignment']) == n
    held_all = []
    for fold in result['folds']:
        held_all.extend(fold['held_ids'])
        assert set(fold['training_ids']).isdisjoint(fold['held_ids'])
        assert set(fold['training_ids']) | set(fold['held_ids']) == set(inputs['support_ids'])
        assert len(fold['stages']) == 6
        for stage in fold['stages']:
            count = len(fold['training_ids'])
            assert stage['train_physical_count'] == count
            assert stage['train_k'] == count//c
            assert stage['physical_loss_mass'] == count
            assert stage['target_norm_squared'] == pytest.approx(count*(1.-1./c))
            assert stage['ridge_coefficient'] == 1.
            assert stage['gradient_norm'] < 1e-12
            assert stage['temporary_state_bytes'] == 8*(stage['design_dim']+1)*stage['output_dim']
            assert stage['factorization_dim'] == min(count, stage['design_dim'])
            assert stage['all_states_estimated_from_trainfold_only']
            assert stage['loss_total'] == pytest.approx(stage['loss_data']+stage['loss_ridge'])
            assert stage['epoch'] is stage['learning_rate'] is None
    assert sorted(held_all) == sorted(inputs['support_ids'])
    for arm, data in result['oof'].items():
        assert len(data['rows']) == n
        assert data['metrics']['macro_accuracy'] == pytest.approx(data['metrics']['accuracy'])
        assert 0 <= data['metrics']['accuracy'] <= 1
    for background, data in result['reconstruction'].items():
        assert len(data['rows']) == n
        assert data['r2_linear'] == pytest.approx(1-data['squared_error_sum']/data['mean_baseline_squared_error_sum'])
    json.dumps(result, allow_nan=False)


def test_held_physical_perturbation_cannot_change_train_state(monkeypatch):
    inputs = fixture(k=5)
    snapshots = []
    original = core._ridge_fit
    def capture(x, y, ridge=1.):
        w, b, audit = original(x, y, ridge)
        snapshots.append((x.copy(), y.copy(), w.copy(), b.copy()))
        return w, b, audit
    monkeypatch.setattr(core, '_ridge_fit', capture)
    first = core.probe_branch_support(**inputs)
    before = snapshots[:6]
    held = set(first['folds'][0]['held_ids'])
    modified = copy.deepcopy(inputs)
    mask = np.array([pid in held for pid in inputs['support_ids']])
    for key in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'):
        modified[key][mask] = modified[key][mask]*-5+12
    snapshots.clear()
    core.probe_branch_support(**modified)
    for a, b in zip(before, snapshots[:6]):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_class_and_physical_permutation_invariance():
    inputs = fixture(k=2)
    first = core.probe_branch_support(**inputs)
    perm = np.array([4, 1, 0, 5, 3, 2])
    other = copy.deepcopy(inputs)
    for key in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local', 'support_labels'):
        other[key] = other[key][perm]
    other['support_ids'] = [inputs['support_ids'][i] for i in perm]
    other['classes'] = inputs['classes'][::-1]
    other['support_labels'] = 2-other['support_labels']
    second = core.probe_branch_support(**other)
    assert first['oof'] == second['oof']
    assert first['reconstruction'] == second['reconstruction']
    assert first['physical_fold_assignment'] == second['physical_fold_assignment']
    assert first['numerical']['spectra'] == second['numerical']['spectra']


def test_all_zero_near_zero_and_constant_numerical_rank():
    inputs = fixture(k=2)
    for key in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'):
        inputs[key].fill(0.)
    result = core.probe_branch_support(**inputs)
    assert all(v['numerical_rank'] == 0 for v in result['numerical']['spectra'].values())
    assert all(v['r2_linear'] is None for v in result['reconstruction'].values())
    for data in result['oof'].values():
        assert all(row['predicted_class'] == 'class00' for row in data['rows'])
    for key in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'):
        inputs[key].fill(1e-20)
    near = core.probe_branch_support(**inputs)
    assert all(v['numerical_rank'] == 0 for v in near['numerical']['spectra'].values())
    json.dumps(near, allow_nan=False)


@pytest.mark.parametrize('change', [
    lambda x: x.update(z_id=np.zeros((6, 159))),
    lambda x: x.update(z_id=np.zeros((6, 160), dtype=object)),
    lambda x: x['fft'].__setitem__((0, 0), np.nan),
    lambda x: x.update(support_labels=np.zeros(6, dtype=int)),
    lambda x: x.update(support_labels=np.arange(6, dtype=float)),
    lambda x: x.update(support_ids=['duplicate']*6),
    lambda x: x.update(classes=['a', 'a', 'b']),
    lambda x: x.update(old_classes=['absent']),
    lambda x: x.update(support_labels=np.array([0, 0, 0, 1, 2, 2])),
    lambda x: x.update(pa_local=np.zeros((7, 160))),
])
def test_invalid_inputs(change):
    inputs = fixture()
    change(inputs)
    with pytest.raises(ValueError):
        core.probe_branch_support(**inputs)


def test_public_api_has_no_query_or_hyperparameter_selection_and_config_is_defensive():
    signature = inspect.signature(core.probe_branch_support)
    assert set(signature.parameters) == {'z_id', 'fft', 't_emb', 'f_emb', 'pa_local',
        'support_labels', 'support_ids', 'classes', 'old_classes'}
    result = core.probe_branch_support(**fixture(k=1))
    result['config']['branches'].append('forbidden')
    assert core.probe_branch_support(**fixture(k=1))['config']['branches'] == ['t_emb', 'f_emb', 'pa_local']


def test_frozen_config_file_matches_code():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'configs/d92_branch_support_probe_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8')) == core.FROZEN_CONFIG


def test_analytic_objective_gradient_against_finite_difference():
    rng = np.random.default_rng(77)
    x, target = rng.normal(size=(7, 4)), rng.normal(size=(7, 2))
    w, b = rng.normal(size=(4, 2)), rng.normal(size=2)
    residual = x@w+b-target
    expected = np.concatenate(((x.T@residual+w).ravel(), residual.sum(axis=0)))
    vector = np.concatenate((w.ravel(), b))
    def loss(v):
        return core._objective(x, target, v[:8].reshape(4, 2), v[8:])['loss_total']
    finite = np.empty_like(vector)
    for i in range(len(vector)):
        left, right = vector.copy(), vector.copy()
        left[i] -= 1e-6
        right[i] += 1e-6
        finite[i] = (loss(right)-loss(left))/2e-6
    np.testing.assert_allclose(finite, expected, rtol=1e-7, atol=1e-7)
