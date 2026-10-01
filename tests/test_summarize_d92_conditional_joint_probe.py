"""Production candidate synthetic traces are verified by independent analysis."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import summarize_d92_conditional_joint_probe as summary
import evaluate_d92_conditional_joint_probe as entry
from run_d92_conditional_joint_probe import budget_for_spec, KS, NEW_COUNTS


def synthetic_selection():
    old = ['a' + str(j) for j in range(6)]
    return dict(receiver_scenes=[['synthetic_rx', 'practical_high']], support_seed=11,
        ks=KS, new_counts=NEW_COUNTS, splits=[dict(split_id='synthetic_' + str(k) + '_' + str(new),
        receiver='synthetic_rx', scenario='practical_high', k=k, new_count=new, support_seed=11,
        registered_classes=old + ['z' + str(j).zfill(2) for j in range(new)]) for k in KS for new in NEW_COUNTS])


def dynamic_spec(rows):
    return dict(run_id='synthetic_run', rows=[dict(row_id='row_' + str(j), cohort='synthetic') for j in range(rows)],
        probe=dict(cohorts=dict(synthetic=dict(selection=synthetic_selection()))))


def _production_case(root, *, k=3, new=1, variant='regular'):
    rng = np.random.default_rng(451)
    old = ['a', 'b']; classes = old + (['z'] if new else []); labels = np.repeat(np.arange(len(classes)), k).astype(np.int64)
    ids = ['physical_' + str(j).zfill(3) for j in range(len(labels))]
    raw = {key: rng.normal(scale=.03, size=(len(labels), width)) for key, width in zip(entry.BRANCHES, (160, 96, 160, 160, 160))}
    for values in raw.values(): values[np.arange(len(labels)), labels] += .05
    if variant == 'zero':
        for values in raw.values(): values.fill(0.)
    elif variant == 'tau0':
        for values in raw.values(): values[k:2 * k] = values[:k]
    split = dict(split_id='synthetic_split', receiver='synthetic_rx', scenario='practical_high', k=k, support_seed=11,
                 registered_classes=classes, new_count=new, support_ids=ids, support_labels=labels.tolist())
    archive = entry.StateArchive(root)
    result = entry.probe_conditional_joint(**raw, support_labels=labels, support_ids=ids, classes=classes, old_classes=old,
        context=dict(run_id='synthetic_run', row_id='synthetic_row', split_id=split['split_id']), state_callback=archive)
    archive.finalize('COMPLETE')
    record = entry.json_native(dict(result, **entry.split_identity(split, old), scope=entry.SCOPE, query_rows_used=0, source_rows_used=0))
    return dict(root=root, record=record, split=split, old=old)


@pytest.fixture(scope='module')
def production(tmp_path_factory):
    # The candidate is a fixture generator only. All following math is independent.
    return _production_case(tmp_path_factory.mktemp('conditional_synthetic_production'))


def _verify(case, resolver=None, record=None):
    return summary.verify_record(case['record'] if record is None else record, case['split'], case['old'],
        summary.StateResolver(case['root']) if resolver is None else resolver,
        dict(run_id='synthetic_run', row_id='synthetic_row', split_id=case['split']['split_id']))


class ChangedResolver:
    def __init__(self, root, target, mutate):
        self.base = summary.StateResolver(root); self.target, self.mutate = target['path'], mutate
        self.analysis_work = self.base.analysis_work
    def __call__(self, ref):
        data = self.base(ref)
        if ref['path'] != self.target: return data
        copied = {key: np.array(value, copy=True) for key, value in data.items()}; self.mutate(copied); return copied
    def verify_tree(self, value): self.base.verify_tree(value)


def test_real_candidate_complete_trace_independent_summary_without_candidate_calls(production, monkeypatch):
    from cvsrffi import d92_conditional_joint_local_ridge as core
    from cvsrffi import d92_conditional_affine_kernel as kernel
    from cvsrffi import d92_affine_joint_local_ridge as affine
    def forbidden(*args, **kwargs): raise AssertionError('Candidate math cannot validate itself')
    for module, names in ((core, ('prepare_conditional_joint_training', 'evaluate_conditional_joint_objective', 'fit_conditional_joint_local_ridge', '_forward', '_backward')),
                          (kernel, ('fit_conditional_affine', 'conditional_affine_adjoint')),
                          (affine, ('_solve_affine_head', '_affine_adjoint'))):
        for name in names: monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(core.ConditionalJointState, 'score', forbidden)
    resolver = summary.StateResolver(production['root']); logs, events, metrics, training = _verify(production, resolver)
    resolver.finalize()
    assert len(training) == production['record']['candidate_stage_count']
    assert len(events) > len(training) and len(logs) > len(training)
    assert set(metrics) == {'oof', 'proxy'} and all(set(value) == set(entry.PATHS) for value in metrics.values())
    assert resolver.analysis_work['independent_general_solve_count'] > 0
    assert resolver.analysis_work['independent_spectral_diagnostic_count'] == 0
    assert all(value['initial_objective'] is None or value['initial_objective']['loss_proximal'] == 0 for value in training)


def _c_stage(case): return case['record']['folds'][0]['candidate_stages'][1]


@pytest.mark.parametrize('kind', ['prior', 'constraint', 'inherit'])
def test_prior_constraint_actual_B_inheritance_tampering_rejected(production, kind):
    target = _c_stage(production)['final_state_ref']
    def mutate(data):
        key = dict(prior='M_train', constraint='beta', inherit='prior_B_U')[kind]
        data[key].flat[0] += .01
    with pytest.raises(ValueError): _verify(production, ChangedResolver(production['root'], target, mutate))


def test_nonzero_constant_projection_adjoint_tampering_rejected(production):
    stage = _c_stage(production); assert stage['gradients']
    target = stage['gradients'][0]['objective']['inner_folds'][0]['head_state_ref']
    resolver = summary.StateResolver(production['root']); assert np.linalg.norm(resolver(target)['adjoint_g_b']) > 1e-10
    with pytest.raises(ValueError):
        _verify(production, ChangedResolver(production['root'], target, lambda data: data['adjoint_g_b'].fill(0.)))


def test_legacy_plus_Z_coordinate_gradient_rejected(production):
    resolver = summary.StateResolver(production['root']); target = None
    for fold in production['record']['folds']:
        for stage in fold['candidate_stages']:
            for gradient in stage['gradients']:
                ref = gradient['state_ref']
                if np.linalg.norm(resolver(ref)['Z']) > 1e-8: target = ref; break
            if target is not None: break
        if target is not None: break
    assert target is not None, 'Synthetic trace must include an accepted nonzero coordinate state'
    def mutate(data): data['g_Z'] += data['Z']
    with pytest.raises(ValueError, match='gradient'):
        _verify(production, ChangedResolver(production['root'], target, mutate))


@pytest.mark.parametrize('kind', ['CE_only', 'counter', 'physical_ids', 'missing_final'])
def test_complete_trace_semantic_tampering_rejected(production, kind):
    record = deepcopy(production['record']); path = record['folds'][0]; stage = path['candidate_stages'][1]
    if kind == 'CE_only': stage['initial_objective']['loss_proximal'] = .01
    elif kind == 'counter': stage['spectral_diagnostic_count'] += 1
    elif kind == 'physical_ids': path['c_training_ids'][0] = 'unrelated_physical_id'
    else: path['training_events'] = [value for value in path['training_events'] if value['event'] != 'CONDITIONAL_JOINT_FINAL']
    with pytest.raises(ValueError): _verify(production, record=record)


@pytest.mark.parametrize('new', [0, 1])
def test_true_K1_full_head_and_new0_reuse_have_no_independent_held_metrics(tmp_path, new):
    case = _production_case(tmp_path, k=1, new=new); _, _, metrics, training = _verify(case)
    assert metrics == {'oof': None, 'proxy': None}
    path = case['record']['full_support']; assert path['prediction_status'] == 'NO_HELD_PREDICTIONS'
    assert len(training) == 1 + int(bool(new))
    for value in path['paths'].values(): assert value['metrics'] == dict.fromkeys(entry.METRICS)
    if not new:
        assert path['c_reuses_b_candidates'] is True
        assert [stage['state'] for stage in path['candidate_stages']] == ['B_CONDITIONAL']


@pytest.mark.parametrize('variant', ['zero', 'tau0'])
def test_zero_kernel_and_exact_equivalence_tau0_independently_verified(tmp_path, variant):
    case = _production_case(tmp_path, k=3, variant=variant); _, _, _, training = _verify(case)
    assert all(value['optimizer_steps'] == 0 for value in training)
    resolver = summary.StateResolver(tmp_path)
    stage = case['record']['folds'][0]['candidate_stages'][1]; data = resolver(stage['final_state_ref'])
    if variant == 'zero':
        assert summary._scalar(data, 'gamma') is None
        np.testing.assert_array_equal(data['scores'], data['M_held'])
    else:
        assert summary._scalar(data, 'tau') == 0 and summary._scalar(data, 'gamma') is not None
        assert len(data['old_representative_indices']) < len(data['old_indices'])
        np.testing.assert_allclose(data['train_scores'][data['old_indices']], data['M_train'][data['old_indices']], atol=1e-12)


@pytest.mark.parametrize('rows', [1, 2, 3, 5])
def test_completion_uses_dynamic_rows_and_declared_axes(rows):
    spec = dynamic_spec(rows); exact = budget_for_spec(spec)['total']['exact']
    done = dict(status=entry.STATUS, run_id=spec['run_id'], model_rows=rows, completed_rows=rows, **exact)
    summary.completion_check(done, spec)
    assert exact['episodes'] == 20 * rows
    done['completed_rows'] -= 1
    with pytest.raises(ValueError, match='incomplete'): summary.completion_check(done, spec)


def test_incomplete_archive_is_rejected_before_numeric_loading(tmp_path, monkeypatch):
    archive = entry.StateArchive(tmp_path)
    namespace = json.dumps(dict(run_id='synthetic_run', row_id='synthetic_row', state='B0'), sort_keys=True, separators=(',', ':'))
    archive(namespace + '/one', dict(value=np.ones(1)))
    archive.finalize('INCOMPLETE')
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Do not load partial numeric archive')))
    with pytest.raises(ValueError, match='Incomplete'): summary.StateResolver(tmp_path)


def test_parent_first_harmonic_mean_is_not_harmonic_of_pooled_means():
    group = {}; base = dict.fromkeys(entry.METRICS)
    for old, new in ((.2, 1.), (1., .2)):
        values = dict(base, C_old_accuracy=old, C_new_accuracy=new, C_h=2 * old * new / (old + new),
                      C_abs_new_old_gap=abs(old - new), total_old_accuracy_drop=.1)
        summary._add(group, ('cell',), values)
    rows = summary._statistics(group, ('cell',)); lookup = {row['metric']: row for row in rows}
    assert lookup['C_h']['mean'] == pytest.approx(1 / 3)
    assert lookup['C_h']['mean'] != pytest.approx(.6)
    assert lookup['C_abs_new_old_gap']['mean'] == pytest.approx(.8)
