"""Synthetic production archives only; no real experiment artifacts."""
if __name__=='__main__':
    import ast
    import json
    from pathlib import Path
    task_root=Path(__file__).resolve().parents[1]
    task_paths=(task_root/'tools/summarize_d92_margin_joint_probe.py',Path(__file__),
        task_root/'tests/test_d92_margin_analysis_math.py',task_root/'docs/D92_MARGIN_JOINT_ANALYSIS_IMPLEMENTATION_20261001.md')
    for task_path in task_paths:
        task_bytes=task_path.read_bytes();task_source=task_bytes.decode('utf-8')
        assert not task_bytes.startswith(b'\xef\xbb\xbf') and '\ufffd' not in task_source and '\r' not in task_source
        if task_path.suffix=='.py':ast.parse(task_source,filename=str(task_path))
    print(json.dumps(dict(status='STATIC_ONLY',paths=[str(v) for v in task_paths])))
    raise SystemExit(0)
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import summarize_d92_margin_joint_probe as summary
import evaluate_d92_margin_joint_probe as entry
from run_d92_margin_joint_probe import budget_for_spec, KS, NEW_COUNTS
QP_RESOURCES=dict(max_transitions=256,max_factor_buffer_bytes=8_000_000)


def synthetic_selection():
    old = ['a' + str(j) for j in range(6)]
    return dict(receiver_scenes=[['synthetic_rx', 'practical_high']], support_seed=11,
        ks=KS, new_counts=NEW_COUNTS, splits=[dict(split_id='synthetic_' + str(k) + '_' + str(new),
        receiver='synthetic_rx', scenario='practical_high', k=k, new_count=new, support_seed=11,
        registered_classes=old + ['z' + str(j).zfill(2) for j in range(new)]) for k in KS for new in NEW_COUNTS])


def dynamic_spec(rows):
    return dict(run_id='synthetic_run', rows=[dict(row_id='row_' + str(j), cohort='synthetic') for j in range(rows)],
        probe=dict(cohorts=dict(synthetic=dict(selection=synthetic_selection())),qp_resources=dict(QP_RESOURCES)))


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
    result = entry.probe_margin_joint(**raw, support_labels=labels, support_ids=ids, classes=classes, old_classes=old,
        **QP_RESOURCES,context=dict(run_id='synthetic_run', row_id='synthetic_row', split_id=split['split_id']), state_callback=archive)
    archive.finalize('COMPLETE')
    record = entry.json_native(dict(result, **entry.split_identity(split, old), scope=entry.SCOPE, query_rows_used=0, source_rows_used=0))
    return dict(root=root, record=record, split=split, old=old)


@pytest.fixture(scope='module')
def production(tmp_path_factory):
    # The candidate is a fixture generator only. All following math is independent.
    return _production_case(tmp_path_factory.mktemp('margin_synthetic_production'))


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
    from cvsrffi import d92_margin_joint_local_ridge as core
    from cvsrffi import d92_margin_qp_head as kernel
    from cvsrffi import d92_affine_joint_local_ridge as affine
    def forbidden(*args, **kwargs): raise AssertionError('Candidate math cannot validate itself')
    for module, names in ((core, ('prepare_margin_joint_training', 'evaluate_margin_joint_objective', 'fit_margin_joint_local_ridge', '_forward', '_backward')),
                          (kernel, ('fit_margin_qp_head','predict_margin_qp_head','margin_qp_head_vjp')),
                          (affine, ('_solve_affine_head', '_affine_adjoint'))):
        for name in names: monkeypatch.setattr(module, name, forbidden)
    for name in ('prepare_margin_joint_training','fit_margin_joint_local_ridge','probe_margin_joint'):
        monkeypatch.setattr(entry,name,forbidden)
    monkeypatch.setattr(core.MarginJointState, 'score', forbidden)
    resolver = summary.StateResolver(production['root']); logs, events, metrics, training = _verify(production, resolver)
    resolver.finalize()
    assert len(training) == production['record']['candidate_stage_count']
    assert len(events) > len(training) and len(logs) > len(training)
    assert set(metrics) == {'oof', 'proxy'} and all(set(value) == set(entry.PATHS) for value in metrics.values())
    assert resolver.analysis_work['independent_general_solve_count'] > 0
    assert resolver.analysis_work['independent_spectral_diagnostic_count'] > 0
    assert all(value['initial_objective'] is None or value['initial_objective']['loss_proximal'] == 0 for value in training)


def _c_stage(case): return case['record']['folds'][0]['candidate_stages'][1]


def test_core_three_field_refs_bind_to_exact_full_manifest_entry(production):
    resolver = summary.StateResolver(production['root'])
    refs = (production['record']['folds'][0]['preparations'][0]['prepared_state_ref'],
            _c_stage(production)['initial_objective']['inner_folds'][0]['head_state_ref'])
    for ref in refs:
        full = resolver.refs[ref['path']]
        assert ref != full and set(ref) == set(full)
        assert all(ref[key] == full[key] for key in full if key != 'arrays')
        assert set(ref['arrays']) == set(full['arrays'])
        for name, meta in ref['arrays'].items():
            assert set(meta) == {'shape', 'dtype', 'nbytes'}
            assert meta == {key: full['arrays'][name][key] for key in meta}
            assert full['arrays'][name]['all_finite'] is True
            assert full['arrays'][name]['nonfinite_count'] == 0
        core_view = resolver(ref); archive_view = resolver(full)
        assert core_view is archive_view
        assert all(not value.flags.writeable for value in core_view.values())


def test_production_C_scale_scalar_and_QP_slack_vector_are_distinct(production):
    resolver = summary.StateResolver(production['root'])
    path = production['record']['folds'][0]
    b_stage, c_stage = path['candidate_stages']
    b = resolver(b_stage['final_state_ref']); c = resolver(c_stage['final_state_ref'])
    assert b['s0'].shape==() and b['s0'].size==1 and np.isfinite(b['s0']).all()
    assert 'reference_s0' not in b
    assert c['reference_s0'].shape==() and c['reference_s0'].size==1
    assert np.isfinite(c['reference_s0']).all()
    assert c['s0'].shape==(len(c['old_indices'])*(len(c_stage['final_fit']['classes'])-1),)
    assert np.isfinite(c['s0']).all()
    assert c_stage['final_state_ref']['arrays']['reference_s0']['shape']==[]
    cert = summary.verify_margin_head(c,c_stage['final_fit']['margin_qp_audit'])
    assert cert['primal_objective']==pytest.approx(cert['dual_objective'],abs=1e-11)
    summary.verify_head(b_stage['final_state_ref'],b_stage['final_fit'],resolver)
    summary.verify_head(c_stage['final_state_ref'],c_stage['final_fit'],resolver,(b,path['b_classes']))


@pytest.mark.parametrize('kind', ['scale_shape', 'scale_value', 'QP_shape', 'QP_value'])
def test_production_scalar_vector_shape_and_value_tampering_are_rejected(production, kind):
    stage = _c_stage(production); target = stage['final_state_ref']
    def mutate(data):
        if kind=='scale_shape': data['reference_s0'] = data['reference_s0'].reshape(1)
        elif kind=='scale_value': data['reference_s0'] = data['reference_s0']+.01
        elif kind=='QP_shape': data['s0'] = np.asarray(data['s0'][0])
        else: data['s0'][0] += .01
    with pytest.raises(ValueError, match='reference_s0|Original old R0 trace|margin primal/dual mismatch: s0'):
        _verify(production,ChangedResolver(production['root'],target,mutate))


@pytest.mark.parametrize('kind', ['namespace', 'key', 'path', 'file_bytes', 'archive_seconds',
    'array_summaries', 'array_inventory', 'shape', 'dtype', 'nbytes', 'extra_array_metadata', 'missing_identity'])
def test_core_view_does_not_relax_identity_or_numeric_metadata(production, monkeypatch, kind):
    resolver = summary.StateResolver(production['root'])
    ref = deepcopy(production['record']['folds'][0]['preparations'][0]['prepared_state_ref'])
    name = next(iter(ref['arrays']))
    if kind == 'namespace': ref[kind] += '_other'
    elif kind == 'key': ref[kind] += '_other'
    elif kind == 'path': ref[kind] = 'state_arrays/unregistered.npz'
    elif kind in ('file_bytes', 'archive_seconds'): ref[kind] += 1
    elif kind == 'array_summaries': ref[kind][name]['norm'] += 1
    elif kind == 'array_inventory': ref['arrays'].pop(name)
    elif kind == 'shape': ref['arrays'][name][kind] = [999]
    elif kind == 'dtype': ref['arrays'][name][kind] = 'float32'
    elif kind == 'nbytes': ref['arrays'][name][kind] += 1
    elif kind == 'extra_array_metadata': ref['arrays'][name]['unknown'] = 0
    else: ref.pop('namespace')
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Reject unbound ref before loading')))
    with pytest.raises(ValueError, match='State reference'): resolver(ref)


@pytest.mark.parametrize('kind', ['all_finite', 'nonfinite_count', 'failed_numeric_state'])
def test_complete_manifest_finiteness_cannot_be_hidden_by_core_view(production, monkeypatch, kind):
    resolver = summary.StateResolver(production['root'])
    ref = production['record']['folds'][0]['preparations'][0]['prepared_state_ref']
    full = resolver.refs[ref['path']]
    if kind == 'failed_numeric_state':
        full[kind] = True
        ref = deepcopy(ref); ref[kind] = True
    else:
        name = next(iter(full['arrays']))
        full['arrays'][name][kind] = False if kind == 'all_finite' else 1
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Reject failed metadata before loading')))
    with pytest.raises(ValueError, match='[Ff]inite|[Ff]ailed'): resolver(ref)


@pytest.mark.parametrize('kind', ['prior', 'constraint', 'inherit'])
def test_prior_constraint_actual_B_inheritance_tampering_rejected(production, kind):
    target = _c_stage(production)['final_state_ref']
    def mutate(data):
        key = dict(prior='M_train', constraint='multipliers', inherit='prior_B_U')[kind]
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
    elif kind == 'counter': stage['margin_qp_forward_spectral_checks'] += 1
    elif kind == 'physical_ids': path['c_training_ids'][0] = 'unrelated_physical_id'
    else: path['training_events'] = [value for value in path['training_events'] if value['event'] != 'MARGIN_JOINT_FINAL']
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
        assert [stage['state'] for stage in path['candidate_stages']] == ['B_MARGIN']


@pytest.mark.parametrize('variant', ['zero', 'tau0'])
def test_zero_kernel_and_exact_equivalence_tau0_independently_verified(tmp_path, variant):
    case = _production_case(tmp_path, k=3, variant=variant); _, _, _, training = _verify(case)
    assert all(value['optimizer_steps'] == 0 for value in training)
    resolver = summary.StateResolver(tmp_path)
    stage = case['record']['folds'][0]['candidate_stages'][1]; data = resolver(stage['final_state_ref'])
    if variant == 'zero':
        assert summary._scalar(data, 'gamma') is None
        np.testing.assert_array_equal(data['K'],0.)
        np.testing.assert_allclose(data['scores'],data['M_held']+data['b'],atol=1e-12)
    else:
        assert summary._scalar(data, 'tau') == 0 and summary._scalar(data, 'gamma') is not None
        assert data['alpha'].shape[0]==len(data['train_labels'])
        assert len(data['slack'])==len(data['old_indices'])*(len(case['record']['classes'])-1)


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



@pytest.mark.parametrize('kind',['rhs','peak_sum','ledger','snapshot'])
def test_actual_QP_ledger_and_peak_tampering_are_rejected(production,kind):
    record=deepcopy(production['record']);stage=record['folds'][0]['candidate_stages'][1]
    if kind=='rhs':stage['head_triangular_rhs_count']+=1
    elif kind=='peak_sum':record['margin_qp_peak_factor_buffer_bytes']+=1
    elif kind=='ledger':stage['final_fit']['margin_qp_audit']['solves'][0]['rhs_columns']+=1
    else:stage['final_fit']['margin_qp_audit']['compact_snapshot_dense_work_units']+=1
    with pytest.raises(ValueError):_verify(production,record=record)


def test_bounded_raw_array_cache_preserves_math_and_tamper_rejection(production):
    uncached=summary.StateResolver(production['root'],cache_budget_bytes=0,max_entries=0)
    cached=summary.StateResolver(production['root'],cache_budget_bytes=64*1024*1024,max_entries=64)
    _,_,a,ta=_verify(production,uncached);_,_,b,tb=_verify(production,cached)
    assert a==b and ta==tb
    assert cached.cache_statistics()['peak_numeric_bytes']<=64*1024*1024
    assert cached.cache_statistics()['hit_count']>0
    target=_c_stage(production)['final_state_ref']
    with pytest.raises(ValueError):_verify(production,ChangedResolver(production['root'],target,lambda data:data['b'].__setitem__(0,data['b'][0]+.01)))


def test_stage_stream_real_serialization_and_no_query_A_fabrication(production):
    logs,_,metrics,training=_verify(production)
    stream=[dict(entry.compact_event(v),schema=entry.SCHEMA,method=entry.METHOD,split_id=production['split']['split_id']) for v in logs]
    summary.verify_stage_stream(iter(stream),logs,production['split']['split_id'])
    for paths in metrics.values():
        for values in paths.values():
            assert values['A_old_accuracy'] is values['adaptation_gain_B_minus_A'] is None
    assert all(v['evidence_scope']=='INNER_SUPPORT_TRAINING_NOT_VALIDATION' for v in training)
    bad=deepcopy(stream);bad[0]['method']='not-margin'
    with pytest.raises(ValueError):summary.verify_stage_stream(iter(bad),logs,production['split']['split_id'])


def test_failed_partial_work_is_preserved_but_never_a_complete_summary(tmp_path):
    from test_d92_margin_joint_local_ridge import fixture
    from cvsrffi import d92_margin_joint_local_ridge as core
    args=fixture(k=3);args['old_classes']=('c0','c1')
    archive=entry.StateArchive(tmp_path)
    with pytest.raises(core.NumericalFailure) as caught:
        entry.probe_margin_joint(**args,max_transitions=1,max_factor_buffer_bytes=8_000_000,
            context=dict(run_id='synthetic-failure',row_id='synthetic-row',split_id='synthetic-split'),state_callback=archive)
    partial=caught.value.registration_context
    assert partial['workload_complete'] is False
    failed=summary.verify_failed_fit(caught.value.audit_dict(),caught.value.arrays)
    assert failed['complete_run'] is failed['optimality_certified'] is failed['cumulative_work_verified'] is False
    assert failed['known_last_solver_work']['transitions']==1
    assert failed['numeric_snapshot_status'].startswith('CURRENT_COMPACT_STATE')
    archive.finalize('INCOMPLETE')
    with pytest.raises(ValueError,match='Incomplete'):summary.StateResolver(tmp_path)


def test_unknown_failed_work_is_not_zero_filled():
    with pytest.raises(ValueError,match='Unknown failed solver work'):
        summary.verify_failed_fit(dict(status='TECHNICAL_FAILURE',failure_code='UNKNOWN',qp_failure_audit={}),{})
