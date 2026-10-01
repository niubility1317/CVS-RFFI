"""Production Margin callbacks and archives from synthetic features only."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import evaluate_d92_margin_joint_probe as entry

QP_RESOURCES = dict(max_transitions=256, max_factor_buffer_bytes=8_000_000)


def probe(**kwargs):
    """Every synthetic call supplies explicit component resource limits."""
    return entry.probe_margin_joint(**kwargs, **QP_RESOURCES)


def synthetic(k=2,new=True,zero=False):
    old=['old-'+str(i) for i in range(6)];classes=np.asarray(old+(['new-0','new-1'] if new else []))
    labels=np.repeat(np.arange(len(classes),dtype=np.int64),k);rng=np.random.default_rng(841)
    raw={name:np.zeros((len(labels),width),dtype=np.float32) if zero else rng.normal(size=(len(labels),width)).astype(np.float32)
         for name,width in zip(entry.BRANCHES,(160,96,160,160,160))}
    return dict(raw,support_labels=labels,support_ids=np.asarray([classes[y]+'-'+str(i%k) for i,y in enumerate(labels)]),
        classes=classes,old_classes=np.asarray(old))


def scope():return dict(run_id='synthetic-run',row_id='synthetic-row',split_id='synthetic-parent')


def production(out,inputs):
    raw={key:inputs[key] for key in entry.BRANCHES}
    k=len(inputs['support_labels'])//len(inputs['classes'])
    split=dict(split_id=scope()['split_id'],receiver='synthetic-rx',scenario='practical_high',k=k,support_seed=0,
        registered_classes=inputs['classes'],support_ids=inputs['support_ids'],support_labels=inputs['support_labels'])
    producer=dict(feature_array_bytes=sum(v.nbytes for v in raw.values()),feature_file_bytes=0)
    config=dict(algorithm=entry.PROBE_CONFIG,producer_matrix={},selection={},qp_resources=dict(QP_RESOURCES))
    with patch.object(entry,'read',return_value=dict(channel=entry.CHANNEL,scenarios=entry.SCENARIOS)), \
         patch.object(entry,'validate_selection'), \
         patch.object(entry,'load_support',return_value=(raw,[],inputs['old_classes'],producer,{},{})), \
         patch.object(entry,'selected_tasks',return_value=[(split,np.arange(len(inputs['support_labels'])),inputs['support_labels'])]), \
         contextlib.redirect_stdout(io.StringIO()):
        return entry.evaluate(support_features='synthetic-cache',capsule='synthetic-capsule',output=out,config=config,
            expected_capsule_id='synthetic-capsule',expected_checkpoint_sha256='a'*64,expected_model_seed=0,
            run_id=scope()['run_id'],row_id=scope()['row_id'])


def lines(path):return [json.loads(row) for row in path.read_text(encoding='utf-8').splitlines()]


@pytest.mark.parametrize('new',[True,False])
def test_actual_production_callback_archives_scope_schema_and_all_fields(tmp_path,new):
    out=tmp_path/'probe';marker=production(out,synthetic(new=new));record=lines(out/'fit_trace.jsonl')[0]
    stages=lines(out/'fit_stages.jsonl');events=lines(out/'training_events.jsonl');small=lines(out/'training_events_compact.jsonl')
    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert marker['schema']==record['schema']==entry.SCHEMA and marker['method']==entry.METHOD
    assert manifest['schema']=='d92_margin_joint_state_archive_v1' and manifest['status']=='COMPLETE'
    paths=record['folds']+record['oneshot_proxy']['trials'];expected=[];actualevents=[]
    order=[('BASE_FIT','B0')]+([('BASE_FIT','C0')] if new else [])
    order += [('MARGIN_PREPARATION','B'),('CANDIDATE_FIT','B_MARGIN')]
    if new:order += [('MARGIN_PREPARATION','C'),('CANDIDATE_FIT','C_MARGIN_seq')]
    for path in paths:
        assert path['prediction_status']=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN'
        rows=[dict(event='BASE_FIT',**stage) for stage in path['stages']]
        for prep,stage in zip(path['preparations'],path['candidate_stages']):
            rows += [dict(event='MARGIN_PREPARATION',**prep),dict(event='CANDIDATE_FIT',**stage)]
        expected.extend(dict(entry.compact_event(row),schema=entry.SCHEMA,method=entry.METHOD,split_id=scope()['split_id']) for row in rows)
        actualevents.extend(path['training_events'])
        assert set(path['paths'])=={'R0','R_MARGIN_seq'}
        for item in path['paths'].values():assert item['metrics']['A_old_accuracy'] is item['metrics']['adaptation_gain_B_minus_A'] is None
        if not new:
            assert path['c_reuses_b_candidates'] and len(path['candidate_stages'])==1
            for value in path['paths'].values():assert value['b_scores']==value['c_scores']
    assert stages==expected and [(v['event'],v['state']) for v in stages]==order*len(paths)
    assert events==actualevents and small==[entry.compact_event(v) for v in events]
    allrefs=manifest['files']
    for ref in allrefs:
        namespace=json.loads(ref['namespace'])
        assert all(namespace[key]==scope()[key] for key in ('run_id','row_id','split_id'))
        assert all(key in namespace for key in ('state','scope','fold','trial','parent_k','train_k'))
        with np.load(out/ref['path'],allow_pickle=False) as arrays:
            assert set(arrays.files)==set(ref['arrays'])
            for key,meta in ref['arrays'].items():assert list(arrays[key].shape)==meta['shape'] and arrays[key].nbytes==meta['nbytes']
    assert record['factorization_count']==sum(record[k] for k in ('baseline_factorization_count','inner_factorization_count','final_factorization_count','prior_factorization_count'))
    assert marker['qp_resources']==record['qp_resources']==QP_RESOURCES
    assert record['candidate_stage_count']==record['ajlr_stage_count']
    assert marker['optimizer_steps']==0  # K2 outer folds and all proxies train at K1.
    assert (out/'training.log').is_file() and (out/'compact.csv').is_file() and (out/'training_events_compact.csv').is_file()
    with pytest.raises(FileExistsError):production(out,synthetic(new=new))


def test_actual_B_object_passed_and_other_row_replacement_rejected():
    original_prepare=entry.prepare_margin_joint_training;original_fit=entry.fit_margin_joint_local_ridge;states=[];calls=[]
    def prepare(**kwargs):
        calls.append(('prepare',kwargs['context']['stage']))
        if kwargs['context']['stage']=='C':assert kwargs['inherited'] is states[-1]
        return original_prepare(**kwargs)
    def fit(prepared,**kwargs):
        calls.append(('fit',kwargs['mode']));state=original_fit(prepared,**kwargs)
        if kwargs['mode']=='C_seq':assert state.prior is states[-1]
        states.append(state);return state
    with patch.object(entry,'prepare_margin_joint_training',side_effect=prepare),patch.object(entry,'fit_margin_joint_local_ridge',side_effect=fit):
        probe(**synthetic(k=1),context=scope())
    assert calls==[('prepare','B'),('fit','B'),('prepare','C'),('fit','C_seq')]
    def replace_B(**kwargs):
        if kwargs['inherited'] is not None:
            old=kwargs['inherited'];audit=old.audit_dict();audit['preparation']['row_id']='different-row'
            kwargs['inherited']=replace(old,audit=audit)
        return original_prepare(**kwargs)
    with patch.object(entry,'prepare_margin_joint_training',side_effect=replace_B),pytest.raises(ValueError,match='row_id'):
        probe(**synthetic(k=1),context=scope())


def test_actual_trained_production_events_and_numeric_refs(tmp_path):
    out=tmp_path/'trained';marker=production(out,synthetic(k=3))
    events=lines(out/'training_events.jsonl')
    names={row['event'] for row in events}
    assert {'MARGIN_JOINT_INITIAL','MARGIN_JOINT_GRADIENT','MARGIN_JOINT_FINAL'}<=names
    assert marker['backward_evaluation_count']>0
    for row in events:
        assert row['schema']==entry.SCHEMA and row['method']==entry.METHOD
        assert all(row[key]==scope()[key] for key in scope())
        assert row['scope'] in ('support_oof','support_oneshot_proxy')
    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert any(ref['key'].startswith('gradient_') for ref in manifest['files'])


def test_support_truth_join_occurs_after_fixed_all_class_scores():
    seen=[];original=entry.assess_paths
    def log(row):seen.append((row['event'],row['state']))
    def scorer(evidence,old):
        assert seen[-1]==('CANDIDATE_FIT','C_MARGIN_seq')
        for path in evidence.values():
            assert isinstance(path['c_scores'],list) and all(len(row)==len(path['c_classes']) for row in path['c_scores'])
        return original(evidence,old)
    with patch.object(entry,'assess_paths',side_effect=scorer):
        probe(**synthetic(),context=scope(),log_callback=log)


def test_forbidden_inputs_true_k1_and_failure_preservation(tmp_path):
    for name in ('query_features','query_truth','query_roles','source_features','checkpoint','inherited','target_statistics'):
        with pytest.raises(TypeError):probe(**synthetic(k=1),**{name:object()})
    result=probe(**synthetic(k=1,zero=True),context=scope())
    assert result['oof'] is result['oneshot_proxy'] is None and result['optimizer_steps']==0
    assert result['full_support']['prediction_status']=='NO_HELD_PREDICTIONS'
    for value in result['full_support']['paths'].values():assert value['metrics']==dict.fromkeys(entry.METRICS)
    original=entry.fit_margin_joint_local_ridge
    def fail_C(prepared,**kwargs):
        if kwargs['mode']=='C_seq':raise FloatingPointError('synthetic C failure')
        return original(prepared,**kwargs)
    out=tmp_path/'failure'
    with patch.object(entry,'fit_margin_joint_local_ridge',side_effect=fail_C),pytest.raises(FloatingPointError):production(out,synthetic(k=1))
    assert not (out/'probe_complete.json').exists()
    failure=json.loads((out/'probe_failed.json').read_text(encoding='utf-8'))
    assert failure['failure_context']['completed_candidate_states'].keys()=={'B_MARGIN'}
    assert json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))['status']=='INCOMPLETE'
    assert failure['workload_complete'] is False
    assert failure['failure_context']['failed_fit'] is None
    assert failure['failure_context']['failed_fit_counters'] is None
    assert failure['failure_context']['counters_scope']=='COMPLETED_BASELINES_PREPARATIONS_AND_CANDIDATE_STAGES_BEFORE_FAILURE'


def test_explicit_qp_limits_are_required_and_invalid_limits_rejected():
    with pytest.raises(TypeError):
        entry.probe_margin_joint(**synthetic(k=1))
    for resources in (
        dict(max_transitions=True,max_factor_buffer_bytes=8_000_000),
        dict(max_transitions=256,max_factor_buffer_bytes=0),
        dict(max_transitions=1.5,max_factor_buffer_bytes=8_000_000),
    ):
        with pytest.raises(ValueError):
            entry.probe_margin_joint(**synthetic(k=1),**resources)


def test_qp_peak_accounting_uses_max_and_work_accounting_uses_sum():
    peaks=entry.PEAK_COUNTERS
    assert peaks
    keys=tuple(peaks)+('synthetic_actual_work',)
    total=dict.fromkeys(keys,0)
    entry._account(total,dict.fromkeys(keys,5),keys)
    entry._account(total,dict.fromkeys(keys,3),keys)
    assert all(total[key]==5 for key in peaks)
    assert total['synthetic_actual_work']==8


def test_failed_numeric_archive_retains_nonfinite_values_with_explicit_metadata(tmp_path):
    archive=entry.StateArchive(tmp_path)
    key=json.dumps(dict(state='FAILED_FIT',run_id='synthetic-run'))+'/state'
    values=np.array([np.nan,np.inf,1.],dtype=np.float64)
    with pytest.raises(ValueError,match='Nonfinite'):
        archive(key,dict(values=values))
    assert not archive.files
    ref=archive.failure(key,dict(values=values))
    assert ref['failed_numeric_state'] is True
    assert ref['arrays']['values']['all_finite'] is False
    assert ref['arrays']['values']['nonfinite_count']==2
    assert ref['array_summaries']['values']==dict(norm=None,minimum=None,maximum=None)
    with np.load(tmp_path/ref['path'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['values'],values)
    manifest=archive.finalize('INCOMPLETE')
    assert manifest['file_count']==1 and manifest['by_phase']['FAILED_FIT']['file_count']==1


def test_raw_qp_failure_audit_and_state_are_retained_separately(tmp_path):
    from cvsrffi.d92_margin_qp_head import MarginQPFailure
    original=entry.fit_margin_joint_local_ridge
    def fail_C(prepared,**kwargs):
        if kwargs['mode']=='C_seq':
            raise MarginQPFailure('TRANSITION_LIMIT','synthetic raw QP failure',
                dict(status='TECHNICAL_FAILURE',factorization_attempts=2,transitions=2),
                dict(rho=np.asarray(.5),V=np.array([[1.,-1.]])))
        return original(prepared,**kwargs)
    out=tmp_path/'raw_failure'
    with patch.object(entry,'fit_margin_joint_local_ridge',side_effect=fail_C),pytest.raises(MarginQPFailure):
        production(out,synthetic(k=1))
    failure=json.loads((out/'probe_failed.json').read_text(encoding='utf-8'))
    evidence=failure['failure_context']
    assert evidence['failed_fit']['factorization_attempts']==2
    assert evidence['failed_fit']['transitions']==2
    assert evidence['failed_fit_audit_unavailable_reason'] is None
    assert evidence['workload_complete'] is False
    ref=evidence['failed_numeric_state_ref']
    with np.load(out/ref['path'],allow_pickle=False) as arrays:
        assert float(arrays['rho'])==.5
        np.testing.assert_array_equal(arrays['V'],[[1.,-1.]])
    assert ref['failed_numeric_state'] is True
    assert not (out/'probe_complete.json').exists()
