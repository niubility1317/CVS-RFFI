"""Synthetic complete callbacks, CE-only diagnostics and data-only transport."""
import ast
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import collect_d92_margin_joint_training_diagnostics as collect


def assert_native_json_scalars(value,path='$'):
    if isinstance(value,dict):
        for key,item in value.items():
            assert type(key) is str,(path,type(key))
            assert_native_json_scalars(item,path+'.'+key)
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):assert_native_json_scalars(item,path+'['+str(index)+']')
    else:
        assert value is None or type(value) in (str,bool,int,float),(path,type(value))


def summary_for(out,marker,root):
    root.mkdir()
    value=dict(status=collect.INPUT_STATUS,summary_schema='d92_margin_joint_support_summary_v1',schema=collect.SCHEMA,
        method=collect.METHOD,scope=collect.SCOPE,run_id=marker['run_id'],release_commit='a'*40,
        algorithm=marker['algorithm'],qp_resources=marker['qp_resources'],coverage=marker,
        resources=dict(actual_counters={key:marker.get(key) for key in
            dict.fromkeys(collect.PREPARATION_WORK_KEYS+collect.STAGE_WORK_KEYS+collect.PEAK_KEYS)}),
        training_stage_count=marker['candidate_stage_count'],
        raw_training_sources=[dict(row_id=marker['row_id'],compact_training_events=str(out/'training_events_compact.jsonl'))],
        query_rows_used=0,source_rows_used=0,statistics={'forbidden_outer_payload':'DO_NOT_DESERIALIZE'})
    (root/'summary.json').write_text(json.dumps(value),encoding='utf-8')
    return root


@pytest.mark.parametrize('k',[2,3])
def test_full_production_training_events_and_all_archives(tmp_path,k):
    from test_evaluate_d92_margin_joint_probe import production,synthetic
    out=tmp_path/'probe';marker=production(out,synthetic(k=k))
    root=summary_for(out,marker,tmp_path/'summary');snapshot=collect.snapshot(root)
    assert 'statistics' not in snapshot['metadata']
    assert all(json.loads(r['namespace'])['state'] in collect.STATES for lane in snapshot['lanes'] for r in lane['training_refs'])
    value=collect.extract(snapshot)
    assert_native_json_scalars(value)
    assert len(value['stages'])==marker['candidate_stage_count']
    assert len(value['curves'])==sum(len(l['event_index']) for l in snapshot['lanes'])
    assert len(value['archives'])==sum(len(l['training_refs']) for l in snapshot['lanes'])
    assert value['proximal_coefficient']==0 and value['objective']=='RMSCE_only'
    assert value['qp_resources']==marker['qp_resources']
    assert value['resources']['known_workload_complete']
    for key in collect.PEAK_KEYS:
        assert value['totals'][key]==max(row['audit'][key] for row in value['stages'])==marker[key]
    assert value['totals']['margin_qp_forward_full_constraint_scans']==marker['margin_qp_forward_full_constraint_scans']
    gradients=[r for r in value['curves'] if r['event']=='MARGIN_JOINT_GRADIENT']
    assert bool(gradients)==(k==3)
    assert all(r['metrics']['proximal_gradient_norm']==0 for r in gradients)
    destination=tmp_path/'derived';collect.write_outputs(destination,value)
    assert len((destination/'curves.jsonl').read_text(encoding='utf-8').splitlines())==len(value['curves'])
    with pytest.raises(FileExistsError):collect.write_outputs(destination,value)
    broken=deepcopy(snapshot);broken['lanes'][0]['event_index'].pop()
    with pytest.raises(ValueError,match='stream changed'):collect.extract(broken)
    # A failure marker is never overwritten or reinterpreted as complete.
    failure=out/'probe_failed.json';failure.write_text('original failure',encoding='utf-8')
    with pytest.raises(ValueError,match='failure'):collect.extract(snapshot)
    assert failure.read_text(encoding='utf-8')=='original failure'


def test_ce_gradient_does_not_subtract_Z_and_armijo_uses_projected_delta():
    arrays={'g':dict(g_Z=np.asarray([[-2.]]),Z=np.asarray([[.49]]),d_Z=np.asarray([[1.]]),
        class_ce_sums=np.asarray([0.,np.sqrt(2.)]),class_ce_counts=np.asarray([1,1])),
        't':dict(Z=np.asarray([[.5]]),delta_Z=np.asarray([[.01]]),d_Z=np.asarray([[1.]]),
        class_ce_sums=np.asarray([.99995,.99995]),class_ce_counts=np.asarray([1,1]))}
    gradient=collect.gradient_metrics(dict(state_ref='g'),arrays.__getitem__)
    assert gradient['CE_gradient_norm']==2 and gradient['proximal_gradient_norm']==0
    event=dict(state_ref='t',gradient_state_ref='g',loss_before=1.,loss_after=.99995,accepted=True,step_size=.125,
        objective=dict(RMSCE=.99995,loss_ce=.99995,loss_total=.99995))
    result=collect.trial_metrics(event,arrays.__getitem__)
    assert_native_json_scalars(result)
    assert type(result['comparison_tolerance']) is float
    assert type(result['accepted_objective_nonincrease']) is bool
    assert type(result['accepted_mean_CE_nonincrease']) is bool
    assert result['actual_delta_gradient_dot']==pytest.approx(-.02)
    assert result['actual_projected_delta_norm']==pytest.approx(.01)
    assert result['raw_step_direction_norm']==.125
    assert result['accepted_objective_nonincrease'] and not result['accepted_mean_CE_nonincrease']
    assert result['mean_CE_change']==pytest.approx(.99995-np.sqrt(.5)) and result['recorded_objective_minus_RMSCE']==0
    arrays['t'].pop('class_ce_counts')
    assert collect.trial_metrics(event,arrays.__getitem__)['mean_CE_after'] is None
    with pytest.raises(ValueError,match='Armijo'):collect.trial_metrics(dict(event,accepted=False),arrays.__getitem__)
    arrays['t']['Z']=np.asarray([[.6]])
    with pytest.raises(ValueError):collect.trial_metrics(event,arrays.__getitem__)


def test_metadata_and_training_namespace_reject_outer_query_and_lineage(tmp_path):
    ref=dict(path='state_arrays/0.npz',namespace=json.dumps(dict(run_id='run',row_id='row',state='B_MARGIN',scope='support_oof')))
    collect.training_reference(ref,'run','row')
    for namespace in (dict(run_id='other',row_id='row',state='B_MARGIN',scope='support_oof'),
        dict(run_id='run',row_id='row',state='OUTER_SUPPORT_HELD',scope='support_oof')):
        with pytest.raises(ValueError):collect.training_reference(dict(ref,namespace=json.dumps(namespace)),'run','row')
    with pytest.raises(ValueError):collect.training_reference(dict(ref,path='../outside.npz'),'run','row')
    assert not collect.event_key('outer_scores') and not collect.event_key('query_truth') and collect.event_key('outer_trial')


def decoded_request(source):
    tree=ast.parse(source);assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SNAPSHOT_REQUEST' for t in n.targets))
    scope=dict(base64=base64,gzip=gzip,json=json)
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'synthetic-transport','exec'),scope)
    return scope['SNAPSHOT_REQUEST']


def test_data_only_transport_unicode_deep_large_and_nonfinite_rejection():
    nested={'unicode':'条件核 λ 中文'}
    for _ in range(80):nested={'child':nested}
    value=dict(nested=nested,payload=random.Random(301).randbytes(1600000).hex())
    encoded=gzip.compress(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'),mtime=0)
    assert len(encoded)>1024**2
    source=collect.transport_source(value)
    assert decoded_request(source)==value and 'gzip.decompress' in source
    assert collect.transport_source(value)==source
    for bad in (float('nan'),float('inf'),-float('inf')):
        with pytest.raises(ValueError):collect.transport_source(dict(bad=bad))
    corrupt='SNAPSHOT_REQUEST=json.loads(gzip.decompress(base64.b64decode('+repr(base64.b64encode(b'bad gzip').decode())+')).decode("utf-8"))'
    with pytest.raises(gzip.BadGzipFile):decoded_request(corrupt)


def test_frozen_counter_contract_matches_core_without_importing_core_in_transport():
    from cvsrffi import d92_margin_joint_local_ridge as core
    assert collect.PREPARATION_WORK_KEYS==core.PREPARATION_COUNTERS
    assert collect.STAGE_WORK_KEYS==core.STAGE_COUNTERS
    assert collect.QP_WORK_KEYS==core.QP_WORK_KEYS


def stage_work():
    audit={key:0 for key in collect.STAGE_WORK_KEYS+collect.PEAK_KEYS}
    audit.update(status='COMPLETED',inner_factorization_count=3,final_factorization_count=2,completed_factorization_count=5)
    audit.update({'margin_qp_forward_'+key:value for key,value in dict(
        factorization_attempts=5,factorizations_completed=5,condition_estimation_calls=5,
        spectral_checks=1,transitions=7,full_constraint_scans=9,triangular_calls=14,
        triangular_rhs_columns=28,triangular_rhs_elements=84,triangular_dense_work_units=252).items()})
    audit[collect.PEAK_KEYS[0]]=100;audit[collect.PEAK_KEYS[1]]=20
    return audit


def test_work_ledger_uses_dynamic_qp_factors_scans_and_rhs_not_conditional_formulas():
    audit=stage_work()
    assert collect.validate_stage_work(audit)==[]
    for key,value in [('margin_qp_adjoint_factorization_attempts',1),
        ('margin_qp_forward_triangular_rhs_elements',99),('margin_qp_forward_transitions',-1),
        ('completed_factorization_count',6)]:
        with pytest.raises(ValueError):collect.validate_stage_work(dict(audit,**{key:value}))


def test_sum_and_peak_max_keep_unknown_work_null_without_zero_imputation():
    first=stage_work();second=stage_work();second[collect.PEAK_KEYS[0]]=70
    prep={key:0 for key in collect.PREPARATION_WORK_KEYS};prep['prior_factorization_count']=3
    totals,missing=collect.aggregate_work([dict(audit=prep)],[dict(audit=first),dict(audit=second)])
    assert totals[collect.PEAK_KEYS[0]]==100 and totals[collect.PEAK_KEYS[1]]==20
    assert totals['margin_qp_forward_factorization_attempts']==10 and totals['prior_factorization_count']==3
    assert not any(missing.values())
    del second['margin_qp_forward_transitions'];second[collect.PEAK_KEYS[0]]=None
    assert set(collect.validate_stage_work(second))=={'margin_qp_forward_transitions',collect.PEAK_KEYS[0]}
    totals,missing=collect.aggregate_work([dict(audit=prep)],[dict(audit=first),dict(audit=second)])
    assert totals['margin_qp_forward_transitions'] is None and totals[collect.PEAK_KEYS[0]] is None
    assert missing['margin_qp_forward_transitions']==missing[collect.PEAK_KEYS[0]]==1
