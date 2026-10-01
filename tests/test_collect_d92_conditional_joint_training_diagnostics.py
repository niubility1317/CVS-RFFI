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
import collect_d92_conditional_joint_training_diagnostics as collect


def summary_for(out,marker,root):
    root.mkdir()
    value=dict(status=collect.INPUT_STATUS,summary_schema='d92_conditional_joint_support_summary_v1',schema=collect.SCHEMA,
        method=collect.METHOD,scope=collect.SCOPE,run_id=marker['run_id'],release_commit='a'*40,
        algorithm=marker['algorithm'],coverage=marker,training_stage_count=marker['candidate_stage_count'],
        raw_training_sources=[dict(row_id=marker['row_id'],compact_training_events=str(out/'training_events_compact.jsonl'))],
        query_rows_used=0,source_rows_used=0,statistics={'forbidden_outer_payload':'DO_NOT_DESERIALIZE'})
    (root/'summary.json').write_text(json.dumps(value),encoding='utf-8')
    return root


@pytest.mark.parametrize('k',[2,3])
def test_full_production_training_events_and_all_archives(tmp_path,k):
    from test_evaluate_d92_conditional_joint_probe import production,synthetic
    out=tmp_path/'probe';marker=production(out,synthetic(k=k))
    root=summary_for(out,marker,tmp_path/'summary');snapshot=collect.snapshot(root)
    assert 'statistics' not in snapshot['metadata']
    assert all(json.loads(r['namespace'])['state'] in collect.STATES for lane in snapshot['lanes'] for r in lane['training_refs'])
    value=collect.extract(snapshot)
    assert len(value['stages'])==marker['candidate_stage_count']
    assert len(value['curves'])==sum(len(l['event_index']) for l in snapshot['lanes'])
    assert len(value['archives'])==sum(len(l['training_refs']) for l in snapshot['lanes'])
    assert value['proximal_coefficient']==0 and value['objective']=='RMSCE_only'
    gradients=[r for r in value['curves'] if r['event']=='CONDITIONAL_JOINT_GRADIENT']
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
    arrays={'g':dict(g_Z=np.asarray([[-2.]]),Z=np.asarray([[.49]]),d_Z=np.asarray([[1.]])),
        't':dict(Z=np.asarray([[.5]]),delta_Z=np.asarray([[.01]]),d_Z=np.asarray([[1.]]))}
    gradient=collect.gradient_metrics(dict(state_ref='g'),arrays.__getitem__)
    assert gradient['CE_gradient_norm']==2 and gradient['proximal_gradient_norm']==0
    event=dict(state_ref='t',gradient_state_ref='g',loss_before=1.,loss_after=.99995,accepted=True,step_size=.125)
    result=collect.trial_metrics(event,arrays.__getitem__)
    assert result['actual_delta_gradient_dot']==pytest.approx(-.02)
    assert result['actual_projected_delta_norm']==pytest.approx(.01)
    assert result['raw_step_direction_norm']==.125
    with pytest.raises(ValueError,match='Armijo'):collect.trial_metrics(dict(event,accepted=False),arrays.__getitem__)
    arrays['t']['Z']=np.asarray([[.6]])
    with pytest.raises(ValueError):collect.trial_metrics(event,arrays.__getitem__)


def test_metadata_and_training_namespace_reject_outer_query_and_lineage(tmp_path):
    ref=dict(path='state_arrays/0.npz',namespace=json.dumps(dict(run_id='run',row_id='row',state='B_CONDITIONAL',scope='support_oof')))
    collect.training_reference(ref,'run','row')
    for namespace in (dict(run_id='other',row_id='row',state='B_CONDITIONAL',scope='support_oof'),
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


def test_work_ledger_distinguishes_two_factors_spectra_and_rhs():
    audit={p+'_'+s:0 for p in collect.PREFIXES for s in collect.SUFFIXES}
    audit.update(status='COMPLETED',inner_factorization_count=0,final_factorization_count=2,spectral_diagnostic_count=3)
    for p in ('projection','residual'):
        audit.update({p+'_factorization_count':1,p+'_triangular_solve_count':2,p+'_triangular_rhs_count':4,
            p+'_triangular_rhs_element_count':12,p+'_triangular_dense_work_unit_count':36})
    collect.validate_stage_work(audit)
    for key,value in [('spectral_diagnostic_count',1),('projection_adjoint_factorization_count',1),('residual_triangular_rhs_element_count',99)]:
        with pytest.raises(ValueError):collect.validate_stage_work(dict(audit,**{key:value}))
