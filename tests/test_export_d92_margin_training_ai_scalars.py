"""Literal scalar diagnostic fixtures only; no model, real run or numeric archive."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import export_d92_margin_training_ai_scalars as exporter


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def save_lines(path,values):
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')


def fixture(tmp_path):
    root=tmp_path/'diagnostics';root.mkdir();streams=dict(stages=[],curves=[],preparations=[])
    for state,prep in [('B_MARGIN','B_prepare'),('C_MARGIN_seq','C_prepare')]:
        coords=dict(run_id='synthetic-run',row_id='model-模型',split_id='literal-parent',scope='support_oof',
            fold=0,outer_trial=None,parent_k=5,train_k=3,state=state)
        audit=dict(status='COMPLETED',fit_seconds=.125,optimizer_steps=1,trainable_parameter_count=8,
            gradient_norm=.25,source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
            unknown_measured_seconds=None,final_fit={'arrays':[1,2,3]},gradients=[{'huge':'not scalars'}],
            state_ref={'path':'forbidden.npz'},config_json='{"not":"categorical"}')
        streams['stages'].append(dict(coords,mode='B' if state=='B_MARGIN' else 'C_seq',audit=audit,
            actual_gradient_records=1,actual_trial_records=2,actual_step_records=1,
            initial_objective=dict(RMSCE=1.,loss_total=1.,weight=.25),
            final_objective=dict(RMSCE=.9,loss_total=.9,prox=0.,class_ce_sums=[.8,1.])))
        prepared=dict(coords,state=prep)
        streams['preparations'].append(dict(prepared,audit=dict(ajlr_preparation_count=1,preparation_seconds=.02,
            prepared_state_ref={'path':'never-open.npz'},text='do not export')))
        for event in ('PREPARED','INITIAL','GRADIENT','TRIAL','TRIAL','STEP','FINAL'):
            row=dict(prepared if event=='PREPARED' else coords,event='MARGIN_JOINT_'+event,
                iteration=1 if event in ('GRADIENT','TRIAL','STEP') else None,
                measured_scalars=dict(learning_rate=None,step_size=.125,elapsed_seconds=.01,method_active=True),
                objective=dict(RMSCE=.9,loss_ce=.9,loss_total=.9,prox=0.,class_ce_means=[.8,1.]),metrics={})
            if event=='GRADIENT':row['metrics']=dict(CE_gradient_norm=.25,proximal_gradient_norm=0.)
            if event=='TRIAL':
                n=sum(r.get('event')=='MARGIN_JOINT_TRIAL' and r['state']==state for r in streams['curves'])
                row['trial']=n
                row['metrics']=dict(accepted=n==0,objective_before=1.,objective_after=.9 if n==0 else 1.2,
                    recorded_objective_minus_RMSCE=0.,mean_CE_before=.7,mean_CE_after=.8,
                    actual_projected_delta_norm=.02)
                if state=='C_MARGIN_seq':row['metrics']['mean_CE_before']=None
            streams['curves'].append(row)
    work={p:None if 'peak_explicit' in p else 0 for p in exporter.PEAKS}
    work.update({'margin_qp_'+op+'_'+key:0 for op in ('forward','adjoint') for key in exporter.QP_WORK})
    meta=dict(status=exporter.INPUT_STATUS,schema=exporter.METHOD_SCHEMA,method=exporter.METHOD,scope=exporter.SCOPE,
        run_id='synthetic-run',release_commit='a'*40,source_summary='/not-opened/source-summary.json',
        objective='RMSCE_only',proximal_coefficient=0.,coordinate_ball_radius=.5,
        qp_resources=dict(max_transitions=12,max_factor_buffer_bytes=8192),
        resources=dict(actual_counters=work,peak_aggregation='MAX',work_aggregation='SUM',scope='preparations_plus_stages'),
        totals=dict(actual_stages=2,curve_records=14,preparation_records=2))
    save(root/'summary.json',meta)
    for key,values in streams.items():save_lines(root/(key+'.jsonl'),values)
    return root,meta,streams


def test_complete_streaming_export_preserves_measured_nested_scalars_and_originals(tmp_path,monkeypatch):
    root,meta,streams=fixture(tmp_path);before={p:p.read_bytes() for p in root.iterdir()};out=tmp_path/'export'
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.npz' and str(path)!='/not-opened/source-summary.json'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=exporter.export_scalars(diagnostics_root=root,output=out)
    assert result['counts']==dict(stages=2,curves=14,preparations=2) and result['readback_verified']
    assert result['qp_resources']==meta['qp_resources'] and result['release_commit']==meta['release_commit']
    assert result['source_actual_counters'][exporter.PEAKS[1]] is None
    assert result['conversion_training_operations']==result['conversion_model_updates']==result['npz_files_read']==0
    assert result['accepted_increase_observations']['RMSCE']==dict(true=0,false=4,unknown=0)
    assert result['accepted_increase_observations']['mean_CE']==dict(true=1,false=2,unknown=1)
    stage=next(exporter.records(out/'stages_compact.jsonl'))
    assert stage['audit__fit_seconds']==.125 and stage['audit__gradient_norm']==.25
    assert stage['audit__trainable_parameter_count']==8 and stage['audit__optimizer_steps']==1
    assert stage['initial_objective__weight']==.25 and stage['final_objective__RMSCE']==.9
    assert stage['audit__unknown_measured_seconds'] is None and stage['source_validation'] is None
    assert stage['row_id']=='model-模型' and not any('epoch' in key for key in stage)
    assert result['field_mapping']['stages']['audit__fit_seconds']=='$.audit.fit_seconds'
    for kind in exporter.STREAMS:
        for row in exporter.records(out/(kind+'_compact.jsonl')):
            assert list(row)==sorted(row)
            assert all(v is None or type(v) in (str,bool,int,float) for v in row.values())
        assert result['output_file_bytes'][kind+'_compact.csv']==(out/(kind+'_compact.csv')).stat().st_size
    assert all(p.read_bytes()==value for p,value in before.items())
    monkeypatch.setattr(exporter,'read_json',lambda *args:pytest.fail('Exclusive output checked too late'))
    with pytest.raises(FileExistsError):exporter.export_scalars(diagnostics_root=root,output=out)


def test_large_nested_and_json_string_payloads_never_masquerade_as_scalars(tmp_path):
    root,_,streams=fixture(tmp_path);huge='载荷'*600000
    row=streams['stages'][0];row.update(text=huge,records={'x':huge},scores=[huge],unknown_category=huge)
    row['audit'].update(text=huge,state_ref={'data':huge},arbitrary='{"array":'+json.dumps(huge)+'}',
        stop_reason='{"leak":1}',device_model='x'*161,gradients=huge)
    save_lines(root/'stages.jsonl',streams['stages'])
    assert (root/'stages.jsonl').stat().st_size>1024**2
    out=tmp_path/'export';exporter.export_scalars(diagnostics_root=root,output=out)
    text=(out/'stages_compact.jsonl').read_text(encoding='utf-8')
    assert len(text)<10000 and '载荷' not in text and 'leak' not in text
    assert all(not any(term in key for term in ('state_ref','gradients','text','config_json')) for key in json.loads(text.splitlines()[0]))


@pytest.mark.parametrize('case',['status','run','commit','qp_missing','qp_bool','qp_zero','qp_float',
    'work_missing','work_negative','work_bool','aggregation','counts','missing_phase','duplicate_stage',
    'missing_preparation','wrong_run','source_validation','failed_marker'])
def test_invalid_complete_binding_resource_and_phase_contract_rejected(tmp_path,case):
    root,meta,streams=fixture(tmp_path)
    if case=='status':meta['status']='PARTIAL'
    elif case=='run':meta['run_id']=''
    elif case=='commit':meta['release_commit']=None
    elif case=='qp_missing':meta['qp_resources'].pop('max_transitions')
    elif case=='qp_bool':meta['qp_resources']['max_transitions']=True
    elif case=='qp_zero':meta['qp_resources']['max_transitions']=0
    elif case=='qp_float':meta['qp_resources']['max_transitions']=12.
    elif case=='work_missing':meta['resources']['actual_counters'].pop(exporter.PEAKS[0])
    elif case=='work_negative':meta['resources']['actual_counters'][exporter.PEAKS[0]]=-1
    elif case=='work_bool':meta['resources']['actual_counters'][exporter.PEAKS[0]]=False
    elif case=='aggregation':meta['resources']['peak_aggregation']='SUM'
    elif case=='counts':meta['totals']['curve_records']+=1
    elif case=='missing_phase':
        streams['curves']=[r for r in streams['curves'] if not(r['event']=='MARGIN_JOINT_INITIAL' and r['state']=='B_MARGIN')]
        meta['totals']['curve_records']-=1
    elif case=='duplicate_stage':streams['stages'][1]=deepcopy(streams['stages'][0])
    elif case=='missing_preparation':streams['preparations'][1]['state']='B_prepare'
    elif case=='wrong_run':streams['stages'][0]['run_id']='other'
    elif case=='source_validation':streams['stages'][0]['audit']['source_validation']=.9
    else:save(root/'failed.json',{'error':'original preserved'})
    save(root/'summary.json',meta)
    for kind,values in streams.items():save_lines(root/(kind+'.jsonl'),values)
    with pytest.raises(ValueError):exporter.export_scalars(diagnostics_root=root,output=tmp_path/'export')
    assert not (tmp_path/'export').exists()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_is_rejected_and_native_bool_and_null_are_preserved(tmp_path,value):
    root,_,streams=fixture(tmp_path);streams['curves'][0]['metrics']['measured']=value
    save_lines(root/'curves.jsonl',streams['curves'])
    with pytest.raises(ValueError,match='Nonfinite'):exporter.export_scalars(diagnostics_root=root,output=tmp_path/'export')
    compact=exporter.compact(dict(metrics=dict(flag=True,unknown=None),event='MARGIN_JOINT_INITIAL'),'curves')
    assert compact['metrics__flag'] is True and compact['metrics__unknown'] is None
    assert json.loads(json.dumps(compact,allow_nan=False))==compact


def test_missing_mean_ce_or_rmsce_equivalence_is_not_imputed():
    row=dict(event='MARGIN_JOINT_TRIAL',metrics=dict(accepted=True,objective_before=1.,objective_after=.9,
        mean_CE_before=None,mean_CE_after=.8))
    compact=exporter.compact(row,'curves')
    assert compact['accepted_RMSCE_increase'] is compact['accepted_mean_CE_increase'] is None
    assert compact['accepted_RMSCE_increase_reason']=='RMSCE_OBJECTIVE_EQUALITY_NOT_RECORDED'
    assert compact['accepted_mean_CE_increase_reason']=='EXACT_MEASURED_BEFORE_AFTER_UNAVAILABLE'


def test_readback_corruption_preserves_failed_output_without_success(tmp_path,monkeypatch):
    root,_,_=fixture(tmp_path);original=exporter.verify_outputs
    def corrupt(out,inspection):
        path=out/'stages_compact.csv';text=path.read_text(encoding='utf-8');path.write_text(text.replace('0.125','999',1),encoding='utf-8')
        original(out,inspection)
    monkeypatch.setattr(exporter,'verify_outputs',corrupt)
    out=tmp_path/'export'
    with pytest.raises(ValueError,match='value mismatch'):exporter.export_scalars(diagnostics_root=root,output=out)
    assert (out/'failed.json').exists() and not (out/'summary.json').exists()
    with pytest.raises(FileExistsError):exporter.export_scalars(diagnostics_root=root,output=out)


def test_output_cannot_be_created_inside_original_diagnostics(tmp_path):
    root,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='outside'):exporter.export_scalars(diagnostics_root=root,output=root/'new')
