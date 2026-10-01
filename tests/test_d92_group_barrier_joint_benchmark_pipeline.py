"""Literal inputs, mock support solver and actual archive/native A boundaries."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_group_barrier_joint_benchmark as entry
import run_d92_group_barrier_joint_benchmark as run
import publish_d92_group_barrier_joint_benchmark as publish
from cvsrffi.d92_ground_classifier_a import GroundClassifierA,GroundHeadMetadata,GroundFeatureContract,WEIGHT_KEY


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def fixture(tmp_path,monkeypatch,*,new=1,k=2):
    old=['g'+str(i) for i in range(6)];classes=old+['n'+str(i) for i in range(new)]
    labels=np.repeat(np.arange(len(classes)),k);n=len(labels)
    ids=np.asarray(['support-'+str(i) for i in range(n)]+['opaque-z','opaque-a','opaque-q'])
    arrays={key:np.zeros((len(ids),96 if key=='fft' else 160),dtype=np.float32) for key in entry.BRANCHES}
    for i in range(len(ids)):
        arrays['z_id'][i,i%6]=1.;arrays['fft'][i,0]=i+.25
    capsule=tmp_path/'capsule';split=dict(split_id='literal-split',capsule_id='literal-capsule',
        protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',receiver='rx-literal',scenario='practical_high',
        k=k,support_seed=7,registered_classes=classes,support_indices=list(range(n)),query_indices=[n+1,n,n+2],support_labels=labels.tolist())
    save(capsule/'splits'/'literal-split.json',split)
    packet=tmp_path/'packet';save(packet/'complete.json',dict(packet_total_file_bytes=4567))
    save(packet/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=17)))
    (packet/'head_weight.float32.bin').write_bytes(bytes(6*160*4))
    native=[old[i] for i in (3,0,5,1,4,2)];weight=torch.zeros((6,160),dtype=torch.float32)
    for j,name in enumerate(native):weight[j,int(name[1:])]=1.
    meta=GroundHeadMetadata('a'*64,WEIGHT_KEY,native,9.5,1e-4,GroundFeatureContract('a'*64,'z_id','feat_joint','raw','float32',160),
        'MATCHED_SOURCE_ONLY_SCRATCH',False,(),'none')
    head=GroundClassifierA(weight=weight,metadata=meta)
    provenance=dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',source_role_comparison='EXACT_MATCH',checkpoint_epoch=200,
        checkpoint_inheritance=[],target_access_before_freeze=False)
    producer=dict(classes=old,feature_array_bytes=sum(a.nbytes for a in arrays.values()),feature_file_bytes=999,model_file_bytes=123)
    cache_calls=[]
    def load(**kwargs):cache_calls.append(kwargs);return arrays,ids,producer,dict(seed=17),provenance
    monkeypatch.setattr(entry,'load_features',load)
    monkeypatch.setattr(entry,'validate_capsule',lambda *a:dict(capsule_id='literal-capsule',split_count=1))
    monkeypatch.setattr(entry,'load_packet',lambda p:head)
    calls=[];states=[]
    def prepare(**kwargs):
        selected=tuple(str(v) for v in kwargs['support_ids'])
        assert set(selected)<=set(ids[:n]) and all('opaque' not in pid for pid in selected)
        assert set(kwargs)==set(entry.BRANCHES)|{'support_labels','support_ids','classes','old_classes','inherited','context',
            'max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes','log_callback','state_callback'}
        assert len(kwargs['support_labels'])==len(selected)
        for key in entry.BRANCHES:
            assert np.array_equal(kwargs[key],arrays[key][[list(ids).index(pid) for pid in selected]])
        inherited=kwargs['inherited'];mode='B' if inherited is None else 'C_seq'
        calls.append(('prepare',mode,selected))
        if inherited is not None:assert inherited is states[-1]
        ref=kwargs['state_callback']('prepared_coordinates',{'U':np.zeros((2,2),dtype=np.float64)})
        audit=dict.fromkeys(entry.core.PREPARATION_COUNTERS,0)
        audit.update(ajlr_preparation_count=1,prepared_state_ref=ref,
            final_problem=dict(prior_ref=None if inherited is None else inherited.audit['final_state_ref']))
        kwargs['log_callback'](dict(audit,event='MARGIN_JOINT_PREPARED',text='prepared synthetic'))
        return SimpleNamespace(audit_dict=lambda:deepcopy(audit),audit=audit,kwargs=kwargs)
    def fit(prepared,*,mode,log_callback,state_callback,**limits):
        assert limits==entry.RESOURCES
        calls.append(('fit',mode))
        ref=state_callback('initial',{'Z':np.zeros((2,2),dtype=np.float64)})
        log_callback(dict(event='MARGIN_JOINT_INITIAL',mode=mode,state_ref=ref,objective=dict(RMSCE=1.,loss_total=1.)))
        count=0 if k==1 else 1
        if count:
            for event in ('GRADIENT','TRIAL','STEP'):
                ref=state_callback(event.lower(),{'Z':np.zeros((2,2),dtype=np.float64)})
                log_callback(dict(event='MARGIN_JOINT_'+event,mode=mode,iteration=1,trial=0,step_size=.125,
                    gradient_norm=.25,accepted=True,state_ref=ref,objective=dict(RMSCE=.9,loss_total=.9)))
        final=state_callback('final',{'U':np.ones((2,2),dtype=np.float64),'alpha':np.ones((len(prepared.kwargs['support_ids']),len(prepared.kwargs['classes'])),dtype=np.float64)})
        audit=dict.fromkeys(entry.STAGE_COUNTERS,0)
        audit.update(status='COMPLETED',optimizer_steps=count,final_head_fit_count=1,final_factorization_count=1 if mode=='B' else 0,
            completed_factorization_count=2,final_state_ref=final,preparation=prepared.audit,
            group_gate_forward_factorization_attempts=2 if mode=='C_seq' else 0,
            group_gate_adjoint_factorization_attempts=1 if mode=='C_seq' else 0,
            new_ridge_factorization_count=1 if mode=='C_seq' else 0,
            group_gate_adjoint_wall_seconds=.25 if mode=='C_seq' else 0,
            gate_peak_factor_buffer_bytes=128 if mode=='C_seq' else 0,
            margin_qp_peak_factor_buffer_bytes=100 if mode=='B' else 80,margin_qp_peak_explicit_solve_temporary_bytes=20)
        class State:
            classes=tuple(sorted(prepared.kwargs['classes']))
            ids=tuple(sorted(str(v) for v in prepared.kwargs['support_ids']))
            def audit_dict(self):return deepcopy(self.audit)
            def _values(self,background):
                return np.arange(len(self.classes),dtype=np.float64)[None,:]+float(background[0,0])
            def score_with_audit(self,**features):
                assert set(features)==set(entry.BRANCHES) and all(a.shape[0]==1 for a in features.values())
                calls.append(('score',mode,float(features['fft'][0,0])))
                background,_=entry.core.interaction._blocks(**features,allow_empty=True)
                return self._values(background),dict(physical_records=1,measured_pair_work=len(self.ids))
            def _score_geometry(self,background,aux,work):
                assert len(background)==1
                calls.append(('C_geometry',mode,float(background[0,0])))
                B=self.prior._values(background)
                h=np.arange(len(self.classes)-6,dtype=np.float64)[None,:]
                g=np.asarray([1.])
                p=SimpleNamespace(classes=self.classes,audit=dict(old_classes=old))
                score,lo,ln=entry.core._compose(p,B,h,g)
                work.update(physical_records=1,measured_pair_work=len(self.ids))
                return score,B,h,g,lo,ln
            def predict(self,**features):
                background,aux=entry.core.interaction._blocks(**features,allow_empty=True)
                _,B,h,g,lo,ln=self._score_geometry(background,aux,{})
                bo,bn=int(np.argmax(B[0])),int(np.argmax(h[0]))
                ow,nw=old[bo],self.classes[self.final_cache['new_columns'][bn]]
                gap=g[0]+lo[0,bo]-ln[0,bn]
                return np.asarray([ow if gap>0 else nw if gap<0 else min(ow,nw)])
        state=State();state.audit=audit;state.prior=prepared.kwargs['inherited']
        state.final_cache=dict(old_columns=np.asarray([state.classes.index(c) for c in old]),new_columns=np.asarray([j for j,c in enumerate(state.classes) if c not in old]))
        states.append(state)
        log_callback(dict(event='MARGIN_JOINT_FINAL',mode=mode,state_ref=final,audit=audit))
        return state
    monkeypatch.setattr(entry.core,'prepare_group_barrier_joint_training',prepare)
    monkeypatch.setattr(entry.core,'fit_group_barrier_joint_local_ridge',fit)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Checkpoint load forbidden'))
    kwargs=dict(run_id='literal-benchmark',row_id='literal-row',release_commit='b'*40,row_root=str(tmp_path/'source-row'),
        capsule=str(capsule),output=str(tmp_path/'prediction'),config=dict(algorithm=deepcopy(entry.core.FROZEN_CONFIG),
        group_barrier_resources=deepcopy(entry.RESOURCES)),expected_capsule_id='literal-capsule',
        expected_checkpoint_sha256='a'*64,branch_features=str(tmp_path/'cache'),ground_packet=str(packet))
    return SimpleNamespace(kwargs=kwargs,arrays=arrays,ids=ids,split=split,head=head,calls=calls,states=states,cache_calls=cache_calls)


def test_preflight_reads_bindings_without_training_scores_or_output(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(GroundClassifierA,'score',lambda *a,**k:pytest.fail('Preflight inference forbidden'))
    value=entry.preflight(**case.kwargs)
    assert value['status']=='GROUP_BARRIER_QUERY_PREFLIGHT_COMPLETE' and value['model_seed']==17
    assert value['source_identity']['source_only_verdict']=='MATCHED_SOURCE_ONLY_SCRATCH'
    assert value['release_commit']=='b'*40 and value['ordered_ground_classes']==list(case.head.classes)
    assert case.calls==[] and not Path(case.kwargs['output']).exists()


@pytest.mark.parametrize('new,k',[(1,2),(0,2),(1,1),(0,1)])
def test_actual_archives_singleton_queries_same_split_B_C_and_truth_free_streams(tmp_path,monkeypatch,new,k):
    case=fixture(tmp_path,monkeypatch,new=new,k=k)
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:pytest.fail('Shared NumPy/Torch input ABI forbidden'))
    monkeypatch.setattr(torch.Tensor,'numpy',lambda *a,**kw:pytest.fail('Shared Torch/NumPy output ABI forbidden'))
    marker=entry.predict(**case.kwargs);out=Path(case.kwargs['output'])
    assert marker['schema']==entry.SCHEMA and marker['status']=='COMPLETE'
    startup=json.loads((out/'startup.json').read_text(encoding='utf-8'))
    for key in ('run_id','row_id','release_commit','source_identity','ground_packet_identity','algorithm','group_barrier_resources'):
        assert marker[key]==startup[key]
    assert marker['actual_stage_count']==(2 if new else 1)
    assert [c[:2] for c in case.calls if c[0] in ('prepare','fit')]==[('prepare','B'),('fit','B')]+([('prepare','C_seq'),('fit','C_seq')] if new else [])
    first_score=next(i for i,c in enumerate(case.calls) if c[0]=='score')
    assert all(c[0] not in ('fit','prepare') for c in case.calls[first_score:])
    meta=marker['splits'][0];assert meta['c_reuses_b'] is (not new)
    assert meta['c_inherited_from_b_state_ref']==meta['b_state_ref']
    if not new:assert meta['c_state_ref']==meta['b_state_ref'] and case.states[-1] is case.states[0]
    expected_ids=case.ids[case.split['query_indices']].tolist()
    for name in ('A','B','C'):
        records=lines(out/marker['streams'][name]['path'])
        assert [r['query_id'] for r in records]==expected_ids
        assert all(set(r)=={'split_id','query_id','classes','scores','prediction'} for r in records)
        assert marker['streams'][name]['record_count']==3 and marker['streams'][name]['file_bytes']==(out/marker['streams'][name]['path']).stat().st_size
        if name!='C':assert all(r['prediction']==r['classes'][int(np.argmax(r['scores']))] for r in records)
    assert (out/'predictions_C.jsonl').read_bytes()==(out/'predictions.jsonl').read_bytes()
    assert marker['resources']['query_score_calls']==dict(A=3,B=3,C=3 if new else 0)
    assert marker['resources']['actual_training_counters']['optimizer_steps']==(2 if new else 1)*(k>1)
    assert marker['resources']['actual_training_counters']['margin_qp_peak_factor_buffer_bytes']==100
    assert marker['resources']['actual_factorization_attempts']==(5 if new else 1)
    assert marker['resources']['actual_training_counters']['group_gate_adjoint_wall_seconds']==(.25 if new else 0)
    assert marker['resources']['C_public_predict_calls']==(3 if new else 0)
    work=lines(out/'query_score_work.jsonl')
    assert len(work)==3 and marker['query_score_work']['file_bytes']==(out/'query_score_work.jsonl').stat().st_size
    assert all(v['C_structural_decision']['prediction']==v['C_structural_prediction'] for v in work)
    assert marker['technical_query_chunk_size']==1
    assert marker['resources']['ground_weight_file_bytes']==marker['resources']['ground_weight_numeric_bytes']==3840
    assert marker['resources']['energy'] is marker['resources']['peak_gpu_memory_bytes'] is marker['resources']['incremental_transfer_bytes'] is None
    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='COMPLETE' and manifest['files']
    assert all(json.loads(r['namespace'])['scope']==entry.TRAIN_SCOPE for r in manifest['files'])
    assert all((out/r['path']).exists() for r in manifest['files'])
    events=lines(out/'training_events.jsonl');compact=lines(out/'training_events_compact.jsonl')
    assert len(events)==len(compact) and (out/'training_events_compact.csv').exists()
    assert not any('accuracy' in key for obj in (marker,startup) for key in obj)
    with pytest.raises(FileExistsError):entry.predict(**case.kwargs)


def test_native_A_oracle_and_query_permutation_do_not_change_individual_predictions(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch)
    case.arrays['z_id'][case.split['query_indices'][0]]=0  # Exact native six-column tie.
    first=entry.predict(**case.kwargs);out=Path(case.kwargs['output'])
    rows={name:{r['query_id']:r for r in lines(out/first['streams'][name]['path'])} for name in ('A','B','C')}
    assert rows['A'][str(case.ids[case.split['query_indices'][0]])]['prediction']==case.head.classes[0]
    for index in case.split['query_indices']:
        tensor=torch.tensor([case.arrays['z_id'][index].tolist()],dtype=torch.float32)
        actual=case.head.score(z_id=tensor,feature_contract=case.head.metadata.feature_contract).tolist()[0]
        assert rows['A'][str(case.ids[index])]['scores']==actual
    case.split['query_indices'].reverse();save(Path(case.kwargs['capsule'])/'splits'/'literal-split.json',case.split)
    second=entry.predict(**dict(case.kwargs,output=str(tmp_path/'second')))
    for name in rows:
        assert {r['query_id']:r for r in lines(tmp_path/'second'/second['streams'][name]['path'])}==rows[name]
    changed=case.split['query_indices'][0];case.arrays['fft'][changed,0]+=100.;case.arrays['z_id'][changed]+=3.
    third=entry.predict(**dict(case.kwargs,output=str(tmp_path/'third')))
    for name in rows:
        observed={r['query_id']:r for r in lines(tmp_path/'third'/third['streams'][name]['path'])}
        assert all(observed[pid]==row for pid,row in rows[name].items() if pid!=str(case.ids[changed]))


@pytest.mark.parametrize('case_name',['algorithm','extra_config','bool_limit','missing_limit','commit','packet_seed','query_truth'])
def test_reject_binding_and_config_without_fitting(tmp_path,monkeypatch,case_name):
    case=fixture(tmp_path,monkeypatch);args=deepcopy(case.kwargs)
    if case_name=='algorithm':args['config']['algorithm']['method']='different'
    elif case_name=='extra_config':args['config']['selection']={}
    elif case_name=='bool_limit':args['config']['group_barrier_resources']['max_newton_iterations']=True
    elif case_name=='missing_limit':args['config']['group_barrier_resources'].pop('max_factor_buffer_bytes')
    elif case_name=='commit':args['release_commit']=None
    elif case_name=='packet_seed':save(Path(args['ground_packet'])/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=99)))
    else:
        case.split['query_truth']=['forbidden'];save(Path(args['capsule'])/'splits'/'literal-split.json',case.split)
    with pytest.raises(ValueError):entry.predict(**args)
    assert case.calls==[] and not Path(args['output']).exists()


def test_partial_C_failure_keeps_actual_B_and_failed_numeric_state_without_retry(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch);original=entry.core.fit_group_barrier_joint_local_ridge;attempts=[]
    class Failed(RuntimeError):
        arrays={'Z':np.asarray([float('nan')],dtype=np.float64)}
        def audit_dict(self):return dict(failure_code='LITERAL_RESOURCE_LIMIT',group_gate_forward_factorization_attempts=7)
    def failing(prepared,**kwargs):
        attempts.append(kwargs['mode'])
        if kwargs['mode']=='C_seq':
            kwargs['state_callback']('failure',dict(Z=np.asarray([np.nan,np.inf])))
            raise Failed('literal failure') from ValueError('original numeric cause')
        return original(prepared,**kwargs)
    monkeypatch.setattr(entry.core,'fit_group_barrier_joint_local_ridge',failing)
    with pytest.raises(Failed):entry.predict(**case.kwargs)
    out=Path(case.kwargs['output']);failure=json.loads((out/'technical_failure.json').read_text(encoding='utf-8'))
    assert attempts==['B','C_seq'] and not (out/'predictions_complete.json').exists()
    assert len(failure['completed_stages'])==1 and failure['failed_work_audit']['group_gate_forward_factorization_attempts']==7
    assert failure['automatic_retry'] is False and failure['failed_state_ref']['failed_numeric_state']
    assert failure['original_cause']==dict(error_type='ValueError',error='original numeric cause')
    manifest=lines(out/'training_events_compact.jsonl')
    assert manifest  # Real successful B events retained before the failed C callback.
    assert all(not lines(out/('predictions_'+name+'.jsonl')) for name in ('A','B','C'))
    with np.load(out/failure['failed_state_ref']['path'],allow_pickle=False) as archive:assert np.isnan(archive['Z']).all()
    with pytest.raises(FileExistsError):entry.predict(**case.kwargs)


@pytest.mark.parametrize('failure_write_error',[False,True])
def test_callback_and_reporting_device_errors_preserve_primary_failure_and_cause(tmp_path,monkeypatch,capsys,failure_write_error):
    case=fixture(tmp_path,monkeypatch);original=entry.core.fit_group_barrier_joint_local_ridge
    class PrimaryFailure(RuntimeError):
        arrays={'Z':np.asarray([np.nan,np.inf],dtype=np.float64)}
        def audit_dict(self):return dict(failure_code='ORIGINAL_NUMERICAL_FAILURE')
    primary=PrimaryFailure('primary numerical failure');cause=ValueError('original numerical cause');callback_errors=[]
    def archive_device_error(*args,**kwargs):raise OSError('archive device unavailable')
    def failing(prepared,**kwargs):
        if kwargs['mode']=='B':return original(prepared,**kwargs)
        try:kwargs['state_callback']('failure',primary.arrays)
        except OSError as callback_error:callback_errors.append(callback_error)
        raise primary from cause
    monkeypatch.setattr(entry.StateArchive,'failure',archive_device_error)
    monkeypatch.setattr(entry.StateArchive,'finalize',archive_device_error)
    monkeypatch.setattr(entry.core,'fit_group_barrier_joint_local_ridge',failing)
    if failure_write_error:
        original_write=entry.write
        def failed_report_write(path,value):
            if Path(path).name=='technical_failure.json':raise OSError('failure report device unavailable')
            return original_write(path,value)
        monkeypatch.setattr(entry,'write',failed_report_write)
    with pytest.raises(PrimaryFailure) as caught:entry.predict(**case.kwargs)
    assert caught.value is primary and caught.value.__cause__ is cause and len(callback_errors)==1
    out=Path(case.kwargs['output']);stderr=capsys.readouterr().err
    failure=json.loads(stderr.split('TECHNICAL_FAILURE best-effort evidence: ')[-1]) if failure_write_error else run.read(out/'technical_failure.json')
    assert failure['error']=='primary numerical failure' and failure['original_cause']['error']=='original numerical cause'
    assert failure['failed_work_audit']['failure_code']=='ORIGINAL_NUMERICAL_FAILURE'
    assert {v['step'] for v in failure['reporting_failures']}=={'failed_numeric_archive','state_manifest_finalize'}|({'technical_failure_write'} if failure_write_error else set())
    assert not (out/'predictions_complete.json').exists()


def test_public_predict_failure_retains_returned_C_geometry_and_attempt_work(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch);original=entry.core.fit_group_barrier_joint_local_ridge
    primary=RuntimeError('public structural decision failed')
    def failing_fit(prepared,**kwargs):
        state=original(prepared,**kwargs)
        if kwargs['mode']=='C_seq':
            def failed_predict(**features):raise primary
            state.predict=failed_predict
        return state
    monkeypatch.setattr(entry.core,'fit_group_barrier_joint_local_ridge',failing_fit)
    with pytest.raises(RuntimeError) as caught:entry.predict(**case.kwargs)
    assert caught.value is primary
    out=Path(case.kwargs['output']);failure=run.read(out/'technical_failure.json');work=failure['active_completed_query_work']
    assert failure['completed_query_score_calls']==dict(A=1,B=1,C=1)
    assert work['C_work']['physical_records']==1 and work['C_structural_decision']['schema']==entry.DECISION_SCHEMA
    assert work['C_public_predict_attempted'] is True and work['C_public_predict_completed'] is False
    assert work['C_public_predict_internal_work'] is None and work['C_public_predict_error']['error']==str(primary)
    assert work['C_public_predict_seconds']>=0 and failure['attempted_C_public_predict_calls']==1
    assert failure['completed_C_public_predict_calls']==0 and failure['attempted_C_public_predict_seconds']>=0
    assert not (out/'predictions_complete.json').exists()


def test_production_core_K3_B_to_C_callbacks_archives_and_query_completion(tmp_path,monkeypatch):
    """One predetermined synthetic case; no seed search or production data.

    Only input/cache/packet I/O is mocked. Restore the actual public core
    functions before calling the predictor, including its real state scorer.
    The resource limits are literal test capacity, not an experiment budget.
    """
    real_prepare=entry.core.prepare_group_barrier_joint_training
    real_fit=entry.core.fit_group_barrier_joint_local_ridge
    case=fixture(tmp_path,monkeypatch,new=1,k=3)
    monkeypatch.setattr(entry.core,'prepare_group_barrier_joint_training',real_prepare)
    monkeypatch.setattr(entry.core,'fit_group_barrier_joint_local_ridge',real_fit)
    rng=np.random.default_rng(341)
    labels=np.asarray(case.split['support_labels'])
    for name in entry.BRANCHES:
        values=rng.normal(size=case.arrays[name].shape)
        values[:len(labels),0]+=labels*.4
        values[:len(labels),1]+=(labels==0)*.6
        case.arrays[name][:]=values.astype(np.float32)

    marker=entry.predict(**case.kwargs)
    out=Path(case.kwargs['output'])
    assert marker['status']=='COMPLETE' and marker['actual_stage_count']==2
    assert case.calls==[] and case.states==[]  # Neither mocked solver was invoked.
    split=marker['splits'][0]
    assert split['c_reuses_b'] is False
    assert split['b_state_ref']!=split['c_state_ref']
    assert split['c_inherited_from_b_state_ref']==split['b_state_ref']
    stages=lines(out/'fit_stages.jsonl')
    preparations=lines(out/'preparations.jsonl')
    assert [s['state'] for s in stages]==['B_MARGIN','C_GROUP_BARRIER_seq']
    assert [p['state'] for p in preparations]==['B_prepare','C_prepare']
    assert preparations[1]['audit']['final_problem']['prior_ref']==split['b_state_ref']
    assert stages[1]['audit']['preparation']['final_problem']['prior_ref']==split['b_state_ref']
    assert stages[0]['training_physical_ids']==sorted(case.ids[:18].tolist())
    assert stages[1]['training_physical_ids']==sorted(case.ids[:21].tolist())
    assert all(s['audit']['optimizer_steps']>=0 for s in stages)  # Fixed synthetic fixture tests completion, not performance.

    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='COMPLETE'
    by_path={ref['path']:ref for ref in manifest['files']}
    for name,ref in [('B_MARGIN',split['b_state_ref']),('C_GROUP_BARRIER_seq',split['c_state_ref'])]:
        saved=by_path[ref['path']]
        assert saved['key']==ref['key'] and saved['namespace']==ref['namespace']
        namespace=json.loads(ref['namespace'])
        assert namespace['state']==name and namespace['scope']==entry.TRAIN_SCOPE
        assert namespace['split_id']==case.split['split_id']
        assert namespace['fold'] is None and namespace['trial'] is None
        with np.load(out/ref['path'],allow_pickle=False) as archive:
            assert {'U','Z'}<=set(archive.files)
            assert ('alpha' if name=='B_MARGIN' else 'new_alpha') in archive.files
            assert all(np.isfinite(archive[key]).all() for key in archive.files)
            for key,meta in ref['arrays'].items():
                assert list(archive[key].shape)==meta['shape']
                assert str(archive[key].dtype)==meta['dtype']
    events=lines(out/'training_events.jsonl')
    compact=lines(out/'training_events_compact.jsonl')
    assert len(events)==len(compact)>6
    for state in ('B_MARGIN','C_GROUP_BARRIER_seq'):
        actual={event['event'] for event in events if event['state']==state}
        prefix='MARGIN_JOINT_' if state=='B_MARGIN' else 'GROUP_BARRIER_'
        assert {prefix+k for k in ('INITIAL','GRADIENT','TRIAL','STEP','FINAL')}<=actual
    assert all(value is None or type(value) in (str,int,float,bool)
               for event in compact for value in event.values())
    assert (out/'training_events_compact.csv').exists()
    assert marker['resources']['query_score_calls']==dict(A=3,B=3,C=3)
    assert len(lines(out/'query_score_work.jsonl'))==3
    query_ids=case.ids[case.split['query_indices']].tolist()
    for name,count in [('A',6),('B',6),('C',7)]:
        records=lines(out/marker['streams'][name]['path'])
        assert [record['query_id'] for record in records]==query_ids
        assert all(len(record['classes'])==len(record['scores'])==count for record in records)
        assert all(np.isfinite(record['scores']).all() for record in records)
        if name!='C':assert all(record['prediction']==record['classes'][int(np.argmax(record['scores']))] for record in records)
    assert (out/'predictions_C.jsonl').read_bytes()==(out/'predictions.jsonl').read_bytes()
    assert marker['query_fit_access'] is marker['source_fit_access'] is marker['truth_read'] is False


def source_spec():return run.read(ROOT/'configs/d92_group_barrier_joint_repeat_20261001.json')


def full_preflight(s,row,commit='b'*40):
    args=run.predict_arguments(s,row,commit);co=s['benchmark']['cohorts'][row['cohort']];matrix=co['matrix'];splits=[]
    old=['g'+str(i) for i in range(6)]
    for i,(rx,scene,k,new,seed) in enumerate(itertools.product(*(matrix[n] for n in ('receivers','scenarios','ks','new_counts','support_seeds')))):
        sid=row['cohort']+'-physical-split-'+str(i);registered=sorted(old+['n'+str(j) for j in range(new)])
        oi=[sid+'-old-'+str(j) for j in range(6*k)];ni=[sid+'-new-'+str(j) for j in range(new*k)]
        splits.append(dict(split_id=sid,receiver=rx,scenario=scene,k=k,new_count=new,support_seed=seed,
            old_classes=old,registered_classes=registered,declared_registered_classes=registered,
            support_ids=sorted(oi+ni),old_support_ids=sorted(oi),new_support_ids=sorted(ni),query_ids=[sid+'-opaque'],query_count=1))
    return dict(status='GROUP_BARRIER_QUERY_PREFLIGHT_COMPLETE',schema=entry.SCHEMA,method=entry.METHOD,
        run_id=s['run_id'],row_id=row['row_id'],release_commit=commit,capsule_id=args['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],algorithm=args['config']['algorithm'],
        group_barrier_resources=args['config']['group_barrier_resources'],run_binding=dict(prediction_output_root=args['output'],
            **{k:args[k] for k in ('row_root','capsule','branch_features','ground_packet')}),
        source_identity=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed']),
        ground_packet_identity=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed']),
        ordered_ground_classes=list(reversed(old)),old_classes=old,splits=splits,split_count=len(splits),query_record_count=len(splits),
        query_fit_access=False,source_fit_access=False,truth_read=False,checkpoint_loaded=False,encoder_called=False,
        cross_split_adapted_state_reuse=False,technical_query_chunk_size=1,A_tie_policy='first_original_native_head_column',
        B_C_tie_policy='physical_class_id_ascending',C_decision_policy=entry.DECISION_POLICY,C_decision_certificate_schema=entry.DECISION_SCHEMA,
        fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS')


def test_repeat_source_literal_exact_4row_2400_parents_and_frozen_resources():
    s=source_spec();assert run.validate_spec(s) is s
    original=run.read(ROOT/'configs/d92_margin_joint_repeat_20261001.json')
    assert s['code']['commit']=='9518d46d2f763e5b080b5337e23155c9de883e09'
    assert s['benchmark']['config']['group_barrier_resources']==entry.RESOURCES
    assert sum(s['benchmark']['cohorts'][r['cohort']]['expected_split_count'] for r in s['rows'])==2400
    for a,b in zip(s['rows'],original['rows']):
        assert {k:v for k,v in a.items() if k!='output_root'}=={k:v for k,v in b.items() if k!='output_root'}
    for row in s['rows']:run.validate_preflight(full_preflight(s,row),s,row,'b'*40)


@pytest.mark.parametrize('change',['row','seed','matrix','resources','bool','algorithm','owner','packet','checkpoint'])
def test_benchmark_reject_changed_fixed_declaration(change):
    s=source_spec()
    if change=='row':s['rows'].pop()
    elif change=='seed':s['rows'][0]['expected_model_seed']=99
    elif change=='matrix':s['benchmark']['cohorts']['rx3']['matrix']['receivers'].pop()
    elif change=='resources':s['benchmark']['config']['group_barrier_resources']['max_newton_iterations']=99
    elif change=='bool':s['benchmark']['config']['group_barrier_resources']['max_line_search_trials']=True
    elif change=='algorithm':s['benchmark']['config']['algorithm']['max_iterations']=5
    elif change=='owner':s['execution']['launch_owner']='another'
    elif change=='packet':s['rows'][2]['ground_packet']+='/changed'
    else:s['rows'][2]['expected_checkpoint_sha256']='c'*64
    with pytest.raises(ValueError):run.validate_spec(s)


def test_full_matrix_cannot_pass_with_same_count_duplicate_coordinates():
    s=source_spec();row=s['rows'][0];p=full_preflight(s,row)
    p['splits'][1].update({k:p['splits'][0][k] for k in ('receiver','scenario','k','new_count','support_seed')})
    with pytest.raises(ValueError,match='complete physical matrix'):run.validate_preflight(p,s,row,'b'*40)


def full_fixed_output(tmp_path,s,row,p):
    out=tmp_path/'fixed';out.mkdir();(out/'state_arrays').mkdir()
    for name in ('B_MARGIN','C_GROUP_BARRIER_seq'):(out/'state_arrays'/ (name+'.npz')).write_bytes(b'pure structural placeholder')
    run.write(out/'startup.json',dict(p,config=s['benchmark']['config'],pid=321,cuda_visible_devices='',
        blas_environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
    rows=dict(A=[],B=[],C=[]);completed=[];work=[]
    for split in p['splits']:
        item=deepcopy(split);sid=item['split_id'];pid=item['query_ids'][0];reuse=not item['new_support_ids']
        for key,state in (('b_state_ref','B_MARGIN'),('c_state_ref','B_MARGIN' if reuse else 'C_GROUP_BARRIER_seq')):
            item[key]=dict(path='state_arrays/'+state+'.npz',namespace=json.dumps(dict(run_id=s['run_id'],row_id=row['row_id'],
                split_id=sid,scope=entry.TRAIN_SCOPE,state=state)))
        item.update(c_reuses_b=reuse,c_inherited_from_b_state_ref=item['b_state_ref'],B_classes=item['old_classes'],
            C_classes=item['registered_classes'],support_only_fit=True,query_fit_access=False,source_fit_access=False)
        completed.append(item);old=item['old_classes'];classes=item['registered_classes'];B=np.arange(6,dtype=float)[None,:]
        rows['A'].append(dict(split_id=sid,query_id=pid,classes=p['ordered_ground_classes'],scores=[0.]*6,prediction=p['ordered_ground_classes'][0]))
        rows['B'].append(entry._prediction(sid,pid,old,B))
        if reuse:
            values=B;prediction=old[-1];decision=dict(schema=entry.DECISION_SCHEMA,policy='EXACT_ACTUAL_B_REUSE',new0_reuses_actual_B=True,prediction=prediction)
        else:
            new=[c for c in classes if c not in old];h=np.zeros((1,len(new)));g=np.asarray([0.])
            values,lo,ln=entry.core._compose(SimpleNamespace(classes=classes,audit=dict(old_classes=old)),B,h,g)
            gap=float(g[0]+lo[0,-1]-ln[0,0]);prediction=old[-1] if gap>0 else new[0] if gap<0 else min(old[-1],new[0])
            decision=dict(schema=entry.DECISION_SCHEMA,policy=entry.DECISION_POLICY,new0_reuses_actual_B=False,
                old_classes=old,new_classes=new,old_raw_scores=B[0].tolist(),new_raw_scores=h[0].tolist(),gate_logit=0.,
                old_log_probabilities=lo[0].tolist(),new_log_probabilities=ln[0].tolist(),old_winner=old[-1],new_winner=new[0],group_gap=gap,prediction=prediction)
        rows['C'].append(entry._prediction(sid,pid,classes,values,prediction))
        work.append(dict(split_id=sid,query_id=pid,B_work={},C_work=None if reuse else {},C_reused_B_scores=reuse,
            C_structural_decision=decision,C_structural_prediction=prediction,C_public_predict_calls=int(not reuse),
            C_public_predict_seconds=0.,C_public_predict_internal_work=None,
            C_public_predict_internal_work_unavailable_reason='synthetic',scope='synthetic_singleton'))
    streams={}
    for name,records in rows.items():
        file=out/('predictions_'+name+'.jsonl');file.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        streams[name]=dict(path=file.name,record_count=len(records),file_bytes=file.stat().st_size)
    (out/'predictions.jsonl').write_bytes((out/'predictions_C.jsonl').read_bytes())
    (out/'query_score_work.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in work),encoding='utf-8')
    run.write(out/'state_manifest.json',dict(status='COMPLETE'))
    marker=dict(p,status='COMPLETE',pid=321,splits=completed,completed_split_count=len(completed),actual_stage_count=sum(1+int(bool(v['new_support_ids'])) for v in completed),
        streams=streams,compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
        query_score_work=dict(path='query_score_work.jsonl',record_count=len(work),file_bytes=(out/'query_score_work.jsonl').stat().st_size),state_manifest='state_manifest.json',
        resources=dict(actual_training_counters=dict.fromkeys(entry.AUDIT_COUNTERS+entry.PEAKS,0),actual_factorization_attempts=0))
    run.write(out/'predictions_complete.json',marker)
    return out


def test_supervisor_full_physical_streams_certificate_pid_and_adjfactor_validation(tmp_path):
    s=source_spec();row=s['rows'][2];p=full_preflight(s,row);out=full_fixed_output(tmp_path,s,row,p)
    run.verify_marker(out,s,row,'b'*40,p)
    marker=run.read(out/'predictions_complete.json');marker['resources']['actual_training_counters']['group_gate_adjoint_factorization_attempts']=1
    run.write(out/'predictions_complete.json',marker,replace=True)
    with pytest.raises(ValueError,match='forward/adjoint factor'):run.verify_marker(out,s,row,'b'*40,p)
    marker['resources']['actual_factorization_attempts']=1;run.write(out/'predictions_complete.json',marker,replace=True)
    run.verify_marker(out,s,row,'b'*40,p)
    records=lines(out/'query_score_work.jsonl');records[0]['C_structural_prediction']='invented'
    (out/'query_score_work.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8')
    marker['query_score_work']['file_bytes']=(out/'query_score_work.jsonl').stat().st_size;run.write(out/'predictions_complete.json',marker,replace=True)
    with pytest.raises(ValueError,match='decision binding'):run.verify_marker(out,s,row,'b'*40,p)


def test_supervisor_all_preflights_first_no_truth_no_retry_healthy_continue(tmp_path):
    s=source_spec();release=tmp_path/'release';release.mkdir()
    s['code']['cwd']=release.as_posix();s['code']['environment']='/synthetic/python'
    s['execution']['remote_run_root']=(tmp_path/'run').as_posix()
    for row in s['rows']:row['output_root']=s['execution']['remote_run_root']+'/'+row['row_id']
    book=[];lookup={r['row_id']:r for r in s['rows']};pid_by_row={r['row_id']:500+i for i,r in enumerate(s['rows'])}
    def pre(argv,log,cwd,environment):
        name=argv[argv.index('--row-id')+1];book.append(('pre',name));Path(log).write_text('pure synthetic preflight',encoding='utf-8')
        assert not any('/truth.json' in v for v in argv)
        return full_preflight(s,lookup[name])
    def launch(argv,log,cwd,environment,started):
        name=argv[argv.index('--row-id')+1];book.append(('launch',name));assert sum(v[0]=='pre' for v in book)==4
        started(pid_by_row[name]);Path(log).write_text('synthetic complete detailed log',encoding='utf-8')
        if name==s['rows'][0]['row_id']:raise RuntimeError('only this row failed')
        assert environment['CUDA_VISIBLE_DEVICES']=='' and environment['OMP_NUM_THREADS']=='2'
        return dict(pid=pid_by_row[name],returncode=0)
    def marker(path,s,row,commit,preflight):return dict(pid=pid_by_row[row['row_id']],status='COMPLETE')
    with pytest.raises(RuntimeError,match='healthy rows'):run.run(s,'b'*40,preflight_fn=pre,launch_fn=launch,marker_fn=marker)
    final=run.read(tmp_path/'run/complete.json')
    assert final['completed_row_count']==3 and final['truth_read'] is final['scorer_invoked'] is False
    assert sum(v[0]=='launch' for v in book)==4 and final['automatic_retry'] is False
    assert all((Path(r['output_root'])/'resolved_config.json').is_file() for r in s['rows'])
    with pytest.raises(FileExistsError):run.run(s,'b'*40,preflight_fn=pre,launch_fn=launch,marker_fn=marker)


def test_publication_source_whitelist_and_pushed_identity_only():
    s=source_spec();paths=publish.release_paths(s)
    assert len(paths)==len(set(paths)) and all((ROOT/p).is_file() for p in paths)
    assert 'tools/score_d92_group_barrier_joint_benchmark.py' in paths
    assert 'tools/evaluate_d92_margin_joint_benchmark.py' not in paths
    assert 'tools/report_d92_margin_joint_benchmark.py' not in paths
    compile(publish.IMPORT_PROGRAM,'group-isolated-source-import','exec')
    compile(publish.REMOTE.replace('CONFIG',repr(publish.remote_config(s,dict(runtime_commit='b'*40),'/exclusive.tar','0'*64))),'group-exclusive-publication','exec')
    calls=[]
    def git(argv,**kw):
        if argv[1]=='rev-parse':return 'b'*40
        if argv[1]=='branch':return 'synthetic'
        if argv[1]=='ls-remote':return 'b'*40+'\trefs/heads/synthetic'
        if argv[1]=='status':return ''
        raise AssertionError(argv)
    binding=publish.git_binding(s,check_output=git,check_call=lambda argv,**kw:calls.append(argv))
    assert binding['remote_oid']=='b'*40 and calls==[['git','merge-base','--is-ancestor',s['code']['commit'],'b'*40]]
