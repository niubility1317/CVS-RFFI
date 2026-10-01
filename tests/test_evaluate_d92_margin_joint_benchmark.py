"""Literal inputs, mock support solver and actual archive/native A boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_margin_joint_benchmark as entry
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
            'max_transitions','max_factor_buffer_bytes','log_callback','state_callback'}
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
    def fit(prepared,*,mode,log_callback,state_callback):
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
        audit=dict.fromkeys(entry.core.STAGE_COUNTERS,0)
        audit.update(status='COMPLETED',optimizer_steps=count,final_head_fit_count=1,final_factorization_count=2,
            completed_factorization_count=2,final_state_ref=final,preparation=prepared.audit,
            margin_qp_peak_factor_buffer_bytes=100 if mode=='B' else 80,margin_qp_peak_explicit_solve_temporary_bytes=20)
        class State:
            classes=tuple(sorted(prepared.kwargs['classes']))
            ids=tuple(sorted(str(v) for v in prepared.kwargs['support_ids']))
            def audit_dict(self):return deepcopy(self.audit)
            def score_with_audit(self,**features):
                assert set(features)==set(entry.BRANCHES) and all(a.shape[0]==1 for a in features.values())
                calls.append(('score',mode,float(features['fft'][0,0])))
                values=np.arange(len(self.classes),dtype=np.float64)[None,:]+float(features['fft'][0,0])
                return values,dict(physical_records=1,measured_pair_work=len(self.ids))
        state=State();state.audit=audit;states.append(state)
        log_callback(dict(event='MARGIN_JOINT_FINAL',mode=mode,state_ref=final,audit=audit))
        return state
    monkeypatch.setattr(entry.core,'prepare_margin_joint_training',prepare)
    monkeypatch.setattr(entry.core,'fit_margin_joint_local_ridge',fit)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Checkpoint load forbidden'))
    kwargs=dict(run_id='literal-benchmark',row_id='literal-row',release_commit='b'*40,row_root=str(tmp_path/'source-row'),
        capsule=str(capsule),output=str(tmp_path/'prediction'),config=dict(algorithm=deepcopy(entry.core.FROZEN_CONFIG),
        qp_resources=dict(max_transitions=12,max_factor_buffer_bytes=8192)),expected_capsule_id='literal-capsule',
        expected_checkpoint_sha256='a'*64,branch_features=str(tmp_path/'cache'),ground_packet=str(packet))
    return SimpleNamespace(kwargs=kwargs,arrays=arrays,ids=ids,split=split,head=head,calls=calls,states=states,cache_calls=cache_calls)


def test_preflight_reads_bindings_without_training_scores_or_output(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(GroundClassifierA,'score',lambda *a,**k:pytest.fail('Preflight inference forbidden'))
    value=entry.preflight(**case.kwargs)
    assert value['status']=='MARGIN_QUERY_PREFLIGHT_COMPLETE' and value['model_seed']==17
    assert value['source_identity']['source_only_verdict']=='MATCHED_SOURCE_ONLY_SCRATCH'
    assert value['release_commit']=='b'*40 and value['ordered_ground_classes']==list(case.head.classes)
    assert case.calls==[] and not Path(case.kwargs['output']).exists()


@pytest.mark.parametrize('new,k',[(1,2),(0,2),(1,1),(0,1)])
def test_actual_archives_singleton_queries_same_split_B_C_and_truth_free_streams(tmp_path,monkeypatch,new,k):
    case=fixture(tmp_path,monkeypatch,new=new,k=k);marker=entry.predict(**case.kwargs);out=Path(case.kwargs['output'])
    assert marker['schema']==entry.SCHEMA and marker['status']=='COMPLETE'
    startup=json.loads((out/'startup.json').read_text(encoding='utf-8'))
    for key in ('run_id','row_id','release_commit','source_identity','ground_packet_identity','algorithm','qp_resources'):
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
        assert all(r['prediction']==r['classes'][int(np.argmax(r['scores']))] for r in records)
    assert (out/'predictions_C.jsonl').read_bytes()==(out/'predictions.jsonl').read_bytes()
    assert marker['resources']['query_score_calls']==dict(A=3,B=3,C=3 if new else 0)
    assert marker['resources']['actual_training_counters']['optimizer_steps']==(2 if new else 1)*(k>1)
    assert marker['resources']['actual_training_counters']['margin_qp_peak_factor_buffer_bytes']==100
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
    elif case_name=='bool_limit':args['config']['qp_resources']['max_transitions']=True
    elif case_name=='missing_limit':args['config']['qp_resources'].pop('max_factor_buffer_bytes')
    elif case_name=='commit':args['release_commit']=None
    elif case_name=='packet_seed':save(Path(args['ground_packet'])/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=99)))
    else:
        case.split['query_truth']=['forbidden'];save(Path(args['capsule'])/'splits'/'literal-split.json',case.split)
    with pytest.raises(ValueError):entry.predict(**args)
    assert case.calls==[] and not Path(args['output']).exists()


def test_partial_C_failure_keeps_actual_B_and_failed_numeric_state_without_retry(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch);original=entry.core.fit_margin_joint_local_ridge;attempts=[]
    class Failed(RuntimeError):
        arrays={'Z':np.asarray([float('nan')],dtype=np.float64)}
        def audit_dict(self):return dict(failure_code='LITERAL_RESOURCE_LIMIT',margin_qp_forward_factorization_attempts=7)
    def failing(prepared,**kwargs):
        attempts.append(kwargs['mode'])
        if kwargs['mode']=='C_seq':raise Failed('literal failure')
        return original(prepared,**kwargs)
    monkeypatch.setattr(entry.core,'fit_margin_joint_local_ridge',failing)
    with pytest.raises(Failed):entry.predict(**case.kwargs)
    out=Path(case.kwargs['output']);failure=json.loads((out/'technical_failure.json').read_text(encoding='utf-8'))
    assert attempts==['B','C_seq'] and not (out/'predictions_complete.json').exists()
    assert len(failure['completed_stages'])==1 and failure['failed_work_audit']['margin_qp_forward_factorization_attempts']==7
    assert failure['automatic_retry'] is False and failure['failed_state_ref']['failed_numeric_state']
    assert all(not lines(out/('predictions_'+name+'.jsonl')) for name in ('A','B','C'))
    with np.load(out/failure['failed_state_ref']['path'],allow_pickle=False) as archive:assert np.isnan(archive['Z']).all()
    with pytest.raises(FileExistsError):entry.predict(**case.kwargs)


def test_production_core_K3_B_to_C_callbacks_archives_and_query_completion(tmp_path,monkeypatch):
    """One predetermined synthetic case; no seed search or production data.

    Only input/cache/packet I/O is mocked. Restore the actual public core
    functions before calling the predictor, including its real state scorer.
    The resource limits are literal test capacity, not an experiment budget.
    """
    real_prepare=entry.core.prepare_margin_joint_training
    real_fit=entry.core.fit_margin_joint_local_ridge
    case=fixture(tmp_path,monkeypatch,new=1,k=3)
    monkeypatch.setattr(entry.core,'prepare_margin_joint_training',real_prepare)
    monkeypatch.setattr(entry.core,'fit_margin_joint_local_ridge',real_fit)
    rng=np.random.default_rng(341)
    labels=np.asarray(case.split['support_labels'])
    for name in entry.BRANCHES:
        values=rng.normal(size=case.arrays[name].shape)
        values[:len(labels),0]+=labels*.4
        values[:len(labels),1]+=(labels==0)*.6
        case.arrays[name][:]=values.astype(np.float32)
    case.kwargs['config']['qp_resources']=dict(max_transitions=128,max_factor_buffer_bytes=1_000_000)

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
    assert [s['state'] for s in stages]==['B_MARGIN','C_MARGIN_seq']
    assert [p['state'] for p in preparations]==['B_prepare','C_prepare']
    assert preparations[1]['audit']['final_problem']['prior_ref']==split['b_state_ref']
    assert stages[1]['audit']['preparation']['final_problem']['prior_ref']==split['b_state_ref']
    assert stages[0]['training_physical_ids']==sorted(case.ids[:18].tolist())
    assert stages[1]['training_physical_ids']==sorted(case.ids[:21].tolist())
    assert all(s['audit']['optimizer_steps']>0 for s in stages)

    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='COMPLETE'
    by_path={ref['path']:ref for ref in manifest['files']}
    for name,ref in [('B_MARGIN',split['b_state_ref']),('C_MARGIN_seq',split['c_state_ref'])]:
        saved=by_path[ref['path']]
        assert saved['key']==ref['key'] and saved['namespace']==ref['namespace']
        namespace=json.loads(ref['namespace'])
        assert namespace['state']==name and namespace['scope']==entry.TRAIN_SCOPE
        assert namespace['split_id']==case.split['split_id']
        assert namespace['fold'] is None and namespace['trial'] is None
        with np.load(out/ref['path'],allow_pickle=False) as archive:
            assert {'U','Z','alpha'}<=set(archive.files)
            assert all(np.isfinite(archive[key]).all() for key in archive.files)
            for key,meta in ref['arrays'].items():
                assert list(archive[key].shape)==meta['shape']
                assert str(archive[key].dtype)==meta['dtype']
    events=lines(out/'training_events.jsonl')
    compact=lines(out/'training_events_compact.jsonl')
    assert len(events)==len(compact)>6
    for state in ('B_MARGIN','C_MARGIN_seq'):
        actual={event['event'] for event in events if event['state']==state}
        assert {'MARGIN_JOINT_INITIAL','MARGIN_JOINT_GRADIENT','MARGIN_JOINT_TRIAL',
                'MARGIN_JOINT_STEP','MARGIN_JOINT_FINAL'}<=actual
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
        assert all(record['prediction']==record['classes'][int(np.argmax(record['scores']))] for record in records)
    assert (out/'predictions_C.jsonl').read_bytes()==(out/'predictions.jsonl').read_bytes()
    assert marker['query_fit_access'] is marker['source_fit_access'] is marker['truth_read'] is False
