"""Synthetic input binding and real ProtoFrame full-support query producer."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    for path in (ROOT/'tools/evaluate_d92_proto_frame_joint_benchmark.py',Path(__file__)):
        source=path.read_bytes().decode('utf-8',errors='strict')
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in source)
        ast.parse(source,filename=str(path))
    print('UTF8/AST PASS: 2 owned ProtoFrame benchmark/test files; no numerical imports')
    raise SystemExit(0)

from copy import deepcopy
from types import SimpleNamespace

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import numpy as np
import pytest
import torch

import evaluate_d92_proto_frame_joint_benchmark as entry
from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
from cvsrffi.d92_ground_classifier_a import GroundClassifierA,GroundFeatureContract,GroundHeadMetadata,WEIGHT_KEY

SHA='a'*64
COMMIT='b'*40
OLD=['g'+str(i) for i in range(6)]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def summary(root,old_classes=OLD):
    root.mkdir()
    core=np.zeros((6,160),dtype=np.int8);core[np.arange(6),np.arange(6)]=50
    payload=dict(schema=np.asarray(codec.SCHEMA),feature_schema=np.asarray(codec.FEATURE_SCHEMA),
        residual_rank=np.asarray(3,dtype=np.int16),center_domain_handle=np.asarray('ground-0'),
        domain_registry=np.asarray(['ground-0','ground-1']),residual_domain_registry=np.asarray(['ground-1']),
        class_registry=np.asarray(old_classes),core_q=core,core_scale=np.full(6,.01,dtype=np.float16),
        residual_basis_q=np.zeros((6,3,160),dtype=np.int8),residual_basis_scale=np.ones((6,3),dtype=np.float16),
        residual_coeff_q=np.zeros((1,6,3),dtype=np.int8),residual_coeff_scale=np.ones((1,6),dtype=np.float16),
        radius_q=np.zeros((2,6),dtype=np.int8),radius_scale=np.ones(6,dtype=np.float16))
    meta=dict(schema=codec.SCHEMA,metadata_schema='cvs.matched.ground.v2',checkpoint_sha256=SHA,
        source_prototype_artifact_sha256='c'*64,provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L',
        formal_phase2_eligible=True,feature_dim=160,residual_rank=3,radius_histogram_bins=4096,
        member_allowlist=[codec.NPZ_NAME],npz_member_allowlist=sorted(codec.ALLOWED_NPZ_MEMBERS),
        resource_audit=codec._numeric_resource_audit(payload),source_role='L_s',target_access=False,
        component_state='CURRENT_MATCHED_SOURCE_ONLY_V2',historical_outer_signature_claim=False)
    np.savez_compressed(root/codec.NPZ_NAME,**payload);save(root/'manifest.json',meta)


def fixture(tmp_path,monkeypatch,*,new=2,k=2,old_classes=None,new_classes=None):
    old=list(OLD if old_classes is None else old_classes)
    classes=old+(list(new_classes) if new_classes is not None else ['n'+str(i) for i in range(new)])
    assert len(old)==6 and len(classes)==6+new
    labels=np.repeat(np.arange(len(classes)),k);n=len(labels)
    ids=np.asarray([f'support-{i:03d}' for i in range(n)]+['opaque-z','opaque-a','opaque-q'])
    arrays={key:np.zeros((len(ids),96 if key=='fft' else 160),dtype=np.float32) for key in entry.BRANCHES}
    # Full support has zero geometry: this deterministic toy avoids tuning the
    # solver and still exercises real new Ridge, free gate and full archive.
    for j in range(3):arrays['z_id'][n+j,j]=1.
    split=dict(split_id='literal-split',capsule_id='literal-capsule',protocol_schema='p2_min_v1',
        phase2_data_status='VALIDATED_ONCE',receiver='rx-literal',scenario='practical_high',k=k,
        support_seed=7,registered_classes=classes,support_indices=list(range(n)),
        query_indices=[n+1,n,n+2],support_labels=labels.tolist())
    capsule=tmp_path/'capsule';save(capsule/'splits'/'literal-split.json',split)
    packet=tmp_path/'packet';save(packet/'complete.json',dict(packet_total_file_bytes=4567))
    save(packet/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=17)))
    (packet/'head_weight.float32.bin').write_bytes(bytes(6*160*4))
    native=tuple(reversed(old));weight=torch.zeros((6,160),dtype=torch.float32)
    metadata=GroundHeadMetadata(SHA,WEIGHT_KEY,native,9.5,1e-4,
        GroundFeatureContract(SHA,'z_id','feat_joint','raw','float32',160),
        'MATCHED_SOURCE_ONLY_SCRATCH',False,(),'none')
    head=GroundClassifierA(weight=weight,metadata=metadata)
    producer=dict(classes=old,feature_array_bytes=sum(a.nbytes for a in arrays.values()),
        feature_file_bytes=999,model_file_bytes=123)
    provenance=dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',source_role_comparison='EXACT_MATCH',
        checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False)
    cache_calls=[]
    def load(**kwargs):
        cache_calls.append(kwargs);return arrays,ids,producer,dict(seed=17),provenance
    monkeypatch.setattr(entry,'load_features',load)
    monkeypatch.setattr(entry,'validate_capsule',lambda *args:dict(capsule_id='literal-capsule',split_count=1))
    monkeypatch.setattr(entry,'load_packet',lambda path:head)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Checkpoint loading forbidden'))
    ground=tmp_path/'summary';summary(ground,old)
    calls=[];states=[]
    actual_prepare=entry.core.prepare_proto_frame_joint_training
    actual_fit=entry.core.fit_proto_frame_joint_local_ridge
    def prepare(**kwargs):
        selected=list(map(str,kwargs['support_ids']))
        assert set(selected)<=set(ids[:n]) and not set(selected)&set(ids[n:])
        assert 'query_labels' not in kwargs and 'query_roles' not in kwargs
        assert kwargs['prototype_frame'].classes==tuple(sorted(old))
        assert kwargs['context']['parent_k']==kwargs['context']['train_k']==k
        calls.append(('prepare','B' if kwargs['inherited'] is None else 'C_seq',selected))
        if kwargs['inherited'] is not None:assert kwargs['inherited'] is states[0]
        return actual_prepare(**kwargs)
    def fit(prepared,**kwargs):
        assert set(kwargs)=={'mode','log_callback','state_callback'}
        calls.append(('fit',kwargs['mode']))
        state=actual_fit(prepared,**kwargs);states.append(state);return state
    monkeypatch.setattr(entry.core,'prepare_proto_frame_joint_training',prepare)
    monkeypatch.setattr(entry.core,'fit_proto_frame_joint_local_ridge',fit)
    args=dict(run_id='literal-benchmark',row_id='literal-row',release_commit=COMMIT,
        row_root=str(tmp_path/'source-row'),capsule=str(capsule),output=str(tmp_path/'predictions'),
        config=dict(algorithm=deepcopy(entry.core.FROZEN_CONFIG),proto_frame_resources=deepcopy(entry.RESOURCES)),
        expected_capsule_id='literal-capsule',expected_checkpoint_sha256=SHA,
        branch_features=str(tmp_path/'cache'),ground_packet=str(packet),ground_summary=str(ground),
        ground_summary_already_deployed=False)
    return SimpleNamespace(args=args,arrays=arrays,ids=ids,split=split,head=head,calls=calls,
        states=states,cache_calls=cache_calls)


def test_preflight_real_reader_bindings_no_fit_score_or_output(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(GroundClassifierA,'score',lambda *a,**kw:pytest.fail('Preflight inference forbidden'))
    marker=entry.preflight(**case.args)
    assert marker['status']=='PROTO_FRAME_QUERY_PREFLIGHT_COMPLETE'
    assert marker['schema']==entry.SCHEMA and marker['method']==entry.core.FROZEN_CONFIG['method']
    assert marker['source_identity']['model_seed']==marker['ground_geometry_identity']['model_seed']==17
    assert marker['ground_geometry_identity']['checkpoint_sha256']==SHA
    assert marker['run_binding']['ground_summary']==str(Path(case.args['ground_summary']).resolve())
    assert marker['ground_summary_already_deployed'] is False
    assert marker['ground_geometry_binding']['prototype_teacher_targets'] is False
    assert marker['ground_geometry_binding']['ground_payload_audit']['incremental_transfer_bytes']>0
    assert case.calls==[] and not Path(case.args['output']).exists()


@pytest.mark.parametrize('new,k',[(2,2),(0,2),(2,1),(0,1)])
def test_real_core_full_support_actual_B_C_lineage_five_fixed_streams(tmp_path,monkeypatch,new,k):
    case=fixture(tmp_path,monkeypatch,new=new,k=k)
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:pytest.fail('Shared NumPy/Torch input bridge forbidden'))
    monkeypatch.setattr(torch.Tensor,'numpy',lambda *a,**kw:pytest.fail('Shared Torch/NumPy output bridge forbidden'))
    marker=entry.predict(**case.args);out=Path(case.args['output'])
    assert marker['status']=='COMPLETE' and marker['actual_stage_count']==(2 if new else 1)
    assert marker['actual_baseline_stage_count']==(2 if new else 1)
    assert marker['candidate_preparation_count']==marker['candidate_stage_count']==(2 if new else 1)
    assert marker['nominal_adapter_parameter_count']==5
    assert marker['query_fit_access'] is marker['query_role_used'] is marker['view_aggregation'] is False
    assert marker['query_count_used_for_decision'] is False
    assert marker['resources']['C_public_predict_internal_work'] is None
    meta=marker['splits'][0];assert meta['c_reuses_b'] is (not new)
    assert meta['c_inherited_from_b_state_ref']==meta['b_state_ref']
    assert len(case.states)==(2 if new else 1)
    if new:assert case.states[1].prior is case.states[0]
    else:
        assert meta['c_state_ref']==meta['b_state_ref']
        assert meta['r0_c_state_ref']==meta['r0_b_state_ref']
    assert len(case.states[0].ids)==6*k
    if new:assert len(case.states[1].ids)==(6+new)*k
    stages=lines(out/'fit_stages.jsonl')
    assert all(stage['train_k']==stage['parent_k']==k and stage['fold'] is None and stage['trial'] is None for stage in stages)
    expected_ids=case.ids[case.split['query_indices']].tolist()
    assert set(marker['streams'])==set(entry.STREAM_NAMES)
    for name in entry.STREAM_NAMES:
        records=lines(out/marker['streams'][name]['path'])
        assert [record['query_id'] for record in records]==expected_ids
        assert all(set(record)=={'split_id','query_id','classes','scores','prediction'} for record in records)
        assert marker['streams'][name]['record_count']==3
        assert marker['streams'][name]['file_bytes']==(out/marker['streams'][name]['path']).stat().st_size
    assert all(row['prediction']==case.head.classes[0] for row in lines(out/'predictions_A.jsonl'))
    assert (out/'predictions_C.jsonl').read_bytes()==(out/'predictions.jsonl').read_bytes()
    assert marker['resources']['query_score_calls']==dict(A=3,B=3,C=3 if new else 0,R0_B=3,R0_C=3 if new else 0)
    for field in ('actual_work','preparation_actual_work','stage_actual_work','score_actual_work'):
        assert set(marker[field])==entry.WORK_KEYS
    for key in entry.WORK_KEYS:
        parts=[marker[field][key] for field in ('preparation_actual_work','stage_actual_work','score_actual_work')]
        expected=max(parts) if key in entry.WORK_MAX_KEYS else sum(parts)
        if key.endswith('_seconds'):
            assert marker['actual_work'][key]==pytest.approx(expected,rel=128*np.finfo(float).eps,
                abs=128*np.finfo(float).eps*max(1.,abs(expected)))
        else:assert marker['actual_work'][key]==expected
        assert marker['actual_work_aggregation'][key]==('MAX' if key in entry.WORK_MAX_KEYS else 'SUM')
    assert marker['resources']['native_wire_bytes'] is marker['resources']['incremental_transfer_bytes'] is None
    assert marker['resources']['newly_generated_ground_statistics_bytes']==0
    work=lines(out/'query_score_work.jsonl')
    assert all(row['C_structural_decision']['prediction']==row['C_structural_prediction'] for row in work)
    assert all(row['R0_score_work'] is None and row['C_public_predict_internal_work'] is None for row in work)
    manifest=json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema']=='d92_proto_frame_joint_state_archive_v1' and manifest['status']=='COMPLETE'
    assert all(json.loads(ref['namespace'])['scope']==entry.TRAIN_SCOPE for ref in manifest['files'])
    assert all((out/ref['path']).is_file() for ref in manifest['files'])
    events=lines(out/'training_events.jsonl');compact=lines(out/'training_events_compact.jsonl')
    assert events and len(events)==len(compact)
    assert 'nominal_parameters=5' in (out/'training.log').read_text(encoding='utf-8')
    assert all((out/name).is_file() for name in ('training_events_compact.csv','fit_stages.csv','preparations.csv'))
    with pytest.raises(FileExistsError):entry.predict(**case.args)


def test_query_permutation_and_other_record_changes_leave_single_record_fixed(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch,k=1)
    first=entry.predict(**case.args);out=Path(case.args['output'])
    expected={name:{r['query_id']:r for r in lines(out/first['streams'][name]['path'])} for name in entry.STREAM_NAMES}
    case.split['query_indices'].reverse()
    save(Path(case.args['capsule'])/'splits'/'literal-split.json',case.split)
    # New split fit uses a fresh actual B; fixture logging no longer checks its
    # inherited object against the previous completed benchmark's state.
    case.states.clear()
    second=entry.predict(**dict(case.args,output=str(tmp_path/'second')))
    for name in entry.STREAM_NAMES:
        assert {r['query_id']:r for r in lines(tmp_path/'second'/second['streams'][name]['path'])}==expected[name]
    changed=case.split['query_indices'][0];case.arrays['z_id'][changed]+=2.
    case.states.clear();third=entry.predict(**dict(case.args,output=str(tmp_path/'third')))
    for name in entry.STREAM_NAMES:
        rows={r['query_id']:r for r in lines(tmp_path/'third'/third['streams'][name]['path'])}
        assert all(rows[pid]==row for pid,row in expected[name].items() if pid!=str(case.ids[changed]))


@pytest.mark.parametrize('new',[0,2])
def test_nonlexical_numeric_class_ids_keep_physical_labels_nonzero_R0_and_standalone_parser(tmp_path,monkeypatch,new):
    # Numeric-style IDs expose the requested-order LocalRidge ABI: lexical
    # order differs from both numeric order and this declared capsule order.
    old=['2','10','1','20','3','11'];new_classes=['7','12'][:new]
    case=fixture(tmp_path,monkeypatch,new=new,k=1,old_classes=old,new_classes=new_classes)
    registry=case.split['registered_classes'];labels=case.split['support_labels']
    support=case.split['support_indices'];physical_labels={str(case.ids[i]):registry[y] for i,y in zip(support,labels)}
    for physical_index,label in zip(support,labels):case.arrays['z_id'][physical_index,label]=1.
    original_fit=entry.fit_branch_local_ridge;fitted=[]
    def fit(**kwargs):
        assert kwargs['classes']==sorted(kwargs['classes'])
        assert [kwargs['classes'][int(y)] for y in kwargs['support_labels']]==[
            physical_labels[str(pid)] for pid in kwargs['support_ids']]
        state=original_fit(**kwargs);fitted.append((kwargs,state));return state
    monkeypatch.setattr(entry,'fit_branch_local_ridge',fit)
    preflight=entry.preflight(**case.args);marker=entry.predict(**case.args);out=Path(case.args['output'])
    assert len(fitted)==1+int(new>0)
    assert marker['splits'][0]['declared_registered_classes']==registry
    for key in entry.STREAM_NAMES:
        rows=lines(out/marker['streams'][key]['path'])
        expected_classes=list(case.head.classes) if key=='A' else sorted(old) if key in ('B','R0_B') else sorted(registry)
        assert all(row['classes']==expected_classes for row in rows)
    assert list(case.head.classes)!=sorted(old)
    # Compare nonzero R0 scores against the unchanged original solver requested
    # in native capsule order, then reorder its returned columns by class ID.
    for key,native_registry in (('R0_B',old),('R0_C',registry)):
        selected=[i for i in support if physical_labels[str(case.ids[i])] in native_registry]
        ys=np.asarray([native_registry.index(physical_labels[str(case.ids[i])]) for i in selected],dtype=np.int64)
        oracle=original_fit(**{k:case.arrays[k][selected] for k in entry.BRANCHES},support_labels=ys,
            support_ids=case.ids[selected],classes=native_registry,old_classes=old,arm='local_ridge')
        assert list(oracle.classes)==native_registry
        columns=[native_registry.index(name) for name in sorted(native_registry)]
        canonical_state=fitted[0 if key=='R0_B' or not new else 1][1]
        np.testing.assert_array_equal(canonical_state.alpha,oracle.alpha[:,columns])
        for name in ('support_background','support_auxiliary','reference_kernel','center_mean'):
            np.testing.assert_array_equal(getattr(canonical_state,name),getattr(oracle,name))
        for name in ('reference_self','center_grand','bandwidth_tau','trace_scale'):
            assert getattr(canonical_state,name)==getattr(oracle,name)
        # The returned alpha column permutation can change the BLAS reduction
        # layout. Two N-term dot products need not be bitwise identical. For
        # rm1 in [-1,0], the anchored diff and its mean have magnitude <=2;
        # hence |centered kernel_i| <= gamma*(4+max|mean|+|grand|). Use the
        # two-dot 2*gamma_N*sum|kernel_i*alpha_ij| bound, with eps (a conservative
        # unit roundoff), while keeping labels, head coefficients and decisions
        # exact. This tolerance exists only in the independent test oracle.
        n=len(canonical_state.alpha);eps=np.finfo(np.float64).eps
        gamma_n=n*eps/(1-n*eps)
        kernel_bound=canonical_state.trace_scale*(4+np.max(np.abs(canonical_state.center_mean))+
            abs(canonical_state.center_grand))
        dot_bound=2*gamma_n*kernel_bound*np.sum(np.abs(canonical_state.alpha),axis=0)
        for record,physical_index in zip(lines(out/marker['streams'][key]['path']),case.split['query_indices']):
            one={k:case.arrays[k][physical_index:physical_index+1] for k in entry.BRANCHES}
            expected=oracle.score(**one)[0,columns]
            assert np.all(np.abs(np.asarray(record['scores'])-expected)<=dot_bound)
            assert np.ptp(expected)>0.
            assert record['prediction']==min(record['classes'][j] for j in np.flatnonzero(expected==expected.max()))
    # The real standalone parser keeps its strict five-stream/NPZ/prior/work
    # checks. Only the fixture's single-parent metadata replaces the fixed full
    # capsule matrix loader; full 2400-parent orchestration is tested separately.
    import score_d92_proto_frame_joint_benchmark as score
    initial=preflight['splits'][0]
    monkeypatch.setattr(score,'load_capsule_metadata',lambda *a:(dict(capsule_id=case.args['expected_capsule_id']),
        {initial['split_id']:initial}))
    row=dict(row_id=case.args['row_id'],cohort='literal',expected_model_seed=17,expected_checkpoint_sha256=SHA,
        output_root=str(out.parent),**{k:case.args[k] for k in ('row_root','branch_features','ground_packet',
            'ground_summary','ground_summary_already_deployed')})
    co=dict(capsule=case.args['capsule'],expected_capsule_id=case.args['expected_capsule_id'],expected_split_count=1,
        matrix=dict(receivers=[initial['receiver']],scenarios=[initial['scenario']],
            support_seeds=[initial['support_seed']],ks=[1],new_counts=[new]))
    spec=dict(schema='d92_proto_frame_joint_query_benchmark_v1',run_id=case.args['run_id'],rows=[row],
        benchmark=dict(config=case.args['config'],cohorts={'literal':co}))
    assert score.validate_row_output(out,spec,row,COMMIT,preflight)==marker


@pytest.mark.parametrize('field',['query_labels','query_truth','query_roles','query_class_counts'])
def test_query_information_rejected_before_training(tmp_path,monkeypatch,field):
    case=fixture(tmp_path,monkeypatch);case.split[field]=['forbidden']
    save(Path(case.args['capsule'])/'splits'/'literal-split.json',case.split)
    with pytest.raises(ValueError,match='Forbidden query information'):entry.predict(**case.args)
    assert not case.calls and not Path(case.args['output']).exists()


@pytest.mark.parametrize('fault',['algorithm','bool_limit','limit_change','packet_seed','summary_checkpoint','deployment_bool'])
def test_ground_and_fixed_controls_rejected_without_fit(tmp_path,monkeypatch,fault):
    case=fixture(tmp_path,monkeypatch);args=deepcopy(case.args)
    if fault=='algorithm':args['config']['algorithm']['coordinates']=6
    elif fault=='bool_limit':args['config']['proto_frame_resources']['max_newton_iterations']=True
    elif fault=='limit_change':args['config']['proto_frame_resources']['max_newton_iterations']=101
    elif fault=='packet_seed':save(Path(args['ground_packet'])/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=99)))
    elif fault=='deployment_bool':args['ground_summary_already_deployed']='false'
    else:
        path=Path(args['ground_summary'])/'manifest.json';meta=json.loads(path.read_text(encoding='utf-8'))
        meta['checkpoint_sha256']='d'*64;save(path,meta)
    with pytest.raises(ValueError):entry.predict(**args)
    assert not case.calls and not Path(args['output']).exists()


def test_partial_C_failure_keeps_B_failed_work_nonfinite_arrays_no_retry(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch);original=entry.core.fit_proto_frame_joint_local_ridge;attempts=[]
    def fit(prepared,**kwargs):
        attempts.append(kwargs['mode'])
        if kwargs['mode']=='C_seq':
            values=dict(partial=np.asarray([np.nan,np.inf]))
            ref=kwargs['state_callback']('failure',values)
            work=entry._empty_work();work['gate_forward_factorization_attempts']=2
            work['gate_forward_wall_seconds']=.125
            raise entry.core.ProtoFrameFailure('SYNTHETIC_LIMIT',dict(status='TECHNICAL_FAILURE',
                failure_state_ref=ref,actual_work=work,operation_audits=[]),values) from ArithmeticError('original cause')
        return original(prepared,**kwargs)
    monkeypatch.setattr(entry.core,'fit_proto_frame_joint_local_ridge',fit)
    with pytest.raises(entry.core.ProtoFrameFailure,match='SYNTHETIC_LIMIT'):entry.predict(**case.args)
    out=Path(case.args['output']);failure=json.loads((out/'technical_failure.json').read_text(encoding='utf-8'))
    assert attempts==['B','C_seq'] and failure['completed_stages'][0]['state']=='B_PROTO_FRAME'
    assert failure['failed_work_audit']['actual_work']['gate_forward_factorization_attempts']==2
    assert failure['failed_work_audit']['actual_work']['gate_forward_wall_seconds']==.125
    assert failure['original_cause']['error_type']=='ArithmeticError'
    assert failure['automatic_retry'] is False and not (out/'predictions_complete.json').exists()
    ref=failure['failed_state_ref']
    with np.load(out/ref['path'],allow_pickle=False) as numeric:
        assert np.isnan(numeric['partial'][0]) and np.isinf(numeric['partial'][1])
    assert json.loads((out/'state_manifest.json').read_text(encoding='utf-8'))['status']=='TECHNICAL_FAILURE'


def test_missing_completed_state_ref_preserves_returned_head_work(tmp_path,monkeypatch):
    case=fixture(tmp_path,monkeypatch);actual=entry.core.fit_proto_frame_joint_local_ridge
    def fit(prepared,**kwargs):
        state=actual(prepared,**kwargs)
        class Broken:
            def __getattr__(self,name):return getattr(state,name)
            def audit_dict(self):
                audit=state.audit_dict();audit['final_state_ref']=None;return audit
        return Broken()
    monkeypatch.setattr(entry.core,'fit_proto_frame_joint_local_ridge',fit)
    with pytest.raises(ValueError,match='Missing completed actual state'):entry.predict(**case.args)
    out=Path(case.args['output']);failure=json.loads((out/'technical_failure.json').read_text(encoding='utf-8'))
    assert failure['failed_work_unknown'] is False
    assert failure['failed_work_audit']['actual_work']['ridge_calls']>0
    assert failure['failed_state_ref'] is not None and failure['completed_stages']==[]
    assert not (out/'predictions_complete.json').exists()


@pytest.mark.parametrize('deployed',[False,True])
def test_deployed_summary_filebytes_separate_from_unknown_native_wire(tmp_path,monkeypatch,deployed):
    case=fixture(tmp_path,monkeypatch);marker=entry.preflight(**dict(case.args,ground_summary_already_deployed=deployed))
    audit=marker['ground_geometry_binding']['ground_payload_audit']
    total=sum(p.stat().st_size for p in Path(case.args['ground_summary']).iterdir())
    assert audit['incremental_transfer_bytes']==(0 if deployed else total)
    assert marker['ground_geometry_binding']['native_wire_bytes'] is None
    assert marker['ground_geometry_identity']['already_deployed'] is deployed


@pytest.mark.parametrize('flag,expected',[('true',True),('false',False)])
def test_CLI_ground_boolean_and_preflight_only(monkeypatch,flag,expected):
    captured=[];monkeypatch.setattr(entry,'read',lambda path:{})
    monkeypatch.setattr(entry,'preflight',lambda **kw:captured.append(kw) or {'status':'SYNTHETIC'})
    monkeypatch.setattr(entry,'predict',lambda **kw:pytest.fail('Preflight must not fit'))
    argv=['benchmark']
    for name in ('run-id','row-id','release-commit','row-root','capsule','output','config',
        'expected-capsule-id','expected-checkpoint-sha256','branch-features','ground-packet','ground-summary'):
        argv+=['--'+name,'synthetic']
    argv+=['--ground-summary-already-deployed',flag,'--preflight-only']
    monkeypatch.setattr(sys,'argv',argv);entry.main()
    assert captured[0]['ground_summary_already_deployed'] is expected
