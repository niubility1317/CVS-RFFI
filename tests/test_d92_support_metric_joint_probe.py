"""Synthetic production-entry contracts; no real cache, run or result is read."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    files=(ROOT/'tools/evaluate_d92_support_metric_joint_probe.py',Path(__file__))
    for path in files:
        text=path.read_bytes().decode('utf-8',errors='strict')
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in text)
        ast.parse(text,filename=str(path))
    print('UTF8/AST PASS: 2 owned SupportMetric evaluator/test files; no numerical imports')
    raise SystemExit(0)

from copy import deepcopy
from types import SimpleNamespace

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import numpy as np
import pytest
import torch

import evaluate_d92_support_metric_joint_probe as evaluate
from cvsrffi import d92_proto_frame_primitives as primitive
from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
from cvsrffi.d92_ground_classifier_a import GroundClassifierA,GroundFeatureContract,GroundHeadMetadata

RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,
    max_integer_bits=16384,max_fraction_operations=1_000_000,max_secular_iterations=128)
SHA='a'*64
COMMIT='b'*40
OLD=tuple('c'+str(i) for i in range(6))


def synthetic(k=3,new=2,*,zero=False):
    classes=list(OLD)+['c'+str(i) for i in range(6,6+new)]
    labels=np.repeat(np.arange(len(classes)),k)
    ids=[f'physical_{i:04d}' for i in range(len(labels))]
    raw={name:np.zeros((len(labels),d),dtype=np.float32)
        for name,d in zip(evaluate.BRANCHES,(160,96,160,160,160))}
    if not zero:
        for values in raw.values():
            values[:,0]=labels+1
            values[:,1]=np.arange(len(labels))+1
    return dict(raw,support_labels=labels,support_ids=ids,classes=classes,old_classes=list(OLD))


def frame():
    return primitive.build_proto_frame_dictionary(prototypes=np.eye(6,160),classes=OLD)


def real_ground():
    classes=tuple(reversed(OLD))
    contract=GroundFeatureContract(SHA,'z_id','feat_joint','raw','float32',160)
    metadata=GroundHeadMetadata(SHA,'id_backbone.cls_head.head.weight',classes,16.,1e-4,
        contract,'MATCHED_SOURCE_ONLY_SCRATCH',False,(),'none')
    return GroundClassifierA(weight=torch.zeros((6,160),dtype=torch.float32),metadata=metadata)


def all_work():
    return dict.fromkeys(tuple(evaluate.WORK_SUM_KEYS)+tuple(evaluate.WORK_MAX_KEYS),0)


def fake_models(monkeypatch,*,fail_c=False,wrong_old_prediction=False,real_baseline=False):
    """Mocks isolate evaluator contracts; the real-core production test is separate."""
    inherited=[];fit_states=[];prepared=[]
    class State:
        def __init__(self,kwargs,mode='baseline'):
            self.kwargs=kwargs;self.classes=tuple(kwargs['classes']);self.mode=mode
            self.support_background=np.zeros((len(kwargs['support_ids']),1))
            self.support_auxiliary=self.support_background.copy()
            self.alpha=np.zeros((len(kwargs['support_ids']),len(self.classes)))
            self.reference_kernel=np.zeros((1,1));self.reference_self=0.
            self.center_mean=np.zeros(1);self.center_grand=0.;self.bandwidth_tau=1.;self.trace_scale=1.
        def audit_dict(self):
            if self.mode=='baseline':
                return dict(persistent_state_bytes=16,final_fit=dict(factorization_calls=1,
                    effective_degrees_of_freedom_extra_triangular_solves=0))
            work=all_work();work['ridge_calls']=2 if self.mode=='C_seq' else 1
            work['gate_forward_wall_seconds']=.25 if self.mode=='C_seq' else 0.
            work['gate_forward_peak_factor_buffer_bytes']=128 if self.mode=='C_seq' else 0
            k=len(self.kwargs['support_ids'])//len(self.classes)
            ops=[dict(operation='ridge',audit={})]
            if k>1:
                ops.append(dict(operation='support_metric_step',audit={}))
                work['support_metric_step_calls']=1
            return dict(schema=evaluate.FROZEN_CONFIG['schema'],method=evaluate.METHOD,status='COMPLETED',
                actual_work=work,operation_audits=ops,nominal_parameter_count=5,effective_parameter_rank=5,
                trainable_parameter_count=5 if k>1 else 0,actual_updated_coordinate_count=0,
                optimizer_steps=0,trial_count=0,trials=[],final_head_complete=True,
                resident_numeric_state_bytes=64,deployment_numeric_state_bytes=64,
                deployment_state_scope='SYNTHETIC_CURRENT_FULL_STATE',fit_seconds=.125)
        def score(self,**features):
            out=np.zeros((len(features['z_id']),len(self.classes)))
            for i,label in enumerate(features['z_id'][:,0].astype(int)-1):
                name='c'+str(label)
                if name in self.classes:out[i,self.classes.index(name)]=2.
            if self.mode=='C_seq':out[:,:6]=0.
            return out
        def score_with_audit(self,**features):
            assert len(features['z_id'])==1
            return self.score(**features),dict(actual_work=all_work(),operation_audits=[],
                single_record_all_registered_classes=True,score_seconds=.01)
        def predict(self,**features):
            assert len(features['z_id'])==1
            label=int(features['z_id'][0,0])-1
            if wrong_old_prediction and label<6:label=(label+1)%6
            return np.asarray(['c'+str(label)])
    def prepare(**kwargs):
        assert kwargs['prototype_frame'].classes==OLD
        prepared.append(kwargs)
        if kwargs['inherited'] is not None:
            assert kwargs['inherited'].mode=='B'
            inherited.append(kwargs['inherited'])
        if kwargs.get('state_callback'):
            kwargs['state_callback']('prepared',dict(theta=np.zeros(5),Q=kwargs['prototype_frame'].Q))
        obj=SimpleNamespace(kwargs=kwargs)
        obj.audit_dict=lambda:dict(status='PREPARED',actual_work=all_work(),operation_audits=[],
            nominal_parameter_count=5,context=kwargs['context'])
        return obj
    def fit(prep,**kwargs):
        assert set(kwargs)<=set(('mode','log_callback','state_callback'))
        if fail_c and kwargs['mode']=='C_seq':
            values=dict(partial=np.asarray([np.nan,np.inf]))
            ref=kwargs['state_callback']('failure',values) if kwargs.get('state_callback') else None
            exc=RuntimeError('synthetic second-stage failure');exc.code='SYNTHETIC_C_FAILURE'
            work=all_work();work['gate_forward_factorization_attempts']=1
            work['gate_forward_wall_seconds']=.125
            exc.audit=dict(status='TECHNICAL_FAILURE',actual_work=work,operation_audits=[],
                failure_state_ref=ref);exc.arrays=values
            raise exc
        state=State(prep.kwargs,kwargs['mode']);fit_states.append(state)
        ref=kwargs['state_callback']('final',dict(theta=np.zeros(5))) if kwargs.get('state_callback') else None
        if kwargs.get('log_callback'):
            kwargs['log_callback'](dict(event='SUPPORT_METRIC_FINAL',schema=evaluate.FROZEN_CONFIG['schema'],
                audit=state.audit_dict(),state_ref=ref))
        return state
    if not real_baseline:monkeypatch.setattr(evaluate,'fit_branch_local_ridge',lambda **kwargs:State(kwargs))
    monkeypatch.setattr(evaluate,'prepare_support_metric_joint_training',prepare)
    monkeypatch.setattr(evaluate,'fit_support_metric_joint_local_ridge',fit)
    return dict(inherited=inherited,fit_states=fit_states,prepared=prepared)


class Ground:
    classes=tuple(reversed(OLD))
    metadata=SimpleNamespace(feature_contract=object())
    def __init__(self):self.inputs=[]
    def score(self,*,z_id,feature_contract):
        assert z_id.dtype==torch.float32 and z_id.shape==(1,160)
        assert feature_contract is self.metadata.feature_contract
        self.inputs.append(z_id.clone());values=torch.zeros((1,6),dtype=torch.float32)
        values[0,self.classes.index('c'+str(int(z_id[0,0])-1))]=3.
        return values


def test_real_production_probe_native_A_actual_core_and_archive_callbacks(tmp_path,monkeypatch):
    frozen=[];events=[];archive=evaluate.StateArchive(tmp_path);real_assess=evaluate.assess_paths
    def assess(evidence,old):
        latest=frozen[-5:]
        assert {row['stream'] for row in latest}=={'A','R0_B','R0_C','R_SUPPORT_METRIC_seq_B','R_SUPPORT_METRIC_seq_C'}
        assert all(row['status']=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN' and 'held_labels' not in row for row in latest)
        return real_assess(evidence,old)
    monkeypatch.setattr(evaluate,'assess_paths',assess)
    result=evaluate.probe_support_metric_joint(**synthetic(zero=True),**RESOURCES,prototype_frame=frame(),
        context=dict(run_id='synthetic',row_id='row',split_id='split'),
        ground_head=real_ground(),state_callback=archive,prediction_callback=frozen.append,event_callback=events.append)
    manifest=archive.finalize('COMPLETE')
    assert result['sequence_paths']==6 and result['candidate_stage_count']==12
    assert result['final_candidate_head_fit_count']==12
    assert result['work_aggregation']=='SUM/MAX' and result['nominal_adapter_parameter_count']==5
    assert len(frozen)==30 and manifest['schema']=='d92_support_metric_joint_state_archive_v1'
    assert manifest['file_count']>0 and events
    assert result['structural_predict_actual_work'] is None
    for row in frozen:
        assert row['outer_scope'] in ('support_oof','support_oneshot_proxy')
        if row['stream']=='A':
            assert row['scope']=='CURRENT_LEGAL_OLD_HELD_ONLY_ORIGINAL_SIX_CLASS_COMPETITION'
            assert row['score_dtype']=='float32' and row['predictions']==['c5']*len(row['physical_ids'])
    for path in result['folds']+result['oneshot_proxy']['trials']:
        assert path['ground_A']['physical_ids']==path['b_ids']
        for name in evaluate.PATHS:
            assert path['paths'][name]['metrics']['A_old_accuracy'] is not None
        for prep in path['preparations']:
            assert set(prep['training_physical_ids']).isdisjoint(path['c_ids'])
    assert result['actual_work']['ridge_calls']>0
    assert result['row_basis_construction_count']==result['basis_calls']==1
    assert result['row_basis_actual_work']['basis_calls']==1
    assert result['preparation_actual_work']['basis_binding_calls']==result['candidate_preparation_count']
    assert result['preparation_actual_work']['basis_binding_physical_gram_evaluation_count']==result['candidate_preparation_count']
    assert result['ggn_step_count']==result['support_metric_step_calls']
    assert result['ggn_parameter_direction_count']==result['row_basis_rank']*result['support_metric_step_calls']
    assert json.loads((tmp_path/'basis_certificate.json').read_text(encoding='utf-8'))['rank']==result['row_basis_rank']
    for stage in events:
        if stage['event'].endswith('INITIAL'):
            assert len(stage['class_ce_means'])==len(stage['class_ce_counts'])
            assert stage['class_weights'] and stage['effective_parameter_rank']==result['row_basis_rank']
    finals=[ref for ref in archive.files if ref['key']=='final']
    assert len(finals)==12
    for ref in finals:
        with np.load(tmp_path/ref['path'],allow_pickle=False) as values:
            assert values['theta'].shape==(5,)
    for field in ('actual_work','preparation_actual_work','stage_actual_work','score_actual_work'):
        assert set(result[field])==evaluate.WORK_KEYS


def test_synthetic_outer_pairing_actual_B_lineage_single_record_and_SUM_MAX(tmp_path,monkeypatch):
    observed=fake_models(monkeypatch);ground=Ground();frozen=[];archive=evaluate.StateArchive(tmp_path)
    result=evaluate.probe_support_metric_joint(**synthetic(),**RESOURCES,prototype_frame=frame(),
        ground_head=ground,state_callback=archive,prediction_callback=frozen.append)
    archive.finalize('COMPLETE')
    assert len(observed['inherited'])==6 and len({id(value) for value in observed['inherited']})==6
    assert result['candidate_preparation_count']==result['candidate_stage_count']==12
    assert result['gate_forward_wall_seconds']==1.5
    assert result['gate_forward_peak_factor_buffer_bytes']==128
    assert result['ridge_calls']==18
    assert result['ggn_step_count']==6 and result['ggn_parameter_direction_count']==30
    assert result['structural_predict_evaluation_count']==result['structural_predict_physical_count']
    assert len(frozen)==30 and all('held_labels' not in row for row in frozen)
    paths=result['folds']+result['oneshot_proxy']['trials']
    for prep,kwargs in zip([p for path in paths for p in path['preparations']],observed['prepared']):
        assert set(kwargs['support_ids'])==set(prep['training_physical_ids'])
        assert 'held_labels' not in kwargs
    assert len({id(kw['support_metric_basis']) for kw in observed['prepared']})==1
    assert result['row_basis_construction_count']==result['basis_calls']==1
    assert result['row_basis_actual_work']['basis_calls']==1
    assert result['preparation_actual_work']['basis_calls']==0
    for path in paths:
        assert set(path['b_training_ids']).isdisjoint(path['b_ids'])
        assert set(path['c_training_ids']).isdisjoint(path['c_ids'])
        assert path['paths']['R_SUPPORT_METRIC_seq']['metrics']['old_order_change']==0.
    assert all(int(vector[0,0])<=6 for vector in ground.inputs)


def test_nonlexical_registry_uses_real_R0_canonical_fit_and_scores(monkeypatch):
    fake_models(monkeypatch,real_baseline=True)
    args=synthetic(k=2,new=2)
    args['classes']=['c5','c0','c2','c1','c4','c3','c10','c6']
    for name in evaluate.BRANCHES:
        args[name][:,0]=[int(args['classes'][int(y)][1:])+1 for y in args['support_labels']]
    result=evaluate.probe_support_metric_joint(**args,**RESOURCES,prototype_frame=frame())
    assert result['classes']==sorted(args['classes'])
    labels={pid:args['classes'][int(y)] for pid,y in zip(args['support_ids'],args['support_labels'])}
    lookup={pid:i for i,pid in enumerate(args['support_ids'])}
    for path in result['folds']+result['oneshot_proxy']['trials']:
        for side in ('b','c'):
            registry=sorted(args['old_classes']) if side=='b' else sorted(args['classes'])
            trainids=path[side+'_training_ids'];heldids=path[side+'_ids']
            train=[lookup[pid] for pid in trainids];held=[lookup[pid] for pid in heldids]
            actual=evaluate.fit_branch_local_ridge(**{name:args[name][train] for name in evaluate.BRANCHES},
                support_labels=np.asarray([registry.index(labels[pid]) for pid in trainids],dtype=np.int64),
                support_ids=trainids,classes=registry,old_classes=sorted(args['old_classes']),arm='local_ridge')
            assert tuple(actual.classes)==tuple(registry)
            expected=actual.score(**{name:args[name][held] for name in evaluate.BRANCHES})
            np.testing.assert_allclose(path['paths']['R0'][side+'_scores'],expected,atol=2e-12)


@pytest.mark.parametrize('k,new',[(1,2),(1,0),(3,0)])
def test_k1_NA_and_new0_exact_B_no_extra_registration(k,new,monkeypatch):
    observed=fake_models(monkeypatch);ground=Ground();frozen=[]
    result=evaluate.probe_support_metric_joint(**synthetic(k,new),**RESOURCES,prototype_frame=frame(),
        ground_head=ground,prediction_callback=frozen.append)
    if k==1:
        assert result['full_support'] is not None and result['oof'] is None
        assert result['oneshot_proxy'] is None and not frozen and not ground.inputs
        assert result['ggn_step_count']==0
        assert all(value is None for path in result['full_support']['paths'].values() for value in path['metrics'].values())
    if new==0:
        assert not observed['inherited']
        assert all(state.mode=='B' for state in observed['fit_states'])
        assert result['candidate_stage_count']==result['sequence_paths']
        paths=[result['full_support']] if k==1 else result['folds']+result['oneshot_proxy']['trials']
        for path in paths:
            assert path['c_reuses_b_candidates']
            assert path['paths']['R_SUPPORT_METRIC_seq']['b_scores']==path['paths']['R_SUPPORT_METRIC_seq']['c_scores']
            assert path['paths']['R_SUPPORT_METRIC_seq']['metrics']['C_new_accuracy'] is None
            assert path['paths']['R_SUPPORT_METRIC_seq']['metrics']['C_h'] is None
            assert path['paths']['R_SUPPORT_METRIC_seq']['metrics']['C_abs_new_old_gap'] is None


def test_failure_callback_preserves_nonfinite_partial_and_completed_cost(tmp_path,monkeypatch):
    fake_models(monkeypatch,fail_c=True);archive=evaluate.StateArchive(tmp_path)
    with pytest.raises(RuntimeError,match='second-stage') as failure:
        evaluate.probe_support_metric_joint(**synthetic(),**RESOURCES,prototype_frame=frame(),state_callback=archive)
    context=failure.value.registration_context
    assert failure.value.code=='SYNTHETIC_C_FAILURE'
    assert context['workload_complete'] is False and context['counters']['candidate_stage_count']==1
    assert context['failed_actual_work']['gate_forward_factorization_attempts']==1
    assert context['failed_actual_work']['gate_forward_wall_seconds']==.125
    ref=context['failed_numeric_state_ref']
    assert ref['failed_numeric_state'] and ref['arrays']['partial']['nonfinite_count']==2
    with np.load(tmp_path/ref['path'],allow_pickle=False) as stored:
        assert np.isnan(stored['partial'][0]) and np.isinf(stored['partial'][1])
    with pytest.raises(ValueError,match='Nonfinite archived successful'):
        archive('success',dict(value=np.asarray([np.nan])))
    assert archive.finalize('INCOMPLETE')['status']=='INCOMPLETE'


def test_public_C_old_winner_mismatch_is_rejected_after_predictions_fixed(monkeypatch):
    fake_models(monkeypatch,wrong_old_prediction=True);frozen=[]
    with pytest.raises(ValueError,match='old winner differs'):
        evaluate.probe_support_metric_joint(**synthetic(),**RESOURCES,prototype_frame=frame(),prediction_callback=frozen.append)
    assert len(frozen)==4 and all(row['status']=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN' for row in frozen)


@pytest.mark.parametrize('bad',[
    dict(max_newton_iterations=True,max_line_search_trials=64,max_factor_buffer_bytes=1),
    dict(max_newton_iterations=0,max_line_search_trials=64,max_factor_buffer_bytes=1),
    dict(max_newton_iterations=100,max_line_search_trials=64.0,max_factor_buffer_bytes=1),
    dict(RESOURCES,extra=1),
])
def test_explicit_resource_types_and_keys_rejected(bad):
    with pytest.raises(ValueError,match='positive integer'):evaluate.validate_resources(bad)


@pytest.mark.parametrize('missing',['max_integer_bits','max_fraction_operations','max_secular_iterations'])
def test_each_new_resource_is_required_without_default(missing):
    values=dict(RESOURCES);values.pop(missing)
    with pytest.raises(ValueError,match='positive integer'):evaluate.validate_resources(values)


def test_missing_frame_rejected_before_fitting(monkeypatch):
    monkeypatch.setattr(evaluate.interaction,'_prepare',lambda *a,**kw:pytest.fail('Frame check must precede fitting'))
    with pytest.raises(ValueError,match='prototype frame'):
        evaluate.probe_support_metric_joint(**synthetic(),**RESOURCES)


def selection_and_tasks():
    classes=list(OLD)+['c'+str(i) for i in range(6,26)]
    selected=[];tasks=[];all_ids=[]
    for receiver,scenario in (('rx-synthetic-a',evaluate.SCENARIOS[0]),('rx-synthetic-b',evaluate.SCENARIOS[1])):
        for k in (1,5,10,20):
            old_ids=[f'{receiver}_{k}_old_{c}_{s}' for c in range(6) for s in range(k)]
            for new in (0,2,5,10,20):
                registry=classes[:6+new];ids=old_ids+[f'{receiver}_{k}_new_{c}_{s}' for c in range(new) for s in range(k)]
                labels=np.repeat(np.arange(len(registry)),k).tolist()
                identity=dict(split_id=f'{receiver}_{k}_{new}',receiver=receiver,scenario=scenario,k=k,
                    support_seed=7,new_count=new,registered_classes=registry)
                selected.append(identity);split=dict(identity,support_ids=ids,support_labels=labels)
                tasks.append((split,None,None));all_ids.extend(ids)
    lookup={pid:i for i,pid in enumerate(sorted(set(all_ids)))}
    actual=[(split,np.asarray([lookup[pid] for pid in split['support_ids']]),np.asarray(split['support_labels']))
        for split,_,_ in tasks]
    selection=dict(receiver_scenes=[['rx-synthetic-a',evaluate.SCENARIOS[0]],['rx-synthetic-b',evaluate.SCENARIOS[1]]],
        support_seed=7,ks=[1,5,10,20],new_counts=[0,2,5,10,20],splits=selected)
    return selection,actual,lookup


def test_production_selector_fixed_matrix_old_physical_pairing_and_extra_parent():
    selection,tasks,_=selection_and_tasks()
    extra=dict(tasks[0][0],split_id='extra_unselected_parent')
    chosen=evaluate.selected_tasks(tasks+[(extra,tasks[0][1],tasks[0][2])],selection,list(OLD))
    assert len(chosen)==40 and all(item[0]['split_id']!='extra_unselected_parent' for item in chosen)
    changed=deepcopy(tasks)
    changed[1][0]['support_ids'][0]='changed_old_physical_record'
    with pytest.raises(ValueError,match='old physical support'):
        evaluate.selected_tasks(changed,selection,list(OLD))


def write_summary(root):
    root.mkdir()
    payload=dict(schema=np.asarray(codec.SCHEMA),feature_schema=np.asarray(codec.FEATURE_SCHEMA),
        residual_rank=np.asarray(3,dtype=np.int16),center_domain_handle=np.asarray('ground-0'),
        domain_registry=np.asarray(['ground-0','ground-1']),residual_domain_registry=np.asarray(['ground-1']),
        class_registry=np.asarray(OLD),core_q=np.eye(6,160,dtype=np.int8)*50,
        core_scale=np.full(6,.01,dtype=np.float16),residual_basis_q=np.zeros((6,3,160),dtype=np.int8),
        residual_basis_scale=np.ones((6,3),dtype=np.float16),residual_coeff_q=np.zeros((1,6,3),dtype=np.int8),
        residual_coeff_scale=np.ones((1,6),dtype=np.float16),radius_q=np.zeros((2,6),dtype=np.int8),
        radius_scale=np.ones(6,dtype=np.float16))
    manifest=dict(schema=codec.SCHEMA,metadata_schema='cvs.matched.ground.v2',checkpoint_sha256=SHA,
        source_prototype_artifact_sha256='c'*64,provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L',
        formal_phase2_eligible=True,feature_dim=160,residual_rank=3,radius_histogram_bins=4096,
        member_allowlist=[codec.NPZ_NAME],npz_member_allowlist=sorted(codec.ALLOWED_NPZ_MEMBERS),
        resource_audit=codec._numeric_resource_audit(payload),source_role='L_s',target_access=False,
        component_state='CURRENT_MATCHED_SOURCE_ONLY_V2',historical_outer_signature_claim=False)
    np.savez_compressed(root/codec.NPZ_NAME,**payload)
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')


@pytest.mark.parametrize('deployed',[False,True])
def test_row_entry_only_selected_support_and_real_summary_reader(tmp_path,monkeypatch,deployed):
    selection,tasks,lookup=selection_and_tasks();count=len(lookup)
    arrays={name:np.zeros((count+1,d),dtype=np.float32)
        for name,d in zip(evaluate.BRANCHES,(160,96,160,160,160))}
    arrays['z_id'][:,0]=1;arrays['z_id'][-1,0]=999
    extra=dict(tasks[0][0],split_id='extra_unselected_parent',support_ids=['extra'])
    monkeypatch.setattr(evaluate,'load_support',lambda **kwargs:(arrays,tasks+[(extra,np.asarray([count]),np.asarray([0]))],
        list(OLD),dict(feature_array_bytes=10,feature_file_bytes=12),{},dict(scope='SYNTHETIC_SUPPORT_ONLY')))
    summary=tmp_path/'summary';write_summary(summary)
    capsule=tmp_path/'capsule';capsule.mkdir()
    (capsule/'manifest.json').write_text(json.dumps(dict(channel=evaluate.CHANNEL,scenarios=evaluate.SCENARIOS)),encoding='utf-8')
    seen=[];native_read=evaluate.read
    def allowed_metadata(path):
        assert Path(path).name not in ('truth.json','query.json','received.npz')
        return native_read(path)
    monkeypatch.setattr(evaluate,'read',allowed_metadata)
    def probe(**kwargs):
        assert not np.any(kwargs['z_id'][:,0]==999)
        assert kwargs['prototype_frame'].classes==OLD
        seen.extend(kwargs['support_ids'])
        result=dict.fromkeys(evaluate.COUNTERS[4:],0)
        result.update(support_count=len(kwargs['support_ids']),old_class_count=6,
            new_class_count=len(kwargs['classes'])-6,fold_count=0,fit_seconds=0.,persistent_state_bytes=0,
            heldout_unavailable_reason='SYNTHETIC_LOADER_BOUNDARY_ONLY',oof=None,oneshot_proxy=None,
            full_support={},actual_work=all_work(),row_basis_actual_work=all_work(),
            row_basis_ref=kwargs['row_basis_owner']['row_basis_ref'],row_basis_rank=kwargs['support_metric_basis'].rank,
            basis_certificate=kwargs['row_basis_owner']['basis_certificate'],
            preparation_actual_work=all_work(),stage_actual_work=all_work(),
            score_actual_work=all_work(),structural_predict_work_unavailable_reason='SYNTHETIC')
        return result
    monkeypatch.setattr(evaluate,'probe_support_metric_joint',probe)
    config=dict(algorithm=deepcopy(evaluate.FROZEN_CONFIG),producer_matrix=dict(synthetic=True),
        selection=selection,support_metric_resources=RESOURCES)
    out=tmp_path/'out'
    marker=evaluate.evaluate(support_features='synthetic-support-cache',capsule=capsule,output=out,config=config,
        expected_capsule_id='synthetic',expected_checkpoint_sha256=SHA,expected_model_seed=1,
        run_id='synthetic',row_id='row',release_commit=COMMIT,ground_summary=summary,
        ground_summary_already_deployed=deployed)
    assert marker['episodes']==40 and 'extra' not in seen
    assert marker['schema']=='d92_support_metric_joint_support_probe_v1' and marker['work_aggregation']=='SUM/MAX'
    assert marker['ground_geometry_binding']['ground_payload_audit']['incremental_transfer_bytes']==(
        0 if deployed else sum(p.stat().st_size for p in summary.iterdir()))
    assert marker['payload_audit']['native_wire_bytes'] is None
    assert marker['selected_support_physical_ids']=={split['split_id']:split['support_ids'] for split,_,_ in tasks}
    assert marker['structural_predict_actual_work'] is None
    assert marker['row_basis_construction_count']==marker['basis_calls']==1
    assert marker['row_basis_actual_work']['basis_calls']==1
    assert marker['preparation_actual_work']['basis_calls']==0
    certificate=marker['basis_certificate'];stored=(out/certificate['path']).read_bytes()
    assert len(stored)==certificate['utf8_bytes']==certificate['file_bytes']
    assert json.loads(stored)['rank']==marker['row_basis_rank']
    assert (out/'row_basis.json').is_file()
    assert set(marker['actual_work_aggregation'])==evaluate.WORK_KEYS
    for key in evaluate.WORK_KEYS:
        phases=[marker[field][key] for field in ('row_basis_actual_work','preparation_actual_work','stage_actual_work','score_actual_work')]
        expected=max(phases) if key in evaluate.WORK_MAX_KEYS else sum(phases)
        assert marker['actual_work'][key]==marker[key]==expected
        assert marker['actual_work_aggregation'][key]==('MAX' if key in evaluate.WORK_MAX_KEYS else 'SUM')
    for name in ('training.log','training_events.jsonl','training_events_compact.jsonl','training_events_compact.csv',
        'fit_stages.jsonl','fit_stages.csv','fixed_predictions.jsonl','state_manifest.json','artifact_manifest.json'):
        assert (out/name).is_file()
    with pytest.raises(FileExistsError):
        evaluate.evaluate(support_features='unused',capsule='unused',output=out,config=config,
            expected_capsule_id='unused',expected_checkpoint_sha256=SHA,expected_model_seed=1,
            run_id='synthetic',row_id='row',release_commit=COMMIT,ground_summary='unused',
            ground_summary_already_deployed=deployed)


@pytest.mark.parametrize('text,expected',[('true',True),('false',False)])
def test_cli_explicit_summary_deployment_boolean(monkeypatch,text,expected):
    captured=[]
    monkeypatch.setattr(evaluate,'evaluate',lambda **kwargs:captured.append(kwargs))
    monkeypatch.setattr(evaluate,'read',lambda path:{})
    argv=['evaluate']
    for flag in ('support-features','capsule','output','config','expected-capsule-id',
        'expected-checkpoint-sha256','run-id','row-id','release-commit','ground-summary'):
        argv.extend(['--'+flag,'synthetic'])
    argv.extend(['--expected-model-seed','1','--ground-summary-already-deployed',text])
    monkeypatch.setattr(sys,'argv',argv);evaluate.main()
    assert captured[0]['ground_summary_already_deployed'] is expected


def test_compact_record_keeps_nested_full_state_descriptors_and_GGN_cost():
    ref=dict(path='state_arrays/00000000.npz',namespace='{"state":"B_SUPPORT_METRIC"}',
        arrays=dict(theta=dict(shape=[5],dtype='float64',nbytes=40)),file_bytes=99)
    row=dict(event='SUPPORT_METRIC_FINAL',effective_parameter_rank=2,class_ce_means=[.4,.6],class_weights=[.3,.7],
        metric_fisher=[[1.,0.],[0.,2.]],step_audit=dict(status='COMPLETE',fisher_constructions_completed=1),
        audit=dict(final_state_ref=ref,actual_work=all_work(),
        preparation=dict(folds=[dict(prior_ref=ref)])),state_ref=ref)
    compact=evaluate.compact_event(row)
    assert compact['state_ref']==ref and compact['audit']['final_state_ref']==ref
    assert compact['audit']['preparation']['folds'][0]['prior_ref']==ref
    assert compact['audit']['actual_work']==all_work()
    assert compact['class_ce_means']==row['class_ce_means'] and compact['class_weights']==row['class_weights']
    assert compact['metric_fisher']==row['metric_fisher'] and compact['step_audit']==row['step_audit']


def test_actual_work_requires_complete_declared_ledger_and_MAX_peak():
    target=evaluate._empty_work();source=all_work()
    source['gate_forward_wall_seconds']=.25;source['gate_forward_peak_factor_buffer_bytes']=128
    evaluate._work_merge(target,source);evaluate._work_merge(target,source)
    assert target['gate_forward_wall_seconds']==.5
    assert target['gate_forward_peak_factor_buffer_bytes']==128
    del source['ridge_calls']
    with pytest.raises(ValueError,match='Complete declared actual'):
        evaluate._work_merge(target,source)


def test_row_basis_failure_preserves_exact_partial_and_numeric_before_any_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate,'fit_branch_local_ridge',lambda **kw:pytest.fail('Basis failure must precede any fitting'))
    archive=evaluate.StateArchive(tmp_path);limits=dict(RESOURCES,max_fraction_operations=1)
    with pytest.raises(evaluate.candidate_core.SupportMetricJointFailure) as caught:
        evaluate.probe_support_metric_joint(**synthetic(k=1),**limits,prototype_frame=frame(),state_callback=archive)
    audit=caught.value.audit_dict()
    assert audit['failed_operation']=='ROW_BASIS_CONSTRUCTION'
    assert audit['actual_work']['basis_calls']==1
    assert audit['actual_work']['basis_fraction_operations_attempted']>1
    assert 'basis_failure_exact_state' in audit and audit['failure_state_ref'] is not None
    with np.load(tmp_path/audit['failure_state_ref']['path'],allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['Q'],frame().Q)
    json.dumps(audit,allow_nan=False)
    assert archive.finalize('INCOMPLETE')['status']=='INCOMPLETE'


def test_certificate_ref_replaces_repeat_serialized_certificate_only():
    owner=dict(row_basis_ref=dict(path='state_arrays/00000000.npz'),
        basis_certificate=dict(path='basis_certificate.json',file_bytes=5,utf8_bytes=5))
    original=dict(preparation=dict(basis_certificate='exact data-only certificate',basis_audit=dict(wall_seconds=.2)),
        theta=[.1],actual_work=all_work())
    bound=evaluate._basis_refs(original,owner)
    assert 'basis_certificate' not in bound['preparation']
    assert bound['preparation']['basis_certificate_ref']==owner['basis_certificate']
    assert bound['preparation']['row_basis_ref']==owner['row_basis_ref']
    assert bound['preparation']['basis_audit']==original['preparation']['basis_audit']
    assert bound['actual_work']==all_work() and original['preparation']['basis_certificate'].startswith('exact')


def test_training_text_displays_actual_loss_GGN_parameters_and_unavailable_values():
    row=dict(event='SUPPORT_METRIC_FINAL',state='B_SUPPORT_METRIC',audit=dict(
        final_objective=dict(RMSCE=.5,loss_total=.5,loss_proximal=0.,objective_seconds=.25),
        gradient_norm=.125,quadratic_multiplier=2.,fit_seconds=.75,resident_numeric_state_bytes=40,
        trainable_parameter_count=5,actual_updated_coordinate_count=2))
    text=evaluate.training_text(row)
    for token in ('RMSCE=0.5','loss_total=0.5','loss_proximal=0.0','damping='+evaluate.FROZEN_CONFIG['damping'],
        'nominal_parameters=5','gradient=0.125','GGN_multiplier=2.0',
        'actual_updated_coordinates=2','source_validation=N/A'):
        assert token in text
    assert 'accepted=N/A' in text
