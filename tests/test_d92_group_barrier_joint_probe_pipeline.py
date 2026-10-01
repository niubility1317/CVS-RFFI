"""Synthetic source-entry contracts only; never read an experiment artifact."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import numpy as np
import pytest
import torch

import evaluate_d92_group_barrier_joint_probe as evaluate
import run_d92_group_barrier_joint_probe as run
import preflight_d92_group_barrier_joint_probe as preflight
import publish_d92_group_barrier_joint_probe as publish

COMMIT='a'*40
RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160)


def literal():
    return run.read(ROOT/'configs/d92_group_barrier_joint_support_20261001.json')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_literal_matrix_same_source_seeds_and_physical_selection():
    s=literal();run.validate_spec(s)
    old=run.read(ROOT/'configs/d92_margin_joint_support_20261001.json')
    assert s['probe']['budget']['total']['exact']['episodes']==160
    assert len(s['rows'])==4 and s['probe']['group_barrier_resources']==RESOURCES
    assert s['code']['commit']=='81985ed377ca2bbb1bc9e6295dbaf9cc208d242f'
    for co in s['probe']['cohorts']:
        assert s['probe']['cohorts'][co]['matrix']==old['probe']['cohorts'][co]['matrix']
        assert s['probe']['cohorts'][co]['selection']==old['probe']['cohorts'][co]['selection']
    for row,original in zip(s['rows'],old['rows']):
        assert row['seeds']==original['seeds']
        assert row['expected_checkpoint_sha256']==original['expected_checkpoint_sha256']
        assert row['support_features']==original['support_features']
    assert s['probe']['candidate']=='R_GROUP_BARRIER_seq' and s['probe']['controls']==['R0']
    assert 'qp_resources' not in s['probe']


@pytest.mark.parametrize('mutation',[lambda s:s['probe']['algorithm'].update(proximal_coefficient=1.),
    lambda s:s['probe']['group_barrier_resources'].update(max_newton_iterations=True),
    lambda s:s['rows'][2].update(expected_checkpoint_sha256='0'*64),
    lambda s:s['rows'][2].update(ground_packet='/another/packet'),
    lambda s:s['permissions'].update(query_use='read query'),
    lambda s:s['probe']['cohorts']['rx3']['selection']['splits'].pop()])
def test_frozen_contract_mutations_rejected(mutation):
    s=literal();mutation(s)
    with pytest.raises((ValueError,KeyError)):run.validate_spec(s)


def test_explicit_runtime_and_per_row_embedded_config_command():
    s=literal();row=s['rows'][0];args=run.command(s,row,COMMIT)
    assert args[args.index('--release-commit')+1]==COMMIT
    assert args[args.index('--config')+1]==row['output_root']+'/resolved_config.json'
    assert args[args.index('--ground-packet')+1]==row['ground_packet']
    assert '--query' not in args and '--checkpoint' not in args
    with pytest.raises(ValueError):run.command(s,row,'bad')


def test_fixed_crossproduct_cannot_remove_row_or_parent_and_rebudget():
    s=literal();s['rows'].pop();s['probe']['budget']=run.budget_for_spec(s)
    with pytest.raises(ValueError,match='four-row matrix'):run.validate_spec(s)
    s=literal();co=s['probe']['cohorts']['rx3'];co['selection']['receiver_scenes']=co['selection']['receiver_scenes'][:1]
    rx,scene=co['selection']['receiver_scenes'][0]
    co['selection']['splits']=[v for v in co['selection']['splits'] if (v['receiver'],v['scenario'])==(rx,scene)]
    co['selected_split_count']=20;co['evaluator_config']=run.evaluator_config(s,'rx3');s['probe']['budget']=run.budget_for_spec(s)
    with pytest.raises(ValueError,match='forty-parent'):run.validate_spec(s)


def synthetic(k=3,new=2):
    classes=['c'+str(i) for i in range(6+new)];labels=np.repeat(np.arange(len(classes)),k)
    n=len(labels);ids=[f'physical_{i:04d}' for i in range(n)]
    raw={name:np.zeros((n,d),dtype=np.float32) for name,d in zip(evaluate.BRANCHES,(160,96,160,160,160))}
    for values in raw.values():values[:,0]=labels+1;values[:,1]=np.arange(n)+1
    return dict(raw,support_labels=labels,support_ids=ids,classes=classes,old_classes=classes[:6])


def fake_models(monkeypatch,*,fail_c=False,rounded_c=True):
    lineage=[]
    class State:
        def __init__(self,kwargs,mode='baseline'):
            self.kwargs=kwargs;self.classes=list(kwargs['classes']);self.mode=mode
            self.support_background=np.zeros((len(kwargs['support_ids']),1));self.support_auxiliary=self.support_background.copy()
            self.alpha=np.zeros((len(kwargs['support_ids']),len(self.classes)));self.reference_kernel=np.zeros((1,1))
            self.reference_self=0.;self.center_mean=np.zeros(1);self.center_grand=0.;self.bandwidth_tau=1.;self.trace_scale=1.
        def audit_dict(self):
            if self.mode=='baseline':return dict(persistent_state_bytes=16,final_fit=dict(factorization_calls=1,effective_degrees_of_freedom_extra_triangular_solves=0))
            return dict(schema='native_B' if self.mode=='B' else evaluate.SCHEMA,method='native_B' if self.mode=='B' else evaluate.METHOD,
                resident_numeric_state_bytes=64,deployment_numeric_state_bytes=32,optimizer_steps=0,ajlr_stage_count=1,
                final_head_fit_count=1,new_ridge_factorization_count=1 if self.mode=='C_seq' else 0,
                group_gate_forward_factorization_attempts=2 if self.mode=='C_seq' else 0,
                group_gate_adjoint_factorization_attempts=1 if self.mode=='C_seq' else 0,
                group_gate_forward_wall_seconds=.25 if self.mode=='C_seq' else 0,
                gate_peak_factor_buffer_bytes=128 if self.mode=='C_seq' else 0)
        def score(self,**features):
            n=len(features['z_id']);out=np.zeros((n,len(self.classes)))
            for i,y in enumerate(features['z_id'][:,0].astype(int)-1):
                cls='c'+str(y)
                if cls in self.classes:out[i,self.classes.index(cls)]=2.
            if self.mode=='C_seq' and rounded_c:out[:,:6]=0.  # structural predict must retain raw B ordering.
            return out
        def score_with_audit(self,**features):return self.score(**features),dict(score_physical_count=len(features['z_id']))
        def predict(self,**features):return np.asarray(['c'+str(y) for y in features['z_id'][:,0].astype(int)-1])
    def base(**kwargs):return State(kwargs)
    def prep(**kwargs):
        if kwargs['inherited'] is not None:
            assert kwargs['inherited'].mode=='B';lineage.append(kwargs['inherited'])
        p=SimpleNamespace(kwargs=kwargs)
        p.audit_dict=lambda:dict(ajlr_preparation_count=1,no_information=True)
        if kwargs.get('state_callback'):kwargs['state_callback']('prepared',dict(scalar=np.asarray(1.)))
        return p
    def fit(p,**kwargs):
        mode=kwargs['mode']
        if fail_c and mode=='C_seq':
            # Invoke the production callback adapter and real StateArchive, without
            # a mocked archiver or an upstream recorder that rejects nonfinite first.
            ref=kwargs['state_callback']('failure',dict(partial=np.asarray([np.nan,np.inf]))) if kwargs.get('state_callback') else None
            error=RuntimeError('synthetic partial gate failure')
            error.audit=dict(group_gate_forward_factorization_attempts=1,group_gate_forward_wall_seconds=.125,status='TECHNICAL_FAILURE',failure_state_ref=ref)
            error.arrays=dict(partial=np.asarray([np.nan]))
            raise error
        state=State(p.kwargs,mode)
        if kwargs.get('state_callback'):kwargs['state_callback']('final',dict(U=np.zeros((2,2))))
        if kwargs.get('log_callback'):kwargs['log_callback'](dict(event='FINAL',schema=state.audit_dict()['schema'],audit=state.audit_dict()))
        return state
    monkeypatch.setattr(evaluate,'fit_branch_local_ridge',base)
    monkeypatch.setattr(evaluate,'prepare_group_barrier_joint_training',prep)
    monkeypatch.setattr(evaluate,'fit_group_barrier_joint_local_ridge',fit)
    return lineage


class Ground:
    classes=('c5','c4','c3','c2','c1','c0')
    metadata=SimpleNamespace(feature_contract=object())
    def __init__(self):self.inputs=[]
    def score(self,*,z_id,feature_contract):
        assert z_id.dtype==torch.float32 and z_id.shape==(1,160) and feature_contract is self.metadata.feature_contract
        self.inputs.append(z_id.clone());scores=torch.zeros((1,6),dtype=torch.float32)
        scores[0,self.classes.index('c'+str(int(z_id[0,0])-1))]=3.
        return scores


def test_synthetic_full_paths_actual_parent_inheritance_structured_decisions_and_native_A(tmp_path,monkeypatch):
    lineage=fake_models(monkeypatch);ground=Ground();frozen=[];events=[];logs=[]
    archive=evaluate.StateArchive(tmp_path)
    def fixed(row):
        assert 'held_labels' not in row;frozen.append(deepcopy(row))
        with (tmp_path/'fixed.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
    result=evaluate.probe_group_barrier_joint(**synthetic(),**RESOURCES,context=dict(run_id='synthetic',row_id='row',split_id='split'),
        ground_head=ground,state_callback=archive,prediction_callback=fixed,event_callback=events.append,log_callback=logs.append)
    manifest=archive.finalize('COMPLETE')
    assert result['sequence_paths']==6 and len(lineage)==6
    assert len({id(b) for b in lineage})==6
    assert result['candidate_stage_count']==12 and result['candidate_preparation_count']==12
    assert result['group_gate_forward_wall_seconds']==1.5 and result['gate_peak_factor_buffer_bytes']==128
    assert result['factorization_count']==36
    assert manifest['status']=='COMPLETE' and manifest['file_count']>0
    assert len(frozen)==30 and result['structural_predict_evaluation_count']==6
    for entry in result['folds']+result['oneshot_proxy']['trials']:
        assert entry['ground_A']['physical_ids']==entry['b_ids']
        assert entry['ground_A']['classes']==list(ground.classes)
        m=entry['paths']['R_GROUP_BARRIER_seq']['metrics']
        assert m['A_old_accuracy']==m['B_old_accuracy']==m['C_old_accuracy']==1.
        assert m['adaptation_gain_B_minus_A']==0. and m['old_order_change']==0.
    assert all(int(t[0,0])<=6 for t in ground.inputs)
    assert logs and events and any(r['executed_core_schema']=='native_B' for r in events)


@pytest.mark.parametrize('k,new',[(1,2),(1,0),(3,0)])
def test_k1_na_and_new0_exact_B_reuse(k,new,monkeypatch):
    lineage=fake_models(monkeypatch);ground=Ground();fixed=[]
    r=evaluate.probe_group_barrier_joint(**synthetic(k,new),**RESOURCES,ground_head=ground,prediction_callback=fixed.append)
    if k==1:
        assert r['full_support'] is not None and r['oof'] is None and r['oneshot_proxy'] is None
        assert not fixed and not ground.inputs
        assert all(v is None for p in r['full_support']['paths'].values() for v in p['metrics'].values())
    if new==0:
        assert not lineage and r['candidate_stage_count']==r['sequence_paths']
        if k>1:
            assert r['oof']['paths']['R_GROUP_BARRIER_seq']['metrics']['C_new_accuracy'] is None
            assert r['oof']['paths']['R_GROUP_BARRIER_seq']['metrics']['C_h'] is None


def test_partial_failure_retains_completed_B_and_actual_gate_prefix(tmp_path,monkeypatch):
    fake_models(monkeypatch,fail_c=True);archive=evaluate.StateArchive(tmp_path)
    with pytest.raises(RuntimeError,match='partial gate') as info:
        evaluate.probe_group_barrier_joint(**synthetic(),**RESOURCES,state_callback=archive)
    context=info.value.registration_context;manifest=archive.finalize('INCOMPLETE')
    assert context['workload_complete'] is False and context['counters']['candidate_stage_count']==1
    assert context['failed_fit_counters']['group_gate_forward_factorization_attempts']==1
    assert context['failed_fit_counters']['group_gate_forward_wall_seconds']==.125
    assert context['failed_numeric_state_ref']['failed_numeric_state'] is True
    assert context['failed_numeric_state_ref']['arrays']['partial']['nonfinite_count']==2
    with pytest.raises(ValueError,match='Nonfinite archived successful state'):
        archive('strict_success',dict(partial=np.asarray([np.nan])))
    assert manifest['status']=='INCOMPLETE'


def test_selected_support_only_extra_cache_records_never_reach_model(tmp_path,monkeypatch):
    s=literal();co=s['probe']['cohorts']['rx3'];selection=co['selection'];item=selection['splits'][0]
    split=dict(item,support_ids=['legal'],support_labels=[0]);extra=dict(split,split_id='NOT_SELECTED',support_ids=['extra'])
    # Exercise the production selector with a complete metadata matrix, then isolate one fake evaluator task.
    tasks=[]
    for ident in selection['splits']:
        c=dict(ident,support_ids=['same_old']+[ident['split_id']+'_new_'+str(j) for j in range(ident['new_count'])],support_labels=[0]+list(range(6,6+ident['new_count'])))
        tasks.append((c,np.asarray([0]),np.asarray([0])))
    chosen=run.selected_tasks(tasks+[(extra,np.asarray([1]),np.asarray([0]))],selection,selection['splits'][0]['registered_classes'][:6])
    assert len(chosen)==40 and all(c[0]['split_id']!='NOT_SELECTED' for c in chosen)
    arrays={name:np.ones((2,d),dtype=np.float32) for name,d in zip(evaluate.BRANCHES,(160,96,160,160,160))}
    arrays['z_id'][1,0]=999
    monkeypatch.setattr(evaluate,'load_support',lambda **kw:(arrays,[(split,np.asarray([0]),np.asarray([0])),(extra,np.asarray([1]),np.asarray([0]))],
        item['registered_classes'][:6],dict(feature_array_bytes=10,feature_file_bytes=12),{},{}))
    monkeypatch.setattr(evaluate,'selected_tasks',lambda tasks,selection,old:tasks[:1])
    monkeypatch.setattr(evaluate,'validate_selection',lambda selection:None)
    monkeypatch.setattr(evaluate,'ground_packet_binding',lambda *args:(None,dict(status='N/A',reason='synthetic')))
    monkeypatch.setattr(np,'load',lambda *args,**kw:pytest.fail('fake entry must not open query/cache numeric files'))
    native_read=evaluate.read
    def support_metadata_only(path):
        assert Path(path).name not in ('truth.json','query.json','received.npz','predictions.jsonl')
        return native_read(path)
    monkeypatch.setattr(evaluate,'read',support_metadata_only)
    manifest=tmp_path/'capsule/manifest.json';dump(manifest,dict(channel=run.CHANNEL,scenarios=run.SCENARIOS))
    seen=[]
    def fake_probe(**kw):
        seen.extend(kw['z_id'][:,0].tolist());assert kw['support_ids']==['legal']
        out=dict.fromkeys(evaluate.COUNTERS[4:],0)
        out.update(classes=item['registered_classes'],old_classes=item['registered_classes'][:6],support_count=1,old_class_count=6,
            new_class_count=0,fold_count=0,fit_seconds=0.,persistent_state_bytes=0,heldout_unavailable_reason='synthetic',
            oof=None,oneshot_proxy=None,full_support={},schema=evaluate.SCHEMA)
        return out
    monkeypatch.setattr(evaluate,'probe_group_barrier_joint',fake_probe)
    out=tmp_path/'result';cfg=run.evaluator_config(s,'rx3')
    evaluate.evaluate(support_features='synthetic',capsule=manifest.parent,output=out,config=cfg,
        expected_capsule_id=co['capsule_id'],expected_checkpoint_sha256='1'*64,expected_model_seed=1,
        run_id='synthetic',row_id='row',release_commit=COMMIT)
    assert seen==[1.] and (out/'training.log').is_file() and (out/'fixed_predictions.jsonl').is_file()
    assert run.read(out/'probe_complete.json')['selected_support_physical_ids']=={item['split_id']:['legal']}
    with pytest.raises(FileExistsError):
        evaluate.evaluate(support_features='synthetic',capsule='unused',output=out,config=cfg,expected_capsule_id='unused',
            expected_checkpoint_sha256='unused',expected_model_seed=1,run_id='synthetic',row_id='row',release_commit=COMMIT)


def test_supervisor_failed_row_does_not_retry_or_stop_healthy(tmp_path,monkeypatch):
    s=literal();s['execution']['remote_run_root']=str(tmp_path/'run');s['code']['cwd']=str(tmp_path)
    s['rows']=s['rows'][:2]
    for r in s['rows']:r['output_root']=str(tmp_path/'run'/r['row_id'])
    monkeypatch.setattr(run,'validate_spec',lambda s:s)
    launched=[]
    def launch(argv,log,cwd,started):
        row=argv[argv.index('--row-id')+1];launched.append(row);pid=100+len(launched);started(pid)
        Path(log).write_text('synthetic detailed log',encoding='utf-8')
        if row==s['rows'][0]['row_id']:raise RuntimeError('synthetic failure')
        return dict(pid=pid,returncode=0)
    def verified(path,s,row,**kw):return dict.fromkeys(evaluate.COUNTERS,0)
    monkeypatch.setattr(run,'verify_marker',verified)
    check=lambda s,**kw:dict(cache_bindings=[dict(row_id=r['row_id'],binding='VERIFIED') for r in s['rows']])
    with pytest.raises(RuntimeError,match='healthy rows retained'):run.run(s,COMMIT,launch_fn=launch,preflight_fn=check)
    terminal=run.read(tmp_path/'run/complete.json')
    assert len(launched)==2 and terminal['completed_rows']==1 and terminal['workload_complete'] is False
    assert terminal['runtime_commit']==COMMIT and terminal['automatic_retry'] is False
    assert (Path(s['rows'][1]['output_root'])/'resolved_config.json').is_file()
    with pytest.raises(ValueError,match='Exclusive run'):run.run(s,COMMIT,launch_fn=launch,preflight_fn=check)


def test_source_whitelist_embedded_scripts_and_exclusive_preflight(monkeypatch,tmp_path):
    s=literal();paths=publish.release_paths(s)
    assert len(paths)==len(set(paths)) and all((ROOT/p).is_file() for p in paths)
    assert 'code/cvsrffi/d92_group_barrier_gate.py' in paths
    assert 'tools/evaluate_d92_margin_joint_benchmark.py' not in paths
    assert 'tools/score_d92_margin_joint_benchmark.py' not in paths
    compile(publish.IMPORT_PROGRAM,'isolated-import','exec')
    compile(publish.REMOTE.replace('CONFIG',repr(publish.remote_config(s,dict(runtime_commit=COMMIT),'/exclusive.tar','0'*64))),'remote-publish','exec')
    code=preflight.remote_script(s);compile(code,'metadata-preflight','exec')
    assert 'support_branch_features.npz' in code and 'np.load' not in code and 'torch.load' not in code
    # Inspect executed file-read expressions, excluding source spec reference strings.
    tree=ast.parse(preflight.REMOTE)
    opened=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('read_text','read_bytes','open'):
            opened.update(v.value for v in ast.walk(node.func.value) if isinstance(v,ast.Constant)
                and isinstance(v.value,str) and v.value.endswith(('.json','.npz','.pth')))
    assert opened=={'manifest.json','features_complete.json','checkpoint_provenance.json','support_splits.json','metadata.json','complete.json'}
    assert not opened.intersection({'truth.json','query.json','received.npz','support_branch_features.npz'})
    out=tmp_path/'existing.json';out.write_text('{}',encoding='utf-8')
    with pytest.raises(FileExistsError):preflight.preflight(s,ssh_host='synthetic',ssh_config='synthetic',remote_python='python',output=out,
        run_fn=lambda *a,**kw:pytest.fail('must not invoke SSH'))


def test_explicit_ground_packet_wrong_binding_rejected_and_absence_is_na(tmp_path,monkeypatch):
    head=Ground();head.metadata=SimpleNamespace(checkpoint_sha256='1'*64,feature_contract=object())
    monkeypatch.setattr(evaluate,'load_packet',lambda packet:head)
    (tmp_path/'head_weight.float32.bin').write_bytes(b'\0'*3840)
    dump(tmp_path/'metadata.json',dict(existing_source_only_provenance=dict(model_seed=7),head_metadata=dict(source_only_verdict='synthetic')))
    _,binding=evaluate.ground_packet_binding(tmp_path,'1'*64,7,list(head.classes))
    assert binding['native_head_weight_bytes']==3840 and binding['new_transfer_bytes'] is None
    for sha,seed,old in (('2'*64,7,list(head.classes)),('1'*64,8,list(head.classes)),('1'*64,7,['wrong']*6)):
        with pytest.raises(ValueError):evaluate.ground_packet_binding(tmp_path,sha,seed,old)
    assert evaluate.ground_packet_binding(None,'1'*64,7,list(head.classes))[1]['status']=='N/A'


def test_real_native_ground_float32_list_bridge_and_original_column_tie_policy(tmp_path,monkeypatch):
    from cvsrffi.d92_ground_classifier_a import GroundClassifierA,GroundFeatureContract,GroundHeadMetadata
    fake_models(monkeypatch)
    contract=GroundFeatureContract('1'*64,'z_id','feat_joint','raw','float32',160)
    metadata=GroundHeadMetadata('1'*64,'id_backbone.cls_head.head.weight',Ground.classes,16.,1e-4,
        contract,'MATCHED_SOURCE_ONLY_SCRATCH',False,(),'none')
    ground=GroundClassifierA(weight=torch.zeros((6,160),dtype=torch.float32),metadata=metadata)
    # Known Torch/NumPy shared ABI is unavailable; entry must use list bridges.
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:pytest.fail('NumPy/Torch shared input bridge used'))
    monkeypatch.setattr(torch.Tensor,'numpy',lambda *a,**kw:pytest.fail('Torch/NumPy shared output bridge used'))
    fixed=[]
    result=evaluate.probe_group_barrier_joint(**synthetic(),**RESOURCES,ground_head=ground,prediction_callback=fixed.append)
    for entry in result['folds']+result['oneshot_proxy']['trials']:
        actual=entry['ground_A']
        assert actual['score_dtype']=='float32' and actual['classes']==list(Ground.classes)
        assert actual['predictions']==['c5']*len(actual['physical_ids'])
        assert entry['paths']['R_GROUP_BARRIER_seq']['metrics']['A_old_accuracy']==pytest.approx(1/6)
        assert entry['paths']['R_GROUP_BARRIER_seq']['metrics']['adaptation_gain_B_minus_A']==pytest.approx(5/6)
    assert all('held_labels' not in row for row in fixed)


def test_publisher_requires_pushed_oid_and_whitelist_clean_without_new_gate():
    s=literal();calls=[]
    def output(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='rev-parse':return COMMIT+'\n'
        if argv[1]=='branch':return 'synthetic-branch\n'
        if argv[1]=='ls-remote':return COMMIT+'\trefs/heads/synthetic-branch\n'
        if argv[1]=='status':return ''
        raise AssertionError(argv)
    checked=[]
    bound=publish.git_binding(s,check_output=output,check_call=lambda argv,**kw:checked.append(argv))
    assert bound['remote_oid']==COMMIT and checked==[['git','merge-base','--is-ancestor',s['code']['commit'],COMMIT]]
    assert not any(argv[1]=='diff' for argv in checked)  # New source follows the preparation parent.
    def unpushed(argv,**kw):return 'b'*40+'\trefs/heads/synthetic-branch\n' if argv[1]=='ls-remote' else output(argv,**kw)
    with pytest.raises(ValueError,match='pushed'):publish.git_binding(s,check_output=unpushed,check_call=lambda *a,**kw:None)


def test_marker_runtime_pid_counter_and_physical_coverage_not_exitcode(tmp_path):
    s=literal();row=deepcopy(s['rows'][0]);co=s['probe']['cohorts'][row['cohort']]
    row['support_features']=str(tmp_path/'cache');plan=[]
    for split in co['selection']['splits']:plan.append(dict(split,support_ids=[split['split_id']+'_physical']))
    dump(tmp_path/'cache/support_splits.json',dict(capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],splits=plan))
    out=tmp_path/'probe';physical={v['split_id']:v['support_ids'] for v in plan}
    startup=dict(schema=run.SCHEMA,method=run.METHOD,run_id=s['run_id'],row_id=row['row_id'],capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],release_commit=COMMIT,
        query_rows_used=0,source_rows_used=0,truth_read=False,group_barrier_resources=RESOURCES,pid=123,
        config=run.evaluator_config(s,row['cohort']),ground_packet=row['ground_packet'],selected_support_physical_ids=physical,
        blas_environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),cuda_visible_devices='')
    marker=dict(startup,**dict.fromkeys(evaluate.COUNTERS,0),status=run.STATUS,workload_complete=True,algorithm=run.PROBE_CONFIG,
        selection=co['selection'],producer_matrix=co['matrix'],state_archive_file_count=0)
    marker.update(s['probe']['budget']['per_row'][row['row_id']]['exact'])
    marker['ajlr_stage_count']=marker['candidate_stage_count'];marker['ajlr_preparation_count']=marker['candidate_preparation_count']
    marker['final_head_fit_count']=marker['candidate_stage_count']
    marker['head_fit_count']=marker['baseline_head_fit_count']+marker['final_head_fit_count']
    dump(out/'startup.json',startup);dump(out/'probe_complete.json',marker)
    dump(out/'state_manifest.json',dict(status='COMPLETE',schema='d92_group_barrier_joint_state_archive_v1',method=run.METHOD,file_count=0))
    with (out/'compact.jsonl').open('w',encoding='utf-8') as f:
        for v in co['selection']['splits']:f.write(json.dumps(dict(v,query_rows_used=0,source_rows_used=0))+'\n')
    run.verify_marker(out/'probe_complete.json',s,row,commit=COMMIT,expected_pid=123)
    for key,value in (('pid',124),('release_commit','b'*40),('episodes',39),('group_gate_forward_wall_seconds',-1.),('selected_support_physical_ids',{})):
        wrong=dict(marker);wrong[key]=value;dump(out/'probe_complete.json',wrong)
        with pytest.raises(ValueError):run.verify_marker(out/'probe_complete.json',s,row,commit=COMMIT,expected_pid=123)
