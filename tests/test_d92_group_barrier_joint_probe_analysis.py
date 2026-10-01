"""Synthetic artifact joins; never access a registered run or input cache."""
import sys
from pathlib import Path

if __name__=='__main__' and '--static-only' in sys.argv:
    import ast
    root=Path(__file__).resolve().parents[1]
    for relative in ('tools/analyze_d92_group_barrier_joint_probe.py',
                     'tests/test_d92_group_barrier_joint_probe_analysis.py'):
        ast.parse((root/relative).read_text(encoding='utf-8'),filename=relative)
    print('STATIC_UTF8_AST_OK: two owned GroupBarrier analysis files')
    raise SystemExit(0)

import copy
import json
from collections import defaultdict
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import analyze_d92_group_barrier_joint_probe as analysis
import evaluate_d92_group_barrier_joint_probe as producer
import run_d92_group_barrier_joint_probe as supervisor

RUNTIME='9'*40
OLD=[f'old{i}' for i in range(6)]


def dump(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def jsonl(path,values):
    with path.open('w',encoding='utf-8') as stream:
        for value in values: stream.write(json.dumps(value,allow_nan=False)+'\n')


def spec_fixture():
    resources=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160)
    spec=dict(schema=analysis.SCHEMA,group_id='d92-group-barrier-joint-support',run_id='synthetic_blind_analysis',
        spec_path='configs/synthetic.json',code=dict(commit='a'*40,preparation_parent_commit='a'*40,cwd='/synthetic/release'),
        execution=dict(cpu_lanes=2,blas_threads_per_lane=2,launch_owner='root',remote_run_root='/synthetic/run'),
        permissions=dict(query_use='none; query IQ/labels/truth/scores never read',source_samples=False,
            source_per_record_features=False,summary_inputs=False,old_target_scores_for_adaptation=False,
            cross_run_result_tuning=False,adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse'),
        probe=dict(algorithm=copy.deepcopy(supervisor.PROBE_CONFIG),channel=supervisor.CHANNEL,
            candidate='R_GROUP_BARRIER_seq',controls=['R0'],query_access=False,reuse_support_cache=True,
            interpretation='support_joint_pilot_no_direct_promotion',group_barrier_resources=resources,cohorts={}),rows=[])
    for cohort in ('rx1','rx3'):
        pairs=[[cohort,scene] for scene in supervisor.SCENARIOS[:2]]
        splits=[dict(split_id=f'{cohort}_{scene}_{k}_{new}',receiver=cohort,scenario=scene,k=k,
            support_seed=11,registered_classes=OLD+[f'new{i}' for i in range(new)],new_count=new)
            for _,scene in pairs for k in supervisor.KS for new in supervisor.NEW_COUNTS]
        matrix=dict(receivers=[cohort],scenarios=supervisor.SCENARIOS,ks=supervisor.KS,new_counts=supervisor.NEW_COUNTS,support_seeds=[11])
        selection=dict(receiver_scenes=pairs,support_seed=11,ks=supervisor.KS,new_counts=supervisor.NEW_COUNTS,splits=splits)
        spec['probe']['cohorts'][cohort]=dict(matrix=matrix,selection=selection,capsule='/synthetic/input/'+cohort,
            capsule_id='residual-noeq-'+cohort,expected_split_count=60,selected_split_count=40)
    for co in spec['probe']['cohorts']:
        spec['probe']['cohorts'][co]['evaluator_config']=supervisor.evaluator_config(spec,co)
        for model in sorted(supervisor.MODEL_SEEDS):
            row_id=f'{co}_{model}'
            spec['rows'].append(dict(row_id=row_id,cohort=co,output_root='/synthetic/run/'+row_id,
                method=analysis.METHOD,initial_step_size=.125,lr=None,support_features='/synthetic/input/'+row_id,
                expected_checkpoint_sha256=('b' if model==min(supervisor.MODEL_SEEDS) else 'c')*64,
                ground_packet='/synthetic/packet/'+str(model),
                seeds=dict(model=model,split=None,data=None,augmentation=None,support=11,evaluation=None)))
    spec['probe']['budget']=supervisor.budget_for_spec(spec)
    return supervisor.validate_spec(spec)


def scores(ids,classes,labels,*,wrong_old=False,displace=False,wrong_new=False):
    value=np.zeros((len(ids),len(classes)),dtype=np.float64)
    for i,pid in enumerate(ids):
        label=labels[pid]; value[i,classes.index(label)]=3.
        if wrong_old and label==OLD[0]: value[i,classes.index(OLD[1])]=4.
        if displace and label==OLD[0]: value[i,classes.index('new0')]=4.
        if wrong_new and label.startswith('new'): value[i,classes.index(OLD[0])]=4.
    return value


def _fake_gate(n,m,q,U,old_indices=None):
    # Analytic zero-kernel finite-barrier equilibrium: total logistic residual
    # is one, and total positive barrier multiplier is one.
    p=(m+1)/n; b=float(np.log(p/(1-p))); zeta=n*1e-4/(m*q); slack=n*1e-4
    old=np.arange(m,dtype=np.int64) if old_indices is None else np.asarray(old_indices,dtype=np.int64)
    targets=np.zeros(n); targets[old]=1.
    bounds=np.full((m,q),b-slack); f=np.full(n,b); qeff=np.full(n,p); qeff[old]-=1+1/m
    ni=np.array([i for i in range(n) if i not in set(old)],dtype=np.int64)
    Y=np.eye(q)[np.arange(n-m)%q]-1/q; nb=Y.mean(axis=0)
    return dict(U=U,anchor_U=U,prior_B_U=U,K=np.zeros((n,n)),new_indices=ni,Ynew=Y,
        new_alpha=Y-nb,new_intercept=nb,new_schur_z=np.ones(n-m),new_schur_s=np.asarray(n-m,dtype=float),
        new_chol=np.eye(n-m),new_rhs=np.column_stack((Y,np.ones(n-m))),
        gate_K=np.zeros((n,n)),gate_alpha=-qeff,gate_b=np.asarray(b),gate_targets=targets,
        gate_old_indices=old,gate_lower_bounds=bounds,gate_zeta=np.asarray(zeta),
        gate_f=f,gate_slacks=np.full((m,q),slack),gate_qeff=qeff)


def parent_fixture(split,row,spec,archive):
    k=split['k']; new=split['new_count']; classes=sorted(split['registered_classes'])
    prefix=split['receiver']+'_'+split['scenario']; ids=sorted(f'{prefix}:{label}:{i:02d}' for label in classes for i in range(k))
    labels={pid:pid.split(':')[-2] for pid in ids}; folds=0 if k==1 else min(k,3)
    context=dict(run_id=spec['run_id'],row_id=row['row_id'],split_id=split['split_id'])
    record=dict(schema=analysis.SCHEMA,method=analysis.METHOD,**{key:split[key] for key in analysis.IDENTITY},
        inheritance_binding=context,group_barrier_resources=spec['probe']['group_barrier_resources'],classes=classes,old_classes=OLD,
        support_count=len(ids),old_class_count=6,new_class_count=new,fold_count=folds,
        scope=analysis.SCOPE,query_rows_used=0,source_rows_used=0,folds=[],oof=None,oneshot_proxy=None,full_support=None,
        physical_fold_assignment=[] if not folds else [dict(physical_id=pid,class_id=labels[pid],fold=int(pid.split(':')[-1])%folds) for pid in ids],
        fit_seconds=.1,persistent_state_bytes=64,heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k==1 else None,
        **dict.fromkeys(producer.COUNTERS[4:],0))
    frozen=[]; U=np.zeros((736,8))
    def make(scope,index,keep):
        train=[pid for pid in ids if keep(pid)]; held=[pid for pid in ids if not keep(pid)]
        btrain=[pid for pid in train if labels[pid] in OLD]; bids=[pid for pid in held if labels[pid] in OLD]
        entry=dict(scope=scope,fold=index if scope=='support_oof' else None,trial=index if scope=='support_oneshot_proxy' else None,
            parent_k=k,train_k=len(train)//len(classes),held_k=len(held)//len(classes),b_training_ids=btrain,c_training_ids=train,
            b_ids=bids,c_ids=held,b_classes=OLD,c_classes=classes,held_labels={pid:labels[pid] for pid in held},
            stages=[],preparations=[],candidate_stages=[],training_events=[],c_reuses_b0=new==0,c_reuses_b_candidates=new==0,
            ground_A=None,actual_A_unavailable_reason=None,prediction_status='FIXED_BEFORE_SUPPORT_TRUTH_JOIN' if held else 'NO_HELD_PREDICTIONS')
        coords=dict(context,**{key:entry[key] for key in ('scope','fold','trial','parent_k','train_k')})
        def save(state,key,arrays):
            return archive(json.dumps(dict(coords,state=state),sort_keys=True,separators=(',',':'))+'/'+key,arrays)
        for stage in ('B0',)+(() if new==0 else ('C0',)):
            entry['stages'].append(dict(coords,state=stage,training_physical_ids=btrain if stage=='B0' else train,
                final_state_ref=save(stage,'baseline',dict(U=U)),factorization_calls=1,fit_and_score_seconds=.01))
        for stage in ('B',)+(() if new==0 else ('C',)):
            entry['preparations'].append(dict(coords,state=stage,inherited_state=stage=='C',
                training_physical_ids=btrain if stage=='B' else train,preparation_seconds=.01))
        bstage=dict(coords,state='B_MARGIN',mode='B',schema='d92_margin_joint_local_ridge_v1',method='D92-MarginJointLocalRidge-v1',
            training_physical_ids=btrain,final_state_ref=save('B_MARGIN','final',dict(U=U)),fit_seconds=.01)
        entry['candidate_stages'].append(bstage)
        if new:
            arrays=_fake_gate(len(train),len(btrain),new,U,[train.index(pid) for pid in btrain])
            cstage=dict(coords,state='C_GROUP_BARRIER_seq',mode='C_seq',schema=analysis.SCHEMA,method=analysis.METHOD,inherited_from_B=True,
                training_physical_ids=train,final_state_ref=save('C_GROUP_BARRIER_seq','final',arrays),fit_seconds=.01,
                group_gate_forward_factorization_attempts=2,group_gate_adjoint_factorization_attempts=1,
                new_ridge_fit_count=1,gate_peak_factor_buffer_bytes=256)
            entry['candidate_stages'].append(cstage)
            entry['preparations'][1]['final_problem']=dict(prior_source='FROZEN_CURRENT_ACTUAL_B',prior_ref=bstage['final_state_ref'])
        if not held:
            entry['paths']={path:dict(b_scores=[],c_scores=[],metrics=dict.fromkeys(producer.METRICS)) for path in analysis.PATHS}
            return entry
        r0b=scores(bids,OLD,labels,wrong_old=True); r0c=r0b if not new else scores(held,classes,labels,wrong_old=True)
        gb=scores(bids,OLD,labels); gc=gb if not new else scores(held,classes,labels,displace=True,wrong_new=True)
        ga=scores(bids,OLD,labels,wrong_old=True).astype(np.float32)
        ground=dict(physical_ids=bids,classes=OLD,scores=ga.tolist(),score_dtype='float32',
            predictions=[OLD[int(j)] for j in np.argmax(ga,axis=1)],scores_state_ref=save('A_GROUND_FIXED_PREDICTIONS','scores',dict(scores=ga)))
        entry['ground_A']=ground
        common=dict(b_ids=bids,c_ids=held,b_classes=OLD,c_classes=classes,held_labels=entry['held_labels'],ground_A=ground)
        evidence={'R0':dict(common,b_scores=r0b.tolist(),c_scores=r0c.tolist()),
            'R_GROUP_BARRIER_seq':dict(common,b_scores=gb.tolist(),c_scores=gc.tolist(),c_predictions=dict(zip(held,analysis._predictions(gc,classes))))}
        independent=producer.assess_paths(evidence,OLD)
        entry['paths']={path:dict(b_scores=value['b_scores'],c_scores=value['c_scores'],c_predictions=value.get('c_predictions',{}),**independent[path]) for path,value in evidence.items()}
        entry['fixed_prediction_state_ref']=save('FIXED_SUPPORT_PREDICTIONS','scores',dict(R0_b=r0b,R0_c=r0c,R_GROUP_BARRIER_seq_b=gb,R_GROUP_BARRIER_seq_c=gc))
        frozen.append(dict(coords,stream='A',status='FIXED_BEFORE_SUPPORT_TRUTH_JOIN',**ground))
        for path,value in evidence.items():
            for side in ('b','c'):
                registry=value[side+'_classes']; arr=np.asarray(value[side+'_scores'])
                preds=[value['c_predictions'][pid] for pid in held] if side=='c' and path==analysis.PATHS[1] else analysis._predictions(arr,registry)
                frozen.append(dict(coords,stream=path+'_'+side.upper(),status='FIXED_BEFORE_SUPPORT_TRUTH_JOIN',
                    physical_ids=value[side+'_ids'],classes=registry,scores=arr.tolist(),predictions=preds))
        return entry
    if k==1:
        record['full_support']=make('support_full_k1',None,lambda pid:True)
    else:
        record['folds']=[make('support_oof',fold,lambda pid,fold=fold:int(pid.split(':')[-1])%folds!=fold) for fold in range(folds)]
        record['oof']=dict(paths=producer.pooled_assess(record['folds'],labels,classes,OLD))
        trials=[make('support_oneshot_proxy',i,lambda pid,i=i:int(pid.split(':')[-1])==i) for i in range(k)]
        record['oneshot_proxy']=dict(trials=trials,trial_count=k,parent_mean_metrics=producer.parent_mean(trials))
    nentries=1 if k==1 else folds+k; stages=1+int(new>0)
    record.update(sequence_paths=nentries,baseline_head_fit_count=nentries*stages,candidate_preparation_count=nentries*stages,
        candidate_stage_count=nentries*stages,diagnostic_fit_count=0)
    if new:
        record.update(group_gate_forward_factorization_attempts=2*nentries,group_gate_adjoint_factorization_attempts=nentries,
                      new_ridge_fit_count=nentries,gate_peak_factor_buffer_bytes=256)
    return record,frozen


@pytest.fixture(scope='module')
def closed_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('blind_group_analysis'); spec=spec_fixture(); rows={}; markers=[]
    rootbinding=dict(schema=analysis.SCHEMA,method=analysis.METHOD,run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=RUNTIME,code_commit=spec['code']['commit'],query_access=False,truth_read=False,source_sample_access=False,scorer_invoked=False)
    dump(root/'startup.json',dict(rootbinding,resolved_spec=spec))
    for ordinal,row in enumerate(spec['rows']):
        directory=root/row['row_id']; directory.mkdir(); probe=directory/'probe'; probe.mkdir()
        co=spec['probe']['cohorts'][row['cohort']]; config=supervisor.evaluator_config(spec,row['cohort']); pid=100+ordinal
        dump(directory/'row_startup.json',dict(run_id=spec['run_id'],row_id=row['row_id'],runtime_commit=RUNTIME,pid=pid,
            model_seed=row['seeds']['model'],checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=co['capsule_id']))
        dump(directory/'resolved_config.json',config); archive=producer.StateArchive(probe); traces=[]; frozen=[]
        for split in co['selection']['splits']:
            record,preds=parent_fixture(split,row,spec,archive); traces.append(record); frozen+=preds
        binding=dict(schema=analysis.SCHEMA,method=analysis.METHOD,run_id=spec['run_id'],row_id=row['row_id'],release_commit=RUNTIME,
            model_seed=row['seeds']['model'],checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=co['capsule_id'],
            group_barrier_resources=spec['probe']['group_barrier_resources'],query_rows_used=0,source_rows_used=0,truth_read=False,pid=pid,
            ground_packet=row['ground_packet'],ground_A_binding=dict(status='MATCHED_SOURCE_ONLY_PACKET',path=row['ground_packet'],
                model_seed=row['seeds']['model'],checkpoint_sha256=row['expected_checkpoint_sha256'],ordered_classes=OLD,packet_file_bytes=17,native_head_weight_bytes=12))
        dump(probe/'startup.json',dict(binding,config=config,hardware=dict(gpu_use=False)))
        jsonl(probe/'fit_trace.jsonl',traces); jsonl(probe/'compact.jsonl',[producer.compact_record(x) for x in traces]); jsonl(probe/'fixed_predictions.jsonl',frozen)
        for filename in ('fit_stages.jsonl','training_events.jsonl','training_events_compact.jsonl'): jsonl(probe/filename,[])
        for filename in ('compact.csv','fit_stages.csv','training_events_compact.csv','training.log'): (probe/filename).write_text('synthetic\n',encoding='utf-8')
        state=archive.finalize('COMPLETE'); ledger=analysis.aggregate_costs(traces,producer.COUNTERS[4:]); counters=dict(ledger['sum'],**ledger['max'])
        counters.update(episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350)
        marker=dict(binding,**counters,status=analysis.STATUS,workload_complete=True,scope=analysis.SCOPE,
            selection=co['selection'],producer_matrix=co['matrix'],algorithm=config['algorithm'],
            state_archive_file_count=state['file_count'],state_archive_file_bytes=state['total_file_bytes'],state_archive_numeric_bytes=state['numeric_array_bytes'],
            payload_audit=dict(new_source_payload_bytes=0,new_ground_statistics_bytes=0,incremental_transfer_bytes=None),wall_seconds=1.,persistent_state_bytes=64)
        dump(probe/'probe_complete.json',marker)
        dump(probe/'artifact_manifest.json',dict(schema='d92_group_barrier_joint_artifacts_v1',method=analysis.METHOD,status=analysis.STATUS,
            files=[dict(path=p.relative_to(probe).as_posix(),file_bytes=p.stat().st_size) for p in sorted(probe.rglob('*')) if p.is_file()]))
        rows[row['row_id']]=dict(status=analysis.STATUS,pid=pid,**{key:marker[key] for key in producer.COUNTERS}); markers.append(marker)
    allcost=analysis.aggregate_costs(markers)
    dump(root/'complete.json',dict(rootbinding,rows=rows,status=analysis.STATUS,workload_complete=True,completed_rows=4,
        **dict(allcost['sum'],**allcost['max'])))
    return spec,root


def test_complete_artifacts_independent_support_join(closed_run):
    spec,root=closed_run
    result=analysis.analyze_run(spec,root,expected_runtime_commit=RUNTIME)
    assert result['parent_count']==160 and result['row_count']==4
    assert result['query_read'] is False and result['parameter_selection'] is False
    assert len(result['tables']['row_K_by_new'])==4*4*5*2*2
    for row in result['metrics']:
        if row['k']==1:
            assert all(row[key] is None for key in analysis.METRIC_NAMES)
        elif row['new_count']==0:
            assert row['C_old_accuracy']==row['B_old_accuracy']
            assert row['C_new_accuracy'] is row['C_h'] is row['C_abs_new_old_gap'] is None
    group=[r for r in result['metrics'] if r['k']>1 and r['new_count']>0 and r['path']==analysis.PATHS[1]]
    assert all(r['adaptation_gain_B_minus_A']>0 and r['total_old_accuracy_drop']>0 for r in group)
    assert result['costs']['max']['gate_peak_factor_buffer_bytes']==256
    assert result['costs']['sum']['group_gate_adjoint_factorization_attempts']>0
    assert all(x['exact_dual_certificate'] is False for x in result['gate_numeric_readbacks'])


@pytest.mark.parametrize('field,value', [('completed_rows',3),('runtime_commit','8'*40),('workload_complete',False)])
def test_incomplete_or_wrong_runtime_never_claims_complete(closed_run,field,value):
    spec,root=closed_run; path=root/'complete.json'; original=path.read_text(encoding='utf-8'); bad=json.loads(original); bad[field]=value
    try:
        dump(path,bad)
        with pytest.raises(ValueError): analysis.analyze_run(spec,root,expected_runtime_commit=RUNTIME)
    finally: path.write_text(original,encoding='utf-8')


def test_cost_sum_peak_and_unknown_are_distinct():
    result=analysis.aggregate_costs([dict(new_ridge_seconds=.25,gate_peak_factor_buffer_bytes=24),
        dict(new_ridge_seconds=.5,gate_peak_factor_buffer_bytes=16)],('new_ridge_seconds','gate_peak_factor_buffer_bytes','new_ridge_adjoint_seconds'))
    assert result['sum']['new_ridge_seconds']==.75
    assert result['max']['gate_peak_factor_buffer_bytes']==24
    assert result['sum']['new_ridge_adjoint_seconds'] is None
    assert result['unavailable_record_counts']['new_ridge_adjoint_seconds']==2


def test_missing_parent_cannot_be_completed_by_marker(closed_run):
    spec,root=closed_run; probe=root/spec['rows'][0]['row_id']/'probe'
    trace=probe/'fit_trace.jsonl'; inventory=probe/'artifact_manifest.json'
    original_trace=trace.read_text(encoding='utf-8'); original_inventory=inventory.read_text(encoding='utf-8')
    try:
        trace.write_text('\n'.join(original_trace.splitlines()[1:])+'\n',encoding='utf-8')
        manifest=json.loads(original_inventory)
        next(x for x in manifest['files'] if x['path']=='fit_trace.jsonl')['file_bytes']=trace.stat().st_size
        dump(inventory,manifest)
        with pytest.raises(ValueError,match='40-parent'): analysis.analyze_run(spec,root,expected_runtime_commit=RUNTIME)
    finally:
        trace.write_text(original_trace,encoding='utf-8'); inventory.write_text(original_inventory,encoding='utf-8')


def test_duplicate_inventory_and_nonfinite_npz_are_rejected(tmp_path):
    archive=producer.StateArchive(tmp_path)
    namespace=json.dumps(dict(state='SYNTHETIC_STATE'),sort_keys=True,separators=(',',':'))
    ref=archive(namespace+'/value',dict(value=np.array([1.])))
    manifest=archive.finalize('COMPLETE')
    marker=dict(state_archive_file_count=1,state_archive_file_bytes=manifest['total_file_bytes'],state_archive_numeric_bytes=manifest['numeric_array_bytes'])
    for name in analysis.REQUIRED_ARTIFACTS:
        if not (tmp_path/name).exists(): (tmp_path/name).write_text('{}',encoding='utf-8')
    artifact=dict(schema='d92_group_barrier_joint_artifacts_v1',method=analysis.METHOD,status=analysis.STATUS,
        files=[dict(path=p.relative_to(tmp_path).as_posix(),file_bytes=p.stat().st_size) for p in tmp_path.rglob('*') if p.is_file()])
    duplicate=copy.deepcopy(artifact); duplicate['files'].append(duplicate['files'][0]); dump(tmp_path/'artifact_manifest.json',duplicate)
    with pytest.raises(ValueError,match='Duplicate artifact'): analysis.GroupStateResolver(tmp_path,marker)
    dump(tmp_path/'artifact_manifest.json',artifact)
    resolver=analysis.GroupStateResolver(tmp_path,marker); bad=copy.deepcopy(ref); bad['arrays']['value']['shape']=[2]
    with pytest.raises(ValueError,match='coordinates'): resolver.bind(bad)
    with pytest.raises(ValueError,match='Unreferenced'): resolver.close_references([])
    with (tmp_path/ref['path']).open('wb') as stream: np.savez_compressed(stream,value=np.array([np.nan]))
    state=analysis.read_json(tmp_path/'state_manifest.json'); size=(tmp_path/ref['path']).stat().st_size
    state['files'][0]['file_bytes']=size; state['total_file_bytes']=size; dump(tmp_path/'state_manifest.json',state)
    marker['state_archive_file_bytes']=size
    for item in artifact['files']: item['file_bytes']=(tmp_path/item['path']).stat().st_size
    dump(tmp_path/'artifact_manifest.json',artifact)
    with pytest.raises(ValueError,match='NPZ numeric'): analysis.GroupStateResolver(tmp_path,marker)


def test_barrier_readback_is_not_exact_dual_certificate():
    arrays=_fake_gate(8,6,2,np.zeros((2,2)))
    diag=analysis.gate_diagnostic(arrays,state_path='synthetic.npz')
    assert diag['canonical_residual']<1e-12 and diag['intercept_residual']<1e-12
    assert diag['minimum_slack']>0 and diag['mu_positive']
    assert diag['exact_dual_certificate'] is False
    arrays['gate_lower_bounds'][:]=float(arrays['gate_b'])+1
    with pytest.raises(ValueError,match='nonpositive slack'): analysis.gate_diagnostic(arrays,state_path='bad.npz')


def test_cross_parent_inherited_prior_is_rejected(tmp_path):
    spec=spec_fixture(); row=spec['rows'][0]; split=next(x for x in spec['probe']['cohorts'][row['cohort']]['selection']['splits'] if x['k']==5 and x['new_count']==2)
    archive=producer.StateArchive(tmp_path); record,_=parent_fixture(split,row,spec,archive); entry=record['folds'][0]
    analysis._check_lineage(record,entry,spec['run_id'],row['row_id'])
    entry['preparations'][1]['final_problem']['prior_ref']=dict(path='other_parent.npz')
    with pytest.raises(ValueError,match='this path actual B'): analysis._check_lineage(record,entry,spec['run_id'],row['row_id'])


def test_fixed_prediction_mismatch_is_not_scored(tmp_path):
    spec=spec_fixture(); row=spec['rows'][0]; split=next(x for x in spec['probe']['cohorts'][row['cohort']]['selection']['splits'] if x['k']==5 and x['new_count']==2)
    record,fixed=parent_fixture(split,row,spec,producer.StateArchive(tmp_path)); entry=record['folds'][0]
    streams={x['stream']:x for x in fixed if analysis._coords(x)==analysis._coords(entry)}
    streams['R_GROUP_BARRIER_seq_C']['predictions'][0]='outside_registry'
    with pytest.raises(ValueError,match='Fixed stream differs'): analysis.assess_entry(entry,OLD,record['classes'],2,streams)


def test_writer_preserves_existing_output_and_marks_unknown(tmp_path):
    output=tmp_path/'existing'; output.mkdir(); (output/'keep').write_text('preserve',encoding='utf-8')
    with pytest.raises(ValueError,match='output exists'): analysis.write_analysis({},output)
    assert (output/'keep').read_text(encoding='utf-8')=='preserve'
    fresh=tmp_path/'fresh'
    analysis.write_analysis(dict(metrics=[dict(A_old_accuracy=None)],entry_metrics=[],tables={},row_count=4,
        parent_count=160,runtime_commit=RUNTIME),fresh)
    assert 'N/A' in (fresh/'parents.csv').read_text(encoding='utf-8')
    assert 'query' in (fresh/'report.md').read_text(encoding='utf-8')


@pytest.fixture
def nested_descriptor_projection(tmp_path):
    """Real production Recorder + compact_event, rather than hand-made shape loss."""
    from cvsrffi import d92_affine_joint_local_ridge as affine
    archive=producer.StateArchive(tmp_path)
    namespace=json.dumps(dict(state='SYNTHETIC_DESCRIPTOR'),sort_keys=True,separators=(',',':'))
    recorder=affine._Recorder(lambda key,arrays:archive(namespace+'/'+key,arrays))
    basic=recorder.save('core',vector=np.array([1.,2.]),scalar=np.asarray(3.))
    native=archive(namespace+'/native',dict(vector=np.array([4.,5.]),scalar=np.asarray(6.)))
    state=archive.finalize('COMPLETE')
    marker=dict(state_archive_file_count=state['file_count'],state_archive_file_bytes=state['total_file_bytes'],
                state_archive_numeric_bytes=state['numeric_array_bytes'])
    for name in analysis.REQUIRED_ARTIFACTS:
        if not (tmp_path/name).exists(): (tmp_path/name).write_text('{}',encoding='utf-8')
    dump(tmp_path/'artifact_manifest.json',dict(schema='d92_group_barrier_joint_artifacts_v1',method=analysis.METHOD,status=analysis.STATUS,
        files=[dict(path=p.relative_to(tmp_path).as_posix(),file_bytes=p.stat().st_size) for p in sorted(tmp_path.rglob('*')) if p.is_file()]))
    full=dict(event='FINAL',audit=dict(preparation=dict(prepared_state_ref=basic),final_state_ref=native))
    compact=producer.compact_event(full)
    return analysis.GroupStateResolver(tmp_path,marker),full,compact


def test_production_nested_compact_descriptors_close_only_in_compact_stream(nested_descriptor_projection):
    resolver,full,compact=nested_descriptor_projection
    projected=compact['audit']['preparation']['prepared_state_ref']
    assert 'shape' in full['audit']['preparation']['prepared_state_ref']['arrays']['vector']
    assert 'shape' not in projected['arrays']['vector']
    assert projected['arrays']['vector']==dict(dtype='float64',nbytes=16)
    resolver.close_references([full],scalar_projected_records=[compact])
    assert resolver.used==set(resolver.entries)
    # The canonical descriptor and the actual NPZ still have the full shape.
    canonical=resolver.entries[projected['path']]
    assert canonical['arrays']['vector']['shape']==[2]
    assert resolver.load(canonical)['vector'].shape==(2,)


@pytest.mark.parametrize('stream',('trace','training_events'))
def test_full_stream_shape_loss_is_rejected(nested_descriptor_projection,stream):
    resolver,full,compact=nested_descriptor_projection
    with pytest.raises(ValueError,match='coordinates missing from full-stream'):
        resolver.close_references([dict(stream=stream,event=compact)])


@pytest.mark.parametrize('change',('mixed_shape','dtype','nbytes','summary','extra_field','unknown_path'))
def test_unknown_or_tampered_compact_descriptor_is_rejected(nested_descriptor_projection,change):
    resolver,full,compact=nested_descriptor_projection; bad=copy.deepcopy(compact)
    ref=bad['audit']['preparation']['prepared_state_ref']
    if change=='mixed_shape': ref['arrays']['vector']['shape']=[2]
    elif change=='dtype': ref['arrays']['vector']['dtype']='float32'
    elif change=='nbytes': ref['arrays']['vector']['nbytes']=8
    elif change=='summary': ref['array_summaries']['vector']['norm']=123.
    elif change=='extra_field': ref['unknown_scalar_field']=0
    else: ref['path']='state_arrays/unregistered.npz'
    with pytest.raises(ValueError,match='state reference projection|Missing state reference'):
        resolver.close_references([full],scalar_projected_records=[bad])
