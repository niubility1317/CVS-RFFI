"""Literal synthetic completeness tests plus the real production callback boundary.

This file never reads a registered spec or experiment artifact. Root alone runs
the numerical tests; --static-only uses only stdlib and parses the two owned files.
"""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    for file in (ROOT/'tools/analyze_d92_support_metric_joint_probe.py',Path(__file__)):
        source=file.read_bytes().decode('utf-8',errors='strict');ast.parse(source,filename=str(file))
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in source)
    print('UTF8/AST PASS: two owned analysis files; no numerical imports or execution')
    raise SystemExit(0)

from copy import deepcopy
import itertools
import math

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code'),str(ROOT/'tests')]
import numpy as np
import pytest
import analyze_d92_support_metric_joint_probe as analyze

COMMIT='b'*40
RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,
    max_integer_bits=16384,max_fraction_operations=1000000,max_secular_iterations=128)
OLD=['c'+str(i) for i in range(6)]


def literal_spec():
    cohorts={}
    for cohort in ('rx3','rx1'):
        pairs=[[cohort+'-receiver','scene-a'],[cohort+'-receiver','scene-b']]
        splits=[]
        for (rx,scene),k,q in itertools.product(pairs,[1,5,10,20],[0,2,5,10,20]):
            splits.append(dict(split_id=f'{cohort}-{scene}-{k}-{q}',receiver=rx,scenario=scene,k=k,new_count=q,support_seed=7,
                registered_classes=OLD+['n'+str(i) for i in range(q)]))
        selection=dict(receiver_scenes=pairs,ks=[1,5,10,20],new_counts=[0,2,5,10,20],support_seed=7,splits=splits)
        cohorts[cohort]=dict(capsule_id='synthetic-'+cohort,selection=selection,matrix={})
    rows=[dict(row_id=f'{cohort}-{seed}',cohort=cohort,seeds=dict(model=seed),expected_checkpoint_sha256=('a' if seed==2026092701 else 'c')*64,
        ground_packet=None,ground_summary='/synthetic/frozen-summary-'+str(seed),ground_summary_already_deployed=True,
        support_features='/synthetic/support-cache-'+cohort+'-'+str(seed)) for cohort,seed in itertools.product(cohorts,[2026092701,2026092702])]
    return dict(schema=analyze.SCHEMA,group_id='d92-support-metric-joint-support',run_id='synthetic-only',rows=rows,
        code=dict(commit='d'*40),execution=dict(remote_run_root='/synthetic/unopened'),
        probe=dict(algorithm=deepcopy(analyze.ALGORITHM),support_metric_resources=RESOURCES.copy(),cohorts=cohorts))


@pytest.mark.parametrize('fault',['missing_row','duplicate_row','missing_cell','duplicate_cell','wrong_axis','resources_bool','method'])
def test_full_declared_matrix_rejects_partial_or_changed_contract(fault):
    spec=literal_spec();analyze.validate_spec(spec)
    co=spec['probe']['cohorts']['rx3']
    if fault=='missing_row':spec['rows'].pop()
    elif fault=='duplicate_row':spec['rows'][-1]=deepcopy(spec['rows'][0])
    elif fault=='missing_cell':co['selection']['splits'].pop()
    elif fault=='duplicate_cell':co['selection']['splits'][-1]=deepcopy(co['selection']['splits'][0])
    elif fault=='wrong_axis':co['selection']['ks']=[1,5,10]
    elif fault=='resources_bool':spec['probe']['support_metric_resources']['max_newton_iterations']=True
    else:spec['probe']['algorithm']['max_updates']=2
    with pytest.raises(ValueError):analyze.validate_spec(spec)


@pytest.fixture(scope='module')
def production_parent(tmp_path_factory):
    # This is the actual producer/core/native A/archive pipeline, not an idealized
    # hand-written callback sequence or a mocked solver. No real inputs are read.
    import test_d92_proto_frame_joint_probe as fixture
    import evaluate_d92_support_metric_joint_probe as producer
    root=tmp_path_factory.mktemp('production-analysis');archive=producer.StateArchive(root)
    data=fixture.synthetic(k=3,new=2,zero=True);stages=[];fixed=[]
    def log(value):stages.append(producer.json_native(dict(producer.compact_event(value),schema=producer.SCHEMA,method=producer.METHOD,split_id='literal')))
    result=producer.probe_support_metric_joint(**data,**RESOURCES,prototype_frame=fixture.frame(),ground_head=fixture.real_ground(),
        context=dict(run_id='literal-run',row_id='literal-row',split_id='literal'),state_callback=archive,
        log_callback=log,prediction_callback=fixed.append)
    archive.finalize('COMPLETE')
    identity=dict(split_id='literal',receiver='receiver',scenario='scene',k=3,support_seed=7,new_count=2,registered_classes=data['classes'])
    result=producer.json_native(dict(result,**{k:identity[k] for k in analyze.IDENTITY},scope=analyze.SCOPE,query_rows_used=0,source_rows_used=0))
    return dict(root=root,parent=result,identity=identity,physical=data['support_ids'],stages=stages,fixed=fixed,
        ground=dict(status='MATCHED_SOURCE_ONLY_PACKET',ordered_classes=list(reversed(OLD))))


def verify_production(value):
    archive=analyze.Archives(value['root'],'literal-run','literal-row')
    stages=iter(value['stages']);fixed=iter(value['fixed'])
    result=analyze.verify_parent(value['parent'],value['identity'],value['physical'],row_id='literal-row',run_id='literal-run',
        archives=archive,stage_stream=stages,prediction_stream=fixed,resources=RESOURCES,ground_binding=value['ground'])
    archive.close();assert next(stages,None) is None and next(fixed,None) is None
    return result


def test_actual_producer_complete_callbacks_numeric_archive_and_native_A(production_parent):
    result=verify_production(production_parent)
    assert result['counters']['sequence_paths']==6 and result['counters']['candidate_stage_count']==12
    assert result['counters']['ridge_calls']>0
    for name in analyze.PATHS:
        pooled=analyze.score_fixed([p for p in result['paths'] if p['path']['scope']=='support_oof'],name,OLD)
        expected=result['parent']['oof']['paths'][name]['metrics']
        for key in set(expected)&set(pooled):assert pooled[key]==pytest.approx(expected[key]) if expected[key] is not None else pooled[key] is None
    assert result['parent']['structural_predict_actual_work'] is None


@pytest.mark.parametrize('k,new,rank_zero',[(1,0,False),(1,2,False),(3,0,False),(3,2,True)])
def test_actual_production_K1_complete_head_and_new0_reuse(tmp_path,k,new,rank_zero):
    import test_d92_proto_frame_joint_probe as fixture
    import evaluate_d92_support_metric_joint_probe as producer
    data=fixture.synthetic(k=k,new=new,zero=True);archive=producer.StateArchive(tmp_path);stages=[];fixed=[]
    def log(value):stages.append(producer.json_native(dict(producer.compact_event(value),schema=producer.SCHEMA,method=producer.METHOD,split_id='literal')))
    frame=fixture.frame()
    if rank_zero:
        from cvsrffi import d92_proto_frame_primitives as primitive
        prototypes=np.zeros((6,160));prototypes[:,0]=1.
        frame=primitive.build_proto_frame_dictionary(prototypes=prototypes,classes=OLD)
    result=producer.probe_support_metric_joint(**data,**RESOURCES,prototype_frame=frame,ground_head=fixture.real_ground(),
        context=dict(run_id='literal-run',row_id='literal-row',split_id='literal'),state_callback=archive,
        log_callback=log,prediction_callback=fixed.append)
    archive.finalize('COMPLETE')
    identity=dict(split_id='literal',receiver='receiver',scenario='scene',k=k,support_seed=7,new_count=new,registered_classes=data['classes'])
    value=dict(root=tmp_path,parent=producer.json_native(dict(result,**{k:identity[k] for k in analyze.IDENTITY},scope=analyze.SCOPE,
        query_rows_used=0,source_rows_used=0)),identity=identity,physical=data['support_ids'],stages=stages,fixed=fixed,
        ground=dict(status='MATCHED_SOURCE_ONLY_PACKET',ordered_classes=list(reversed(OLD))))
    checked=verify_production(value)
    assert checked['counters']['optimizer_steps']==0
    if rank_zero:
        assert checked['counters']['peak_effective_adapter_rank']==0
        assert checked['counters']['support_metric_step_calls']==0
    count=(1 if new==0 else 2)*(1 if k==1 else 6)
    assert checked['counters']['candidate_stage_count']==checked['counters']['final_candidate_head_fit_count']==count
    if k==1:
        assert not fixed and all(v is None for v in analyze.score_fixed(checked['paths'],'R_SUPPORT_METRIC_seq',OLD).values())
    elif new==0:
        for value in checked['paths']:
            metrics=analyze.score_fixed([value],'R_SUPPORT_METRIC_seq',OLD)
            assert metrics['C_new_accuracy'] is metrics['C_h'] is metrics['C_abs_new_old_gap'] is None
            assert metrics['B_old_accuracy']==metrics['C_old_accuracy']


@pytest.mark.parametrize('fault',['stage_order','held_id','fixed_prediction','outer_scope','actual_B_ref','work_counter','missing_A','matrix_shape',
    'basis_double_charge','basis_origin','basis_rank','basis_certificate_path'])
def test_production_artifact_tampering_rejected(production_parent,fault):
    value=deepcopy(production_parent)
    if fault=='stage_order':value['stages'][1],value['stages'][2]=value['stages'][2],value['stages'][1]
    elif fault=='held_id':value['parent']['folds'][0]['c_ids'][0]='foreign-physical-id'
    elif fault=='fixed_prediction':value['fixed'][0]['predictions'][0]='unregistered'
    elif fault=='outer_scope':value['fixed'][0]['outer_scope']='foreign-scope'
    elif fault=='actual_B_ref':value['parent']['folds'][0]['candidate_stages'][1]['final_prior_ref']=value['parent']['folds'][1]['candidate_stages'][0]['final_state_ref']
    elif fault=='work_counter':value['parent']['actual_work']['ridge_triangular_rhs_columns']+=1
    elif fault=='missing_A':value['fixed'].pop(0)
    elif fault=='matrix_shape':value['fixed'][1]['scores']=[sum(value['fixed'][1]['scores'],[])]
    elif fault=='basis_double_charge':value['parent']['folds'][0]['preparations'][0]['basis_construction_charged_here']=True
    elif fault=='basis_origin':value['parent']['row_basis_audit']['context']['row_id']='other-row'
    elif fault=='basis_rank':value['parent']['row_basis_rank']=0
    else:value['parent']['row_basis_audit']['basis_certificate']['path']='../unrelated.json'
    with pytest.raises((ValueError,KeyError)):verify_production(value)


def literal_evidence(old_correct,new_correct,*,old_count=2,new_count=2,has_A=True):
    old=['o0','o1'];new=['n0','n1'];classes=old+new;oi=['old0','old1'];ni=['new0','new1'];ids=oi+ni
    truth=dict(zip(ids,classes));bp=['o0','o1'];cp=([truth[i] if n<old_correct else 'n0' for n,i in enumerate(oi)]+
        [truth[i] if n<new_correct else 'o0' for n,i in enumerate(ni)])
    fixed={}
    for name in analyze.PATHS:
        for side,pids,registry,pred in [('B',oi,old,bp),('C',ids,classes,cp)]:
            fixed[name+'_'+side]=dict(physical_ids=pids,classes=registry,predictions=pred,
                scores=[[float(c==winner) for c in registry] for winner in pred])
    if has_A:fixed['A']=dict(physical_ids=oi,classes=old,predictions=['o1','o0'],scores=[[0.,1.],[1.,0.]])
    path=dict(held_labels=truth,c_classes=classes,b_ids=oi,c_ids=ids,scope='support_oneshot_proxy',
        fold=None,trial=0,parent_k=5,train_k=1,paths={})
    value=dict(path=path,fixed=fixed)
    for name in analyze.PATHS:path['paths'][name]=dict(metrics=analyze.score_fixed([value],name,old))
    return value


def test_parent_first_H_and_absolute_gap_and_missing_A():
    one=literal_evidence(2,0);two=literal_evidence(0,2);two['path']['trial']=1
    for name in analyze.PATHS:
        left=analyze.score_fixed([one],name,['o0','o1']);right=analyze.score_fixed([two],name,['o0','o1'])
        assert (left['C_h']+right['C_h'])/2==0
        assert (left['C_abs_new_old_gap']+right['C_abs_new_old_gap'])/2==1
        assert 2*.5*.5/(.5+.5)==.5  # Recomputing H from grand means would be wrong.
    missing=analyze.score_fixed([literal_evidence(1,1,has_A=False)],'R_SUPPORT_METRIC_seq',['o0','o1'])
    assert missing['A_old_accuracy'] is missing['adaptation_gain_B_minus_A'] is None
    assert missing['B_old_accuracy']==1
    assert all(v is None for v in analyze.score_fixed([], 'R0',OLD).values())


def test_full_literal_parent_path_tables_keep_K1_and_all_proxy_anchors():
    # Pure literal score records test aggregation only; production state validation
    # is above. Each of the 160 parents has the exact declared outer path count.
    spec=literal_spec();rows=[]
    for row in spec['rows']:
        parents=[]
        for identity in spec['probe']['cohorts'][row['cohort']]['selection']['splits']:
            k=identity['k'];paths=[]
            if k==1:
                paths=[dict(path=dict(held_labels={},c_classes=['o0','o1','n0','n1'],b_ids=[],c_ids=[],scope='support_full_k1',
                    fold=None,trial=None,parent_k=1,train_k=1,paths={n:dict(metrics=dict.fromkeys(analyze.METRICS)) for n in analyze.PATHS}),fixed={},parameter_counts=[])]
            else:
                for scope,limit in [('support_oof',3),('support_oneshot_proxy',k)]:
                    for index in range(limit):
                        value=literal_evidence(2 if index%2==0 else 0,0 if index%2==0 else 2)
                        value['parameter_counts']=[];value['path'].update(scope=scope,fold=index if scope=='support_oof' else None,
                            trial=index if scope=='support_oneshot_proxy' else None,parent_k=k,train_k=k-1 if scope=='support_oof' else 1)
                        if scope=='support_oof':
                            remap=lambda p:p+'-fold'+str(index)
                            p=value['path'];p['held_labels']={remap(i):v for i,v in p['held_labels'].items()}
                            for field in ('b_ids','c_ids'):p[field]=list(map(remap,p[field]))
                            for record in value['fixed'].values():record['physical_ids']=list(map(remap,record['physical_ids']))
                        paths.append(value)
            parent=dict(**{key:identity[key] for key in analyze.IDENTITY},old_classes=['o0','o1'],classes=['o0','o1','n0','n1'],oof=None,oneshot_proxy=None)
            if k>1:
                parent['oof']={'paths':{n:{'metrics':analyze.score_fixed(paths[:3],n,parent['old_classes'])} for n in analyze.PATHS}}
                parent['oneshot_proxy']={'parent_mean_metrics':{n:{metric:sum(analyze.score_fixed([v],n,parent['old_classes'])[metric]
                    for v in paths[3:])/k for metric in analyze.METRICS} for n in analyze.PATHS}}
            parents.append(dict(parent=parent,paths=paths))
        rows.append(dict(row=row,parents=parents))
    parents,paths,tables=analyze.derive_tables(dict(rows=rows))
    assert len(parents)==560 and len(paths)==3600
    assert len({(r['scope'],r['path'],r['k'],r['new_count']) for r in tables['by_k_new_count']})==70
    assert all(r['C_h'] is None for r in parents if r['k']==1)
    proxy=[r for r in parents if r['scope']=='support_oneshot_proxy']
    assert all(r['anchor_count']==r['k'] and r['C_h']==0 and r['C_abs_new_old_gap']==1 for r in proxy)


def supervisor_bundle(tmp_path,monkeypatch,*,bad_last=False):
    """Literal full four-row/160-parent supervisor boundary; per-row numerical
    verification is exercised separately with the actual producer above."""
    spec=literal_spec();root=tmp_path/'run';root.mkdir();counter=analyze._empty_counters()
    counter.update(episodes=40,sequence_paths=450,row_basis_construction_count=1,peak_effective_adapter_rank=2)
    rows={};verified={}
    for i,row in enumerate(spec['rows']):
        work=analyze._empty_work();physical={s['split_id']:['physical-'+s['split_id']] for s in spec['probe']['cohorts'][row['cohort']]['selection']['splits']}
        marker=dict(pid=100+i,actual_work=work)
        verified[row['row_id']]=dict(counters=counter.copy(),marker=marker,startup=dict(selected_support_physical_ids=physical))
        rows[row['row_id']]=dict(status=analyze.STATUS,release_commit=COMMIT,pid=100+i,actual_work=work,**counter)
    binding=dict(schema=analyze.SCHEMA,method=analyze.METHOD,run_id=spec['run_id'],group_id=spec['group_id'],runtime_commit=COMMIT,
        code_commit=spec['code']['commit'],pid=12,query_access=False,truth_read=False,source_sample_access=False,scorer_invoked=False)
    analyze.write_json(root/'startup.json',dict(binding,status='SUPPORT_METRIC_JOINT_SUPERVISOR_STARTED',resolved_spec=spec))
    total=analyze._empty_counters()
    for _ in rows:analyze._merge(total,counter)
    analyze.write_json(root/'complete.json',dict(binding,status=analyze.STATUS,workload_complete=True,automatic_retry=False,
        model_rows=4,completed_rows=4,rows=rows,actual_work_by_row={k:v['actual_work'] for k,v in rows.items()},**total))
    calls=[]
    def verify(root,spec,row,commit):
        calls.append(row['row_id'])
        if bad_last and len(calls)==4:raise ValueError('fourth row incomplete')
        return verified[row['row_id']]
    monkeypatch.setattr(analyze,'verify_row',verify)
    return spec,root,calls


def test_complete_four_row_closure_before_any_truth_join(tmp_path,monkeypatch):
    spec,root,calls=supervisor_bundle(tmp_path,monkeypatch,bad_last=True)
    monkeypatch.setattr(analyze,'derive_tables',lambda value:pytest.fail('Truth join before all four fixed rows verified'))
    out=tmp_path/'analysis'
    with pytest.raises(ValueError,match='fourth row'):analyze.analyze(spec=spec,run_root=root,output=out,expected_runtime_commit=COMMIT)
    assert len(calls)==4 and (out/'analysis_failed.json').exists() and not (out/'summary.json').exists()
    with pytest.raises(FileExistsError):analyze.analyze(spec=spec,run_root=root,output=out,expected_runtime_commit=COMMIT)


def test_full_supervisor_actual_SUM_MAX_and_runtime(tmp_path,monkeypatch):
    spec,root,calls=supervisor_bundle(tmp_path,monkeypatch)
    result=analyze.verify_run(spec=spec,run_root=root,expected_runtime_commit=COMMIT)
    assert result['counters']['episodes']==160 and result['counters']['sequence_paths']==1800 and len(calls)==4
    with pytest.raises(ValueError,match='identity'):analyze.verify_run(spec=spec,run_root=root,expected_runtime_commit='e'*40)


def test_no_fit_cache_ground_or_truth_import_and_strict_json():
    tree=ast.parse((ROOT/'tools/analyze_d92_support_metric_joint_probe.py').read_text(encoding='utf-8'))
    names={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    names|={item.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for item in node.names}
    assert names<={'argparse','collections','csv','itertools','json','math','pathlib','time','numpy'}
    for text in ('{"x":NaN}','{"x":1,"x":2}'):
        with pytest.raises(ValueError):analyze.loads(text)
    with pytest.raises(ValueError):analyze._scores([[1,2,3,4]],['a','b'],['x','y'])


def test_scalar_tables_preserve_NA_and_measured_zero(tmp_path):
    analyze.write_table(tmp_path,'literal',[dict(metric='零',mean=0.,missing=None,seen=True)])
    assert list(analyze.jsonl(tmp_path/'literal.jsonl'))==[dict(metric='零',mean=0.,missing=None,seen=True)]
    assert 'missing' in (tmp_path/'literal.csv').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):analyze.write_table(tmp_path,'literal',[])


def metric_certificate():
    """Production scalar-objective and step results on literal saved-score/JVP input.
    No head, source data, cache or fixture selected from observed results."""
    from cvsrffi import d92_proto_frame_primitives as primitive
    from cvsrffi import d92_support_metric_step as step
    scores=np.array([[.4,-.1],[-.2,.6],[.1,.2],[-.3,.4]])
    labels=np.array([0,1,0,1]);J=np.array([[[.1,.2],[-.2,.1]],[[.3,-.1],[.2,.1]],
        [[.1,.4],[-.1,.2]],[[-.2,.3],[.2,-.1]]])
    padded=np.zeros((4,2,5));padded[...,:2]=J
    rms=primitive.class_balanced_rmsce(scores=scores,labels=labels,score_jacobian=padded)
    gram=np.array([[1.,.125],[.125,1.5]])
    initial=dict(scores=scores,labels=labels,probabilities=rms.arrays['probabilities'],score_jacobian=J,
        gradient=rms.gradient[:2],curvature=rms.curvature[:2,:2],physical_gram=gram)
    result=step.solve_support_metric_step(gradient=initial['gradient'],ggn=initial['curvature'],physical_gram=gram,
        score_jvp=J,probabilities=initial['probabilities'],labels=labels,max_secular_iterations=128)
    saved=dict(score_jvp=J,probabilities=initial['probabilities'],labels=labels,**result.arrays)
    return initial,saved,result.audit_dict(),float(rms.loss)


def test_independent_saved_objective_Fisher_metric_and_KKT_readback():
    initial,saved,audit,loss=metric_certificate()
    analyze.verify_objective_arrays(initial,2,['a','b','c','d'],['old','new'],
        dict(a='old',b='new',c='old',d='new'),loss)
    direction,M=analyze.verify_metric_direction(initial,saved,audit,2)
    assert direction.shape==(2,) and float(direction@M@direction)<=.25
    assert audit['direction_error_bound'] is None and not audit['precise_original_problem_certificate']
    assert not np.array_equal(saved['fisher'],initial['curvature'])


@pytest.mark.parametrize('fault',['fisher','gram_identity','ggn_is_fisher','multiplier','direction','label_binding','certificate'])
def test_saved_metric_evidence_tampering_rejected(fault):
    initial,saved,audit,loss=metric_certificate()
    initial={k:v.copy() for k,v in initial.items()};saved={k:v.copy() for k,v in saved.items()}
    if fault=='fisher':saved['fisher'][0,0]+=.05
    elif fault=='gram_identity':saved['physical_gram']=np.eye(2)
    elif fault=='ggn_is_fisher':saved['ggn']=saved['fisher'].copy()
    elif fault=='multiplier':saved['multiplier']=np.asarray(-1.)
    elif fault=='direction':saved['direction']=saved['direction']+.25
    elif fault=='label_binding':saved['labels']=1-saved['labels']
    else:audit['precise_original_problem_certificate']=True
    with pytest.raises(ValueError):analyze.verify_metric_direction(initial,saved,audit,2)


def test_basis_U_enclosure_actual_Gram_and_rank_zero():
    U=np.zeros((160,2));U[0,0]=1.;U[1,1]=1.
    arrays=dict(original_dictionary_Q=np.zeros((160,5)),U=U,U_lower=U.copy(),U_upper=U.copy(),
        U_error_bounds=np.zeros_like(U),exact_column_indices=np.array([0,2]),physical_gram=U.T@U)
    analyze.verify_basis_arrays(arrays,2)
    changed=deepcopy(arrays);changed['physical_gram'][0,0]+=1e-5
    with pytest.raises(ValueError,match='Gram'):analyze.verify_basis_arrays(changed,2)
    changed=deepcopy(arrays);changed['U_upper'][0,0]=.9
    with pytest.raises(ValueError,match='enclosure'):analyze.verify_basis_arrays(changed,2)
    zero={k:(np.zeros((160,0)) if k in ('U','U_lower','U_upper','U_error_bounds') else v) for k,v in arrays.items()}
    zero.update(exact_column_indices=np.array([],dtype=int),physical_gram=np.zeros((0,0)))
    analyze.verify_basis_arrays(zero,0)


def test_basis_construction_SUM_and_step_peak_MAX_are_different_scopes():
    ops=[dict(operation='basis',audit=dict(fraction_operations_attempted=12,max_observed_integer_bits=400)),
         dict(operation='basis_binding',audit=dict(wall_seconds=.2,physical_gram_evaluation_count=1)),
         dict(operation='support_metric_step',audit=dict(factorization_attempts=2,peak_single_factor_input_output_bytes=144)),
         dict(operation='support_metric_step',audit=dict(factorization_attempts=3,peak_single_factor_input_output_bytes=96))]
    work=analyze._empty_work()
    for item in ops:
        op=item['operation'];work[op+'_calls']+=1
        for key,value in item['audit'].items():
            full=op+'_'+key
            if full in analyze.WORK_KEYS:work[full]=max(work[full],value) if full in analyze.WORK_MAX else work[full]+value
    assert analyze.verify_work(dict(actual_work=work,operation_audits=ops))==work
    assert work['basis_fraction_operations_attempted']==12 and work['basis_binding_calls']==1
    assert work['support_metric_step_factorization_attempts']==5
    assert work['support_metric_step_peak_single_factor_input_output_bytes']==144
    corrupt=work.copy();corrupt['support_metric_step_peak_single_factor_input_output_bytes']=240
    with pytest.raises(ValueError):analyze.verify_work(dict(actual_work=corrupt,operation_audits=ops))


def test_independent_declared_analysis_ABI_matches_frozen_producer_and_core():
    import evaluate_d92_support_metric_joint_probe as producer
    from cvsrffi import d92_support_metric_joint_local_ridge as core
    assert analyze.ALGORITHM==core.FROZEN_CONFIG
    assert analyze.SCHEMA==producer.SCHEMA and analyze.METHOD==producer.METHOD
    assert analyze.STATUS==producer.STATUS and analyze.SCOPE==producer.SCOPE
    assert analyze.WORK_SUM==core.WORK_SUM_KEYS and analyze.WORK_MAX==core.WORK_MAX_KEYS
    assert analyze.COUNTERS==producer.COUNTERS and analyze.PEAKS==set(producer.PEAK_COUNTERS)


@pytest.mark.parametrize('failed_numeric',[False,True])
def test_failure_artifacts_cannot_become_completed_analysis(tmp_path,failed_numeric):
    import evaluate_d92_support_metric_joint_probe as producer
    archive=producer.StateArchive(tmp_path)
    ns=json.dumps(dict(run_id='synthetic',row_id='row',state='FAILED_FIT'))
    if failed_numeric:
        archive.failure(ns+'/failure',dict(current=np.array([np.nan]),last_accepted=np.array([0.])))
        archive.finalize('COMPLETE')
    else:archive.finalize('INCOMPLETE')
    with pytest.raises(ValueError,match='Incomplete numeric archive|Failed numeric state'):
        analyze.Archives(tmp_path,'synthetic','row')
