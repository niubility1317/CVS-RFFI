"""Literal synthetic fixed predictions only; no fitting, native weights or real files."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import score_d92_group_barrier_joint_benchmark as score

OLD=['old_'+str(i) for i in range(6)]
NATIVE=list(reversed(OLD))
COMMIT='a'*40
CONFIG=dict(algorithm=deepcopy(score.FROZEN_ALGORITHM),group_barrier_resources=dict(
    max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def lines(path,values):
    path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values),encoding='utf-8')


def decision(old_scores,new_scores,g):
    def ls(values):
        values=np.asarray(values,dtype=np.float64);shift=values-values.max()
        return (shift-np.log(np.exp(shift).sum())).tolist()
    new=['new_'+str(i).zfill(2) for i in range(len(new_scores))]
    lo,ln=ls(old_scores),ls(new_scores)
    oi=int(np.argmax(old_scores));ni=int(np.argmax(new_scores))
    gap=g+lo[oi]-ln[ni];ow,nw=OLD[oi],new[ni]
    winner=ow if gap>0 else nw if gap<0 else min(ow,nw)
    cert=dict(schema=score.DECISION_SCHEMA,policy=score.DECISION_POLICY,new0_reuses_actual_B=False,
        old_classes=OLD,new_classes=new,old_raw_scores=old_scores,new_raw_scores=new_scores,gate_logit=g,
        old_log_probabilities=lo,new_log_probabilities=ln,old_winner=ow,new_winner=nw,group_gap=gap,prediction=winner)
    mapping={name:float(-np.logaddexp(0.,-g)+lo[i]) for i,name in enumerate(OLD)}
    mapping.update({name:float(-np.logaddexp(0.,g)+ln[i]) for i,name in enumerate(new)})
    classes=sorted(OLD+new)
    return cert,[mapping[name] for name in classes]


def capsule_fixture(root,co,full):
    path=root/co;path.mkdir();(path/'splits').mkdir()
    matrix=dict(receivers=score.RECEIVERS[co],scenarios=score.SCENARIOS,ks=list(score.KS),
        new_counts=list(score.NEWS),support_seeds=score.SUPPORT_SEEDS)
    axes=[matrix[k] for k in ('receivers','scenarios','support_seeds','ks','new_counts')]
    if not full:axes=[axes[0][:1],axes[1][:1],axes[2][:1],axes[3],axes[4]]
    physical=[];positions={};truth={};splits=[];cid='synthetic_'+co
    def pos(pid):
        if pid not in positions:positions[pid]=len(physical);physical.append(pid)
        return positions[pid]
    for rx,scene,seed,k,new in itertools.product(*axes):
        classes=OLD+['new_'+str(i).zfill(2) for i in range(new)]
        support=[];labels=[];queries=[]
        for i,name in enumerate(classes):
            for j in range(k):support.append(pos('s_'+rx+'_'+scene+'_'+name+'_'+str(j)));labels.append(i)
            pid='q_'+rx+'_'+scene+'_'+name;queries.append(pos(pid))
            truth[pid]=dict(pool_role='query',receiver=rx,scene=scene,transmitter=name,old=name in OLD)
        sid='split_'+str(len(splits)).zfill(4)
        split=dict(split_id=sid,capsule_id=cid,protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',
            receiver=rx,scenario=scene,support_seed=seed,k=k,registered_classes=classes,
            support_indices=support,support_labels=labels,query_indices=queries)
        dump(path/'splits'/(sid+'.json'),split);splits.append(split)
    dump(path/'manifest.json',dict(capsule_id=cid,protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',split_count=len(splits)))
    np.savez(path/'received.npz',ids=np.asarray(physical),received=np.asarray([np.nan]))
    truth_path=root/(co+'_truth.json');dump(truth_path,truth)
    return dict(capsule=str(path),truth=str(truth_path),expected_capsule_id=cid,matrix=matrix,
        expected_split_count=score.COHORT_COUNTS[co]),splits,physical,truth


def bundle(tmp_path,full=False):
    root=tmp_path/'run';root.mkdir();cohorts={};data={}
    for co in (score.COHORT_COUNTS if full else ['rx1']):
        value,*items=capsule_fixture(tmp_path,co,full);cohorts[co]=value;data[co]=items
    declarations=[];lanes={};pending={};bindings={}
    pairs=itertools.product(score.COHORT_COUNTS,score.MODEL_SEEDS) if full else [('rx1',score.MODEL_SEEDS[0])]
    for co,seed in pairs:
        rid=co+'_'+str(seed);rowout=root/rid;out=rowout/'predictions';out.mkdir(parents=True)
        sha=('1' if seed==score.MODEL_SEEDS[0] else '2')*64
        row=dict(row_id=rid,cohort=co,expected_model_seed=seed,expected_checkpoint_sha256=sha,
            row_root=str(tmp_path/('source_'+rid)),branch_features=str(tmp_path/('features_'+rid)),
            ground_packet=str(tmp_path/('packet_'+str(seed))),output_root=str(rowout));declarations.append(row)
        source=dict(checkpoint_sha256=sha,model_seed=seed,source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',
            source_role_comparison='EXACT_MATCH',checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,
            cache_schema='d92_branch_received_features_v1',feature_contract=score.RAW_FEATURE_CONTRACT)
        ground=dict(path=row['ground_packet'],checkpoint_sha256=sha,model_seed=seed,ordered_classes=NATIVE,scale=30.,norm_eps=1e-4,
            packet_total_file_bytes=100,source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',feature_contract=dict(
                checkpoint_sha256=sha,cache_key='z_id',source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160))
        splits,physical,truth=data[co];archive=[];metadata=[];records={k:[] for k in score.STREAMS};work=[]
        def ref(sid,state):
            namespace=json.dumps(dict(run_id='synthetic_run',row_id=rid,split_id=sid,scope='query_benchmark_support_training',
                state=state,fold=None,trial=None),sort_keys=True,separators=(',',':'))
            value=dict(key='final',path='state_arrays/'+str(len(archive))+'.npz',namespace=namespace,
                arrays=dict(value=dict(shape=[1],dtype='float64',nbytes=8,all_finite=True,nonfinite_count=0)),failed_numeric_state=False)
            archive.append(value);return deepcopy(value)
        for s in splits:
            classes=sorted(s['registered_classes']);new=len(classes)-6;sid=s['split_id']
            support=sorted(physical[i] for i in s['support_indices'])
            oldsupport=sorted(physical[i] for i,y in zip(s['support_indices'],s['support_labels']) if y<6)
            queries=[physical[i] for i in s['query_indices']]
            b=ref(sid,'B_MARGIN');c=ref(sid,'C_GROUP_BARRIER_seq') if new else b
            metadata.append(dict(**{k:s[k] for k in ('split_id','receiver','scenario','k','support_seed')},new_count=new,
                declared_registered_classes=s['registered_classes'],registered_classes=classes,old_classes=OLD,
                support_ids=support,old_support_ids=oldsupport,new_support_ids=sorted(set(support)-set(oldsupport)),
                query_ids=queries,query_count=len(queries),B_classes=OLD,C_classes=classes,b_state_ref=b,c_state_ref=c,
                c_inherited_from_b_state_ref=b,c_reuses_b=not bool(new),support_only_fit=True,query_fit_access=False,source_fit_access=False))
            for pid in queries:
                name=truth[pid]['transmitter'];old=name in OLD
                aw=name if old and OLD.index(name)<3 else next(v for v in NATIVE if v!=name)
                bw=name if old and OLD.index(name)<5 else next(v for v in OLD if v!=name)
                bvalues=[float(v==bw) for v in OLD]
                records['A'].append(dict(split_id=sid,query_id=pid,classes=NATIVE,scores=[float(v==aw) for v in NATIVE],prediction=aw))
                brecord=dict(split_id=sid,query_id=pid,classes=OLD,scores=bvalues,prediction=bw);records['B'].append(brecord)
                if new:
                    cert,cvalues=decision(bvalues,[float(v==name) for v in classes if v not in OLD],10. if old else -10.)
                    crecord=dict(split_id=sid,query_id=pid,classes=classes,scores=cvalues,prediction=cert['prediction'])
                else:
                    cert=dict(schema=score.DECISION_SCHEMA,policy='EXACT_ACTUAL_B_REUSE',new0_reuses_actual_B=True,prediction=bw)
                    crecord=deepcopy(brecord)
                records['C'].append(crecord);work.append(dict(split_id=sid,query_id=pid,C_structural_decision=cert,
                    B_work=dict(score_seconds=.001),C_work=dict(score_seconds=.001) if new else None,
                    C_reused_B_scores=not bool(new),C_structural_prediction=crecord['prediction'],
                    C_public_predict_calls=int(bool(new)),C_public_predict_seconds=.001 if new else 0.,
                    C_public_predict_internal_work=None,C_public_predict_internal_work_unavailable_reason=(
                        'PUBLIC_PREDICT_RECOMPUTES_GEOMETRY_WITHOUT_RETURNING_AUDIT' if new else 'EXACT_B_REUSE_NO_EXTRA_CALL'),
                    scope='ACTUAL_SINGLETON_CORE_SCORE_GEOMETRY_PLUS_PUBLIC_STRUCTURAL_PREDICT; C_REUSE_HAS_NO_EXTRA_CALL'))
        streams={}
        for stage,name in score.STREAMS.items():
            lines(out/name,records[stage]);streams[stage]=dict(path=name,record_count=len(records[stage]),file_bytes=(out/name).stat().st_size)
        lines(out/'predictions.jsonl',records['C']);lines(out/'query_score_work.jsonl',work)
        dump(out/'state_manifest.json',dict(schema='d92_group_barrier_joint_state_archive_v1',method=score.METHOD,status='COMPLETE',files=archive,file_count=len(archive)))
        binding=dict(run_id='synthetic_run',row_id=rid,release_commit=COMMIT,capsule_id=cohorts[co]['expected_capsule_id'],checkpoint_sha256=sha,model_seed=seed)
        public=dict(binding,schema=score.PREDICTION_SCHEMA,method=score.METHOD,algorithm=CONFIG['algorithm'],
            group_barrier_resources=CONFIG['group_barrier_resources'],ordered_ground_classes=NATIVE,old_classes=OLD,
            source_identity=source,ground_packet_identity=ground,run_binding=dict(prediction_output_root=str(out),
                row_root=row['row_root'],branch_features=row['branch_features'],ground_packet=row['ground_packet'],capsule=cohorts[co]['capsule']),
            splits=metadata,split_count=len(metadata),query_record_count=len(records['C']),query_fit_access=False,source_fit_access=False,truth_read=False,
            checkpoint_loaded=False,encoder_called=False,cross_split_adapted_state_reuse=False,
            C_decision_policy=score.DECISION_POLICY,C_decision_certificate_schema=score.DECISION_SCHEMA,
            A_tie_policy='first_original_native_head_column',B_C_tie_policy='physical_class_id_ascending',
            fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS')
        dump(out/'startup.json',dict(public,config=CONFIG))
        counters=dict.fromkeys(score.FACTOR_COMPONENTS+tuple(score.PEAK_COUNTERS),0)
        counters.update(heads=2,group_gate_forward_wall_seconds=.25,gate_peak_factor_buffer_bytes=64)
        ccalls=sum(w['C_public_predict_calls'] for w in work)
        marker=dict(public,status='COMPLETE',streams=streams,completed_split_count=len(metadata),
            actual_stage_count=sum(1+int(s['new_count']>0) for s in metadata),state_manifest='state_manifest.json',
            query_score_work=dict(path='query_score_work.jsonl',record_count=len(work),file_bytes=(out/'query_score_work.jsonl').stat().st_size),
            compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
            resources=dict(actual_training_counters=counters,actual_factorization_attempts=0,
                query_score_calls=dict(A=len(work),B=len(work),C=ccalls),C_public_predict_calls=ccalls,
                query_score_seconds=dict(A=.003*len(work),B=.002*len(work),C=.002*ccalls),
                C_public_predict_seconds=sum(w['C_public_predict_seconds'] for w in work),
                work_aggregation='SUM',peak_aggregation='MAX'))
        dump(out/'predictions_complete.json',marker)
        lanes[rid]=dict(status='COMPLETE',release_commit=COMMIT,expected_model_seed=seed,expected_checkpoint_sha256=sha,
            expected_capsule_id=binding['capsule_id'],output_root=str(rowout),prediction_output=str(out),marker=marker,
            source_paths={k:public['run_binding'][k] for k in ('row_root','branch_features','ground_packet','capsule')})
        pending[rid]=dict(lanes[rid],status='PENDING');bindings[rid]=binding
    spec=dict(schema='d92_group_barrier_joint_query_benchmark_v1',run_id='synthetic_run',group_id='synthetic_group',
        code=dict(commit='b'*40),execution=dict(remote_run_root=str(root)),benchmark=dict(config=CONFIG,cohorts=cohorts),rows=declarations)
    startup=dict(schema=spec['schema'],status='GROUP_BARRIER_QUERY_SUPERVISOR_STARTED',run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=COMMIT,code_commit=spec['code']['commit'],resolved_spec=spec,rows=pending)
    dump(root/'startup.json',startup)
    dump(root/'complete.json',dict(startup,status='GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_COMPLETE',rows=lanes,row_count=len(lanes),
        completed_row_count=len(lanes),all_predictions_fixed=True,truth_read=False,scorer_invoked=False,automatic_retry=False))
    return dict(spec=spec,root=root,bindings=bindings,output=tmp_path/'scores.json')


def load_one(case):
    row=case['spec']['rows'][0];co=case['spec']['benchmark']['cohorts'][row['cohort']]
    return score.load_fixed_predictions(predictions=Path(row['output_root'])/'predictions',capsule=co['capsule'],
        config=CONFIG,expected_binding=case['bindings'][row['row_id']])


def test_full_2400_parent_matrix_truth_last_pairing_K1_new0_macroF1_and_seed_SD(tmp_path,monkeypatch):
    case=bundle(tmp_path,full=True);original=score.read;markers=[]
    truths={co['truth'] for co in case['spec']['benchmark']['cohorts'].values()}
    def monitored(path):
        if Path(path).name=='predictions_complete.json':markers.append(str(path))
        if str(path) in truths:assert len(markers)==8
        return original(path)
    monkeypatch.setattr(score,'read',monitored)
    result=score.score_benchmark(spec=case['spec'],output=case['output'])
    assert result['parent_count']==2400 and result['row_count']==4 and result['status']==score.STATUS
    assert result['automatic_promotion'] is False and result['prediction_validation_complete_before_truth'] is True
    for p in result['parents']:
        assert p['metrics']['A_old_accuracy']==.5 and p['metrics']['B_old_accuracy']==5/6
        assert p['metrics']['C_old_accuracy']==5/6
        assert p['metrics']['C_macro_f1'] is not None
        if not p['new_count']:
            assert all(p['metrics'][k] is None for k in ('C_new_accuracy','C_h','C_abs_new_old_gap','C_new_macro_f1'))
        else:assert p['metrics']['C_h']==pytest.approx(10/11)
    assert all(p['metrics']['A_old_accuracy'] is not None for p in result['parents'] if p['k']==1)
    seedrows=result['statistics']['model_seed_mean_sd']['by_k_new_count']
    assert all(r['sample_standard_deviation']==0. for r in seedrows if r['mean'] is not None)
    assert score.read(case['output'])==result
    # A late failed row cannot trigger scoring of the already verified prefix.
    last=Path(case['spec']['rows'][-1]['output_root'])/'predictions'/'predictions_B.jsonl'
    last.write_text('',encoding='utf-8')
    def guarded(path):
        assert str(path) not in truths,'No partial-prefix truth access'
        return original(path)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):score.score_benchmark(spec=case['spec'],output=tmp_path/'never_scored.json')


def test_rounded_logscore_argmax_may_differ_from_valid_structured_C():
    b=[0.,1e-16,0.,0.,0.,0.];cert,values=decision(b,[0.,0.],100.)
    classes=sorted(OLD+['new_00','new_01'])
    c=dict(split_id='x',query_id='opaque',classes=classes,scores=values,prediction='old_1')
    record=dict(c,classes=OLD,scores=b)
    assert classes[int(np.argmax(values))]=='old_0'
    score.verify_structured_decision(cert,c,record,dict(new_count=2,old_classes=OLD,registered_classes=classes))


def test_exact_group_tie_uses_lexical_physical_class_and_native_A_column_order(tmp_path):
    cert,values=decision([0.]*6,[0.]*6,0.)
    classes=sorted(OLD+cert['new_classes'])
    assert cert['group_gap']==0. and cert['prediction']=='new_00'
    b=dict(split_id='x',query_id='q',classes=OLD,scores=[0.]*6,prediction=OLD[0])
    score.verify_structured_decision(cert,dict(b,classes=classes,scores=values,prediction='new_00'),b,
        dict(new_count=6,old_classes=OLD,registered_classes=classes))
    case=bundle(tmp_path);row=case['spec']['rows'][0];root=Path(row['output_root'])/'predictions'
    marker=score.read(root/'predictions_complete.json')
    _,splits=score.load_capsule_metadata(case['spec']['benchmark']['cohorts']['rx1']['capsule'],'synthetic_rx1')
    path=root/score.STREAMS['A'];items=[json.loads(v) for v in path.read_text().splitlines()]
    items[0].update(scores=[0.]*6,prediction=NATIVE[0]);lines(path,items)
    marker['streams']['A']['file_bytes']=path.stat().st_size
    score._stream(root,'A',marker,splits,NATIVE)
    assert NATIVE[0]!=OLD[0]


@pytest.mark.parametrize('change',['old_winner','new_winner','gap','gate','logprob','classes','prediction','actual_B','new0'])
def test_structured_C_tampering_rejects(change):
    cert,values=decision([0.,1.,0.,0.,0.,0.],[1.,0.],2.)
    classes=sorted(OLD+['new_00','new_01']);b=dict(split_id='x',query_id='q',classes=OLD,scores=[0.,1.,0.,0.,0.,0.],prediction='old_1')
    c=dict(b,classes=classes,scores=values,prediction=cert['prediction'])
    if change=='old_winner':cert['old_winner']='old_2'
    elif change=='new_winner':cert['new_winner']='new_01'
    elif change=='gap':cert['group_gap']+=1.
    elif change=='gate':cert['gate_logit']+=1.
    elif change=='logprob':cert['old_log_probabilities'][0]+=.1
    elif change=='classes':cert['new_classes'].pop()
    elif change=='prediction':cert['prediction']=c['prediction']='new_01'
    elif change=='actual_B':cert['old_raw_scores'][0]+=.1
    else:cert['new0_reuses_actual_B']=True
    with pytest.raises(ValueError):score.verify_structured_decision(cert,c,b,dict(new_count=2,old_classes=OLD,registered_classes=classes))


@pytest.mark.parametrize('change',['inherit','namespace','source','runtime','limits','missing','duplicate','work_order','work_missing','C_classes','forbidden_truth'])
def test_fixed_row_binding_corruption_never_opens_truth(tmp_path,monkeypatch,change):
    case=bundle(tmp_path);row=case['spec']['rows'][0];root=Path(row['output_root'])/'predictions'
    marker=score.read(root/'predictions_complete.json')
    if change=='inherit':marker['splits'][1]['c_inherited_from_b_state_ref']=marker['splits'][0]['b_state_ref']
    elif change=='namespace':marker['splits'][1]['c_state_ref']['namespace']='{}'
    elif change=='source':marker['source_identity']['target_access_before_freeze']=True
    elif change=='runtime':marker['release_commit']='c'*40
    elif change=='limits':marker['group_barrier_resources']['max_newton_iterations']+=1
    elif change in ('work_order','work_missing'):
        path=root/'query_score_work.jsonl';items=[json.loads(v) for v in path.read_text().splitlines()]
        if change=='work_order':items[0],items[1]=items[1],items[0]
        else:items.pop()
        lines(path,items);marker['query_score_work']['file_bytes']=path.stat().st_size
    else:
        path=root/'predictions_C.jsonl';items=[json.loads(v) for v in path.read_text().splitlines()]
        if change=='missing':items.pop()
        elif change=='duplicate':items[1]=deepcopy(items[0])
        elif change=='C_classes':items[0]['classes'].pop()
        else:items[0]['truth']='old_0'
        lines(path,items);marker['streams']['C']['file_bytes']=path.stat().st_size
    dump(root/'predictions_complete.json',marker);original=score.read
    def guarded(path):
        assert not str(path).endswith('_truth.json'),'Premature truth access'
        return original(path)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):load_one(case)


def test_subset_spec_rejected_before_any_truth_or_marker_read(tmp_path,monkeypatch):
    case=bundle(tmp_path)
    monkeypatch.setattr(score,'read',lambda p:pytest.fail('Subset must reject before reading inputs'))
    with pytest.raises(ValueError):score.score_benchmark(spec=case['spec'],output=case['output'])


def test_only_ids_npz_member_and_no_training_imports(tmp_path,monkeypatch):
    case=bundle(tmp_path);original=np.load;members=[]
    class IDsOnly:
        def __init__(self,path,**kw):self.data=original(path,**kw)
        def __enter__(self):return self
        def __exit__(self,*args):self.data.close()
        def __getitem__(self,key):
            assert key=='ids';members.append(key);return self.data[key]
    monkeypatch.setattr(np,'load',IDsOnly);load_one(case);assert members==['ids']
    import ast
    tree=ast.parse((ROOT/'tools/score_d92_group_barrier_joint_benchmark.py').read_text(encoding='utf-8'))
    modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    modules|={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
    assert not any(any(k in v for k in ('cvsrffi','torch','evaluate_','export_','run_','packet')) for v in modules)


def test_counter_SUM_MAX_float_seconds_unknown_and_nonfinite_rejection():
    rows=[dict(resources=dict(actual_training_counters=dict(count=2,group_gate_forward_wall_seconds=.25,gate_peak_factor_buffer_bytes=40,one_row=1))),
        dict(resources=dict(actual_training_counters=dict(count=3,group_gate_forward_wall_seconds=.5,gate_peak_factor_buffer_bytes=30)))]
    out=score.aggregate_training_resources(rows)['actual_training_counters']
    assert out==dict(count=5,group_gate_forward_wall_seconds=.75,gate_peak_factor_buffer_bytes=40,one_row=None)
    for value in (True,float('nan'),float('inf'),-.1):
        rows[0]['resources']['actual_training_counters']['group_gate_forward_wall_seconds']=value
        with pytest.raises(ValueError):score.aggregate_training_resources(rows)


def test_exclusive_output_and_algorithm_resource_contract(tmp_path):
    output=tmp_path/'preserve.json';output.write_text('original',encoding='utf-8')
    with pytest.raises(ValueError,match='Exclusive'):score.score_benchmark(spec={},output=output)
    assert output.read_text()=='original'
    for key,value in [('max_newton_iterations',True),('max_line_search_trials',0),('max_factor_buffer_bytes',1.5)]:
        config=deepcopy(CONFIG);config['group_barrier_resources'][key]=value
        with pytest.raises(ValueError):score.validate_config(config)
    config=deepcopy(CONFIG);config['algorithm']['temperature']=2.
    with pytest.raises(ValueError):score.validate_config(config)


def test_full_matrix_contract_cannot_be_resized_or_swap_registered_seed():
    cohorts={co:dict(matrix=dict(receivers=score.RECEIVERS[co],scenarios=score.SCENARIOS,ks=list(score.KS),
        new_counts=list(score.NEWS),support_seeds=score.SUPPORT_SEEDS),expected_split_count=n)
        for co,n in score.COHORT_COUNTS.items()}
    rows=[dict(row_id=co+'_'+str(seed),cohort=co,expected_model_seed=seed,
        expected_checkpoint_sha256=('1' if seed==score.MODEL_SEEDS[0] else '2')*64,
        ground_packet='packet_'+str(seed),output_root='/synthetic/'+co+'_'+str(seed))
        for co,seed in itertools.product(score.COHORT_COUNTS,score.MODEL_SEEDS)]
    spec=dict(rows=rows,benchmark=dict(cohorts=cohorts),execution=dict(remote_run_root='/synthetic'))
    score.validate_declared_matrix(spec)
    changed=deepcopy(spec);changed['rows'].pop()
    with pytest.raises(ValueError):score.validate_declared_matrix(changed)
    changed=deepcopy(spec);changed['rows'][-1]['expected_model_seed']=123
    with pytest.raises(ValueError):score.validate_declared_matrix(changed)
    changed=deepcopy(spec);changed['benchmark']['cohorts']['rx3']['matrix']['support_seeds'].pop()
    with pytest.raises(ValueError):score.validate_declared_matrix(changed)


def test_parent_macro_F1_is_independent_of_accuracy_and_H_averaging(tmp_path):
    case=bundle(tmp_path);fixed=load_one(case)
    truth=score.read(case['spec']['benchmark']['cohorts']['rx1']['truth'])
    sid=next(sid for sid,s in fixed['splits'].items() if s['new_count']==2)
    parent=score._truth_parent(fixed,sid,truth)
    # A has three correct classes. Its two wrong classes go to old_5 and
    # old_5 goes to old_4: 3 perfect F1s out of six, despite new-class records.
    assert parent['metrics']['A_old_macro_f1']==.5
    # B has five correct; old_5 is predicted as old_0, so F1_0=2/3.
    assert parent['metrics']['B_old_macro_f1']==pytest.approx((4+2/3)/6)
    assert parent['metrics']['C_new_macro_f1']==1.


def test_independent_algorithm_literal_matches_frozen_source_AST_without_importing_core():
    import ast
    path=ROOT/'code/cvsrffi/d92_group_barrier_joint_local_ridge.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    call=next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(v,ast.Name) and v.id=='FROZEN_CONFIG' for v in n.targets))
    actual={}
    for item in call.keywords:
        if item.arg=='B_objective':actual[item.arg]='RMS_across_classes_of_cross_fold_mean_CE_only'
        elif item.arg=='barrier_average_original_objective_gap':actual[item.arg]=1e-4
        else:actual[item.arg]=ast.literal_eval(item.value)
    assert actual==score.FROZEN_ALGORITHM


@pytest.mark.parametrize('kind',['missing_marker_seconds','negative_A','missing_public_seconds','wrong_public_sum',
    'negative_B_work','missing_C_work_seconds','nonfinite_public','bool_seconds'])
def test_measured_seconds_missing_negative_or_inconsistent_reject_before_truth(tmp_path,monkeypatch,kind):
    case=bundle(tmp_path);row=case['spec']['rows'][0];root=Path(row['output_root'])/'predictions'
    marker=score.read(root/'predictions_complete.json');workpath=root/'query_score_work.jsonl'
    work=[json.loads(v) for v in workpath.read_text().splitlines()]
    if kind=='missing_marker_seconds':marker['resources'].pop('query_score_seconds')
    elif kind=='negative_A':marker['resources']['query_score_seconds']['A']=-.001
    elif kind=='missing_public_seconds':marker['resources'].pop('C_public_predict_seconds')
    elif kind=='wrong_public_sum':marker['resources']['C_public_predict_seconds']+=.1
    elif kind=='negative_B_work':work[0]['B_work']['score_seconds']=-.1
    elif kind=='missing_C_work_seconds':next(v for v in work if v['C_work'] is not None)['C_work'].pop('score_seconds')
    elif kind=='bool_seconds':marker['resources']['query_score_seconds']['B']=True
    else:
        marker['resources']['C_public_predict_seconds']=float('nan')
        with pytest.raises(ValueError):score._fixed_resources(marker,work)
        return
    lines(workpath,work);marker['query_score_work']['file_bytes']=workpath.stat().st_size
    dump(root/'predictions_complete.json',marker);original=score.read
    def guarded(path):
        assert not str(path).endswith('_truth.json'),'Seconds validation must precede truth'
        return original(path)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):load_one(case)


def test_seconds_readback_allows_only_machine_accumulation_error():
    values=[.1]*1000
    score._seconds_sum_readback(sum(values),values,'public_predict')
    with pytest.raises(ValueError):score._seconds_sum_readback(sum(values)+.0001,values,'public_predict')
