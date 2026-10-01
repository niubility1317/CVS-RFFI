"""Literal opaque-ID fixtures only; no real data, packets, checkpoints or fitting."""
if __name__=='__main__':
    import ast
    import json
    from pathlib import Path
    task_root=Path(__file__).resolve().parents[1]
    task_paths=[task_root/'tools/score_d92_margin_joint_benchmark.py',task_root/'tools/report_d92_margin_joint_benchmark.py',
        Path(__file__),task_root/'tests/test_report_d92_margin_joint_benchmark.py',task_root/'docs/D92_MARGIN_JOINT_QUERY_SCORER_20261001.md']
    for task_path in task_paths:
        task_bytes=task_path.read_bytes();task_source=task_bytes.decode('utf-8')
        assert not task_bytes.startswith(b'\xef\xbb\xbf') and '\ufffd' not in task_source and '\r' not in task_source
        if task_path.suffix=='.py':ast.parse(task_source,filename=str(task_path))
    print(json.dumps(dict(status='STATIC_ONLY',paths=[str(p) for p in task_paths])))
    raise SystemExit(0)

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import score_d92_margin_joint_benchmark as scorer

COMMIT='a'*40
CHECKPOINT='1'*64
OLD=['old_'+str(j) for j in range(6)]
NATIVE=list(reversed(OLD))
CONFIG=dict(algorithm=dict(schema='d92_margin_joint_local_ridge_v1',method=scorer.METHOD),
    qp_resources=dict(max_transitions=7,max_factor_buffer_bytes=800_000))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False,ensure_ascii=False),encoding='utf-8')


def lines(path,values):
    path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values),encoding='utf-8')


def synthetic_bundle(tmp_path,rows=1):
    capsule=tmp_path/'capsule';capsule.mkdir();(capsule/'splits').mkdir()
    physical=[];lookup={};truth={};splits=[]
    def add(pid,name):
        if pid not in lookup:lookup[pid]=len(physical);physical.append(pid)
        return lookup[pid]
    for k in scorer.KS:
        for new in scorer.NEWS:
            classes=OLD+['new_'+str(j).zfill(2) for j in range(new)]
            support=[];labels=[];query=[]
            for ci,name in enumerate(classes):
                for j in range(k):support.append(add('s_'+str(k)+'_'+name+'_'+str(j),name));labels.append(ci)
                pid='q_'+name;query.append(add(pid,name))
                truth[pid]=dict(pool_role='query',receiver='synthetic_rx',scene='synthetic_scene',transmitter=name,old=name in OLD)
            sid='split_k'+str(k)+'_n'+str(new)
            s=dict(split_id=sid,capsule_id='synthetic_capsule',protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',receiver='synthetic_rx',scenario='synthetic_scene',
                k=k,support_seed=11,registered_classes=classes,support_indices=support,support_labels=labels,query_indices=query)
            splits.append(s);dump(capsule/'splits'/ (sid+'.json'),s)
    dump(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='synthetic_capsule',split_count=len(splits)))
    # A poison IQ member proves only ids is read; its shape/value never matters.
    np.savez(capsule/'received.npz',ids=np.asarray(physical),received=np.asarray([np.nan]))
    truth_path=tmp_path/'opaque_truth.json';dump(truth_path,truth)
    root=tmp_path/'new_run';root.mkdir();declared=[];lane_states={};pending={}
    for row_index in range(rows):
        rid='row_'+str(row_index);out=root/rid/'predictions';out.mkdir(parents=True)
        row=dict(row_id=rid,cohort='synthetic',expected_model_seed=7+row_index,expected_checkpoint_sha256=CHECKPOINT,
            row_root=str(tmp_path/('source_'+rid)),branch_features=str(tmp_path/('features_'+rid)),ground_packet=str(tmp_path/('packet_'+rid)),
            output_root=str(root/rid));declared.append(row)
        source=dict(checkpoint_sha256=CHECKPOINT,model_seed=7+row_index,source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',
            source_role_comparison='EXACT_MATCH',checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,
            cache_schema='d92_branch_received_features_v1',feature_contract=scorer.RAW_FEATURE_CONTRACT)
        ground=dict(path=row['ground_packet'],checkpoint_sha256=CHECKPOINT,model_seed=7+row_index,ordered_classes=NATIVE,
            scale=30.,norm_eps=.0001,packet_total_file_bytes=100,
            source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',feature_contract=dict(checkpoint_sha256=CHECKPOINT,
                cache_key='z_id',source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160))
        archive=[];metadata=[];records={stage:[] for stage in scorer.STREAMS}
        def ref(sid,state):
            namespace=json.dumps(dict(run_id='synthetic_run',row_id=rid,split_id=sid,scope='query_benchmark_support_training',
                fold=None,trial=None,state=state),sort_keys=True,separators=(',',':'))
            meta=dict(shape=[1],dtype='float64',nbytes=8,all_finite=True,nonfinite_count=0)
            full=dict(key='final',namespace=namespace,path='state_arrays/'+str(len(archive)).zfill(8)+'.npz',
                arrays=dict(value=meta),array_summaries=dict(value=dict(norm=0.,minimum=0.,maximum=0.)),
                file_bytes=20,archive_seconds=0.,failed_numeric_state=False)
            archive.append(deepcopy(full));full['arrays']['value']={k:meta[k] for k in ('shape','dtype','nbytes')}
            return full
        for s in splits:
            support_ids=sorted(physical[i] for i in s['support_indices'])
            old_support=sorted(physical[i] for i,y in zip(s['support_indices'],s['support_labels']) if s['registered_classes'][y] in OLD)
            new_support=sorted(set(support_ids)-set(old_support));query_ids=[physical[i] for i in s['query_indices']]
            b=ref(s['split_id'],'B_MARGIN');c=b if not new_support else ref(s['split_id'],'C_MARGIN_seq')
            item=dict(**{k:s[k] for k in ('split_id','receiver','scenario','k','support_seed')},new_count=len(s['registered_classes'])-6,
                old_classes=OLD,registered_classes=sorted(s['registered_classes']),declared_registered_classes=s['registered_classes'],
                support_ids=support_ids,old_support_ids=old_support,new_support_ids=new_support,query_ids=query_ids,query_count=len(query_ids),
                b_state_ref=b,c_state_ref=c,c_inherited_from_b_state_ref=b,c_reuses_b=not bool(new_support),
                B_classes=OLD,C_classes=sorted(s['registered_classes']),support_only_fit=True,query_fit_access=False,source_fit_access=False)
            metadata.append(item)
            for pid in query_ids:
                name=truth[pid]['transmitter']
                # Literal controlled errors: A=1/2 old, B=5/6 old; C old=2/3
                # with new, and new=1 (new0 C reuses B exactly).
                for stage in records:
                    classes=NATIVE if stage=='A' else OLD if stage=='B' else item['registered_classes']
                    correct=(name in OLD and int(name[-1])<3) if stage=='A' else (name in OLD and int(name[-1])<5) if stage=='B' else (
                        name not in OLD or int(name[-1])<(5 if not new_support else 4))
                    winner=name if correct else next(v for v in classes if v!=name)
                    scores=[1. if v==winner else 0. for v in classes]
                    records[stage].append(dict(split_id=s['split_id'],query_id=pid,classes=classes,scores=scores,prediction=winner))
        streams={}
        for stage,name in scorer.STREAMS.items():
            lines(out/name,records[stage]);streams[stage]=dict(path=name,record_count=len(records[stage]),file_bytes=(out/name).stat().st_size)
        lines(out/'predictions.jsonl',records['C'])
        dump(out/'state_manifest.json',dict(schema='d92_margin_joint_state_archive_v1',method=scorer.METHOD,status='COMPLETE',
            files=archive,file_count=len(archive)))
        public=dict(schema=scorer.PREDICTION_SCHEMA,method=scorer.METHOD,run_id='synthetic_run',row_id=rid,release_commit=COMMIT,
            capsule_id='synthetic_capsule',checkpoint_sha256=CHECKPOINT,model_seed=7+row_index,
            algorithm=CONFIG['algorithm'],qp_resources=CONFIG['qp_resources'],ordered_ground_classes=NATIVE,old_classes=OLD,
            source_identity=source,ground_packet_identity=ground,
            run_binding=dict(prediction_output_root=str(out),row_root=row['row_root'],branch_features=row['branch_features'],ground_packet=row['ground_packet'],capsule=str(capsule)),
            splits=metadata,split_count=len(metadata),query_record_count=len(records['A']),query_fit_access=False,source_fit_access=False,truth_read=False)
        dump(out/'startup.json',dict(public,config=CONFIG,device='cpu'))
        marker=dict(public,status='COMPLETE',streams=streams,completed_split_count=len(metadata),
            compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
            resources=dict(actual_training_counters={'synthetic_counter':2},wall_seconds=1.,peak_gpu_memory_bytes=None))
        dump(out/'predictions_complete.json',marker)
        lane_states[rid]=dict(status='COMPLETE',release_commit=COMMIT,expected_model_seed=row['expected_model_seed'],
            expected_checkpoint_sha256=CHECKPOINT,expected_capsule_id='synthetic_capsule',output_root=row['output_root'],
            prediction_output=str(out),source_paths=dict(row_root=row['row_root'],capsule=str(capsule),branch_features=row['branch_features'],ground_packet=row['ground_packet']),marker=marker)
        pending[rid]=dict(lane_states[rid],status='PENDING')
    spec=dict(schema='d92_margin_joint_query_benchmark_v1',run_id='synthetic_run',group_id='synthetic_group',
        code=dict(commit='b'*40,cwd='synthetic_code',environment='synthetic_environment'),
        execution=dict(remote_run_root=str(root),launch_owner='synthetic_root',cpu_lanes=2,blas_threads=2),
        benchmark=dict(config=CONFIG,cohorts=dict(synthetic=dict(capsule=str(capsule),truth=str(truth_path),expected_capsule_id='synthetic_capsule'))),rows=declared)
    startup=dict(schema=spec['schema'],status='MARGIN_QUERY_SUPERVISOR_STARTED',run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=COMMIT,code_commit=spec['code']['commit'],resolved_spec=spec,rows=pending)
    complete=dict(startup,status='MARGIN_QUERY_BENCHMARK_PREDICTIONS_COMPLETE',rows=lane_states,row_count=rows,completed_row_count=rows,
        all_predictions_fixed=True,truth_read=False,scorer_invoked=False,automatic_retry=False)
    dump(root/'startup.json',startup);dump(root/'complete.json',complete)
    return dict(spec=spec,root=root,capsule=capsule,truth=truth_path,output=tmp_path/'new_score.json')


def test_complete_literal_matrix_truth_last_K1_and_new0(tmp_path,monkeypatch):
    case=synthetic_bundle(tmp_path,rows=2);calls=[];original=scorer.read
    def monitored(path):
        if Path(path)==case['truth']:
            assert len([v for v in calls if v.endswith('predictions_complete.json')])==4
        calls.append(str(path));return original(path)
    monkeypatch.setattr(scorer,'read',monitored)
    value=scorer.score_benchmark(spec=case['spec'],output=case['output'])
    assert value['status']==scorer.STATUS and value['parent_count']==40
    assert value['release_commit']!=value['code_commit']
    assert scorer.read(case['output'])==value
    for parent in value['parents']:
        assert parent['metrics']['A_old_accuracy']==.5
        assert parent['metrics']['B_old_accuracy']==5/6
        if parent['new_count']==0:
            assert all(parent['metrics'][m] is None for m in ('C_new_accuracy','C_h','C_abs_new_old_gap'))
        else:assert parent['metrics']['C_h']==pytest.approx(.8)
    assert all(p['metrics']['A_old_accuracy'] is not None for p in value['parents'] if p['k']==1)


def _change_marker(case,mutate):
    path=case['root']/'row_0'/'predictions'/'predictions_complete.json';value=scorer.read(path);mutate(value);dump(path,value)


@pytest.mark.parametrize('kind',['classes','argmax','missing','duplicate','truth','order','alias'])
def test_any_stream_failure_precedes_truth_access(tmp_path,monkeypatch,kind):
    case=synthetic_bundle(tmp_path);root=case['root']/'row_0'/'predictions';path=root/'predictions_C.jsonl'
    values=[json.loads(v) for v in path.read_text().splitlines()]
    if kind=='classes':values[0]['classes']=values[0]['classes'][:-1];values[0]['scores']=values[0]['scores'][:-1]
    elif kind=='argmax':values[0]['prediction']='wrong'
    elif kind=='missing':values.pop()
    elif kind=='duplicate':values[1]=deepcopy(values[0])
    elif kind=='truth':values[0]['truth']='old_0'
    elif kind=='order':values[0],values[1]=values[1],values[0]
    else:values[0]['scores'][0]+=.125
    lines(path if kind!='alias' else root/'predictions.jsonl',values)
    if kind!='alias':_change_marker(case,lambda v:v['streams']['C'].update(file_bytes=path.stat().st_size,record_count=len(values)))
    original=scorer.read
    def guarded(path):
        if Path(path)==case['truth']:raise AssertionError('Truth opened before all predictions validated')
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError):scorer.score_benchmark(spec=case['spec'],output=case['output'])
    assert not case['output'].exists()


@pytest.mark.parametrize('kind',['inherit','namespace','new0','source','release','budget','matrix','incomplete'])
def test_lineage_source_matrix_and_runtime_failures_are_truth_blind(tmp_path,monkeypatch,kind):
    case=synthetic_bundle(tmp_path)
    def mutate(v):
        if kind=='inherit':v['splits'][1]['c_inherited_from_b_state_ref']=v['splits'][0]['b_state_ref']
        elif kind=='namespace':v['splits'][1]['c_state_ref']['namespace']='{}'
        elif kind=='new0':v['splits'][0]['c_reuses_b']=False
        elif kind=='source':v['source_identity']['target_access_before_freeze']=True
        elif kind=='release':v['release_commit']='c'*40
        elif kind=='budget':v['qp_resources']['max_transitions']+=1
        elif kind=='matrix':v['splits'].pop()
        else:v['status']='TECHNICAL_FAILURE'
    _change_marker(case,mutate);original=scorer.read
    def guarded(path):
        if Path(path)==case['truth']:raise AssertionError('Forbidden early truth')
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError):scorer.score_benchmark(spec=case['spec'],output=case['output'])


def test_late_row_failure_never_scores_a_successful_prefix(tmp_path,monkeypatch):
    case=synthetic_bundle(tmp_path,rows=2)
    (case['root']/'row_1'/'predictions'/'predictions_B.jsonl').write_text('',encoding='utf-8')
    original=scorer.read
    def guarded(path):
        if Path(path)==case['truth']:raise AssertionError('Truth should not be opened for a partial prefix')
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError):scorer.score_benchmark(spec=case['spec'],output=case['output'])


def test_truth_role_binding_is_checked_only_after_freeze(tmp_path):
    case=synthetic_bundle(tmp_path);truth=scorer.read(case['truth']);truth['q_old_0']['old']=False;dump(case['truth'],truth)
    with pytest.raises(ValueError,match='truth binding'):scorer.score_benchmark(spec=case['spec'],output=case['output'])


def test_no_model_packet_feature_or_training_imports():
    import ast
    source=ast.parse((ROOT/'tools/score_d92_margin_joint_benchmark.py').read_text(encoding='utf-8'))
    imports={node.module for node in ast.walk(source) if isinstance(node,ast.ImportFrom)}
    imports|={a.name for node in ast.walk(source) if isinstance(node,ast.Import) for a in node.names}
    assert not any(any(term in name for term in ('cvsrffi','torch','evaluate_','export_','run_','packet')) for name in imports)


def test_exclusive_output_is_preserved(tmp_path):
    case=synthetic_bundle(tmp_path);case['output'].write_text('do not overwrite',encoding='utf-8')
    with pytest.raises(ValueError,match='Exclusive'):scorer.score_benchmark(spec=case['spec'],output=case['output'])
    assert case['output'].read_text()=='do not overwrite'


def test_only_opaque_ids_member_is_loaded_and_no_IQ_truth_enters_predictor(tmp_path,monkeypatch):
    case=synthetic_bundle(tmp_path);original=np.load;members=[]
    class IDsOnly:
        def __init__(self,path,**kwargs):self.source=original(path,**kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):self.source.close()
        def __getitem__(self,name):
            members.append(name);assert name=='ids','Scorer must not load IQ or training arrays'
            return self.source[name]
    monkeypatch.setattr(np,'load',IDsOnly)
    scorer.score_benchmark(spec=case['spec'],output=case['output'])
    assert members==['ids','ids']


def test_native_A_tie_and_canonical_B_C_tie_are_validated_separately(tmp_path):
    case=synthetic_bundle(tmp_path);root=case['root']/'row_0'/'predictions'
    marker=scorer.read(root/'predictions_complete.json')
    _,splits=scorer.load_capsule_metadata(case['capsule'],'synthetic_capsule')
    for stage,native in (('A',NATIVE),('B',sorted(OLD))):
        path=root/scorer.STREAMS[stage];records=[json.loads(line) for line in path.read_text().splitlines()]
        records[0]['scores']=[0.]*6;records[0]['prediction']=native[0];lines(path,records)
        marker['streams'][stage]['file_bytes']=path.stat().st_size
        scorer._stream(root,stage,marker,splits,NATIVE)
    assert NATIVE[0]!=sorted(OLD)[0]


def test_independent_reread_detects_change_before_truth(tmp_path,monkeypatch):
    case=synthetic_bundle(tmp_path);original=scorer.load_fixed_predictions;reads=0
    def changed(**kwargs):
        nonlocal reads
        value=original(**kwargs);reads+=1
        if reads==1:
            (Path(kwargs['predictions'])/'predictions.jsonl').write_text('{}\n',encoding='utf-8')
        return value
    monkeypatch.setattr(scorer,'load_fixed_predictions',changed)
    original_read=scorer.read
    def guarded(path):
        if Path(path)==case['truth']:raise AssertionError('Changed frozen predictions must reject before truth')
        return original_read(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError,match='alias'):scorer.score_benchmark(spec=case['spec'],output=case['output'])


def test_reported_actual_work_uses_SUM_two_MAX_and_unknown_not_zero():
    rows=[dict(resources=dict(actual_training_counters=dict(heads=3,margin_qp_peak_factor_buffer_bytes=40,
        margin_qp_peak_explicit_solve_temporary_bytes=9,known_on_one_row=2))),
        dict(resources=dict(actual_training_counters=dict(heads=5,margin_qp_peak_factor_buffer_bytes=30,
        margin_qp_peak_explicit_solve_temporary_bytes=12)))]
    values=scorer.aggregate_training_resources(rows)['actual_training_counters']
    assert values==dict(heads=8,margin_qp_peak_factor_buffer_bytes=40,margin_qp_peak_explicit_solve_temporary_bytes=12,known_on_one_row=None)
