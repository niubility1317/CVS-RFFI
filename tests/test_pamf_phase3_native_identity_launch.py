"""Native identity lifecycle, CPU truth boundary and own-head score evidence."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.native_identity_launch')


def setup(root,mp):
    m=api();root.mkdir();(root/'logs').mkdir()
    mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_FRESH')
    t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'train'))
    e=dict(run_id=m.RUN,code_commit='a'*40,train_config=t,reference_config=ref,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='native-identity-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8')
    mp.setattr(m,'_available',lambda r:[5]);return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':
        a.update(status='NATIVE_IDENTITY_TRAIN_COMPLETE',updates_per_arm=444,total_optimizer_updates=888,SOURCE_completed_before_U_open=True,
            source_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            target_U_reads=576,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':
        a.update(status='NATIVE_IDENTITY_PREDICTIONS_COMPLETE',models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
            all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,controls_reused_exactly=True,classification_unchanged=False,
            new_density_fits=0,new_threshold_sets_fitted=4,new_student_optimizer_updates=888,private_truth_reads=0,target_U_reads=0,U_truth_reads=0)
    else:
        a.update(status='NATIVE_IDENTITY_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',
            adaptation_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);m._write(out/'complete.json',a);return a


def test_runtime_rejects_private_override_or_mismatched_identity(tmp_path,monkeypatch,subtests):
    m,r,path=setup(tmp_path/'run',monkeypatch);assert m.validate(r) is r
    for change in (dict(U_truth='FORBIDDEN'),dict(code_commit='b'*40),dict(max_added_workers_per_gpu=3),dict(minimum_free_mb=0),
                   dict(eval_config=dict(r['eval_config'],reference_config={})),dict(train_config=dict(r['train_config'],output_root='FOREIGN'))):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate(dict(r,**change))
    for stage in ('train','predict'):
        assert all(v not in m._command(r,stage) for v in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'))
    assert 'PRIVATE_STAR' in m._command(r,'score')


def test_actual_job_environments_complete_scientific_failure_and_preserve_outputs(tmp_path,monkeypatch):
    m,r,path=setup(tmp_path/'run',monkeypatch);events=[]
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            stage=next(s for s in ('train','predict','score') if '.native_identity_'+('run' if s=='predict' else s) in argv[4])
            events.append((stage,argv,kwargs));terminal(m,r,stage)
        def wait(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child);m.driver(path)
    assert [v[0] for v in events]==['train','predict','score']
    assert [v[2]['env']['CUDA_VISIBLE_DEVICES'] for v in events]==['5','5','']
    assert all(v[2]['cwd']==r['release_root'] for v in events)
    state=m._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['next_step']=='NO_PROMOTION' and not state['final_phase3_complete']
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)


def test_wrong_terminal_or_job_failure_stops_later_stages(tmp_path,monkeypatch,subtests):
    cases=(('train','total_optimizer_updates',444),('train','SOURCE_completed_before_U_open',False),
           ('predict','classification_unchanged',True),('predict','new_density_fits',1),('predict','new_threshold_sets_fitted',0),
           ('predict','all_three_DURA_conditions_complete',False),('train','job',False))
    for stage,key,value in cases:
        with subtests.test(stage=stage,key=key):
            m,r,path=setup(tmp_path/(stage+key),monkeypatch);events=[]
            def job(r,state,p,current,argv,gpu):
                events.append(current);a=terminal(m,r,current)
                if stage==current:
                    if key=='job':return False
                    a[key]=value;m._write(Path(r['run_root'])/current/'complete.json',a)
                return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            assert events==['train'] if stage=='train' else events==['train','predict']
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
            assert Path(r['run_root'],'train','complete.json').is_file()


def test_gpu_capacity_keeps_foreign_processes_and_allows_one_existing_worker(tmp_path,monkeypatch):
    m=api();r=dict(allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2)
    def snapshot(argv,**kwargs):
        if '--query-gpu=index,memory.free' in argv:return '2, 5000\n3, 3000\n4, 7000\n'
        if '--query-gpu=index,uuid' in argv:return '2, GPU2\n3, GPU3\n4, GPU4\n'
        return '10, GPU2\n11, GPU2\n20, GPU4\n21, GPU4\n'
    monkeypatch.setattr(m.subprocess,'check_output',snapshot)
    def cmdline(path):
        pid=path.parent.name
        return b'foreign_user_job' if pid=='11' else b'python\x00-m\x00experiments.pamf_phase3_open_world.native_identity_run\x00'
    monkeypatch.setattr(Path,'read_bytes',cmdline)
    assert m._available(r)==[2]


def test_actual_training_status_connects_launcher_terminal(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import native_identity_train as training
    m,r,_=setup(tmp_path/'run',monkeypatch);done=terminal(m,r,'train');done['status']=training.STATUS
    m._write(Path(r['run_root'])/'train/complete.json',done)
    assert m._terminal(r,'train')['status']==training.STATUS


def score_fixture(s):
    scenes=s.common.SCENES;rxs=s.common.RECEIVERS;unknown=s.common.UNKNOWN_TX
    spec=[(y,r,d,c) for y in range(6) for r in rxs for d in (1,2,3) for c in scenes]
    ns=[(tx,r,d,c) for tx in unknown for r in rxs for d in (1,2,3) for c in scenes]
    y=np.asarray([v[0] for v in spec],np.int64);base=y.copy();base[y==0]=1
    known=dict(ids=np.asarray(['L'+str(i) for i in range(len(y))]),y=y,rx_i=np.asarray([v[1] for v in spec],np.int64),day_i=np.asarray([v[2] for v in spec],np.int64),scenario=np.asarray([v[3] for v in spec]))
    novel=dict(ids=np.asarray(['N'+str(i) for i in range(len(ns))]),rx_i=np.asarray([v[1] for v in ns],np.int64),day_i=np.asarray([v[2] for v in ns],np.int64))
    nt=dict(ids=novel['ids'],tx=np.asarray([v[0] for v in ns]),scenario=np.asarray([v[3] for v in ns]))
    rows={'source_V':{},'source_NV':{}}
    for arm in s.ARMS:
        rows['source_V'][arm]=dict(known,closed=y.copy() if arm=='TARGET_ID' else base.copy(),baseline_closed=base.copy(),scores=np.full((len(y),2),0. if arm=='TARGET_ID' else -1.))
        rows['source_NV'][arm]=dict(novel,closed=np.zeros(len(ns),np.int64),baseline_closed=np.zeros(len(ns),np.int64),scores=-np.ones((len(ns),2)))
    targets={};qspec=[(y,d,c) for d,labels in [('2021_03_15',range(10)),('2021_03_23',(0,1,2,4,6,7,8,9))] for y in labels for c in scenes]
    qy=np.asarray([v[0] for v in qspec],np.int64);qc=np.where(qy<6,qy,0).astype(np.int64);qb=qc.copy();qb[qy==0]=1
    for condition in s.CONDITIONS[1:]:
        t=dict(ids=np.asarray(['Q'+str(i) for i in range(len(qy))]),y=qy,tx=np.asarray(s.fresh.TX_BY_LABEL)[qy],receiver=np.full(len(qy),34,np.int64),day=np.asarray([v[1] for v in qspec]),scenario=np.asarray(['clean' if condition=='clean_DURA' else v[2] for v in qspec]))
        targets[condition]=t;rows['Q_'+condition]={}
        for arm in s.ARMS:
            scores=np.column_stack([np.where(qy<6,0. if arm=='TARGET_ID' else -1.,-1.)]*2)
            rows['Q_'+condition][arm]=dict(ids=t['ids'],closed=qc.copy() if arm=='TARGET_ID' else qb.copy(),baseline_closed=qb.copy(),scores=scores)
    done=dict(run_id=s.RUN_ID,code_commit='a'*40,method_names=list(s.METHODS),primary_method=s.PRIMARY,
        training_identity=dict(total_optimizer_updates=888,identities={a:{} for a in s.ARMS[1:]}),student_identities={a:{} for a in s.ARMS[1:]})
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    return done,cal,rows,nt,targets


def test_score_own_student_identity_original_baseline_and_truth_last(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.native_identity_score');done,cal,rows,nt,truths=score_fixture(s);events=[]
    monkeypatch.setattr(s.evaluation,'_prediction_gate',lambda root:(events.append('predictions'),(done,None,None,cal,rows))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(events.append('source'),nt)[1])
    monkeypatch.setattr(s,'_future_truth',lambda path,*a:(events.append(path),truths[path])[1])
    result=s.score('synthetic','NV','clean_DURA','star_DURA',tmp_path/'score')
    assert events==['predictions','source','clean_DURA','star_DURA'] and result['development_gate_passed']
    primary=result['primary_summary']['star_DURA']
    assert primary['candidate_macro_closed_accuracy']==pytest.approx(1.)
    assert primary['original_F0_macro_closed_accuracy']==pytest.approx(5/6)
    assert primary['original_closed_minus_new_open']==pytest.approx(-1/6)
    assert result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
    pair=result['paired_primary_vs_OLD_PCD_predicted_ratio']['star_DURA']['counts']
    assert pair['known_closed_correct_net']>0 and not result['classification_unchanged'] and result['A_N'] is None
    events.clear();monkeypatch.setattr(s.evaluation,'_prediction_gate',lambda root:(_ for _ in ()).throw(ValueError('partial predictions')))
    with pytest.raises(ValueError,match='partial'):s.score('partial','NV','clean_DURA','star_DURA',tmp_path/'bad')
    assert events==[] and not (tmp_path/'bad').exists()
