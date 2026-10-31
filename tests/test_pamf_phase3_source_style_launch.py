"""Source-only style lifecycle and own-head, original-baseline scoring."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_style_launch')


def setup(root,mp):
    m=api();root.mkdir();(root/'logs').mkdir()
    mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c));mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_FRESH');t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'train'))
    e=dict(run_id=m.RUN,code_commit='a'*40,train_config=t,reference_config=ref,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='source-style-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_available',lambda r:[4])
    return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':
        a.update(status='SOURCE_STYLE_TRAIN_COMPLETE',updates_per_arm=444,total_optimizer_updates=888,source_sampling_matched=True,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':
        a.update(status='SOURCE_STYLE_PREDICTIONS_COMPLETE',source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
            all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_control_readouts_reused_exactly=True,classification_unchanged=False,
            source_density_models_fitted=2,new_threshold_sets_fitted=4,training_target_U_count=0,U_ID_metadata_reads=0,
            private_truth_reads=0,target_U_reads=0,U_truth_reads=0)
    else:
        a.update(status='SOURCE_STYLE_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',
            source_style_success=False,scientific_success=False,development_gate_passed=False,new_target_DA=False,adaptation_success=None,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);m._write(out/'complete.json',a);return a


def test_runtime_and_commands_exclude_private_gpu_inputs(tmp_path,monkeypatch,subtests):
    m,r,_=setup(tmp_path/'run',monkeypatch);assert m.validate(r) is r
    for change in (dict(U_truth='PRIVATE'),dict(code_commit='b'*40),dict(max_added_workers_per_gpu=3),dict(minimum_free_mb=0),
                   dict(eval_config=dict(r['eval_config'],reference_config={})),dict(train_config=dict(r['train_config'],output_root='FOREIGN'))):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate(dict(r,**change))
    for stage in ('train','predict'):
        assert all(v not in m._command(r,stage) for v in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'))
    assert 'PRIVATE_STAR' in m._command(r,'score')


def test_real_job_environment_source_scientific_failure_still_completes(tmp_path,monkeypatch):
    m,r,path=setup(tmp_path/'run',monkeypatch);events=[]
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            stage=next(s for s in ('train','predict','score') if '.source_style_'+('run' if s=='predict' else s) in argv[4]);events.append((stage,kwargs));terminal(m,r,stage)
        def wait(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child);m.driver(path)
    assert [v[0] for v in events]==['train','predict','score']
    assert [v[1]['env']['CUDA_VISIBLE_DEVICES'] for v in events]==['4','4','']
    state=m._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['next_step']=='NO_PROMOTION' and state['new_target_DA'] is False
    assert state['adaptation_success'] is None
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)


def test_bad_counts_or_target_reads_stop_before_truth(tmp_path,monkeypatch,subtests):
    cases=(('train','total_optimizer_updates',444),('train','target_U_reads',576),('predict','classification_unchanged',True),
           ('predict','source_density_models_fitted',0),('predict','new_threshold_sets_fitted',0),('predict','U_ID_metadata_reads',576),
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
            assert events==(['train'] if stage=='train' else ['train','predict'])
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
            assert Path(r['run_root'],'train','complete.json').is_file()


def test_actual_training_status_connects_launcher(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import source_style_train as train
    m,r,_=setup(tmp_path/'run',monkeypatch);a=terminal(m,r,'train');a['status']=train.STATUS
    m._write(Path(r['run_root'])/'train/complete.json',a);assert m._terminal(r,'train')['status']==train.STATUS


def score_fixture(s):
    from test_pamf_phase3_native_identity_launch import score_fixture as previous
    proxy=SimpleNamespace(common=s.common,fresh=s.fresh,ARMS=('OLD_PCD','SOURCE_ID','TARGET_ID'),CONDITIONS=s.CONDITIONS,RUN_ID=s.RUN_ID,METHODS=s.METHODS,PRIMARY=s.PRIMARY)
    done,cal,old,nt,truths=previous(proxy);names=dict(E0='OLD_PCD',OLD_PCD='OLD_PCD',SOURCE_NATIVE='SOURCE_ID',STYLE_NATIVE='TARGET_ID')
    rows={'source_DURA':{arm:dict(known=deepcopy(old['source_V'][v]),novel=deepcopy(old['source_NV'][v])) for arm,v in names.items()}}
    rows.update({c:{arm:deepcopy(old['Q_'+c][v]) for arm,v in names.items()} for c in s.CONDITIONS[1:]})
    done.update(training_costs={},combined_new_real_signal_forward_rows=0,combined_new_G_forward_rows=0,combined_new_real_F_forward_rows=0)
    for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows'):done[k]=0
    done['real_signal_forward_rows_by_arm']={a:0 for a in s.ARMS};return done,cal,rows,nt,truths


def test_source_style_score_uses_own_identity_and_does_not_claim_target_DA(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.source_style_score');done,cal,rows,nt,truths=score_fixture(s);events=[]
    monkeypatch.setattr(s,'_prediction',lambda root:(events.append('predictions'),(done,cal,rows))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(events.append('source'),nt)[1]);monkeypatch.setattr(s,'_future_truth',lambda path,*a:(events.append(path),truths[path])[1])
    result=s.score('synthetic','NV','clean_DURA','star_DURA',tmp_path/'score')
    assert events==['predictions','source','clean_DURA','star_DURA'] and result['source_style_success']
    main=result['primary_summary']['star_DURA'];assert main['candidate_macro_closed_accuracy']==pytest.approx(1.)
    assert main['original_F0_macro_closed_accuracy']==pytest.approx(5/6) and main['original_closed_minus_new_open']==pytest.approx(-1/6)
    assert result['adaptation_success'] is None and not result['new_target_DA'] and result['primary_deployed_neural_parameters']==602632
    assert result['next_step']=='SOURCE_STYLE_STAGE_PASSED_REQUIRES_CONFIRMATION'
    assert result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
    assert 'by_RX' in result['paired_primary_vs_controls']['SOURCE_NATIVE_predicted_ratio']['source_DURA']
    i=int(np.flatnonzero(truths['star_DURA']['y']>=6)[0]);rows['star_DURA']['STYLE_NATIVE']['scores'][i,1]=1.
    failed=s.score('synthetic','NV','clean_DURA','star_DURA',tmp_path/'failed')
    assert failed['primary_summary']['star_DURA']['R_U']>.9 and not failed['star_R_U_not_lower_than_all_controls'] and failed['next_step']=='NO_PROMOTION'
    events.clear();monkeypatch.setattr(s,'_prediction',lambda root:(_ for _ in ()).throw(ValueError('partial predictions')))
    with pytest.raises(ValueError,match='partial'):s.score('bad','NV','clean_DURA','star_DURA',tmp_path/'partial')
    assert events==[]
