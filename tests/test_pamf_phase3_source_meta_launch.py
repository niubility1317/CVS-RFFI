"""Source-only style lifecycle and own-head, original-baseline scoring."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_meta_launch')


def setup(root,mp):
    m=api();root.mkdir();(root/'logs').mkdir()
    mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c));mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_FRESH');t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'train'))
    e=dict(run_id=m.RUN,code_commit='a'*40,train_config=t,reference_config=ref,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='source-meta-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_available',lambda r:[4])
    return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':
        a.update(status='SOURCE_META_TRAIN_COMPLETE',updates_per_arm=444,total_optimizer_updates=888,source_sampling_matched=True,full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,meta_alpha=.001,meta_beta=.5,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':
        a.update(status='SOURCE_META_PREDICTIONS_COMPLETE',source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
            all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_control_readouts_reused_exactly=True,classification_unchanged=False,
            source_density_models_fitted=2,new_threshold_sets_fitted=4,training_target_U_count=0,U_ID_metadata_reads=0,
            private_truth_reads=0,target_U_reads=0,U_truth_reads=0)
    else:
        a.update(status='SOURCE_META_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',
            source_meta_success=False,scientific_success=False,development_gate_passed=False,new_target_DA=False,adaptation_success=None,next_step='NO_PROMOTION')
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
            stage=next(s for s in ('train','predict','score') if '.source_meta_'+('run' if s=='predict' else s) in argv[4]);events.append((stage,kwargs));terminal(m,r,stage)
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
           ('predict','all_three_DURA_conditions_complete',False),('train','job',False),('train','full_second_order',False),('train','all_receiver_splits_disjoint',False),('train','virtual_updates_committed',True))
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
    from experiments.pamf_phase3_open_world import source_meta_train as train
    m,r,_=setup(tmp_path/'run',monkeypatch);a=terminal(m,r,'train');a['status']=train.STATUS
    m._write(Path(r['run_root'])/'train/complete.json',a);assert m._terminal(r,'train')['status']==train.STATUS

