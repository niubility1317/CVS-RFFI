"""Matched tail lifecycle and independent CPU-only truth stage."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest


def setup(root,mp):
    from experiments.pamf_phase3_open_world import ground_tail_launch as m
    root.mkdir();(root/'logs').mkdir()
    mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_GROUND',data_root=str(root.parent/'OLD_DATA'))
    t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'train'))
    e=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,train_config=t,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='tail-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_available',lambda r:[4])
    return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':a.update(status='GROUND_TAIL_TRAIN_COMPLETE',updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1136640,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,same_input_teacher=True,original_models_unchanged=True,
        all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0)
    elif stage=='predict':a.update(status='GROUND_TAIL_PREDICTIONS_COMPLETE',neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_controls_reused_exactly=True,
        original_F0_baseline_independent_of_method_columns=True,private_truth_reads=0,target_U_reads=0,source_Us_reads=0,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,source_query_count=16650,cross_query_count=10238,
        calibration_known_count=13500,real_student_forward_rows=134552,G_forward_rows=20476,new_threshold_values=12,reused_threshold_values=6,threshold_values=18)
    else:a.update(status='GROUND_TAIL_SCORES_COMPLETE',truth_last=True,all_four_predictions_before_truth=True,all_12_points_before_truth=True,
        legacy_SSL_overall_exact=True,full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None,ground_training_success=False,
        frozen_rejection_success=False,scientific_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);m._write(out/'complete.json',a);return a


def test_runtime_and_commands_keep_source_roles_and_truth_separate(tmp_path,monkeypatch,subtests):
    m,r,_=setup(tmp_path/'run',monkeypatch);assert m.validate(r) is r
    for change in (dict(U_truth='PRIVATE'),dict(code_commit='b'*40),dict(max_added_workers_per_gpu=3),dict(minimum_free_mb=0),
                   dict(eval_config=dict(r['eval_config'],reference_config={})),dict(train_config=dict(r['train_config'],output_root='FOREIGN'))):
        with subtests.test(key=next(iter(change))),pytest.raises(ValueError):m.validate(dict(r,**change))
    for stage in ('train','predict'):
        a=m._command(r,stage);assert not any(v.startswith('PRIVATE_') for v in a) and '--source-known-truth' not in a
    a=m._command(r,'score');assert '--source-known-truth' in a and 'PRIVATE_STAR' in a and any('OLD_DATA' in v for v in a)


def test_actual_job_environment_orders_stage_and_keeps_goal_active(tmp_path,monkeypatch):
    m,r,path=setup(tmp_path/'run',monkeypatch);events=[]
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            stage=next(s for s in ('train','predict','score') if '.ground_tail_'+('run' if s=='predict' else s) in argv[4]);events.append((stage,kwargs));terminal(m,r,stage)
        def wait(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child);m.driver(path)
    assert [v[0] for v in events]==['train','predict','score']
    assert [v[1]['env']['CUDA_VISIBLE_DEVICES'] for v in events]==['4','4','']
    state=m._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['next_step']=='NO_PROMOTION' and state['primary']=='DTAIL_energy'
    assert state['adaptation_success'] is None and state['full_goal_status']=='ACTIVE' and not state['frozen_rejection_success']
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)


def test_invalid_ancestors_or_incomplete_predictions_stop_before_truth(tmp_path,monkeypatch,subtests):
    cases=(('train','source_Us_reads',1),('train','real_student_forward_rows',1420800),('train','same_input_teacher',False),
        ('predict','all_12_points_complete',False),('predict','original_F0_baseline_independent_of_method_columns',False),
        ('predict','source_controls_reused_exactly',False),('predict','all_thresholds_frozen_before_Q',False),('train','job',False))
    stages=('train','predict','score')
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
            assert events==list(stages[:stages.index(stage)+1])
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
            assert not Path(r['run_root'],'score').exists()
