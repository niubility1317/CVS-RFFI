"""Ground lifecycle: private truth only on the final independent CPU stage."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest


def setup(root,mp):
    from experiments.pamf_phase3_open_world import ground_open_launch as m,ground_open_data as data
    root.mkdir();(root/'logs').mkdir()
    mp.setattr(data,'validate_config',lambda c:c)
    mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_FRESH')
    d=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'data'))
    t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,data_root=d['output_root'],output_root=str(root/'train'))
    e=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,data_root=d['output_root'],train_config=t,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='ground-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,data_config=d,train_config=t,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_available',lambda r:[4])
    return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='data':a.update(status='GROUND_OPEN_DATA_COMPLETE',phase3_data_status='VALIDATED_ONCE',source_only=True,role_input_path='G(clean)',
        source_roles_disjoint=True,V_calibration_test_disjoint=True,paired_query_ID_order_equal=True,known_TEST_labels_public=False,
        real_F_forward_rows=36450,G_forward_rows=99284,private_truth_reads=0,target_U_reads=0,target_Q_reads=0,channel_observations_generated=0)
    elif stage=='train':a.update(status='GROUND_OPEN_TRAIN_COMPLETE',updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1420800,
        source_L_OE_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_U_GT_reads=0)
    elif stage=='predict':a.update(status='GROUND_OPEN_PREDICTIONS_COMPLETE',neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        private_truth_reads=0,target_U_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,
        source_query_count=16650,cross_query_count=10238,calibration_known_count=13500,real_student_forward_rows=134552,G_forward_rows=20476)
    else:a.update(status='GROUND_OPEN_SCORES_COMPLETE',truth_last=True,all_four_predictions_before_truth=True,all_12_points_before_truth=True,
        full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None,ground_training_success=False,scientific_success=False,
        development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);m._write(out/'complete.json',a);return a


def test_runtime_and_commands_no_gpu_truth(tmp_path,monkeypatch,subtests):
    m,r,_=setup(tmp_path/'run',monkeypatch);assert m.validate(r) is r
    cases=(dict(U_truth='PRIVATE'),dict(code_commit='b'*40),dict(max_added_workers_per_gpu=3),dict(minimum_free_mb=0),
        dict(data_config=dict(r['data_config'],output_root='OTHER')),
        dict(eval_config=dict(r['eval_config'],data_root='OTHER')),dict(train_config=dict(r['train_config'],reference_config={})))
    for change in cases:
        with subtests.test(key=next(iter(change))),pytest.raises(ValueError):m.validate(dict(r,**change))
    for stage in ('data','train','predict'):
        a=m._command(r,stage);assert not any(v.startswith('PRIVATE_') for v in a) and '--source-known-truth' not in a
    a=m._command(r,'score');assert '--source-known-truth' in a and 'PRIVATE_STAR' in a


def test_actual_job_environment_orders_all_stages(tmp_path,monkeypatch):
    m,r,path=setup(tmp_path/'run',monkeypatch);events=[]
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            stage=next(s for s in ('data','train','predict','score') if '.ground_open_'+('run' if s=='predict' else s) in argv[4]);events.append((stage,kwargs));terminal(m,r,stage)
        def wait(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child);m.driver(path)
    assert [v[0] for v in events]==['data','train','predict','score']
    assert [v[1]['env']['CUDA_VISIBLE_DEVICES'] for v in events]==['4','4','4','']
    state=m._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['next_step']=='NO_PROMOTION' and state['primary']=='SSL_energy'
    assert state['adaptation_success'] is None and state['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)


def test_invalid_intermediate_state_stops_before_truth(tmp_path,monkeypatch,subtests):
    cases=(('data','role_input_path','G(G10_received)'),('data','V_calibration_test_disjoint',False),('data','channel_observations_generated',1),
        ('train','source_U_GT_reads',1),('train','real_student_forward_rows',1136640),('predict','all_12_points_complete',False),
        ('predict','source_test_labels_read',13500),('predict','all_thresholds_frozen_before_Q',False),('train','job',False))
    stages=('data','train','predict','score')
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
