import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_body_prototype_launch as launch


def setup(tmp_path,mp):
    from experiments.pamf_phase3_open_world import star_body_prototype_train as train,star_body_prototype_run as predict
    mp.setattr(train,'validate_config',lambda c:c);mp.setattr(predict,'validate_config',lambda c:c)
    root=tmp_path/launch.RUN;root.mkdir();(root/'logs').mkdir();t=dict(run_id=launch.TRAIN_RUN,code_commit=launch.TRAIN_CODE,output_root=str(tmp_path/launch.TRAIN_RUN/'train'))
    e=dict(run_id=launch.RUN,code_commit='a'*40,train_config=t,output_root=str(root/'predict'))
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='ssl-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,source_novel_truth='SOURCE_PRIVATE',clean_truth='CLEAN_PRIVATE',star_truth='STAR_PRIVATE')
    terminal(r,'train');p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');mp.setattr(launch,'_gpu_free',lambda r:[2]);return r,p


def terminal(r,stage):
    m=dict(run_id=launch.TRAIN_RUN if stage=='train' else launch.RUN,code_commit=launch.TRAIN_CODE if stage=='train' else r['code_commit'],final_phase3_complete=False)
    if stage=='train':m.update(status='BODY_PROTOTYPE_TRAINING_COMPLETE',updates_per_arm=444,total_optimizer_updates=888,SOURCE_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True, all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':m.update(status='BODY_PROTOTYPE_PREDICTIONS_COMPLETE',source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0,classification_unchanged=True,source_density_models_fitted=2,new_threshold_sets_fitted=4)
    else:m.update(status='BODY_PROTOTYPE_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',adaptation_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    folder=Path(r['train_config']['output_root']) if stage=='train' else Path(r['run_root'])/stage;folder.mkdir(parents=True,exist_ok=True);launch._write(folder/'complete.json',m);return m,folder


def test_runtime_exact_paths_resources_and_no_target_truth(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for delta in (dict(code_commit='dirty'),dict(max_added_workers_per_gpu=3),dict(U_truth='private'),dict(allowed_gpus=[0]),dict(train_config=dict(r['train_config'],code_commit='c'*40))):
        with pytest.raises(ValueError):launch.validate(dict(r,**delta))
    with pytest.raises(ValueError,match='optimization is forbidden'):launch._command(r,'train')


def test_two_training_arms_then_all_predictions_then_only_CPU_receives_truth(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,s,path,stage,argv,gpu):events.append((stage,argv,gpu));terminal(r,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [v[0] for v in events]==['predict','score'] and [v[2] for v in events]==[2,None]
    for name in ('SOURCE_PRIVATE','CLEAN_PRIVATE','STAR_PRIVATE'):
        assert name not in events[0][1] and name in events[1][1]
    s=launch._read(Path(r['run_root'])/'pipeline_state.json');assert s['status']=='ANALYZED_DEVELOPMENT' and s['all_stages_complete'] and not s['adaptation_success'] and s['training_reused'] and s['this_run_optimizer_updates']==0


@pytest.mark.parametrize('stage,key,value',[('train','source_sampling_matched',False),('train','target_Q_reads',1),('train','original_models_unchanged',False),('train','all_student_Sinc_unchanged',False),('train','all_student_bodies_changed',False),('train','all_student_projections_changed',False),
    ('predict','all_three_DURA_conditions_complete',False),('predict','all_thresholds_frozen_before_novel_and_Q',False),('predict','private_truth_reads',1)])
def test_illegal_training_or_partial_predictions_never_reach_truth(tmp_path,monkeypatch,stage,key,value):
    r,p=setup(tmp_path,monkeypatch);events=[]
    if stage=='train':
        m,out=terminal(r,'train');m[key]=value;launch._write(out/'complete.json',m)
    def job(r,s,path,current,argv,gpu):
        events.append(current);m,out=terminal(r,current)
        if current==stage:m[key]=value;launch._write(out/'complete.json',m)
        return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert 'score' not in events and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_pipeline_is_preserved(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);path=Path(r['run_root'])/'pipeline_state.json';path.write_text('original',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert path.read_text(encoding='utf-8')=='original'
