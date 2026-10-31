"""Input fitting, legal predictions, and independent CPU truth boundaries."""
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import receiver_input_launch as m


def setup(root,mp):
    from experiments.pamf_phase3_open_world import receiver_input_train as train,receiver_input_run as run
    mp.setattr(train,'validate_config',lambda c:c);mp.setattr(run,'validate_config',lambda c:c)
    root.mkdir();(root/'logs').mkdir()
    ref={'train_config':{'source':'reference'},'output_root':'FROZEN_REFERENCE'}
    t=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref['train_config'],reference_prediction_root=ref['output_root'],output_root=str(root/'train'))
    c=dict(run_id=m.RUN,code_commit='a'*40,train_config=t,reference_config=ref,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='input-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=c,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_gpu_free',lambda r:[5]);return r,path


def terminal(r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':a.update(status='RECEIVER_INPUT_TRAINING_COMPLETE',updates_per_arm=m.STEPS,total_optimizer_updates=2*m.STEPS,
        source_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_adapters_changed=True,
        NN_optimizer_updates=0,target_Q_reads=0,validation_reads=0,novel_validation_reads=0,private_truth_reads=0,U_truth_reads=0,
        source_L_reads=m.SOURCE_COUNT,target_U_reads=m.U_COUNT)
    elif stage=='predict':a.update(status='RECEIVER_INPUT_PREDICTIONS_COMPLETE',all_adapters_frozen_before_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,all_thresholds_inherited_unchanged=True,
        controls_reused_exactly=True,original_models_unchanged=True,classification_unchanged=False,new_density_fits=0,new_threshold_sets_fitted=0,
        private_truth_reads=0,U_truth_reads=0,target_U_reads=0)
    else:a.update(status='RECEIVER_INPUT_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',adaptation_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);return a,out/'complete.json'


def test_source_and_U_fit_then_predict_then_CPU_truth_no_restart(tmp_path,monkeypatch):
    r,path=setup(tmp_path/'run',monkeypatch);events=[]
    def job(r,state,p,stage,argv,gpu):
        events.append((stage,argv,gpu));a,out=terminal(r,stage);m._write(out,a);return True
    monkeypatch.setattr(m,'_job',job);m.driver(path)
    assert [e[0] for e in events]==['train','predict','score'] and [e[2] for e in events]==[5,5,None]
    for private in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):
        assert all(private not in e[1] for e in events[:2]) and private in events[2][1]
    assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='ANALYZED_DEVELOPMENT'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)
    with pytest.raises(ValueError):m.validate(dict(r,quota=2))


def test_illegal_fit_or_predict_stops_before_truth(tmp_path,monkeypatch,subtests):
    cases=(('train','source_completed_before_U_open',False),('train','source_sampling_matched',False),('train','target_Q_reads',1),('train','NN_optimizer_updates',1),('train','target_U_reads',0),
        ('predict','all_adapters_frozen_before_Q',False),('predict','all_thresholds_inherited_unchanged',False),('predict','classification_unchanged',True),('predict','private_truth_reads',1))
    for stage,field,value in cases:
        with subtests.test(stage=stage,field=field):
            r,path=setup(tmp_path/(stage+field),monkeypatch);events=[]
            def job(r,state,p,current,argv,gpu):
                events.append(current);a,out=terminal(r,current)
                if current==stage:a[field]=value
                m._write(out,a);return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            assert 'score' not in events and m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
