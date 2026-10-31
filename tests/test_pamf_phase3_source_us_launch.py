"""Source Us materialization/training precede all predictions and CPU truths."""
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import source_us_launch as m

def setup(root,mp):
    from experiments.pamf_phase3_open_world import source_us_cache,source_us_train,source_us_run
    for module in (source_us_cache,source_us_train,source_us_run):mp.setattr(module,'validate_config',lambda c:c)
    root.mkdir();(root/'logs').mkdir()
    base=dict(run_id=m.RUN,code_commit='a'*40);k=dict(base,output_root=str(root/'cache'));t=dict(base,output_root=str(root/'train'),Us_public_root=str(root/'cache/U_s'),reference_config={'source_control':'pureSOURCE'})
    e=dict(base,train_config=t,source_control_config=dict(source_control_config=t['reference_config']),output_root=str(root/'predict'))
    r=dict(base,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='source-us-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
        cache_config=k,train_config=t,eval_config=e,source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r));mp.setattr(m,'_gpu_free',lambda r:[5]);return r,path

def terminal(r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='cache':a.update(status='SOURCE_US_G10_CACHE_COMPLETE',original_U_s_reads=56700,Us_labels_read=0,private_truth_reads=0,target_reads=0,G_forward_rows=56700,real_F_forward_rows=0,original_models_unchanged=True)
    elif stage=='train':a.update(status='SOURCE_US_REPLAY_TRAINING_COMPLETE',updates=444,total_optimizer_updates=444,source_sampling_matched=True,source_only=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,Us_ground_truth_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0)
    elif stage=='predict':a.update(status='SOURCE_US_REPLAY_PREDICTIONS_COMPLETE',models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0,classification_unchanged=True,source_density_models_fitted=1,new_threshold_sets_fitted=2,source_control_readouts_reused_exactly=True)
    else:a.update(status='SOURCE_US_REPLAY_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',adaptation_success=None,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);return a,out/'complete.json'

def test_four_stages_then_only_three_CPU_truths_and_no_restart(tmp_path,monkeypatch):
    r,path=setup(tmp_path/'run',monkeypatch);events=[]
    def job(r,state,p,stage,argv,gpu):
        events.append((stage,argv,gpu));a,out=terminal(r,stage);m._write(out,a);return True
    monkeypatch.setattr(m,'_job',job);m.driver(path)
    assert [e[0]for e in events]==['cache','train','predict','score'] and [e[2]for e in events]==[5,5,5,None]
    for private in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):
        assert all(private not in e[1] for e in events[:3]) and private in events[-1][1]
    assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='ANALYZED_DEVELOPMENT'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)
    with pytest.raises(ValueError):m.validate(dict(r,Us_truth='FORBIDDEN'))

def test_labels_role_or_freeze_failure_stops_before_next_stage(tmp_path,monkeypatch,subtests):
    for stage,field,value in [('cache','Us_labels_read',1),('cache','original_U_s_reads',6300),('train','source_sampling_matched',False),('train','target_Q_reads',1),('predict','classification_unchanged',False),('predict','all_three_DURA_conditions_complete',False)]:
        with subtests.test(stage=stage,field=field):
            r,path=setup(tmp_path/(stage+field),monkeypatch);events=[]
            def job(r,state,p,current,argv,gpu):
                events.append(current);a,out=terminal(r,current)
                if current==stage:a[field]=value
                m._write(out,a);return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            expected=['cache','train','predict','score'];assert events==expected[:expected.index(stage)+1]
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'

def test_actual_prediction_artifact_reaches_predict_terminal(tmp_path,monkeypatch):
    from test_pamf_phase3_source_us_run import fixture
    run_root=tmp_path/'actual_launch';run_root.mkdir()
    with fixture(tmp_path,monkeypatch) as (producer,c,_,_,_):
        c['output_root']=str(run_root/'predict')
        done=producer.run(c,'cpu')
        actual=m._terminal(dict(run_root=str(run_root),code_commit=c['code_commit']),'predict')
        assert actual['status']==done['status']=='SOURCE_US_REPLAY_PREDICTIONS_COMPLETE'
