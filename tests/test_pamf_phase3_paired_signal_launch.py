"""Three-stage handoff preserves data roles and never sends truth to GPU jobs."""
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import paired_signal_launch as m


def setup(root,mp):
    from experiments.pamf_phase3_open_world import paired_signal_train as train,paired_signal_run as predict
    mp.setattr(train,'validate_config',lambda c:c);mp.setattr(predict,'validate_config',lambda c:c)
    root.mkdir();(root/'logs').mkdir();base=dict(run_id=m.RUN,code_commit='a'*40)
    t=dict(base,output_root=str(root/'train'),reference_config={'source_control':'f4b5'})
    e=dict(base,train_config=t,source_control_config=t['reference_config'],output_root=str(root/'predict'))
    r=dict(base,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='paired-signal-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
        train_config=t,eval_config=e,source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r));mp.setattr(m,'_available',lambda r:[5]);return r,path


def terminal(r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':a.update(status='PAIRED_SIGNAL_TRAIN_COMPLETE',updates_per_arm=444,total_optimizer_updates=1332,target_U_reads=576,source_sampling_matched=True,original_models_unchanged=True,
        all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':a.update(status='PAIRED_SIGNAL_PREDICTIONS_COMPLETE',models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,
        all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0,classification_unchanged=True,source_density_models_fitted=3,new_threshold_sets_fitted=6,source_control_readouts_reused_exactly=True)
    else:a.update(status='PAIRED_SIGNAL_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',adaptation_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);return a,out/'complete.json'


def test_training_prediction_then_only_cpu_truths_and_no_restart(tmp_path,monkeypatch):
    r,path=setup(tmp_path/'run',monkeypatch);events=[]
    def job(r,state,p,stage,argv,gpu):
        events.append((stage,argv,gpu));a,out=terminal(r,stage);m._write(out,a);return True
    monkeypatch.setattr(m,'_job',job);m.driver(path)
    assert[e[0]for e in events]==['train','predict','score']and[e[2]for e in events]==[5,5,None]
    for private in('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):assert all(private not in e[1]for e in events[:2])and private in events[-1][1]
    assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='ANALYZED_DEVELOPMENT'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)
    with pytest.raises(ValueError):m.validate(dict(r,U_truth='FORBIDDEN'))


def test_wrong_training_or_incomplete_predictions_stop_before_scoring(tmp_path,monkeypatch,subtests):
    for stage,key,value in[('train','target_Q_reads',1),('train','total_optimizer_updates',444),('train','source_sampling_matched',False),('predict','all_three_DURA_conditions_complete',False),('predict','classification_unchanged',False)]:
        with subtests.test(stage=stage,key=key):
            r,path=setup(tmp_path/(stage+key),monkeypatch);events=[]
            def job(r,state,p,current,argv,gpu):
                events.append(current);a,out=terminal(r,current)
                if current==stage:a[key]=value
                m._write(out,a);return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            assert events==['train','predict','score'][:['train','predict','score'].index(stage)+1]
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_actual_three_student_prediction_terminal(tmp_path,monkeypatch):
    from test_pamf_phase3_paired_signal_run import fixture
    run_root=tmp_path/'actual';run_root.mkdir()
    with fixture(tmp_path,monkeypatch)as(producer,c,*_):
        c['output_root']=str(run_root/'predict');done=producer.run(c,'cpu')
        actual=m._terminal(dict(run_root=str(run_root),code_commit=c['code_commit']),'predict')
        assert actual['status']==done['status']=='PAIRED_SIGNAL_PREDICTIONS_COMPLETE'
