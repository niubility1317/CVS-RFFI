"""Predictor/independent CPU boundaries and immutable run ownership."""
from pathlib import Path
import json
import pytest
from experiments.pamf_phase3_open_world import pcd_relative_metric_launch as m


def setup(root,mp):
    from experiments.pamf_phase3_open_world import pcd_relative_metric_run as run
    mp.setattr(run,'validate_config',lambda c:c)
    root.mkdir();(root/'logs').mkdir()
    c=dict(run_id=m.RUN,code_commit='a'*40,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='rmd-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,run_config=c,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_gpu_free',lambda r:[4]);return r,path


def terminal(r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':a.update(status='PCD_RELATIVE_METRIC_PREDICTIONS_COMPLETE',models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        private_truth_reads=0,U_truth_reads=0,target_U_reads=0,source_L_feature_reads=6300,
        real_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,optimizer_updates=0,new_threshold_sets_fitted=4,
        source_metric_fits=1,original_models_unchanged=True,classification_unchanged=True)
    else:a.update(status='PCD_RELATIVE_METRIC_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',adaptation_success=None,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);return a,out/'complete.json'


def test_predict_then_truth_only_CPU_immutable_no_restart(tmp_path,monkeypatch):
    r,path=setup(tmp_path/'run',monkeypatch);events=[]
    def job(r,state,p,stage,argv,gpu):
        events.append((stage,argv,gpu));a,out=terminal(r,stage);m._write(out,a);return True
    monkeypatch.setattr(m,'_job',job);m.driver(path)
    assert [e[0] for e in events]==['predict','score'] and [e[2] for e in events]==[4,None]
    for private in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):
        assert private not in events[0][1] and private in events[1][1]
    state=m._read(Path(r['run_root'])/'pipeline_state.json');assert state['status']=='ANALYZED_DEVELOPMENT' and state['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)
    with pytest.raises(ValueError):m.validate(dict(r,quota=2))


def test_cost_role_or_freeze_mismatch_prevents_truth(tmp_path,monkeypatch,subtests):
    for field,value in (('models_frozen_before_V',False),('source_L_feature_reads',0),('real_signal_forward_rows',1),('all_three_DURA_conditions_complete',False),('U_truth_reads',1),('new_threshold_sets_fitted',0),('classification_unchanged',False)):
        with subtests.test(field=field):
            r,path=setup(tmp_path/field,monkeypatch);events=[]
            def job(r,state,p,stage,argv,gpu):
                events.append(stage);a,out=terminal(r,stage);a[field]=value;m._write(out,a);return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            assert events==['predict'] and m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
