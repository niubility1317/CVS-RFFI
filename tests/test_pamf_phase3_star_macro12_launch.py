from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_macro12_launch as launch


def setup(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_macro12_run as run
    monkeypatch.setattr(run,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir()
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='macro12-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,run_config=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(root/'predict')),novel_truth='PRIVATE_ONLY_SCORE')
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[6]);return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='predict':m.update(status='STAR_MACRO12_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,private_truth_reads=0,native_updates=0,neural_updates=0,
        real_signal_forward_rows=0,synthetic_signal_forward_rows=2,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,controls_reused_exactly=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:m.update(status='STAR_MACRO12_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE',source_gate_passed=False,eligible_for_target_prereg=False,next_step='NO_PROMOTION')
    p=Path(r['run_root'])/stage;p.mkdir(exist_ok=True);launch._write(p/'complete.json',m);return m,p


def test_fixed_owned_config_and_resources(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(target_truth='bad'),dict(max_added_workers_per_gpu=3),dict(code_commit='working-tree'),dict(allowed_gpus=[0])):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))
    bad=deepcopy(r);bad['run_config']['output_root']='old'
    with pytest.raises(ValueError):launch.validate(bad)


def test_complete_predict_before_independent_CPU_truth(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);calls=[]
    def job(r,state,path,stage,command,gpu):calls.append((stage,command,gpu));terminal(r,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [v[0] for v in calls]==['predict','score'] and [v[2] for v in calls]==[6,None]
    assert 'PRIVATE_ONLY_SCORE' not in calls[0][1] and 'PRIVATE_ONLY_SCORE' in calls[1][1]
    s=launch._read(Path(r['run_root'])/'pipeline_state.json');assert s['status']=='ANALYZED_SOURCE_ONLY' and s['all_stages_complete'] and not s['eligible_for_target_prereg']
    assert s['full_goal_status']=='ACTIVE' and not s['final_phase3_complete']


@pytest.mark.parametrize('field,value',[('private_truth_reads',1),('neural_updates',1),('all_thresholds_frozen_before_novel',False),('controls_reused_exactly',False)])
def test_bad_prediction_never_connects_truth(tmp_path,monkeypatch,field,value):
    r,p=setup(tmp_path,monkeypatch);calls=[]
    def job(r,state,path,stage,command,gpu):
        calls.append(stage);m,folder=terminal(r,stage);m[field]=value;launch._write(folder/'complete.json',m);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert calls==['predict'] and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_output_preserved(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);q=Path(r['run_root'])/'pipeline_state.json';q.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert q.read_text(encoding='utf-8')=='old'
