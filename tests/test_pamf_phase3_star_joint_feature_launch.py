from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_joint_feature_launch as launch


def setup(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_joint_feature_run as run
    monkeypatch.setattr(run,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir()
    c=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(root/'predict'))
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='joint-feature-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,run_config=c,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[4]);return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':m.update(status='JOINT_FEATURE_PREDICTIONS_COMPLETE',models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0)
    else:m.update(status='JOINT_FEATURE_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/('predict' if stage=='predict' else 'score');out.mkdir(exist_ok=True);launch._write(out/'complete.json',m);return m,out


def test_exact_owned_output_identity_and_resources(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(max_added_workers_per_gpu=3),dict(allowed_gpus=[0]),dict(code_commit='notcommitted'),dict(target_labels='forbidden')):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))
    bad=deepcopy(r);bad['run_config']['output_root']='old-output'
    with pytest.raises(ValueError):launch.validate(bad)


def test_all_three_predictions_then_only_CPU_receives_three_truths(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,stage,argv,gpu):events.append((stage,argv,gpu));terminal(r,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [v[0] for v in events]==['predict','score'] and [v[2] for v in events]==[4,None]
    for value in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):
        assert value not in events[0][1] and value in events[1][1]
    s=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert s['status']=='ANALYZED_DEVELOPMENT' and s['all_stages_complete'] and not s['development_gate_passed']
    assert s['full_goal_status']=='ACTIVE' and not s['final_phase3_complete']


@pytest.mark.parametrize('field,value',[('all_three_conditions_complete',False),('all_conditions_through_DURA',False),
    ('all_thresholds_frozen_before_novel_and_Q',False),('private_truth_reads',1)])
def test_missing_test_DURA_or_freeze_stops_before_truth(tmp_path,monkeypatch,field,value):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,stage,argv,gpu):
        events.append(stage);m,out=terminal(r,stage);m[field]=value;launch._write(out/'complete.json',m);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert events==['predict'] and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_root_and_pipeline_not_restarted(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);s=Path(r['run_root'])/'pipeline_state.json';s.write_text('original',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert s.read_text(encoding='utf-8')=='original'
