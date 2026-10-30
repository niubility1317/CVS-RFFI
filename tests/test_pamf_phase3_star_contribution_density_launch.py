from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_contribution_density_launch as launch


def setup(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_contribution_density_run as run
    monkeypatch.setattr(run,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir()
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',
        launch_owner='density-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
        run_config=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(root/'predict')),novel_truth='private-only-score')
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps(r),encoding='utf-8')
    monkeypatch.setattr(launch,'_gpu_free',lambda x:[7])
    return r,runtime


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='predict':m.update(status='STAR_CONTRIBUTION_DENSITY_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
        private_truth_reads=0,real_F_forward_rows=30150,synthetic_F_forward_rows=2,G_forward_rows=0,G_checkpoint_loaded=False,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:m.update(status='STAR_CONTRIBUTION_DENSITY_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE',source_gate_passed=False,next_step='NO_PROMOTION')
    p=Path(r['run_root'])/('predict' if stage=='predict' else 'score');p.mkdir(exist_ok=True)
    (p/'complete.json').write_text(json.dumps(m),encoding='utf-8');return m


def test_fixed_runtime_and_owned_prediction_path(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(target_truth='bad'),dict(max_added_workers_per_gpu=3),dict(allowed_gpus=[0]),dict(code_commit='uncommitted')):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))
    bad=deepcopy(r);bad['run_config']['output_root']='old-output'
    with pytest.raises(ValueError):launch.validate(bad)


def test_full_prediction_before_truthside_CPU_score(tmp_path,monkeypatch):
    r,runtime=setup(tmp_path,monkeypatch);calls=[]
    def job(r,state,path,name,command,gpu):
        calls.append((name,command,gpu));terminal(r,name);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(runtime)
    assert [c[0] for c in calls]==['predict','score'] and [c[2] for c in calls]==[7,None]
    assert r['novel_truth'] not in calls[0][1] and r['novel_truth'] in calls[1][1]
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_SOURCE_ONLY' and state['source_gate_passed'] is False and state['next_step']=='NO_PROMOTION'
    assert state['full_goal_status']=='ACTIVE' and not state['final_phase3_complete']


@pytest.mark.parametrize('failure',['process','identity','truth_reads','unfrozen_models','wrong_real_forward_count'])
def test_failed_or_illegal_prediction_never_connects_scorer(tmp_path,monkeypatch,failure):
    r,runtime=setup(tmp_path,monkeypatch);calls=[]
    def job(r,state,path,name,command,gpu):
        calls.append(name)
        if failure=='process':return False
        m=terminal(r,name)
        field=dict(identity='identity_predictions_unchanged',truth_reads='private_truth_reads',unfrozen_models='models_frozen_before_V',wrong_real_forward_count='real_F_forward_rows')[failure]
        m[field]=False if field in ('identity_predictions_unchanged','models_frozen_before_V') else 1
        launch._write(Path(r['run_root'])/'predict/complete.json',m);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(runtime)
    assert calls==['predict']
    assert launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_pipeline_and_root_are_preserved(tmp_path,monkeypatch):
    r,runtime=setup(tmp_path,monkeypatch)
    path=Path(r['run_root'])/'pipeline_state.json';path.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.driver(runtime)
    with pytest.raises(FileExistsError):launch.start(runtime)
    assert path.read_text(encoding='utf-8')=='old'
