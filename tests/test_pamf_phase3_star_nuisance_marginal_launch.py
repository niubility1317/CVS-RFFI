"""Prediction is truth-free; only the independent CPU stage receives truths."""
import importlib
import json
from pathlib import Path
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_marginal_launch')


def setup(root,monkeypatch):
    from experiments.pamf_phase3_open_world import star_nuisance_marginal_run as run
    launch=api();monkeypatch.setattr(run,'validate_config',lambda c:c);folder=root/'run';folder.mkdir(parents=True);(folder/'logs').mkdir()
    c=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(folder/'predict'))
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(folder),release_root=str(root/'release'),python='python',launch_owner='nuisance-marginal-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,run_config=c,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    p=root/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[4]);return launch,r,p


def terminal(launch,r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':m.update(status='NUISANCE_MARGINAL_PREDICTIONS_COMPLETE',models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_conditions_complete=True,all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0)
    else:m.update(status='NUISANCE_MARGINAL_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',development_gate_passed=False,next_step='NO_PROMOTION')
    p=Path(r['run_root'])/('predict' if stage=='predict' else 'score');p.mkdir(exist_ok=True);launch._write(p/'complete.json',m);return m,p


def test_only_CPU_score_gets_truth_after_complete_three_condition_predictions(tmp_path,monkeypatch):
    launch,r,p=setup(tmp_path,monkeypatch);events=[]
    with pytest.raises(ValueError):launch.validate(dict(r,target_labels='forbidden'))
    def job(runtime,state,path,stage,argv,gpu):events.append((stage,argv,gpu));terminal(launch,runtime,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [e[0] for e in events]==['predict','score'] and [e[2] for e in events]==[4,None]
    for truth in ('PRIVATE_SOURCE','PRIVATE_CLEAN','PRIVATE_STAR'):assert truth not in events[0][1] and truth in events[1][1]
    m=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert m['status']=='ANALYZED_DEVELOPMENT' and m['full_goal_status']=='ACTIVE' and not m['final_phase3_complete']


def test_prediction_failure_missing_DURA_or_freeze_never_starts_score(tmp_path,monkeypatch,subtests):
    for defect in ('failed_process','missing_DURA','not_frozen','missing_condition'):
        with subtests.test(defect=defect):
            launch,r,p=setup(tmp_path/defect,monkeypatch);events=[]
            def job(runtime,state,path,stage,argv,gpu):
                events.append(stage)
                if defect=='failed_process':return False
                m,folder=terminal(launch,runtime,stage)
                key={'missing_DURA':'all_conditions_through_DURA','not_frozen':'all_thresholds_frozen_before_novel_and_Q','missing_condition':'all_three_conditions_complete'}[defect]
                m[key]=False;launch._write(folder/'complete.json',m);return True
            monkeypatch.setattr(launch,'_job',job);launch.driver(p)
            assert events==['predict'] and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_launch_root_and_pipeline_preserve_owned_artifacts(tmp_path,monkeypatch):
    launch,r,p=setup(tmp_path,monkeypatch);state=Path(r['run_root'])/'pipeline_state.json';state.write_text('original',encoding='utf-8')
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('Do not relaunch existing root'))
    monkeypatch.setattr(launch,'_job',lambda *a:pytest.fail('Do not rerun existing pipeline'))
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert state.read_text(encoding='utf-8')=='original'
