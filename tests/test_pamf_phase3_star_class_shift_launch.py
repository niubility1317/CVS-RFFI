import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_class_shift_launch as launch


def setup(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_class_shift_run as producer
    monkeypatch.setattr(producer,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir()
    c=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(root/'predict'))
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='mixture-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,run_config=c,source_novel_truth='SOURCE_PRIVATE',clean_truth='CLEAN_PRIVATE',star_truth='STAR_PRIVATE')
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[1]);return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':m.update(status='CLASS_SHIFT_PREDICTIONS_COMPLETE',source_models_frozen_before_U=True,
        adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,private_truth_reads=0,target_U_reads=384,
        new_neural_updates=0,threshold_fit_updates=0,classification_unchanged=True)
    else:m.update(status='CLASS_SHIFT_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',
        adaptation_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/('predict' if stage=='predict' else 'score');out.mkdir(exist_ok=True);launch._write(out/'complete.json',m);return m,out


def test_runtime_resources_identity_and_no_U_truth(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(code_commit='uncommitted'),dict(max_added_workers_per_gpu=3),dict(U_truth='notpermitted'),dict(allowed_gpus=[0])):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))


def test_U_fit_and_all_predictions_before_independent_future_truth(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,stage,argv,gpu):events.append((stage,argv,gpu));terminal(r,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [v[0] for v in events]==['predict','score'] and [v[2] for v in events]==[1,None]
    for name in ('SOURCE_PRIVATE','CLEAN_PRIVATE','STAR_PRIVATE'):assert name not in events[0][1] and name in events[1][1]
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['all_stages_complete'] and not state['adaptation_success']
    assert not state['development_gate_passed'] and not state['final_phase3_complete'] and state['full_goal_status']=='ACTIVE'


@pytest.mark.parametrize('key,value',[('adapters_frozen_before_source_and_Q',False),('source_models_frozen_before_U',False),
    ('private_truth_reads',1),('threshold_fit_updates',1)])
def test_illegal_fit_or_partial_freeze_never_reaches_truth(tmp_path,monkeypatch,key,value):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,stage,argv,gpu):
        events.append(stage);m,out=terminal(r,stage);m[key]=value;launch._write(out/'complete.json',m);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert events==['predict'] and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_run_or_pipeline_cannot_restart(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);path=Path(r['run_root'])/'pipeline_state.json';path.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert path.read_text(encoding='utf-8')=='old'
