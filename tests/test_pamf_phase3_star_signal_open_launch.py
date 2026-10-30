from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_signal_open_launch as launch


def setup(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_signal_open_train as training,star_signal_open_eval as evaluation
    monkeypatch.setattr(training,'validate_config',lambda c:c);monkeypatch.setattr(evaluation,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir()
    t=dict(run_id=launch.RUN,code_commit='a'*40,arm=launch.ARM,IQ_root='fixed-IQ',feature_root='frozen-source-features',control_root='frozen-control',output_root=str(root/'rows'/launch.ARM/'train'))
    e=dict(t,train_config=t,reference_prediction_root=t['control_root'],output_root=str(root/'rows'/launch.ARM/'predict'))
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='signal-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,train_config=t,eval_config=e,novel_truth='PRIVATE_ONLY_SCORER')
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[4])
    return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],arm=launch.ARM,source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='train':m.update(status='STAR_SIGNAL_OPEN_SOURCE_FROZEN',updates=2220,native_updates=0,open_signal_updates=2220,
        validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
    elif stage=='eval':m.update(status='STAR_SIGNAL_OPEN_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
        native_updates=0,open_signal_updates=2220,private_truth_reads=0,real_signal_forward_rows=42584,synthetic_signal_forward_rows=2,real_F_forward_rows=0,
        synthetic_F_forward_rows=0,G_forward_rows=0,identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:m.update(status='STAR_SIGNAL_OPEN_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE',source_gate_passed=False,eligible_for_target_prereg=False,next_step='NO_PROMOTION')
    p=Path(r['run_root'])/'rows'/launch.ARM/dict(train='train',eval='predict',score='score')[stage];p.mkdir(parents=True,exist_ok=True)
    launch._write(p/'complete.json',m);return m,p


def test_fixed_resources_parent_and_owned_paths(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(target_truth='bad'),dict(max_added_workers_per_gpu=3),dict(code_commit='uncommitted'),dict(allowed_gpus=[0])):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))
    for key,value in [('IQ_root','other-IQ'),('reference_prediction_root','other-control'),('output_root','old-output')]:
        bad=deepcopy(r);bad['eval_config'][key]=value
        with pytest.raises(ValueError):launch.validate(bad)


def test_frozen_training_then_eval_then_CPU_truth_only(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,stage,argv,gpu):events.append((stage,argv,gpu));terminal(r,stage);return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert [x[0] for x in events]==['train','eval','score'] and [x[2] for x in events]==[4,4,None]
    assert all(('PRIVATE_ONLY_SCORER' in command)==(stage=='score') for stage,command,gpu in events)
    m=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert m['status']=='ANALYZED_SOURCE_ONLY' and m['all_stages_complete'] and not m['eligible_for_target_prereg']
    assert m['full_goal_status']=='ACTIVE' and not m['final_phase3_complete']


@pytest.mark.parametrize('stage,field,value',[('train','native_updates',2220),('train','validation_reads',1),('eval','all_thresholds_frozen_before_novel',False),('eval','real_signal_forward_rows',0)])
def test_bad_terminal_stops_next_stage_without_touching_inputs(tmp_path,monkeypatch,stage,field,value):
    r,p=setup(tmp_path,monkeypatch);events=[]
    def job(r,state,path,name,argv,gpu):
        events.append(name);m,folder=terminal(r,name)
        if name==stage:m[field]=value;launch._write(folder/'complete.json',m)
        return True
    monkeypatch.setattr(launch,'_job',job);launch.driver(p)
    assert events==(['train'] if stage=='train' else ['train','eval'])
    assert launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_existing_run_and_pipeline_never_restarts(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch);q=Path(r['run_root'])/'pipeline_state.json';q.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert q.read_text(encoding='utf-8')=='old'
