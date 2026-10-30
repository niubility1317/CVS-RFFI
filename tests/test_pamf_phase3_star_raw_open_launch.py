from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_raw_open_launch as launch


def setup(tmp_path, monkeypatch):
    from experiments.pamf_phase3_open_world import star_raw_open_train as training, star_raw_open_eval as evaluation
    monkeypatch.setattr(training,'validate_config',lambda c:c); monkeypatch.setattr(evaluation,'validate_config',lambda c:c)
    root=tmp_path/'run'; root.mkdir(); (root/'logs').mkdir()
    parent=dict(output_root=str(tmp_path/'signal/rows/E_SIGNAL/train'))
    t=dict(run_id=launch.RUN,code_commit='a'*40,arm=launch.ARM,IQ_root='fixed-IQ',control_root='macro12-control',
           control_train_config=parent,output_root=str(root/'rows'/launch.ARM/'train'))
    e=dict(run_id=launch.RUN,code_commit='a'*40,arm=launch.ARM,train_config=t,
           reference_prediction_root=t['control_root'],signal_reference_root=str(tmp_path/'signal/rows/E_SIGNAL/predict'),
           output_root=str(root/'rows'/launch.ARM/'predict'),batch_size=512)
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',
           launch_owner='raw-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
           train_config=t,eval_config=e,novel_truth='PRIVATE_ONLY_SCORER')
    p=tmp_path/'runtime.json'; p.write_text(json.dumps(r),encoding='utf-8'); monkeypatch.setattr(launch,'_gpu_free',lambda r:[4])
    return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],arm=launch.ARM,source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='train':
        m.update(status='STAR_RAW_OPEN_SOURCE_FROZEN',updates=2220,native_updates=0,open_signal_updates=2220,
                 validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
    elif stage=='eval':
        m.update(status='STAR_RAW_OPEN_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
                 native_updates=0,open_signal_updates=2220,private_truth_reads=0,real_signal_forward_rows=42584,synthetic_signal_forward_rows=2,
                 real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,classification_path='P1',identity_baseline_path='P1',
                 rejection_input_path='P0',teacher_input_path='P0',identity_predictions_unchanged=True,models_frozen_before_V=True,
                 all_thresholds_frozen_before_novel=True,source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:
        m.update(status='STAR_RAW_OPEN_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE',source_gate_passed=False,
                 eligible_for_target_prereg=False,next_step='NO_PROMOTION')
    p=Path(r['run_root'])/'rows'/launch.ARM/dict(train='train',eval='predict',score='score')[stage]
    p.mkdir(parents=True,exist_ok=True); launch._write(p/'complete.json',m); return m,p


def test_fixed_resources_both_parents_and_owned_paths(tmp_path,monkeypatch):
    r,_=setup(tmp_path,monkeypatch); assert launch.validate(r) is r
    for change in (dict(target_truth='bad'),dict(max_added_workers_per_gpu=3),dict(code_commit='uncommitted'),dict(allowed_gpus=[0])):
        with pytest.raises(ValueError): launch.validate(dict(r,**change))
    for key in ('reference_prediction_root','signal_reference_root','output_root'):
        bad=deepcopy(r); bad['eval_config'][key]='other-control-or-output'
        with pytest.raises(ValueError): launch.validate(bad)


def test_train_then_prediction_then_independent_CPU_truth(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch); events=[]
    def job(r,state,path,stage,argv,gpu): events.append((stage,argv,gpu)); terminal(r,stage); return True
    monkeypatch.setattr(launch,'_job',job); launch.driver(p)
    assert [x[0] for x in events]==['train','eval','score'] and [x[2] for x in events]==[4,4,None]
    assert all(('PRIVATE_ONLY_SCORER' in command)==(stage=='score') for stage,command,gpu in events)
    m=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert m['status']=='ANALYZED_SOURCE_ONLY' and m['all_stages_complete'] and not m['eligible_for_target_prereg']
    assert m['primary']=='E_RAW::P0_known12' and m['full_goal_status']=='ACTIVE' and not m['final_phase3_complete']


@pytest.mark.parametrize('stage,field,value',[
    ('train','native_updates',2220),('train','validation_reads',1),('eval','rejection_input_path','P1'),
    ('eval','identity_baseline_path','P0'),('eval','all_thresholds_frozen_before_novel',False)])
def test_invalid_terminal_never_advances(tmp_path,monkeypatch,stage,field,value):
    r,p=setup(tmp_path,monkeypatch); events=[]
    def job(r,state,path,name,argv,gpu):
        events.append(name); m,folder=terminal(r,name)
        if name==stage: m[field]=value; launch._write(folder/'complete.json',m)
        return True
    monkeypatch.setattr(launch,'_job',job); launch.driver(p)
    assert events==(['train'] if stage=='train' else ['train','eval'])
    assert launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_no_duplicate_launch_or_resume(tmp_path,monkeypatch):
    r,p=setup(tmp_path,monkeypatch); q=Path(r['run_root'])/'pipeline_state.json'; q.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError): launch.start(p)
    with pytest.raises(FileExistsError): launch.driver(p)
    assert q.read_text(encoding='utf-8')=='old'
