"""CPU recovery cannot regenerate, overwrite, or relabel producer provenance."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from experiments.pamf_phase3_open_world import star_binary_graph_score_recovery as m


def fixture(tmp_path):
    root=tmp_path/'runs'/m.RUN;pred=root.parent/m.producer.RUN_ID/'predict';pred.mkdir(parents=True)
    marker=dict(status=m.producer.STATUS,run_id=m.producer.RUN_ID,code_commit=m.PREDICTION_COMMIT)
    (pred/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
    release=tmp_path/'releases'/m.RUN;release.mkdir(parents=True)
    r=dict(run_id=m.RUN,code_commit='c'*40,run_root=str(root),release_root=str(release),python='python',launch_owner='test-owner',
        predictions_root=str(pred),U_truth=str(tmp_path/'private/U.npz'),Q_truth=str(tmp_path/'private/Q.npz'),targets_path=str(release/'targets.json'))
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps(r),encoding='utf-8')
    return r,runtime,root,pred


def test_fixed_completed_source_and_isolated_immutable_output(tmp_path):
    r,path,root,pred=fixture(tmp_path);assert m.validate(r)==r
    for key,value in [('run_id','wrong'),('predictions_root',str(root/'predict')),('extra',True),('code_commit','bad'),('U_truth',str(root/'U.npz'))]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):m.validate(bad)
    (pred/'complete.json').write_text(json.dumps(dict(status='INCOMPLETE',run_id=m.producer.RUN_ID,code_commit=m.PREDICTION_COMMIT)))
    with pytest.raises(ValueError):m.validate(r)


def test_start_is_separate_CPU_process_and_does_not_modify_predictions(tmp_path):
    r,path,root,pred=fixture(tmp_path);before=(pred/'complete.json').read_bytes()
    with patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=321)) as spawn:m.start(path)
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']==''
    assert spawn.call_args.kwargs['cwd']==r['release_root']
    assert (pred/'complete.json').read_bytes()==before
    assert json.loads((root/'launch.json').read_text())['pid']==321
    with pytest.raises(FileExistsError):m.start(path)


def test_driver_scores_once_and_preserves_original_and_recovery_ids(tmp_path):
    r,path,root,pred=fixture(tmp_path);root.mkdir()
    result=dict(status=m.scorer.STATUS,run_id=m.producer.RUN_ID,code_commit=m.PREDICTION_COMMIT,
        primary_state='DA1_REG1',primary_pass=False,next_step='NO_PROMOTION',truth_last=True)
    with patch.object(m.scorer,'score',return_value=result) as score:m.driver(path)
    score.assert_called_once_with(r['predictions_root'],r['U_truth'],r['Q_truth'],r['targets_path'],root/'score')
    state=json.loads((root/'pipeline_state.json').read_text())
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['code_commit']=='c'*40
    assert state['prediction_code_commit']==m.PREDICTION_COMMIT and state['GPU_workers']==0
    assert not state['predictions_regenerated'] and not state['primary_pass']
    with pytest.raises(FileExistsError):m.driver(path)
