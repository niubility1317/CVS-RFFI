import json
from types import SimpleNamespace

import pytest

from experiments.pamf_phase3_restoration import registered_restoration_dispatch as registered
from experiments.pamf_phase3_restoration import packet_filter_resource_dispatch as dispatch


def test_mismatched_waits_for_actual_modulation_gpu_end_not_score(tmp_path):
    case=registered.CASES['modulation'];out=tmp_path/'runs'/case['run'];out.mkdir(parents=True)
    assert not registered.predecessor_finished(tmp_path)
    (out/f'terminal-{case["arm"]}-train.json').write_text(json.dumps(dict(exit_code=0)))
    assert not registered.predecessor_finished(tmp_path)
    (out/f'terminal-{case["arm"]}-cross_both.json').write_text(json.dumps(dict(exit_code=0)))
    assert registered.predecessor_finished(tmp_path)
    assert not (out/f'terminal-{case["arm"]}-score.json').exists()


@pytest.mark.parametrize('candidate',['modulation','mismatched'])
def test_fixed_original_code_and_normal_gpu5_route(tmp_path,monkeypatch,candidate):
    case=registered.CASES[candidate];legacy=tmp_path/'releases'/(case['run']+'-release');legacy.mkdir(parents=True)
    (legacy/'release_commit.txt').write_text(case['commit'])
    args=SimpleNamespace(candidate=candidate,root=tmp_path,legacy_release=legacy)
    calls=[]
    monkeypatch.setattr(registered,'predecessor_finished',lambda root:True)
    monkeypatch.setattr(dispatch,'controller',lambda args,**kwargs:calls.append((dispatch.RUN,dispatch.ARM,dispatch.LEGACY_MODULE,dispatch.LEGACY_COMMIT,kwargs)))
    for key in ('RUN','ARM','LEGACY_MODULE','LEGACY_COMMIT'):monkeypatch.setattr(dispatch,key,getattr(dispatch,key))
    registered.run_candidate(args)
    assert calls[0][:4]==(case['run'],case['arm'],case['module'],case['commit'])
    assert calls[0][4]['allowed_gpu']==5 and calls[0][4]['reuse_unstarted_root']


def test_reuse_rejects_any_already_submitted_gpu_stage(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'release_commit.txt').write_text(dispatch.LEGACY_COMMIT)
    out=tmp_path/'runs'/dispatch.RUN;out.mkdir(parents=True)
    (out/f'submitted-{dispatch.ARM}-train.json').write_text('{}')
    args=SimpleNamespace(root=tmp_path,legacy_release=legacy,commit='new-resource',cache=tmp_path/'cache')
    with pytest.raises(ValueError,match='unstarted'):dispatch.controller(args,reuse_unstarted_root=True)


def test_reuse_preserves_original_launch_before_any_worker(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'release_commit.txt').write_text(dispatch.LEGACY_COMMIT)
    out=tmp_path/'runs'/dispatch.RUN;(out/'logs').mkdir(parents=True);(out/'rows'/dispatch.ARM/'prediction').mkdir(parents=True)
    original='{"old_owner":"retired"}';(out/'launch.json').write_text(original)
    args=SimpleNamespace(root=tmp_path,legacy_release=legacy,commit='new-resource',cache=tmp_path/'cache')
    monkeypatch.setattr(dispatch,'recovery_gpu_choice',lambda **kwargs:[dict(GPU=5,CUDA_ID='GPU-card5')])
    monkeypatch.setattr(dispatch.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(pid=7,wait=lambda:2))
    with pytest.raises(RuntimeError):dispatch.controller(args,reuse_unstarted_root=True)
    assert (out/'launch.json').read_text()==original
    assert json.loads((out/'resource_controller.json').read_text())['original_launch_preserved']
