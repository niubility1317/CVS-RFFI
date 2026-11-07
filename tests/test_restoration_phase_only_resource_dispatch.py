import json
import os
from types import SimpleNamespace

import pytest

from experiments.pamf_phase3_restoration import phase_only_resource_dispatch as dispatch


def make_original(tmp_path):
    legacy=tmp_path/'release';legacy.mkdir()
    (legacy/'release_commit.txt').write_text(dispatch.LEGACY_COMMIT)
    out=tmp_path/'runs'/dispatch.RUN;out.mkdir(parents=True)
    (out/'logs').mkdir()
    (out/'rows'/dispatch.ARM/'prediction').mkdir(parents=True)
    (out/'launch.json').write_text(json.dumps(dict(pid=4321,code_commit=dispatch.LEGACY_COMMIT,CWD=str(legacy))))
    return out,legacy


def test_handover_rejects_live_or_already_submitted_old_owner(tmp_path):
    out,legacy=make_original(tmp_path)
    proc=tmp_path/'proc'/'4321';proc.mkdir(parents=True)
    command=proc/'cmdline';command.write_bytes(b'old-controller')
    with pytest.raises(ValueError,match='still live'):
        dispatch.verify_unstarted_retired(out,legacy,tmp_path/'proc')
    command.write_bytes(b'')
    submitted=out/f'submitted-{dispatch.ARM}-cross_rx.json';submitted.write_text('{}')
    with pytest.raises(ValueError,match='unstarted'):
        dispatch.verify_unstarted_retired(out,legacy,tmp_path/'proc')
    submitted.unlink()
    dispatch.verify_unstarted_retired(out,legacy,tmp_path/'proc')


def test_handover_rejects_different_original_code(tmp_path):
    out,legacy=make_original(tmp_path)
    (legacy/'release_commit.txt').write_text('different')
    with pytest.raises(ValueError,match='prediction code'):
        dispatch.verify_unstarted_retired(out,legacy,tmp_path/'proc')


def test_legacy_prediction_commands_never_train_and_keep_original_identity(tmp_path):
    args=SimpleNamespace(root=tmp_path,legacy_release=tmp_path/'old-release',cache=tmp_path/'cache')
    for stage in dispatch.STAGES:
        command=dispatch.stage_command(args,stage)
        assert command[3]==dispatch.LEGACY_MODULE and command[4] in ('predict','score')
        assert command[command.index('--commit')+1]==dispatch.LEGACY_COMMIT
        assert command[command.index('--inputs')+1]==str(args.legacy_release/'inputs.json')
    env=dispatch.legacy_environment(tmp_path,args.legacy_release,'GPU-card3')
    assert env['PYTHONPATH'].split(os.pathsep)[0]==str(args.legacy_release)
    assert env['CUDA_VISIBLE_DEVICES']=='GPU-card3'


def test_controller_runs_exactly_two_populations_then_independent_cpu_score(tmp_path,monkeypatch):
    out,legacy=make_original(tmp_path)
    args=SimpleNamespace(root=tmp_path,legacy_release=legacy,cache=tmp_path/'cache',commit='dispatcher-commit')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(dispatch,'verify_unstarted_retired',lambda *args:None)
    calls=[];queries=[]
    def choice(**kwargs):
        queries.append(kwargs)
        return [dict(GPU=3,CUDA_ID='GPU-card3',existing_compute_PIDs=2,free_memory_MB=18000)]
    monkeypatch.setattr(dispatch,'recovery_gpu_choice',choice)
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=100+len(calls);calls.append((command,kwargs))
        def wait(self):
            if len(calls)==3:
                (out/'rows'/dispatch.ARM/'results.json').write_text(json.dumps(dict(status='ANALYZED',run_id=dispatch.RUN,arm=dispatch.ARM,code_commit=dispatch.LEGACY_COMMIT)))
            return 0
    monkeypatch.setattr(dispatch.subprocess,'Popen',Child)
    dispatch.controller(args)
    assert queries==[dict(allowed_gpus=(3,),maximum_processes=3)]*2
    assert [x[0][4] for x in calls]==['predict','predict','score']
    assert [x[1]['env']['CUDA_VISIBLE_DEVICES'] for x in calls]==['GPU-card3','GPU-card3','']
    assert all(x[1]['cwd']==legacy for x in calls)
    assert json.loads((out/'complete.json').read_text())['new_G_training_updates']==0
    assert json.loads((out/'launch.json').read_text())['pid']==4321


def test_child_failure_preserves_and_prevents_further_stages(tmp_path,monkeypatch):
    out,legacy=make_original(tmp_path)
    args=SimpleNamespace(root=tmp_path,legacy_release=legacy,cache=tmp_path/'cache',commit='dispatcher-commit')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(dispatch,'verify_unstarted_retired',lambda *args:None)
    monkeypatch.setattr(dispatch,'recovery_gpu_choice',lambda **kwargs:[dict(GPU=3,CUDA_ID='GPU-card3')])
    calls=[]
    def child(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(pid=81,wait=lambda:2)
    monkeypatch.setattr(dispatch.subprocess,'Popen',child)
    with pytest.raises(RuntimeError,match='preserve'):
        dispatch.controller(args)
    assert len(calls)==1 and not (out/'complete.json').exists()
    assert json.loads((out/f'terminal-{dispatch.ARM}-cross_rx.json').read_text())['exit_code']==2
