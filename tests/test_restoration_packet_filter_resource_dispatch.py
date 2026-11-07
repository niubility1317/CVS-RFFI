import json
from types import SimpleNamespace

import pytest

from experiments.pamf_phase3_restoration import packet_filter_resource_dispatch as dispatch


def args_for(tmp_path):
    legacy=tmp_path/'legacy';legacy.mkdir()
    (legacy/'release_commit.txt').write_text(dispatch.LEGACY_COMMIT)
    (tmp_path/'runs').mkdir()
    return SimpleNamespace(root=tmp_path,legacy_release=legacy,cache=tmp_path/'cache',commit='dispatcher-commit')


def test_commands_keep_original_science_and_full_training_stage(tmp_path):
    args=args_for(tmp_path)
    commands=[dispatch.stage_command(args,stage) for stage in dispatch.STAGES]
    assert [x[4] for x in commands]==['train','predict','predict','score']
    assert all(x[3]==dispatch.LEGACY_MODULE and x[x.index('--commit')+1]==dispatch.LEGACY_COMMIT for x in commands)
    assert all('--task' not in commands[i] for i in (0,3))
    with pytest.raises(ValueError):dispatch.stage_command(args,'resume')


def test_exclusive_root_rejects_any_existing_run(tmp_path,monkeypatch):
    args=args_for(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    (tmp_path/'runs'/dispatch.RUN).mkdir()
    with pytest.raises(FileExistsError):dispatch.controller(args)


def test_four_stages_keep_normal_GPU5_cap_and_independent_score(tmp_path,monkeypatch):
    args=args_for(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    calls=[];queries=[]
    def choice(**kwargs):
        queries.append(kwargs)
        return [dict(GPU=5,CUDA_ID='GPU-card5',existing_compute_PIDs=2,free_memory_MB=18000)]
    monkeypatch.setattr(dispatch,'recovery_gpu_choice',choice)
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=101+len(calls);calls.append((command,kwargs))
        def wait(self):
            row=tmp_path/'runs'/dispatch.RUN/'rows'/dispatch.ARM
            if len(calls)==1:
                (row/'training').mkdir()
                (row/'training/complete.json').write_text(json.dumps(dict(status='SOURCE_TRAINED',code_commit=dispatch.LEGACY_COMMIT,arm=dispatch.ARM,epoch=30,updates=7410)))
            if len(calls)==4:
                (row/'results.json').write_text(json.dumps(dict(status='ANALYZED',run_id=dispatch.RUN,arm=dispatch.ARM,code_commit=dispatch.LEGACY_COMMIT)))
            return 0
    monkeypatch.setattr(dispatch.subprocess,'Popen',Child)
    dispatch.controller(args)
    assert queries==[dict(allowed_gpus=(5,),maximum_processes=3)]*3
    assert [x[1]['env']['CUDA_VISIBLE_DEVICES'] for x in calls]==['GPU-card5']*3+['']
    assert all(x[1]['cwd']==args.legacy_release for x in calls)
    assert json.loads((tmp_path/'runs'/dispatch.RUN/'complete.json').read_text())['all_rows_trained30_and_main_seven_scored']


def test_failed_training_never_enters_prediction_or_score(tmp_path,monkeypatch):
    args=args_for(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(dispatch,'recovery_gpu_choice',lambda **kwargs:[dict(GPU=5,CUDA_ID='GPU-card5')])
    calls=[]
    def child(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(pid=77,wait=lambda:2)
    monkeypatch.setattr(dispatch.subprocess,'Popen',child)
    with pytest.raises(RuntimeError,match='preserve'):dispatch.controller(args)
    assert len(calls)==1 and not (tmp_path/'runs'/dispatch.RUN/'complete.json').exists()
