import json
from types import SimpleNamespace

import torch

from experiments.pamf_phase3_restoration import signed_covariance_parallel_trial as parallel
from experiments.pamf_phase3_restoration import signed_covariance_trial as method
from experiments.pamf_phase3_restoration import multiview_trial as trial


def runner(counts,free=None,calls=None):
    def run(command,**kwargs):
        gpu=int(command[command.index('-i')+1])
        if calls is not None:calls.append(gpu)
        if '--query-gpu=index,uuid,memory.free' in command:
            available=18000 if free is None else free[gpu]
            return SimpleNamespace(returncode=0,stdout=f'{gpu}, GPU-card{gpu}, {available}\n',stderr='')
        return SimpleNamespace(returncode=0,stdout=''.join(f'GPU-card{gpu}, {100+i}\n' for i in range(counts[gpu])),stderr='')
    return run


def predecessor_marker(root,stage,exit_code):
    folder=root/'runs'/method.predecessor.RUN
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f'terminal-{method.predecessor.ARM}-{stage}.json').write_text(json.dumps(dict(exit_code=exit_code)))


def test_gpu4_can_use_normal_capacity_before_predecessor_finishes(tmp_path):
    choices=parallel.resource_choice(tmp_path,runner({4:2,6:5}))
    assert choices==[dict(GPU=4,CUDA_ID='GPU-card4',free_memory_MB=18000,existing_compute_PIDs=2)]
    calls=[]
    assert parallel.resource_choice(tmp_path,runner({4:3,6:0},calls=calls))==[]
    assert set(calls)=={4}


def test_gpu6_keeps_last_gpu_stage_dependency_and_five_process_limit(tmp_path):
    predecessor_marker(tmp_path,'train',0)
    assert parallel.resource_choice(tmp_path,runner({4:3,6:4}))==[]
    predecessor_marker(tmp_path,'cross_both',0)
    choices=parallel.resource_choice(tmp_path,runner({4:3,6:4}))
    assert len(choices)==1 and choices[0]['GPU']==6 and choices[0]['existing_compute_PIDs']==4
    assert parallel.resource_choice(tmp_path,runner({4:3,6:5}))==[]
    assert parallel.resource_choice(tmp_path,runner({4:2,6:4},free={4:11999,6:11999}))==[]


def test_recipe_and_all_scratch_tensors_are_identical_except_run_id():
    before=method.configuration(method.ARM)
    after=parallel.configuration(parallel.ARM)
    assert after.pop('run_id')==parallel.RUN
    before.pop('run_id')
    assert before==after
    a=method.factory(method.ARM).state_dict()
    b=parallel.method.factory(parallel.ARM).state_dict()
    assert set(a)==set(b) and all(torch.equal(value,b[key]) for key,value in a.items())


def test_wrapper_delegates_training_with_only_run_metadata_override(monkeypatch,tmp_path):
    observed={}
    monkeypatch.setattr(method,'train',lambda args,out,**kwargs:observed.update(args=args,out=out,**kwargs))
    args=SimpleNamespace(root=tmp_path)
    parallel.train(args,tmp_path/'training')
    assert observed==dict(args=args,out=tmp_path/'training',config_for=parallel.configuration)


def test_controller_uses_one_generic_owner_with_bound_resource_root(monkeypatch,tmp_path):
    observed={}
    monkeypatch.setattr(trial,'gpu_choice',None)
    monkeypatch.setattr(parallel,'resource_choice',lambda root:observed.update(root=root) or [])
    monkeypatch.setattr(method,'original_controller',lambda args:observed.update(args=args,choices=trial.gpu_choice()))
    args=SimpleNamespace(root=tmp_path)
    parallel.controller(args)
    assert observed==dict(root=tmp_path,args=args,choices=[])
