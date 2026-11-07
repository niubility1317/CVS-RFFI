from types import SimpleNamespace

import torch

from experiments.pamf_phase3_restoration import conditioned_covariance_shared_trial as shared
from experiments.pamf_phase3_restoration import conditioned_covariance_trial as original
from experiments.pamf_phase3_restoration.variance_trial import recovery_gpu_choice


def runner(processes,free=17938):
    def run(command,**kwargs):
        gpu=int(command[command.index('-i')+1])
        if '--query-gpu=index,uuid,memory.free' in command:
            return SimpleNamespace(returncode=0,stdout=f'{gpu}, GPU-card{gpu}, {free}\n',stderr='')
        return SimpleNamespace(returncode=0,stdout=''.join(f'GPU-card{gpu}, {i}\n' for i in range(processes)),stderr='')
    return run


def test_authorized_single_card_budget_admits_four_and_excludes_five():
    choices=recovery_gpu_choice(runner(4),allowed_gpus=(6,),maximum_processes=5)
    assert choices==[dict(GPU=6,CUDA_ID='GPU-card6',free_memory_MB=17938,existing_compute_PIDs=4)]
    assert recovery_gpu_choice(runner(5),allowed_gpus=(6,),maximum_processes=5)==[]
    assert recovery_gpu_choice(runner(4,11999),allowed_gpus=(6,),maximum_processes=5)==[]
    assert recovery_gpu_choice(runner(4))==[]


def test_resource_variant_preserves_recipe_and_every_scratch_tensor():
    before=original.configuration(original.ARM)
    after=shared.configuration(shared.ARM)
    assert after.pop('run_id')==shared.RUN
    before.pop('run_id')
    assert after==before
    a=original.factory(original.ARM).state_dict()
    b=shared.factory(shared.ARM).state_dict()
    assert set(a)==set(b) and all(torch.equal(a[key],b[key]) for key in a)
