import torch
from experiments.phase1_dura_orthogonal.benchmark_cross_expert_resident import tree_equal
from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def test_nested_training_state_requires_exact_values_and_structure():
    a={'model':{'w':torch.tensor([1.,2.])},'optimizer':[{'m':torch.tensor([3.]),'step':4}]}
    b={'model':{'w':torch.tensor([1.,2.])},'optimizer':[{'m':torch.tensor([3.]),'step':4}]}
    assert tree_equal(a,b)
    b['optimizer'][0]['m'][0]=4.
    assert not tree_equal(a,b)
    assert not tree_equal(a,{'model':a['model']})


def test_gpu_capacity_respects_existing_owners_and_memory():
    g='0, GPU-a, 9000\n'
    assert capacity(g,'GPU-a, 1\nGPU-a, 2\n',0,minimum_mib=8192)[0]
    assert not capacity(g,'GPU-a, 1\nGPU-a, 2\nGPU-a, 3\n',0,minimum_mib=8192)[0]
    assert not capacity('0, GPU-a, 8000\n','',0,minimum_mib=8192)[0]
