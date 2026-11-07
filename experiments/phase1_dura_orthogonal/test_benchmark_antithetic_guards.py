import torch

from experiments.phase1_dura_orthogonal.benchmark_antithetic_guards import equal_tree


def test_nested_optimizer_and_rng_comparison_detects_tensor_changes():
    state = {'params': [torch.tensor([1., 2.])], 'step': 3, 'rng': (torch.tensor([2]),)}
    same = {'params': [torch.tensor([1., 2.])], 'step': 3, 'rng': (torch.tensor([2]),)}
    assert equal_tree(state, same)
    same['params'][0][0] += 1
    assert not equal_tree(state, same)
