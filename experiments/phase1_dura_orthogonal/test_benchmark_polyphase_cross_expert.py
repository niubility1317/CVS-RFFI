import torch
from experiments.phase1_dura_orthogonal.benchmark_polyphase_cross_expert import tree_bits
from experiments.phase1_dura_orthogonal.benchmark_polyphase_cross_expert import compare_model,compare_tensors


def test_comparison_detects_differences_in_model_and_output():
    a={'weight':torch.tensor([1.,0.]),'bias':torch.tensor([.1])}
    b={'weight':torch.tensor([1.01,0.]),'bias':torch.tensor([.1])}
    result=compare_model(a,b)
    assert not result['bitwise_equal'] and result['different_tensors']==1
    assert result['outside_tolerance_tensors']==1 and result['max_abs']>.009
    assert not compare_tensors(a['weight'],b['weight'])['bitwise_equal']


def test_recursive_bit_comparison_detects_signed_zero():
    assert not tree_bits({'state':[torch.tensor([0.])]}, {'state':[torch.tensor([-0.])]})
    assert not tree_bits({'eps':0.0},{'eps':-0.0})
    assert tree_bits({'state':[torch.tensor([1.])]}, {'state':[torch.tensor([1.])]})
