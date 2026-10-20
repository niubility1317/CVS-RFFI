import copy
import pytest
import torch
from torch import nn
from tools.inspect_pamf_r17_prior_weights import weight_structure


def test_identical_stage_kernels_have_zero_algebraic_error_and_smaller_shared_storage():
    torch.manual_seed(3)
    prior=nn.Sequential(nn.Conv1d(3,4,3),nn.GroupNorm(2,4),nn.Conv1d(4,2,1)).double()
    value=weight_structure([copy.deepcopy(prior) for _ in range(3)],100)
    first=value['hypothetical_shared_base_plus_independent_lowrank_deltas'][0]
    assert first['full_parameters']<value['original_full_parameters']
    assert first['relative_kernel_Frobenius_error']<1e-14
    assert value['forward_calls']==value['physical_IQ_reads']==value['optimizer_updates']==0


def test_rank_one_stage_specialization_is_representable_without_zeroing_dense_parameters():
    torch.manual_seed(4)
    original=nn.Sequential(nn.Conv1d(3,4,3)).double();priors=[copy.deepcopy(original) for _ in range(3)]
    delta=torch.randn(4,1,dtype=torch.float64)@torch.randn(1,9,dtype=torch.float64)
    with torch.no_grad():
        for index,prior in enumerate(priors):prior[0].weight.add_(index*.01*delta.reshape_as(prior[0].weight))
    value=weight_structure(priors,100);rows=value['hypothetical_shared_base_plus_independent_lowrank_deltas']
    assert rows[0]['relative_kernel_Frobenius_error']>0
    assert rows[1]['relative_kernel_Frobenius_error']<1e-14
    assert rows[1]['full_parameters']<value['original_full_parameters']


def test_single_stage_or_unmatched_structures_are_rejected():
    prior=nn.Sequential(nn.Conv1d(3,4,3))
    with pytest.raises(ValueError):weight_structure([prior],100)
    with pytest.raises(ValueError):weight_structure([prior,prior,nn.Sequential(nn.Conv1d(3,5,3))],100)
