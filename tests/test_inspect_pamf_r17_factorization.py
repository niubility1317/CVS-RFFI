import math
import pytest
import torch
from torch import nn
from tools.inspect_pamf_r17_factorization import factorization_structure


def test_rank_one_kernels_are_exactly_compressible_and_other_state_is_retained():
    priors=[]
    for _ in range(3):
        value=nn.Sequential(nn.Conv1d(4,8,3),nn.GroupNorm(4,8),nn.Conv1d(8,2,1))
        with torch.no_grad():value[0].weight.copy_(torch.ones_like(value[0].weight))
        priors.append(value)
    original=123+sum(p.numel() for v in priors for p in v.parameters())
    result=factorization_structure(priors,123)
    half=next(v for v in result['hypothetical_costs'] if v['strategy']=='half_all')
    energy=next(v for v in result['hypothetical_costs'] if v['strategy']=='energy999')
    assert result['original_full_parameters']==original and result['long_layers']==3
    assert half['saved_parameters']==3*(8*4*3-4*(8+4*3))
    assert half['relative_long_kernel_Frobenius_error']<1e-12
    assert energy['saved_parameters']==3*(8*4*3-(8+4*3))
    assert all(v['energy_ranks']['0.999']==1 for v in result['layers'])
    assert all(v['full_parameters']>=123 for v in result['hypothetical_costs'])


def test_full_energy_rank_does_not_add_a_larger_factorization():
    priors=[]
    for _ in range(3):
        conv=nn.Conv1d(1,2,3)
        with torch.no_grad():
            conv.weight.zero_();conv.weight[0,0,0]=1;conv.weight[1,0,1]=1
        priors.append(nn.Sequential(conv))
    result=factorization_structure(priors,0)
    energy=next(v for v in result['hypothetical_costs'] if v['strategy']=='energy99')
    assert energy['saved_parameters']==0 and energy['changed_long_layers']==0
    assert energy['relative_long_kernel_Frobenius_error']==0
    half=next(v for v in result['hypothetical_costs'] if v['strategy']=='half_all')
    assert half['saved_parameters']==3 and math.isclose(half['relative_long_kernel_Frobenius_error'],2**-.5)


def test_excluded_widths_and_invalid_inputs_do_not_silently_approximate():
    priors=[nn.Sequential(nn.Conv1d(4,8,3)) for _ in range(3)]
    result=factorization_structure(priors,7)
    for row in result['hypothetical_costs']:
        if row['strategy'] in ('half_output_ge64','half_output_ge96'):assert row['saved_parameters']==0
    with pytest.raises(ValueError):factorization_structure(priors[:2],7)
    with pytest.raises(ValueError):factorization_structure([nn.Conv1d(4,4,3,groups=4) for _ in range(3)],7)
    with pytest.raises(ValueError):factorization_structure([nn.Conv1d(4,4,1) for _ in range(3)],7)
