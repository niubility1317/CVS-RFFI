import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration import source_fingerprint_response as m


def test_four_direction_energy_and_linear_injection_preserve_same_noise():
    torch.manual_seed(7);clean=torch.randn(3,2,256);noise=.03*torch.randn_like(clean)
    diagonal=torch.linspace(.8,1.2,512);action=torch.diag(diagonal).expand(3,-1,-1).double()
    received=(action@clean.flatten(1).double()[:,:,None]).squeeze(-1).float().reshape_as(clean)+noise
    delta=m.perturbations(clean)
    assert delta.shape==(3,4,2,256)
    torch.testing.assert_close(delta.square().mean((2,3)),.0001*clean.square().mean((1,2))[:,None].expand(-1,4),rtol=2e-6,atol=1e-10)
    for j in range(4):
        changed=m.inject(received,delta[:,j],action)
        noisy_old=received.flatten(1).double()-(action@clean.flatten(1).double()[:,:,None]).squeeze(-1)
        noisy_new=changed.flatten(1).double()-(action@(clean+delta[:,j]).flatten(1).double()[:,:,None]).squeeze(-1)
        torch.testing.assert_close(noisy_old,noisy_new,rtol=0,atol=5e-7)


def test_gain_error_and_zero_reference_are_reported_without_hidden_floor():
    target=torch.tensor([[1.,2.,3.],[2.,-1.,.5]])
    identity=m.summarize(m.response_statistics(target,target).tolist())
    half=m.summarize(m.response_statistics(target,.5*target).tolist())
    assert identity['projection_gain']==1 and identity['relative_squared_error']==0
    assert half['projection_gain']==.5 and half['relative_squared_error']==.25 and half['response_norm_ratio']==.5
    zero=m.summarize(m.response_statistics(torch.zeros_like(target),target).tolist())
    assert zero['zero_reference_rows']==2 and zero['projection_gain'] is None and zero['squared_error_sum']>0


def test_degenerate_or_nonfinite_inputs_reject_without_case_omission():
    with pytest.raises(ValueError):m.perturbations(torch.zeros(1,2,256))
    with pytest.raises(ValueError):m.response_statistics(torch.ones(2,4),torch.full((2,4),float('nan')))
    with pytest.raises(ValueError):m.summarize(np.array([[1,1,1,float('nan')]]))
    with pytest.raises(ValueError):m.inject(torch.zeros(2,2,256),torch.zeros(1,2,256),torch.eye(512)[None])
