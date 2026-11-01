import pytest
import torch
from experiments.pamf_phase3_restoration.covariance_feature_trial import feature_window,micro_objective,posterior_feature_guide
from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance


def test_feature_sampling_budget_and_partial_logical_batch():
    assert feature_window(256)==slice(0,32)
    assert feature_window(24)==slice(0,24)
    assert sum(feature_window(min(256,63000-i)).stop for i in range(0,63000,256))==7896
    with pytest.raises(ValueError):feature_window(0)
    with pytest.raises(ValueError):feature_window(257)


def test_physics_and_feature_microbatch_normalization():
    physical=torch.arange(1.,6.,requires_grad=True);guide=torch.tensor([2.,3.],requires_grad=True)
    total=micro_objective(physical[:2],guide,5)+micro_objective(physical[2:4],None,5)+micro_objective(physical[4:],None,5)
    torch.testing.assert_close(total,physical.mean()+guide.mean())
    total.backward()
    torch.testing.assert_close(physical.grad,torch.full((5,),.2))
    torch.testing.assert_close(guide.grad,torch.full((2,),.5))


def test_frozen_feature_loss_reaches_physical_heads_through_posterior():
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.projection=torch.nn.Linear(512,160)
        def features(self,x):return self.projection(x.flatten(1))
    torch.manual_seed(27);g=scratch_covariance(4);reader=Reader().eval().requires_grad_(False)
    received=torch.randn(2,2,256);clean=torch.randn_like(received)
    target=reader.features(clean).detach().requires_grad_(True)
    guide,metrics=posterior_feature_guide(reader,g.estimate(received),received,clean,target,torch.ones(160),g)
    guide.mean().backward()
    for head in (g.kernel_head,g.coefficient_head,g.noise_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in reader.parameters()) and target.grad is None
    assert metrics['feature_error'].shape==(2,) and metrics['iq_nmse'].shape==(2,)
    assert metrics['relative_solve_residual'].max()<1e-8
