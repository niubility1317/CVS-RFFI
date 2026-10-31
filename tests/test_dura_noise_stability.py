import pytest
import torch
from experiments.pamf_dura_efficiency.noise_stability import paired_noise, qualifies


def test_antithetic_pair_preserves_reference_and_clean_records():
    y=torch.arange(3*2*256,dtype=torch.float32).reshape(3,2,256)/1000
    before=y.clone();z=torch.linspace(-1,1,y.numel()).reshape_as(y)
    variance=torch.tensor([.25,1.,0.]);clean=torch.tensor([False,True,False])
    plus,minus,noise=paired_noise(y,variance,clean,z)
    torch.testing.assert_close(noise[0],.5*z[0]);assert torch.count_nonzero(noise[1:])==0
    torch.testing.assert_close((plus+minus)/2,y);assert torch.equal(y,before)
    assert torch.equal(plus[1],y[1]) and torch.equal(minus[1],y[1])


def test_identity_response_has_exact_unit_noise_gain():
    g=torch.Generator().manual_seed(31);y=torch.randn(4,2,256,generator=g)
    z=torch.randn(4,2,256,generator=g);plus,minus,noise=paired_noise(y,torch.full((4,),.01),torch.zeros(4,dtype=torch.bool),z)
    incoming=noise.square().mean((1,2));outgoing=((plus-y).square().mean((1,2))+(minus-y).square().mean((1,2)))/2
    torch.testing.assert_close(outgoing/incoming,torch.ones(4),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('variance',[torch.tensor([-1.]),torch.tensor([float('nan')])])
def test_invalid_source_variance_is_rejected(variance):
    with pytest.raises(ValueError):paired_noise(torch.zeros(1,2,256),variance,torch.zeros(1,dtype=torch.bool),torch.ones(1,2,256))


def test_both_fixed_source_views_must_qualify_without_threshold_relaxation():
    rows=[dict(view=v,noise_gain_ratio=1.25,primary_MSE_ratio=1.10) for v in ['1','20']]
    assert qualifies(rows)
    rows[1]['noise_gain_ratio']=1.249
    assert not qualifies(rows)
    with pytest.raises(ValueError):qualifies(rows[:1])
