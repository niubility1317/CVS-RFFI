import torch

from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance
from experiments.pamf_phase3_restoration.signed_covariance import (
    adapt_covariance,signed_coefficients,source_modes,scratch_signed_covariance,
)
from experiments.pamf_phase3_restoration.projected_prior_detail import fixed_posterior_frame


def test_signed_modes_preserve_spd_and_exact_zero_prior():
    generator=torch.Generator().manual_seed(83)
    matrix=torch.randn(6,6,generator=generator,dtype=torch.float64)
    covariance=(matrix@matrix.T+.01*torch.eye(6,dtype=torch.float64)).float().double()
    modes=source_modes(covariance,rank=3).double()
    raw=torch.tensor([[-1e6,1e6,-50.],[50.,-1e6,1e6]],dtype=torch.float64)
    changed=adapt_covariance(covariance,modes,signed_coefficients(raw))
    assert torch.linalg.eigvalsh(changed).min()>0
    zero=adapt_covariance(covariance,modes,torch.zeros(2,3,dtype=torch.float64))
    assert torch.equal(zero,covariance.expand(2,-1,-1))
    one_mode=adapt_covariance(covariance,modes,torch.tensor([[-.5,0.,0.]],dtype=torch.float64))[0]
    assert not torch.allclose(one_mode,covariance)
    difference=one_mode-covariance
    assert difference.diagonal().sum()<0 and (difference-torch.diag(difference.diagonal())).abs().sum()>0


def test_signed_variance_can_learn_opposite_directions_at_identity():
    covariance=torch.tensor([[.01]],dtype=torch.float64)
    modes=source_modes(covariance,rank=1).double()
    raw=torch.zeros(2,1,dtype=torch.float64,requires_grad=True)
    adapted=adapt_covariance(covariance,modes,signed_coefficients(raw))
    observed=torch.tensor([[1.],[4.]],dtype=torch.float64)
    clean=torch.tensor([[1.],[.3]],dtype=torch.float64)
    restored,*_=fixed_posterior_frame(torch.ones(2,1,1,dtype=torch.float64),observed,
        torch.zeros(1,dtype=torch.float64),adapted,.1*torch.ones(2,dtype=torch.float64))
    (restored-clean).square().sum().backward()
    assert raw.grad[0,0]<0 and raw.grad[1,0]>0
    tail=torch.tensor([-50.,0.,50.],dtype=torch.float64,requires_grad=True)
    signed_coefficients(tail).sum().backward()
    assert tail.grad.min()>3e-4


def test_original_physical_initialization_is_preserved():
    old=scratch_covariance(2026092701)
    new=scratch_signed_covariance(2026092701)
    assert all(torch.equal(value,new.physical.state_dict()[key]) for key,value in old.state_dict().items())
    assert new.detail.output.weight.count_nonzero()==0 and new.detail.output.bias.count_nonzero()==0
    assert all(not buffer.requires_grad for name,buffer in new.physical.named_buffers())


def test_actual_noisy_readout_updates_signed_head_only_and_preserves_deployed_prior():
    from experiments.pamf_phase3_restoration.signed_covariance_trial import signed_readout,configuration,ARM
    from experiments.pamf_phase3_restoration.covariance_detail import detail_objective
    from experiments.pamf_phase3_restoration.objective import PairedObjective
    config=configuration(ARM)
    assert (config['epochs'],config['updates'],config['training_stages'])==(30,7410,1)
    model=scratch_signed_covariance(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    received=torch.randn(1,2,256,generator=torch.Generator().manual_seed(89))
    clean=.7*received
    class Reader:
        def features(self,value):return value.flatten(1)[:,:160]
    reader=Reader();objective=PairedObjective();target=reader.features(clean);scale=torch.ones(160)
    other=.08*torch.eye(512)
    priors=(torch.ones(5,512)*.1,other[None].expand(5,-1,-1),
            source_modes(other)[None].expand(5,-1,-1))
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    with torch.no_grad():deployed=model(received)
    restored,aux,loss,metrics=signed_readout(model,reader,received,clean,target,scale,objective,
        source_indices=torch.tensor([0]),epoch=2,priors=priors,assignment=torch.tensor([0]))
    expected,_=detail_objective(reader,restored,clean,target,scale,objective)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    assert metrics['receiver_mismatch_prior_fraction'].item()==1
    assert not torch.equal(deployed,restored)
    loss.mean().backward()
    assert all(parameter.grad is None for parameter in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0
    assert torch.isfinite(model.detail.output.weight.grad).all()
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    with torch.no_grad():torch.testing.assert_close(model(received),deployed,rtol=0,atol=0)
