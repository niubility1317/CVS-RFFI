import pytest
import torch
from experiments.pamf_dura_efficiency.phase_prior import PhasePrior,coordinate,rotate_iq,rotate_inputs,wrap_priors


def fixture():
    torch.manual_seed(31)
    core=torch.nn.Sequential(torch.nn.Conv1d(9,8,3,padding=1),torch.nn.Tanh(),torch.nn.Conv1d(8,2,1))
    return PhasePrior(core),torch.randn(3,9,256)


def test_prior_is_phase_equivariant_and_keeps_amplitude_relative_phase_and_scalar():
    prior,x=fixture();angle=torch.tensor([.37,-1.13,2.7]);unit=torch.stack((angle.cos(),angle.sin()),dim=1)[:,:,None]
    rotated=rotate_inputs(x,unit);canonical,u,valid=coordinate(x)
    canonical2,_,_=coordinate(rotated)
    torch.testing.assert_close(canonical2,canonical,atol=3e-6,rtol=2e-5)
    torch.testing.assert_close(rotate_inputs(canonical,u),x,atol=2e-6,rtol=2e-5)
    assert torch.equal(canonical[:,8],x[:,8]) and valid.all()
    torch.testing.assert_close(prior(rotated),rotate_iq(prior(x),unit),atol=3e-6,rtol=2e-5)


def test_zero_base_fallback_zero_signal_and_backprop_are_finite():
    prior,x=fixture();x[:,2:4]=0
    angle=torch.full((3,),.73);unit=torch.stack((angle.cos(),angle.sin()),dim=1)[:,:,None]
    torch.testing.assert_close(prior(rotate_inputs(x,unit)),rotate_iq(prior(x),unit),atol=3e-6,rtol=2e-5)
    x.requires_grad_();prior(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in prior.parameters())
    zero=torch.zeros(3,9,256,requires_grad=True);zero.data[:,8]=2
    out=prior(zero);assert not out.any();out.sum().backward();assert torch.isfinite(zero.grad).all()
    with pytest.raises(ValueError):coordinate(torch.randn(2,8,256))


def test_wrapping_preserves_parameter_identity_aliases_and_rng():
    model=torch.nn.Module();a=torch.nn.Conv1d(9,2,3,padding=1);b=torch.nn.Conv1d(9,2,3,padding=1);b.weight=a.weight
    model.prior=torch.nn.ModuleList([a,b]);model.initialization_report={}
    original=list(model.parameters());state=torch.get_rng_state().clone();wrap_priors(model)
    assert torch.equal(state,torch.get_rng_state()) and all(a is b for a,b in zip(original,model.parameters()))
    assert model.prior[0].core.weight is model.prior[1].core.weight
    assert len(list(model.parameters()))==len(original)
