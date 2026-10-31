import pytest
import torch
from experiments.pamf_dura_efficiency.tangent_balance import balanced_tangent_loss
from experiments.pamf_dura_efficiency.clean_tangent import tangent_loss


def test_loss_and_current_IQ_gradient_bound():
    torch.manual_seed(42);clean=torch.randn(12,2,256);r=(clean+torch.randn_like(clean)*3).requires_grad_(True)
    jac=torch.randn(12,5,2,256)*torch.logspace(-3,3,12)[:,None,None,None];alpha=.0033130025035573727
    loss,info=balanced_tangent_loss(r,clean,jac,alpha);energy=512*clean.square().mean((1,2)).clamp_min(1e-4)
    iq=(r-clean).square().sum((1,2))/energy;guide=alpha*loss
    gradient=torch.autograd.grad(guide.sum(),r)[0];iqgrad=2*(r-clean)/energy[:,None,None]
    assert torch.all(guide<=iq*(1+2e-5))
    assert torch.all(gradient.flatten(1).norm(dim=1)<=iqgrad.flatten(1).norm(dim=1)*(1+2e-5))
    assert info['limited'].any() and (~info['limited']).any()


def test_frozen_step_weights_preserve_raw_direction_and_detach_references():
    clean=torch.randn(3,2,256,requires_grad=True);r=torch.randn_like(clean,requires_grad=True);jac=torch.randn(3,5,2,256,requires_grad=True)
    alpha=.2;loss,info=balanced_tangent_loss(r,clean,jac,alpha)
    actual=torch.autograd.grad((alpha*loss).sum(),r,retain_graph=True)[0]
    raw=torch.autograd.grad((alpha*tangent_loss(r,clean,jac)).sum(),r)[0]
    torch.testing.assert_close(actual,info['weight'][:,None,None]*raw,atol=2e-6,rtol=2e-6)
    assert not info['weight'].requires_grad and clean.grad is None and jac.grad is None
    cosine=torch.nn.functional.cosine_similarity(actual.flatten(1),raw.flatten(1),dim=1)
    torch.testing.assert_close(cosine,torch.ones(3),atol=2e-6,rtol=2e-6)


def test_uncapped_samples_match_original_metric():
    clean=torch.randn(4,2,256);r=torch.randn_like(clean);jac=torch.randn(4,5,2,256)*1e-4
    loss,info=balanced_tangent_loss(r,clean,jac,.0033130025035573727)
    assert torch.equal(info['weight'],torch.ones(4))
    torch.testing.assert_close(loss,tangent_loss(r,clean,jac))


def test_zero_and_invalid_inputs():
    clean=torch.randn(2,2,256);jac=torch.zeros(2,5,2,256)
    loss,info=balanced_tangent_loss(clean,clean,jac,.1)
    assert torch.equal(loss,torch.zeros(2)) and torch.isfinite(info['weight']).all()
    with pytest.raises(ValueError):balanced_tangent_loss(clean,clean,jac[:,:4],.1)
    with pytest.raises(ValueError):balanced_tangent_loss(clean,clean,jac,0.)
