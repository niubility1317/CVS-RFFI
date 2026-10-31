import math
import pytest
import torch
from experiments.pamf_dura_efficiency.clean_tangent import clean_jacobian,tangent_loss,eligibility


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(160,512,dtype=torch.float64)/32)
    def features(self,x):return torch.sin(x.flatten(1)@self.weight.T)


def test_clean_jacobian_is_per_packet_and_matches_finite_difference():
    torch.manual_seed(9);reader=Reader().eval().requires_grad_(False)
    x=torch.randn(2,2,256,dtype=torch.float64);basis=torch.linalg.qr(torch.randn(160,5,dtype=torch.float64)).Q
    value,jac=clean_jacobian(reader,x,basis)
    assert jac.shape==(2,5,2,256) and not jac.requires_grad and not value.requires_grad
    solo=clean_jacobian(reader,x[:1],basis)
    torch.testing.assert_close(value[:1],solo[0]);torch.testing.assert_close(jac[:1],solo[1])
    direction=torch.randn_like(x);step=1e-6
    def phi(z):return torch.nn.functional.normalize(reader.features(z),dim=1,eps=1e-4)@basis*4
    numerical=(phi(x+step*direction)-phi(x-step*direction))/(2*step)
    torch.testing.assert_close((jac*direction[:,None]).sum((2,3)),numerical,atol=2e-8,rtol=2e-6)
    assert all(p.grad is None for p in reader.parameters()) and x.grad is None


def test_tangent_loss_and_gradient_are_fixed_linear_metric():
    x=torch.randn(3,2,256,dtype=torch.float64,requires_grad=True);clean=torch.randn_like(x,requires_grad=True)
    jac=torch.randn(3,5,2,256,dtype=torch.float64,requires_grad=True)
    loss=tangent_loss(x,clean,jac);error=x.detach()-clean.detach();v=(jac.detach()*error[:,None]).sum((2,3))
    expected=2*(v[:,:,None,None]*jac.detach()).sum(1)
    torch.testing.assert_close(loss,v.square().sum(1));loss.sum().backward()
    torch.testing.assert_close(x.grad,expected);assert clean.grad is None and jac.grad is None
    torch.testing.assert_close(tangent_loss(clean,clean,jac),torch.zeros(3,dtype=torch.float64))


def test_frozen_boundaries_and_shapes():
    reader=Reader();x=torch.randn(2,2,256,dtype=torch.float64);basis=torch.randn(160,5,dtype=torch.float64)
    with pytest.raises(ValueError):clean_jacobian(reader,x,basis)
    reader.eval().requires_grad_(False)
    with pytest.raises(ValueError):clean_jacobian(reader,x,basis[:159])
    with pytest.raises(ValueError):tangent_loss(x,x,torch.randn(2,4,2,256))


def test_eligibility_is_both_views_all_receivers_and_no_auto_training():
    def row():return dict(mean_tangent_cosine=.7,mean_IQ_cosine=.4,nonnegative_tangent=400,count=512,defined_tangent=512,defined_IQ=512,
        rx_mean_tangent={str(rx):.4 for rx in [1,3,4,6,8]},median_tangent_to_IQ_norm=8.)
    rows={'1':row(),'20':row()};decision=eligibility(rows)
    assert decision['eligible'] and decision['fixed_source_weight']==pytest.approx(.25/8) and not decision['automatic_training']
    rows['20']['rx_mean_tangent']['8']=-.01;assert not eligibility(rows)['eligible']
    rows['20']=row();rows['1']['mean_tangent_cosine']=.49;assert not eligibility(rows)['eligible']
    rows['1']=row();rows['1']['defined_tangent']=511;assert not eligibility(rows)['eligible']
