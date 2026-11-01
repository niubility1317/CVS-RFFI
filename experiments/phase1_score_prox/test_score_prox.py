import torch
import pytest
from score_prox_model import GroundDenoiser,data_prox

def test_prox_identity_and_scaled_operator():
    torch.manual_seed(5);n=7;x=torch.randn(3,n);p=torch.randn(3,n);scale=torch.tensor([.25,1.,3.]);a=torch.eye(n)[None]*scale[:,None,None];lam=.3
    got=data_prox(a,x,p,lam);expected=(scale[:,None]*x+lam*scale[:,None].square()*p)/(scale[:,None].square()*(1+lam))
    torch.testing.assert_close(got,expected,rtol=2e-5,atol=2e-5)

def test_prox_solves_normal_equations():
    torch.manual_seed(6);a=torch.randn(2,8,8);x=torch.randn(2,8);p=torch.randn(2,8);lam=.4;z=data_prox(a,x,p,lam);e=a.square().sum((1,2))/8
    residual=torch.bmm(a.transpose(1,2),torch.bmm(a,z.unsqueeze(-1))-x.unsqueeze(-1)).squeeze(-1)+lam*e[:,None]*(z-p)
    torch.testing.assert_close(residual,torch.zeros_like(residual),atol=2e-5,rtol=0)
    with pytest.raises(ValueError):data_prox(a,x,p,0)

def test_denoiser_identity_and_finite_gradient():
    torch.manual_seed(7);m=GroundDenoiser();x=torch.randn(3,2,256);x[0]=0;sig=torch.tensor([0.,.1,.4]);y=m(x,sig);torch.testing.assert_close(y,x)
    (y-x*.5).square().mean().backward();assert torch.isfinite(m.output.weight.grad).all() and m.output.weight.grad.abs().sum()>0
