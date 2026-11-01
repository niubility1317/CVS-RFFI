import pytest
import torch
from noise_prox import noise_adaptive_prox

def test_identity_system_adapts_each_record_to_its_noise():
    torch.manual_seed(5);x=torch.randn(3,8);p=torch.randn(3,8);A=torch.eye(8).repeat(3,1,1);v=torch.tensor([.001,.1,1.]);sigma=torch.tensor([.1,.2,.4]);alpha=2.;weight=alpha*v/(.5*sigma.square())
    got=noise_adaptive_prox(A,x,p,v,sigma,alpha);want=(x+weight[:,None]*p)/(1+weight[:,None]);torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-5)

def test_invalid_uncertainty_rejected():
    A=torch.eye(4)[None];x=torch.zeros(1,4)
    with pytest.raises(ValueError):noise_adaptive_prox(A,x,x,torch.tensor([0.]),torch.tensor([1.]),1.)
