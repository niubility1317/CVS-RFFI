import pytest
import torch
from experiments.pamf_phase3_restoration.covariance_inverse import dense_action,gaussian_posterior
from experiments.pamf_phase3_restoration.physical_operators import Action


def test_dense_causal_reflect_and_real_adjoint():
    torch.manual_seed(4);n=12;k=5;b=3
    h=torch.randn(b,k,dtype=torch.complex128)
    main=torch.randn(b,n,dtype=torch.complex128);image=torch.randn(b,n,dtype=torch.complex128)
    a=Action(h,main,image);dense=dense_action(h,main,image)
    x=torch.randn(b,n,dtype=torch.complex128);y=torch.randn(b,n,dtype=torch.complex128)
    flat=lambda z:torch.cat((z.real,z.imag),1)
    torch.testing.assert_close((dense@flat(x)[:,:,None]).squeeze(-1),flat(a(x)))
    torch.testing.assert_close((dense.transpose(1,2)@flat(y)[:,:,None]).squeeze(-1),flat(a.adjoint(y)))


def test_full_covariance_posterior_matches_normal_equations():
    torch.manual_seed(9);b=3;d=8
    a=torch.randn(b,d,d,dtype=torch.float64);y=torch.randn(b,d,dtype=torch.float64)
    mu=torch.randn(d,dtype=torch.float64);factor=torch.randn(d,d,dtype=torch.float64)
    cov=factor@factor.T+.1*torch.eye(d,dtype=torch.float64);noise=torch.tensor([.1,1.,2.],dtype=torch.float64)
    z,residual=gaussian_posterior(a,y,mu,cov,noise)
    precision=torch.linalg.inv(cov);normal=a.transpose(1,2)@a+noise[:,None,None]*precision
    rhs=(a.transpose(1,2)@y[:,:,None]).squeeze(-1)+noise[:,None]*(precision@mu)
    torch.testing.assert_close(z,torch.linalg.solve(normal,rhs))
    assert residual.max()<1e-12


def test_identity_diagonal_and_invalid_noise():
    a=torch.eye(4,dtype=torch.float64)[None];mu=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    cov=torch.diag(torch.tensor([.1,.3,1.,2.],dtype=torch.float64));y=torch.zeros(1,4,dtype=torch.float64)
    z,_=gaussian_posterior(a,y,mu,cov,torch.ones(1,dtype=torch.float64))
    torch.testing.assert_close(z,mu[None]/(cov.diag()+1))
    with pytest.raises(ValueError,match='positive real noise'):
        gaussian_posterior(a,y,mu,cov,torch.zeros(1,dtype=torch.float64))
