import math
import pytest
import torch
from lowrank_posterior import factors_from_covariance, lowrank_posterior

def direct(y,action,variance,means,factors,priors):
 cov=factors@factors.transpose(-1,-2);cross=cov.unsqueeze(0)@action[:,None].transpose(-1,-2);S=action[:,None]@cross+variance[:,None,None,None]*torch.eye(y.shape[-1],dtype=y.dtype);res=y[:,None]-torch.einsum('bij,kj->bki',action,means);sol=torch.linalg.solve(S,res.unsqueeze(-1)).squeeze(-1);states=means[None]+(cross@sol.unsqueeze(-1)).squeeze(-1);sign,logdet=torch.linalg.slogdet(S);assert torch.all(sign>0);scores=torch.log(priors/priors.sum())[None]-.5*((res*sol).sum(-1)+logdet+y.shape[-1]*math.log(2*math.pi));return states,scores-torch.logsumexp(scores,dim=-1,keepdim=True)

@pytest.mark.parametrize('rank',[2,7])
def test_lowrank_matches_direct_covariance(rank):
 torch.manual_seed(412);dtype=torch.float64;b,k,d=3,4,7;y=torch.randn(b,d,dtype=dtype);action=torch.randn(b,d,d,dtype=dtype);means=torch.randn(k,d,dtype=dtype);matrix=torch.randn(k,d,d,dtype=dtype);cov=matrix@matrix.transpose(-1,-2)+.1*torch.eye(d,dtype=dtype);var=torch.tensor([.1,.7,2.],dtype=dtype);priors=torch.tensor([.1,.2,.3,.4],dtype=dtype);f=factors_from_covariance(cov,rank);actual=lowrank_posterior(y,action,var,means,f,priors);expect=direct(y,action,var,means,f,priors);torch.testing.assert_close(actual[0],expect[0],rtol=1e-9,atol=1e-9);torch.testing.assert_close(actual[1],expect[1],rtol=1e-9,atol=1e-9)

def test_full_rank_reconstructs_covariance_and_rejects_invalid_noise():
 torch.manual_seed(7);m=torch.randn(3,6,6,dtype=torch.float64);cov=m@m.transpose(-1,-2)+.2*torch.eye(6,dtype=m.dtype);f=factors_from_covariance(cov,6);torch.testing.assert_close(f@f.transpose(-1,-2),cov,rtol=1e-10,atol=1e-10)
 with pytest.raises(ValueError):lowrank_posterior(torch.zeros(2,6,dtype=m.dtype),torch.eye(6,dtype=m.dtype).expand(2,-1,-1),torch.tensor([1.,0.],dtype=m.dtype),torch.zeros(3,6,dtype=m.dtype),f,torch.ones(3,dtype=m.dtype))
