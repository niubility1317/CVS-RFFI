import torch
from posterior_uncertainty import posterior_moments,uncertainty_gram,expected_channel_update
from channel_update import design_matrix,update_channel

def test_posterior_moments_match_scalar_gaussian_formula():
 y=torch.tensor([[1.3]]);a=torch.tensor([[[2.]]]);noise=torch.tensor([[[.5]]]);means=torch.tensor([[.2]]);cov=torch.tensor([[[.8]]]);mu,var=posterior_moments(y,a,noise,means,cov,torch.tensor([1.]));expected=.2+(.8*2)/(4*.8+.5)*(1.3-2*.2);expected_var=.8-(.8*2)**2/(4*.8+.5);assert torch.allclose(mu,torch.tensor([[expected]]),atol=1e-6);assert torch.allclose(var,torch.tensor([[[expected_var]]]),atol=1e-6)

def test_uncertain_design_gram_matches_rank_one_exactly():
 torch.manual_seed(11);v=torch.randn(1,2,256);flat=v.reshape(1,512,1);cov=flat@flat.transpose(1,2);design=design_matrix(v);actual=uncertainty_gram(cov);expected=design.transpose(1,2)@design;assert torch.allclose(actual,expected,atol=2e-4,rtol=2e-5)

def test_zero_covariance_recovers_existing_channel_update():
 torch.manual_seed(12);s=torch.randn(1,2,256);x=torch.randn_like(s);h=torch.randn(1,2,64);zero=torch.zeros(1,512,512);a=update_channel(s,x,h,1.);b=expected_channel_update(s,zero,x,h,1.);assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)


def test_mixture_covariance_includes_between_component_uncertainty():
 y=torch.tensor([[0.]]);a=torch.tensor([[[0.]]]);noise=torch.tensor([[[1.]]]);means=torch.tensor([[-1.],[1.]]);cov=torch.tensor([[[.25]],[[.25]]]);mu,var=posterior_moments(y,a,noise,means,cov,torch.tensor([.75,.25]));assert torch.allclose(mu,torch.tensor([[-.5]]),atol=1e-6);assert torch.allclose(var,torch.tensor([[[1.]]]),atol=1e-6)


def test_mixture_means_match_existing_posterior():
 from experiments.phase1_conditional_bayes.gaussian_posterior import posterior
 torch.manual_seed(23);y=torch.randn(2,3);a=torch.eye(3).repeat(2,1,1);noise=torch.eye(3).repeat(2,1,1)*.4;means=torch.randn(2,3);cov=torch.eye(3).repeat(2,1,1)*.5;priors=torch.tensor([.2,.8]);states,logp=posterior(y,a,noise,means,cov,priors);expected=(states*logp.exp().unsqueeze(-1)).sum(1);mean,var=posterior_moments(y,a,noise,means,cov,priors);assert torch.allclose(mean,expected,atol=1e-6,rtol=1e-6);assert torch.linalg.eigvalsh(var).min()>=-1e-6
