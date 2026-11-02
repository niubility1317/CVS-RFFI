import unittest
import importlib.util
from pathlib import Path
import torch
from posterior_sampling import conditional_gaussian_draws

class PosteriorSamplingTest(unittest.TestCase):
 def test_zero_noise_draw_matches_analytic_conditional_mean(self):
  y=torch.tensor([[1.]],dtype=torch.float64);A=torch.ones(1,1,1,dtype=torch.float64);var=torch.ones(1,dtype=torch.float64);mean=torch.zeros(1,1,dtype=torch.float64);factor=torch.ones(1,1,1,dtype=torch.float64);prior=torch.ones(1,dtype=torch.float64);z=torch.zeros(1,1,1,dtype=torch.float64);draw,estimate,logw=conditional_gaussian_draws(y,A,var,mean,factor,prior,z,z);self.assertTrue(torch.allclose(draw,torch.tensor([[[.5]]],dtype=torch.float64)));self.assertTrue(torch.allclose(draw,estimate));self.assertEqual(logw.item(),0.)
 def test_monte_carlo_has_correct_conditional_variance(self):
  torch.manual_seed(21);torch.set_num_threads(1);n=10000;y=torch.ones(n,1);A=torch.ones(n,1,1);var=torch.ones(n);mean=torch.zeros(1,1);factor=torch.ones(1,1,1);prior=torch.ones(1);z=torch.randn(n,1,1);e=torch.randn(n,1,1);draw,_,_=conditional_gaussian_draws(y,A,var,mean,factor,prior,z,e);self.assertLess(abs(draw.mean().item()-.5),.03);self.assertLess(abs(draw.var(unbiased=False).item()-.5),.03)
 def test_correlated_observation_matches_analytic_posterior(self):
  torch.set_num_threads(1);torch.manual_seed(811);n=18000;dtype=torch.float64;obs=torch.tensor([.8,-.4],dtype=dtype);A=torch.tensor([[1.,.3],[-.2,.9]],dtype=dtype);mu=torch.tensor([.2,-.1],dtype=dtype);L=torch.tensor([[1.,0.],[.4,.7]],dtype=dtype);C=L@L.T;v=.3;S=A@C@A.T+v*torch.eye(2,dtype=dtype);gain=torch.linalg.solve(S,A@C).T;expected_mu=mu+gain@(obs-A@mu);expected_cov=C-gain@A@C;z=torch.randn(n,1,2,dtype=dtype);e=torch.randn(n,1,2,dtype=dtype);draw,means,_=conditional_gaussian_draws(obs.expand(n,-1),A.expand(n,-1,-1),torch.full((n,),v,dtype=dtype),mu[None],L[None],torch.ones(1,dtype=dtype),z,e);sample=draw[:,0];center=sample-sample.mean(0);empirical_cov=center.T@center/n;self.assertTrue(torch.allclose(means[0,0],expected_mu,atol=1e-10,rtol=1e-10));self.assertTrue(torch.allclose(sample.mean(0),expected_mu,atol=.02,rtol=0));self.assertTrue(torch.allclose(empirical_cov,expected_cov,atol=.02,rtol=0))
 def test_mixture_likelihood_weights_match_analytic_result(self):
  y=torch.tensor([[1.]],dtype=torch.float64);A=torch.ones(1,1,1,dtype=torch.float64);var=torch.ones(1,dtype=torch.float64);means=torch.tensor([[-1.],[1.]],dtype=torch.float64);f=torch.ones(2,1,1,dtype=torch.float64);p=torch.tensor([.5,.5],dtype=torch.float64);eps=torch.zeros(1,2,1,dtype=torch.float64);draw,estimate,logw=conditional_gaussian_draws(y,A,var,means,f,p,eps,eps);expected=torch.softmax(torch.tensor([-1.,0.],dtype=torch.float64),dim=0);self.assertTrue(torch.allclose(logw.exp()[0],expected));self.assertTrue(torch.allclose(draw,estimate))
 def test_original_covariance_is_used_for_ill_conditioned_posterior(self):
  torch.set_num_threads(1);torch.manual_seed(0);d=16;q=torch.linalg.qr(torch.randn(d,d)).Q;cov=(q@torch.diag(torch.logspace(0,-5,d))@q.T)[None];factors=torch.linalg.cholesky(cov);y=torch.randn(1,d);action=torch.eye(d)[None];var=torch.tensor([1e-5]);means=torch.zeros(1,d);zero=torch.zeros(1,1,d);args=(y,action,var,means,factors,torch.ones(1),zero,zero)
  _,estimate,_=conditional_gaussian_draws(*args,covariance=cov);_,reconstructed,_=conditional_gaussian_draws(*args)
  spec=importlib.util.spec_from_file_location('reference_posterior',Path(__file__).resolve().parents[1]/'phase1_conditional_bayes'/'gaussian_posterior.py');reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
  reference,_=reference_module.posterior(y,action,var[:,None,None]*torch.eye(d)[None],means,cov,torch.ones(1))
  torch.testing.assert_close(estimate,reference,rtol=1e-6,atol=1e-6)
  self.assertGreater(float((reconstructed-estimate).abs().max()),1e-4)
 def test_rejects_nonpositive_observation_variance(self):
  y=torch.zeros(1,1);A=torch.ones(1,1,1);f=torch.ones(1,1,1)
  with self.assertRaises(ValueError):conditional_gaussian_draws(y,A,torch.zeros(1),torch.zeros(1,1),f,torch.ones(1),f,f)
if __name__=='__main__':unittest.main()
