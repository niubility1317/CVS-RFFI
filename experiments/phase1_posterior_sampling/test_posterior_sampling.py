import unittest
import torch
from posterior_sampling import conditional_gaussian_draws

class PosteriorSamplingTest(unittest.TestCase):
 def test_zero_noise_draw_matches_analytic_conditional_mean(self):
  y=torch.tensor([[1.]],dtype=torch.float64);A=torch.ones(1,1,1,dtype=torch.float64);var=torch.ones(1,dtype=torch.float64);mean=torch.zeros(1,1,dtype=torch.float64);factor=torch.ones(1,1,1,dtype=torch.float64);prior=torch.ones(1,dtype=torch.float64);z=torch.zeros(1,1,1,dtype=torch.float64);draw,estimate,logw=conditional_gaussian_draws(y,A,var,mean,factor,prior,z,z);self.assertTrue(torch.allclose(draw,torch.tensor([[[.5]]],dtype=torch.float64)));self.assertTrue(torch.allclose(draw,estimate));self.assertEqual(logw.item(),0.)
 def test_monte_carlo_has_correct_conditional_variance(self):
  torch.manual_seed(21);torch.set_num_threads(1);n=10000;y=torch.ones(n,1);A=torch.ones(n,1,1);var=torch.ones(n);mean=torch.zeros(1,1);factor=torch.ones(1,1,1);prior=torch.ones(1);z=torch.randn(n,1,1);e=torch.randn(n,1,1);draw,_,_=conditional_gaussian_draws(y,A,var,mean,factor,prior,z,e);self.assertLess(abs(draw.mean().item()-.5),.03);self.assertLess(abs(draw.var(unbiased=False).item()-.5),.03)
 def test_mixture_likelihood_weights_match_analytic_result(self):
  y=torch.tensor([[1.]],dtype=torch.float64);A=torch.ones(1,1,1,dtype=torch.float64);var=torch.ones(1,dtype=torch.float64);means=torch.tensor([[-1.],[1.]],dtype=torch.float64);f=torch.ones(2,1,1,dtype=torch.float64);p=torch.tensor([.5,.5],dtype=torch.float64);eps=torch.zeros(1,2,1,dtype=torch.float64);draw,estimate,logw=conditional_gaussian_draws(y,A,var,means,f,p,eps,eps);expected=torch.softmax(torch.tensor([-1.,0.],dtype=torch.float64),dim=0);self.assertTrue(torch.allclose(logw.exp()[0],expected));self.assertTrue(torch.allclose(draw,estimate))
 def test_rejects_nonpositive_observation_variance(self):
  y=torch.zeros(1,1);A=torch.ones(1,1,1);f=torch.ones(1,1,1)
  with self.assertRaises(ValueError):conditional_gaussian_draws(y,A,torch.zeros(1),torch.zeros(1,1),f,torch.ones(1),f,f)
if __name__=='__main__':unittest.main()
