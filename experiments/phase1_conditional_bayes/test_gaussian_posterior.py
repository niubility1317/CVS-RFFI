import unittest,torch,numpy as np
try:from gaussian_posterior import posterior
except ImportError:posterior=None
class PosteriorTests(unittest.TestCase):
 def test_identity_scalar_case_and_probabilities(self):
  self.assertIsNotNone(posterior);y=torch.tensor([[1.,-1.]]);a=torch.eye(2).unsqueeze(0);n=torch.eye(2).unsqueeze(0)*.5;mu=torch.tensor([[0.,0.],[2.,-2.]]);cov=torch.stack([torch.eye(2),torch.eye(2)]);prior=torch.tensor([.5,.5]);means,logp=posterior(y,a,n,mu,cov,prior)
  self.assertTrue(torch.allclose(means[0,0],y[0]/1.5,atol=1e-6));self.assertTrue(torch.allclose(means[0,1],mu[1]+(y[0]-mu[1])/1.5,atol=1e-6));self.assertTrue(torch.allclose(logp.exp().sum(1),torch.ones(1),atol=1e-6));self.assertAlmostEqual(logp.exp()[0,0].item(),.5,places=6)
 def test_confidence_mask_uses_supported_float_dlpack(self):
  confidence=torch.tensor([.94,.95,.99]);mask=np.from_dlpack(confidence.contiguous())>=.95;self.assertEqual(np.count_nonzero(mask),2)
 def test_finite_batch_and_nontrivial_likelihood(self):
  torch.manual_seed(4);y=torch.randn(3,4);a=torch.eye(4).repeat(3,1,1);n=.1*torch.eye(4).repeat(3,1,1);mu=torch.tensor([[0.]*4,[8.]*4]);cov=torch.eye(4).repeat(2,1,1);m,p=posterior(y,a,n,mu,cov,torch.tensor([.5,.5]));self.assertEqual(m.shape,(3,2,4));self.assertTrue(torch.isfinite(m).all());self.assertTrue(torch.isfinite(p).all());self.assertTrue(torch.all(p[:,0]>p[:,1]))
if __name__=='__main__':unittest.main()
