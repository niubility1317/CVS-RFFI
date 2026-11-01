import unittest,torch,sys
from pathlib import Path
local=Path(__file__).resolve().parents[1]/'phase1_blind_bayes_validation'
remote=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-blind-bayes-source-validation-s2026092701-r01-release/experiments/phase1_blind_bayes_validation')
sys.path.insert(0,str(local if local.exists() else remote))
from blind_action_matrix import action_matrix
try:from differentiable_posterior import posterior
except ImportError:posterior=None
class PosteriorGradientTests(unittest.TestCase):
 def test_gradients_are_finite_and_nonzero(self):
  self.assertIsNotNone(posterior);torch.manual_seed(3);A=torch.eye(3).unsqueeze(0).requires_grad_();y=torch.randn(1,3);noise=.1*torch.eye(3).unsqueeze(0);means=torch.stack([torch.zeros(3),torch.ones(3)]);cov=torch.eye(3).repeat(2,1,1);out,lp=posterior(y,A,noise,means,cov,torch.tensor([.5,.5]));z=(out*lp.exp().unsqueeze(-1)).sum();zero=torch.autograd.grad(z*0,A,retain_graph=True)[0];self.assertTrue(torch.equal(zero,torch.zeros_like(zero)));z.backward();self.assertTrue(torch.isfinite(A.grad).all());self.assertGreater(A.grad.abs().sum().item(),0.)
 def test_action_matrix_propagates_filter_gradients(self):
  h=torch.randn(1,2,64,requires_grad=True);x=torch.randn(1,512,1);y=torch.bmm(action_matrix(h),x).square().mean();y.backward();self.assertTrue(torch.isfinite(h.grad).all());self.assertGreater(h.grad.abs().sum().item(),0.)
if __name__=='__main__':unittest.main()
