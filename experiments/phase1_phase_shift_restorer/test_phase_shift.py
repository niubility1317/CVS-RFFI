import unittest,torch
import phase_shift_model
from phase_factor_model import AdaptivePhaseFactor
from phase_shift_model import PhaseShiftRestorer
class ArchitectureTests(unittest.TestCase):
 def test_identity_anchor_frozen_phase_and_gradient(self):
  torch.manual_seed(52);m=PhaseShiftRestorer(AdaptivePhaseFactor());m.train();x=torch.randn(2,2,256);a=m.anchor(x);y=m(x)
  self.assertEqual(tuple(y.shape),(2,2,256));self.assertTrue(torch.allclose(a,y,atol=2e-5,rtol=2e-5));self.assertFalse(m.phase.training);self.assertTrue(all(not p.requires_grad for p in m.phase.parameters()));y.square().mean().backward();self.assertTrue(torch.isfinite(y).all());self.assertTrue(m.body.temporal_output.weight.grad.abs().sum()>0);self.assertTrue(m.body.output.weight.grad.abs().sum()>0);self.assertTrue(all(p.grad is None for p in m.phase.parameters()))
 def test_cpu_backend_prevents_nonzero_zero_gradient(self):
  prior=torch.backends.mkldnn.enabled
  try:
   torch.backends.mkldnn.enabled=True
   configure=getattr(phase_shift_model,'configure_training_backend',None)
   if configure is not None:configure('cpu')
   self.assertFalse(torch.backends.mkldnn.enabled,'CPU training must disable the backend that produced nonzero zero-gradients')
   torch.manual_seed(123);m=phase_shift_model.PhaseShiftRestorer(phase_shift_model.AdaptivePhaseFactor()).train();p=[v for v in m.parameters() if v.requires_grad];x=torch.randn(4,2,256);loss=m(x).square().mean();zero=torch.autograd.grad(loss*0,p,allow_unused=True)
   self.assertTrue(all(g is None or torch.count_nonzero(g)==0 for g in zero))
  finally:torch.backends.mkldnn.enabled=prior
if __name__=='__main__':unittest.main()
