import unittest,torch
from phase_factor_model import AdaptivePhaseFactor
from phase_shift_model import PhaseShiftRestorer
class ArchitectureTests(unittest.TestCase):
 def test_identity_anchor_frozen_phase_and_gradient(self):
  torch.manual_seed(52);m=PhaseShiftRestorer(AdaptivePhaseFactor());m.train();x=torch.randn(2,2,256);a=m.anchor(x);y=m(x)
  self.assertEqual(tuple(y.shape),(2,2,256));self.assertTrue(torch.allclose(a,y,atol=2e-5,rtol=2e-5));self.assertFalse(m.phase.training);self.assertTrue(all(not p.requires_grad for p in m.phase.parameters()));y.square().mean().backward();self.assertTrue(torch.isfinite(y).all());self.assertTrue(m.body.temporal_output.weight.grad.abs().sum()>0);self.assertTrue(m.body.output.weight.grad.abs().sum()>0);self.assertTrue(all(p.grad is None for p in m.phase.parameters()))
if __name__=='__main__':unittest.main()
