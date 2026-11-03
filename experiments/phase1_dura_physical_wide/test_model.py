import unittest
import torch
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion
from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion

class WideTest(unittest.TestCase):
 def test_wide_delta_is_double_same_weights_and_initial_identity(self):
  torch.manual_seed(45);torch.set_num_threads(2);a=PhysicalFusion(width=8);b=WidePhysicalFusion(width=8)
  x=torch.randn(2,2,256);anchor=torch.randn_like(x);physical=torch.randn_like(x);pack=torch.cat((anchor,physical),1);lv=torch.full((2,),-2.)
  b.load_state_dict(a.state_dict());self.assertTrue(torch.equal(b(x,pack,lv),anchor))
  with torch.no_grad():a.gate_output.weight.normal_(0,.04);a.gate_output.bias.normal_(0,.04);b.load_state_dict(a.state_dict())
  reference=a(x,pack,lv);wide=b(x,pack,lv)
  self.assertTrue(torch.allclose(wide-anchor,2*(reference-anchor),atol=1e-6,rtol=1e-6))
  wide.square().mean().backward();self.assertTrue(torch.isfinite(b.gate_output.weight.grad).all())
  self.assertGreater(float(b.gate_output.weight.grad.abs().sum()),0.)

if __name__=='__main__':unittest.main()
