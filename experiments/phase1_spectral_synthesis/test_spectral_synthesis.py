import importlib,importlib.util,unittest
import torch

def rotate(x,angle):
 c=torch.cos(torch.tensor(angle));s=torch.sin(torch.tensor(angle))
 return torch.stack((c*x[...,0,:]-s*x[...,1,:],s*x[...,0,:]+c*x[...,1,:]),dim=-2)

class SpectralSynthesisTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('spectral_synthesis'),'spectral synthesis network must exist');self.mod=importlib.import_module('spectral_synthesis');torch.manual_seed(91);self.base=torch.randn(3,2,256);self.candidates=torch.randn(3,7,2,256);self.lv=torch.tensor([-4.,-2.,0.]);self.net=self.mod.SpectralSynthesis()
 def test_initial_prediction_preserves_physical_base(self):
  out=self.net(self.base,self.candidates,self.lv);torch.testing.assert_close(out,self.base,atol=2e-6,rtol=2e-6)
 def test_learned_network_equivariant_to_global_iq_phase(self):
  with torch.no_grad():self.net.output.weight.normal_(0,.03);self.net.output.bias.normal_(0,.03)
  out=self.net(self.base,self.candidates,self.lv);rotated=self.net(rotate(self.base,.7),rotate(self.candidates,.7),self.lv);torch.testing.assert_close(rotated,rotate(out,.7),atol=3e-5,rtol=3e-5)
 def test_no_change_when_all_hypotheses_equal_base(self):
  with torch.no_grad():self.net.output.weight.normal_(0,.1);self.net.output.bias.normal_(0,.1)
  out=self.net(self.base,self.base[:,None].expand(-1,7,-1,-1),self.lv);torch.testing.assert_close(out,self.base,atol=2e-6,rtol=2e-6)
 def test_training_gradient_finite_and_zero_signal_is_finite(self):
  out=self.net(self.base,self.candidates,self.lv);out.square().mean().backward();g=self.net.output.weight.grad;self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0);zero=torch.zeros_like(self.base);self.assertTrue(torch.isfinite(self.net(zero,torch.zeros_like(self.candidates),self.lv)).all())
if __name__=='__main__':unittest.main()
