import unittest,torch
from phase1_self_reference_model import SelfReferenceNet
try:import self_reference_inference as inference
except ImportError:inference=None
class FixedPhase:
 def anchor(self,x):return x.roll(3,-1)
class FixedGate:
 def __call__(self,x):return torch.tensor([-10.,10.],device=x.device)[:len(x)]
class ClosedGate:
 def __call__(self,x):return torch.full((len(x),),-10.,device=x.device)
class NeverPhase:
 def anchor(self,x):raise AssertionError('disabled records must bypass restoration')
class InferenceTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(inference);torch.manual_seed(13);self.net=SelfReferenceNet().eval()
  with torch.no_grad():self.net.output.bias.fill_(.1)
 def test_protection_and_full_batch_equivalence(self):
  x=torch.randn(2,2,256);a=FixedPhase().anchor(x);full=self.net(a);y,on=inference.apply(self.net,FixedPhase(),FixedGate(),.5,.8,x)
  self.assertFalse(on[0]);self.assertTrue(on[1]);self.assertTrue(torch.equal(y[0],x[0]));self.assertTrue(torch.allclose(y[1:],(a+.5*(full-a))[1:],atol=1e-6,rtol=1e-6))
 def test_all_disabled_skips_restoration(self):
  x=torch.randn(3,2,256);y,on=inference.apply(self.net,NeverPhase(),ClosedGate(),1.,.8,x);self.assertTrue(torch.equal(y,x));self.assertFalse(on.any())
if __name__=='__main__':unittest.main()
