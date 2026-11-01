import math,unittest
import torch
from phase_factor_model import AdaptivePhaseFactor
from phase_factor_source import target_gain,choose

class PhaseTests(unittest.TestCase):
 def test_rotation_structure(self):
  torch.manual_seed(71);m=AdaptivePhaseFactor();x=torch.randn(8,2,256);theta=.83;c=math.cos(theta);s=math.sin(theta);xr=torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),1)
  a,o,g=m.components(x);ar,orr,gr=m.components(xr);rot=complex(c,-s)
  self.assertTrue(torch.allclose(ar,a*rot,atol=2e-5,rtol=2e-5));self.assertTrue(torch.allclose(o,orr,atol=2e-5,rtol=2e-5));self.assertTrue(torch.allclose(g,gr,atol=2e-5,rtol=2e-5))
  y=m(x);self.assertTrue(torch.isfinite(y).all());self.assertTrue(torch.allclose(y.abs(),torch.ones_like(y.abs()),atol=1e-5));y.real.sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_gain_and_delay(self):
  torch.manual_seed(72);x=torch.randn(4,2,256);z=torch.complex(x[:,0],x[:,1]);theta=.71;rot=complex(math.cos(theta),math.sin(theta));target=z*rot;ref=torch.stack((target.real,target.imag),1);phase,valid=target_gain(x,ref)
  self.assertTrue(valid.all());self.assertTrue(torch.allclose(phase,torch.full_like(phase,rot),atol=1e-5))
  zero=torch.zeros_like(x);_,valid=target_gain(zero,zero);self.assertFalse(valid.any())
 def test_selection_guards(self):
  a={'1:4:view':{'count':10,'baseline':.5,'candidate':.3},'1:6:view':{'count':10,'baseline':.5,'candidate':.4},'2:8:view':{'count':10,'baseline':.5,'candidate':1.9}}
  self.assertEqual(choose({1:a})['kind'],'adaptive_phase_factor');a['1:6:view']['candidate']=.56;self.assertEqual(choose({1:a})['kind'],'identity')
if __name__=='__main__':unittest.main()
