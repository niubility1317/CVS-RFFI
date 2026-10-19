import importlib.util,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
s=importlib.util.spec_from_file_location('loss84',Path(__file__).with_name('loss.py'));loss=importlib.util.module_from_spec(s);s.loader.exec_module(loss)
def old(a,b,y,alpha):
 c0=F.cross_entropy(a,y,label_smoothing=.01);c1=F.cross_entropy(b,y,label_smoothing=.01);p,q=a.log_softmax(1),b.log_softmax(1);k=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean();return .5*(c0+c1)+alpha*k,.5*(c0+c1),k,c0,c1
class TemperatureTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(84);self.a=torch.randn(128,6,dtype=torch.float64,requires_grad=True);self.b=torch.randn(128,6,dtype=torch.float64,requires_grad=True);self.y=torch.arange(128)%6;self.mask=torch.arange(128)%2==0
 def test_T1_value_and_gradient_exact(self):
  x=loss.pair(self.a,self.b,self.y,1.)[0];y=old(self.a,self.b,self.y,0.)[0];self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(torch.autograd.grad(x,self.a,retain_graph=True)[0],torch.autograd.grad(y,self.a)[0]))
 def test_T6_compensated_gradient(self):
  v=loss.pair(self.a,self.b,self.y,6.)[0];g=torch.autograd.grad(v,self.a)[0];target=F.one_hot(self.y,6).to(self.a)*.99+.01/6;expected=.5*((self.a/6).softmax(1)-target)/128;self.assertTrue(torch.allclose(g,expected,atol=1e-12,rtol=1e-12))
 def test_only_As_changes_and_ordinary_KL_preserved(self):
  a=loss.mixed(self.a,self.b,self.y,self.mask,1.,old)[0];b=loss.mixed(self.a,self.b,self.y,self.mask,6.,old)[0];ga=torch.autograd.grad(a,self.a,retain_graph=True)[0];gb=torch.autograd.grad(b,self.a)[0];self.assertTrue(torch.equal(ga[~self.mask],gb[~self.mask]));self.assertGreater(float((ga[self.mask]-gb[self.mask]).abs().sum()),0);v=loss.mixed(self.a,self.b,self.y,self.mask,6.,old);self.assertTrue(torch.equal(v[2],.5*old(self.a[~self.mask],self.b[~self.mask],self.y[~self.mask],1.)[2]))
 def test_hidden_labels_and_invalid_temperature_rejected(self):
  for t in (0,-1,float('nan'),float('inf'),True):
   with self.assertRaises(ValueError):loss.pair(self.a,self.b,self.y,t)
  y=self.y.clone();y[0]=-1
  with self.assertRaises(ValueError):loss.pair(self.a,self.b,y,6)
  with self.assertRaises(ValueError):loss.mixed(self.a,self.b,self.y,~(torch.arange(128)==0),6.,old)
 def test_logits_unmodified_argmax_unchanged(self):
  before=self.a.detach().clone();loss.pair(self.a,self.b,self.y,6);self.assertTrue(torch.equal(before,self.a));self.assertTrue(torch.equal(self.a.argmax(1),(self.a/6).argmax(1)))
if __name__=='__main__':unittest.main()
