import unittest
import torch
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

class IterativeModelTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(31);torch.set_num_threads(2);self.m=IterativeTimeFreq(width=16,steps=3);self.x=torch.randn(3,2,256);self.lv=torch.full((3,),-2.)
 def test_zero_initialization_is_exact_identity(self):
  self.assertTrue(torch.equal(self.m(self.x,self.x,self.lv),self.x))
 def test_sample_independence_after_learning(self):
  with torch.no_grad():
   self.m.gate_output.weight.normal_(0,.01);self.m.frequency_gate.weight.normal_(0,.01)
   batch=self.m(self.x,self.x,self.lv);single=self.m(self.x[:1],self.x[:1],self.lv[:1])
  self.assertTrue(torch.allclose(batch[:1],single,atol=2e-6,rtol=2e-6))
 def test_global_phase_rotation_equivariance_after_learning(self):
  with torch.no_grad():
   self.m.gate_output.weight.normal_(0,.01);self.m.frequency_gate.weight.normal_(0,.01)
   a=self.m(self.x,self.x,self.lv);rot=torch.complex(torch.tensor(.6),torch.tensor(.8));z=torch.complex(self.x[:,0],self.x[:,1])*rot;rx=torch.stack((z.real,z.imag),1);b=self.m(rx,rx,self.lv)
   expected=torch.complex(a[:,0],a[:,1])*rot;target=torch.stack((expected.real,expected.imag),1)
  self.assertTrue(torch.allclose(b,target,atol=5e-6,rtol=5e-6))
 def test_multiple_steps_accumulate_nonzero_corrections(self):
  one=IterativeTimeFreq(width=16,steps=1);one.load_state_dict(self.m.state_dict())
  with torch.no_grad():
   self.m.gate_output.bias[0]=.1;one.gate_output.bias[0]=.1
   one_step=one(self.x,self.x,self.lv)-self.x;three=self.m(self.x,self.x,self.lv)-self.x
  self.assertGreater(float(three.abs().max()),0);self.assertTrue(torch.allclose(three,3*one_step,atol=3e-6,rtol=3e-6))
 def test_both_paths_receive_finite_gradients(self):
  out=self.m(self.x,self.x,self.lv);loss=(out-torch.randn_like(out)).square().mean();loss.backward()
  for layer in (self.m.gate_output,self.m.frequency_gate):
   self.assertIsNotNone(layer.weight.grad);self.assertTrue(torch.isfinite(layer.weight.grad).all());self.assertGreater(float(layer.weight.grad.abs().sum()),0.)
 def test_invalid_noise_and_nonfinite_input_rejected(self):
  with self.assertRaises(ValueError):self.m(self.x,self.x,self.lv[:1])
  x=self.x.clone();x[0,0,0]=float('nan')
  with self.assertRaises(FloatingPointError):self.m(x,x,self.lv)

if __name__=='__main__':unittest.main()
