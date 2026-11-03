import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.phase1_dura_iterative_timefreq import train_source
from experiments.phase1_dura_iterative_timefreq.model import IterativeDura
from experiments.phase1_dura_volterra.pipeline import predict_batch

class Gate(nn.Module):
 def forward(self,x):return x[:,0,0]*10
class Blind(nn.Module):
 def forward(self,x):return torch.zeros(len(x),4),torch.zeros(len(x))
class Reader(nn.Module):
 def features(self,x):return x.mean(-1)
 def classify_features(self,x):return x.repeat(1,3)

class IterativeDuraTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(19);torch.set_num_threads(2);self.x=torch.randn(2,2,256);self.x[0,0,0]=-2;self.x[1,0,0]=2;self.m=IterativeDura(width=16)
 def test_zero_head_and_closed_gate_preserve_dura(self):
  dura=lambda x:x*.7;base,out,active,_=predict_batch(self.x,self.m,Blind(),Gate(),dura,None,Reader(),None)
  self.assertEqual(active.tolist(),[False,True]);self.assertTrue(torch.equal(base,out))
  with torch.no_grad():self.m.gate_output.bias[0]=.1
  base,out,active,_=predict_batch(self.x,self.m,Blind(),Gate(),dura,None,Reader(),None)
  self.assertTrue(torch.equal(base[~active],out[~active]));self.assertGreater(float((out[active]-base[active]).abs().sum()),0)
 def test_joint_rotation_of_received_and_dura_rotates_output(self):
  a=torch.randn_like(self.x);lv=torch.full((2,),-2.)
  with torch.no_grad():
   self.m.gate_output.weight.normal_(0,.01);self.m.frequency_gate.weight.normal_(0,.01)
   y=self.m(self.x,a,lv);rot=complex(.6,.8)
   def rotate(x):
    z=torch.complex(x[:,0],x[:,1])*rot;return torch.stack((z.real,z.imag),1)
   self.assertTrue(torch.allclose(self.m(rotate(self.x),rotate(a),lv),rotate(y),atol=6e-6,rtol=6e-6))
 def test_protected_source_loss_trains_both_paths_without_updating_reader(self):
  class FrozenReader(nn.Module):
   def __init__(self):super().__init__();self.f=nn.Linear(512,160);self.head=nn.Linear(160,6);self.eval().requires_grad_(False)
   def features(self,x):return self.f(x.flatten(1))
   def classify_features(self,x):return self.head(x)
  r=FrozenReader();a=(self.x*.7).detach();clean=torch.randn_like(self.x);y=self.m(self.x,a,torch.full((2,),-2.))
  loss,_=train_source.protected_source_objective(r,y,a,clean,r.features(clean),torch.tensor([4,8]));loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(self.m.gate_output.weight.grad.abs().sum()),0);self.assertGreater(float(self.m.frequency_gate.weight.grad.abs().sum()),0)
  self.assertTrue(all(p.grad is None for p in r.parameters()))
 def test_training_adapter_freezes_dura_and_restores_globals_on_failure(self):
  base=train_source.base;names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components');old=tuple(getattr(base,k) for k in names)
  def fail():
   self.assertIs(base.DynamicVolterra,IterativeDura);self.assertIs(base.source_objective,train_source.protected_source_objective);self.assertTrue(base.USES_DURA);raise RuntimeError('stop')
  with patch.object(base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'stop'):train_source.main()
  self.assertEqual(old,tuple(getattr(base,k) for k in names))

if __name__=='__main__':unittest.main()
