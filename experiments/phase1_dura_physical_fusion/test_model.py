import unittest
import torch
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion


def rotate(x,a):
 c,s=torch.cos(a),torch.sin(a)
 return torch.stack((x[:,0]*c-x[:,1]*s,x[:,0]*s+x[:,1]*c),1)


class ModelTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(11);torch.set_num_threads(2)
  self.x=torch.randn(2,2,256);self.anchor=torch.randn(2,2,256);self.posterior=torch.randn(2,2,256)
  self.lv=torch.full((2,),-2.);self.model=PhysicalFusion(width=8).eval()
 def test_initial_equals_anchor_and_has_gradient(self):
  y=self.model(self.x,torch.cat((self.anchor,self.posterior),1),self.lv)
  self.assertTrue(torch.equal(y,self.anchor));y.square().mean().backward()
  self.assertGreater(float(self.model.gate_output.weight.grad.abs().sum()),0.)
 def test_learned_path_phase_and_batch_independent(self):
  with torch.no_grad():
   self.model.gate_output.weight.normal_(0,.05);self.model.gate_output.bias.normal_(0,.05)
   y=self.model(self.x,torch.cat((self.anchor,self.posterior),1),self.lv)
   one=self.model(self.x[:1],torch.cat((self.anchor[:1],self.posterior[:1]),1),self.lv[:1])
   a=torch.tensor(.7);turn=self.model(rotate(self.x,a),torch.cat((rotate(self.anchor,a),rotate(self.posterior,a)),1),self.lv)
  self.assertTrue(torch.allclose(one,y[:1],atol=2e-5,rtol=2e-5))
  self.assertTrue(torch.allclose(turn,rotate(y,a),atol=2e-5,rtol=2e-5))
 def test_bad_packed_shape_rejected(self):
  with self.assertRaises(ValueError): self.model(self.x,self.anchor,self.lv)

if __name__=='__main__':unittest.main()
