import unittest
import torch
from torch import nn
from experiments.phase1_frozen_feature_projector.model import FeatureProjector

class Reader(nn.Module):
 def __init__(self):
  super().__init__();self.projection=nn.Linear(512,160);self.head=nn.Linear(160,6)
  self.eval().requires_grad_(False)
 def features(self,x):return self.projection(x.flatten(1))
 def classify_features(self,x):return self.head(x)

class ProjectorTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(91);torch.set_num_threads(2);self.reader=Reader();self.model=FeatureProjector(self.reader,width=8)
  self.x=torch.randn(3,2,256);self.lv=torch.full((3,),-2.)
 def test_initial_is_input_and_reader_frozen(self):
  out=self.model(self.x,self.x,self.lv)
  self.assertTrue(torch.equal(out,self.x));self.assertFalse(self.reader.training)
  self.assertTrue(all(not p.requires_grad for p in self.reader.parameters()))
 def test_target_feature_is_trainable_without_reader_updates(self):
  clean=torch.randn_like(self.x);self.model(self.x,self.x,self.lv)
  target=self.model.latest_target;loss=1-torch.nn.functional.cosine_similarity(target,self.reader.features(clean),dim=1).mean()
  loss.backward();self.assertGreater(float(self.model.feature_head[-1].weight.grad.abs().sum()),0.)
  self.assertTrue(all(p.grad is None for p in self.reader.parameters()))
 def test_guidance_works_inside_no_grad_and_is_batch_independent(self):
  with torch.no_grad():
   self.model.feature_head[-1].weight.normal_(0,.03);self.model.step.fill_(.5)
   both=self.model(self.x,self.x,self.lv);one=self.model(self.x[:1],self.x[:1],self.lv[:1])
   self.model.step.zero_();off=self.model(self.x,self.x,self.lv)
  self.assertTrue(torch.allclose(one,both[:1],rtol=1e-5,atol=1e-5))
  self.assertGreater(float((both-off).abs().max()),1e-6)
  self.assertTrue(all(p.grad is None for p in self.reader.parameters()))
 def test_feature_direction_and_direct_gate_have_finite_gradients(self):
  out=self.model(self.x,self.x,self.lv);loss=out.square().mean();loss.backward()
  self.assertTrue(torch.isfinite(self.model.gate_output.weight.grad).all())
  self.assertGreater(float(self.model.gate_output.weight.grad.abs().sum()),0.)
 def test_rejects_mismatched_noise(self):
  with self.assertRaises(ValueError):self.model(self.x,self.x,self.lv[:1])

if __name__=='__main__':unittest.main()
