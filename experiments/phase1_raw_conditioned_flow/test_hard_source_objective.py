import unittest
import torch
from hard_source_objective import select_targets,hard_source_loss
from global_spectral_flow import GlobalSpectralFlow

class Teacher(torch.nn.Module):
 def __init__(self):
  super().__init__();self.head=torch.nn.Linear(512,6);self.requires_grad_(False)
 def features(self,x):return x.flatten(1)
 def classify_features(self,x):return self.head(x)

class HardSourceTest(unittest.TestCase):
 def test_labeled_targets_override_pseudo_and_low_confidence_is_excluded(self):
  labels=torch.tensor([4,-1,-1]);confidence=torch.tensor([.7,.96,.2]);pseudo=torch.tensor([2,3,1]);raw=torch.tensor([2,3,0]);target,valid,weight=select_targets(labels,confidence,pseudo,raw)
  self.assertEqual(target.tolist(),[4,3,1]);self.assertEqual(valid.tolist(),[True,True,False]);self.assertEqual(weight.tolist(),[3.,1.,3.])
 def test_training_gradient_reaches_flow_but_not_teacher(self):
  torch.manual_seed(5);net=GlobalSpectralFlow();teacher=Teacher();x=torch.randn(3,2,256);clean=torch.randn_like(x);labels=torch.tensor([0,1,2]);loss,stats=hard_source_loss(net,teacher,x,clean,labels);loss.backward();self.assertGreater(float(net.frequency_bias.grad.abs().sum()),0.);self.assertTrue(all(p.grad is None for p in teacher.parameters()));self.assertEqual(stats['labeled'],3)
if __name__=='__main__':unittest.main()
