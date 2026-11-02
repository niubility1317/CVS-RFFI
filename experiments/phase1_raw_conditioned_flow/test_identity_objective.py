import unittest
import torch
from raw_conditioned_flow import RawConditionedFlow
from identity_objective import identity_objective

class Teacher(torch.nn.Module):
 def __init__(self):
  super().__init__();self.head=torch.nn.Linear(512,6);self.requires_grad_(False)
 def features(self,x):return x.flatten(1)
 def classify_features(self,x):return self.head(x)

class IdentityObjectiveTest(unittest.TestCase):
 def test_identical_pairs_start_with_zero_loss(self):
  torch.manual_seed(2);net=RawConditionedFlow();teacher=Teacher();x=torch.randn(3,2,256);t=torch.tensor([.1,.4,.8]);loss,parts=identity_objective(net,teacher,x,x,x,t);self.assertLess(float(loss.detach().abs()),1e-6);self.assertEqual(float(parts['clean'].detach()),0.)
 def test_distorted_pairs_train_network_without_teacher_gradients(self):
  torch.manual_seed(3);net=RawConditionedFlow();teacher=Teacher();x=torch.randn(3,2,256);y=x*.7;t=torch.tensor([.1,.4,.8]);loss,parts=identity_objective(net,teacher,x,y,y,t);loss.backward();self.assertGreater(float(net.output.weight.grad.abs().sum()),0.);self.assertTrue(all(p.grad is None for p in teacher.parameters()));self.assertTrue(torch.isfinite(loss))
if __name__=='__main__':unittest.main()
