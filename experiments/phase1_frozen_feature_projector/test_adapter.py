import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.phase1_frozen_feature_projector import train_source as a
from experiments.phase1_frozen_feature_projector.model import FeatureProjector

class Reader(nn.Module):
 def __init__(self):super().__init__();self.f=nn.Linear(512,160);self.head=nn.Linear(160,6);self.eval().requires_grad_(False)
 def features(self,x):return self.f(x.flatten(1))
 def classify_features(self,x):return self.head(x)
class AdapterTest(unittest.TestCase):
 def test_reader_frozen_and_both_model_heads_learn(self):
  torch.manual_seed(13);torch.set_num_threads(2);reader=Reader();m=FeatureProjector(reader,width=8);a._current[0]=m
  x=torch.randn(3,2,256);clean=torch.randn_like(x);lv=torch.full((3,),-2.);out=m(x,x,lv)
  loss,terms=a.projector_objective(reader,out,x,clean,reader.features(clean),torch.tensor([4,4,8]))
  loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(terms['predicted_feature']))
  self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0.)
  self.assertGreater(float(m.feature_head[-1].weight.grad.abs().sum()),0.)
  self.assertTrue(all(p.grad is None for p in reader.parameters()));a._current[0]=None
 def test_adapter_restores_contract_on_failure(self):
  names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA','extra_frozen_model_components')
  before=tuple(getattr(a.base,k) for k in names)
  def fail():
   self.assertIs(a.base.source_objective,a.projector_objective);self.assertFalse(a.base.USES_DURA)
   raise RuntimeError('synthetic failure')
  with patch.object(a.base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'synthetic failure'):a.main()
  self.assertEqual(before,tuple(getattr(a.base,k) for k in names));self.assertIsNone(a._current[0])

if __name__=='__main__':unittest.main()
