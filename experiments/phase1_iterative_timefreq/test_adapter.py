import unittest
from unittest.mock import patch
import torch
from torch import nn
class Reader(nn.Module):
 def __init__(self):super().__init__();self.f=nn.Linear(512,160);self.head=nn.Linear(160,6);self.eval().requires_grad_(False)
 def features(self,x):return self.f(x.flatten(1))
 def classify_features(self,x):return self.head(x)
from experiments.phase1_iterative_timefreq import train_source as adapter
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

class TrainingAdapterTest(unittest.TestCase):
 def test_training_loss_updates_both_outputs_and_freezes_reader(self):
  torch.manual_seed(81);torch.set_num_threads(2);reader=Reader();model=IterativeTimeFreq(width=16);x=torch.randn(3,2,256);clean=torch.randn_like(x);lv=torch.full((3,),-2.);y=model(x,x,lv)
  loss,terms=adapter.source_loss(reader,y,clean,reader.features(clean),torch.tensor([4,4,8]));loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertTrue(all(k in terms for k in ('aligned','feature','distill','margin')))
  self.assertGreater(float(model.gate_output.weight.grad.abs().sum()),0);self.assertGreater(float(model.frequency_gate.weight.grad.abs().sum()),0)
  self.assertTrue(all(p.grad is None for p in reader.parameters()))
 def test_adapter_restores_base_contract_on_failure(self):
  base=adapter.base;names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_loss');before=tuple(getattr(base,k) for k in names)
  def fail():
   self.assertIs(base.DynamicVolterra,IterativeTimeFreq);self.assertIs(base.source_loss,adapter.source_loss);raise RuntimeError('synthetic')
  with patch.object(base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'synthetic'):adapter.main()
  self.assertEqual(before,tuple(getattr(base,k) for k in names))

if __name__=='__main__':unittest.main()
