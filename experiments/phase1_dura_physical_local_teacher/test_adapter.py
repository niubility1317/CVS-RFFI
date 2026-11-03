import unittest
import torch
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion
from experiments.phase1_dura_physical_fusion.test_objective import Reader
from experiments.phase1_dura_physical_local_teacher.train_source import combined_objective

class LocalTeacherAdapterTest(unittest.TestCase):
 def test_trainable_fusion_receives_gradient_and_fixed_reader_remains_frozen(self):
  torch.manual_seed(14);torch.set_num_threads(2);m=PhysicalFusion(width=8);reader=Reader()
  x=torch.randn(3,2,256);clean=torch.randn_like(x);anchor=torch.randn_like(x);physical=torch.randn_like(x)
  packed=torch.cat((anchor,physical),1);lv=torch.full((3,),-2.)
  out=m(x,packed,lv);loss,terms=combined_objective(reader,out,packed,clean,reader.features(clean),torch.tensor([4,4,8]),x,lv)
  loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(terms['local_teacher']))
  self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in reader.parameters()))

class ContractRestoreTest(unittest.TestCase):
 def test_training_globals_restore_after_failure(self):
  from unittest.mock import patch
  from experiments.phase1_dura_physical_local_teacher import train_source as a
  names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components')
  old=tuple(getattr(a.base,k) for k in names)
  def fail():
   self.assertIs(a.base.training_source_objective,a.combined_objective);raise RuntimeError('test stop')
  with patch.object(a.base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'test stop'):a.main()
  self.assertEqual(old,tuple(getattr(a.base,k) for k in names))

if __name__=='__main__':unittest.main()
