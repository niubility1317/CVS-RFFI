import unittest
from unittest.mock import patch
from experiments.phase1_dura_physical_wide import train_source as a

class AdapterTest(unittest.TestCase):
 def test_restores_contract_on_failure(self):
  names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components')
  before=tuple(getattr(a.base,k) for k in names)
  def fail():
   self.assertIs(a.base.DynamicVolterra,a.WidePhysicalFusion);self.assertTrue(a.base.USES_DURA);raise RuntimeError('synthetic')
  with patch.object(a.base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'synthetic'):a.main()
  self.assertEqual(before,tuple(getattr(a.base,k) for k in names))

if __name__=='__main__':unittest.main()
