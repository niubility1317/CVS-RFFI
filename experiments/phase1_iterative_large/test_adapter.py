import unittest
from unittest.mock import patch
from experiments.phase1_iterative_large import train_source,score_source,ground
from experiments.phase1_iterative_large.sampling import select_large_fit

class AdapterTest(unittest.TestCase):
 def test_training_installs_nested_fit_and_restores_on_failure(self):
  a=train_source;base=a.base;names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_loss','FIT_SIZE','select_fit');before=tuple(getattr(base,k) for k in names)
  def fail():
   self.assertEqual(base.FIT_SIZE,32768);self.assertIs(base.select_fit,select_large_fit);raise RuntimeError('stop')
  with patch.object(base,'main',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'stop'):a.main()
  self.assertEqual(before,tuple(getattr(base,k) for k in names))
 def test_scoring_and_ground_restore_size_after_failure(self):
  for a in (score_source,ground):
   base=a.base;old=base.FIT_SIZE
   def fail():
    self.assertEqual(base.FIT_SIZE,32768);raise RuntimeError('stop')
   with patch.object(base,'main',side_effect=fail):
    with self.assertRaisesRegex(RuntimeError,'stop'):a.main()
   self.assertEqual(base.FIT_SIZE,old)

if __name__=='__main__':unittest.main()
