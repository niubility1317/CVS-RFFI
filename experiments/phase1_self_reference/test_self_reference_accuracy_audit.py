import unittest
from test_self_reference_selection import make_cells
try:from self_reference_accuracy_audit import choose_accuracy
except ImportError:choose_accuracy=None
class AccuracyAuditTests(unittest.TestCase):
 def test_allows_impaired_waveform_tradeoff_but_reports_it(self):
  self.assertIsNotNone(choose_accuracy);cells=make_cells()
  for k,v in cells.items():
   if not k.endswith(':clean'):v['candidate_iq']=1.
  result=choose_accuracy({1:{'1.0:0.8':cells}});self.assertEqual(result['kind'],'self_reference_net');self.assertEqual(result['validation_net'],24);self.assertGreater(result['max_impaired_iq_ratio'],9.)
 def test_protects_clean_and_each_classification_cell(self):
  cells=make_cells();cells['1:4:clean']['candidate_iq']=.102
  self.assertEqual(choose_accuracy({1:{'1.0:0.8':cells}})['kind'],'none')
  cells=make_cells();cells['1:4:practical_low_urban']['candidate_correct']=88
  self.assertEqual(choose_accuracy({1:{'1.0:0.8':cells}})['kind'],'none')
if __name__=='__main__':unittest.main()
