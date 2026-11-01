import unittest
try:
 import self_reference_selection as selection
except ImportError:
 selection=None
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def make_cells():
 return {f'{part}:{rx}:{v}':dict(count=100,valid_count=100,baseline_correct=90,candidate_correct=90 if v=='clean' else 92,baseline_iq=.1,candidate_iq=.1) for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for v in VIEWS}
class SelectionTests(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(selection,'source selection implementation required')
 def test_source_guards_accept_improvement_and_ignore_holdout(self):
  a=make_cells();a['2:8:practical_low_urban']['candidate_correct']=0
  result=selection.choose({1:{'1.0:0.8':a}});self.assertEqual(result['kind'],'self_reference_net');self.assertEqual(result['validation_net'],24)
 def test_rejects_regression_or_missing_scene(self):
  a=make_cells();a['1:4:practical_low_urban']['candidate_correct']=88
  self.assertEqual(selection.choose({1:{'1.0:0.8':a}})['kind'],'dura')
  del a['1:6:practical_high']
  with self.assertRaises(ValueError):selection.choose({1:{'1.0:0.8':a}})
if __name__=='__main__':unittest.main()
