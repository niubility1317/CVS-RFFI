import unittest
try:from residual_selection import choose,VIEWS
except ImportError:choose=None;VIEWS=()
def make():
 return {f'{part}:{rx}:{v}':dict(count=100,valid_count=100,baseline_correct=90,parent_correct=90,candidate_correct=90 if v=='clean' else 92,baseline_iq=.1,parent_iq=.1,candidate_iq=.1) for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for v in VIEWS}
class SelectionTests(unittest.TestCase):
 def test_accept_and_holdout_excluded(self):
  self.assertIsNotNone(choose);a=make();a['2:8:practical_low_urban']['candidate_correct']=0;self.assertEqual(choose({'0.99':a},True,'baseline')['validation_net'],24)
 def test_clean_and_cell_regression_rejected(self):
  a=make();a['1:4:clean']['candidate_iq']=.11;self.assertEqual(choose({'0.99':a},False,'baseline')['kind'],'none');a=make();a['1:4:practical_low_urban']['candidate_correct']=88;self.assertEqual(choose({'0.99':a},False,'baseline')['kind'],'none')
 def test_iq_guard_and_parent_comparison(self):
  a=make();a['1:4:practical_low_urban']['candidate_iq']=.13;self.assertEqual(choose({'0.99':a},True,'baseline')['kind'],'none');self.assertEqual(choose({'0.99':a},False,'parent')['kind'],'residual')
if __name__=='__main__':unittest.main()
