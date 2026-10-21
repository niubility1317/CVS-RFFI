import copy,unittest
from . import plan as p

class PlanTests(unittest.TestCase):
 def test_complete_six_readers_and_unique_cells(self):
  self.assertEqual(len(p.rows()),6)
  reused=sum(p.reused(r,m,v) for r in p.rows() for m in ('raw','dura') for v in p.VIEWS)
  self.assertEqual(reused,37);self.assertEqual(84-reused,47)
 def test_bad_row_and_view_closed(self):
  for rid in ('NOAS-s392005','SARD-s2026092704','T1-s2026092701'):
   with self.assertRaises(ValueError):p.row(rid)
  with self.assertRaises(ValueError):p.reused(p.rows()[0],'raw','leo_clear_weak')
 def test_recovery_preserves_completed_NoAs(self):
  self.assertEqual(len(p.pending_rows()),3)
  self.assertTrue(all(r['arm']=='SARD' for r in p.pending_rows()))
  self.assertTrue(all(p.row_root(r).parent==p.PREVIOUS_BASE for r in p.rows() if r['arm']=='NOAS'))
  self.assertTrue(all(p.row_root(r).parent==p.BASE for r in p.pending_rows()))
 def test_training_and_runtime_mutations_closed(self):
  for key,value in [('new_training_updates',1),('head_fit_updates',1),('target_fit',True),('restorer','old_R17')]:
   x=copy.deepcopy(p.policy());x[key]=value
   with self.assertRaises(ValueError):p.validate(x)
  x=copy.deepcopy(p.policy());x['restorer_runtime']['native']=True
  with self.assertRaises(ValueError):p.validate(x)

if __name__=='__main__':unittest.main()
