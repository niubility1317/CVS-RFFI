"""Focused placement/replay/source-rule negatives; no real data or models."""
import unittest
import numpy as np
from .maths import replay_guard,source_rule
from .probe import prior_guard
class Checks(unittest.TestCase):
 def test_replay_and_G1(self):
  x=np.arange(256,dtype=np.float32).astype(complex);h=np.ones(62,dtype=complex);g=np.ones(1,dtype=complex);self.assertEqual(replay_guard(x,x,h,h,g,g),0.)
  cases=[(x+1e-3,x,h,h,g,g),(x,x[:255],h,h,g,g),(x*np.nan,x,h,h,g,g),(x,x,h,h+1e-5,g,g),(x,x,h,h,g,np.ones(2)),(x,x,h,h,g,np.array([.9]))]
  for args in cases:
   with self.assertRaises(ValueError):replay_guard(*args)
 def test_source_rule(self):
  exact=100*(1215/1800-1224/1800);r=source_rule(True,[[exact]*2]*2,'B01');self.assertEqual(r['status'],'PASS_SOURCE_PLACEMENT_ONLY');self.assertEqual(r['protected_cells'],4)
  self.assertEqual(source_rule(True,[[-.500001,0],[0,0]],'B01')['status'],'FAIL_FIXED_PRE_RECEIVER_CYCLE_ORDER')
  self.assertEqual(source_rule(False,[[0]*2]*2,'B01')['status'],'FAIL_FIXED_PRE_RECEIVER_CYCLE_ORDER')
  self.assertEqual(source_rule(True,[[-5]*2]*2,'B00')['status'],'PASS_SOURCE_PLACEMENT_ONLY')
  for clean,x,arm in [(True,[[0,0]],'B01'),(True,[[np.nan,0],[0,0]],'B01'),(1,[[0]*2]*2,'B01'),(True,[[0]*2]*2,'other')]:
   with self.assertRaises(ValueError):source_rule(clean,x,arm)
 def test_clean_guard_scope(self):
  checks={k:True for k in ('clean_protected','every_RX_protected','every_TX_protected','clean_correct_retained','periodic_reference_supported','nontrivial_view','clean_phase_all_defined')};checks['urban_protected']=False
  self.assertTrue(prior_guard({'decision':{'checks':checks}}))
  for key in list(checks):
   if key=='urban_protected':continue
   bad=dict(checks);bad[key]=False;self.assertFalse(prior_guard({'decision':{'checks':bad}}))
  bad=dict(checks);del bad['every_RX_protected']
  with self.assertRaises(ValueError):prior_guard({'decision':{'checks':bad}})
if __name__=='__main__':unittest.main()
