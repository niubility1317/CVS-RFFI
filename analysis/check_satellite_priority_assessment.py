"""Focused checks for absolute-pp reporting, matching and seed accounting."""
import copy
import importlib.util
import unittest
from pathlib import Path
p=Path(__file__).with_name('cvs_satellite_priority_assessment_20261018.py')
s=importlib.util.spec_from_file_location('assessment',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def row(seed,correct):
 return dict(model_seed=seed,count=10000,source={'cost':dict(new_epochs=30,new_updates=6660,L_identity_NN=5114880)},views={v:dict(count=10000,correct=correct if v=='clean' else 5000,accuracy=(correct if v=='clean' else 5000)/10000,macro_f1=.5,worst_RX_accuracy=.4) for v in m.VIEWS})
def summary():
 return dict(status='ANALYZED',rows={'N1':row(1,8500),'A1':row(1,8490)},paired_repairs=[])
class Checks(unittest.TestCase):
 def test_exact_absolute_boundary(self):
  r=m.assess(summary(),[('G','N1','A1')])['groups']['G'];self.assertTrue(r['clean_tolerance_mean_met']);self.assertAlmostEqual(r['paired']['clean']['mean_pp'],-.1);self.assertIsNone(r['paired']['clean']['sample_SD_pp'])
 def test_just_over_boundary(self):
  d=summary();d['rows']['A1']=row(1,8489);self.assertFalse(m.assess(d,[('G','N1','A1')])['groups']['G']['clean_tolerance_mean_met'])
 def test_not_relative_percentage(self):
  d=summary();d['rows']['N1']=row(1,5000);d['rows']['A1']=row(1,4990);self.assertTrue(m.assess(d,[('G','N1','A1')])['groups']['G']['clean_tolerance_mean_met'])
 def test_seed_and_budget_mismatch(self):
  for key,value in [('seed',2),('new_updates',6659),('L_identity_NN',5114879)]:
   d=summary()
   if key=='seed':d['rows']['A1']['model_seed']=value
   else:d['rows']['A1']['source']['cost'][key]=value
   with self.assertRaises(ValueError):m.assess(d,[('G','N1','A1')])
 def test_no_duplicate_or_missing_seed(self):
  with self.assertRaises(ValueError):m.assess(summary(),[('G','N1','A1'),('G','N1','A1')])
  with self.assertRaises(ValueError):m.assess(summary(),[('G','N1','MISSING')])
 def test_all_six_views_and_saved_score_consistency(self):
  for fault in ('missing','incorrect'):
   d=summary()
   if fault=='missing':del d['rows']['A1']['views']['practical_mid']
   else:d['rows']['A1']['views']['clean']['accuracy']=.9
   with self.assertRaises(ValueError):m.assess(d,[('G','N1','A1')])
 def test_all_seeds_reported_separately_from_mean(self):
  d=summary();d['rows'].update(N2=row(2,8500),A2=row(2,8480),N3=row(3,8500),A3=row(3,8520))
  r=m.assess(d,[('G','N1','A1'),('G','N2','A2'),('G','N3','A3')])['groups']['G']
  self.assertEqual(r['model_seed_count'],3);self.assertTrue(r['clean_tolerance_mean_met']);self.assertFalse(r['clean_tolerance_every_seed_met']);self.assertEqual(len(r['per_seed']),3);self.assertFalse(r['all_observed_clean_gains_above_0_1pp'])
 def test_no_cross_group_seed_inflation(self):
  d=summary();d['rows'].update(N2=row(1,8500),A2=row(1,8510));r=m.assess(d,[('S','N1','A1'),('C','N2','A2')]);self.assertEqual(r['unique_model_seed_count'],1);self.assertEqual(len(r['groups']),2)
 def test_unfinished_summary_rejected(self):
  d=summary();d['status']='WAIT_SOURCE'
  with self.assertRaises(ValueError):m.assess(d,[('G','N1','A1')])
if __name__=='__main__':unittest.main()
