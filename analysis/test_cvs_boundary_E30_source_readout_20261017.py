"""Hand-counted source metrics, strata boundaries and preregistered interpretation."""
import importlib.util,unittest
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).with_name('cvs_boundary_E30_source_readout_20261017.py')
spec=importlib.util.spec_from_file_location('source_readout30',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReadoutTests(unittest.TestCase):
 def test_hand_counted_metric(self):
  q=m.metric(np.asarray([0,0,1,1,2,3,4,5]),np.asarray([0,1,1,1,2,3,4,5]))
  self.assertEqual(q['count'],8);self.assertEqual(q['correct'],7);self.assertEqual(q['confusion'][0][:2],[1,1]);self.assertAlmostEqual(q['macro_f1'],(2/3+4/5+4)/6)
  for y,p in [([0],[6]),([],[]),([0,1],[1])]:
   with self.assertRaises(ValueError):m.metric(np.asarray(y),np.asarray(p))
 def rows(self,full=.75,edge=.69):
  return [dict(arm=a,scene=s,draw=d,condition=c,dimension='overall',stratum='ALL',accuracy=v) for s in m.SCENES[1:] for d in (0,1) for a,pair in [('B01',(.75,.65)),('BCTX',(full,edge))] for c,v in zip(m.CONDITIONS,pair)]
 def test_interpretation_requires_all_four_cells_and_full_preservation(self):
  q=m.hypothesis(self.rows());self.assertTrue(q['source_condition_supported']);self.assertEqual(len(q['cells']),4);self.assertAlmostEqual(q['cells'][0]['boundary_penalty_improvement_pp'],4)
  self.assertFalse(m.hypothesis(self.rows(full=.744,edge=.684))['source_condition_supported']);self.assertFalse(m.hypothesis(self.rows(edge=.67))['source_condition_supported'])
  with self.assertRaises(ValueError):m.hypothesis(self.rows()[:-1])
 def test_completion_requires_no_target_or_updates(self):
  good=dict(status='SOURCE_BOUNDARY_MECHANISM_COMPLETE',NN=70206,models=3,model_seed_count=1,channel_draws=2,unique_source_V=1800,training_updates=0,head_fit_updates=0,target_access=False,target_scores_or_truth_read=False,state_unchanged=True)
  m.require_done(good)
  for k,v in [('NN',70205),('models',2),('training_updates',1),('head_fit_updates',1),('target_access',True),('target_scores_or_truth_read',True),('state_unchanged',False)]:
   q=dict(good);q[k]=v
   with self.subTest(k=k),self.assertRaises(ValueError):m.require_done(q)
 def test_exact_quality_bin_edges(self):
  rx=np.asarray([1,3,4,6,8]);state=np.asarray([0,1,2,2,0]);lock=np.asarray([0,1,2,1,0]);quality=np.asarray([2.99,3.,7.99,8.,15.])
  q=dict(dimension='added_noise_ratio',stratum='3-8');self.assertEqual(m.strata_mask(q,rx,state,lock,quality).tolist(),[False,True,True,False,False])
  q=dict(dimension='state_lock',stratum='BLOCKED/unlocked');self.assertEqual(m.strata_mask(q,rx,state,lock,quality).tolist(),[False,False,True,False,False])
if __name__=='__main__':unittest.main()
