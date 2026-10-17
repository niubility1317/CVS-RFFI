"""Synthetic source-boundary and paired attribution regressions; no IQ/CP/query."""
import unittest
import numpy as np
from experiments.cvs_practical_attribution33 import design as d
from experiments.cvs_practical_attribution33.stats import metric,effect,stable_reduction
class AttributionTests(unittest.TestCase):
 def test_boundary(self):
  q=dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),V_panel=1800,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0)
  o=dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=1800,draws=2,scenes=list(d.SCENES),conditions=list(d.CONDITIONS),physical_draws=10800,target_access=False,training_updates=0,formal_test_changed=False)
  d.validate_metadata(q,o)
  for key,value in [('role','U'),('held_RX',1),('U_truth_used',True),('target_access',True)]:
   z=dict(q);z[key]=value
   with self.assertRaises(ValueError):d.validate_metadata(z,o)
  z=dict(o);z['conditions']=list(reversed(d.CONDITIONS))
  with self.assertRaises(ValueError):d.validate_metadata(q,z)
 def test_metric_and_scale(self):
  y=np.array([0,1]);z=np.array([[30.,0,0,0,0,0],[30.,0,0,0,0,0]])
  a=metric(z,y,np.array([True,False]));self.assertEqual(a['accuracy'],.5);self.assertEqual(a['mean_margin_over30'],0.);self.assertEqual(a['reference_clean_retention'],1.)
  with self.assertRaises(ValueError):metric(z*np.nan,y,np.ones(2,bool))
 def test_effect_sign_and_pairing(self):
  y=np.array([0,0,0,0]);full=np.array([[2.,0],[0,2],[0,2],[0,2]]);removed=full.copy();removed[1]=[2,0];removed[2]=[2,0]
  a=effect(full,removed,y);self.assertEqual(a['removal_delta_pp'],50.);self.assertEqual(a['repairs'],2);self.assertEqual(a['breaks'],0)
  b=effect(removed,full,y);self.assertEqual(a['removal_delta_pp']-b['removal_delta_pp'],100.)
 def test_two_draw_criterion(self):
  self.assertTrue(stable_reduction([1.,1.]));self.assertFalse(stable_reduction([2.,-.1]));self.assertFalse(stable_reduction([2.]));self.assertFalse(stable_reduction([np.nan,2.]))
 def test_complete_collect_preserves_condition_pairing(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from experiments.cvs_practical_attribution33 import stats
  y=np.arange(6);clean=np.eye(6,dtype=np.float32)*30;inputs={a:dict(TX=y,RX=np.array([1,3,4,6,8,1]),clean=clean.copy()) for a in d.ARMS};z={a:np.tile(clean,(2,3,11,1,1)) for a in d.ARMS}
  z['B00'][:,:,0,:,0]=31 #full mostly wrong;all interventions perfect
  physics=dict(state=np.zeros((2,3,11,6),int),lock=np.zeros((2,3,11,6),int),signal_to_added_noise_db=np.full((2,3,11,6),20.))
  with tempfile.TemporaryDirectory() as q,patch.object(stats,'plot'):
   r=stats.collect(Path(q),inputs,z,physics);self.assertTrue((Path(q)/'summary.json').exists());self.assertTrue(all(v['at_least_1pp_both_draws'] for v in r['descriptive_criteria']))
   a=next(v for v in r['sensitivity_reduction'] if v['draw']==-1 and v['dimension']=='overall');self.assertAlmostEqual(a['sensitivity_reduction_pp'],100*5/6)
if __name__=='__main__':unittest.main()
