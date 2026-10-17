"""Synthetic kernel and source-only factorial tests; no real IQ/CP/query."""
import unittest
import numpy as np
from experiments.cvs_delay_phase_attribution34.kernel import flat_kernel
from experiments.cvs_delay_phase_attribution34.stats import contrast,criterion
from experiments.cvs_delay_phase_attribution34 import design as d
class Tests(unittest.TestCase):
 def test_kernel_energy_phase_and_position(self):
  h=np.array([1+2j,0,3-4j,0],complex)
  for delay in (0,2):
   for zero in (False,True):
    k=flat_kernel(h,delay,zero);self.assertAlmostEqual(float(np.vdot(k,k).real),30.);self.assertEqual(np.flatnonzero(k).tolist(),[delay]);self.assertAlmostEqual(float(np.angle(k[delay])),0. if zero else float(np.angle(h[2])))
  for delay in (-1,4):
   with self.assertRaises(ValueError):flat_kernel(h,delay,False)
 def test_source_boundaries(self):
  q=d.config();d.validate(q)
  for key,value in [('target_access',True),('training_updates',1),('held_RX',1)]:
   bad=dict(q);bad[key]=value
   with self.assertRaises(ValueError):d.validate(bad)
 def test_factorial_contrast_signs(self):
  q=contrast([.4,.8,.5,.9],[.6,.8,.6,.9]);self.assertAlmostEqual(q['B00_delay_gain_pp'],40.);self.assertAlmostEqual(q['As_advantage_reduction_after_delay_pp'],20.);self.assertAlmostEqual(q['B00_phase_gain_pp'],10.)
  self.assertAlmostEqual(q['B00_delay_phase_interaction_pp'],0.)
 def test_source_contribution_criterion(self):
  good=[dict(B00_delay_gain_pp=5.,As_advantage_reduction_after_delay_pp=5.,original_As_advantage_pp=10.)]*4
  self.assertTrue(criterion(good,'delay'));self.assertFalse(criterion(good[:3],'delay'))
  bad=[dict(v) for v in good];bad[2]['B00_delay_gain_pp']=4.99;self.assertFalse(criterion(bad,'delay'))
 def test_complete_collect_and_major_contribution(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from experiments.cvs_delay_phase_attribution34 import stats
  y=np.arange(6);clean=np.eye(6,dtype=np.float32)*30;inp={a:dict(TX=y,RX=np.array([1,3,4,6,8,1]),clean=clean.copy()) for a in d.ARMS};z={a:np.tile(clean,(2,2,4,1,1)) for a in d.ARMS}
  z['B00'][:,:,0,:,0]=31;z['B00'][:,:,2,:,0]=31
  with tempfile.TemporaryDirectory() as q,patch.object(stats,'plot'):
   r=stats.collect(Path(q),inp,z);self.assertTrue(r['source_decision']['delay_major_contribution']);self.assertFalse(r['source_decision']['phase_major_contribution']);self.assertTrue((Path(q)/'contrasts.csv').exists())
if __name__=='__main__':unittest.main()
