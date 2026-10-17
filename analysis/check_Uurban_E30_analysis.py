import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_Uurban_E30_results_20261018 as a
class Cases(unittest.TestCase):
 def test_macroF1_and_confusion_recount(self):
  m=a.base.metric([0,1,1,2],[0,0,1,2]);self.assertEqual(m['correct'],3);self.assertAlmostEqual(m['macro_f1'],7/18);self.assertEqual(m['confusion'][1][0],1)
 def test_source_gate_all_cells_and_teacher_required(self):
  clean=dict(B0=.98,URAW=.98,UAS=.98);urban={k:[[.8,.6,.5],[.8,.6,.5]] for k in a.ARMS};urban['UAS']=[[.8,.61,.51],[.8,.61,.51]];t={k:True for k in a.ARMS};self.assertTrue(a.independent_source_decision(clean,urban,t)['source_qualified'])
  urban['UAS'][1][2]=.505;self.assertFalse(a.independent_source_decision(clean,urban,t)['source_qualified']);urban['UAS'][1][2]=.51;t['UAS']=False;self.assertFalse(a.independent_source_decision(clean,urban,t)['source_qualified'])
 def test_missing_draw_or_nonfinite_source_fails(self):
  clean=dict(B0=.98,URAW=.98,UAS=.98);u={k:[[.8,.6,.5]] for k in a.ARMS};t={k:True for k in a.ARMS}
  with self.assertRaises(ValueError):a.independent_source_decision(clean,u,t)
  u={k:[[.8,.6,.5],[.8,.6,.5]] for k in a.ARMS};u['UAS'][0][1]=float('nan')
  with self.assertRaises(ValueError):a.independent_source_decision(clean,u,t)
 def test_formal_recount_stays_closed_on_incomplete_predictions(self):
  def read(p):
   if p.name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False)
   return dict(checks=dict(source_only=True))
  with tempfile.TemporaryDirectory() as tmp,patch.object(a,'read',side_effect=read),patch.object(a,'completed',side_effect=ValueError('Incomplete seven views')),patch.object(a.base,'collect_row') as collect:
   with self.assertRaisesRegex(ValueError,'Incomplete seven'):a.final_analysis(Path(tmp),Path(tmp)/'source',Path(tmp)/'truth.json')
   collect.assert_not_called()
 def test_receiver_thresholds_and_residual_scale(self):
  c=dict(quality_adaptation_enabled=True,unlocked_snr_db=5.,locked_snr_db=15.,quality_reference_snr_db=20.,max_residual_scale=10.)
  self.assertEqual(a.receiver_quality_policy(c,5.)[0],'degraded');self.assertEqual(a.receiver_quality_policy(c,15.)[0],'locked');self.assertEqual(a.receiver_quality_policy(c,4.999)[0],'unlocked')
  self.assertAlmostEqual(a.receiver_quality_policy(c,10.)[1],10**.5);self.assertEqual(a.receiver_quality_policy(c,-100.)[1],10.);self.assertEqual(a.receiver_quality_policy(c,100.)[1],1.)
  with self.assertRaises(ValueError):a.receiver_quality_policy(c,float('nan'))
 def test_added_noise_and_reference_proxy_are_distinct(self):
  c=dict(quality_adaptation_enabled=True,unlocked_snr_db=5.,locked_snr_db=15.,quality_reference_snr_db=20.,max_residual_scale=10.)
  r=dict(quality_snr_db=15.,receiver_lock_status='locked',residual_scale=10**.25,measured_added_noise_power_pre_agc=.01,added_noise_variance_reference=.001,signal_to_added_noise_db=14.8)
  d=a.quality_sensitivity(r,c);self.assertEqual(d['actual_state'],'locked');self.assertEqual(d['signal_added_noise_state'],'degraded');self.assertEqual(d['signal_reference_noise_state'],'locked');self.assertAlmostEqual(d['signal_reference_noise_ratio_db'],24.8);self.assertAlmostEqual(d['signal_reference_noise_minus_white_proxy_db'],9.8)
  r['receiver_lock_status']='degraded'
  with self.assertRaisesRegex(ValueError,'Recorded receiver state'):a.quality_sensitivity(r,c)
if __name__=='__main__':unittest.main()
