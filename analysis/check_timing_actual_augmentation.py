"""Synthetic actual-telemetry protocol/summary tests;no real data or neural calls."""
import copy,unittest
from cvs_timing_actual_augmentation_20261017 import extract,compare
class Cases(unittest.TestCase):
 def fixture(self):return dict(head_fit_updates=0,U_truth_read=False,target_access=False,records=[dict(epoch=e,steps=e*222,L_forward_packets=e*222*512,ordinary=dict(calls=e*222,packets=e*222*128,waveform_MSE=.1,max_abs=3.,probabilities=dict(p_pa=.01,p_cfo=.03)),parameter_movement={'cls_head.weight':float(e)}) for e in range(1,31)])
 def test_actual_schedule_retains_values_and_mismatches(self):
  q=self.fixture();a,m=extract(q);self.assertEqual(len(a),30);self.assertEqual(len(m),30);self.assertEqual(a[0]['p_cfo'],.03);b=copy.deepcopy(a);b[12]['p_cfo']=.04;b[4]['cumulative_max_abs']=4.;r=compare({'B00':a,'B10':b});self.assertEqual(r['B10']['probability_epoch_mismatches'],1);self.assertEqual(r['B10']['waveform_observation_epoch_mismatches'],1)
 def test_illegal_or_incomplete_source_records(self):
  for change in ('U_truth_read','target_access','head_fit_updates','short','nonfinite'):
   q=self.fixture()
   if change=='short':q['records'].pop()
   elif change=='nonfinite':q['records'][0]['ordinary']['probabilities']['p_pa']=float('nan')
   else:q[change]=True
   with self.assertRaises(ValueError):extract(q)
if __name__=='__main__':unittest.main()
