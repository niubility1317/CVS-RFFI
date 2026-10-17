"""Synthetic arithmetic and scope tests;no real artifact reads."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import cvs_hard_LCE_E30_results_20261018 as a
class Checks(unittest.TestCase):
 def metrics(self):
  c=dict(B00=.8,H00=.803,B01=.8,H01=.803);u={k:[[.8,.6,.5],[.8,.6,.5]] for k in a.ARMS};u['H01']=[[.8,.61,.51],[.8,.61,.51]];r={k:{str(rx):.8 for rx in (1,3,4,6,8)} for k in a.ARMS};return c,u,r
 def test_source_pass_has_no_goalclaim(self):
  v=a.independent_source(*self.metrics());self.assertTrue(v['source_qualified']);self.assertFalse(v['goal_complete']);self.assertFalse(v['Ag_proved'])
 def test_each_scientific_failure_is_preserved(self):
  c,u,r=self.metrics();c['H00']=.8;self.assertFalse(a.independent_source(c,u,r)['checks']['hard_clean_independent'])
  c,u,r=self.metrics();u['H01'][1][2]=.5;self.assertFalse(a.independent_source(c,u,r)['checks']['both_urban_each_draw'])
  c,u,r=self.metrics();r['H01']['8']=.79;self.assertFalse(a.independent_source(c,u,r)['checks']['each_source_RX_preserved'])
 def test_source_incomplete_or_nonfinite(self):
  c,u,r=self.metrics();c.pop('B00')
  with self.assertRaises(ValueError):a.independent_source(c,u,r)
  c,u,r=self.metrics();u['H01'][1][2]=float('nan')
  with self.assertRaises(ValueError):a.independent_source(c,u,r)
 def test_run_scope_restored_on_exception(self):
  run,sat=a.old.RUN,a.old.satellite
  with self.assertRaises(RuntimeError):
   with a.run_scope(Path('/never/read/newrun/row')):
    self.assertEqual(a.old.RUN,'newrun');self.assertTrue(a.old.satellite('H01'));raise RuntimeError('Synthetic only')
  self.assertEqual(a.old.RUN,run);self.assertIs(a.old.satellite,sat)
 def test_factorial_units_and_interaction(self):
  values=dict(B00=.5,H00=.52,B01=.7,H01=.73);results={k:dict(views={v:dict(accuracy=x,macro_f1=x,worst_RX_accuracy=x) for v in a.base.VIEWS}) for k,x in values.items()};effects=a.effect_table(results);self.assertEqual(len(effects),21)
  self.assertAlmostEqual(effects[0]['hardL_without_As_pp'],2.);self.assertAlmostEqual(effects[0]['hardL_with_As_pp'],3.);self.assertAlmostEqual(effects[0]['interaction_pp'],1.);self.assertAlmostEqual(effects[0]['As_under_hardL_pp'],21.)
 def test_metric_and_paired_error_identity(self):
  y=[0,0,1,1,2,2];p=[0,1,1,1,2,0];m=a.base.metric(y,p);self.assertEqual(m['correct'],4);self.assertEqual(sum(map(sum,m['confusion'])),6);self.assertAlmostEqual(m['accuracy'],4/6)
  with self.assertRaises(ValueError):a.base.metric([0],[6])
 def test_formal_phase_reaches_collect_only_after_source_and_all_frozen_rows(self):
  from unittest.mock import patch
  decisions=dict(checks=dict(synthetic=True))
  def fake_read(p):
   p=Path(p)
   if p.name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False)
   if p.name in ('independent_source_decision.json','source_matrix_decision.json'):return decisions
   if p.name.endswith('_source.json'):return dict(synthetic=True)
   raise AssertionError('Unexpected artifact read: '+str(p))
  saved=dict(a.old.PARSED)
  try:
   with patch.object(a,'read',side_effect=fake_read),patch.object(a,'completed',return_value=None) as frozen,patch.object(a.base,'completed'),patch.object(a.base,'parse_source'),patch.object(a.base,'collect_row',side_effect=RuntimeError('SYNTHETIC_STOP_BEFORE_TRUTH')):
    with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_STOP_BEFORE_TRUTH'):a.final_analysis(Path('/never/output'),Path('/never/source'),Path('/never/truth'))
    self.assertEqual(frozen.call_count,4)
  finally:a.old.PARSED.clear();a.old.PARSED.update(saved)
if __name__=='__main__':unittest.main()
