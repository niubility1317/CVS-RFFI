"""Protocol-negative analysis wiring: synthetic files only, no IQ/CP/NN."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import cvs_cycleorder_factorial_E30_results_20261017 as mod
class Stop(Exception):pass
class Cases(unittest.TestCase):
 def test_all_rows_complete_before_any_source_or_truth_analysis(self):
  events=[]
  def completed(root):
   events.append('complete')
   if len(events)==4:raise Stop()
  with tempfile.TemporaryDirectory() as td:
   target=Path(td)/'result'
   with patch.object(sys,'argv',['analysis','--truth',str(Path(td)/'DO_NOT_OPEN_TRUTH'),'--output',str(target),'--commit','test']),patch.object(mod.base,'completed',side_effect=completed),patch.object(mod,'source_readout') as source,patch.object(mod.base,'collect_row') as collect:
    with self.assertRaises(Stop):mod.main()
    source.assert_not_called();collect.assert_not_called();self.assertFalse(target.exists())
 def test_source_rule_precedes_first_truth_consumer(self):
  events=[];source_result={'sentinel':1}
  def source(out):events.append('source');return source_result
  def decide(out,sources):self.assertIs(sources,source_result);events.append('decide');return {'pass':False}
  def collect(*args):events.append('truth');raise Stop()
  with tempfile.TemporaryDirectory() as td:
   with patch.object(sys,'argv',['analysis','--truth',str(Path(td)/'DO_NOT_OPEN_TRUTH'),'--output',str(Path(td)/'result'),'--commit','test']),patch.object(mod.base,'completed',side_effect=lambda root:events.append('complete')),patch.object(mod,'source_readout',side_effect=source),patch.object(mod,'independent_source_rule',side_effect=decide),patch.object(mod.base,'collect_row',side_effect=collect):
    with self.assertRaises(Stop):mod.main()
  self.assertEqual(events,['complete']*4+['source','decide','truth'])
 def test_factorial_scale_and_rows(self):
  values={a:{v:x for v in mod.base.VIEWS} for a,x in [('B00',.5),('B10',.6),('B01',.7),('B11',.85)]}
  self.assertTrue(all(abs(x-5.)<1e-10 for x in mod.factorial(values).values()));self.assertTrue(all(run==mod.RUN for run,row in mod.base.ROWS.values()))
  with self.assertRaises(ValueError):mod.factorial({k:v for k,v in values.items() if k!='B00'})
if __name__=='__main__':unittest.main()
