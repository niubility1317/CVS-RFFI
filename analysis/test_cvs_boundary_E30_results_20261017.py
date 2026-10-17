"""Boundary metadata checks and all-row completion before truth analysis."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SOURCE=Path(__file__).with_name('cvs_boundary_E30_results_20261017.py')
spec=importlib.util.spec_from_file_location('boundary_E30',SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class BoundaryAnalysisTests(unittest.TestCase):
 def evidence(self):
  cells=[dict(TX=t,RX=r,scene=s,boundary=b,count=4736) for t in range(6) for r in (1,3,4,6,8) for s in ('practical_low_suburban','practical_mid_urban','practical_low_urban') for b in ('reflect','edge')]
  return dict(training_boundary_counts=dict(reflect=426240,edge=426240),boundary_coverage=cells),dict(records=[dict(epoch=e,steps=222*e,training_boundary_packets=dict(reflect=14208*e,edge=14208*e)) for e in range(1,31)])
 def test_full_training_support(self):
  r,p=self.evidence();q=a.boundary_evidence(r,p)
  self.assertEqual(q['coverage_cells'],180);self.assertEqual(q['counts']['edge'],426240)
 def test_incomplete_boundary_evidence_rejected(self):
  for issue in ('counts','missing_cell','zero_cell','cell_total','epochs','steps','schedule'):
   r,p=self.evidence()
   if issue=='counts':r['training_boundary_counts']['edge']-=128
   if issue=='missing_cell':r['boundary_coverage'].pop()
   if issue=='zero_cell':r['boundary_coverage'][0]['count']=0
   if issue=='cell_total':r['boundary_coverage'][0]['count']+=1
   if issue=='epochs':p['records'].pop()
   if issue=='steps':p['records'][1]['steps']-=1
   if issue=='schedule':p['records'][1]['training_boundary_packets']['edge']-=128
   with self.subTest(issue=issue),self.assertRaises(ValueError):a.boundary_evidence(r,p)
 def test_new_row_completion_before_any_truth_collection(self):
  a.install()
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'absent'
   argv=['readout','--row','all','--truth',str(Path(td)/'missing_truth.json'),'--output',str(out),'--commit','synthetic']
   with patch.object(a.m,'completed',side_effect=[None,None,ValueError('BCTX unfinished')]) as check,patch.object(a.m,'collect_row') as collect,patch('sys.argv',argv),self.assertRaises(ValueError):a.m.main()
   self.assertEqual(check.call_count,3);collect.assert_not_called();self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
