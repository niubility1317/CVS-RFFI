"""Narrow checks: independent metrics and prediction-before-truth analysis."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
SOURCE=Path(__file__).with_name('cvs_fullsource_practical_results_20261017.py')
spec=importlib.util.spec_from_file_location('results24',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AnalysisTests(unittest.TestCase):
 def test_hand_computed_confusion(self):
  r=m.metric([0,0,1,1,2,3,4,5],[0,1,1,1,2,3,4,5])
  self.assertEqual(r['correct'],7);self.assertEqual(r['count'],8)
  self.assertEqual(r['confusion'][0][:2],[1,1])
  self.assertAlmostEqual(r['accuracy'],7/8)
  self.assertAlmostEqual(r['macro_f1'],(2/3+4/5+4)/6)
 def test_invalid_decisions(self):
  for y,p in [([0],[6]),([0],[-1]),([0,1],[1]),([],[])]:
   with self.subTest(y=y,p=p),self.assertRaises(ValueError):m.metric(y,p)
 def test_incomplete_or_unfrozen_blocks_analysis(self):
  good=[dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(m.VIEWS),truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0),
        dict(status='SCORED_COMPLETE',views=list(m.VIEWS),truth_last=True),
        dict(status='SOURCE_TRAINED',epoch=20,optimizer_steps=4440,target_access=False,head_fit_updates=0)]
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   def store(rows):
    for name,q in zip(['prediction/complete.json','scoring/complete.json','source/completion.json'],rows):
     p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(q),encoding='utf8')
   store(good);self.assertEqual(m.completed(root)[0]['count'],168000)
   for part,key,value in [(0,'views',list(m.VIEWS[:-1])),(0,'truth_read',True),(0,'all_parameters_frozen',False),(0,'head_fit_updates',1),(1,'truth_last',False),(2,'optimizer_steps',4439)]:
    rows=[dict(q) for q in good];rows[part][key]=value;store(rows)
    with self.subTest(key=key),self.assertRaises(ValueError):m.completed(root)
   rows=[dict(q) for q in good];del rows[0]['truth_read'];store(rows)
   with self.assertRaises(ValueError):m.completed(root)
if __name__=='__main__':unittest.main()
