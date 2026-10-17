"""Narrow checks: independent metrics and prediction-before-truth analysis."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
SOURCE=Path(__file__).with_name('cvs_fullsource_E30_results_20261017.py')
spec=importlib.util.spec_from_file_location('results30',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
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
        dict(status='SOURCE_TRAINED',epoch=30,optimizer_steps=6660,target_access=False,head_fit_updates=0,config=dict(epochs=30,label_epochs=20,pseudo_epochs=10,updates=6660,held_RX=None,source_roles=dict(L_s=6300,U_s=56700,V=27000)))]
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   def store(rows):
    for name,q in zip(['prediction/complete.json','scoring/complete.json','source/completion.json'],rows):
     p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(q),encoding='utf8')
   store(good);self.assertEqual(m.completed(root)[0]['count'],168000)
   for part,key,value in [(0,'views',list(m.VIEWS[:-1])),(0,'truth_read',True),(0,'all_parameters_frozen',False),(0,'head_fit_updates',1),(1,'truth_last',False),(2,'optimizer_steps',6659),(2,'epoch',20),(2,'optimizer_steps',4440),(2,'config',dict(epochs=20,label_epochs=13,pseudo_epochs=7,updates=4440,held_RX=None,source_roles=dict(L_s=6300,U_s=56700,V=27000)))]:
    rows=[dict(q) for q in good];rows[part][key]=value;store(rows)
    with self.subTest(key=key),self.assertRaises(ValueError):m.completed(root)
   rows=[dict(q) for q in good];del rows[0]['truth_read'];store(rows)
   with self.assertRaises(ValueError):m.completed(root)

 def test_all_rows_guard_before_truth_or_output(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);rows=list(m.ROWS)
   for name in rows:
    run,row=m.ROWS[name];root=base/'runs'/run/row
    records={'prediction/complete.json':dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(m.VIEWS),truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0),'scoring/complete.json':dict(status='SCORED_COMPLETE',views=list(m.VIEWS if name==rows[0] else m.VIEWS[:-1]),truth_last=True),'source/completion.json':dict(status='SOURCE_TRAINED',epoch=30,optimizer_steps=6660,target_access=False,head_fit_updates=0,config=dict(epochs=30,label_epochs=20,pseudo_epochs=10,updates=6660,held_RX=None,source_roles=dict(L_s=6300,U_s=56700,V=27000)))}
    for filename,value in records.items():
     p=root/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf8')
   output=base/'must_not_exist';args=['readout','--row','all','--truth',str(base/'missing_truth.json'),'--output',str(output),'--commit','synthetic']
   with patch.object(m,'PROJECT',base),patch('sys.argv',args),self.assertRaises(ValueError):m.main()
   self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
