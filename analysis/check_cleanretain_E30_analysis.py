"""Focused independent recount and real objective/slot-log failure checks."""
import gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
import cvs_cleanretain_E30_results_20261017 as a
class Cases(unittest.TestCase):
 def test_confusion_independent_known_counts(self):
  r=a.base.metric([0,1,1,2],[0,0,1,2]);self.assertEqual(r['correct'],3);self.assertEqual(r['count'],4);self.assertAlmostEqual(r['accuracy'],.75);self.assertEqual(r['confusion'][1][0],1);self.assertAlmostEqual(r['macro_f1'],7/18)
 def test_full_actual_objective_logs_and_corrupt_weight(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'source';p.mkdir();q=p/'SupCon_steps.jsonl.gz';parts=dict(clean_CE=.2,second_CE=.6,clean_KL=.1,second_KL=.3,satellite_CE=.8,secondary_ordinary_CE=.4,satellite_KL=.4,secondary_ordinary_KL=.2)
   def emit(corrupt=False):
    with gzip.open(q,'wt',encoding='utf8') as f:
     for i in range(1,6661):f.write(json.dumps(dict(step=i,satellite_L=64,ordinary_weight=.75,satellite_weight=.5 if corrupt and i==4000 else .25,classification_terms=parts))+'\n')
   emit();curve,error=a.parts_curve(Path(t));self.assertEqual(len(curve),30);self.assertLess(error,1e-10);self.assertAlmostEqual(curve[-1]['satellite_CE'],.8)
   emit(True)
   with self.assertRaises(ValueError):a.parts_curve(Path(t))
 def test_incomplete_training_components_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'source';p.mkdir()
   with gzip.open(p/'SupCon_steps.jsonl.gz','wt',encoding='utf8') as f:pass
   with self.assertRaises(ValueError):a.parts_curve(Path(t))
 def test_actual_physical_slot_misalignment_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'R25-s2026092701';(root/'source').mkdir(parents=True)
   with gzip.open(root/'source/training_physics.jsonl.gz','wt',encoding='utf8') as f:f.write(json.dumps(dict(step=1,records=[{}]*64,positions=list(range(64))))+'\n')
   with self.assertRaisesRegex(ValueError,'alternating64'):a.physics_summary(root,dict(config=dict(As=True)))
if __name__=='__main__':unittest.main()
