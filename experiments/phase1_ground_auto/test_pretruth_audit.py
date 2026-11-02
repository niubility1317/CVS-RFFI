import importlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np

class PretruthAuditTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('pretruth_audit'));self.check=importlib.import_module('pretruth_audit').check;self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup);self.c=self.root/'candidate';self.b=self.root/'baseline';self.c.mkdir();self.b.mkdir();self.views=['clean','low'];self.ids=np.array([101,102,103,104]);self.meta={'status':'PREDICTIONS_COMPLETE','count':4,'views':self.views,'F_run':'tri','F_row':'frozen','TX_truth_reads':0,'model_updates':0}
  for d in (self.c,self.b):
   np.save(d/'physical_ids.npy',self.ids);(d/'complete.json').write_text(json.dumps(dict(self.meta,commit='candidate' if d==self.c else 'baseline')))
   for view in self.views:np.save(d/(view+'.raw.npy'),np.array([0,1,2,3],dtype=np.int64));np.save(d/(view+'.restored.npy'),np.array([1,1,2,4],dtype=np.int64))
 def test_complete_matching_predictions_pass(self):
  result=self.check(self.c,self.b,4,'candidate',self.views);self.assertEqual(result['status'],'PASS');self.assertEqual(result['count'],4);self.assertEqual(len(result['sha256']),10)
 def test_mismatched_reference_or_incomplete_result_fails(self):
  np.save(self.c/'low.raw.npy',np.array([0,1,2,4],dtype=np.int64))
  with self.assertRaises(ValueError):self.check(self.c,self.b,4,'candidate',self.views)
  np.save(self.c/'low.raw.npy',np.array([0,1,2,3],dtype=np.int64));meta=dict(self.meta,commit='candidate',TX_truth_reads=1);(self.c/'complete.json').write_text(json.dumps(meta))
  with self.assertRaises(ValueError):self.check(self.c,self.b,4,'candidate',self.views)
 def test_invalid_prediction_and_changed_identity_fail(self):
  np.save(self.c/'clean.restored.npy',np.array([1,1,2,6],dtype=np.int64))
  with self.assertRaises(ValueError):self.check(self.c,self.b,4,'candidate',self.views)
  np.save(self.c/'clean.restored.npy',np.array([1,1,2,4],dtype=np.int64));np.save(self.c/'physical_ids.npy',self.ids[::-1])
  with self.assertRaises(ValueError):self.check(self.c,self.b,4,'candidate',self.views)
if __name__=='__main__':unittest.main()
