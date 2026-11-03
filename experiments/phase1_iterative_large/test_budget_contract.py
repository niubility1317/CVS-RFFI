import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.phase1_dynamic_volterra import ground,score_source
from experiments.phase1_dynamic_volterra import test_score_source as fixtures

class FitSizeContractTest(unittest.TestCase):
 def test_scoring_accepts_registered_fit_size_only(self):
  f=fixtures.DynamicVolterraScoreTest('test_fixed_predictions_bind_only_authorized_labeled_rows');f.setUp()
  try:
   manifest=f.source/'prediction_complete.json';data=json.loads(manifest.read_text(encoding='utf8'));data['fit_ids']=32768;manifest.write_text(json.dumps(data),encoding='utf8')
   torch.save(dict(schema='phase1_dynamic_volterra_v1',commit='fixed',epochs=5,fit_ids=np.arange(100000,132768)),f.source/'checkpoint.pt')
   with patch.object(score_source,'FIT_SIZE',32768):f.run_score()
   self.assertEqual(json.loads(f.output.read_text(encoding='utf8'))['U_s_truth_reads'],0)
  finally:f.tearDown()
 def test_score_rejects_mismatched_checkpoint_before_truth_access(self):
  f=fixtures.DynamicVolterraScoreTest('test_fixed_predictions_bind_only_authorized_labeled_rows');f.setUp()
  try:
   manifest=f.source/'prediction_complete.json';data=json.loads(manifest.read_text(encoding='utf8'));data['fit_ids']=32768;manifest.write_text(json.dumps(data),encoding='utf8')
   (f.root/'source_metadata/L_s_metadata.npz').unlink()
   with patch.object(score_source,'FIT_SIZE',32768):
    with self.assertRaisesRegex(ValueError,'Fixed source checkpoint'):f.run_score()
   self.assertFalse(f.output.exists())
  finally:f.tearDown()
 def test_ground_rejects_wrong_fit_size_before_loading_model(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'runs'/ground.SOURCE;folder.mkdir(parents=True)
   done=dict(status='SOURCE_PREDICTIONS_FIXED',commit='fixed',count=672,fit_ids=8192,F_frozen=True,blind_frozen=True,gate_frozen=True,DURA_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates_after_prediction=0)
   (folder/'prediction_complete.json').write_text(json.dumps(done),encoding='utf8')
   deps=SimpleNamespace(core=SimpleNamespace(read=lambda p:json.loads(p.read_text(encoding='utf8'))))
   with patch.object(ground,'FIT_SIZE',32768):
    with self.assertRaisesRegex(ValueError,'fixed source predictions'):ground.load_fixed(root,'fixed','cpu',deps)

if __name__=='__main__':unittest.main()
