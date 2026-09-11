import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import json
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from core90_game_target_test import check_checkpoint, check_target, TargetInputs
from dataset_wisig import WiSigIndex

class FakeBase:
    index = [WiSigIndex(0,2,0,0,0)]
    def __len__(self): return 1
    def __getitem__(self,i):
        return torch.ones(2,256),0,0,{'tx_i':0,'truth':0}

class TargetTests(unittest.TestCase):
    def test_truth_is_removed(self):
        x, meta = TargetInputs(FakeBase())[0]
        self.assertEqual(set(meta),{'sample_id','rx_i','day_i'})
        self.assertEqual(tuple(x.shape),(2,256))
    def test_receiver_overlap_rejected(self):
        info = dict(source_rxs=[2],role_ids={'L_s':[]})
        with self.assertRaises(ValueError): check_target(FakeBase(),info,[2],[0])
    def test_physical_overlap_rejected(self):
        from cvsrffi.game_tracking.data import opaque_id
        info = dict(source_rxs=[1],role_ids={'L_s':[opaque_id(FakeBase.index[0])]})
        with self.assertRaises(ValueError): check_target(FakeBase(),info,[2],[0])
    def test_checkpoint_rejects_contact_and_nonfinal(self):
        saved = dict(epoch=200,initialization='scratch_only',checkpoint_selection='final_only',target_contact=False,
            args=dict(seed=392005,game_synthetic=False),source_info=dict(checkpoint_init='scratch_only',target_access_before_freeze=False))
        check_checkpoint(saved)
        for key,value in [('epoch',199),('target_contact',True),('initialization','inherited')]:
            with self.assertRaises(ValueError): check_checkpoint(dict(saved,**{key:value}))

    def test_explicit_multiseed_must_match_frozen_seed(self):
        saved = dict(epoch=200,initialization='scratch_only',checkpoint_selection='final_only',target_contact=False,
            args=dict(seed=392006,game_synthetic=False),source_info=dict(checkpoint_init='scratch_only',target_access_before_freeze=False))
        check_checkpoint(saved,expected_seed=392006)
        with self.assertRaises(ValueError):check_checkpoint(saved,expected_seed=392005)

    def test_manifest_duplicate_models_rejected_before_output(self):
        from core90_game_target_test import main
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);entry=dict(run_id='V2_A_seed392006',seed=392006,checkpoint='unused')
            (p/'input.json').write_text(json.dumps(dict(scope='frozen_completed_v2_target_recheck',selection_permitted=False,models=[entry,entry])))
            with self.assertRaisesRegex(ValueError,'Duplicate'):main(['--input-manifest',str(p/'input.json'),'--output-dir',str(p/'output')])
            self.assertFalse((p/'output').exists())

    def test_manifest_incomplete_source_rejected_before_checkpoint_load(self):
        from core90_game_target_test import main
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);run=p/'V2_A_seed392006';run.mkdir();ckpt=run/'final_ssdg.pth';ckpt.write_bytes(b'not a checkpoint')
            (run/'completion.json').write_text(json.dumps(dict(status='SOURCE_ARTIFACTS_COMPLETE',epoch=199,target_evaluated=False)))
            entry=dict(run_id=run.name,seed=392006,checkpoint=str(ckpt))
            (p/'input.json').write_text(json.dumps(dict(scope='frozen_completed_v2_target_recheck',selection_permitted=False,models=[entry])))
            with self.assertRaisesRegex(ValueError,'Incomplete source row'):main(['--input-manifest',str(p/'input.json'),'--output-dir',str(p/'output')])

if __name__ == '__main__': unittest.main()
