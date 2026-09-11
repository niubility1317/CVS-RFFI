import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
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

if __name__ == '__main__': unittest.main()
