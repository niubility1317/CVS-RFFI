import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.phase1_cumulant_context.model import cumulant_loss
from experiments.phase1_cumulant_rank import ground
from experiments.phase1_cumulant_rank.objective import raw_features
from experiments.phase1_cumulant_rank.train import source_reference

class RankTrainingTest(unittest.TestCase):
    def test_reference_retains_matching_cumulant(self):
        torch.manual_seed(2)
        clean=torch.randn(64,2,256)
        physical,stats,reference,cumulants=source_reference((clean,clean,clean,torch.ones(64),clean),True)
        self.assertEqual(physical,64)
        self.assertEqual(tuple(reference.shape),(64,53))
        output=clean[:5]+.1*torch.randn_like(clean[:5])
        raw=raw_features(output)[1]
        torch.testing.assert_close((raw-cumulants[:5]).square().mean(),cumulant_loss(output,clean[:5]))
    def test_bad_checkpoint_rejected_before_recognizer(self):
        with patch.object(ground.core,'read',return_value={'status':'SOURCE_TRAINED','schema':'other'}), \
             patch.object(ground.torch,'load',return_value={}), \
             patch.object(ground.core,'teacher',side_effect=AssertionError('recognizer accessed')):
            with self.assertRaises(ValueError):ground.load_fixed(Path('.'),'source','bad','cpu')

if __name__=='__main__':unittest.main()
