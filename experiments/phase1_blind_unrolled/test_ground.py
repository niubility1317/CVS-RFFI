import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.phase1_blind_unrolled import ground
from experiments.phase1_blind_unrolled.train import loss_terms
import torch

class BlindContractTest(unittest.TestCase):
    def test_invalid_checkpoint_rejected_before_F(self):
        with patch.object(ground.core,'read',return_value={'status':'SOURCE_TRAINED','schema':'other'}), \
             patch.object(ground.torch,'load',return_value={}), \
             patch.object(ground.core,'teacher',side_effect=AssertionError('F accessed')):
            with self.assertRaises(ValueError):ground.load_fixed(Path('.'),'source','wrong','cpu')
    def test_training_loss_uses_paired_iq_and_reprojection(self):
        y=torch.randn(2,2,256)
        target=y+.2*torch.randn_like(y)
        iq,physics=loss_terms(y,y,y,target)
        self.assertGreater(iq.item(),0)
        self.assertEqual(physics.item(),0)

if __name__=='__main__':unittest.main()
