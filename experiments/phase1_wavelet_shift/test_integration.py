import unittest
from unittest.mock import patch
from experiments.phase1_wavelet_shift import train_source,ground
from experiments.phase1_wavelet_shift.model import ShiftAugmentedWavelet

class ShiftAdapterTest(unittest.TestCase):
    def test_train_adapter_restores_state(self):
        base=train_source.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_loss')
        before=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,ShiftAugmentedWavelet)
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),before)
    def test_fixed_ground_model_matches_training(self):
        self.assertIs(ground.ShiftAugmentedWavelet,ShiftAugmentedWavelet)
        self.assertEqual(ground.SOURCE,'20261103-phase1-wavelet-shift-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
