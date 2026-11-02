import unittest
from unittest.mock import patch
from experiments.phase1_wavelet_lifting import train_source,ground
from experiments.phase1_dynamic_volterra import train_source as base

class AdapterTest(unittest.TestCase):
    def test_training_adapter_restores_after_error(self):
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_loss')
        before=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,train_source.WaveletLifting)
            self.assertIs(base.source_loss,train_source.source_loss)
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),before)

    def test_ground_uses_the_same_model(self):
        self.assertIs(ground.WaveletLifting,train_source.WaveletLifting)
        self.assertEqual(ground.SOURCE,'20261103-phase1-wavelet-lifting-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
