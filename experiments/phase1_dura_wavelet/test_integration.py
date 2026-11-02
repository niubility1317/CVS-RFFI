import unittest
from unittest.mock import patch
from experiments.phase1_dura_wavelet import train_source,ground
from experiments.phase1_wavelet_lifting.model import WaveletLifting

class DuraWaveletAdapterTest(unittest.TestCase):
    def test_source_uses_dura_protection_and_restores_adapters(self):
        base=train_source.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA',
               'load_components','input_noise','predict_batch','extra_frozen_components')
        before=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,WaveletLifting)
            self.assertIs(base.source_objective,train_source.protected_source_objective)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),before)

    def test_ground_uses_exact_same_architecture(self):
        self.assertIs(ground.DynamicVolterra,WaveletLifting)
        self.assertEqual(ground.SOURCE,'20261103-phase1-dura-wavelet-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
