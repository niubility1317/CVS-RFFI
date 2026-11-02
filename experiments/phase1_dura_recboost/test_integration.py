import unittest
from unittest.mock import patch
from experiments.phase1_dura_recboost import train_source,ground
from experiments.phase1_dura_recboost.model import WideContextRecovery

class RecognitionAdapterTest(unittest.TestCase):
    def test_train_installs_pseudo_guidance_and_restores(self):
        base=train_source.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA',
               'load_components','input_noise','predict_batch','extra_frozen_components')
        before=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,WideContextRecovery)
            self.assertIs(base.source_objective,train_source.recognition_source_objective)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),before)
    def test_ground_uses_same_wide_model(self):
        self.assertIs(ground.DynamicVolterra,WideContextRecovery)
        self.assertEqual(ground.SOURCE,'20261103-phase1-dura-recboost-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
