import unittest
from unittest.mock import patch
from experiments.phase1_dura_viewcvar import train_source,ground
from experiments.phase1_dura_context.model import ContextRecovery

class ViewAdapterTest(unittest.TestCase):
    def test_installs_paired_source_objective_and_restores(self):
        base=train_source.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','PAIRED_VIEWS','USES_DURA',
               'load_components','input_noise','predict_batch','extra_frozen_components')
        old=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,ContextRecovery)
            self.assertIs(base.source_objective,train_source.viewcvar_source_objective)
            self.assertTrue(base.PAIRED_VIEWS and base.USES_DURA)
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),old)

    def test_ground_uses_one_view_and_same_model(self):
        self.assertIs(ground.DynamicVolterra,ContextRecovery)
        self.assertEqual(ground.SOURCE,'20261103-phase1-dura-viewcvar-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
