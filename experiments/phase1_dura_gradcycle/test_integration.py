import unittest
from unittest.mock import patch
from experiments.phase1_dura_gradcycle import train_source


class TrainingAdapterTest(unittest.TestCase):
    def test_installs_gradient_balance_and_restores_on_failure(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective',
                 'auxiliary_source_objective', 'combine_source_objectives', 'PAIRED_VIEWS', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        before = tuple(getattr(base, name) for name in names)
        def fail():
            self.assertIs(base.auxiliary_source_objective, train_source.crossview_objective)
            self.assertIs(base.combine_source_objectives, train_source.balanced_objective)
            self.assertTrue(base.USES_DURA and base.PAIRED_VIEWS)
            raise RuntimeError('synthetic training error')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic training error'):
                train_source.main()
        self.assertEqual(tuple(getattr(base, name) for name in names), before)


if __name__ == '__main__': unittest.main()
