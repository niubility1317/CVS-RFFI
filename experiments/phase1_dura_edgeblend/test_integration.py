import unittest
from unittest.mock import patch
from torch import nn

from experiments.phase1_dura_edgeblend import train_source


class TrainingAdapterTest(unittest.TestCase):
    def test_installs_expert_freeze_and_restores_on_failure(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components',
                 'extra_frozen_model_components')
        before = tuple(getattr(base, name) for name in names)
        def fail():
            candidate = nn.Module()
            candidate.expert = nn.Linear(2, 2)
            self.assertEqual(base.extra_frozen_model_components(candidate), (candidate.expert,))
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('synthetic training error')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic training error'):
                train_source.main()
        self.assertEqual(tuple(getattr(base, name) for name in names), before)


if __name__ == '__main__': unittest.main()
