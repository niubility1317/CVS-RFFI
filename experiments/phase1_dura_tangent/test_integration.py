import unittest
from unittest.mock import patch

from experiments.phase1_dura_tangent import train_source


class TrainingAdapterTest(unittest.TestCase):
    def test_installs_protected_objective_and_restores_on_failure(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        before = tuple(getattr(base, name) for name in names)
        def fail():
            self.assertIs(base.source_objective, train_source.tangent_source_objective)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('synthetic training error')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic training error'):
                train_source.main()
        self.assertEqual(tuple(getattr(base, name) for name in names), before)


if __name__ == '__main__': unittest.main()
