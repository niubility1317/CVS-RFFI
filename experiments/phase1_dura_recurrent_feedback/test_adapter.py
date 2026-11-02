import unittest
from unittest.mock import patch
from experiments.phase1_dura_recurrent_feedback import train_source as adapter
from experiments.phase1_dura_recurrent_feedback import ground


class AdapterTest(unittest.TestCase):
    def test_training_hooks_and_restoration(self):
        base = adapter.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        old = tuple(getattr(base, name) for name in names)

        def check():
            self.assertTrue(base.USES_DURA)
            self.assertIs(base.DynamicVolterra, adapter.RecurrentFeedbackRecovery)
            self.assertIs(base.predict_batch, adapter.predict_batch)
            raise RuntimeError('restore')

        with patch.object(base, 'main', check):
            with self.assertRaisesRegex(RuntimeError, 'restore'):
                adapter.main()
        self.assertEqual(old, tuple(getattr(base, name) for name in names))

    def test_ground_uses_same_model_and_pipeline(self):
        self.assertIs(ground.DynamicVolterra, adapter.RecurrentFeedbackRecovery)
        self.assertIs(ground.predict_batch, adapter.predict_batch)


if __name__ == '__main__':
    unittest.main()
