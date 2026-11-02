import unittest
from unittest.mock import patch
from experiments.phase1_dura_augment import train_source as augmented
from experiments.phase1_dura_augment_control import train_source as control


class MatchedControlTest(unittest.TestCase):
    def test_shared_model_loss_and_pipeline(self):
        self.assertIs(control.ContextRecovery, augmented.ContextRecovery)
        self.assertIs(control.source_loss, augmented.source_loss)
        self.assertIs(control.load_components, augmented.load_components)
        self.assertIs(control.input_noise, augmented.input_noise)
        self.assertIs(control.predict_batch, augmented.predict_batch)

    def test_control_preserves_training_input_and_restores_hooks(self):
        base = control.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        original = tuple(getattr(base, name) for name in names)
        original_preparation = base.prepare_training_input

        def inspect():
            self.assertIs(base.DynamicVolterra, control.ContextRecovery)
            self.assertIs(base.source_loss, control.source_loss)
            self.assertIs(base.prepare_training_input, original_preparation)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('restore')

        with patch.object(base, 'main', inspect):
            with self.assertRaisesRegex(RuntimeError, 'restore'):
                control.main()
        self.assertEqual(original, tuple(getattr(base, name) for name in names))


if __name__ == '__main__':
    unittest.main()
