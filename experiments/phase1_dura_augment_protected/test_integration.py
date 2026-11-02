import unittest
from unittest.mock import patch
from experiments.phase1_dura_augment_protected import train_source


class ProtectedAugmentationContract(unittest.TestCase):
    def test_installs_protected_loss_and_restores_all_hooks_on_failure(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss', 'source_objective', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components', 'prepare_training_input')
        previous = tuple(getattr(base, key) for key in names)
        def fail():
            self.assertIs(base.source_objective, train_source.protected_source_objective)
            self.assertIs(base.source_loss, train_source.source_loss)
            self.assertIsInstance(base.prepare_training_input, train_source.SourceChannelAugment)
            self.assertEqual(base.SCHEMA, 'phase1_dura_augment_protected_v1')
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('test controlled failure')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'test controlled failure'):
                train_source.main()
        self.assertEqual(tuple(getattr(base, key) for key in names), previous)

if __name__ == '__main__': unittest.main()
