import unittest
from unittest.mock import patch
from experiments.phase1_dura_crossview_latent import train_source as adapter


class AdapterTest(unittest.TestCase):
    def test_paired_dura_hooks_are_installed_and_restored(self):
        base = adapter.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss',
                 'auxiliary_source_objective', 'PAIRED_VIEWS', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        old = tuple(getattr(base, name) for name in names)

        def inspect():
            self.assertTrue(base.PAIRED_VIEWS)
            self.assertTrue(base.USES_DURA)
            self.assertIs(base.source_loss, adapter.source_loss)
            self.assertIs(base.auxiliary_source_objective, adapter.paired_auxiliary)
            self.assertIs(base.DynamicVolterra, adapter.make_model)
            self.assertIs(base.predict_batch, adapter.predict_batch)
            raise RuntimeError('test restoration')

        with patch.object(base, 'main', inspect):
            with self.assertRaisesRegex(RuntimeError, 'test restoration'):
                adapter.main()
        self.assertEqual(old, tuple(getattr(base, name) for name in names))


if __name__ == '__main__':
    unittest.main()
