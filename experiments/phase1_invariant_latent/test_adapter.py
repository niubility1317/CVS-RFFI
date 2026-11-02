import unittest
from unittest.mock import patch

from experiments.phase1_invariant_latent import train_source as adapter


class AdapterTest(unittest.TestCase):
    def test_paired_training_hooks_are_installed_and_restored(self):
        base = adapter.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss',
                 'auxiliary_source_objective', 'PAIRED_VIEWS', 'USES_DURA')
        old = tuple(getattr(base, name) for name in names)

        def inspect():
            self.assertTrue(base.PAIRED_VIEWS)
            self.assertFalse(base.USES_DURA)
            self.assertIs(base.source_loss, adapter.source_loss)
            self.assertIs(base.auxiliary_source_objective, adapter.paired_auxiliary)
            self.assertIs(base.DynamicVolterra, adapter.make_model)
            raise RuntimeError('test restoration')

        with patch.object(base, 'main', inspect):
            with self.assertRaisesRegex(RuntimeError, 'test restoration'):
                adapter.main()
        self.assertEqual(old, tuple(getattr(base, name) for name in names))


if __name__ == '__main__':
    unittest.main()
