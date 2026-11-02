import unittest
from unittest.mock import patch
import torch
from experiments.phase1_dura_augment_protected.augmentation import SourceChannelAugment
from experiments.phase1_dura_augment_protected import train_source as adapter


class AugmentationTest(unittest.TestCase):
    def test_deterministic_partial_augmentation_preserves_input(self):
        torch.manual_seed(12)
        original = torch.randn(64, 2, 256)
        x = original.clone()
        one, two = SourceChannelAugment(), SourceChannelAugment()
        y, z = one(x), two(x)
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        changed = (y != x).flatten(1).any(1)
        self.assertEqual(int(changed.sum()), one.augmented)
        self.assertGreater(one.augmented, 0)
        self.assertLess(one.augmented, len(x))
        self.assertTrue(torch.isfinite(y).all())

    def test_invalid_source_data_rejected(self):
        augment = SourceChannelAugment()
        with self.assertRaises(ValueError):
            augment(torch.zeros(2, 2, 255))
        x = torch.zeros(2, 2, 256)
        x[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            augment(x)

    def test_training_hook_is_restored_on_error(self):
        base = adapter.base
        original = base.prepare_training_input
        with patch.object(base, 'main', side_effect=RuntimeError('stop')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                adapter.main()
        self.assertIs(base.prepare_training_input, original)


if __name__ == '__main__':
    unittest.main()
