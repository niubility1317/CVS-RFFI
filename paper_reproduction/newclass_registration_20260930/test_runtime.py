import unittest
import numpy as np
import torch
from runtime import expand_head, channel_spectrogram, validate_split

class RuntimeTests(unittest.TestCase):
    def test_expansion_preserves_old_weights_and_teacher(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.fc = torch.nn.Linear(4, 3)
        model = Net(); weights = model.fc.weight.detach().clone()
        model, teacher = expand_head(model, 2)
        self.assertEqual(model.fc.out_features, 5)
        self.assertIsNone(model.fc.bias)
        self.assertTrue(torch.equal(model.fc.weight[:3], weights))
        self.assertEqual(teacher.fc.out_features, 3)
        self.assertNotEqual(model.fc.weight.data_ptr(), teacher.fc.weight.data_ptr())

    def test_short_iq_cannot_use_original_stft(self):
        x = np.random.default_rng(42).normal(size=(2, 256, 2))
        with self.assertRaises(ValueError): channel_spectrogram(x, 256, 128)
        self.assertEqual(channel_spectrogram(x, 64, 32).shape, (2, 26, 6, 1))

    def test_reject_query_training_overlap(self):
        with self.assertRaises(ValueError):
            validate_split(np.array(['a','b']), np.array(['b','c']))

if __name__ == '__main__': unittest.main()
