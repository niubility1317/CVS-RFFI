import unittest

import torch

from experiments.phase1_dura_posterior.numerics import checked_logits


class Reader:
    def __init__(self, bad=False): self.bad = bad
    def features(self, x): return x
    def classify_features(self, x):
        logits = torch.zeros(len(x), 6)
        logits[:, 3] = float('nan') if self.bad else 1.
        return logits


class NumericTest(unittest.TestCase):
    def test_nonfinite_logits_cannot_be_converted_to_a_valid_class(self):
        with self.assertRaises(FloatingPointError):
            checked_logits(Reader(bad=True), torch.zeros(2, 2, 256))

    def test_nonfinite_waveform_is_rejected_before_reader(self):
        x = torch.zeros(2, 2, 256)
        x[0, 0, 5] = float('inf')
        with self.assertRaises(FloatingPointError):
            checked_logits(Reader(), x)

    def test_finite_reader_logits_are_preserved(self):
        logits = checked_logits(Reader(), torch.zeros(2, 2, 256))
        self.assertEqual(logits.argmax(1).tolist(), [3, 3])


if __name__ == '__main__':
    unittest.main()
