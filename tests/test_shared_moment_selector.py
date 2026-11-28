import unittest

import torch

from experiments.pamf_lora_main.shared_moment_selector_run import moment_features


class SharedMomentSelectorTest(unittest.TestCase):
    def test_features_are_finite_and_invariant_to_complex_gain(self):
        torch.manual_seed(19)
        x = torch.randn(4, 256, dtype=torch.complex64)
        gain = torch.tensor([1 + 2j, -.7 + .2j, 3 - 1j, .3j])[:, None]
        torch.testing.assert_close(moment_features(x), moment_features(x * gain), rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isfinite(moment_features(torch.zeros_like(x))).all())

    def test_features_detect_temporal_distortion(self):
        torch.manual_seed(29)
        x = torch.randn(4, 256, dtype=torch.complex64)
        filtered = x + .8 * x.roll(1, dims=-1)
        delta = (moment_features(x) - moment_features(filtered)).abs().sum(-1)
        self.assertTrue((delta > .05).all())

    def test_rejects_non_complex_or_short_inputs(self):
        with self.assertRaises(ValueError):
            moment_features(torch.ones(4, 256))
        with self.assertRaises(ValueError):
            moment_features(torch.ones(4, 8, dtype=torch.complex64))


if __name__ == '__main__':
    unittest.main()
