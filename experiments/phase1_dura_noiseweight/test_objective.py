import math
import unittest
import torch

from experiments.phase1_dura_noiseweight.objective import noise_weights


class NoiseWeightTest(unittest.TestCase):
    def test_higher_estimated_noise_gets_more_weight(self):
        x = torch.ones(3, 2, 256)
        lv = torch.tensor([-9., -3., 2.])
        weights = noise_weights(x, lv)
        self.assertTrue(bool((weights[1:] > weights[:-1]).all()))
        self.assertTrue(bool(((weights >= 1) & (weights <= 8)).all()))

    def test_consistent_gain_does_not_change_relative_weight(self):
        torch.manual_seed(41)
        x = torch.randn(3, 2, 256)
        lv = torch.tensor([-7., -3., 1.])
        a = noise_weights(x, lv)
        b = noise_weights(x * 2, lv + math.log(4))
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_weight_is_detached_from_inputs(self):
        x = torch.ones(2, 2, 256, requires_grad=True)
        lv = torch.tensor([-6., -2.], requires_grad=True)
        self.assertFalse(noise_weights(x, lv).requires_grad)

    def test_invalid_estimate_is_rejected(self):
        with self.assertRaises(ValueError):
            noise_weights(torch.ones(2, 2, 256), torch.tensor([0.]))


if __name__ == '__main__': unittest.main()
