import unittest

import torch

from experiments.phase1_hankel_posterior.hankel import hankel_shrink


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)


class HankelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_zero_noise_is_identity_and_silence_is_finite(self):
        x = torch.randn(3, 2, 256)
        self.assertTrue(torch.equal(hankel_shrink(x, torch.zeros(3)), x))
        out = hankel_shrink(torch.zeros_like(x), torch.ones(3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out, torch.zeros_like(x)))

    def test_common_phase_rotation_and_batch_independence(self):
        x = torch.randn(3, 2, 256)
        var = torch.tensor([.01, .02, .03])
        angle = torch.tensor(.57)
        expected = rotate(hankel_shrink(x, var), angle)
        actual = hankel_shrink(rotate(x, angle), var)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        pieces = torch.cat([hankel_shrink(x[i:i+1], var[i:i+1]) for i in range(3)])
        torch.testing.assert_close(hankel_shrink(x, var), pieces, rtol=3e-5, atol=3e-5)

    def test_rejects_invalid_noise_estimate(self):
        x = torch.randn(2, 2, 256)
        with self.assertRaises(ValueError):
            hankel_shrink(x, torch.tensor([.1, -.1]))

if __name__ == '__main__':
    unittest.main()
