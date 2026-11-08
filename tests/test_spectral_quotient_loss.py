import unittest
import numpy as np
import torch

from experiments.phase1_dura_orthogonal.spectral_quotient_loss import spectral_quotient_loss
from experiments.phase1_dura_orthogonal.spectral_quotient_probe import spectral_features


class SpectralQuotientLossTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(41)
        self.clean = torch.randn(6, 2, 256, generator=generator, dtype=torch.float64)
        self.received = self.clean + 0.4 * torch.randn(6, 2, 256, generator=generator, dtype=torch.float64)

    def test_matches_source_diagnostic_similarity(self):
        a = spectral_features(self.clean.numpy(), 'quotient')
        b = spectral_features(self.received.numpy(), 'quotient')
        reference = np.mean(1 - np.abs(np.sum(b * a.conj(), axis=1)) ** 2)
        self.assertAlmostEqual(spectral_quotient_loss(self.received, self.clean).item(), reference, places=10)

    def test_invariant_to_complex_gain(self):
        z = torch.complex(self.clean[:, 0], self.clean[:, 1]) * (0.6 + 0.7j)
        scaled = torch.stack((z.real, z.imag), dim=1)
        self.assertLess(spectral_quotient_loss(scaled, self.clean).item(), 1e-10)

    def test_differentiable_and_finite_including_zero(self):
        out = self.received.clone().requires_grad_()
        loss = spectral_quotient_loss(out, self.clean)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(out.grad).all())
        self.assertGreater(out.grad.abs().sum().item(), 0)
        zero = torch.zeros_like(self.clean, requires_grad=True)
        zero_loss = spectral_quotient_loss(zero, self.clean)
        zero_loss.backward()
        self.assertTrue(torch.isfinite(zero_loss))
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            spectral_quotient_loss(self.received[:, :, :-1], self.clean)


if __name__ == '__main__':
    unittest.main()
