import unittest
import torch

from experiments.phase1_spectral_margin.test_objective import Reader
from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss


class AlignmentObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(722); torch.set_num_threads(2)
        self.clean = torch.randn(3, 2, 256)

    def test_delay_and_complex_gain_are_invariant(self):
        z = torch.complex(self.clean[:, 0], self.clean[:, 1])
        z = torch.roll(z, 7, dims=-1)*(1.4+0.6j)
        wave = torch.stack((z.real, z.imag), 1)
        self.assertLess(float(aligned_iq_error(wave, self.clean).max()), 1e-10)

    def test_selected_alignment_produces_finite_waveform_gradient(self):
        wave = torch.randn_like(self.clean, requires_grad=True)
        error = aligned_iq_error(wave, self.clean).mean(); error.backward()
        self.assertTrue(torch.isfinite(wave.grad).all()); self.assertGreater(float(wave.grad.abs().sum()), 0.)

    def test_selected_gain_gradient_matches_finite_difference(self):
        target = torch.randn(1, 2, 256, dtype=torch.float64)
        wave = torch.randn(1, 2, 256, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(aligned_iq_error(wave, target).sum(), wave)[0]
        index = int(gradient.abs().reshape(-1).argmax()); epsilon = 1e-5
        plus = wave.detach().clone(); minus = wave.detach().clone()
        plus.reshape(-1)[index] += epsilon; minus.reshape(-1)[index] -= epsilon
        numeric = (aligned_iq_error(plus, target)-aligned_iq_error(minus, target))/(2*epsilon)
        self.assertAlmostEqual(float(gradient.reshape(-1)[index]), float(numeric), delta=2e-7)

    def test_combined_objective_matches_clean_and_penalizes_amplitude_loss(self):
        reader = Reader().eval().requires_grad_(False); features = reader.features(self.clean).detach(); groups = torch.tensor([4, 4, 8])
        loss, terms = source_loss(reader, self.clean, self.clean, features, groups)
        self.assertLess(abs(float(loss)), 1e-6)
        diminished = (self.clean*.1).requires_grad_(True)
        loss, terms = source_loss(reader, diminished, self.clean, features, groups)
        self.assertGreater(float(terms['power'].detach().min()), 1.)
        loss.backward(); self.assertTrue(torch.isfinite(diminished.grad).all())
        self.assertTrue(all(p.grad is None for p in reader.parameters()))


if __name__ == '__main__': unittest.main()
