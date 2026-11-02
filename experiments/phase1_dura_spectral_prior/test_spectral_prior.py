import unittest

import torch

from experiments.phase1_dura_spectral_prior.spectral_prior import SpectralPosteriorCorrection


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)


class SpectralPriorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        torch.set_num_threads(2)
        self.inputs = [torch.randn(3, 2, 256) for _ in range(4)]
        self.logvar = torch.randn(3)

    def test_zero_initialization_is_exact_identity(self):
        model = SpectralPosteriorCorrection()
        result = model(*self.inputs, self.logvar)
        self.assertTrue(torch.equal(result, self.inputs[2]))

    def test_nonzero_model_is_phase_equivariant(self):
        model = SpectralPosteriorCorrection()
        torch.nn.init.normal_(model.coefficients.weight, std=.1)
        torch.nn.init.normal_(model.coefficients.bias, std=.1)
        angle = torch.tensor(.71)
        expected = rotate(model(*self.inputs, self.logvar), angle)
        actual = model(*(rotate(v, angle) for v in self.inputs), self.logvar)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_frequency_branch_has_nonlocal_temporal_gradient(self):
        model = SpectralPosteriorCorrection()
        torch.nn.init.normal_(model.coefficients.weight, std=.1)
        posterior = self.inputs[3].clone().requires_grad_(True)
        result = model(*self.inputs[:3], posterior, self.logvar)
        grad = torch.autograd.grad(result[0, 0, 0], posterior)[0][0, :, 1:]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 1e-5)

    def test_zero_head_receives_gradient(self):
        model = SpectralPosteriorCorrection()
        output = model(*self.inputs, self.logvar)
        (output-self.inputs[0]).square().mean().backward()
        grad = model.coefficients.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_rejects_wrong_shapes(self):
        model = SpectralPosteriorCorrection()
        with self.assertRaises(ValueError):
            model(*self.inputs, torch.zeros(2))

if __name__ == '__main__':
    unittest.main()
