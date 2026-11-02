import unittest
import torch
from experiments.phase1_dura_spectral_token.model import SpectralTokenRecovery


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((x[:, 0] * c - x[:, 1] * s, x[:, 0] * s + x[:, 1] * c), dim=1)


class SpectralTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        torch.set_num_threads(2)
        self.net = SpectralTokenRecovery()
        self.x = torch.randn(3, 2, 256)
        self.anchor = torch.randn(3, 2, 256)
        self.logvar = torch.tensor([-3., -2., -1.])

    def test_initial_output_is_exact_anchor(self):
        torch.testing.assert_close(self.net(self.x, self.anchor, self.logvar), self.anchor, rtol=0, atol=0)

    def test_phase_equivariance_with_nonzero_head(self):
        torch.nn.init.normal_(self.net.output.weight, std=.04)
        torch.nn.init.normal_(self.net.output.bias, std=.02)
        angle = torch.tensor(.713)
        expected = rotate(self.net(self.x, self.anchor, self.logvar), angle)
        actual = self.net(rotate(self.x, angle), rotate(self.anchor, angle), self.logvar)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_global_amplitude_equivariance(self):
        torch.nn.init.normal_(self.net.output.weight, std=.04)
        torch.nn.init.normal_(self.net.output.bias, std=.02)
        scale = 1.9
        y = self.net(self.x, self.anchor, self.logvar)
        scaled = self.net(self.x * scale, self.anchor * scale, self.logvar + 2 * torch.log(torch.tensor(scale)))
        torch.testing.assert_close(scaled, y * scale, rtol=3e-5, atol=3e-5)

    def test_backpropagates_and_does_not_mix_records(self):
        torch.nn.init.normal_(self.net.output.weight, std=.04)
        y = self.net(self.x, self.anchor, self.logvar)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(self.net.input.weight.grad).all())
        self.assertGreater(self.net.input.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            a = self.net(self.x[:1], self.anchor[:1], self.logvar[:1])
            b = self.net(self.x, self.anchor, self.logvar)[:1]
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_and_nonfinite(self):
        with self.assertRaises(ValueError): self.net(self.x[:, :, :-1], self.anchor, self.logvar)
        x = self.x.clone(); x[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError): self.net(x, self.anchor, self.logvar)

if __name__ == '__main__': unittest.main()
