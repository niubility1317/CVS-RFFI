import unittest

import torch

from experiments.phase1_equivariant_dictionary.model import EquivariantPatchDictionary


class DictionaryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        self.model = EquivariantPatchDictionary(atoms=8, width=12)
        self.received = torch.randn(3, 2, 256)
        self.initial = torch.randn(3, 2, 256)
        self.logvar = torch.randn(3)

    def test_initial_output_is_exactly_the_input_estimate(self):
        out = self.model(self.received, self.initial, self.logvar)
        self.assertTrue(torch.equal(out, self.initial))
        (out-self.received).square().mean().backward()
        gradient = self.model.gate_output.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_common_phase_rotation_commutes_with_reconstruction(self):
        torch.nn.init.normal_(self.model.gate_output.weight, std=0.2)
        torch.nn.init.normal_(self.model.gate_output.bias, std=0.2)
        angle = torch.tensor(0.61)
        c, s = angle.cos(), angle.sin()

        def rotate(x):
            return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)

        actual = self.model(rotate(self.received), rotate(self.initial), self.logvar)
        expected = rotate(self.model(self.received, self.initial, self.logvar))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    def test_dictionary_receives_gradients_when_gate_is_open(self):
        with torch.no_grad():
            self.model.gate_output.bias.fill_(0.5)
        self.model(self.received, self.initial, self.logvar).square().mean().backward()
        for parameter in (self.model.atom_real, self.model.atom_imag):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_positive_gain_and_noise_rescaling_commute_with_reconstruction(self):
        torch.nn.init.normal_(self.model.gate_output.weight, std=0.2)
        torch.nn.init.normal_(self.model.gate_output.bias, std=0.2)
        scale = 2.5
        actual = self.model(scale*self.received, scale*self.initial,
                            self.logvar + 2*torch.tensor(scale).log())
        expected = scale*self.model(self.received, self.initial, self.logvar)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-6)

    def test_identical_constant_atoms_match_explicit_overlap_projection(self):
        with torch.no_grad():
            self.model.atom_real.fill_(1.)
            self.model.atom_imag.zero_()
            self.model.gate_output.bias.fill_(0.5)
        actual = self.model(self.received, self.initial, self.logvar)
        delta = torch.zeros_like(self.initial)
        counts = torch.zeros_like(self.initial)
        strength = 0.5*torch.tensor(0.5).tanh()
        for start in range(0, 256-32+1, 8):
            segment = self.initial[..., start:start+32]
            delta[..., start:start+32] += strength*(segment.mean(-1, keepdim=True)-segment)
            counts[..., start:start+32] += 1
        expected = self.initial+delta/counts
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_silent_signal_stays_finite(self):
        zero = torch.zeros_like(self.initial)
        output = self.model(zero, zero, self.logvar)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, zero))

    def test_rejects_incompatible_iq_shape(self):
        with self.assertRaises(ValueError):
            self.model(self.received, self.initial[..., :-1], self.logvar)


if __name__ == '__main__':
    unittest.main()
