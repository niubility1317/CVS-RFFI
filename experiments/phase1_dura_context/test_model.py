import math
import unittest

import torch
from experiments.phase1_dura_context.model import ContextRecovery


def rotate(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]), 1)


class ContextRecoveryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(184)
        torch.set_num_threads(2)
        self.model = ContextRecovery().eval()
        self.received = torch.randn(3, 2, 256)
        self.anchor = torch.randn(3, 2, 256)
        self.log_variance = torch.randn(3)

    def test_zero_initialization_is_exact_anchor(self):
        torch.testing.assert_close(self.model(self.received, self.anchor, self.log_variance),
                                   self.anchor, rtol=0, atol=0)

    def test_phase_equivariance_after_nonzero_correction(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0, 0.02)
            self.model.gate_output.bias.normal_(0, 0.02)
            first = self.model(self.received, self.anchor, self.log_variance)
            actual = self.model(rotate(self.received, 0.73), rotate(self.anchor, 0.73), self.log_variance)
        torch.testing.assert_close(actual, rotate(first, 0.73), rtol=2e-5, atol=2e-5)

    def test_each_input_is_independent_of_batch_composition(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0, 0.02)
            full = self.model(self.received, self.anchor, self.log_variance)
            single = self.model(self.received[:1], self.anchor[:1], self.log_variance[:1])
        torch.testing.assert_close(full[:1], single, rtol=2e-5, atol=2e-5)

    def test_output_layer_receives_gradient(self):
        out = self.model(self.received, self.anchor, self.log_variance)
        out.square().mean().backward()
        self.assertGreater(float(self.model.gate_output.weight.grad.abs().sum()), 0.0)


if __name__ == '__main__': unittest.main()
