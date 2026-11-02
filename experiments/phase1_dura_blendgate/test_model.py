import math
import unittest
import torch
from torch import nn

from experiments.phase1_dura_blendgate.model import AlignedBlend


class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(0.85))

    def forward(self, received, initial, log_variance):
        return self.factor * received + 0.1 * torch.roll(received, 1, -1)


def rotate(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]), 1)


class AlignedBlendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(157)
        torch.set_num_threads(2)
        self.model = AlignedBlend(FakeExpert())
        self.received = torch.randn(3, 2, 256)
        self.anchor = self.received + 0.1 * torch.randn(3, 2, 256)
        self.log_variance = torch.randn(3)

    def test_initial_output_is_exact_anchor(self):
        torch.testing.assert_close(self.model(self.received, self.anchor, self.log_variance),
                                   self.anchor, rtol=0, atol=0)

    def test_expert_stays_frozen_and_in_evaluation_mode(self):
        self.model.train()
        self.assertFalse(self.model.expert.training)
        self.assertFalse(any(p.requires_grad for p in self.model.expert.parameters()))

    def test_phase_equivariance_with_nonzero_gate(self):
        self.model.eval()
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0, 0.02)
            a = self.model(self.received, self.anchor, self.log_variance)
            b = self.model(rotate(self.received, .83), rotate(self.anchor, .83), self.log_variance)
        torch.testing.assert_close(b, rotate(a, .83), rtol=2e-5, atol=2e-5)

    def test_single_sample_independent_of_batch(self):
        self.model.eval()
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0, 0.02)
            a = self.model(self.received, self.anchor, self.log_variance)
            b = self.model(self.received[:1], self.anchor[:1], self.log_variance[:1])
        torch.testing.assert_close(a[:1], b, rtol=2e-5, atol=2e-5)

    def test_gate_receives_gradient_and_expert_does_not(self):
        out = self.model(self.received, self.anchor, self.log_variance)
        out.square().mean().backward()
        self.assertGreater(float(self.model.gate_output.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.expert.factor.grad)


if __name__ == '__main__': unittest.main()
