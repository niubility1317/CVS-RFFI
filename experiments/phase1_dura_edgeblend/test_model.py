import math
import unittest
import torch
from torch import nn
from experiments.phase1_dura_blendgate.model import AlignedBlend, align_to_anchor


class FakeExpert(nn.Module):
    def forward(self, received, initial, log_variance):
        return torch.roll(received, 1, -1)


class EdgeAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(490)
        self.wave = torch.randn(2, 2, 256)

    def test_missing_right_edge_uses_anchor(self):
        anchor = torch.roll(self.wave, -24, -1)
        aligned = align_to_anchor(self.wave, anchor, preserve_edges=True)
        torch.testing.assert_close(aligned[:, :, -24:], anchor[:, :, -24:], rtol=0, atol=0)
        torch.testing.assert_close(aligned[:, :, :-24], anchor[:, :, :-24], rtol=1e-5, atol=1e-5)

    def test_missing_left_edge_uses_anchor(self):
        anchor = torch.roll(self.wave, 24, -1)
        aligned = align_to_anchor(self.wave, anchor, preserve_edges=True)
        torch.testing.assert_close(aligned[:, :, :24], anchor[:, :, :24], rtol=0, atol=0)
        torch.testing.assert_close(aligned[:, :, 24:], anchor[:, :, 24:], rtol=1e-5, atol=1e-5)

    def test_default_and_explicit_reflection_agree(self):
        anchor = torch.roll(self.wave, -24, -1)
        a = align_to_anchor(self.wave, anchor)
        b = align_to_anchor(self.wave, anchor, preserve_edges=False)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_phase_equivariance_and_batch_independence(self):
        model = AlignedBlend(FakeExpert(), preserve_edges=True).eval()
        anchor = self.wave + 0.1 * torch.randn_like(self.wave)
        with torch.no_grad():
            model.gate_output.weight.normal_(0, 0.02)
            full = model(self.wave, anchor, torch.zeros(2))
            single = model(self.wave[:1], anchor[:1], torch.zeros(1))
            c, s = math.cos(0.7), math.sin(0.7)
            rotate = lambda v: torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]), 1)
            rotated = model(rotate(self.wave), rotate(anchor), torch.zeros(2))
        torch.testing.assert_close(full[:1], single, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(rotated, rotate(full), rtol=2e-5, atol=2e-5)

    def test_initial_gate_returns_anchor(self):
        model = AlignedBlend(FakeExpert(), preserve_edges=True)
        anchor = self.wave + 0.1 * torch.randn_like(self.wave)
        out = model(self.wave, anchor, torch.zeros(2))
        torch.testing.assert_close(out, anchor, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
