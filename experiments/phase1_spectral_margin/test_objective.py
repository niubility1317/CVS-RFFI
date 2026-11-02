import unittest
import torch
from torch import nn

from experiments.phase1_spectral_margin.objective import source_loss


class Reader(nn.Module):
    def __init__(self):
        super().__init__(); self.head = nn.Linear(2, 6, bias=False)
        with torch.no_grad(): self.head.weight.copy_(torch.tensor([[2., 0.], [-2., 0.], [0., 1.], [0., -1.], [.5, .5], [-.5, -.5]]))
    def features(self, wave): return wave.mean(-1)
    def classify_features(self, features): return self.head(features)


class SourceObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12); self.reader = Reader().eval().requires_grad_(False)
        self.clean = torch.randn(3, 2, 256); self.features = self.reader.features(self.clean).detach()
        self.groups = torch.tensor([4, 4, 8])

    def test_equal_wave_has_zero_matching_penalties(self):
        loss, terms = source_loss(self.reader, self.clean, self.clean, self.features, self.groups)
        self.assertLess(abs(float(loss)), 1e-6)
        for key in ('wave', 'feature', 'distill', 'margin'):
            self.assertLess(float(terms[key].abs().max()), 1e-6)

    def test_incorrect_prediction_receives_gradient_without_updating_reader(self):
        changed = (-self.clean).requires_grad_(True)
        loss, terms = source_loss(self.reader, changed, self.clean, self.features, self.groups)
        self.assertGreater(float(terms['margin'].detach().sum()), 0.)
        self.assertGreater(float(terms['distill'].detach().sum()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(changed.grad).all()); self.assertGreater(float(changed.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.reader.parameters()))

    def test_trainable_reader_is_rejected(self):
        self.reader.requires_grad_(True)
        with self.assertRaises(ValueError): source_loss(self.reader, self.clean, self.clean, self.features, self.groups)


if __name__ == '__main__': unittest.main()
