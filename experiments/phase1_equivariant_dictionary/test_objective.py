import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_equivariant_dictionary.objective import source_loss


class Reader(nn.Module):
    def features(self, x): return x.flatten(1)
    def classify_features(self, x): return x[:, :6]


class ObjectiveTest(unittest.TestCase):
    def test_perfect_unlabeled_pair_has_zero_error(self):
        x = torch.randn(4, 2, 256)
        reader = Reader().eval().requires_grad_(False)
        features = reader.features(x)
        soft = F.softmax(reader.classify_features(features)/2, dim=1)
        loss, parts = source_loss(reader, x, x, features, torch.tensor([4, 4, 6, 6]))
        self.assertLess(abs(float(loss)), 1e-6)
        self.assertEqual(float(parts['wave'].max()), 0)

    def test_frozen_feature_loss_has_finite_gradient(self):
        clean = torch.randn(3, 2, 256)
        output = torch.randn_like(clean, requires_grad=True)
        reader = Reader().eval().requires_grad_(False)
        loss, _ = source_loss(reader, output, clean, reader.features(clean), torch.tensor([4, 6, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
