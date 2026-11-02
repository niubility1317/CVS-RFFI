import unittest

import torch
from torch import nn

from experiments.phase1_dura_posterior.source_objective import source_objective


class Reader(nn.Module):
    def features(self, x):
        return x.flatten(1)

    def classify_features(self, x):
        return torch.stack((x[:, 0]*4, -x[:, 0]*4), 1)


class SourceObjectiveTest(unittest.TestCase):
    def test_perfect_pair_has_no_wave_or_feature_error(self):
        x = torch.randn(4, 2, 256)
        reader = Reader().eval().requires_grad_(False)
        with torch.no_grad():
            feat = reader.features(x)
            logits = reader.classify_features(feat)
        loss, parts = source_objective(reader, x, x, x, feat, logits, logits, torch.tensor([4, 4, 6, 6]))
        self.assertLess(abs(float(loss)), 1e-6)
        self.assertEqual(float(parts['wave'].max()), 0)
        self.assertLess(float(parts['feature'].abs().max()), 1e-6)

    def test_gradients_enter_output_only_and_bad_groups_are_rejected(self):
        reader = Reader().eval().requires_grad_(False)
        clean = torch.randn(4, 2, 256)
        output = torch.randn_like(clean, requires_grad=True)
        with torch.no_grad():
            feat = reader.features(clean)
            logits = reader.classify_features(feat)
        loss, _ = source_objective(reader, output, clean, clean, feat, logits, logits, torch.tensor([4, 6, 4, 6]))
        loss.backward()
        self.assertGreater(float(output.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(output.grad).all())
        with self.assertRaises(ValueError):
            source_objective(reader, output, clean, clean, feat, logits, logits, torch.tensor([4, 6]))

    def test_training_reader_is_rejected(self):
        reader = Reader().train()
        x = torch.randn(1, 2, 256)
        feat = reader.features(x)
        logits = reader.classify_features(feat)
        with self.assertRaises(ValueError):
            source_objective(reader, x, x, x, feat, logits, logits, torch.tensor([4]))


if __name__ == '__main__':
    unittest.main()
