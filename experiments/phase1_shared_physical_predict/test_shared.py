import unittest
import torch
from torch import nn
from experiments.phase1_dura_physical_fusion.pipeline import predict_batch
from experiments.phase1_shared_physical_predict.shared import predict_shared


class CountDura(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def forward(self, x):
        self.calls += 1; return x * .7


class Gate(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode; self.calls = 0
    def forward(self, x):
        self.calls += 1
        if self.mode == 'none': return x.new_full((len(x),), -20)
        if self.mode == 'all': return x.new_full((len(x),), 20)
        return x[:, 0, 0] * 20


class CountPhysical(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def forward(self, x):
        self.calls += 1; return x * .2, x.new_full((len(x),), -2.)


class Correction(nn.Module):
    def __init__(self, factor):
        super().__init__(); self.factor = factor
    def forward(self, x, packed, log_variance):
        return packed[:, :2] + self.factor * (packed[:, 2:] - packed[:, :2])


class Reader(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def features(self, x):
        self.calls += 1; return x.mean(-1)
    def classify_features(self, features):
        return torch.cat([features, -features, features * .5], dim=1)


class SharedPredictionTest(unittest.TestCase):
    def test_shared_matches_three_separate_paths_for_all_gates(self):
        torch.manual_seed(5); torch.set_num_threads(1)
        for n in (1, 8):
            for mode in ('none', 'all', 'mixed'):
                with self.subTest(n=n, mode=mode):
                    x = torch.randn(n, 2, 256)
                    original = x.clone()
                    models = [Correction(x).eval() for x in (.1, .2, -.1)]
                    d, p, g, f = CountDura(), CountPhysical(), Gate(mode), Reader()
                    expected = [predict_batch(x, m, None, g, d, p, f, None) for m in models]
                    self.assertEqual((d.calls, g.calls, f.calls), (3, 3, 9))
                    self.assertEqual(p.calls, 0 if mode == 'none' else 3)
                    d, p, g, f = CountDura(), CountPhysical(), Gate(mode), Reader()
                    got = predict_shared(x, models, g, d, p, f)
                    self.assertEqual((d.calls, g.calls, f.calls), (1, 1, 5))
                    self.assertEqual(p.calls, 0 if mode == 'none' else 1)
                    self.assertTrue(torch.equal(x, original))
                    for reference, candidate in zip(expected, got):
                        for a, b in zip(reference[:3], candidate[:3]):
                            self.assertTrue(torch.equal(a, b))
                        for a, b in zip(reference[3], candidate[3]):
                            self.assertTrue(torch.equal(a, b))

    def test_models_must_be_frozen_for_shared_prediction(self):
        x = torch.zeros(1, 2, 256)
        models = [Correction(.1).eval(), Correction(.2).train()]
        with self.assertRaises(ValueError):
            predict_shared(x, models, Gate('all'), CountDura(), CountPhysical(), Reader())


if __name__ == '__main__':
    unittest.main()
