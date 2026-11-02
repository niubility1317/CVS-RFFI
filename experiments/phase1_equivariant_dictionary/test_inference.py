import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from experiments.phase1_equivariant_dictionary import train_source


class Gate(nn.Module):
    def forward(self, x): return x[:, 0, 0]


class Model(nn.Module):
    def forward(self, x, base, logvar): return base+0.5


class Reader(nn.Module):
    def features(self, x): return x.flatten(1)
    def classify_features(self, x): return x[:, :6]


class InferenceTest(unittest.TestCase):
    def test_conditional_mixture_mean_uses_current_observation(self):
        class Blind(nn.Module):
            def forward(self, x): return torch.ones(len(x), 1), torch.zeros(len(x))
        def posterior(y, action, noise, *stats):
            self.assertEqual(noise.shape, (2, 512, 512))
            weights = torch.tensor([.25, .75]).log().expand(len(y), -1)
            return torch.stack((y, y+4), dim=1), weights
        deps = SimpleNamespace(action_matrix=lambda x: x, posterior=posterior)
        x = torch.randn(2, 2, 256)
        mean, logvar = train_source.initial_posterior(x, Blind(), (), torch.eye(512), deps)
        torch.testing.assert_close(mean, x+3)
        torch.testing.assert_close(logvar, torch.zeros(2))

    def test_gate_preserves_closed_inputs_and_runs_prior_only_on_open_inputs(self):
        x = torch.zeros(3, 2, 256)
        x[:, 0, 0] = torch.tensor([-3., 2., 3.])
        def physical(selected, *unused): return selected+1, torch.zeros(len(selected))
        with patch.object(train_source, 'initial_posterior', side_effect=physical) as inference:
            parent, output, active, classes = train_source.predict_batch(x, Model(), None, Gate(), None, None, Reader(), None)
        self.assertEqual(active.tolist(), [False, True, True])
        self.assertEqual(len(inference.call_args.args[0]), 2)
        torch.testing.assert_close(output[0], x[0], rtol=0, atol=0)
        torch.testing.assert_close(parent[1:], x[1:]+1, rtol=0, atol=0)
        torch.testing.assert_close(output[1:], x[1:]+1.5, rtol=0, atol=0)
        self.assertEqual(len(classes), 3)

    def test_nonfinite_reader_scores_cannot_become_predictions(self):
        class BadReader(Reader):
            def classify_features(self, x): return torch.full((len(x), 6), float('nan'))
        x = torch.zeros(2, 2, 256)
        with self.assertRaises(FloatingPointError):
            train_source.predict_batch(x, Model(), None, Gate(), None, None, BadReader(), None)

    def test_closed_gate_never_computes_posterior(self):
        x = torch.zeros(2, 2, 256)
        with patch.object(train_source, 'initial_posterior', side_effect=AssertionError('must not run')):
            parent, output, active, _ = train_source.predict_batch(x, Model(), None, Gate(), None, None, Reader(), None)
        self.assertFalse(active.any())
        self.assertTrue(torch.equal(parent, x))
        self.assertTrue(torch.equal(output, x))

    def test_wrong_gate_output_shape_is_rejected(self):
        class BadGate(nn.Module):
            def forward(self, x): return torch.zeros(len(x), 2)
        with self.assertRaises(FloatingPointError):
            train_source.predict_batch(torch.zeros(2, 2, 256), Model(), None, BadGate(), None, None, Reader(), None)


if __name__ == '__main__':
    unittest.main()
