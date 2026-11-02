import unittest
import torch
from torch import nn

from experiments.phase1_spectral_coupling.model import SpectralCoupling
from experiments.phase1_dura_volterra.pipeline import predict_batch


class Gate(nn.Module):
    def forward(self, x): return x[:, 0, 0]*10


class Blind(nn.Module):
    def forward(self, x): return torch.zeros(len(x), 4), torch.zeros(len(x))


class Reader(nn.Module):
    def features(self, x): return x.mean(-1)
    def classify_features(self, x): return x.repeat(1, 3)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(318); torch.set_num_threads(2)
        self.x = torch.randn(2, 2, 256); self.x[0, 0, 0] = -2.; self.x[1, 0, 0] = 2.
        self.dura = lambda x: x*.7

    def test_zero_head_and_closed_gate_preserve_dura(self):
        model = SpectralCoupling()
        base, out, active, _ = predict_batch(self.x, model, Blind(), Gate(), self.dura, None, Reader(), None)
        self.assertEqual(active.tolist(), [False, True])
        self.assertTrue(torch.equal(base, self.dura(self.x)))
        self.assertTrue(torch.equal(out, base))
        torch.nn.init.normal_(model.gate_output.bias, std=.1)
        base, out, active, _ = predict_batch(self.x, model, Blind(), Gate(), self.dura, None, Reader(), None)
        self.assertTrue(torch.equal(out[~active], base[~active]))
        self.assertGreater(float((out[active]-base[active]).abs().sum()), 0.)


if __name__ == '__main__': unittest.main()
