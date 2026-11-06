import unittest

import torch

from experiments.phase1_dura_orthogonal.model import InternalDuraRecovery
from experiments.phase1_dura_orthogonal.train_source import source_loss


class InternalDuraTest(unittest.TestCase):
    def test_owned_model_trains_without_using_external_context(self):
        torch.manual_seed(91)
        inner = torch.nn.Conv1d(2, 2, 1)
        model = InternalDuraRecovery(inner)
        received, clean = torch.randn(2, 2, 256), torch.randn(2, 2, 256)
        first = model(received, torch.zeros_like(received), torch.zeros_like(received), torch.ones(2))
        second = model(received, torch.ones_like(received), torch.ones_like(received), torch.ones(2)*2)
        self.assertTrue(torch.equal(first, second))
        source_loss(first, received, clean, objective='dura_guard').backward()
        self.assertIsNotNone(inner.weight.grad)
        self.assertTrue(torch.isfinite(inner.weight.grad).all())
        self.assertGreater(float(inner.weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
