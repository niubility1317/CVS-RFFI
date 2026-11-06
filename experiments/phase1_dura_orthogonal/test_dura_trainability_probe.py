import unittest

import torch

from experiments.phase1_dura_orthogonal.dura_trainability_probe import gradient_summary


class TrainabilityProbeTest(unittest.TestCase):
    def test_gradient_summary_distinguishes_unconnected_and_nonfinite(self):
        model = torch.nn.Module()
        model.left = torch.nn.Linear(2, 1)
        model.right = torch.nn.Linear(2, 1)
        model.left.weight.grad = torch.tensor([[1., float('nan')]])
        summary = gradient_summary(model)
        self.assertFalse(summary['left']['gradient_finite'])
        self.assertEqual(summary['right']['with_gradient'], 0)
        self.assertGreater(summary['left']['with_gradient'], 0)


if __name__ == '__main__':
    unittest.main()
