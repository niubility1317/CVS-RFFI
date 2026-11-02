import unittest
import torch

from experiments.phase1_dura_crosscycle.objective import cycle_residual, next_view


class CrossViewCycleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.g = torch.Generator().manual_seed(2568)

    def test_exact_complex_gain_is_explained(self):
        candidate = torch.randn(2, 2, 256, generator=self.g)
        target = torch.stack((0.7 * candidate[:, 0] - 0.4 * candidate[:, 1],
                              0.4 * candidate[:, 0] + 0.7 * candidate[:, 1]), 1)
        self.assertLess(float(cycle_residual(candidate, target).max()), 1e-10)

    def test_unrelated_views_give_differentiable_positive_residual(self):
        candidate = torch.randn(2, 2, 256, generator=self.g).requires_grad_()
        target = torch.randn(2, 2, 256, generator=self.g)
        loss = cycle_residual(candidate, target).mean()
        self.assertGreater(float(loss), 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.abs().sum()), 0.0)

    def test_pairs_same_ids_across_three_views(self):
        inputs = torch.tensor([[10], [11], [20], [21], [30], [31]])
        expected = torch.tensor([[20], [21], [30], [31], [10], [11]])
        torch.testing.assert_close(next_view(inputs), expected)

    def test_rejects_incomplete_view_batch(self):
        with self.assertRaises(ValueError):
            next_view(torch.zeros(5, 2, 256))


if __name__ == '__main__': unittest.main()
