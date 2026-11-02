import unittest
import torch

from experiments.phase1_dura_safe_memory.objective import degradation_penalty


class DegradationPenaltyTest(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(1841)
        self.clean = torch.randn(3, 2, 256, generator=rng)
        self.noise = torch.randn(3, 2, 256, generator=rng)
        self.groups = torch.tensor([4, 4, 8])

    def test_same_as_anchor_has_no_penalty(self):
        anchor = self.clean + 0.1 * self.noise
        p = degradation_penalty(anchor.clone(), anchor, self.clean, self.groups)
        self.assertEqual(float(p), 0.0)

    def test_worse_than_anchor_is_penalized_and_differentiable(self):
        candidate = (self.clean + 0.4 * self.noise).requires_grad_()
        anchor = self.clean.clone().requires_grad_()
        p = degradation_penalty(candidate, anchor, self.clean, self.groups)
        self.assertGreater(float(p), 0.01)
        p.backward()
        self.assertGreater(float(candidate.grad.abs().sum()), 0.0)
        self.assertIsNone(anchor.grad)

    def test_improvement_over_anchor_is_not_penalized(self):
        anchor = self.clean + 0.4 * self.noise
        p = degradation_penalty(self.clean, anchor, self.clean, self.groups)
        self.assertEqual(float(p), 0.0)


if __name__ == '__main__':
    unittest.main()
