import unittest
import torch
from experiments.phase1_spectral_moe.model import SpectralMixture

class SpectralMixtureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = SpectralMixture(16)
        self.y = torch.randn(3, 2, 256)
        self.anchor = torch.randn_like(self.y)
        self.joint = torch.randn_like(self.y)
        self.diffusion = torch.randn_like(self.y)

    def test_clean_gate_is_exact_anchor(self):
        value = self.model(self.y, self.anchor, self.joint, self.diffusion, torch.zeros(3))
        self.assertTrue(torch.equal(value, self.anchor))

    def test_equal_experts_and_batch_independence(self):
        variance = torch.ones(3)
        result = self.model(self.y, self.anchor, self.anchor, self.anchor, variance)
        torch.testing.assert_close(result, self.anchor, rtol=1e-5, atol=1e-6)
        together = self.model(self.y, self.anchor, self.joint, self.diffusion, variance)
        separately = torch.cat([self.model(self.y[i:i+1], self.anchor[i:i+1], self.joint[i:i+1],
                            self.diffusion[i:i+1], variance[i:i+1]) for i in range(3)])
        torch.testing.assert_close(together, separately, rtol=1e-5, atol=1e-6)

    def test_source_gradient(self):
        result = self.model(self.y, self.anchor, self.joint, self.diffusion, torch.ones(3))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(self.model.network[-1].weight.grad).all())
        self.assertGreater(self.model.network[-1].weight.grad.abs().sum().item(), 0)

if __name__ == '__main__': unittest.main()
