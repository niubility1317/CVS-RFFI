import unittest
import torch

from experiments.phase1_dura_orthogonal.model import OrthogonalRecovery, split_direction


class OrthogonalRecoveryTest(unittest.TestCase):
    def test_projection_separates_per_sample_and_handles_zero(self):
        torch.manual_seed(4)
        residual = torch.randn(3, 2, 256)
        direction = torch.randn_like(residual)
        direction[1].zero_()
        parallel, orthogonal = split_direction(residual, direction)
        torch.testing.assert_close(parallel + orthogonal, residual)
        dot = (orthogonal * direction).sum((1, 2))
        torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-4, rtol=0)
        torch.testing.assert_close(orthogonal[1], residual[1])

    def test_initial_identity_gradient_and_batch_independence(self):
        torch.manual_seed(5)
        model = OrthogonalRecovery(width=8)
        x = torch.randn(3, 2, 256)
        anchor = torch.randn_like(x)
        direction = torch.randn_like(x)
        noise = torch.tensor([.01, .1, .2])
        output = model(x, anchor, direction, noise)
        self.assertTrue(torch.equal(output, anchor))
        loss = (output - x).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.wave.weight.grad).all())
        self.assertGreater(float(model.wave.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.step.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.wave.bias.fill_(.1)
            model.step.bias.fill_(-.2)
            whole = model(x, anchor, direction, noise)
            single = torch.cat([model(x[i:i+1], anchor[i:i+1], direction[i:i+1], noise[i:i+1])
                                for i in range(len(x))])
            torch.testing.assert_close(whole, single, atol=2e-6, rtol=2e-6)

    def test_invalid_noise_and_shape_rejected(self):
        model = OrthogonalRecovery(width=8)
        x = torch.ones(2, 2, 256)
        with self.assertRaises(ValueError):
            model(x, x, x, torch.tensor([.1, 0.]))
        with self.assertRaises(ValueError):
            model(x, x, x[:1], torch.ones(2))


if __name__ == '__main__':
    unittest.main()
