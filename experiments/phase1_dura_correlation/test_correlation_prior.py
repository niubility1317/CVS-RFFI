import unittest
import torch
from experiments.phase1_dura_correlation.correlation_prior import CorrelationPriorCorrection


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


class CorrelationPriorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913); torch.set_num_threads(2)
        self.x = [torch.randn(3, 2, 256) for _ in range(4)]
        self.lv = torch.randn(3)

    def test_initialization_exactly_preserves_prior(self):
        self.assertTrue(torch.equal(CorrelationPriorCorrection()(*self.x, self.lv), self.x[2]))

    def test_nonzero_model_preserves_phase_and_batch_independence(self):
        model = CorrelationPriorCorrection()
        torch.nn.init.normal_(model.coefficients.weight, std=.1)
        torch.nn.init.normal_(model.coefficients.bias, std=.1)
        torch.nn.init.normal_(model.phase_coeff, std=.1)
        angle = torch.tensor(.71)
        full = model(*self.x, self.lv)
        rotated = model(*(rotate(v, angle) for v in self.x), self.lv)
        torch.testing.assert_close(rotated, rotate(full, angle), rtol=4e-5, atol=4e-5)
        alone = model(*(v[1:2] for v in self.x), self.lv[1:2])
        torch.testing.assert_close(alone, full[1:2], rtol=3e-5, atol=3e-5)

    def test_attention_reaches_distant_time_samples(self):
        model = CorrelationPriorCorrection()
        torch.nn.init.normal_(model.coefficients.weight, std=.1)
        posterior = self.x[3].clone().requires_grad_(True)
        value = model(*self.x[:3], posterior, self.lv)[0, 0, 0]
        gradient = torch.autograd.grad(value, posterior)[0][0, :, 64:]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 1e-5)

    def test_gate_and_attention_learn_with_two_steps(self):
        model = CorrelationPriorCorrection(); optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (model(*self.x, self.lv)-self.x[0]).square().mean().backward()
            gate = model.coefficients.weight.grad
            self.assertTrue(torch.isfinite(gate).all()); self.assertGreater(float(gate.abs().sum()), 0.)
            optimizer.step()
        query = model.query.weight.grad
        self.assertIsNotNone(query); self.assertTrue(torch.isfinite(query).all())
        self.assertGreater(float(query.abs().sum()), 0.)

    def test_bad_shape_is_rejected(self):
        with self.assertRaises(ValueError): CorrelationPriorCorrection()(*self.x, torch.zeros(2))


if __name__ == '__main__': unittest.main()
