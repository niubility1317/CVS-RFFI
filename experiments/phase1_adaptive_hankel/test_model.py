import unittest
import torch
from experiments.phase1_adaptive_hankel.model import AdaptiveHankel
from experiments.phase1_equivariant_dictionary.objective import source_loss
from experiments.phase1_equivariant_dictionary.test_objective import Reader


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)


class AdaptiveHankelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(231)
        torch.set_num_threads(2)
        self.raw = torch.randn(3, 2, 256)
        self.initial = torch.randn_like(self.raw)
        self.logvar = torch.randn(3)

    def test_zero_initialization_is_exact_identity(self):
        model = AdaptiveHankel()
        self.assertTrue(torch.equal(model(self.raw, self.initial, self.logvar), self.initial))

    def test_nonzero_model_is_phase_equivariant_and_batch_stable(self):
        model = AdaptiveHankel()
        torch.nn.init.normal_(model.gate_output.weight, std=.1)
        torch.nn.init.normal_(model.gate_output.bias, std=.1)
        angle = torch.tensor(.67)
        expected = rotate(model(self.raw, self.initial, self.logvar), angle)
        actual = model(rotate(self.raw, angle), rotate(self.initial, angle), self.logvar)
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
        batch = model(self.raw, self.initial, self.logvar)
        one = torch.cat([model(self.raw[i:i+1], self.initial[i:i+1], self.logvar[i:i+1]) for i in range(3)])
        torch.testing.assert_close(batch, one, rtol=4e-5, atol=4e-5)

    def test_learns_without_updating_physical_input(self):
        model = AdaptiveHankel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        output = model(self.raw, self.initial, self.logvar)
        (output-self.raw).square().mean().backward()
        grad = model.gate_output.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        optimizer.step()
        self.assertGreater(float((model(self.raw, self.initial, self.logvar)-self.initial).detach().abs().sum()), 0.)

    def test_two_source_updates_learn_rank_features_without_reader_updates(self):
        model = AdaptiveHankel()
        reader = Reader().eval().requires_grad_(False)
        clean = torch.randn_like(self.raw)
        features = reader.features(clean)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            out = model(self.raw, self.initial, self.logvar)
            loss, _ = source_loss(reader, out, clean, features, torch.tensor([4, 4, 6]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(model.entry.weight.grad).all())
        self.assertGreater(float(model.entry.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_silence_is_finite_and_invalid_inputs_rejected(self):
        model = AdaptiveHankel()
        x = torch.zeros(2, 2, 256)
        self.assertTrue(torch.isfinite(model(x, x, torch.zeros(2))).all())
        with self.assertRaises(ValueError): model(x, x, torch.zeros(1))

if __name__ == '__main__': unittest.main()
