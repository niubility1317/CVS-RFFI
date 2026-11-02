import importlib
import unittest

import torch
from torch import nn


class FrozenReader(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(32)
        self.proj = nn.Linear(512, 24)
        self.head = nn.Linear(24, 6)
        self.eval().requires_grad_(False)

    def features(self, x):
        return torch.tanh(self.proj(x.flatten(1)))

    def classify_features(self, x):
        return self.head(x)


class TangentLossTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.reader = FrozenReader()
        self.clean = torch.randn(4, 2, 256)
        self.direction = torch.randn_like(self.clean)
        self.groups = torch.tensor([4, 4, 6, 6])
        self.clean_features = self.reader.features(self.clean).detach()

    def loss(self, output):
        try:
            mod = importlib.import_module('experiments.phase1_dura_tangent.objective')
        except ModuleNotFoundError as exc:
            if exc.name == 'experiments.phase1_dura_tangent.objective':
                self.fail('feature-response consistency loss is not implemented')
            raise
        return mod.tangent_consistency_loss(self.reader, output, self.clean,
                    self.clean_features, self.groups, direction=self.direction)

    def test_matching_waveforms_have_identical_perturbation_response(self):
        value = self.loss(self.clean.clone().requires_grad_())
        self.assertLess(float(value.detach()), 1e-10)

    def test_mismatched_waveforms_backpropagate_without_reader_updates(self):
        output = (self.clean + 0.5 * torch.randn_like(self.clean)).requires_grad_()
        value = self.loss(output)
        self.assertGreater(float(value.detach()), 1e-8)
        value.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.reader.parameters()))

    def test_complete_loss_trains_dura_correction_without_reader_updates(self):
        from experiments.phase1_dura_tangent.objective import tangent_source_objective
        from experiments.phase1_dura_context.model import ContextRecovery
        model = ContextRecovery()
        received = self.clean + 0.1 * torch.randn_like(self.clean)
        anchor = received + 0.02 * torch.randn_like(self.clean)
        output = model(received, anchor, torch.full((4,), -2.0))
        loss, terms = tangent_source_objective(self.reader, output, anchor, self.clean,
                                                self.clean_features, self.groups)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms['feature_response']))
        loss.backward()
        self.assertTrue(torch.isfinite(model.gate_output.weight.grad).all())
        self.assertGreater(float(model.gate_output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.reader.parameters()))

    def test_rejects_wrong_receiver_group_count(self):
        self.groups = self.groups[:2]
        with self.assertRaises(ValueError):
            self.loss(self.clean.clone().requires_grad_())


if __name__ == '__main__': unittest.main()
