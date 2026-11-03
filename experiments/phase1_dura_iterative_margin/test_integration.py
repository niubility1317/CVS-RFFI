import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.phase1_dura_iterative_margin import train_source, ground, score_source
from experiments.phase1_dura_iterative_margin.objective import recognition_protected_objective
from experiments.phase1_dura_iterative_timefreq.model import IterativeDura


class FrozenReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 160)
        self.head = nn.Linear(160, 6)
        self.eval().requires_grad_(False)

    def features(self, x):
        return self.encoder(x.flatten(1))

    def classify_features(self, x):
        return self.head(x)


class IntegrationTest(unittest.TestCase):
    def test_full_source_objective_backpropagates_without_unfreezing_reader(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        x = torch.randn(2, 2, 256)
        clean = torch.randn_like(x)
        anchor = (x * .7).detach()
        reader = FrozenReader()
        model = IterativeDura(width=16)
        output = model(x, anchor, torch.full((2,), -2.))
        loss, terms = recognition_protected_objective(reader, output, anchor, clean,
                                                        reader.features(clean), torch.tensor([4, 8]))
        self.assertIn('relative_recognition_margin', terms)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.gate_output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.frequency_gate.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in reader.parameters()))

    def test_training_adapter_restores_shared_configuration_on_failure(self):
        base = train_source.base
        names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
                 'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
        old = tuple(getattr(base, k) for k in names)
        def fail():
            self.assertIs(base.DynamicVolterra, IterativeDura)
            self.assertIs(base.source_objective, recognition_protected_objective)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('stop')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                train_source.main()
        self.assertEqual(old, tuple(getattr(base, k) for k in names))

    def test_ground_adapter_loads_the_registered_dura_schema(self):
        base = ground.base
        names = ('DynamicVolterra', 'SOURCE', 'SCHEMA', 'LOG_PREFIX', 'USES_DURA', 'load_components', 'predict_batch')
        old = tuple(getattr(base, k) for k in names)
        def fail():
            self.assertIs(base.DynamicVolterra, IterativeDura)
            self.assertEqual(base.SOURCE, ground.SOURCE)
            self.assertEqual(base.SCHEMA, 'phase1_dura_iterative_margin_v1')
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('stop')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                ground.main()
        self.assertEqual(old, tuple(getattr(base, k) for k in names))

    def test_source_score_adapter_checks_dura_schema(self):
        base = score_source.base
        old = base.SCHEMA, base.USES_DURA
        def fail():
            self.assertEqual(base.SCHEMA, 'phase1_dura_iterative_margin_v1')
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('stop')
        with patch.object(base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                score_source.main()
        self.assertEqual(old, (base.SCHEMA, base.USES_DURA))


if __name__ == '__main__':
    unittest.main()
