import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_phase_gain_run as run


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


class PhaseAwareGainTest(unittest.TestCase):
    def test_shapes_bound_and_global_phase_equivariance(self):
        torch.manual_seed(19)
        for width, anchor_real in ((32, False), (1024, True)):
            model = run.PhaseAwareBoundedGain(width, anchor_real=anchor_real)
            torch.nn.init.normal_(model.net[-1].weight, std=.02)
            received = torch.randn(1, 8*width, dtype=torch.complex64)
            real_gain = received + .2*torch.randn_like(received)
            r17 = received + .2*torch.randn_like(received)
            output = model(received, real_gain, r17, IdentityAction())
            anchor = real_gain if anchor_real else r17
            correction = (output-anchor).abs().square().mean(-1).sqrt()
            bound = .05*anchor.abs().square().mean(-1).sqrt()
            self.assertTrue(bool(torch.all(correction <= bound + 1e-6)))
            rotation = torch.polar(torch.tensor(1.), torch.tensor(.71))
            rotated = model(received*rotation, real_gain*rotation, r17*rotation, IdentityAction())
            torch.testing.assert_close(rotated, output*rotation, rtol=1e-4, atol=1e-5)

    def test_worker_reconstructs_phase_model_and_anchor(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertIsInstance(model, run.PhaseAwareBoundedGain)
            self.assertEqual(bool(model.anchor_real), run.ANCHOR_REAL[dataset])
            self.assertEqual(model.correction_cap, .05)
            self.assertTrue(kwargs['allow_source_f_guidance'])
        with patch.object(run, 'read', return_value={}), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict', side_effect=check), \
             patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            for dataset in ('lora', 'wisig'):
                run.worker('runtime.json', dataset)


if __name__ == '__main__':
    unittest.main()
