import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_aligned_gain_run as run


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


class AlignedGainTest(unittest.TestCase):
    def test_scalar_alignment_recovers_anchor_and_handles_zero(self):
        torch.manual_seed(31)
        anchor = torch.randn(3, 256, dtype=torch.complex64)
        scalar = torch.tensor(0.4 + 1.2j, dtype=torch.complex64)
        torch.testing.assert_close(run.align_to_anchor(anchor, anchor*scalar), anchor, rtol=1e-5, atol=1e-5)
        self.assertTrue(bool(torch.isfinite(run.align_to_anchor(anchor, torch.zeros_like(anchor))).all()))

    def test_both_packet_shapes_keep_anchor_at_initialization(self):
        for width, anchor_real in ((32, False), (1024, True)):
            model = run.AlignedBoundedGain(width, anchor_real=anchor_real, correction_cap=.05)
            received = torch.ones(1, 8*width, dtype=torch.complex64)
            real_gain, r17 = received*(1+1j), received*(2-1j)
            output = model(received, real_gain, r17, IdentityAction())
            torch.testing.assert_close(output, real_gain if anchor_real else r17, rtol=0, atol=0)

    def test_alignment_does_not_change_original_network_features(self):
        torch.manual_seed(41)
        aligned = run.AlignedBoundedGain(32, anchor_real=True, correction_cap=.05)
        original = run.SharedBoundedGain(32, anchor_real=True, correction_cap=.05)
        original.load_state_dict(aligned.state_dict())
        captured = []
        hooks = [model.net[0].register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
                 for model in (original, aligned)]
        received = torch.randn(2, 256, dtype=torch.complex64)
        real_gain = torch.randn_like(received)
        r17 = (2+3j)*torch.randn_like(received)
        for model in (original, aligned):
            model(received, real_gain, r17, IdentityAction())
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)

    def test_worker_reconstructs_aligned_model(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertIsInstance(model, run.AlignedBoundedGain)
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
