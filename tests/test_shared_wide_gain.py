import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_wide_gain_run as run


class IdentityAction:
    def __call__(self, value):
        return value

    def adjoint(self, value):
        return value


class WideSharedGainTest(unittest.TestCase):
    def test_both_packet_lengths_initialize_to_selected_anchor(self):
        for width, anchor_real in ((32, False), (1024, True)):
            model = run.WideSharedBoundedGain(width, anchor_real=anchor_real)
            received = torch.ones(1, 8 * width, dtype=torch.complex64)
            real_gain, r17 = received * 2, received * 3
            output = model(received, real_gain, r17, IdentityAction())
            torch.testing.assert_close(output, real_gain if anchor_real else r17, rtol=0, atol=0)
            self.assertEqual(model.correction_cap, .05)

    def test_worker_reconstructs_wide_model(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertIsInstance(model, run.WideSharedBoundedGain)
            self.assertEqual(bool(model.anchor_real), run.ANCHOR_REAL[dataset])
            self.assertEqual(model.correction_cap, .05)
            self.assertTrue(kwargs['allow_source_f_guidance'])
        with patch.object(run, 'read', return_value={}), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict', side_effect=check), \
             patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            run.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
