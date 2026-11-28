import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_confidence_router_run as run
from experiments.pamf_lora_main.shared_confidence_router_run import SharedConfidenceRouter, switch_targets


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


def teacher(iq):
    value = iq.real.mean(-1)
    return torch.stack((value, -value), dim=-1)


class SharedConfidenceRouterTest(unittest.TestCase):
    def test_initial_route_keeps_dataset_anchor(self):
        observed = torch.ones(2, 256, dtype=torch.complex64)
        real_gain, r17 = observed * 2, observed * 3
        for anchor_real in (True, False):
            model = SharedConfidenceRouter(32, anchor_real=anchor_real, teacher=teacher)
            output = model(observed, real_gain, r17, IdentityAction())
            torch.testing.assert_close(output, real_gain if anchor_real else r17)

    def test_nonfinite_classifier_logits_rejected(self):
        logits = torch.tensor([[1., 0.]])
        with self.assertRaises(ValueError):
            switch_targets(logits, logits, torch.tensor([[float('nan'), 1.]]))

    def test_wisig_teacher_load_follows_archive_activation(self):
        order = []
        def activated(_):
            order.append('archive')
        def make_teacher(_rt, _dataset):
            self.assertEqual(order, ['bootstrap', 'archive'])
            return teacher
        with patch.object(run, 'read', return_value={'project': '/p'}), \
             patch('experiments.cric_dura_comparison.bootstrap.activate', side_effect=lambda _: order.append('bootstrap')), \
             patch('experiments.pamf_lora_main.dura_real_gain_fit.activate_wisig_archive', side_effect=activated), \
             patch.object(run, 'source_teacher', side_effect=make_teacher), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict'), patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            run.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
