import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_risk_router_run as run


class SharedRiskRouterTest(unittest.TestCase):
    def test_pseudolabel_routing_ignores_both_wrong_and_prefers_anchor_on_tie(self):
        clean = torch.tensor([[4., 0., 0.]] * 4)
        anchor = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 4., 0.], [4., 0., 0.]])
        other = torch.tensor([[4., 0., 0.], [4., 0., 0.], [0., 0., 4.], [0., 4., 0.]])
        target, informative = run.switch_targets(clean, anchor, other)
        torch.testing.assert_close(target, torch.tensor([0., 1., 0., 0.]))
        torch.testing.assert_close(informative, torch.tensor([True, True, False, True]))

    def test_worker_reconstructs_dataset_anchor(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertEqual(bool(model.anchor_real), run.ANCHOR_REAL[dataset])
            self.assertEqual(float(model.head.bias.item()), -2.)
            self.assertTrue(kwargs['allow_source_f_guidance'])
        with patch.object(run, 'read', return_value={}), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict', side_effect=check), \
             patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            for dataset in ('lora', 'wisig'):
                run.worker('runtime.json', dataset)


if __name__ == '__main__':
    unittest.main()
