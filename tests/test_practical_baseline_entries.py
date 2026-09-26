import importlib
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class EntryTests(unittest.TestCase):
    def test_all_five_backbones_complete_real_augmented_update(self):
        torch.set_num_threads(2)
        batch = dict(iq=torch.randn(4, 2, 256), label=torch.arange(4),
                     receiver=torch.tensor([1, 3, 4, 6]), domain=torch.arange(4),
                     meta=[dict(sample_id=f'sample{i}', session_id=f'rx{i}/day1') for i in range(4)])
        split = SimpleNamespace(num_classes=6, num_receivers=12, input_len=256,
                                split_info={'train_rxs_idx': [1,3,4,6,8]})
        loaders = SimpleNamespace(split=split, train=[batch], val=[batch],
            named_tests={}, unlabeled=[dict(batch, label=torch.full((4,), -1))])
        for method in ['cvcnn_ce', 'riei_fd', 'drift', 'poster', 'radionet']:
            with self.subTest(method=method):
                module = importlib.import_module(f"baselines.{'cvcnn_ce' if method in {'poster','radionet'} else method}.train_cvs")
                argv = ['test', '--device', 'cpu', '--source_only', '--use_source_ssl_split',
                    '--practical_residual_noeq', '--practical_fs_hz', '25000000',
                    '--concat_sat_ce_only', '--use_concat_sat_channel_aug', '--use_pseudo_labels']
                if method in {'poster','radionet'}:
                    argv += ['--author_backbone', method]

                def exercise(**kw):
                    self.assertTrue(kw['source_only'])
                    self.assertEqual(kw['named_test_loaders'], {})
                    self.assertIsNone(kw['extra_test_fn'])
                    model = kw['model']
                    before = [p.detach().clone() for p in model.parameters()]
                    loss = kw['train_step_fn'](model, batch, torch.device('cpu'), 151, 0)
                    self.assertTrue(math.isfinite(loss['loss']))
                    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
                    result = kw['pseudo_step_fn'](model, torch.device('cpu'), 151, 0)
                    self.assertTrue(math.isfinite(result['loss']))
                with patch.object(module, 'build_cvs_loaders', return_value=loaders), \
                     patch.object(module, 'run_validation_gated_training', side_effect=exercise), \
                     patch.object(sys, 'argv', argv):
                    module.main()


if __name__ == '__main__':
    unittest.main()
