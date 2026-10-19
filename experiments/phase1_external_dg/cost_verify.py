"""Focused cost controls; no GPU, query or author dependency needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .cost import tasks, policy, mac_profile, dependency, summarize
from .plan import SEEDS


class CostControls(unittest.TestCase):
    def test_frozen_scope_and_three_seed_crossing(self):
        rows = [dict(row_id='%s-%s-s%d' % (family, variant, seed), family=family,
            variant=variant, model_seed=seed) for seed in SEEDS
            for family in ('riei', 'drift', 'ibrff', 'sdg') for variant in ('full', 'ce')]
        values = tasks(rows)
        self.assertEqual(len(values), 69)
        self.assertEqual({x['model_seed'] for x in values}, set(SEEDS))
        self.assertFalse(any('pamf' in str(x).lower() for x in values))
        valid = dict(allowed_gpus=[0, 1], batches=[1, 128], warmup=100, repeats=1000)
        policy(valid)
        for key, value in [('allowed_gpus', [2, 3]), ('repeats', 10), ('truth', 'hidden.json')]:
            with self.assertRaises(ValueError):
                policy(dict(valid, **{key: value}))

    def test_mac_counter_includes_conv_and_linear_without_double_count(self):
        net = torch.nn.Sequential(torch.nn.Conv1d(2, 4, 3, padding=1),
            torch.nn.Flatten(), torch.nn.Linear(32, 3)).eval()
        count = mac_profile(net, net, torch.zeros(1, 2, 8))
        self.assertEqual(count['conv_macs'], 192)
        self.assertEqual(count['macs'], 288)
        self.assertEqual(count['twice_macs'], 576)

    def test_upstream_process_must_exit_and_complete_all_registered_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            rt = dict(suite_pid=123, suite_release_root='owned-release', suite_run_root=folder)
            path = Path(folder) / 'completion.json'
            value = dict(status='ANALYZED', complete=True, parent_training=24, new_training=162, frozen_eval=6)
            path.write_text(json.dumps(value), encoding='utf-8')
            with patch('experiments.phase1_external_dg.cost.process_alive', return_value=True):
                self.assertEqual(dependency(rt), 'LIVE')
            with patch('experiments.phase1_external_dg.cost.process_alive', return_value=False):
                self.assertEqual(dependency(rt), 'DONE')
                value['new_training'] = 1
                path.write_text(json.dumps(value), encoding='utf-8')
                self.assertEqual(dependency(rt), 'FAILED')

    def test_summary_keeps_three_model_seeds_separate_from_timing_repeats(self):
        results = []
        for seed in SEEDS:
            results.append(dict(task=dict(origin='parent', protocol_id='F0', family='sdg',
                variant='full', model_seed=seed), parameter_cost=dict(identity_parameters=10),
                mac_profile=dict(macs=100), measurements=[dict(batch=batch, median_ms=1.,
                    p95_ms=2., examples_per_second=1000., peak_allocated_bytes=20)
                    for batch in (1, 128)]))
        records = summarize(results)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(x['actual_seeds'] == 3 and x['median_ms_sample_sd'] == 0. for x in records))
        with self.assertRaises(ValueError):
            summarize(results + [results[0]])


if __name__ == '__main__':
    unittest.main()
