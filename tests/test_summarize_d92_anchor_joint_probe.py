"""Independent synthetic AJLR certificate/score tampering checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_anchor_joint_probe as probe
import summarize_d92_anchor_joint_probe as summary
from test_evaluate_d92_anchor_joint_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.inputs = synthetic()
        archive = probe.StateArchive(cls.directory.name)
        cls.fit_stages = Path(cls.directory.name)/'fit_stages.jsonl'
        with cls.fit_stages.open('x', encoding='utf-8') as stream:
            def log_callback(stage):
                row = dict(probe.compact_event(stage), split_id='synthetic-parent')
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            result = probe.probe_anchor_joint(**cls.inputs, state_callback=archive, log_callback=log_callback)
        cls.record, cls.split = as_record(result, cls.inputs, 3)
        archive.finalize('COMPLETE')

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def resolver(self): return summary.StateResolver(self.directory.name)

    def test_actual_callback_stage_stream_matches_strict_reconstruction(self):
        resolver = self.resolver()
        logs, _, _, _ = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        emitted = list(summary.jsonlines(self.fit_stages))
        expected = [dict(probe.compact_event(value), split_id=self.record['split_id']) for value in logs]
        self.assertEqual(emitted, expected)
        per_path = [('BASE_FIT', 'B0'), ('BASE_FIT', 'C0'),
            ('AJLR_PREPARATION', 'B'), ('CANDIDATE_FIT', 'B_AJLR'),
            ('AJLR_PREPARATION', 'C'), ('CANDIDATE_FIT', 'C_AJLR_seq')]
        path_count = len(self.record['folds'])+len(self.record['oneshot_proxy']['trials'])
        self.assertEqual([(value['event'], value['state']) for value in emitted], per_path*path_count)

    def test_complete_prior_reference_ce_solver_scores_and_workload_recomputed(self):
        resolver = self.resolver()
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        self.assertEqual(len(training), self.record['ajlr_stage_count'])
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(v['event'] == 'AJLR_GRADIENT' for v in events), self.record['backward_evaluation_count'])
        self.assertEqual(sum(v['event'] == 'AJLR_TRIAL' for v in events), self.record['trial_count'])
        self.assertEqual(sum(v['event'] == 'AJLR_STEP' for v in events), self.record['optimizer_steps'])
        for entry in self.record['folds']:
            b, c = entry['candidate_stages']
            initial = resolver(c['initialization_state_ref']); bfinal = resolver(b['final_state_ref'])
            np.testing.assert_array_equal(initial['U'], bfinal['U']); np.testing.assert_array_equal(initial['Z'], np.zeros_like(initial['Z']))
            self.assertIsNotNone(c['score_workload']); self.assertIn('prior', c['score_workload'])
        measured_reference_cases = set()
        for entry in self.record['folds']:
            for stage in entry['candidate_stages']:
                for objective in [v['objective'] for v in stage['trials']]:
                    for fold in objective['inner_folds']:
                        if fold['identity_forward']: continue
                        m = len(fold['old_reference_physical_ids']); n = fold['train_physical_count']; h = fold['held_physical_count']
                        expected_pairs = m*(m-1)//2+m*(n-m)+h*m
                        self.assertEqual(fold['reference_distance_pair_count'], expected_pairs)
                        measured_reference_cases.add('all_old' if m == n else 'old_subset')
        self.assertEqual(measured_reference_cases, {'all_old', 'old_subset'})
        resources = {}; summary.accumulate_resources(resources, self.record, logs)
        self.assertIn('outer_prior_raw_distance_evaluation_count_sum', resources)

    def test_prior_reference_ce_trial_cost_event_score_tampering_rejected(self):
        for change in ('prior_leak', 'reference', 'ce', 'cost', 'trial_acceptance', 'stop', 'drop_event', 'proxy_score', 'rank', 'A'):
            record = deepcopy(self.record); entry = record['folds'][0]; b, c = entry['candidate_stages']
            if change == 'prior_leak': entry['preparations'][1]['prior_folds'][0]['training_physical_ids'][0] = entry['c_ids'][0]
            if change == 'reference': entry['preparations'][1]['inner_folds'][0]['old_reference_physical_ids'][0] = entry['c_ids'][0]
            if change == 'ce': b['initial_objective']['class_ce_means'][0] += .1
            if change == 'cost': record['derivative_triangular_solve_count'] += 1
            if change == 'trial_acceptance': b['trials'][0]['objective_nonincrease_pass'] = not b['trials'][0]['objective_nonincrease_pass']
            if change == 'stop': b['stop_reason'] = 'INVENTED_CONVERGENCE'
            if change == 'drop_event': entry['training_events'].pop()
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_AJLR_seq']['c_scores'][0][0] += 20.
            if change == 'rank': entry['preparations'][0]['latent_rank'] += 1
            if change == 'A': entry['paths']['R_AJLR_seq']['metrics']['A_old_accuracy'] = .5
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                summary.verify_record(record, self.split, self.inputs['old_classes'], self.resolver())

    def test_archive_path_shape_and_numeric_coordinate_changes_rejected(self):
        ref = self.record['folds'][0]['candidate_stages'][0]['final_state_ref']
        for change in ('path', 'shape', 'norm'):
            bad = deepcopy(ref)
            if change == 'path': bad['path'] = '../escaped.npz'
            if change == 'shape': bad['arrays']['U']['shape'] = [5888]
            if change == 'norm': bad['array_summaries']['U']['norm'] += 1.
            with self.subTest(change=change), self.assertRaises(ValueError): self.resolver()(bad)

    def test_true_k1_and_zero_old_scale_keep_real_closed_heads_and_no_held_metrics(self):
        for zero in (False, True):
            inputs = synthetic(k=1, zero=zero)
            with tempfile.TemporaryDirectory() as directory:
                archive = probe.StateArchive(directory)
                record, split = as_record(probe.probe_anchor_joint(**inputs, state_callback=archive), inputs, 1)
                archive.finalize('COMPLETE'); resolver = summary.StateResolver(directory)
                _, _, metrics, training = summary.verify_record(record, split, inputs['old_classes'], resolver); resolver.finalize()
            self.assertEqual(metrics, dict(oof=None, proxy=None)); self.assertEqual(len(training), 2)
            self.assertEqual(record['optimizer_steps'], 0); self.assertEqual(record['final_head_fit_count'], 2)

    def test_incomplete_run_does_not_open_held_trace_or_arrays(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores, \
             patch.object(summary, 'StateResolver') as archives:
            with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(spec=spec, output=Path(directory)/'out')
            scores.assert_not_called(); archives.assert_not_called()


if __name__ == '__main__': unittest.main()
