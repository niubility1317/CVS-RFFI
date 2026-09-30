"""Synthetic paired evidence and complete-pilot guard checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools'), str(ROOT/'tests')]
import evaluate_d92_within_class_metric_probe as probe
import summarize_d92_within_class_metric_probe as summary
from test_evaluate_d92_within_class_metric_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic()
        cls.record, cls.split = as_record(probe.probe_within_class_metric(**cls.inputs), cls.inputs, 3)

    def test_recompute_pairing_actual_costs_and_train_only_diagnostics(self):
        logs, metrics, diagnostics = summary.verify_record(self.record, self.split, self.inputs['old_classes'])
        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(sum(value['loco'] is not None for value in diagnostics), 3)
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        resources = {}; summary.accumulate_resources(resources, self.record, logs)
        self.assertGreater(resources['metric_fit_fit_seconds_sum'], 0.)
        self.assertGreater(resources['train_only_loco_diagnostic_fit_seconds_sum'], 0.)
        self.assertTrue(all(value['evidence_scope'] == 'TRAIN_ONLY_NOT_OUTER_HELD_ACCURACY' for value in diagnostics))

    def test_actual_counts_identity_and_training_binding_mutations_rejected(self):
        for change in ('count', 'metric_ids', 'proxy_score', 'diagnostic_count', 'trace_target', 'loco_class_leak'):
            record = deepcopy(self.record)
            if change == 'count': record['metric_head_fit_count'] += 1
            if change == 'metric_ids': record['folds'][0]['metric_fit']['training_physical_ids'][0] = 'wrong-physical-id'
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_metric']['b_scores'][0][0] += 20.
            if change == 'diagnostic_count': record['folds'][0]['train_only_diagnostic']['diagnostic_fit_count'] -= 1
            if change == 'trace_target': record['folds'][0]['metric_stages'][0]['final_fit']['interaction_centered_trace'] += 1.
            if change == 'loco_class_leak':
                fold = record['folds'][0]['train_only_diagnostic']['folds'][0]
                fold['training_physical_ids'][0] = fold['held_physical_ids'][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'])

    def test_true_k1_remains_unmeasured(self):
        inputs = synthetic(k=1)
        record, split = as_record(probe.probe_within_class_metric(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], dict(oof=None, proxy=None), []))
        record['metric_fit_count'] = 1
        with self.assertRaisesRegex(ValueError, 'K1'):
            summary.verify_record(record, split, inputs['old_classes'])

    def test_incomplete_pilot_never_opens_score_trace(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                summary.summarize(spec=spec, output=str(Path(directory)/'out'))
            scores.assert_not_called()

    def test_six_correctness_transitions_partition_same_old_physical_samples(self):
        for value in self.record['oof']['paths'].values():
            fractions = [number for key, number in value['metrics'].items() if key.startswith('correctness_B_Cold_C_')]
            self.assertEqual(len(fractions), 6); self.assertAlmostEqual(sum(fractions), 1.)


if __name__ == '__main__': unittest.main()
