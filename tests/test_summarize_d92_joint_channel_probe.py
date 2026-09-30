"""Synthetic complete-only, nested isolation and objective-accounting checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools'), str(ROOT/'tests')]
import evaluate_d92_joint_channel_probe as probe
import summarize_d92_joint_channel_probe as summary
from test_evaluate_d92_joint_channel_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic()
        cls.record, cls.split = as_record(probe.probe_joint_channel(**cls.inputs), cls.inputs, 5)

    def test_paired_recomputation_nested_objectives_and_actual_costs(self):
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'])
        self.assertEqual(len(training), 24)
        self.assertEqual(sum(stage['optimizer_steps'] for stage in training), 72)
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(event['event'] == 'JOINT_CHANNEL_STEP' for event in events), 72)
        resources = {}; summary.accumulate_resources(resources, self.record, logs)
        self.assertGreater(resources['channel_preparation_prepare_seconds_sum'], 0.)
        self.assertNotIn('channel_preparation_preparation_seconds_sum', resources)
        self.assertGreater(resources['candidate_fit_maximum_persistent_state_bytes'], 5888)

    def test_inner_fold_leak_parameter_anchor_and_cost_mutations_rejected(self):
        for change in ('inner_leak', 'count', 'proximal', 'anchor', 'proxy_score', 'drop_event'):
            record = deepcopy(self.record); entry = record['folds'][0]
            if change == 'inner_leak':
                fold = entry['preparations'][0]['inner_folds'][0]
                fold['training_physical_ids'][0] = fold['held_physical_ids'][0]
            if change == 'count': record['inner_head_fit_count'] += 1
            if change == 'proximal': entry['candidate_stages'][0]['steps'][0]['loss_proximal'] += .1
            if change == 'anchor': entry['candidate_stages'][1]['anchor'] = [.1]+[0.]*735
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_channel_seq']['b_scores'][0][0] += 20.
            if change == 'drop_event': entry['training_events'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'])

    def test_true_k1_remains_unmeasured(self):
        inputs = synthetic(k=1)
        record, split = as_record(probe.probe_joint_channel(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], [], dict(oof=None, proxy=None), []))
        record['channel_preparation_count'] = 1
        with self.assertRaisesRegex(ValueError, 'K1'): summary.verify_record(record, split, inputs['old_classes'])

    def test_inner_margin_objective_does_not_add_head_training_ridge_loss(self):
        entry = self.record['folds'][0]; stage = entry['candidate_stages'][0]; prep = entry['preparations'][0]
        value = deepcopy(stage['steps'][0])
        value['inner_folds'][0]['head_training_loss_total'] = 1e12
        summary.verify_objective(value, value['u_pre'], stage['anchor'], prep)
        value['inner_folds'][0]['held_margin_loss_sum'] += 1.
        with self.assertRaisesRegex(ValueError, 'physical margin'):
            summary.verify_objective(value, value['u_pre'], stage['anchor'], prep)

    def test_incomplete_pilot_never_opens_outer_score_trace(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores:
            with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(spec=spec, output=str(Path(directory)/'out'))
            scores.assert_not_called()

    def test_six_correctness_transitions_partition_paired_old_samples(self):
        for value in self.record['oof']['paths'].values():
            fractions = [number for key, number in value['metrics'].items() if key.startswith('correctness_B_Cold_C_')]
            self.assertEqual(len(fractions), 6); self.assertAlmostEqual(sum(fractions), 1.)

    def test_independent_zero_sum_box_projection_conditions(self):
        summary.verify_box_projection([.01]*736, [0.]*736)
        proposal = [0.]*736; actual = [0.]*736
        proposal[0], proposal[1] = 2., -2.
        actual[0], actual[1] = summary.PARAMETER_BOUND, -summary.PARAMETER_BOUND
        summary.verify_box_projection(proposal, actual)
        wrong = [0.]*736; wrong[0], wrong[1] = .02, -.02
        with self.assertRaisesRegex(ValueError, 'KKT'): summary.verify_box_projection([0.]*736, wrong)
        with self.assertRaisesRegex(ValueError, 'constraint'): summary.verify_box_projection([0.]*736, [.01]*736)


if __name__ == '__main__': unittest.main()
