"""Independent support-risk, trial, lineage and complete-only verification."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools'), str(ROOT/'tests')]
import evaluate_d92_prototype_transport_probe as probe
import summarize_d92_prototype_transport_probe as summary
from test_evaluate_d92_prototype_transport_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic()
        cls.record, cls.split = as_record(probe.probe_prototype_transport(**cls.inputs), cls.inputs, 3)

    def test_paired_scores_full_solver_events_and_actual_costs(self):
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'])
        self.assertEqual(len(training), self.record['transport_stage_count'])
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(e['event'] == 'TRANSPORT_STEP' for e in events), self.record['optimizer_steps'])
        self.assertEqual(sum(e['event'] == 'TRANSPORT_TRIAL' for e in events), self.record['trial_count'])
        self.assertEqual(sum(e['event'] == 'TRANSPORT_GRADIENT' for e in events), self.record['backward_evaluation_count'])
        resources = {}; summary.accumulate_resources(resources, self.record, logs)
        self.assertGreater(resources['transport_preparation_prepare_seconds_sum'], 0.)
        self.assertGreater(resources['candidate_fit_maximum_persistent_state_bytes'], 80)

    def test_binding_cost_stop_risk_anchor_proxy_and_event_tampering_rejected(self):
        for change in ('inner_leak', 'count', 'class_risk', 'anchor', 'proxy_score', 'drop_event', 'stop', 'prototype'):
            record = deepcopy(self.record); entry = record['folds'][0]; stage = entry['candidate_stages'][0]
            if change == 'inner_leak':
                fold = entry['preparations'][0]['inner_folds'][0]
                fold['training_physical_ids'][0] = fold['held_physical_ids'][0]
            if change == 'count': record['inner_head_fit_count'] += 1
            if change == 'class_risk': stage['initial_objective']['class_loss_values'][0] += .1
            if change == 'anchor': entry['candidate_stages'][1]['anchor'] = [.1]+[0.]*9
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_transport_seq']['b_scores'][0][0] += 20.
            if change == 'drop_event': entry['training_events'].pop()
            if change == 'stop': stage['stop_reason'] = 'INVENTED_CONVERGENCE'
            if change == 'prototype': entry['preparations'][1]['full_prototypes']['old_prototypes_bitwise_inherited'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'])

    def test_trial_armijo_bound_and_cache_cost_tampering_rejected(self):
        entries = [e for e in self.record['folds'] if any(s['trials'] for s in e['candidate_stages'])]
        self.assertTrue(entries, 'Deterministic nondegenerate fixture must exercise at least one trial')
        for change in ('armijo', 'cached_head'):
            record = deepcopy(self.record)
            stage = next(s for e in record['folds'] for s in e['candidate_stages'] if s['trials'])
            if change == 'armijo': stage['trials'][0]['armijo_rhs'] += 1.
            if change == 'cached_head': stage['inner_head_fit_count'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'])

    def test_class_rms_pools_each_class_across_unequal_folds(self):
        entry = self.record['folds'][0]; stage = entry['candidate_stages'][0]; prep = entry['preparations'][0]
        value = deepcopy(stage['initial_objective'])
        value['inner_folds'][0]['head_training_loss_total'] = 1e12
        summary.verify_objective(value, stage['anchor'], stage['anchor'], prep)
        value['inner_folds'][0]['class_margin_loss_sums'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'Pooled class loss'):
            summary.verify_objective(value, stage['anchor'], stage['anchor'], prep)

    def test_unequal_folds_pool_before_rms_instead_of_averaging_fold_risks(self):
        prep = dict(classes=['a', 'b'], train_physical_count=6, inner_folds=[])
        folds = []
        for index, (counts, losses) in enumerate((([1, 1], [0., 4.]), ([2, 2], [6., 0.]))):
            prepared = dict(training_physical_ids=[f'train-{index}'], held_physical_ids=[f'held-{index}'],
                            held_physical_count=sum(counts))
            prep['inner_folds'].append(prepared)
            folds.append(dict(prepared, class_margin_loss_sums=losses, class_physical_counts=counts,
                held_training_correct_count=0, normal_equation_residual=0., trace_relative_error=0., numerical_tolerance=1e-10,
                factorization_count=1, head_fit_count=1, transport_forward_evaluation_count=1,
                derivative_triangular_solve_count=0))
        risk = math.hypot(2., 4/3)/math.sqrt(2)
        value = dict(inner_folds=folds, classes=['a', 'b'], class_loss_values=[2., 4/3], class_held_counts=[3, 3],
            loss_data=risk, loss_proximal=0., loss_total=risk, loss_scope='CLASS_RMS_INNER_HELD_MARGIN_PLUS_PROXIMAL',
            inner_head_fit_count=2, inner_factorization_count=2, transport_forward_evaluation_count=2,
            derivative_triangular_solve_count=0, inner_objective_evaluation_count=1, forward_cache_reused=False)
        summary.verify_objective(value, [0.]*10, [0.]*10, prep)
        value['loss_data'] = value['loss_total'] = (math.hypot(0., 4.)+math.hypot(3., 0.))/(2*math.sqrt(2))
        with self.assertRaisesRegex(ValueError, 'Class RMS'):
            summary.verify_objective(value, [0.]*10, [0.]*10, prep)

    def test_true_k1_remains_numerical_only(self):
        inputs = synthetic(k=1)
        record, split = as_record(probe.probe_prototype_transport(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], [], dict(oof=None, proxy=None), []))
        record['transport_preparation_count'] = 1
        with self.assertRaisesRegex(ValueError, 'K1'): summary.verify_record(record, split, inputs['old_classes'])

    def test_incomplete_pilot_never_opens_outer_scores(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores:
            with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(spec=spec, output=str(Path(directory)/'out'))
            scores.assert_not_called()

    def test_independent_ten_parameter_projection_kkt(self):
        summary.verify_box_projection([.1]*5+[.01]*5, [.1]*5+[0.]*5)
        proposal = [0.]*5+[2., -2., 0., 0., 0.]
        actual = [0.]*5+[summary.PARAMETER_BOUND, -summary.PARAMETER_BOUND, 0., 0., 0.]
        summary.verify_box_projection(proposal, actual)
        with self.assertRaisesRegex(ValueError, 'KKT'):
            summary.verify_box_projection([0.]*10, [0.]*5+[.02, -.02, 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, 'constraint'):
            summary.verify_box_projection([0.]*10, [0.]*5+[.01]*5)


if __name__ == '__main__': unittest.main()
