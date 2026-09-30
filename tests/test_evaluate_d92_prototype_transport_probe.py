"""Synthetic physical-support integration tests; no production cache access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_prototype_transport_probe as probe


def synthetic(k=3, new=True, zero=False):
    classes = ['old-a', 'old-b']+(['new-z'] if new else [])
    labels = np.repeat(np.arange(len(classes)), k); rng = np.random.default_rng(3411)
    features = {key: np.zeros((len(labels), width)) if zero else rng.normal(size=(len(labels), width))
        for key, width in (('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}
    return dict(features, support_labels=labels, support_ids=[f'{classes[y]}-{i % k}' for i, y in enumerate(labels)],
        classes=classes, old_classes=['old-a', 'old-b'])


def as_record(result, inputs, k):
    split = dict(split_id='synthetic-parent', receiver='synthetic-rx', scenario='practical_high', k=k,
        support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'],
        support_labels=inputs['support_labels'].tolist())
    return dict(result, **probe.split_identity(split, inputs['old_classes']), scope=probe.SCOPE,
                query_rows_used=0, source_rows_used=0), split


class ProbeTests(unittest.TestCase):
    def test_true_k1_never_prepares_or_fits(self):
        with patch.object(probe, 'fit_branch_local_ridge', side_effect=AssertionError('no fit')), \
             patch.object(probe, 'prepare_prototype_transport_training', side_effect=AssertionError('no preparation')):
            result = probe.probe_prototype_transport(**synthetic(k=1))
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertTrue(all(result[key] == 0 for key in probe.COUNTERS[4:]))

    def test_three_paths_shared_b_preparation_and_actual_costs(self):
        inputs = synthetic(); events = []
        with patch.object(probe, 'prepare_prototype_transport_training', wraps=probe.prepare_prototype_transport_training) as prepare:
            result = probe.probe_prototype_transport(**inputs, event_callback=events.append)
        self.assertEqual(result['sequence_paths'], 6)
        self.assertEqual(prepare.call_count, result['transport_preparation_count'])
        self.assertEqual(result['transport_stage_count'], 18)
        self.assertEqual(result['diagnostic_fit_count'], 0)
        self.assertEqual(result['head_fit_count'], result['baseline_head_fit_count']+
            result['inner_head_fit_count']+result['final_head_fit_count'])
        self.assertLessEqual(result['inner_objective_evaluation_count'], 13*9)
        self.assertLessEqual(result['derivative_triangular_solve_count'], 24*9)
        trials = [event for event in events if event['event'] == 'TRANSPORT_TRIAL']
        self.assertTrue(trials)
        for event in trials:
            self.assertIn(event['trial'], (1, 2, 3))
            self.assertIsNone(event['outer_trial'])
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertFalse(set(entry['c_training_ids']) & set(entry['c_ids']))
            self.assertEqual(entry['paths']['R_transport_seq']['b_scores'], entry['paths']['R_transport_reset']['b_scores'])
            self.assertIsNone(entry['paths']['R_transport_seq']['metrics']['A_old_accuracy'])
            states = {stage['state']: stage for stage in entry['candidate_stages']}
            self.assertEqual(states['C_transport']['anchor'], states['B_transport']['u'])
            self.assertEqual(states['C_reset']['anchor'], [0.]*10)
            self.assertEqual(states['C_transport']['preparation'], states['C_reset']['preparation'])
        for entry in result['oneshot_proxy']['trials']:
            self.assertTrue(all(entry['paths'][name] == entry['paths']['R0'] for name in probe.PATHS))
            self.assertTrue(all(stage['optimizer_steps'] == 0 for stage in entry['candidate_stages']))

    def test_n0_reuses_b_with_no_extra_c_fit(self):
        result = probe.probe_prototype_transport(**synthetic(new=False, zero=True))
        self.assertEqual(result['transport_preparation_count'], 6)
        self.assertEqual(result['transport_stage_count'], 6)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertEqual([stage['state'] for stage in entry['candidate_stages']], ['B_transport'])
            for name in probe.PATHS:
                self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])
                self.assertEqual(entry['paths'][name]['metrics']['total_old_accuracy_drop'], 0.)

    def test_completed_training_preserved_on_held_score_failure(self):
        original = probe.fit_prototype_transport_local_ridge
        class FailedScore:
            def __init__(self, state): self.state = state; self.u = state.u
            def audit_dict(self): return self.state.audit_dict()
            def score(self, **kwargs): raise FloatingPointError('synthetic held score failure')
        with patch.object(probe, 'fit_prototype_transport_local_ridge',
                          side_effect=lambda *a, **kw: FailedScore(original(*a, **kw))):
            with self.assertRaises(FloatingPointError) as caught:
                probe.probe_prototype_transport(**synthetic(zero=True))
        context = caught.exception.registration_context
        self.assertEqual(context['counters']['transport_stage_count'], 1)
        self.assertIn('B_transport', context['completed_candidate_states'])
        self.assertIsNone(context['current_path']['candidate_stages'][0]['score_seconds'])

    def test_compact_logs_separate_beta_eta_and_retained_training_metrics(self):
        value = probe.compact_event(dict(u=[0.]*10, gradient=[.1]*5+[0.]*5, inner_folds=[
            dict(held_physical_count=2, held_training_correct_count=2),
            dict(held_physical_count=1, held_training_correct_count=0)]))
        self.assertEqual(value['gradient_summary']['blocks']['eta']['norm'], 0.)
        self.assertAlmostEqual(value['gradient_summary']['blocks']['beta']['sum'], .5)
        self.assertEqual(value['inner_training_accuracy'], 2/3)
        self.assertNotIn('gradient', value)


if __name__ == '__main__': unittest.main()
