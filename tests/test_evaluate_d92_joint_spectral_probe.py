"""Synthetic support-only integration tests for all four fixed paths."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_joint_spectral_probe as probe


def synthetic(k=5, new=True, zero=False):
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
    def test_true_k1_never_fits(self):
        with patch.object(probe, 'fit_branch_local_ridge', side_effect=AssertionError('no fit')), \
             patch.object(probe, 'prepare_joint_training', side_effect=AssertionError('no preparation')):
            result = probe.probe_joint_spectral(**synthetic(k=1))
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertTrue(all(result[key] == 0 for key in probe.COUNTERS[4:]))

    def test_shared_preparation_b_and_fixed_budget_with_exact_proxy(self):
        inputs = synthetic(); events = []
        with patch.object(probe, 'prepare_joint_training', wraps=probe.prepare_joint_training) as prepare:
            result = probe.probe_joint_spectral(**inputs, event_callback=events.append)
        self.assertEqual(result['sequence_paths'], 8)
        self.assertEqual(result['baseline_head_fit_count'], 16)
        self.assertEqual(prepare.call_count, result['joint_preparation_count'])
        self.assertEqual(prepare.call_count, 16)
        self.assertEqual(result['joint_stage_count'], 24)
        self.assertEqual(result['fixed_stage_count'], 16)
        self.assertEqual(result['trained_joint_stage_count'], 9)
        self.assertEqual(result['optimizer_steps'], 72)
        self.assertEqual(result['inner_objective_evaluation_count'], 81)
        self.assertEqual(result['inner_head_fit_count'], 243)
        self.assertEqual(result['geometry_fit_count'], 26)
        self.assertEqual(result['diagnostic_fit_count'], 0)
        self.assertEqual(result['head_fit_count'], result['baseline_head_fit_count']+result['inner_head_fit_count']+result['final_head_fit_count'])
        self.assertEqual(sum(event['event'] == 'JOINT_SPECTRAL_STEP' for event in events), 72)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertFalse(set(entry['c_training_ids']) & set(entry['c_ids']))
            self.assertEqual(entry['paths']['R_joint']['b_scores'], entry['paths']['R_reset']['b_scores'])
            self.assertIsNone(entry['paths']['R_joint']['metrics']['A_old_accuracy'])
            states = {stage['state']: stage for stage in entry['candidate_stages']}
            self.assertEqual(states['C_joint']['theta_anchor'], states['B_joint']['theta'])
            self.assertEqual(states['C_reset']['theta_anchor'], [0., 0.])
            self.assertEqual(states['C_joint']['preparation'], states['C_reset']['preparation'])
        for entry in result['oneshot_proxy']['trials']:
            self.assertTrue(all(entry['paths'][name] == entry['paths']['R0'] for name in probe.PATHS))
            self.assertTrue(all(stage['optimizer_steps'] == 0 for stage in entry['candidate_stages']))

    def test_n0_reuses_corresponding_b_states_and_no_c_training(self):
        result = probe.probe_joint_spectral(**synthetic(k=3, new=False, zero=True))
        self.assertEqual(result['joint_preparation_count'], 6)
        self.assertEqual(result['joint_stage_count'], 6)
        self.assertEqual(result['fixed_stage_count'], 6)
        self.assertEqual(result['optimizer_steps'], 0)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertEqual([stage['state'] for stage in entry['candidate_stages']], ['B_joint', 'B_fixed'])
            for name in probe.PATHS:
                self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])
                self.assertEqual(entry['paths'][name]['metrics']['total_old_accuracy_drop'], 0.)

    def test_train_k2_joint_no_information_does_not_disable_fixed_ablation(self):
        result = probe.probe_joint_spectral(**synthetic(k=3))
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertEqual(result['inner_objective_evaluation_count'], 0)
        for entry in result['folds']:
            states = {stage['state']: stage for stage in entry['candidate_stages']}
            self.assertTrue(states['B_joint']['no_information'])
            self.assertEqual(states['B_joint']['theta'], [0., 0.])
            self.assertEqual(states['B_fixed']['theta'], [1., 0.])
            self.assertFalse(states['B_fixed']['identity_forward'])

    def test_completed_training_survives_outer_score_failure(self):
        original = probe.fit_joint_spectral_local_ridge
        class FailedScore:
            def __init__(self, state): self.state = state; self.theta = state.theta
            def audit_dict(self): return self.state.audit_dict()
            def score(self, **kwargs): raise FloatingPointError('synthetic outer score failure')
        def fail(*args, **kwargs): return FailedScore(original(*args, **kwargs))
        with patch.object(probe, 'fit_joint_spectral_local_ridge', side_effect=fail):
            with self.assertRaises(FloatingPointError) as caught: probe.probe_joint_spectral(**synthetic())
        context = caught.exception.registration_context
        self.assertEqual(context['counters']['optimizer_steps'], 8)
        self.assertEqual(context['counters']['trained_joint_stage_count'], 1)
        self.assertIn('B_joint', context['completed_candidate_states'])
        self.assertIsNone(context['current_path']['candidate_stages'][0]['score_seconds'])

    def test_compact_events_keep_actual_parameter_gradient_and_training_accuracy(self):
        value = probe.compact_event(dict(theta_pre=[0., .2], gradient=[.1, -.3], inner_folds=[
            dict(held_physical_count=2, held_training_correct_count=2),
            dict(held_physical_count=1, held_training_correct_count=0)]))
        self.assertEqual(value['theta_pre_1'], .2)
        self.assertEqual(value['gradient_1'], -.3)
        self.assertEqual(value['inner_training_accuracy'], 2/3)
        self.assertNotIn('inner_folds', value)

    def test_oof_unequal_folds_are_physically_pooled(self):
        labels = {f'a{i}': 'a' for i in range(5)} | {f'b{i}': 'b' for i in range(5)}; entries = []
        for fold in range(3):
            ids = [pid for pid in labels if int(pid[-1]) % 3 == fold]
            scores = [[1., 0.] if labels[pid] == 'a' else [0., 1.] for pid in ids]
            if fold == 2: scores = [row[::-1] for row in scores]
            entries.append(dict(b_ids=ids, c_ids=ids,
                paths={name: dict(b_scores=deepcopy(scores), c_scores=deepcopy(scores)) for name in probe.PATHS}))
        value = probe.pooled_assess(entries, labels, ['a', 'b'], ['a', 'b'])
        self.assertEqual(value['R_joint']['metrics']['B_old_accuracy'], .8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            probe.pooled_assess(entries+[entries[0]], labels, ['a', 'b'], ['a', 'b'])


if __name__ == '__main__': unittest.main()
