"""Synthetic-only integration checks; no experiment artifacts or target queries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_sequential_residual_probe as probe


def synthetic(k=2, new=True):
    classes = ['old-a', 'old-b']+(['new-z'] if new else [])
    labels = np.repeat(np.arange(len(classes)), k)
    rng = np.random.default_rng(3401)
    features = {key: rng.normal(size=(len(labels), width)) for key, width in
        (('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}
    return dict(features, support_labels=labels, support_ids=[f'{classes[y]}-{i % k}' for i, y in enumerate(labels)],
        classes=classes, old_classes=['old-a', 'old-b'])


def as_record(result, inputs, k):
    split = dict(split_id='synthetic-parent', receiver='synthetic-rx', scenario='practical_high', k=k,
        support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'],
        support_labels=inputs['support_labels'].tolist())
    return dict(result, **probe.split_identity(split, inputs['old_classes']), scope=probe.SCOPE,
                query_rows_used=0, source_rows_used=0), split


class ProbeTests(unittest.TestCase):
    def test_real_k1_has_no_fit_no_held_result(self):
        with patch.object(probe, 'fit_branch_local_ridge', side_effect=AssertionError('K1 fit forbidden')):
            result = probe.probe_sequential_residual(**synthetic(k=1))
        self.assertIsNone(result['oof'])
        self.assertIsNone(result['oneshot_proxy'])
        for key in ('head_fit_count', 'residual_training_stages', 'optimizer_steps', 'sequence_paths'):
            self.assertEqual(result[key], 0)

    def test_shared_b_base_and_c_preparation_actual_counts(self):
        inputs = synthetic()
        stages, steps = [], []
        with patch.object(probe, 'prepare_residual_training', wraps=probe.prepare_residual_training) as prepared:
            result = probe.probe_sequential_residual(**inputs, log_callback=stages.append, step_callback=steps.append)
        self.assertEqual(result['sequence_paths'], 4)
        self.assertEqual(result['head_fit_count'], 8)
        self.assertEqual(prepared.call_count, 8)
        self.assertEqual(result['residual_training_stages'], 12)
        self.assertEqual(result['optimizer_steps'], len(steps))
        self.assertEqual(len(steps), 12*64)
        self.assertEqual(len(stages), 20)
        for path in result['folds']+result['oneshot_proxy']['trials']:
            self.assertFalse(set(path['c_training_ids']) & set(path['c_ids']))
            self.assertEqual(path['paths']['R_reset']['b_scores'], path['paths']['R_seq']['b_scores'])
            self.assertEqual(path['residual_stages'][1]['q'], path['residual_stages'][2]['q'])
            self.assertEqual(path['residual_stages'][2]['actual_prepare_seconds'], 0.)
            self.assertIsNone(path['paths']['R_seq']['metrics']['A_old_accuracy'])
            self.assertIsNone(path['paths']['R_seq']['metrics']['adaptation_gain_B_minus_A'])

    def test_n0_reuses_b_bitwise_and_performs_no_c_training(self):
        result = probe.probe_sequential_residual(**synthetic(new=False))
        self.assertEqual(result['head_fit_count'], 4)
        self.assertEqual(result['residual_training_stages'], 4)
        self.assertEqual(result['optimizer_steps'], 256)
        for path in result['folds']+result['oneshot_proxy']['trials']:
            self.assertEqual([s['state'] for s in path['residual_stages']], ['B'])
            for name in probe.PATHS:
                self.assertEqual(path['paths'][name]['b_scores'], path['paths'][name]['c_scores'])
                self.assertEqual(path['paths'][name]['metrics']['total_old_accuracy_drop'], 0.)
                self.assertIsNone(path['paths'][name]['metrics']['C_new_accuracy'])

    def test_successful_training_survives_held_inference_failure(self):
        original = probe.train_residual_head
        class FailedInference:
            def __init__(self, state):
                self.state = state
                for key in ('U', 'V', 'q', 'classes'): setattr(self, key, getattr(state, key))
            def audit_dict(self): return self.state.audit_dict()
            def score(self, **kwargs): raise FloatingPointError('Synthetic held score failure')
        def train(*args, **kwargs): return FailedInference(original(*args, **kwargs))
        with patch.object(probe, 'train_residual_head', side_effect=train):
            with self.assertRaises(FloatingPointError) as caught:
                probe.probe_sequential_residual(**synthetic())
        context = caught.exception.registration_context
        self.assertEqual(context['counters']['optimizer_steps'], 64)
        self.assertEqual(context['counters']['residual_training_stages'], 1)
        self.assertEqual(context['current_path']['residual_stages'][0]['optimizer_steps'], 64)
        self.assertEqual(np.asarray(context['completed_residual_states']['B']['U']).shape, (160, 8))
        self.assertIsNone(context['current_path']['residual_stages'][0]['score_seconds'])

    def test_oof_pool_is_physical_weighted_and_rejects_duplicates(self):
        labels = {f'a{i}': 'a' for i in range(5)} | {f'b{i}': 'b' for i in range(5)}
        entries = []
        for fold in range(3):
            ids = [pid for pid in labels if int(pid[-1]) % 3 == fold]
            scores = [[1., 0.] if labels[pid] == 'a' else [0., 1.] for pid in ids]
            if fold == 2: scores = [row[::-1] for row in scores]
            entries.append(dict(b_ids=ids, c_ids=ids,
                paths={name: dict(b_scores=deepcopy(scores), c_scores=deepcopy(scores)) for name in probe.PATHS}))
        result = probe.pooled_assess(entries, labels, ['a', 'b'], ['a', 'b'])
        self.assertEqual(result['R0']['metrics']['B_old_accuracy'], .8)
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'):
            probe.pooled_assess(entries+[entries[0]], labels, ['a', 'b'], ['a', 'b'])


if __name__ == '__main__': unittest.main()
