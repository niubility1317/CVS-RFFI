"""Synthetic-only physical splitting, identity sharing and accounting checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_within_class_metric_probe as probe


def synthetic(k=3, new=True, zero=False):
    classes = ['old-a', 'old-b']+(['new-z'] if new else [])
    labels = np.repeat(np.arange(len(classes)), k); rng = np.random.default_rng(3407)
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
    def test_true_k1_does_not_fit_or_diagnose(self):
        with patch.object(probe, 'fit_branch_local_ridge', side_effect=AssertionError('no fit')), \
             patch.object(probe, 'fit_within_class_metric', side_effect=AssertionError('no metric')):
            result = probe.probe_within_class_metric(**synthetic(k=1))
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertTrue(all(result[key] == 0 for key in probe.COUNTERS[4:]))

    def test_shared_metric_and_proxy_exact_identity_actual_counts(self):
        inputs = synthetic(); stages = []
        with patch.object(probe, 'fit_within_class_metric', wraps=probe.fit_within_class_metric) as fit, \
             patch.object(probe, 'diagnose_leave_one_class_out', wraps=probe.diagnose_leave_one_class_out) as loco:
            result = probe.probe_within_class_metric(**inputs, log_callback=stages.append)
        self.assertEqual(result['sequence_paths'], 6)
        self.assertEqual(result['baseline_head_fit_count'], 12)
        self.assertEqual(fit.call_count, result['metric_fit_count'])
        self.assertEqual(fit.call_count, 6)
        self.assertEqual(loco.call_count, 3)
        self.assertEqual(result['diagnostic_fit_count'], 6)
        self.assertEqual(result['metric_nonidentity_count'], 3)
        self.assertEqual(result['metric_head_fit_count'], 6)
        self.assertEqual(result['head_fit_count'], 18)
        self.assertEqual(result['optimizer_steps'], 0)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertFalse(set(entry['c_training_ids']) & set(entry['c_ids']))
            self.assertEqual(entry['metric_fit']['training_physical_ids'], entry['b_training_ids'])
            self.assertIsNone(entry['paths']['R_metric']['metrics']['A_old_accuracy'])
        for entry in result['oneshot_proxy']['trials']:
            self.assertEqual(entry['paths']['R0'], entry['paths']['R_metric'])
            self.assertEqual(entry['metric_stages'], [])
            self.assertIsNone(entry['train_only_diagnostic'])

    def test_n0_and_structural_identity_reuse(self):
        result = probe.probe_within_class_metric(**synthetic(new=False, zero=True))
        self.assertEqual(result['head_fit_count'], result['baseline_head_fit_count'])
        self.assertEqual(result['metric_head_fit_count'], 0)
        self.assertEqual(result['metric_nonidentity_count'], 0)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            for name in probe.PATHS:
                self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])
                self.assertEqual(entry['paths'][name]['metrics']['total_old_accuracy_drop'], 0.)
            self.assertEqual(entry['paths']['R0'], entry['paths']['R_metric'])

    def test_metric_head_success_then_score_failure_preserves_actual_counts(self):
        original = probe.fit_within_class_local_ridge
        class FailedScore:
            def __init__(self, state): self.state = state
            def audit_dict(self): return self.state.audit_dict()
            def score(self, **kwargs): raise FloatingPointError('synthetic held inference failure')
        def fail(*args, **kwargs): return FailedScore(original(*args, **kwargs))
        with patch.object(probe, 'fit_within_class_local_ridge', side_effect=fail):
            with self.assertRaises(FloatingPointError) as caught:
                probe.probe_within_class_metric(**synthetic())
        context = caught.exception.registration_context
        self.assertEqual(context['counters']['metric_head_fit_count'], 1)
        self.assertEqual(context['counters']['head_fit_count'], 3)
        self.assertIn('B', context['completed_head_audits'])
        self.assertIsNone(context['current_path']['metric_stages'][0]['score_seconds'])

    def test_oof_unequal_fold_pool_and_duplicate_rejection(self):
        labels = {f'a{i}': 'a' for i in range(5)} | {f'b{i}': 'b' for i in range(5)}; entries = []
        for fold in range(3):
            ids = [pid for pid in labels if int(pid[-1]) % 3 == fold]
            scores = [[1., 0.] if labels[pid] == 'a' else [0., 1.] for pid in ids]
            if fold == 2: scores = [row[::-1] for row in scores]
            entries.append(dict(b_ids=ids, c_ids=ids,
                paths={name: dict(b_scores=deepcopy(scores), c_scores=deepcopy(scores)) for name in probe.PATHS}))
        result = probe.pooled_assess(entries, labels, ['a', 'b'], ['a', 'b'])
        self.assertEqual(result['R_metric']['metrics']['B_old_accuracy'], .8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            probe.pooled_assess(entries+[entries[0]], labels, ['a', 'b'], ['a', 'b'])


if __name__ == '__main__': unittest.main()
