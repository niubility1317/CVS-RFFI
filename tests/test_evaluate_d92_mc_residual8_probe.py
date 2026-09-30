"""Synthetic support/serialization boundaries; never open target/query data."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_mc_residual8_probe as probe


def synthetic(k=3, new=True, zero=False, six_old=False):
    old = [f'old-{i}' for i in range(6 if six_old else 2)]
    classes = np.asarray(old+(['new-0', 'new-1'] if new else []), dtype=np.str_)
    labels = np.repeat(np.arange(len(classes), dtype=np.int64), k)
    rng = np.random.default_rng(7351)
    arrays = {key: np.zeros((len(labels), width)) if zero else rng.normal(size=(len(labels), width))
              for key, width in (('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}
    return dict(arrays, support_labels=labels,
        support_ids=np.asarray([f'{classes[y]}-{i % k}' for i, y in enumerate(labels)], dtype=np.str_),
        classes=classes, old_classes=np.asarray(old, dtype=np.str_))


def as_record(result, inputs, k):
    split = probe.json_native(dict(split_id='synthetic-parent', receiver='synthetic-rx', scenario='practical_high', k=k,
        support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'], support_labels=inputs['support_labels']))
    return probe.json_native(dict(result, **probe.split_identity(split, inputs['old_classes']), scope=probe.SCOPE,
        query_rows_used=0, source_rows_used=0)), split


class ProbeTests(unittest.TestCase):
    def test_true_k1_does_not_prepare_fit_or_archive(self):
        with patch.object(probe, 'fit_branch_local_ridge', side_effect=AssertionError('no fit')), \
             patch.object(probe, 'prepare_mc_residual8_training', side_effect=AssertionError('no prepare')):
            result = probe.probe_mc_residual8(**synthetic(k=1), state_callback=lambda *a: self.fail('no archive'))
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertTrue(all(result[k] == 0 for k in probe.COUNTERS[4:]))

    def test_archive_lossless_native_finite_exclusive_and_phase_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            values = dict(U=np.arange(5888, dtype=np.float64).reshape(736, 8), flag=np.asarray([True, False]))
            ref = archive('{"state":"B_MC"}/initial', values)
            self.assertEqual(ref['arrays']['U']['nbytes'], 47104)
            with np.load(Path(directory)/ref['path'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['U'], values['U'])
            self.assertEqual(json.loads(json.dumps(ref, allow_nan=False)), ref)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): archive('bad', dict(U=np.asarray([np.nan])))
            manifest = archive.finalize('COMPLETE')
            self.assertEqual(manifest['by_phase']['B_MC']['file_count'], 1)
            with self.assertRaises(FileExistsError): archive.finalize('COMPLETE')
            with self.assertRaises(FileExistsError): probe.StateArchive(directory)

    def test_n0_reuses_b_and_proxy_is_exact_r0_without_new_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            result = probe.probe_mc_residual8(**synthetic(new=False, zero=True), state_callback=archive)
            archive.finalize('COMPLETE')
        self.assertEqual(result['mc_preparation_count'], 6); self.assertEqual(result['mc_stage_count'], 6)
        self.assertEqual(result['teacher_head_fit_count'], 0)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertEqual([s['state'] for s in entry['candidate_stages']], ['B_MC'])
            for name in probe.PATHS: self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])
        for entry in result['oneshot_proxy']['trials']:
            self.assertTrue(all(entry['paths'][name] == entry['paths']['R0'] for name in probe.PATHS))

    def test_real_evaluator_writes_full_compact_text_csv_and_npz_with_numpy_ids(self):
        inputs = synthetic(six_old=True); arrays = {key: inputs[key] for key in probe.BRANCHES}; k = 3
        split = dict(split_id='stream-parent', receiver='synthetic-rx', scenario='practical_high', k=k,
            support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'], support_labels=inputs['support_labels'])
        producer = dict(feature_array_bytes=sum(v.nbytes for v in arrays.values()), feature_file_bytes=0)
        config = dict(algorithm=probe.PROBE_CONFIG, producer_matrix={}, selection={})
        original = probe.fit_mc_residual8_local_ridge
        class ScalarAudit:
            def __init__(self, state): self.state = state
            def __getattr__(self, key): return getattr(self.state, key)
            def audit_dict(self):
                result = self.state.audit_dict()
                result['identity_forward'] = np.bool_(result['identity_forward'])
                result['trainable_parameter_count'] = np.int64(result['trainable_parameter_count'])
                for trial in result['trials']: trial['accepted'] = np.bool_(trial['accepted'])
                return result
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            with patch.object(probe, 'read', return_value=dict(channel=probe.CHANNEL, scenarios=probe.SCENARIOS)), \
                 patch.object(probe, 'validate_selection'), \
                 patch.object(probe, 'load_support', return_value=(arrays, [], inputs['old_classes'], producer, {}, {})), \
                 patch.object(probe, 'selected_tasks', return_value=[(split, np.arange(len(inputs['support_labels'])), inputs['support_labels'])]), \
                 patch.object(probe, 'fit_mc_residual8_local_ridge', side_effect=lambda *a, **kw: ScalarAudit(original(*a, **kw))), \
                 contextlib.redirect_stdout(io.StringIO()):
                marker = probe.evaluate(support_features='synthetic-cache', capsule='synthetic-capsule', output=out,
                    config=config, expected_capsule_id='synthetic-capsule', expected_checkpoint_sha256='a'*64, expected_model_seed=0)
            streams = {name: [json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines()]
                       for name in ('fit_trace.jsonl', 'compact.jsonl', 'fit_stages.jsonl', 'training_events.jsonl', 'training_events_compact.jsonl')}
            record = streams['fit_trace.jsonl'][0]
            self.assertEqual(streams['compact.jsonl'], [probe.compact_record(record)])
            self.assertEqual(streams['training_events_compact.jsonl'], [probe.compact_event(v) for v in streams['training_events.jsonl']])
            stage = record['folds'][0]['candidate_stages'][0]
            self.assertIs(type(stage['identity_forward']), bool); self.assertIs(type(stage['trainable_parameter_count']), int)
            self.assertTrue(stage['trials']); self.assertIs(type(stage['trials'][0]['accepted']), bool)
            self.assertNotIn('U', stage); self.assertNotIn('V', stage)
            from summarize_d92_mc_residual8_probe import StateResolver, verify_artifact_inventory
            resolver = StateResolver(out); resolver.verify_tree(record); resolver.finalize(); verify_artifact_inventory(out, marker)
            self.assertGreater(marker['state_archive_file_count'], 0)
            self.assertEqual(record['head_fit_count'], record['baseline_head_fit_count']+record['inner_head_fit_count']+
                record['teacher_head_fit_count']+record['final_head_fit_count'])
            self.assertEqual(record['derivative_triangular_solve_count'], record['task_derivative_triangular_solve_count']+
                record['keep_derivative_triangular_solve_count'])
            for name in ('compact.csv', 'fit_stages.csv', 'training_events_compact.csv'):
                with (out/name).open(encoding='utf-8', newline='') as stream: self.assertTrue(list(csv.DictReader(stream)))
            for line in (out/'training.log').read_text(encoding='utf-8').splitlines():
                json.loads(line.split(' ', 1)[1] if line.startswith(('STARTUP ', 'MC_RESIDUAL8_TRAINING ')) else line)
            self.assertEqual(json.loads((out/'probe_complete.json').read_text(encoding='utf-8')), marker)


if __name__ == '__main__': unittest.main()
