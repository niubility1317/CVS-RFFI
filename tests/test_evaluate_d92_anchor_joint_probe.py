"""Deterministic synthetic support boundaries; no target/query files are opened."""
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
import evaluate_d92_anchor_joint_probe as probe


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
    def test_true_k1_executes_closed_heads_without_holdout_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            result = probe.probe_anchor_joint(**synthetic(k=1), state_callback=archive)
            archive.finalize('COMPLETE')
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertEqual(result['sequence_paths'], 1); self.assertEqual(result['baseline_head_fit_count'], 2)
        self.assertEqual(result['ajlr_preparation_count'], 2); self.assertEqual(result['ajlr_stage_count'], 2)
        self.assertEqual(result['final_head_fit_count'], 2); self.assertEqual(result['optimizer_steps'], 0)
        self.assertEqual(result['final_score_evaluation_count'], 0)
        self.assertEqual(set(result['full_support']['paths']), {'R0', 'R_AJLR_seq'})
        for path in result['full_support']['paths'].values():
            self.assertEqual(path['metrics'], dict.fromkeys(probe.METRICS)); self.assertEqual(path['b_scores'], [])
        self.assertGreater(result['persistent_state_bytes'], 0)

    def test_archive_numeric_native_finite_exclusive_and_phase_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            ref = archive('{"state":"B_AJLR"}/initial', dict(U=np.arange(5888, dtype=float).reshape(736, 8), flag=np.asarray([True, False])))
            self.assertEqual(ref['arrays']['U']['nbytes'], 47104)
            with np.load(Path(directory)/ref['path'], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['U'], np.arange(5888, dtype=float).reshape(736, 8))
            self.assertEqual(json.loads(json.dumps(ref, allow_nan=False)), ref)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): archive('bad', dict(U=np.asarray([np.nan])))
            with self.assertRaisesRegex(ValueError, 'nonnumeric'): archive('bad', dict(ids=np.asarray(['x'])))
            self.assertEqual(archive.finalize('COMPLETE')['by_phase']['B_AJLR']['file_count'], 1)
            with self.assertRaises(FileExistsError): archive.finalize('COMPLETE')

    def test_n0_reuses_actual_b_and_proxy_k1_has_real_candidate_head(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            result = probe.probe_anchor_joint(**synthetic(new=False, zero=True), state_callback=archive)
            archive.finalize('COMPLETE')
        self.assertEqual(result['ajlr_preparation_count'], 6); self.assertEqual(result['ajlr_stage_count'], 6)
        self.assertEqual(result['prior_head_fit_count'], 0)
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertEqual([v['state'] for v in entry['candidate_stages']], ['B_AJLR'])
            for name in probe.PATHS: self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])
        for entry in result['oneshot_proxy']['trials']:
            self.assertEqual(entry['candidate_stages'][0]['final_head_fit_count'], 1)
            self.assertEqual(entry['candidate_stages'][0]['optimizer_steps'], 0)

    def test_streamed_full_compact_csv_text_npz_and_actual_costs(self):
        inputs = synthetic(k=1, six_old=True); arrays = {key: inputs[key] for key in probe.BRANCHES}
        split = dict(split_id='stream-parent', receiver='synthetic-rx', scenario='practical_high', k=1,
            support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'], support_labels=inputs['support_labels'])
        producer = dict(feature_array_bytes=sum(v.nbytes for v in arrays.values()), feature_file_bytes=0)
        config = dict(algorithm=probe.PROBE_CONFIG, producer_matrix={}, selection={})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            with patch.object(probe, 'read', return_value=dict(channel=probe.CHANNEL, scenarios=probe.SCENARIOS)), \
                 patch.object(probe, 'validate_selection'), \
                 patch.object(probe, 'load_support', return_value=(arrays, [], inputs['old_classes'], producer, {}, {})), \
                 patch.object(probe, 'selected_tasks', return_value=[(split, np.arange(len(inputs['support_labels'])), inputs['support_labels'])]), \
                 contextlib.redirect_stdout(io.StringIO()):
                marker = probe.evaluate(support_features='synthetic-cache', capsule='synthetic-capsule', output=out,
                    config=config, expected_capsule_id='synthetic-capsule', expected_checkpoint_sha256='a'*64, expected_model_seed=0)
            streams = {name: [json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines()]
                for name in ('fit_trace.jsonl', 'compact.jsonl', 'fit_stages.jsonl', 'training_events.jsonl', 'training_events_compact.jsonl')}
            record = streams['fit_trace.jsonl'][0]
            self.assertEqual(streams['compact.jsonl'], [probe.compact_record(record)])
            self.assertEqual(streams['training_events_compact.jsonl'], [probe.compact_event(v) for v in streams['training_events.jsonl']])
            self.assertEqual(record['head_fit_count'], sum(record[key] for key in ('baseline_head_fit_count', 'inner_head_fit_count', 'prior_head_fit_count', 'final_head_fit_count')))
            self.assertEqual(record['baseline_triangular_solve_count'], record['baseline_head_triangular_solve_count']+record['baseline_effective_df_triangular_solve_count'])
            from summarize_d92_anchor_joint_probe import StateResolver, verify_artifact_inventory
            resolver = StateResolver(out); resolver.verify_tree(record); resolver.finalize(); verify_artifact_inventory(out, marker)
            for name in ('compact.csv', 'fit_stages.csv', 'training_events_compact.csv'):
                with (out/name).open(encoding='utf-8', newline='') as stream: self.assertTrue(list(csv.DictReader(stream)))
            for line in (out/'training.log').read_text(encoding='utf-8').splitlines():
                json.loads(line.split(' ', 1)[1] if line.startswith(('STARTUP ', 'ANCHOR_JOINT_TRAINING ')) else line)
            self.assertEqual(json.loads((out/'probe_complete.json').read_text(encoding='utf-8')), marker)


if __name__ == '__main__': unittest.main()
