"""Deterministic synthetic support boundaries; no target/query cache is opened."""
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
import evaluate_d92_affine_joint_probe as probe


def synthetic(k=2, new=True, zero=False, six_old=False):
    old = [f'old-{i}' for i in range(6 if six_old else 2)]
    classes = np.asarray(old+(['new-0', 'new-1'] if new else []), dtype=np.str_)
    labels = np.repeat(np.arange(len(classes), dtype=np.int64), k)
    rng = np.random.default_rng(7351)
    arrays = {key: np.zeros((len(labels), width)) if zero else rng.normal(size=(len(labels), width))
        for key, width in (('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}
    return dict(arrays, support_labels=labels,
        support_ids=np.asarray([f'{classes[y]}-{i % k}' for i, y in enumerate(labels)], dtype=np.str_),
        classes=classes, old_classes=np.asarray(old, dtype=np.str_))


def scope():
    return dict(run_id='synthetic-affine-run', row_id='synthetic-affine-row', split_id='synthetic-parent')


class ProbeTests(unittest.TestCase):
    def test_method_config_exactly_matches_new_core_and_fixed_structure(self):
        from cvsrffi.d92_affine_joint_local_ridge import FROZEN_CONFIG
        method = json.loads((ROOT/'configs/d92_affine_joint_frozen_20261001.json').read_text(encoding='utf-8'))
        self.assertEqual(method, FROZEN_CONFIG)
        self.assertEqual(probe.PROBE_CONFIG, FROZEN_CONFIG)
        self.assertEqual(method['schema'], probe.SCHEMA)
        self.assertEqual(method['method'], probe.METHOD)
        self.assertTrue(method['free_intercept'])
        self.assertEqual(method['rank'], 8)
        self.assertEqual((method['max_iterations'], method['max_trials']), (4, 12))
        self.assertFalse(method['parameter_search'])
        self.assertEqual(probe.PATHS, ('R0', 'R_AFFINE_seq'))
        self.assertEqual(probe.EXACT_COUNTS['episodes'], 160)
        self.assertEqual(probe.EXACT_COUNTS['sequence_paths'], 1800)

    def test_public_boundary_rejects_query_source_and_external_adapted_state(self):
        inputs = synthetic(k=1)
        for forbidden in ('query_features', 'query_labels', 'query_truth', 'query_roles',
                          'source_features', 'source_labels', 'inherited', 'checkpoint', 'teacher', 'target_statistics'):
            with self.subTest(forbidden=forbidden), self.assertRaises(TypeError):
                probe.probe_affine_joint(**inputs, context=scope(), **{forbidden: object()})
        with patch.object(probe, 'read') as read, patch.object(probe, 'load_support') as load:
            with self.assertRaisesRegex(ValueError, 'run/row'):
                probe.evaluate(support_features='unopened', capsule='unopened', output='synthetic-uncreated-output',
                    config=dict(algorithm=probe.PROBE_CONFIG, producer_matrix={}, selection={}),
                    expected_capsule_id='synthetic', expected_checkpoint_sha256='a'*64,
                    expected_model_seed=0, run_id='', row_id='synthetic-row')
            read.assert_not_called(); load.assert_not_called()

    def test_real_event_order_and_actual_b_inheritance_are_path_local(self):
        calls, events, states = [], [], []
        prepare, fit = probe.prepare_affine_joint_training, probe.fit_affine_joint_local_ridge

        def prepare_record(**kwargs):
            stage = kwargs['context']['stage']
            calls.append(('prepare', stage, dict(kwargs['context'])))
            if stage == 'C':
                self.assertIs(kwargs['inherited'], states[-1])
                self.assertEqual(kwargs['context']['run_id'], 'synthetic-affine-run')
                self.assertEqual(kwargs['context']['row_id'], 'synthetic-affine-row')
            else:
                self.assertIsNone(kwargs['inherited'])
            return prepare(**kwargs)

        def fit_record(prepared, **kwargs):
            mode = kwargs['mode']; calls.append(('fit', mode, {}))
            state = fit(prepared, **kwargs)
            if mode == 'C_seq':
                self.assertIs(state.prior, states[-1])
                np.testing.assert_array_equal(state.anchor_U, states[-1].U)
            states.append(state)
            return state

        with patch.object(probe, 'prepare_affine_joint_training', side_effect=prepare_record), \
             patch.object(probe, 'fit_affine_joint_local_ridge', side_effect=fit_record):
            result = probe.probe_affine_joint(**synthetic(k=2), context=scope(), event_callback=events.append)
        self.assertEqual(len(calls), 4*result['sequence_paths'])
        for begin in range(0, len(calls), 4):
            self.assertEqual([(kind, stage) for kind, stage, _ in calls[begin:begin+4]],
                [('prepare', 'B'), ('fit', 'B'), ('prepare', 'C'), ('fit', 'C_seq')])
            bctx, cctx = calls[begin][2], calls[begin+2][2]
            self.assertEqual({key: bctx[key] for key in bctx if key != 'stage'},
                             {key: cctx[key] for key in cctx if key != 'stage'})
        self.assertEqual(events, [event for entry in result['folds']+result['oneshot_proxy']['trials']
                                  for event in entry['training_events']])
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            sequence = []
            for event in entry['training_events']:
                if not sequence or sequence[-1] != event['state']: sequence.append(event['state'])
                self.assertIsNone(event['source_validation'])
                self.assertEqual(event['objective_scope'], 'INNER_SUPPORT_TRAINING_NOT_VALIDATION')
            self.assertEqual(sequence, ['B_prepare', 'B_AFFINE', 'C_prepare', 'C_AFFINE_seq'])
            self.assertEqual(set(entry['paths']), set(probe.PATHS))
        for path in result['oof']['paths'].values():
            self.assertIsNone(path['metrics']['A_old_accuracy'])
            self.assertIsNone(path['metrics']['adaptation_gain_B_minus_A'])

    def test_true_k1_has_affine_head_rhs_archive_and_no_adapter_update(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            result = probe.probe_affine_joint(**synthetic(k=1), context=scope(), state_callback=archive)
            manifest = archive.finalize('COMPLETE')
            self.assertEqual(manifest['schema'], 'd92_affine_joint_state_archive_v1')
            self.assertEqual(manifest['method'], probe.METHOD)
            for stage in result['full_support']['candidate_stages']:
                arrays = stage['final_state_ref']['arrays']
                classes = stage['class_count']; n = stage['train_physical_count']
                self.assertEqual(arrays['intercept']['shape'], [classes])
                self.assertEqual(arrays['intercept']['nbytes'], 8*classes)
                self.assertEqual(arrays['schur_z']['shape'], [n])
                self.assertEqual(arrays['schur_s']['shape'], [])
                self.assertEqual(arrays['combined_rhs']['shape'], [n, classes+1])
                self.assertEqual(stage['head_triangular_rhs_count'], 2*(classes+1))
                self.assertEqual(stage['head_triangular_rhs_element_count'], 2*n*(classes+1))
                self.assertEqual(stage['head_triangular_dense_work_unit_count'], 2*n*n*(classes+1))
                self.assertEqual(stage['intercept_fit_count'], 1)
                self.assertEqual(stage['final_fit']['analytic_intercept_parameter_count'], classes)
                self.assertEqual(stage['final_fit']['analytic_intercept_contrast_count'], classes-1)
                self.assertEqual(stage['trained_parameter_count'], 0)
                self.assertEqual(stage['optimizer_steps'], 0)
                with np.load(Path(directory)/stage['final_state_ref']['path'], allow_pickle=False) as saved:
                    np.testing.assert_allclose(saved['alpha'].sum(axis=0), 0., atol=1e-11, rtol=0.)
                    self.assertTrue(np.isfinite(saved['intercept']).all())
                    if stage['state'] == 'C_AFFINE_seq':
                        self.assertIn('prior_B_intercept', saved.files)
                        self.assertEqual(saved['prior_B_intercept'].shape, (result['old_class_count'],))
        self.assertIsNone(result['oof']); self.assertIsNone(result['oneshot_proxy'])
        self.assertEqual(result['schema'], probe.SCHEMA)
        self.assertEqual(result['sequence_paths'], 1)
        self.assertEqual(result['baseline_head_fit_count'], 2)
        self.assertEqual(result['ajlr_preparation_count'], 2)
        self.assertEqual(result['ajlr_stage_count'], 2)
        self.assertEqual(result['final_head_fit_count'], 2)
        self.assertEqual(result['final_score_evaluation_count'], 0)
        for value in result['full_support']['paths'].values():
            self.assertEqual(value['metrics'], dict.fromkeys(probe.METRICS))
            self.assertEqual(value['b_scores'], [])
        self.assertGreater(result['persistent_state_bytes'], 0)

    def test_new_zero_bypasses_all_c_preparation_fit_and_intercept_solves(self):
        prepare, fit = probe.prepare_affine_joint_training, probe.fit_affine_joint_local_ridge
        with patch.object(probe, 'prepare_affine_joint_training', wraps=prepare) as prep, \
             patch.object(probe, 'fit_affine_joint_local_ridge', wraps=fit) as fitting:
            result = probe.probe_affine_joint(**synthetic(k=2, new=False, zero=True), context=scope())
        self.assertEqual(prep.call_count, result['sequence_paths'])
        self.assertEqual(fitting.call_count, result['sequence_paths'])
        self.assertEqual(result['prior_head_fit_count'], 0)
        self.assertEqual(result['head_triangular_rhs_count'], 0)
        self.assertEqual(result['intercept_fit_count'], result['sequence_paths'])
        for entry in result['folds']+result['oneshot_proxy']['trials']:
            self.assertTrue(entry['c_reuses_b_candidates'])
            self.assertEqual([value['state'] for value in entry['candidate_stages']], ['B_AFFINE'])
            self.assertEqual([value['state'] for value in entry['preparations']], ['B'])
            for name in probe.PATHS:
                self.assertEqual(entry['paths'][name]['b_scores'], entry['paths'][name]['c_scores'])

    def test_archive_numeric_native_finite_exclusive_and_formula_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            ref = archive('{"state":"B_AFFINE"}/initial', dict(U=np.arange(5888, dtype=float).reshape(736, 8)))
            self.assertEqual(ref['arrays']['U']['nbytes'], 47104)
            with np.load(Path(directory)/ref['path'], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['U'], np.arange(5888, dtype=float).reshape(736, 8))
            self.assertEqual(json.loads(json.dumps(ref, allow_nan=False)), ref)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): archive('bad', dict(U=np.asarray([np.nan])))
            with self.assertRaisesRegex(ValueError, 'nonnumeric'): archive('bad', dict(ids=np.asarray(['x'])))
            manifest = archive.finalize('COMPLETE')
            self.assertEqual(manifest['by_phase']['B_AFFINE']['file_count'], 1)
            self.assertIn('analytic_intercept', manifest['prediction_formula'])
            with self.assertRaises(FileExistsError): archive.finalize('COMPLETE')

    def test_streamed_full_compact_csv_text_and_new_schemas_preserve_actual_costs(self):
        inputs = synthetic(k=1, six_old=True); arrays = {key: inputs[key] for key in probe.BRANCHES}
        split = dict(split_id='stream-parent', receiver='synthetic-rx', scenario='practical_high', k=1,
            support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'], support_labels=inputs['support_labels'])
        producer = dict(feature_array_bytes=sum(value.nbytes for value in arrays.values()), feature_file_bytes=0)
        config = dict(algorithm=probe.PROBE_CONFIG, producer_matrix={}, selection={})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            with patch.object(probe, 'read', return_value=dict(channel=probe.CHANNEL, scenarios=probe.SCENARIOS)), \
                 patch.object(probe, 'validate_selection'), \
                 patch.object(probe, 'load_support', return_value=(arrays, [], inputs['old_classes'], producer, {}, {})), \
                 patch.object(probe, 'selected_tasks', return_value=[(split, np.arange(len(inputs['support_labels'])), inputs['support_labels'])]), \
                 contextlib.redirect_stdout(io.StringIO()):
                marker = probe.evaluate(support_features='synthetic-cache', capsule='synthetic-capsule', output=out,
                    config=config, expected_capsule_id='synthetic-capsule', expected_checkpoint_sha256='a'*64,
                    expected_model_seed=0, run_id='synthetic-affine-run', row_id='synthetic-affine-row')
            streams = {name: [json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines()]
                for name in ('fit_trace.jsonl', 'compact.jsonl', 'fit_stages.jsonl', 'training_events.jsonl', 'training_events_compact.jsonl')}
            record = streams['fit_trace.jsonl'][0]
            self.assertEqual(streams['compact.jsonl'], [probe.compact_record(record)])
            self.assertEqual(streams['training_events_compact.jsonl'], [probe.compact_event(value) for value in streams['training_events.jsonl']])
            self.assertEqual(record['head_fit_count'], sum(record[key] for key in
                ('baseline_head_fit_count', 'inner_head_fit_count', 'prior_head_fit_count', 'final_head_fit_count')))
            for key in probe.AUDIT_COUNTERS:
                self.assertEqual(marker[key], record[key])
                self.assertEqual(streams['compact.jsonl'][0][key], record[key])
            for name in ('compact.csv', 'fit_stages.csv', 'training_events_compact.csv'):
                with (out/name).open(encoding='utf-8', newline='') as stream:
                    self.assertTrue(list(csv.DictReader(stream)))
            for line in (out/'training.log').read_text(encoding='utf-8').splitlines():
                json.loads(line.split(' ', 1)[1] if line.startswith(('STARTUP ', 'AFFINE_JOINT_TRAINING ')) else line)
            startup = json.loads((out/'startup.json').read_text(encoding='utf-8'))
            self.assertIsNone(startup['actual_A'])
            self.assertEqual((startup['query_rows_used'], startup['source_rows_used']), (0, 0))
            self.assertEqual(startup['run_id'], marker['run_id'])
            self.assertEqual(startup['row_id'], marker['row_id'])
            self.assertEqual(json.loads((out/'probe_complete.json').read_text(encoding='utf-8')), marker)
            inventory = json.loads((out/'artifact_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(inventory['schema'], 'd92_affine_joint_artifacts_v1')
            self.assertEqual(inventory['method'], probe.METHOD)
            for value in inventory['files']:
                self.assertEqual((out/value['path']).stat().st_size, value['file_bytes'])


if __name__ == '__main__': unittest.main()
