"""Independent FCR8 coordinate, teacher, risk and complete-only checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools'), str(ROOT/'tests')]
import evaluate_d92_fcr8_probe as probe
import summarize_d92_fcr8_probe as summary
from cvsrffi import d92_function_coordinate_residual8_local_ridge as core
from test_evaluate_d92_fcr8_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.inputs = synthetic()
        archive = probe.StateArchive(cls.directory.name)
        result = probe.probe_fcr8(**cls.inputs, state_callback=archive)
        cls.manifest = archive.finalize('COMPLETE')
        cls.record, cls.split = as_record(result, cls.inputs, 3)

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def resolver(self): return summary.StateResolver(self.directory.name)

    def test_complete_coordinate_solver_teacher_and_paired_score_recomputation(self):
        resolver = self.resolver()
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        self.assertEqual(len(training), self.record['fcr_stage_count'])
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(v['event'] == 'FCR_GRADIENT' for v in events), self.record['backward_evaluation_count'])
        self.assertEqual(sum(v['event'] == 'FCR_TRIAL' for v in events), self.record['trial_count'])
        self.assertEqual(sum(v['event'] == 'FCR_STEP' for v in events), self.record['optimizer_steps'])
        for entry in self.record['folds']:
            b, seq, reset = entry['candidate_stages']
            self.assertEqual(seq['preparation']['teacher_folds'], reset['preparation']['teacher_folds'])
            initial = resolver(seq['initialization_state_ref']); final_b = resolver(b['final_state_ref'])
            np.testing.assert_array_equal(initial['anchor_U'], final_b['U'])
            np.testing.assert_array_equal(initial['U'], final_b['U'])
            np.testing.assert_array_equal(initial['Z'], np.zeros_like(initial['Z']))
            self.assertNotIn('V', initial)
        costs = {}; summary.accumulate_resources(costs, self.record, logs)
        self.assertGreater(costs['fcr_preparation_prepare_seconds_sum'], 0.)
        self.assertIn('inner_task_adjoint_seconds_sum', costs); self.assertIn('inner_keep_adjoint_seconds_sum', costs)

    def test_teacher_binding_risk_keep_limit_counts_stops_and_event_tampering(self):
        for change in ('teacher_leak', 'class_keep', 'limit', 'cost', 'nonincrease', 'reject_reason', 'stop', 'drop_event', 'proxy_score', 'rank', 'proximal', 'coordinate_ids'):
            record = deepcopy(self.record); entry = record['folds'][0]; stage = entry['candidate_stages'][0]
            if change == 'teacher_leak': entry['preparations'][1]['teacher_folds'][0]['training_physical_ids'][0] = entry['c_ids'][0]
            if change == 'class_keep': stage['initial_objective']['class_keep_loss_means'][0] += .1
            if change == 'limit': stage['keep_limit'] += .1
            if change == 'cost': record['task_derivative_triangular_solve_count'] += 1
            if change == 'nonincrease': stage['trials'][0]['objective_nonincrease_pass'] = not stage['trials'][0]['objective_nonincrease_pass']
            if change == 'reject_reason': stage['trials'][0]['reject_reason'] = 'INVENTED'
            if change == 'stop': stage['stop_reason'] = 'INVENTED_CONVERGED'
            if change == 'drop_event': entry['training_events'].pop()
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_FCR8_seq']['b_scores'][0][0] += 20.
            if change == 'rank': entry['preparations'][0]['latent_rank'] += 1
            if change == 'proximal': stage['initial_objective']['loss_proximal'] += .1
            if change == 'coordinate_ids': entry['preparations'][0]['optimizer_coordinate_ids'][0] = entry['c_ids'][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'], self.resolver())

    def test_reference_inventory_and_npz_coordinate_tampering_rejected(self):
        ref = self.record['folds'][0]['candidate_stages'][0]['final_state_ref']
        for change in ('path', 'shape', 'norm'):
            bad = deepcopy(ref)
            if change == 'path': bad['path'] = '../escaped.npz'
            if change == 'shape': bad['arrays']['U']['shape'] = [5888]
            if change == 'norm': bad['array_summaries']['U']['norm'] += 1.
            with self.subTest(change=change), self.assertRaises(ValueError): self.resolver()(bad)
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory); values = dict(U=np.zeros((736, 8)), V=summary.dct_initial())
            ref = archive('final', values); archive.finalize('COMPLETE')
            # Rewrite a synthetic archive and update only its physical byte inventory.
            with (Path(directory)/ref['path']).open('wb') as stream: np.savez_compressed(stream, U=np.ones((736, 8)), V=values['V'])
            manifest_path = Path(directory)/'state_manifest.json'; manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            size = (Path(directory)/ref['path']).stat().st_size; manifest['files'][0]['file_bytes'] = size
            manifest['total_file_bytes'] = size; manifest['by_phase']['unscoped']['file_bytes'] = size
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'norm summary'): summary.StateResolver(directory)(manifest['files'][0])

    def test_rank_two_whitening_and_exact_original_anchor_mapping(self):
        H = np.zeros((4, 8)); H[0, 0] = 4.; H[1, 1] = 2.
        W = np.zeros((8, 2)); W[0, 0] = .5; W[1, 1] = 1.
        spectrum = np.asarray([2., 1., 0., 0., 0., 0., 0., 0.])
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            ref = archive('coordinates', dict(H=H, W=W, singular_values=spectrum)); archive.finalize('COMPLETE')
            prep = dict(coordinate_state_ref=ref, train_physical_count=4, latent_rank=2, rank_estimated=True,
                rank_energy_threshold=128*sys.float_info.epsilon*8, latent_svd_count=1, singular_values=spectrum.tolist(),
                whitening_residual=0., whitening_tolerance=128*sys.float_info.epsilon*8*2,
                dictionary_rms=np.linalg.norm(H)/2, dictionary_physical_evaluation_count=4,
                optimizer_coordinate_ids=['a', 'b', 'c', 'd'], training_physical_ids=['a', 'b', 'c', 'd'],
                optimizer_coordinate_scope='ALL_CURRENT_OUTER_TRAIN_UNLABELLED', fixed_dictionary_trainable=False)
            summary.verify_latent_coordinates(prep, summary.StateResolver(directory))
            bad = deepcopy(prep); bad['singular_values'][0] += .1
            with self.assertRaisesRegex(ValueError, 'spectrum'): summary.verify_latent_coordinates(bad, summary.StateResolver(directory))
        anchor = np.arange(5888, dtype=float).reshape(736, 8)
        Z = np.zeros((736, 2)); summary.verify_coordinate_map(Z, anchor.copy(), anchor, W)
        Z[0, 0] = .25; U = anchor+Z@W.T
        summary.verify_coordinate_map(Z, U, anchor, W)
        U[0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'reconstruction'): summary.verify_coordinate_map(Z, U, anchor, W)

    def test_all_geometry_degenerate_preserves_anchor_and_actual_preparation_cost(self):
        inputs = synthetic(zero=True)
        with tempfile.TemporaryDirectory() as directory:
            archive = probe.StateArchive(directory)
            record, split = as_record(probe.probe_fcr8(**inputs, state_callback=archive), inputs, 3)
            archive.finalize('COMPLETE')
            summary.verify_record(record, split, inputs['old_classes'], summary.StateResolver(directory))
        self.assertEqual(record['latent_svd_count'], 0)
        self.assertGreater(record['dictionary_physical_evaluation_count'], 0)
        self.assertEqual(record['optimizer_steps'], 0)
        for entry in record['folds']:
            self.assertTrue(all(p['no_information_reason'] == 'ALL_INNER_GEOMETRY_DEGENERATE' for p in entry['preparations']))

    def test_mixed_zero_bandwidth_keeps_explicit_unmeasured_adapted_distance(self):
        inputs = synthetic(k=3, new=False)
        for name in probe.BRANCHES:
            first, different = inputs[name][0].copy(), inputs[name][5].copy()
            inputs[name][:] = first
            inputs[name][5] = different
        positions = np.asarray([1, 2, 4, 5])
        selected = {name: inputs[name][positions] for name in probe.BRANCHES}
        prepared = core.prepare_function_coordinate_training(**selected,
            support_labels=inputs['support_labels'][positions],
            support_ids=inputs['support_ids'][positions], classes=inputs['classes'], old_classes=inputs['old_classes'])
        prep = prepared.audit_dict()
        self.assertFalse(prep['no_information'])
        self.assertEqual(sum(p['original_bandwidth_tau'] == 0 for p in prep['inner_folds']), 1)
        anchor = np.zeros((736, 8)); Z = np.zeros((736, prepared.W.shape[1]))
        _, _, _, _, initial = core.evaluate_function_coordinate_objective(prepared, Z, anchor, gradient=False)
        Z[0, 0] = 1e-4
        U = core.reconstruct_U(Z, anchor, prepared.W)
        _, _, _, objective, cache = core.evaluate_function_coordinate_objective(prepared, Z, anchor, gradient=False)
        core._annotate_kernel_changes(objective, cache, initial)
        summary.verify_objective(objective, Z, U, anchor, prep)
        zero = next(p for p in objective['inner_folds'] if p['original_bandwidth_tau'] == 0)
        self.assertIsNone(zero['adapted_distance_relative_change'])
        self.assertEqual(zero['adapted_distance_unmeasured_reason'], 'ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE')
        for change in ('wrong_reason', 'positive_fold_missing', 'zero_U_missing'):
            bad = deepcopy(objective)
            if change == 'wrong_reason':
                next(p for p in bad['inner_folds'] if p['original_bandwidth_tau'] == 0)['adapted_distance_unmeasured_reason'] = None
            if change == 'positive_fold_missing':
                next(p for p in bad['inner_folds'] if p['original_bandwidth_tau'] > 0)['adapted_distance_relative_change'] = None
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'bypass'):
                summary.verify_objective(bad, Z, np.zeros_like(U) if change == 'zero_U_missing' else U, anchor, prep)

    def test_true_k1_numerical_only_without_archive(self):
        inputs = synthetic(k=1); record, split = as_record(probe.probe_fcr8(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], [], dict(oof=None, proxy=None), []))
        record['fcr_stage_count'] = 1
        with self.assertRaisesRegex(ValueError, 'K1'): summary.verify_record(record, split, inputs['old_classes'])

    def test_incomplete_pilot_does_not_open_held_scores_or_archives(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores, \
             patch.object(summary, 'StateResolver') as archives:
            with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(spec=spec, output=Path(directory)/'out')
            scores.assert_not_called(); archives.assert_not_called()


if __name__ == '__main__': unittest.main()
