"""Independent MC archive, teacher, risk, projection and complete-only checks."""
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
import evaluate_d92_mc_residual8_probe as probe
import summarize_d92_mc_residual8_probe as summary
from test_evaluate_d92_mc_residual8_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.inputs = synthetic()
        archive = probe.StateArchive(cls.directory.name)
        result = probe.probe_mc_residual8(**cls.inputs, state_callback=archive)
        cls.manifest = archive.finalize('COMPLETE')
        cls.record, cls.split = as_record(result, cls.inputs, 3)

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def resolver(self): return summary.StateResolver(self.directory.name)

    def test_complete_coordinate_solver_teacher_and_paired_score_recomputation(self):
        resolver = self.resolver()
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        self.assertEqual(len(training), self.record['mc_stage_count'])
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(v['event'] == 'MC_GRADIENT' for v in events), self.record['backward_evaluation_count'])
        self.assertEqual(sum(v['event'] == 'MC_TRIAL' for v in events), self.record['trial_count'])
        self.assertEqual(sum(v['event'] == 'MC_STEP' for v in events), self.record['optimizer_steps'])
        for entry in self.record['folds']:
            b, seq, reset = entry['candidate_stages']
            self.assertEqual(seq['preparation']['teacher_folds'], reset['preparation']['teacher_folds'])
            initial = resolver(seq['initialization_state_ref']); final_b = resolver(b['final_state_ref'])
            np.testing.assert_array_equal(initial['anchor_U'], final_b['U'])
            np.testing.assert_array_equal(initial['anchor_V'], final_b['V'])
        costs = {}; summary.accumulate_resources(costs, self.record, logs)
        self.assertGreater(costs['mc_preparation_prepare_seconds_sum'], 0.)
        self.assertIn('inner_task_adjoint_seconds_sum', costs); self.assertIn('inner_keep_adjoint_seconds_sum', costs)

    def test_teacher_binding_risk_keep_limit_counts_stops_and_event_tampering(self):
        for change in ('teacher_leak', 'class_keep', 'limit', 'cost', 'nonincrease', 'reject_reason', 'stop', 'drop_event', 'proxy_score'):
            record = deepcopy(self.record); entry = record['folds'][0]; stage = entry['candidate_stages'][0]
            if change == 'teacher_leak': entry['preparations'][1]['teacher_folds'][0]['training_physical_ids'][0] = entry['c_ids'][0]
            if change == 'class_keep': stage['initial_objective']['class_keep_loss_means'][0] += .1
            if change == 'limit': stage['keep_limit'] += .1
            if change == 'cost': record['task_derivative_triangular_solve_count'] += 1
            if change == 'nonincrease': stage['trials'][0]['objective_nonincrease_pass'] = not stage['trials'][0]['objective_nonincrease_pass']
            if change == 'reject_reason': stage['trials'][0]['reject_reason'] = 'INVENTED'
            if change == 'stop': stage['stop_reason'] = 'INVENTED_CONVERGED'
            if change == 'drop_event': entry['training_events'].pop()
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_MC_seq']['b_scores'][0][0] += 20.
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

    def test_independent_product_ball_projection_and_constraints(self):
        V0 = summary.dct_initial(); U = np.zeros((736, 8)); U[0, 0] = 2.
        delta = np.zeros((8, 736)); delta[0, 0] = 3.
        summary.verify_ball_projection(U, V0+delta, U/2, V0+delta/3)
        with self.assertRaisesRegex(ValueError, 'projection'): summary.verify_ball_projection(U, V0+delta, U/3, V0+delta/3)
        with self.assertRaisesRegex(ValueError, 'constraint'): summary.verify_balls(U, V0)

    def test_true_k1_numerical_only_without_archive(self):
        inputs = synthetic(k=1); record, split = as_record(probe.probe_mc_residual8(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], [], dict(oof=None, proxy=None), []))
        record['mc_stage_count'] = 1
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
