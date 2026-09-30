"""Independent synthetic AFFINE_JOINT certificate/score tampering checks."""
from copy import deepcopy
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
import summarize_d92_affine_joint_probe as summary
from test_evaluate_d92_affine_joint_probe import synthetic, scope


def as_record(result, inputs, k):
    split = probe.json_native(dict(split_id='synthetic-parent', receiver='synthetic-rx', scenario='practical_high', k=k,
        support_seed=0, registered_classes=inputs['classes'], support_ids=inputs['support_ids'], support_labels=inputs['support_labels']))
    return probe.json_native(dict(result, **probe.split_identity(split, inputs['old_classes']), scope=probe.SCOPE,
        query_rows_used=0, source_rows_used=0)), split


class ChangedArrays:
    """Change loaded numeric evidence while leaving archive receipts untouched."""
    def __init__(self, resolver, target, key, mutate):
        self.resolver, self.target, self.key, self.mutate = resolver, target, key, mutate
    def __call__(self, ref):
        arrays = self.resolver(ref)
        if ref['path'] != self.target['path']: return arrays
        arrays = dict(arrays); arrays[self.key] = np.array(arrays[self.key], copy=True)
        self.mutate(arrays[self.key]); return arrays
    def verify_tree(self, value): self.resolver.verify_tree(value)


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.inputs = synthetic(k=3)
        archive = probe.StateArchive(cls.directory.name)
        cls.fit_stages = Path(cls.directory.name)/'fit_stages.jsonl'
        with cls.fit_stages.open('x', encoding='utf-8') as stream:
            def log_callback(stage):
                row = dict(probe.compact_event(stage), split_id='synthetic-parent')
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            result = probe.probe_affine_joint(**cls.inputs, context=scope(), state_callback=archive, log_callback=log_callback)
        cls.record, cls.split = as_record(result, cls.inputs, 3)
        archive.finalize('COMPLETE')

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def resolver(self): return summary.StateResolver(self.directory.name)

    def test_actual_callback_stage_stream_matches_strict_reconstruction(self):
        resolver = self.resolver()
        logs, _, _, _ = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        emitted = list(summary.jsonlines(self.fit_stages))
        expected = [dict(probe.compact_event(value), split_id=self.record['split_id']) for value in logs]
        self.assertEqual(emitted, expected)
        per_path = [('BASE_FIT', 'B0'), ('BASE_FIT', 'C0'),
            ('AFFINE_PREPARATION', 'B'), ('CANDIDATE_FIT', 'B_AFFINE'),
            ('AFFINE_PREPARATION', 'C'), ('CANDIDATE_FIT', 'C_AFFINE_seq')]
        path_count = len(self.record['folds'])+len(self.record['oneshot_proxy']['trials'])
        self.assertEqual([(value['event'], value['state']) for value in emitted], per_path*path_count)

    def test_complete_prior_reference_ce_solver_scores_and_workload_recomputed(self):
        resolver = self.resolver()
        logs, events, metrics, training = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        resolver.finalize()
        self.assertEqual(len(training), self.record['ajlr_stage_count'])
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        self.assertEqual(sum(v['event'] == 'AFFINE_JOINT_GRADIENT' for v in events), self.record['backward_evaluation_count'])
        self.assertEqual(sum(v['event'] == 'AFFINE_JOINT_TRIAL' for v in events), self.record['trial_count'])
        self.assertEqual(sum(v['event'] == 'AFFINE_JOINT_STEP' for v in events), self.record['optimizer_steps'])
        for entry in self.record['folds']:
            b, c = entry['candidate_stages']
            initial = resolver(c['initialization_state_ref']); bfinal = resolver(b['final_state_ref'])
            np.testing.assert_array_equal(initial['U'], bfinal['U']); np.testing.assert_array_equal(initial['Z'], np.zeros_like(initial['Z']))
            self.assertIsNotNone(c['score_workload']); self.assertIn('prior', c['score_workload'])
        measured_reference_cases = set()
        for entry in self.record['folds']:
            for stage in entry['candidate_stages']:
                for objective in [v['objective'] for v in stage['trials']]:
                    for fold in objective['inner_folds']:
                        if fold['identity_forward']: continue
                        m = len(fold['old_reference_physical_ids']); n = fold['train_physical_count']; h = fold['held_physical_count']
                        expected_pairs = m*(m-1)//2+m*(n-m)+h*m
                        self.assertEqual(fold['reference_distance_pair_count'], expected_pairs)
                        measured_reference_cases.add('all_old' if m == n else 'old_subset')
        self.assertEqual(measured_reference_cases, {'all_old', 'old_subset'})
        resources = {}; summary.accumulate_resources(resources, self.record, logs)
        self.assertIn('outer_prior_raw_distance_evaluation_count_sum', resources)

    def test_prior_reference_ce_trial_cost_event_score_tampering_rejected(self):
        for change in ('prior_leak', 'reference', 'ce', 'cost', 'trial_acceptance', 'stop', 'drop_event', 'proxy_score', 'rank', 'A'):
            record = deepcopy(self.record); entry = record['folds'][0]; b, c = entry['candidate_stages']
            if change == 'prior_leak': entry['preparations'][1]['prior_folds'][0]['training_physical_ids'][0] = entry['c_ids'][0]
            if change == 'reference': entry['preparations'][1]['inner_folds'][0]['old_reference_physical_ids'][0] = entry['c_ids'][0]
            if change == 'ce': b['initial_objective']['class_ce_means'][0] += .1
            if change == 'cost': record['derivative_triangular_solve_count'] += 1
            if change == 'trial_acceptance': b['trials'][0]['objective_nonincrease_pass'] = not b['trials'][0]['objective_nonincrease_pass']
            if change == 'stop': b['stop_reason'] = 'INVENTED_CONVERGENCE'
            if change == 'drop_event': entry['training_events'].pop()
            if change == 'proxy_score': record['oneshot_proxy']['trials'][0]['paths']['R_AFFINE_seq']['c_scores'][0][0] += 20.
            if change == 'rank': entry['preparations'][0]['latent_rank'] += 1
            if change == 'A': entry['paths']['R_AFFINE_seq']['metrics']['A_old_accuracy'] = .5
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                summary.verify_record(record, self.split, self.inputs['old_classes'], self.resolver())

    def test_archive_path_shape_and_numeric_coordinate_changes_rejected(self):
        ref = self.record['folds'][0]['candidate_stages'][0]['final_state_ref']
        for change in ('path', 'shape', 'norm'):
            bad = deepcopy(ref)
            if change == 'path': bad['path'] = '../escaped.npz'
            if change == 'shape': bad['arrays']['U']['shape'] = [5888]
            if change == 'norm': bad['array_summaries']['U']['norm'] += 1.
            with self.subTest(change=change), self.assertRaises(ValueError): self.resolver()(bad)

    def test_intercept_schur_combined_rhs_and_complete_companion_tampering_rejected(self):
        stage = self.record['folds'][0]['candidate_stages'][0]
        final = stage['final_state_ref']; gradient = stage['gradients'][0]['state_ref']
        cases = [(final, 'intercept'), (final, 'schur_z'), (final, 'combined_rhs'),
            (gradient, 'fold_0_adjoint_g_b'), (gradient, 'fold_0_adjoint_eta'),
            (gradient, 'fold_0_adjoint_T'), (gradient, 'g_Z')]
        for ref, key in cases:
            def mutate(a): a.flat[0] += .1
            resolver = ChangedArrays(self.resolver(), ref, key, mutate)
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)

    def test_rhs_width_accounting_and_current_B_binding_tampering_rejected(self):
        for change in ('head_rhs', 'prior_rhs', 'derivative_rhs', 'intercept_add', 'run', 'row', 'fold', 'expected_run'):
            record = deepcopy(self.record); entry = record['folds'][0]
            if change == 'head_rhs': entry['candidate_stages'][0]['final_fit']['head_triangular_rhs_count'] -= 2
            if change == 'prior_rhs': entry['preparations'][1]['prior_triangular_rhs_element_count'] += 1
            if change == 'derivative_rhs': entry['candidate_stages'][0]['gradients'][0]['objective']['derivative_triangular_rhs_count'] += 1
            if change == 'intercept_add': entry['candidate_stages'][1]['score_workload']['prior']['intercept_addition_count'] -= 1
            if change == 'run': entry['preparations'][1]['run_id'] = 'other-run'
            if change == 'row': entry['candidate_stages'][1]['row_id'] = 'other-row'
            if change == 'fold': entry['candidate_stages'][1]['fold'] = 20
            binding = dict(run_id='other-run') if change == 'expected_run' else dict(run_id=scope()['run_id'], row_id=scope()['row_id'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'], self.resolver(), binding)

    def test_analytic_head_parameters_and_actual_training_vs_deployment_buffers(self):
        resolver = self.resolver()
        for entry in self.record['folds']:
            for stage in entry['candidate_stages']:
                data = resolver(stage['final_state_ref']); n, c = data['alpha'].shape
                self.assertEqual(data['schur_s'].shape, ())
                self.assertEqual(stage['analytic_head_parameter_count'], (n+1)*c)
                self.assertEqual(stage['analytic_intercept_parameter_count'], c)
                self.assertGreater(stage['persistent_state_bytes'], stage['deployment_numeric_state_bytes'])
                self.assertEqual(stage['final_fit']['head_triangular_rhs_count'], 2*(c+1))
        gradients = [resolver(g['state_ref']) for entry in self.record['folds'] for stage in entry['candidate_stages'] for g in stage['gradients']]
        self.assertTrue(any(np.linalg.norm(a['fold_0_adjoint_g_b']) > 1e-12 for a in gradients))

    def test_positive_equivalence_zero_bandwidth_and_new0_exactreuse(self):
        for new in (True, False):
            inputs = synthetic(k=3, new=new); nclasses = len(inputs['classes'])
            for key in probe.BRANCHES:
                pattern = inputs[key][:3].copy(); inputs[key] = np.tile(pattern, (nclasses, 1))
            with tempfile.TemporaryDirectory() as directory:
                archive = probe.StateArchive(directory)
                result = probe.probe_affine_joint(**inputs, context=scope(), state_callback=archive)
                record, split = as_record(result, inputs, 3); archive.finalize('COMPLETE')
                resolver = summary.StateResolver(directory)
                summary.verify_record(record, split, inputs['old_classes'], resolver); resolver.finalize()
                for entry in record['folds']:
                    for stage in entry['candidate_stages']:
                        final = resolver(stage['final_state_ref'])
                        self.assertEqual(summary._scalar(final, 'tau'), 0.)
                        self.assertGreater(summary._scalar(final, 'gamma'), 0.)
                        self.assertEqual(stage['derivative_triangular_rhs_count'], 0)
                    if not new:
                        self.assertEqual(len(entry['candidate_stages']), 1)
                        self.assertEqual(entry['paths']['R_AFFINE_seq']['b_scores'], entry['paths']['R_AFFINE_seq']['c_scores'])

    def test_true_k1_and_zero_old_scale_keep_real_closed_heads_and_no_held_metrics(self):
        for zero in (False, True):
            inputs = synthetic(k=1, zero=zero)
            with tempfile.TemporaryDirectory() as directory:
                archive = probe.StateArchive(directory)
                record, split = as_record(probe.probe_affine_joint(**inputs, context=scope(), state_callback=archive), inputs, 1)
                archive.finalize('COMPLETE'); resolver = summary.StateResolver(directory)
                _, _, metrics, training = summary.verify_record(record, split, inputs['old_classes'], resolver); resolver.finalize()
            self.assertEqual(metrics, dict(oof=None, proxy=None)); self.assertEqual(len(training), 2)
            self.assertEqual(record['optimizer_steps'], 0); self.assertEqual(record['final_head_fit_count'], 2)

    def test_incomplete_run_does_not_open_held_trace_or_arrays(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores, \
             patch.object(summary, 'StateResolver') as archives:
            with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(spec=spec, output=Path(directory)/'out')
            scores.assert_not_called(); archives.assert_not_called()


if __name__ == '__main__': unittest.main()
