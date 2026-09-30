import contextlib
from copy import deepcopy
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
import evaluate_d92_registration_diagnostic as entry
import summarize_d92_registration_diagnostic as summary
import analyze_d92_registration_diagnostic as analyzer


def arrays_for(n):
    rng = np.random.default_rng(46)
    arrays = {key: rng.normal(size=(n, 96 if key == 'fft' else 160)).astype(np.float32) for key in entry.BRANCHES}
    arrays['z_id'][:, 0] = np.arange(n)
    return arrays


def small_case(k=5, new=True):
    old = ['old-A', 'old-B']; classes = old+(['new-C'] if new else [])
    ids = [f'{cls}-{i:02}' for cls in classes for i in range(k)]
    split = dict(split_id='synthetic', receiver='RX', scenario='practical_high', k=k, support_seed=7,
        registered_classes=classes, support_ids=ids, support_indices=list(range(len(ids))),
        support_labels=[i for i in range(len(classes)) for _ in range(k)])
    arrays = arrays_for(len(ids))
    return arrays, split, old


def arguments(arrays, split, old):
    return dict(arrays, support_labels=np.asarray(split['support_labels']), support_ids=split['support_ids'],
                classes=split['registered_classes'], old_classes=old)


def fake_fitter(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes, arm):
    """Synthetic deterministic scores; no numerical solver or feature export."""
    if arm != 'local_ridge': raise AssertionError('Additional arm requested')
    n, c = len(support_ids), len(classes)
    class State:
        def audit_dict(self):
            return dict(final_fit=dict(train_physical_count=n, optimizer_steps=0, factorization_calls=int(c > 1),
                status='CLOSED_FORM_SOLVED' if c > 1 else 'EXACT_ZERO_CLASSIFIER',
                interaction_centered_trace=1., radial_centered_trace=1., trace_scale=1., bandwidth_tau=1.,
                normal_equation_residual=0., trace_relative_error=0., numerical_tolerance=1e-12,
                physical_loss_mass=float(n), sample_weight=1., ridge_coefficient=1.,
                loss_data=.1, loss_ridge=.2, loss_total=.3, fit_seconds=.001,
                training_accuracy=.5, gradient_norm=0., learning_rate=None))
        def score(self, **features):
            values = features['z_id'][:, 0]
            return np.asarray([[np.sin(float(v)+sum(map(ord, cls))) for cls in classes] for v in values])
    return State()


def record_for(arrays, split, old):
    audit = entry.probe_registration(**arguments(arrays, split, old))
    return dict(audit, **entry.split_identity(split, old), scope=entry.SCOPE, query_rows_used=0, source_rows_used=0)


def pilot_fixture(root):
    old = [f'old-{i:02}' for i in range(6)]; classes = old+[f'new-{i:02}' for i in range(20)]
    ids = [f'{cls}-physical-{i:02}' for cls in classes for i in range(20)]
    raw = arrays_for(len(ids)); tasks = []
    for scenario in entry.SCENARIOS:
        for k in entry.KS:
            for new in entry.NEW_COUNTS:
                registry = classes[:6+new]
                positions = [20*c+i for c in range(6+new) for i in range(k)]
                labels = [c for c in range(6+new) for _ in range(k)]
                split = dict(split_id=f'{scenario}-{k}-{new}', receiver='RX', scenario=scenario,
                    k=k, support_seed=7, registered_classes=registry,
                    support_ids=[ids[i] for i in positions], support_indices=positions, support_labels=labels)
                tasks.append((split, np.asarray(positions), np.asarray(labels)))
    selection = dict(receiver_scenes=[['RX', s] for s in entry.SCENARIOS[:2]], support_seed=7,
        ks=entry.KS, new_counts=entry.NEW_COUNTS,
        splits=[entry.split_identity(s, old) for s, _, _ in tasks if s['scenario'] in entry.SCENARIOS[:2]])
    matrix = dict(receivers=['RX'], scenarios=entry.SCENARIOS, ks=entry.KS, new_counts=entry.NEW_COUNTS, support_seeds=[7])
    capsule = root/'capsule'; capsule.mkdir()
    entry.write(capsule/'manifest.json', dict(channel=entry.CHANNEL, scenarios=entry.SCENARIOS))
    producer = dict(feature_array_bytes=sum(v.nbytes for v in raw.values()), feature_file_bytes=100)
    loaded = (raw, tasks, old, producer, {}, {'checkpoint_sha256': 'a'*64})
    args = dict(support_features=root/'cache', capsule=capsule, output=root/'output',
        expected_capsule_id='residual-noeq-synthetic', expected_checkpoint_sha256='a'*64, expected_model_seed=3,
        config=dict(algorithm=entry.DIAGNOSTIC_CONFIG, producer_matrix=matrix, selection=selection))
    return args, loaded


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class RegistrationDiagnosticTests(unittest.TestCase):
    def test_only_local_ridge_train_ids_no_mixing_and_complete_scores(self):
        arrays, split, old = small_case()
        features_by_id = {pid: arrays['z_id'][i].tolist() for i, pid in enumerate(split['support_ids'])}
        calls = []
        def fit(**kw):
            self.assertEqual(kw['arm'], 'local_ridge')
            self.assertEqual(len(kw['z_id']), len(kw['support_ids']))
            self.assertEqual(kw['z_id'].tolist(), [features_by_id[pid] for pid in kw['support_ids']])
            self.assertTrue(all(kw['classes'][int(label)] == pid.rsplit('-', 1)[0]
                                for pid, label in zip(kw['support_ids'], kw['support_labels'])))
            calls.append(kw['support_ids'])
            return fake_fitter(**kw)
        with patch.object(entry, 'fit_branch_local_ridge', side_effect=fit):
            record = record_for(arrays, split, old)
        self.assertEqual((record['sequence_paths'], record['head_fit_count']), (8, 16))
        self.assertEqual(len(calls), 16)
        for path in record['folds']+record['oneshot_proxy']['trials']:
            self.assertFalse(set(path['c_training_ids']) & set(path['c_ids']))
            self.assertEqual(set(path['b_training_ids']), {pid for pid in path['c_training_ids'] if pid.startswith('old-')})
            self.assertEqual(set(path['b_ids']), {pid for pid in path['c_ids'] if pid.startswith('old-')})
            self.assertEqual(len(path['c_scores']), len(path['c_ids']))
            self.assertTrue(all(len(row) == 3 for row in path['c_scores']))
        stages, metrics = summary.verify_record(record, split, old)
        self.assertEqual(len(stages), 16)
        self.assertEqual(metrics['proxy'], entry.parent_mean(record['oneshot_proxy']['trials']))

    def test_true_k1_no_fit_or_holdout_and_no_new_single_fit(self):
        for k, new in ((1, True), (5, False)):
            arrays, split, old = small_case(k, new)
            with patch.object(entry, 'fit_branch_local_ridge', side_effect=fake_fitter) as fit:
                record = record_for(arrays, split, old)
            if k == 1:
                fit.assert_not_called(); self.assertIsNone(record['oof']); self.assertIsNone(record['oneshot_proxy'])
            else:
                self.assertEqual(fit.call_count, 8)
                for path in record['folds']+record['oneshot_proxy']['trials']:
                    self.assertTrue(path['c_reuses_b0']); self.assertEqual(path['b_scores'], path['c_scores'])
                self.assertIsNone(record['oof']['metrics']['C0_new_accuracy'])
                self.assertEqual(record['oof']['metrics']['total_old_accuracy_drop'], 0.)
            summary.verify_record(record, split, old)

    def test_oof_pools_unequal_folds_by_physical_rows(self):
        arrays, split, old = small_case(new=False)
        def fixed(**kw):
            state = fake_fitter(**kw)
            registry = kw['classes']
            def score(**features):
                out = []
                for value in features['z_id'][:, 0]:
                    i = int(value); truth = old[i//5]
                    winner = truth if i % 5 % 3 == 0 else old[1-i//5]
                    out.append([float(cls == winner) for cls in registry])
                return np.asarray(out)
            state.score = score
            return state
        with patch.object(entry, 'fit_branch_local_ridge', side_effect=fixed):
            record = record_for(arrays, split, old)
        self.assertEqual(record['oof']['metrics']['B0_old_accuracy'], .4)
        self.assertAlmostEqual(sum(p['metrics']['B0_old_accuracy'] for p in record['folds'])/3, 1/3)
        summary.verify_record(record, split, old)

    def test_summary_recomputes_scores_and_rejects_pair_and_aggregation_tampering(self):
        arrays, split, old = small_case()
        with patch.object(entry, 'fit_branch_local_ridge', side_effect=fake_fitter): record = record_for(arrays, split, old)
        changes = [lambda r: r['folds'][0]['b_training_ids'].append('foreign-id'),
                   lambda r: r['folds'][0]['held_labels'].update({r['folds'][0]['c_ids'][0]: 'wrong-class'}),
                   lambda r: r['oof']['metrics'].update(B0_old_accuracy=.12345),
                   lambda r: r['oneshot_proxy']['trials'].pop(),
                   lambda r: r['folds'][0]['diagnostic'].update(B_acc=.12345),
                   lambda r: r.update(head_fit_count=0)]
        for change in changes:
            with self.subTest(change=change):
                altered = deepcopy(record); change(altered)
                with self.assertRaises((ValueError, KeyError)): summary.verify_record(altered, split, old)

    def test_real_frozen_local_ridge_small_synthetic_path(self):
        arrays, split, old = small_case(k=2)
        record = record_for(arrays, split, old)
        self.assertEqual(record['head_fit_count'], 8)
        self.assertEqual(record['optimizer_steps'], 0)
        summary.verify_record(record, split, old)

    def test_selection_exact_metadata_and_cross_newcount_old_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, loaded = pilot_fixture(Path(tmp)); selection = args['config']['selection']
            self.assertEqual(len(entry.selected_tasks(loaded[1], selection, loaded[2])), 40)
            altered = deepcopy(selection); altered['splits'][1]['registered_classes'][0] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'identity'): entry.selected_tasks(loaded[1], altered, loaded[2])
            tasks = deepcopy(loaded[1]); tasks[1][0]['support_ids'][0] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'Old support'): entry.selected_tasks(tasks, selection, loaded[2])
            altered = deepcopy(selection); altered['splits'].pop()
            with self.assertRaises(ValueError): entry.validate_selection(altered)
            altered = deepcopy(selection); altered['splits'][0] = altered['splits'][1]
            with self.assertRaises(ValueError): entry.validate_selection(altered)

    def test_entry_loads_full_producer_but_executes_only_explicit_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, loaded = pilot_fixture(Path(tmp)); calls = []
            def numeric_stub(**kw):
                calls.append(tuple(kw['support_ids']))
                k = len(kw['support_ids'])//len(kw['classes']); c = len(kw['classes'])
                return dict(classes=sorted(kw['classes']), old_classes=sorted(kw['old_classes']),
                    support_count=k*c, old_class_count=6, new_class_count=c-6, fold_count=0 if k == 1 else 3,
                    numerical={}, physical_fold_assignment=[], folds=[], oof=None, oneshot_proxy=None,
                    sequence_paths=0, head_fit_count=0, factorization_count=0, optimizer_steps=0,
                    persistent_state_bytes=0, fit_seconds=0., heldout_unavailable_reason='synthetic-no-fit')
            with patch.object(entry, 'load_support', return_value=loaded) as load, \
                 patch.object(entry, 'probe_registration', side_effect=numeric_stub), contextlib.redirect_stdout(io.StringIO()):
                marker = entry.evaluate(**args)
            self.assertEqual(load.call_args.kwargs['config']['matrix'], args['config']['producer_matrix'])
            self.assertEqual(len(loaded[1]), 60); self.assertEqual(len(calls), 40)
            self.assertEqual(marker['episodes'], 40)
            self.assertEqual({r['split_id'] for r in lines(args['output']/'fit_trace.jsonl')},
                             {r['split_id'] for r in args['config']['selection']['splits']})
            self.assertEqual(entry.read(args['output']/'startup.json')['channel'], entry.CHANNEL)
            with self.assertRaises(FileExistsError): entry.evaluate(**args)

    def test_wrong_channel_rejected_before_cache_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = pilot_fixture(Path(tmp))
            (args['capsule']/'manifest.json').write_text(json.dumps(dict(
                channel=dict(entry.CHANNEL, equalization_enabled=True), scenarios=entry.SCENARIOS)), encoding='utf-8')
            with patch.object(entry, 'load_support') as load:
                with self.assertRaisesRegex(ValueError, 'channel|capsule'): entry.evaluate(**args)
            load.assert_not_called(); self.assertFalse(args['output'].exists())

    def test_frozen_base_code_and_config_must_match_before_cache_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = pilot_fixture(Path(tmp))
            changed = deepcopy(entry.DIAGNOSTIC_CONFIG); changed['base_algorithm'] = {'changed': True}
            args['config']['algorithm'] = changed
            with patch.object(entry, 'DIAGNOSTIC_CONFIG', changed), patch.object(entry, 'load_support') as load:
                with self.assertRaisesRegex(ValueError, 'code/config'): entry.evaluate(**args)
            load.assert_not_called(); self.assertFalse(args['output'].exists())

    def test_resources_sum_measured_work_without_averaging_or_inventing_steps(self):
        resources = {}
        summary.accumulate_resources(resources, dict(fit_seconds=3.), [dict(
            fit_seconds=1., score_seconds=.2, fit_and_score_seconds=1.25)])
        summary.accumulate_resources(resources, dict(fit_seconds=4.), [dict(
            fit_seconds=2., score_seconds=.3, fit_and_score_seconds=2.4)])
        self.assertEqual(resources['parent_fit_seconds_sum'], 7.)
        self.assertEqual(resources['stage_fit_seconds_sum'], 3.)
        self.assertEqual(resources['stage_score_seconds_sum'], .5)
        self.assertEqual(resources['stage_fit_and_score_seconds_sum'], 3.65)
        self.assertEqual(resources['fit_stage_count'], 2)
        self.assertNotIn('run_wall_seconds', resources)

    def test_failure_keeps_current_scores_and_prior_completed_paths(self):
        arrays, split, old = small_case()
        counter = 0
        def failing(**kw):
            nonlocal counter
            counter += 1
            if counter == 4: raise FloatingPointError('synthetic-fourth-fit')
            return fake_fitter(**kw)
        with patch.object(entry, 'fit_branch_local_ridge', side_effect=failing):
            with self.assertRaises(FloatingPointError) as raised: entry.probe_registration(**arguments(arrays, split, old))
        failure = raised.exception.registration_context
        self.assertEqual(failure['completed_paths'], 1)
        self.assertEqual(len(failure['completed_path_evidence']), 1)
        self.assertIn('c_scores', failure['completed_path_evidence'][0])
        self.assertEqual(len(failure['completed_stages']), 2)
        self.assertIn('b_scores', failure['current_path'])
        self.assertEqual(len(failure['current_path']['stages']), 1)
        json.dumps(failure, allow_nan=False)

    def test_independent_summary_rejects_partial_before_score_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = dict(execution=dict(remote_run_root=str(root)))
            entry.write(root/'startup.json', dict(spec=spec))
            entry.write(root/'complete.json', dict(status='RUNNING', completed_rows=3))
            entry.write(root/'state.json', {})
            with patch.object(summary, 'validate_spec'), patch.object(summary, 'jsonlines') as scores:
                with self.assertRaises(ValueError): summary.summarize(spec=spec, output=root/'out')
            scores.assert_not_called(); self.assertFalse((root/'out').exists())
        for count in (0, 159):
            with self.assertRaises(ValueError): analyzer.completion_check(dict(status=entry.STATUS, completed_rows=4, model_rows=4, episodes=count))
        analyzer.completion_check(dict(status=entry.STATUS, completed_rows=4, model_rows=4, episodes=160))
        compile(analyzer.REMOTE.replace('CONFIG', '{}'), 'remote-synthetic', 'exec')
        self.assertNotIn('code', analyzer.PATHS)
        self.assertFalse(any('residual_local_ridge' in path for path in analyzer.PATHS))


if __name__ == '__main__': unittest.main()
