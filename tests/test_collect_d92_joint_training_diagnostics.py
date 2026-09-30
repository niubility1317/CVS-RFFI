"""Synthetic full-stream fixtures only: no fitting, scoring, or real run access."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import collect_d92_joint_training_diagnostics as collect

ALGORITHM = dict(optimizer_steps=8, learning_rate=.1, arms=['R0', 'joint'])


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def objective(value, gradient):
    return dict(loss_data=value, loss_proximal=.001, loss_total=value+.001, gradient=gradient,
        inner_folds=[dict(held_training_correct_count=1, held_physical_count=2),
                     dict(held_training_correct_count=2, held_physical_count=3)],
        inner_head_fit_count=2, inner_factorization_count=2,
        derivative_triangular_solve_count=8 if gradient is not None else 0)


def save_fixture(lane, stages, events):
    compact = [collect.compact_event(event) for event in events]
    jsonl(lane/'fit_stages.jsonl', stages); jsonl(lane/'training_events.jsonl', events)
    jsonl(lane/'training_events_compact.jsonl', compact)
    for name, rows in (('fit_stages', stages), ('training_events_compact', compact)):
        fields = sorted({key for row in rows for key in row})
        with (lane/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({key: collect.csv_value(row.get(key)) for key in fields} for row in rows)
    # Interleaving order does not affect each independently reconciled stream.
    text = [json.dumps(row) for row in stages]+['JOINT_SPECTRAL_TRAINING '+json.dumps(row) for row in compact]
    (lane/'training.log').write_text('\n'.join(text)+'\nWarning: synthetic tail\n', encoding='utf-8')
    (lane.parent/'probe.log').write_text(json.dumps(dict(event='STARTUP', hardware={'gpu_use': False}))+'\n'+
        '\n'.join(text)+'\n'+json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', oof={'do_not_analyze_held': .99}))+'\n', encoding='utf-8')


def make_fixture(root):
    lane = root/'row-a'/'probe'; lane.mkdir(parents=True)
    stages, events = [], []
    for proxy in (False, True):
        coords = dict(row_id='row-a', split_id='proxy' if proxy else 'parent',
            scope='support_oneshot_proxy' if proxy else 'support_oof', fold=None if proxy else 0,
            trial=0 if proxy else None, parent_k=5, train_k=1 if proxy else 3)
        for state in (('B',) if proxy else ('B', 'C')):
            stages.append(dict(coords, event='BASE_FIT', state=state+'0', factorization_calls=1, fit_seconds=.1))
            folded = dict(inner_fold=0, training_physical_ids=['train0', 'train1'], held_physical_ids=['held0'],
                geometry_training_physical_ids=['train0'], geometry_identity=False,
                geometry_audit=dict(spectral_eigenvalues=[.2, .8]), train_physical_count=2, held_physical_count=1,
                direct_difference_pair_count=0)
            prep = dict(coords, stage=state, geometry_fit_count=1, geometry_factorization_count=int(not proxy),
                inner_folds=[] if proxy else [folded], prepare_seconds=.1, no_information=proxy)
            stages.append(collect.compact_event(dict(prep, event='JOINT_PREPARATION', state=state)))
            if not proxy:
                events.append(dict(coords, **folded, event='JOINT_INNER_PREPARED', state=state+'_prepare',
                    source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
            modes = [('B_joint', 'B', 'cancel'), ('B_fixed', 'fixed', 'fixed')] if state == 'B' else [
                ('C_joint', 'C_seq', 'zero'), ('C_reset', 'C_reset', 'move'), ('C_fixed', 'fixed', 'fixed')]
            for name, mode, behavior in modes:
                embedded = []; theta = [1., 0.] if mode == 'fixed' else [0., 0.]
                trained = not proxy and mode != 'fixed'; anchor = [0., 0.]
                if trained:
                    for step in range(1, 9):
                        before = theta[:]
                        grad = [.2, .1] if behavior == 'cancel' else ([0., 0.] if behavior == 'zero' else [-.1, 0.])
                        if behavior == 'move': theta = [theta[0]+.01, 0.]
                        obj = objective(1-step/100., grad)
                        record = dict(obj, step=step, theta_pre=before, theta_post=theta[:], theta_anchor=anchor,
                            learning_rate=.1, gradient_norm=math.hypot(*grad), unprojected_update_norm=.1*math.hypot(*grad),
                            update_norm=math.dist(before, theta), active_nonnegative_constraints=[v == 0 for v in theta],
                            active_sum_constraint=sum(theta) == 1, step_seconds=.01)
                        embedded.append(record)
                        events.append(dict(record, **coords, state=name, mode=mode, event='JOINT_SPECTRAL_STEP',
                            source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
                final = objective(.90, None) if trained else None
                full = dict(coords, state=name, mode=mode, event='JOINT_SPECTRAL_FIT', status='JOINT_STAGE_COMPLETE',
                    optimizer_steps=8 if trained else 0, inner_objective_evaluation_count=9 if trained else 0,
                    inner_head_fit_count=18 if trained else 0, inner_factorization_count=18 if trained else 0,
                    derivative_triangular_solve_count=64 if trained else 0,
                    final_head_fit_count=int(mode == 'fixed' and not proxy), final_factorization_count=int(mode == 'fixed' and not proxy),
                    steps=deepcopy(embedded), theta=theta, theta_anchor=anchor, anchor=anchor,
                    theta_changed_from_anchor=theta != anchor, nonzero_projected_update_count=8 if trained and behavior == 'move' else 0,
                    no_information=proxy, preparation=prep, config=ALGORITHM, final_objective=final,
                    no_update_reason=None if trained else 'FIXED_METRIC_ABLATION' if mode == 'fixed' else 'NO_IDENTIFIABLE_INNER_GEOMETRY',
                    identity_forward=proxy or theta == [0., 0.], source_validation=None,
                    objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION', fit_seconds=.2,
                    persistent_state_bytes=1016, shared_geometry_state_bytes=800, head_state_bytes=200,
                    theta_state_bytes=16, trainable_parameter_count=0 if mode == 'fixed' else 2)
                events.append(full)
                stages.append(collect.compact_event(dict(full, event='CANDIDATE_FIT', score_seconds=.01)))
    save_fixture(lane, stages, events)
    return lane, stages, events


class CollectTests(unittest.TestCase):
    def test_all_updates_finals_physical_stages_and_projection_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = make_fixture(Path(directory))
            result = collect.collect_lane(lane, 'row-a', ALGORITHM)
        self.assertEqual(result['coverage']['optimizer_steps'], 24)
        self.assertEqual(result['coverage']['trained_joint_stage_count'], 3)
        self.assertEqual(result['coverage']['joint_stage_count'], 4)
        self.assertEqual(result['coverage']['fixed_stage_count'], 3)
        self.assertEqual(result['coverage']['inner_objective_evaluation_count'], 27)
        self.assertEqual(len(result['stages']), 7)
        first = result['stages'][0]
        self.assertEqual(first['counts']['projection_cancelled_updates'], 8)
        self.assertEqual(first['counts']['gradient_zero_steps'], 0)
        self.assertTrue(first['all_projected_zero']); self.assertFalse(first['all_gradient_zero'])
        self.assertEqual(len(first['curve']), 9)
        self.assertEqual([p['step'] for p in first['curve']], list(range(1, 9))+[None])
        self.assertAlmostEqual(first['first']['loss_data'], .99)
        self.assertAlmostEqual(first['last']['loss_data'], .92)
        self.assertAlmostEqual(first['final']['loss_data'], .90)
        self.assertAlmostEqual(first['final_minus_initial']['loss_data'], -.09)
        self.assertEqual(first['final']['inner_training_accuracy'], .6)
        self.assertIsNone(first['final']['gradient_norm'])
        self.assertIsNone(first['final']['residual_rms'])
        reset = next(s for s in result['stages'] if s['identity']['state'] == 'C_reset')
        self.assertEqual(reset['counts']['nonzero_projected_updates'], 8)
        self.assertTrue(reset['theta_changed_from_anchor'])
        self.assertAlmostEqual(reset['final']['theta_sum'], .08)
        self.assertAlmostEqual(reset['final']['operator_contraction_universal_bound'], .04)
        self.assertAlmostEqual(reset['final']['inner_geometry_max_spectral_gain'], .8/1.8*.08)
        self.assertIsNone(reset['final']['full_final_geometry_max_spectral_gain'])
        self.assertEqual(result['text_logs'][0]['marker_line_counts']['warning'], 1)
        self.assertEqual(result['text_logs'][1]['events']['SUPPORT_PARENT_COMPLETE'], 1)
        self.assertNotIn('do_not_analyze_held', json.dumps(result))
        groups = collect.aggregate(result['stages'])
        self.assertEqual(sum(g['stages'] for g in groups['group_counts']), 7)
        self.assertEqual(sum(g.get('nonzero_projected_updates', 0) for g in groups['group_counts']), 8)

    def test_skipped_and_fixed_objectives_stay_null_and_not_learned_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = make_fixture(Path(directory)); result = collect.collect_lane(lane, 'row-a', ALGORITHM)
        proxy = next(s for s in result['stages'] if s['identity']['scope'] == 'support_oneshot_proxy' and not s['fixed_control'])
        self.assertEqual(proxy['steps'], 0); self.assertIsNone(proxy['final']['loss_total'])
        self.assertIsNone(proxy['final']['inner_training_accuracy']); self.assertFalse(proxy['all_projected_zero'])
        self.assertEqual(proxy['theta'], [0., 0.])
        fixed = next(s for s in result['stages'] if s['fixed_control'])
        self.assertEqual(fixed['theta'], [1., 0.]); self.assertEqual(fixed['steps'], 0)
        self.assertFalse(fixed['trained']); self.assertIsNone(fixed['first']['loss_total'])

    def test_csv_and_compact_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = make_fixture(Path(directory)); path = lane/'training_events_compact.csv'
            path.write_text(path.read_text(encoding='utf-8').replace('0.99', '0.98', 1), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CSV/JSON mismatch'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, _, events = make_fixture(Path(directory)); events[1]['gradient'][0] = 88.
            jsonl(lane/'training_events.jsonl', events)
            with self.assertRaisesRegex(ValueError, 'Full/compact'): collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_missing_final_embedded_step_and_late_text_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, events = make_fixture(Path(directory))
            events = [e for e in events if not (e['event'] == 'JOINT_SPECTRAL_FIT' and e['state'] == 'C_reset')]
            save_fixture(lane, stages, events)
            with self.assertRaisesRegex(ValueError, 'Missing candidate final'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, events = make_fixture(Path(directory))
            final = next(e for e in events if e['event'] == 'JOINT_SPECTRAL_FIT' and e['state'] == 'B_joint')
            final['steps'][3]['step_seconds'] = 42.
            save_fixture(lane, stages, events)
            with self.assertRaisesRegex(ValueError, 'Embedded/event'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, _, events = make_fixture(Path(directory))
            with (lane.parent/'probe.log').open('a', encoding='utf-8') as stream:
                stream.write('JOINT_SPECTRAL_TRAINING '+json.dumps(collect.compact_event(events[-1]))+'\n')
            with self.assertRaisesRegex(ValueError, 'Text/event'): collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_identity_and_inner_partition_checks(self):
        record = dict(split_id='p', scope='support_oof', fold=0, trial=None, state='B_joint', train_k=3)
        key = collect.stage_key('row1', record)
        for row, changed in [('row2', record), ('row1', dict(record, fold=1)), ('row1', dict(record, state='C_joint'))]:
            self.assertNotEqual(key, collect.stage_key(row, changed))
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, events = make_fixture(Path(directory))
            events[0]['held_physical_ids'] = ['train0']; save_fixture(lane, stages, events)
            with self.assertRaisesRegex(ValueError, 'physical support partition'): collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_incomplete_run_rejected_before_training_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = dict(startup=dict(spec={'probe': {'algorithm': ALGORITHM}, 'rows': []}, commit='abc'),
                complete=dict(status='FAILED', model_rows=4, completed_rows=3, commit='abc'), state={})
            for name, obj in documents.items(): (root/(name+'.json')).write_text(json.dumps(obj), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Complete fixed four-row'): collect.collect(root)

    def test_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane, _, _ = make_fixture(root)
            lane_result = collect.collect_lane(lane, 'row-a', ALGORITHM)
            result = dict(status='SYNTHETIC', run_id='synthetic', coverage=lane_result['coverage'], totals={},
                stage_summaries=lane_result['stages'], resource_groups=[], text_marker_line_counts={}, limitations=[],
                **collect.aggregate(lane_result['stages']))
            destination = root/'new-output'; collect.write_outputs(destination, result)
            for filename in ('report.md', 'summary.json', 'stages.csv', 'stage_curves.csv', 'curves.csv'):
                self.assertTrue((destination/filename).is_file())
            saved = json.loads((destination/'summary.json').read_text(encoding='utf-8'))
            self.assertNotIn('stage_summaries', saved)
            self.assertEqual(len(list(collect.csvlines(destination/'stage_curves.csv'))), 31)
            with self.assertRaises(FileExistsError): collect.write_outputs(destination, result)

    def test_ssh_config_and_readonly_stdin(self):
        result = dict(status='SYNTHETIC', coverage={})
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'new-local-output')
            argv = ['collector', '--run-root', '/remote/run root', '--run-log', '/release/run.log', '--output', output,
                '--ssh-host', 'n607', '--ssh-config', 'E:/type10-7/tools/n607_ssh_config', '--remote-python', '/remote/python']
            with patch.object(sys, 'argv', argv), patch.object(collect.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')) as ssh, \
                    patch.object(collect, 'write_outputs') as write, patch('builtins.print'):
                collect.main()
            self.assertEqual(ssh.call_args.args[0][:-1], ['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config',
                '-T', '-o', 'BatchMode=yes', 'n607'])
            command = collect.shlex.split(ssh.call_args.args[0][-1])
            self.assertEqual(command, ['/remote/python', '-', '--run-root', '/remote/run root', '--collect-stdout',
                '--run-log', '/release/run.log'])
            self.assertNotIn('--output', command)
            self.assertEqual(ssh.call_args.kwargs['input'], Path(collect.__file__).read_text(encoding='utf-8'))
            write.assert_called_once_with(output, result)


if __name__ == '__main__': unittest.main()
