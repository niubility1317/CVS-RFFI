"""Synthetic channel log streams only; no model import, fit, score, or remote I/O."""
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
import collect_d92_channel_training_diagnostics as collect

ALGORITHM = dict(optimizer_steps=8, learning_rate=.02, adam_beta1=.9, adam_beta2=.999,
    adam_epsilon=1e-8, gradient_clip_norm=1., early_stopping=False, block_dimensions=[160, 96, 160, 160, 160])


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def save(lane, stages, events):
    compact = [collect.compact_event(row) for row in events]
    jsonl(lane/'fit_stages.jsonl', stages); jsonl(lane/'training_events.jsonl', events)
    jsonl(lane/'training_events_compact.jsonl', compact)
    for name, rows in (('fit_stages', stages), ('training_events_compact', compact)):
        fields = sorted({key for row in rows for key in row})
        with (lane/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({key: collect.csv_value(row.get(key)) for key in fields} for row in rows)
    text = [json.dumps(row) for row in stages]+['JOINT_CHANNEL_TRAINING '+json.dumps(row) for row in compact]
    (lane/'training.log').write_text('\n'.join(text)+'\nWarning: synthetic tail\n', encoding='utf-8')
    (lane.parent/'probe.log').write_text(json.dumps(dict(event='STARTUP', hardware={'gpu_use': False}, config=ALGORITHM))+'\n'+
        '\n'.join(text)+'\n'+json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', oof={'never_extract_outer_held': .99}))+'\n', encoding='utf-8')


def objective(prep, u, anchor, position, gradient):
    folds = []
    n = prep['train_physical_count']; loss = 1.-.01*position
    for before in prep['inner_folds']:
        held = before['held_physical_count']
        folds.append(dict(before, held_margin_loss_sum=loss*held, head_training_loss_total=999.,
            factorization_count=1, head_fit_count=1, derivative_triangular_solve_count=2 if gradient is not None else 0,
            held_training_correct_count=min(held, held//2+int(position == 9)),
            held_training_margin_mean=.1*position, held_training_margin_min=-.5+.01*position))
    proximal=math.fsum((a-b)**2 for a, b in zip(u, anchor))/(2*n)
    return dict(loss_data=loss, loss_proximal=proximal, loss_total=loss+proximal,
        loss_scope='PHYSICAL_INNER_HELD_MARGIN_PLUS_PROXIMAL', gradient=gradient, inner_folds=folds,
        inner_head_fit_count=len(folds), inner_factorization_count=len(folds),
        derivative_triangular_solve_count=2*len(folds) if gradient is not None else 0)


def fixture(root):
    lane = root/'row-a'/'probe'; lane.mkdir(parents=True)
    stages, events = [], []
    for proxy in (False, True):
        k = 1 if proxy else 4
        coords = dict(row_id='row-a', split_id='proxy' if proxy else 'parent', parent_k=5, train_k=k,
            scope='support_oneshot_proxy' if proxy else 'support_oof', fold=None if proxy else 0, trial=0 if proxy else None)
        for state in (('B',) if proxy else ('B', 'C')):
            classes = ['old-a', 'old-b'] if state == 'B' else ['new-z', 'old-a', 'old-b']
            ids = [f'{cls}-{i}' for cls in classes for i in range(k)]
            stages.append(dict(coords, event='BASE_FIT', state=state+'0', factorization_calls=1, fit_seconds=.1))
            folds = []
            for index in range(0 if proxy else 3):
                held = [pid for pid in ids if int(pid[-1]) % 3 == index]; train = [pid for pid in ids if pid not in held]
                folds.append(dict(inner_fold=index, training_physical_ids=train, held_physical_ids=held,
                    train_physical_count=len(train), held_physical_count=len(held),
                    original_interaction_centered_trace=1., original_bandwidth_tau=1., all_head_statistics_from_inner_train_only=True))
            prep = dict(coords, stage=state, classes=classes, old_classes=['old-a', 'old-b'], training_physical_ids=ids,
                train_physical_count=len(ids), channel_preparation_count=1, prepare_seconds=.1,
                inner_folds=folds, no_information=proxy, inherited_state=state == 'C')
            stages.append(collect.compact_event(dict(prep, event='CHANNEL_PREPARATION', state=state)))
            for folded in folds:
                events.append(dict(coords, **folded, event='CHANNEL_INNER_PREPARED', state=state+'_prepare',
                    source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
            for name, mode, behavior in ([('B_channel', 'B', 'cancel')] if state == 'B' else
                    [('C_channel', 'C_seq', 'zero'), ('C_reset', 'C_reset', 'move')]):
                u = [0.]*736; anchor = u[:]; m = u[:]; v = u[:]; embedded = []
                for step in range(1, 1 if proxy else 9):
                    before = u[:]; g = [.2]*736 if behavior == 'cancel' else [0.]*736
                    if behavior == 'move': g[0], g[1] = -1., 1.
                    norm = math.hypot(*g); scale = min(1., 1./norm) if norm else 1.; used = [x*scale for x in g]
                    m = [.9*a+.1*b for a, b in zip(m, used)]
                    v = [.999*a+.001*b*b for a, b in zip(v, used)]
                    proposal = [a-.02*(b/(1-.9**step))/(math.sqrt(c/(1-.999**step))+1e-8) for a, b, c in zip(u, m, v)]
                    u = [0.]*736 if behavior == 'cancel' else proposal[:]
                    record = dict(objective(prep, before, anchor, step, g), step=step, u_pre=before, u_post=u[:], anchor=anchor,
                        learning_rate=.02, gradient_norm=norm, gradient_clip_scale=scale, clipped_gradient_norm=math.hypot(*used),
                        unprojected_update_norm=math.dist(proposal, before), update_norm=math.dist(u, before), active_box_count=0,
                        projection_zero_sum_residuals=[math.fsum(block) for block in collect.parameter_blocks(u)],
                        u_rms=math.hypot(*u)/math.sqrt(736), u_max_abs=max(abs(x) for x in u), step_seconds=.01)
                    embedded.append(deepcopy(record))
                    events.append(dict(record, **coords, mode=mode, state=name, event='JOINT_CHANNEL_STEP',
                        source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
                trained = not proxy; changed = u != anchor; final = objective(prep, u, anchor, 9, None) if trained else None
                full = dict(coords, classes=classes, old_classes=['old-a', 'old-b'], mode=mode, state=name,
                    event='JOINT_CHANNEL_FIT', status='CHANNEL_STAGE_COMPLETE', u=u, anchor=anchor, u_anchor=anchor,
                    u_changed_from_anchor=changed, u_update_norm=math.dist(u, anchor), optimizer_state_reset=True, optimizer_state_bytes=11776,
                    nonzero_projected_update_count=8 if behavior == 'move' and trained else 0,
                    optimizer_steps=8 if trained else 0, inner_objective_evaluation_count=9 if trained else 0,
                    inner_head_fit_count=27 if trained else 0, inner_factorization_count=27 if trained else 0,
                    derivative_triangular_solve_count=48 if trained else 0,
                    final_head_fit_count=int(changed), final_factorization_count=int(changed),
                    steps=embedded, final_objective=final, preparation=prep, no_information=proxy,
                    no_update_reason='PHYSICAL_K1' if proxy else None, identity_forward=not changed, config=ALGORITHM,
                    source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION', fit_seconds=.2,
                    persistent_state_bytes=8888, head_state_bytes=1000, adapter_state_bytes=5888, lineage_state_bytes=2000,
                    trainable_parameter_count=736, effective_parameter_count=731,
                    baseline_binding_seconds=.001, baseline_binding_distance_evaluation_count=1, baseline_binding_factorization_count=0)
                events.append(full); stages.append(collect.compact_event(dict(full, event='CANDIDATE_FIT', score_seconds=.01)))
    save(lane, stages, events)
    return lane, stages, events


class CollectorTests(unittest.TestCase):
    def test_full_stream_updates_margins_and_compact_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); result = collect.collect_lane(lane, 'row-a', ALGORITHM)
        self.assertEqual(result['coverage']['optimizer_steps'], 24)
        self.assertEqual(result['coverage']['channel_stage_count'], 4)
        self.assertEqual(result['coverage']['inner_head_fit_count'], 81)
        self.assertEqual(result['coverage']['derivative_triangular_solve_count'], 144)
        self.assertEqual(result['coverage']['head_fit_count'], 85)
        first = result['stages'][0]
        self.assertTrue(first['all_projected_zero']); self.assertFalse(first['all_gradient_zero'])
        self.assertEqual(first['counts']['projection_cancelled_updates'], 8)
        self.assertEqual(first['counts']['clipped_steps'], 8)
        self.assertEqual([p['step'] for p in first['curve']], list(range(1, 9))+[None])
        self.assertAlmostEqual(first['first']['inner_training_margin_mean'], .1)
        self.assertAlmostEqual(first['final']['inner_training_margin_mean'], .9)
        self.assertAlmostEqual(first['final_minus_initial']['inner_training_margin_mean'], .8)
        self.assertEqual(first['parent_new_count'], 1)
        self.assertIsNone(first['inner_prediction_change_count'])
        self.assertFalse(first['inner_correct_count_unchanged'])
        reset = next(s for s in result['stages'] if s['mode'] == 'C_reset')
        self.assertEqual(reset['counts']['nonzero_projected_updates'], 8)
        self.assertGreater(reset['final']['u_z_id_gain_condition_number'], 1.)
        self.assertEqual(reset['final']['u_fft_gain_condition_number'], 1.)
        self.assertIsNone(reset['final']['gradient_norm']); self.assertIsNone(reset['final']['residual_rms'])
        self.assertNotIn('never_extract_outer_held', json.dumps(result))
        def no_large_arrays(value):
            if isinstance(value, list):
                self.assertNotEqual(len(value), 736)
                for item in value: no_large_arrays(item)
            elif isinstance(value, dict):
                for item in value.values(): no_large_arrays(item)
        no_large_arrays(result)
        groups = collect.aggregate(result['stages'])
        self.assertEqual(sum(row['stages'] for row in groups['group_counts']), 4)
        self.assertEqual(sum(row['stages'] for row in groups['stratified_counts']), 4)

    def test_noinfo_remains_null_and_zero_gradient_still_has_eight_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); stages = collect.collect_lane(lane, 'row-a', ALGORITHM)['stages']
        skipped = next(s for s in stages if s['no_information'])
        self.assertEqual(skipped['steps'], 0); self.assertIsNone(skipped['final']['loss_total'])
        self.assertIsNone(skipped['final']['inner_training_margin_mean']); self.assertFalse(skipped['all_gradient_zero'])
        zero = next(s for s in stages if s['mode'] == 'C_seq')
        self.assertEqual(zero['steps'], 8); self.assertTrue(zero['all_gradient_zero'])

    def test_csv_and_full_compact_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); path = lane/'training_events_compact.csv'
            path.write_text(path.read_text(encoding='utf-8').replace('0.99', '0.98', 1), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CSV/JSON'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, _, events = fixture(Path(directory)); event = next(e for e in events if e['event'] == 'JOINT_CHANNEL_STEP')
            event['gradient'][0] += 1.; jsonl(lane/'training_events.jsonl', events)
            with self.assertRaisesRegex(ValueError, 'Full/compact'): collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_parameter_adam_mutation_and_final_backward_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, events = fixture(Path(directory))
            final = next(e for e in events if e['event'] == 'JOINT_CHANNEL_FIT' and e['mode'] == 'C_reset')
            step = final['steps'][0]
            step['u_post'][0] += .001; step['u_post'][1] -= .001
            event = next(e for e in events if e['event'] == 'JOINT_CHANNEL_STEP' and e['mode'] == 'C_reset')
            event['u_post'] = step['u_post'][:]
            # Update the compact final copy too, so the independent Adam check is reached.
            position = next(i for i, row in enumerate(stages) if row.get('state') == 'C_reset')
            stages[position] = collect.compact_event(dict(final, event='CANDIDATE_FIT', score_seconds=.01))
            save(lane, stages, events)
            with self.assertRaisesRegex(ValueError, 'KKT'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, events = fixture(Path(directory))
            full = next(e for e in events if e['event'] == 'JOINT_CHANNEL_FIT')
            full['final_objective']['derivative_triangular_solve_count'] = 2
            index = next(i for i, row in enumerate(stages) if row.get('state') == 'B_channel')
            stages[index] = collect.compact_event(dict(full, event='CANDIDATE_FIT', score_seconds=.01)); save(lane, stages, events)
            with self.assertRaisesRegex(ValueError, 'differentiated'): collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_all_text_tail_and_false_earlystop_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, events = fixture(Path(directory)); result = collect.collect_lane(lane, 'row-a', ALGORITHM)
            scan = result['text_logs'][1]
            self.assertGreater(scan['marker_line_counts']['early_stop_disabled_declaration'], 0)
            self.assertEqual(scan['marker_line_counts']['early_stop_actual_message'], 0)
            with (lane.parent/'probe.log').open('a', encoding='utf-8') as stream:
                stream.write('JOINT_CHANNEL_TRAINING '+json.dumps(collect.compact_event(events[-1]))+'\n')
            with self.assertRaisesRegex(ValueError, 'Text/event'): collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'log'
            path.write_text('{"early_stopping": false}\nEarly stopping triggered\n{"event":"EARLY_STOPPED"}\n', encoding='utf-8')
            scan = collect.scan_text(path)
            self.assertEqual(scan['marker_line_counts']['early_stop_actual_message'], 2)
        observed = collect.structured_observations(dict(status='TECHNICAL_FAILURE', failure_reason='bad', x=float('nan'), y='Infinity'))
        self.assertEqual(observed['technical_failure_statuses'], 1)
        self.assertEqual(observed['numeric_nonfinite_values'], 1)
        self.assertEqual(observed['encoded_nonfinite_values'], 1)

    def test_incomplete_run_refuses_to_open_training_or_performance_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = dict(startup=dict(spec={'probe': {'algorithm': ALGORITHM}, 'rows': []}, commit='abc'),
                complete=dict(status='FAILED', model_rows=4, completed_rows=3, commit='abc'), state={})
            for name, value in documents.items(): (root/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
            with patch.object(collect, 'jsonlines') as read:
                with self.assertRaisesRegex(ValueError, 'Complete fixed four-row'): collect.collect(root)
                read.assert_not_called()

    def test_output_and_existing_destination_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane, _, _ = fixture(root); parsed = collect.collect_lane(lane, 'row-a', ALGORITHM)
            result = dict(status='SYNTHETIC', run_id='synthetic', coverage=parsed['coverage'], totals={},
                stage_summaries=parsed['stages'], resource_groups=parsed['resource_groups'], text_marker_line_counts={}, limitations=[],
                **collect.aggregate(parsed['stages']))
            out = root/'new'; collect.write_outputs(out, result)
            self.assertEqual(len(list(collect.csvlines(out/'stage_curves.csv'))), 28)
            self.assertTrue((out/'stratified_counts.csv').is_file()); self.assertTrue((out/'report.md').is_file())
            with self.assertRaises(FileExistsError): collect.write_outputs(out, result)

    def test_ssh_stdin_is_readonly_and_honors_config(self):
        result = dict(status='SYNTHETIC', coverage={})
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'new')
            argv = ['collector', '--run-root', '/remote/run root', '--run-log', '/release/run.log', '--output', output,
                '--ssh-host', 'n607', '--ssh-config', 'E:/type10-7/tools/n607_ssh_config', '--remote-python', '/remote/python']
            with patch.object(sys, 'argv', argv), patch.object(collect.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')) as ssh, \
                    patch.object(collect, 'write_outputs') as write, patch('builtins.print'):
                collect.main()
            self.assertEqual(ssh.call_args.args[0][:-1], ['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config', '-T', '-o', 'BatchMode=yes', 'n607'])
            command = collect.shlex.split(ssh.call_args.args[0][-1]); self.assertNotIn('--output', command)
            self.assertEqual(command, ['/remote/python', '-', '--run-root', '/remote/run root', '--collect-stdout', '--run-log', '/release/run.log'])
            self.assertEqual(ssh.call_args.kwargs['input'], Path(collect.__file__).read_text(encoding='utf-8'))
            write.assert_called_once_with(output, result)


if __name__ == '__main__': unittest.main()
