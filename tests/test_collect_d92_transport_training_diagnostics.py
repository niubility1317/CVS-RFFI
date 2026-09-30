"""Synthetic frozen transport log records only; no model, fit, score or data I/O."""
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
import collect_d92_transport_training_diagnostics as collector

ALGORITHM = dict(method='D92-PrototypeTransportLocalRidge-v1', max_iterations=4, max_trials=3,
                 initial_step_size=.125, backtrack_factor=.5, armijo_coefficient=1e-4)


def jsonl(path, records):
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')


def save(lane, stages, events):
    compact = [collector.compact_event(e) for e in events]
    jsonl(lane/'fit_stages.jsonl', stages); jsonl(lane/'training_events.jsonl', events)
    jsonl(lane/'training_events_compact.jsonl', compact)
    for name, records in (('fit_stages', stages), ('training_events_compact', compact)):
        keys = sorted({key for r in records for key in r})
        with (lane/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
            writer.writerows({key: collector.csv_value(r.get(key)) for key in keys} for r in records)
    text = [json.dumps(r) for r in stages]+['PROTOTYPE_TRANSPORT_TRAINING '+json.dumps(r) for r in compact]
    (lane/'training.log').write_text('STARTUP {"config":{}}\n'+'\n'.join(text)+'\nWarning: synthetic tail\n', encoding='utf-8')
    # Deliberately malformed parent-result payload: the training collector must not deserialize it.
    (lane.parent/'probe.log').write_text('\n'.join(text)+'\n{"event":"SUPPORT_PARENT_COMPLETE","oof": DO_NOT_PARSE_THIS}\n', encoding='utf-8')


def objective(prep, u, anchor, loss=1., cached=False, backward=False):
    folds = []
    for original in prep['inner_folds']:
        per_class = original['held_physical_count']//len(prep['classes'])
        folds.append(dict(original, class_margin_loss_sums=[loss*per_class]*len(prep['classes']),
            class_physical_counts=[per_class]*len(prep['classes']), held_training_correct_count=per_class,
            held_training_margin_mean=.1, normal_equation_residual=0., trace_relative_error=0., numerical_tolerance=1e-10,
            head_training_loss_total=999., head_fit_count=int(not cached), factorization_count=int(not cached),
            transport_forward_evaluation_count=int(not cached), derivative_triangular_solve_count=2 if backward else 0,
            forward_seconds=.002, **({'adjoint_seconds': .003} if backward else {})))
    n = prep['train_physical_count']; proximal = math.fsum((x-y)**2 for x, y in zip(u, anchor))/(2*n)
    return dict(classes=prep['classes'], class_loss_values=[loss]*len(prep['classes']),
        class_held_counts=[prep['train_k']]*len(prep['classes']), loss_data=loss, loss_proximal=proximal, loss_total=loss+proximal,
        loss_scope='CLASS_RMS_INNER_HELD_MARGIN_PLUS_PROXIMAL', inner_folds=folds, objective_seconds=.01,
        inner_objective_evaluation_count=int(not cached), inner_head_fit_count=len(folds)*int(not cached),
        inner_factorization_count=len(folds)*int(not cached), transport_forward_evaluation_count=len(folds)*int(not cached),
        backward_evaluation_count=int(backward), derivative_triangular_solve_count=len(folds)*2*int(backward),
        forward_cache_reused=cached, gradient=None)


def preparation(coords, state, classes):
    ids = [f'{c}-{i}' for c in classes for i in range(coords['train_k'])]
    old = ['old-a', 'old-b']; noinfo = coords['train_k'] == 1; folds = []
    for index in range(0 if noinfo else 3):
        held = [pid for pid in ids if int(pid[-1]) % 3 == index]; train = [pid for pid in ids if pid not in held]
        folds.append(dict(inner_fold=index, training_physical_ids=train, held_physical_ids=held,
            train_physical_count=len(train), held_physical_count=len(held),
            prototypes=dict(training_physical_ids=train, classes=classes, prototype_seconds=.001)))
    proto = dict(training_physical_ids=ids, classes=classes, old_prototypes_bitwise_inherited=state == 'C',
        inherited_old_prototype_classes=old if state == 'C' else [], prototype_seconds=.001)
    return dict(coords, stage=state, classes=classes, old_classes=old, training_physical_ids=ids, train_physical_count=len(ids),
        inner_folds=folds, full_prototypes=proto, no_information=noinfo, no_information_reason='PHYSICAL_K1' if noinfo else None,
        transport_preparation_count=1, prototype_construction_count=len(folds)+1,
        prototype_distance_evaluation_count=3*(len(folds)+1), prepared_distance_evaluation_count=2*(len(folds)+1),
        prepare_seconds=.1, prepared_numeric_state_bytes=1234, inherited_state=state == 'C')


def stage_events(prep, state, mode, anchor, behavior):
    coords = {key: prep[key] for key in ('row_id', 'split_id', 'scope', 'fold', 'trial', 'parent_k', 'train_k')}
    events = []; u = anchor[:]; trials = []; gradients = []; steps = []; initial = None; final = None
    def emit(kind, record):
        value = dict(prep); value.update(record)
        value.update(state=state, mode=mode, event=kind, outer_trial=coords['trial'],
            source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'); events.append(value)
    if not prep['no_information']:
        initial = objective(prep, u, anchor); emit('TRANSPORT_INITIAL', dict(u=u[:], anchor=anchor, objective=initial))
        iteration_count = 2 if behavior == 'move' else 1
        for iteration in range(1, iteration_count+1):
            g = [-1.]+[0.]*9 if iteration == 1 and behavior in ('move', 'reject') else [0.]*10
            norm = math.hypot(*g); loss = .8 if steps else 1.
            grad = dict(iteration=iteration, u=u[:], gradient=g, gradient_norm=norm,
                        objective=objective(prep, u, anchor, loss, cached=True, backward=True))
            grad['objective']['gradient'] = g[:]
            gradients.append(grad); emit('TRANSPORT_GRADIENT', grad)
            if norm == 0: break
            for index in range(1, 2 if behavior == 'move' else 4):
                size = .125*(.5**(index-1)); candidate = [x-size*v/norm for x, v in zip(u, g)]
                obj = objective(prep, candidate, anchor, .8 if behavior == 'move' else 1.2)
                rhs = grad['objective']['loss_total']+1e-4*math.fsum(v*(new-old) for v, new, old in zip(g, candidate, u))
                tol = 128*sys.float_info.epsilon*max(1., abs(grad['objective']['loss_total']), abs(obj['loss_total']), abs(rhs))
                accepted = obj['loss_total'] <= rhs+tol
                trial = dict(iteration=iteration, trial=index, step_size=size, u_pre=u[:], u_trial=candidate, anchor=anchor,
                    gradient=g, gradient_norm=norm, loss_before=grad['objective']['loss_total'], loss_after=obj['loss_total'],
                    armijo_rhs=rhs, armijo_tolerance=tol, accepted=accepted, update_norm=math.dist(u, candidate), objective=obj)
                trials.append(trial); emit('TRANSPORT_TRIAL', trial)
                if accepted:
                    step = dict(step=len(steps)+1, iteration=iteration, trial=index, u_pre=u[:], u_post=candidate, anchor=anchor,
                        gradient=g, gradient_norm=norm, step_size=size, learning_rate=size, loss_before=trial['loss_before'],
                        loss_after=trial['loss_after'], update_norm=trial['update_norm'], projection_zero_sum_residual=0., step_seconds=.02, objective=obj)
                    steps.append(step); emit('TRANSPORT_STEP', step); u = candidate; break
        final = objective(prep, u, anchor, .8 if steps else 1., cached=True)
    reason = 'PHYSICAL_K1' if prep['no_information'] else 'ARMIJO_BUDGET_EXHAUSTED' if behavior == 'reject' else 'ZERO_GRADIENT'
    forward = ([] if initial is None else [initial])+[t['objective'] for t in trials]
    head = int(any(u[:5])); full = dict(coords, mode=mode, state=state, classes=prep['classes'], old_classes=prep['old_classes'],
        preparation=prep, no_information=prep['no_information'], stop_reason=reason, status='TRANSPORT_STAGE_COMPLETE', config=ALGORITHM,
        u=u, theta=u, anchor=anchor, u_anchor=anchor, u_changed_from_anchor=u != anchor, u_update_norm=math.dist(u, anchor),
        identity_forward=not any(u[:5]), initial_objective=initial, final_objective=final, gradients=gradients, trials=trials, steps=steps,
        optimizer_steps=len(steps), optimizer_iterations=len(gradients), trial_count=len(trials), trial_attempt_count=len(trials),
        accepted_trial_count=len(steps), rejected_trial_count=len(trials)-len(steps), nonzero_projected_update_count=len(steps),
        inner_objective_evaluation_count=len(forward), inner_head_fit_count=sum(o['inner_head_fit_count'] for o in forward),
        inner_factorization_count=sum(o['inner_factorization_count'] for o in forward), final_head_fit_count=head, final_factorization_count=head,
        backward_evaluation_count=len(gradients), derivative_triangular_solve_count=sum(g['objective']['derivative_triangular_solve_count'] for g in gradients),
        transport_forward_evaluation_count=sum(o['transport_forward_evaluation_count'] for o in forward)+head,
        persistent_state_bytes=2592, adapter_state_bytes=80, prototype_state_bytes=512, head_state_bytes=1000, lineage_state_bytes=1000,
        optimizer_state_bytes=0, trainable_parameter_count=10, effective_parameter_count=9, fit_seconds=.2,
        baseline_binding_seconds=.001, baseline_binding_distance_evaluation_count=1, baseline_binding_factorization_count=0,
        source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
    emit('TRANSPORT_FIT', full)
    return full, events


def fixture(root):
    lane = root/'row-a'/'probe'; lane.mkdir(parents=True); stages = []; events = []
    for split, proxy, new in (('parent-n1', False, 1), ('parent-n2', False, 2), ('proxy-n1', True, 1)):
        coords = dict(row_id='row-a', split_id=split, scope='support_oneshot_proxy' if proxy else 'support_oof',
            fold=None if proxy else 0, trial=0 if proxy else None, parent_k=5, train_k=1 if proxy else 4)
        b_u = [0.]*10
        for prep_state in ('B', 'C'):
            classes = ['old-a', 'old-b']+([f'new-{i}' for i in range(new)] if prep_state == 'C' else [])
            prep = preparation(coords, prep_state, classes)
            stages.append(dict(coords, event='BASE_FIT', state=prep_state+'0', factorization_calls=1,
                               held_physical_count=len(classes), fit_seconds=.1, score_seconds=.01))
            stages.append(collector.compact_event(dict(prep, state=prep_state, event='TRANSPORT_PREPARATION')))
            for f in prep['inner_folds']:
                events.append(dict(coords, **f, state=prep_state+'_prepare', event='TRANSPORT_INNER_PREPARED', outer_trial=coords['trial'],
                    source_validation=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
            modes = [('B_transport', 'B', 'move')] if prep_state == 'B' else [('C_transport', 'C_seq', 'zero'), ('C_reset', 'C_reset', 'reject')]
            for state, mode, behavior in modes:
                anchor = b_u if mode == 'C_seq' else [0.]*10
                full, stream = stage_events(prep, state, mode, anchor, behavior)
                if mode == 'B': b_u = full['u']
                events.extend(stream)
                stages.append(collector.compact_event(dict(full, event='CANDIDATE_FIT', preparation_ref=prep_state,
                    held_physical_count=len(classes), score_seconds=.01)))
    save(lane, stages, events); return lane, stages, events


class CollectorTests(unittest.TestCase):
    def test_full_stream_costs_zero_update_information_and_B_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); result = collector.collect_lane(lane, 'row-a', ALGORITHM)
        self.assertEqual(result['coverage']['transport_stage_count'], 9)
        self.assertEqual(result['coverage']['information_stage_count'], 6)
        self.assertEqual(result['coverage']['trained_transport_stage_count'], 2)
        self.assertEqual(result['coverage']['optimizer_steps'], 2)
        self.assertEqual(result['coverage']['rejected_trial_count'], 6)
        self.assertEqual(result['coverage']['inner_objective_evaluation_count'], 14)
        self.assertEqual(len(result['B_reuse_contexts']), 6)
        self.assertEqual(len(result['B_physical_training_groups']), 2)
        self.assertEqual(sum(g['repeated_context_count'] for g in result['B_physical_training_groups']), 1)
        self.assertEqual(len(result['deduplicated_B_stages']), 2)
        zero = next(s for s in result['stages'] if s['mode'] == 'C_seq' and s['information_stage'])
        self.assertFalse(zero['updated_stage']); self.assertEqual(zero['counts']['backward_evaluation_count'], 1)
        self.assertEqual(zero['counts']['inner_objective_evaluation_count'], 1)
        self.assertEqual(zero['counts']['final_head_fit_count'], 1)
        self.assertEqual([p['kind'] for p in zero['curve']], ['initial', 'gradient', 'final_cached'])
        self.assertEqual(zero['curve'][1]['metrics']['inner_forward_seconds'], 0.)
        self.assertIsNone(zero['final']['source_validation']); self.assertIsNone(zero['final']['prediction_change_count'])
        self.assertNotIn('DO_NOT_PARSE_THIS', json.dumps(result))
        self.assertEqual(sum(g['actual_stages'] for g in collector.aggregate(result['stages'])['group_counts']), 9)

    def test_noinfo_null_and_all_rejected_trials_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); stages = collector.collect_lane(lane, 'row-a', ALGORITHM)['stages']
        noinfo = next(s for s in stages if not s['information_stage'])
        self.assertIsNone(noinfo['initial']['loss_total']); self.assertEqual(noinfo['counts']['inner_objective_evaluation_count'], 0)
        rejected = next(s for s in stages if s['stop_reason'] == 'ARMIJO_BUDGET_EXHAUSTED')
        self.assertEqual([p['trial'] for p in rejected['curve'] if p['kind'] == 'trial'], [1, 2, 3])
        self.assertTrue(all(p['accepted'] is False for p in rejected['curve'] if p['kind'] == 'trial'))

    def test_csv_full_compact_and_text_tail_tampering_rejected(self):
        for change in ('csv', 'full', 'text'):
            with tempfile.TemporaryDirectory() as directory:
                lane, _, events = fixture(Path(directory))
                if change == 'csv':
                    p = lane/'training_events_compact.csv'; p.write_text(p.read_text(encoding='utf-8').replace('0.125', '0.126', 1), encoding='utf-8')
                if change == 'full':
                    next(e for e in events if e['event'] == 'TRANSPORT_TRIAL')['accepted'] = False
                    jsonl(lane/'training_events.jsonl', events)
                if change == 'text':
                    with (lane/'training.log').open('a', encoding='utf-8') as source:
                        source.write('PROTOTYPE_TRANSPORT_TRAINING '+json.dumps(collector.compact_event(events[-1]))+'\n')
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'CSV/JSON|Full/compact|Text/event'):
                    collector.collect_lane(lane, 'row-a', ALGORITHM)

    def test_independent_armijo_RMS_cache_stop_and_anchor_mutations(self):
        for change in ('armijo', 'risk', 'cache', 'stop', 'anchor', 'trial_coords'):
            with tempfile.TemporaryDirectory() as directory:
                lane, stages, events = fixture(Path(directory))
                full = next(e for e in events if e['event'] == 'TRANSPORT_FIT' and e['mode'] == 'B')
                if change == 'armijo': full['trials'][0]['armijo_rhs'] += 1.
                if change == 'risk': full['initial_objective']['class_loss_values'][0] += .1
                if change == 'cache': full['final_objective']['forward_cache_reused'] = False
                if change == 'stop': full['stop_reason'] = 'MAX_ITERATIONS'
                if change == 'anchor': full['anchor'][0] = .01
                if change == 'trial_coords': full['outer_trial'] = 0
                # Refresh redundant copies/compact output; independent math/event checks must still reject.
                position = next(i for i, s in enumerate(stages) if s.get('state') == 'B_transport')
                stages[position] = collector.compact_event(dict(full, event='CANDIDATE_FIT', preparation_ref='B', held_physical_count=2, score_seconds=.01))
                save(lane, stages, events)
                with self.subTest(change=change), self.assertRaises(ValueError): collector.collect_lane(lane, 'row-a', ALGORITHM)

    def test_incomplete_metadata_never_opens_training_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in dict(startup=dict(spec={'rows': [], 'probe': {'algorithm': ALGORITHM}}, commit='same'),
                                    complete=dict(status='FAILED', model_rows=4, completed_rows=3), state={}).items():
                (root/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
            with patch.object(collector, 'jsonlines') as records:
                with self.assertRaisesRegex(ValueError, 'Complete fixed four-row'): collector.collect(root)
                records.assert_not_called()

    def test_output_native_JSON_CSV_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane, _, _ = fixture(root); result = collector.collect_lane(lane, 'row-a', ALGORITHM)
            stages = result['stages']; delivered = dict(status='SYNTHETIC', run_id='synthetic', coverage=result['coverage'], totals={},
                stage_summaries=stages, B_reuse_contexts=result['B_reuse_contexts'], B_physical_training_groups=result['B_physical_training_groups'],
                resource_groups=result['resource_groups'], limitations=[], **collector.aggregate(stages))
            out = root/'new'; collector.write_outputs(out, delivered)
            self.assertEqual(len(list(collector.csvlines(out/'stages.csv'))), 9)
            self.assertEqual(len(list(collector.csvlines(out/'B_reuse_contexts.csv'))), 6)
            self.assertTrue((out/'training_resource_groups.csv').is_file())
            self.assertEqual(json.loads((out/'training_diagnostics.json').read_text(encoding='utf-8')), delivered)
            with self.assertRaises(FileExistsError): collector.write_outputs(out, delivered)

    def test_scan_every_text_line_reports_warning_without_parsing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, _, _ = fixture(Path(directory)); result = collector.collect_lane(lane, 'row-a', ALGORITHM)
        self.assertEqual(result['text_logs'][0]['marker_line_counts']['warning'], 1)
        self.assertEqual(result['text_logs'][1]['events']['SUPPORT_PARENT_COMPLETE'], 1)
        self.assertEqual(collector.structured_observations({'status': 'TECHNICAL_FAILURE', 'x': 'NaN'})['encoded_nonfinite_values'], 1)

    def test_ssh_stdin_is_self_contained_and_readonly(self):
        result = dict(status='SYNTHETIC', coverage={})
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'new')
            args = ['collector', '--run-root', '/remote/run root', '--output', output, '--ssh-host', 'n607',
                    '--ssh-config', 'E:/type10-7/tools/n607_ssh_config', '--remote-python', '/remote/python']
            with patch.object(sys, 'argv', args), patch.object(collector.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')) as ssh, \
                    patch.object(collector, 'write_outputs') as writer, patch('builtins.print'):
                collector.main()
            self.assertEqual(collector.shlex.split(ssh.call_args.args[0][-1]),
                             ['/remote/python', '-', '--run-root', '/remote/run root', '--collect-stdout'])
            self.assertEqual(ssh.call_args.kwargs['input'], Path(collector.__file__).read_text(encoding='utf-8'))
            writer.assert_called_once_with(output, result)


if __name__ == '__main__': unittest.main()
