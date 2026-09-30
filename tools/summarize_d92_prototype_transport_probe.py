"""Verify all four pilot rows before recomputing fixed support comparisons."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_prototype_transport_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, PROBE_CONFIG, PATHS, METRICS, COUNTERS,
    check, read, scalars, csv_record, split_identity, selected_tasks, compact_record, compact_event,
    assess_paths, pooled_assess, parent_mean, baseline_metrics, STAGE_COUNTERS, PREPARATION_COUNTERS,
)
from evaluate_d92_prototype_transport_probe import EXACT_COUNTS, MAX_COUNTS, validate_spec, verify_marker
import evaluate_d92_registration_diagnostic as baseline
from summarize_d92_registration_diagnostic import verify_record as verify_baseline, _statistics, write_json
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close

SUMMARY_STATUS = 'COMPLETE_PROTOTYPE_TRANSPORT_PROBE_VERIFIED'
PARAMETER_BOUND = math.log(2)/2


def parameter_blocks(values):
    yield values[:5]
    yield values[5:]


def valid_u(values):
    if not isinstance(values, list) or len(values) != 10: return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values): return False
    tolerance = 128*sys.float_info.epsilon*5*PARAMETER_BOUND
    return (max(abs(v) for v in values[:5]) <= math.pi/4+tolerance
        and max(abs(v) for v in values[5:]) <= PARAMETER_BOUND+tolerance
        and abs(math.fsum(values[5:])) <= tolerance)


def verify_box_projection(proposal, actual):
    """Independent KKT check for beta box and eta zero-sum box projection."""
    check(len(proposal) == 10 and valid_u(actual), 'Transport parameter constraint violation')
    for v, u in zip(proposal[:5], actual[:5]):
        close(u, min(math.pi/4, max(-math.pi/4, v)), 'Beta box projection mismatch')
    proposed, block = proposal[5:], actual[5:]
    tolerance = 128*sys.float_info.epsilon*max(1., max(abs(v) for v in proposed))
    free = [v-u for v, u in zip(proposed, block) if -PARAMETER_BOUND < u < PARAMETER_BOUND]
    lower = max((v-u for v, u in zip(proposed, block) if u <= -PARAMETER_BOUND), default=-math.inf)
    upper = min((v-u for v, u in zip(proposed, block) if u >= PARAMETER_BOUND), default=math.inf)
    if free:
        multiplier = math.fsum(free)/len(free)
        check(max(abs(v-multiplier) for v in free) <= tolerance
            and lower <= multiplier+tolerance and upper >= multiplier-tolerance, 'Eta projection KKT mismatch')
    else: check(lower <= upper+tolerance, 'Eta active-set KKT mismatch')


def baseline_view(record):
    def view(entry):
        result = dict(entry, **entry['paths']['R0'])
        result['metrics'] = baseline_metrics(result['diagnostic']); return result
    entries = record['folds']+([] if record['oneshot_proxy'] is None else record['oneshot_proxy']['trials'])
    result = dict(record, scope=baseline.SCOPE, persistent_state_bytes=0, optimizer_steps=0,
        head_fit_count=record['baseline_head_fit_count'],
        factorization_count=sum(stage['factorization_calls'] for entry in entries for stage in entry['stages']),
        folds=[view(entry) for entry in record['folds']])
    if record['oof'] is not None:
        value = record['oof']['paths']['R0']['diagnostic']
        result['oof'] = dict(diagnostic=value, metrics=baseline_metrics(value), aggregation='one_record_per_physical_held_id')
        trials = [view(entry) for entry in record['oneshot_proxy']['trials']]
        result['oneshot_proxy'] = dict(record['oneshot_proxy'], trials=trials, parent_mean_metrics=baseline.parent_mean(trials))
    return result


def verify_preparation(prep, entry, labels, old):
    b = prep['state'] == 'B'; ids = entry['b_training_ids' if b else 'c_training_ids']
    classes = entry['b_classes' if b else 'c_classes']; k = entry['train_k']; folds = prep['inner_folds']
    noinfo = k == 1 or len(classes) == 1
    check(prep['training_physical_ids'] == ids and prep['classes'] == classes and sorted(prep['old_classes']) == old
        and prep['train_physical_count'] == len(ids) and prep['class_count'] == len(classes)
        and prep['transport_preparation_count'] == 1 and prep['inherited_state'] is (not b)
        and prep['inherited_adapter_from'] == (None if b else 'B_transport'), 'Preparation physical/lineage mismatch')
    expected_count = min(k, 3) if not noinfo else 0
    check(len(folds) == expected_count and prep['prepare_seconds'] >= 0, 'Preparation fold/cost mismatch')
    check(prep['no_information'] is noinfo and prep['no_information_reason'] ==
        ('PHYSICAL_K1' if k == 1 else 'SINGLE_REGISTERED_CLASS' if len(classes) == 1 else None), 'No-information declaration mismatch')
    groups = {cls: sorted(pid for pid in ids if labels[pid] == cls) for cls in classes}
    assignment = {pid: index % expected_count for values in groups.values() for index, pid in enumerate(values)} if expected_count else {}
    seen = set()
    for index, fold in enumerate(folds):
        held = {pid for pid in ids if assignment[pid] == index}; train = set(ids)-held
        check(fold['inner_fold'] == index and set(fold['training_physical_ids']) == train
            and len(fold['training_physical_ids']) == len(train) and set(fold['held_physical_ids']) == held
            and len(fold['held_physical_ids']) == len(held) and not seen.intersection(held)
            and not (train | held).intersection(entry['c_ids']), 'Inner head/outer-held isolation mismatch')
        check(fold['train_physical_count'] == len(train) and fold['held_physical_count'] == len(held)
            and fold['all_head_statistics_from_inner_train_only'] is True
            and fold['original_interaction_centered_trace'] >= 0
            and (fold['original_bandwidth_tau'] is None or fold['original_bandwidth_tau'] >= 0), 'Inner physical/statistic audit mismatch')
        seen.update(held)
    check(not folds or seen == set(ids), 'Each inner training sample must be held exactly once')
    check(prep['prototype_construction_count'] == len(folds)+1
        and prep['prototype_distance_evaluation_count'] == 3*(len(folds)+1)
        and prep['prepared_distance_evaluation_count'] == 2*(len(folds)+1), 'Preparation prototype cost mismatch')
    for fold in folds:
        check(fold['prototypes']['training_physical_ids'] == fold['training_physical_ids']
            and fold['prototypes']['classes'] == classes, 'Inner-train prototype isolation mismatch')
    full = prep['full_prototypes']
    check(full['training_physical_ids'] == ids and full['classes'] == classes
        and full['old_prototypes_bitwise_inherited'] is (not b)
        and sorted(full['inherited_old_prototype_classes']) == ([] if b else old), 'Full old-prototype inheritance mismatch')


def verify_objective(value, u, anchor, prep):
    """Recompute class RMS after pooling physical losses across all inner folds."""
    folds = value['inner_folds']; n = prep['train_physical_count']; classes = prep['classes']
    check(value['loss_scope'] == 'CLASS_RMS_INNER_HELD_MARGIN_PLUS_PROXIMAL'
        and len(folds) == len(prep['inner_folds']) and value['classes'] == classes
        and len(value['class_loss_values']) == len(classes), 'Transport objective scope/fold/class mismatch')
    sums = [0.]*len(classes); counts = [0]*len(classes)
    for result, prepared in zip(folds, prep['inner_folds']):
        check(result['training_physical_ids'] == prepared['training_physical_ids']
            and result['held_physical_ids'] == prepared['held_physical_ids']
            and result['held_physical_count'] == prepared['held_physical_count']
            and 0 <= result['held_training_correct_count'] <= result['held_physical_count']
            and result['factorization_count'] in (0, 1)
            and 0 <= result['normal_equation_residual'] <= result['numerical_tolerance']
            and 0 <= result['trace_relative_error'] <= result['numerical_tolerance'], 'Uncertified inner solve/physical binding')
        check(len(result['class_margin_loss_sums']) == len(classes)
            and len(result['class_physical_counts']) == len(classes), 'Missing class-risk evidence')
        for index, (loss, count) in enumerate(zip(result['class_margin_loss_sums'], result['class_physical_counts'])):
            check(loss >= 0 and type(count) is int and count >= 0, 'Invalid physical class loss/count')
            sums[index] += loss; counts[index] += count
    for total, field in (('inner_head_fit_count', 'head_fit_count'), ('inner_factorization_count', 'factorization_count'),
                        ('transport_forward_evaluation_count', 'transport_forward_evaluation_count'),
                        ('derivative_triangular_solve_count', 'derivative_triangular_solve_count')):
        check(value[total] == sum(fold[field] for fold in folds), 'Objective actual per-fold cost mismatch: '+total)
    check(value['inner_objective_evaluation_count'] == int(not value['forward_cache_reused']), 'Objective forward-cache count mismatch')
    if value['forward_cache_reused']:
        check(value['inner_head_fit_count'] == value['inner_factorization_count'] == value['transport_forward_evaluation_count'] == 0,
            'Cache reuse refitted an inner head')
    check(sum(counts) == n and all(count > 0 for count in counts), 'Class-risk physical coverage mismatch')
    losses = [loss/count for loss, count in zip(sums, counts)]
    check(value['class_held_counts'] == counts, 'Pooled class count mismatch')
    for actual, expected in zip(value['class_loss_values'], losses): close(actual, expected, 'Pooled class loss mismatch')
    close(value['loss_data'], math.hypot(*losses)/math.sqrt(len(classes)), 'Class RMS loss mismatch')
    close(value['loss_proximal'], math.fsum((x-y)**2 for x, y in zip(u, anchor))/(2*n), 'Proximal scaling mismatch')
    close(value['loss_total'], value['loss_data']+value['loss_proximal'], 'Training loss component mismatch')


def verify_candidate(stage, prep, entry, b_transport):
    mode = stage['mode']; anchor = b_transport['u'] if mode == 'C_seq' else [0.]*10
    check(stage['anchor'] == stage['u_anchor'] == anchor and valid_u(anchor)
        and stage['preparation_ref'] == prep['state'] and stage['training_physical_ids'] == prep['training_physical_ids']
        and stage['preparation']['inner_folds'] == prep['inner_folds']
        and stage['config'] == PROBE_CONFIG and stage['source_validation'] is None
        and stage['status'] == 'TRANSPORT_STAGE_COMPLETE' and stage['no_information'] is prep['no_information'],
        'Transport state/anchor/preparation mismatch')
    steps, trials, gradients = stage['steps'], stage['trials'], stage['gradients']
    check(stage['optimizer_steps'] == len(steps) <= 4 and stage['trial_count'] == len(trials) <= 12
        and stage['trial_attempt_count'] == len(trials)
        and stage['optimizer_iterations'] == len(gradients)
        and stage['accepted_trial_count'] == len(steps)
        and stage['rejected_trial_count'] == len(trials)-len(steps)
        and stage['backward_evaluation_count'] == len(gradients) <= 4, 'Actual bounded optimizer count mismatch')
    if prep['no_information']:
        check(not steps and not trials and not gradients and stage['initial_objective'] is None
            and stage['final_objective'] is None and stage['stop_reason'] == prep['no_information_reason'],
            'Invented no-information training')
    else:
        verify_objective(stage['initial_objective'], anchor, anchor, prep)
        check(stage['initial_objective']['forward_cache_reused'] is False
            and stage['initial_objective']['backward_evaluation_count'] == 0, 'Initial objective cache/gradient mismatch')
    current = anchor; accepted = []; objectives = [] if prep['no_information'] else [stage['initial_objective']]
    used_trials = 0
    for index, event in enumerate(gradients, 1):
        check(event['iteration'] == index and event['u'] == current and len(event['gradient']) == 10,
            'Gradient cache/accepted state mismatch')
        gradient = event['gradient']; norm = math.hypot(*gradient)
        close(event['gradient_norm'], norm, 'Measured gradient norm mismatch')
        verify_objective(event['objective'], current, anchor, prep)
        check(event['objective']['forward_cache_reused'] is True and event['objective']['backward_evaluation_count'] == 1,
            'Gradient must reuse accepted forward cache')
        group = [trial for trial in trials if trial['iteration'] == index]
        check(len(group) <= 3, 'Armijo trial budget exceeded')
        if norm == 0: check(not group, 'Zero gradient must stop without trial')
        accepted_trial = None
        for number, trial in enumerate(group, 1):
            check(accepted_trial is None and trial['trial'] == number and trial['u_pre'] == current
                and trial['step_size'] == (1/8)/(2**(number-1)), 'Fixed bounded trial sequence mismatch')
            candidate = trial['u_trial']; proposal = [x-trial['step_size']*g/norm for x, g in zip(current, gradient)]
            verify_box_projection(proposal, candidate)
            close(trial['update_norm'], math.dist(current, candidate), 'Actual projected displacement mismatch')
            close(trial['loss_before'], event['objective']['loss_total'], 'Trial accepted-cache loss mismatch')
            rhs = trial['loss_before']+1e-4*math.fsum(g*(new-old) for g, new, old in zip(gradient, candidate, current))
            close(trial['armijo_rhs'], rhs, 'Armijo directional bound mismatch')
            close(trial['armijo_tolerance'], 128*sys.float_info.epsilon*
                max(1., abs(trial['loss_before']), abs(trial['objective']['loss_total']), abs(rhs)),
                'Declared float64 Armijo tolerance mismatch')
            verify_objective(trial['objective'], candidate, anchor, prep)
            check(trial['objective']['forward_cache_reused'] is False and trial['objective']['backward_evaluation_count'] == 0,
                'Trial objective reused or differentiated the wrong cache')
            expected = trial['objective']['loss_total'] <= rhs+trial['armijo_tolerance']
            check(trial['accepted'] is expected, 'Armijo acceptance mismatch')
            objectives.append(trial['objective']); used_trials += 1
            if expected: accepted_trial = trial
        if accepted_trial:
            step = steps[len(accepted)]
            check(step['step'] == len(accepted)+1 and step['u_pre'] == current
                and step['u_post'] == accepted_trial['u_trial'] and step['anchor'] == anchor
                and step['gradient'] == gradient and step['trial'] == accepted_trial['trial'], 'Accepted step/trial mismatch')
            close(step['loss_after'], accepted_trial['objective']['loss_total'], 'Accepted loss mismatch')
            current = step['u_post']; accepted.append(step)
        else:
            check(index == len(gradients), 'Optimizer continued after bounded stop')
    check(used_trials == len(trials) and accepted == steps and stage['u'] == current and valid_u(current),
        'Final accepted parameters/trial coverage mismatch')
    if not prep['no_information']:
        verify_objective(stage['final_objective'], current, anchor, prep)
        check(stage['final_objective']['forward_cache_reused'] is True
            and stage['final_objective']['backward_evaluation_count'] == 0
            and stage['final_objective']['derivative_triangular_solve_count'] == 0,
            'Final objective must reuse accepted cache without backward work')
        close(stage['final_objective']['loss_total'], objectives[-1 if trials and trials[-1]['accepted'] else 0]['loss_total']
            if not steps else steps[-1]['loss_after'], 'Final objective must be last accepted cache')
        reason = stage['stop_reason']
        check(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'ZERO_PROJECTED_STEP', 'ARMIJO_BUDGET_EXHAUSTED'),
            'Unknown bounded solver stop reason')
        if reason == 'MAX_ITERATIONS': check(len(gradients) == 4 and len(steps) == 4, 'Premature maximum-iteration stop')
        if reason == 'ZERO_GRADIENT': check(gradients and gradients[-1]['gradient_norm'] == 0, 'False zero-gradient stop')
        if reason == 'ZERO_PROJECTED_STEP':
            last = gradients[-1]; number = len([t for t in trials if t['iteration'] == last['iteration']])+1
            check(1 <= number <= 3 and last['gradient_norm'] > 0, 'False projected-step stop')
            proposed = [x-(1/8)/(2**(number-1))*g/last['gradient_norm'] for x, g in zip(current, last['gradient'])]
            verify_box_projection(proposed, current)
        if reason == 'ARMIJO_BUDGET_EXHAUSTED':
            check(gradients and len([t for t in trials if t['iteration'] == len(gradients)]) == 3
                and not trials[-1]['accepted'], 'False exhausted-Armijo stop')
    check(stage['inner_objective_evaluation_count'] == len(objectives), 'Actual forward objective count mismatch')
    for key in ('inner_head_fit_count', 'inner_factorization_count'):
        check(stage[key] == sum(value[key] for value in objectives), 'Actual forward cost mismatch: '+key)
    check(stage['transport_forward_evaluation_count'] == stage['inner_head_fit_count']+stage['final_head_fit_count'],
        'Actual inner/final transport forward call mismatch')
    check(stage['derivative_triangular_solve_count'] == sum(g['objective']['derivative_triangular_solve_count'] for g in gradients),
        'Actual cached adjoint solve count mismatch')
    check(stage['u_changed_from_anchor'] is (current != anchor)
        and stage['nonzero_projected_update_count'] == len(steps), 'Adapter update evidence mismatch')
    close(stage['u_update_norm'], math.dist(current, anchor), 'Final anchor distance mismatch')
    check(stage['persistent_state_bytes'] == stage['head_state_bytes']+stage['lineage_state_bytes']+stage['adapter_state_bytes']+stage['prototype_state_bytes']
        and stage['adapter_state_bytes'] == 80 and stage['trainable_parameter_count'] == 10
        and stage['effective_parameter_count'] == 9 and stage['fit_seconds'] >= 0 and stage['score_seconds'] >= 0,
        'Full deployed state/parameter/cost mismatch')
    final = stage['final_fit']; baseline_stage = entry['stages'][0 if prep['state'] == 'B' else -1]
    check(final['interaction_centered_trace'] == baseline_stage['interaction_centered_trace'], 'Original LocalRidge trace target changed')
    check(0 <= final['normal_equation_residual'] <= final['numerical_tolerance']
        and 0 <= final['trace_relative_error'] <= final['numerical_tolerance'], 'Uncertified final head solve')


def verify_record(record, split, old):
    finite_tree(record); check(record['scope'] == SCOPE and record['query_rows_used'] == record['source_rows_used'] == 0, 'Forbidden parent access')
    verify_baseline(baseline_view(record), split, old)
    if split['k'] == 1:
        check(all(record[key] == 0 for key in COUNTERS[4:]) and record['persistent_state_bytes'] == 0, 'True K1 fabricated training')
        return [], [], dict(oof=None, proxy=None), []
    old = sorted(old); classes = sorted(split['registered_classes'])
    labels = {pid: split['registered_classes'][label] for pid, label in zip(split['support_ids'], split['support_labels'])}
    counts = dict.fromkeys(COUNTERS[4:], 0); logs = []; events = []; training = []; maximum = 0
    for entry in record['folds']+record['oneshot_proxy']['trials']:
        check(set(entry['paths']) == set(PATHS), 'Three fixed paths required')
        common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
        evidence = {name: dict(common, b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
        fresh = assess_paths(evidence, old)
        for name in PATHS:
            check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]), 'Fixed-score paired result mismatch')
        reuse = classes == old
        check(entry['c_reuses_b_candidates'] is reuse and entry['paths']['R_transport_seq']['b_scores'] == entry['paths']['R_transport_reset']['b_scores'], 'Shared B/N0 mismatch')
        if reuse:
            for name in PATHS: check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'N0 must reuse B scores exactly')
        if entry['train_k'] == 1:
            check(all(entry['paths'][name] == entry['paths']['R0'] for name in PATHS), 'Proxy trainK1 must exactly equal R0')
        check([p['state'] for p in entry['preparations']] == (['B'] if reuse else ['B', 'C']), 'Preparation sharing mismatch')
        expected_stages = ['B_transport']+([] if reuse else ['C_transport', 'C_reset'])
        check([s['state'] for s in entry['candidate_stages']] == expected_stages, 'Candidate stage coverage/order mismatch')
        by_stage = {stage['state']: stage for stage in entry['candidate_stages']}; by_prep = {prep['state']: prep for prep in entry['preparations']}
        for base in entry['stages']:
            logs.append(dict(event='BASE_FIT', **base)); counts['baseline_head_fit_count'] += 1; counts['head_fit_count'] += 1
            counts['baseline_factorization_count'] += base['factorization_calls']; counts['factorization_count'] += base['factorization_calls']
            counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += base['held_physical_count']
        for prep in entry['preparations']:
            verify_preparation(prep, entry, labels, old); logs.append(dict(event='TRANSPORT_PREPARATION', **prep)); counts['transport_preparation_count'] += 1
            for key in PREPARATION_COUNTERS: counts[key] += prep[key]
            for stage in (s for s in entry['candidate_stages'] if s['preparation_ref'] == prep['state']):
                verify_candidate(stage, prep, entry, by_stage['B_transport']); logs.append(dict(event='CANDIDATE_FIT', **stage))
                counts['transport_stage_count'] += 1
                counts['trained_transport_stage_count'] += int(stage['optimizer_steps'] > 0)
                for key in STAGE_COUNTERS: counts[key] += stage[key]
                counts['head_fit_count'] += stage['inner_head_fit_count']+stage['final_head_fit_count']
                counts['factorization_count'] += stage['inner_factorization_count']+stage['final_factorization_count']
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += stage['held_physical_count']
                small_training = compact_event(dict(u=stage['u'], anchor=stage['anchor'],
                    no_information=stage['no_information'], optimizer_steps=stage['optimizer_steps'],
                    nonzero_projected_update_count=stage['nonzero_projected_update_count'],
                    u_changed_from_anchor=stage['u_changed_from_anchor'], u_update_norm=stage['u_update_norm'],
                    final_objective=stage['final_objective']))
                training.append(dict(split_id=record['split_id'], scope=entry['scope'], fold=entry['fold'], trial=entry['trial'],
                    k=record['k'], new_count=record['new_count'], state=stage['state'], train_k=entry['train_k'],
                    **small_training, stop_reason=stage['stop_reason'],
                    steps=[compact_event(step) for step in stage['steps']],
                    gradients=[compact_event(value) for value in stage['gradients']],
                    trials=[compact_event(value) for value in stage['trials']],
                    evidence_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION',
                    full_vectors_source='Original row probe/fit_trace.jsonl and training_events.jsonl; use row/split/scope/fold-or-trial/state keys'))
        per_stage = {name: dict(steps=0, trials=0, gradients=0, initial=0, final=0) for name in by_stage}
        prepared_seen = set()
        for event in entry['training_events']:
            check(event['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION' and event['source_validation'] is None,
                'Invented validation event')
            check(event['scope'] == entry['scope'] and event['fold'] == entry['fold']
                and event['outer_trial'] == entry['trial'], 'Training event outer-parent coordinates mismatch')
            name = event['state']; kind = event['event']
            if kind == 'TRANSPORT_INNER_PREPARED':
                prep_name = name.removesuffix('_prepare'); index = event['inner_fold']; key = (prep_name, index)
                check(prep_name in by_prep and key not in prepared_seen
                    and 0 <= index < len(by_prep[prep_name]['inner_folds']), 'Unknown/duplicate inner preparation event')
                expected_inner = by_prep[prep_name]['inner_folds'][index]
                check(all(event.get(field) == value for field, value in expected_inner.items()), 'Inner preparation binding mismatch')
                prepared_seen.add(key); continue
            check(name in by_stage, 'Unknown training event stage'); stage = by_stage[name]; seen = per_stage[name]
            mapping = {'TRANSPORT_GRADIENT': 'gradients', 'TRANSPORT_TRIAL': 'trials', 'TRANSPORT_STEP': 'steps'}
            if kind in mapping:
                field = mapping[kind]; index = seen[field]
                check(index < len(stage[field]), 'Training event exceeds full solver trace length')
                mismatched = [key for key, value in stage[field][index].items() if event.get(key) != value]
                check(not mismatched, 'Training event/full solver trace mismatch: '+name+'/'+kind+'/'+str(index)+
                    ' fields='+','.join(mismatched))
                seen[field] += 1
            elif kind == 'TRANSPORT_INITIAL':
                check(seen['initial'] == 0 and event['u'] == stage['anchor']
                    and event['objective'] == stage['initial_objective'], 'Initial accepted-cache event mismatch')
                seen['initial'] += 1
            elif kind == 'TRANSPORT_FIT':
                check(seen['final'] == 0 and event['u'] == stage['u'] and event['final_objective'] == stage['final_objective']
                    and event['stop_reason'] == stage['stop_reason'], 'Final solver event mismatch')
                for key in STAGE_COUNTERS: check(event[key] == stage[key], 'Final event actual cost mismatch')
                seen['final'] += 1
            else: raise ValueError('Unexpected transport training event kind')
        for name, stage in by_stage.items():
            seen = per_stage[name]
            check(seen['final'] == 1 and seen['initial'] == int(not stage['no_information'])
                and all(seen[field] == len(stage[field]) for field in ('steps', 'trials', 'gradients')),
                'Full solver event coverage mismatch')
        check(prepared_seen == {(p['state'], index) for p in entry['preparations'] for index in range(len(p['inner_folds']))},
            'Inner preparation event coverage mismatch')
        events.extend(entry['training_events']); counts['sequence_paths'] += 1
        for name, stage_name in (('R_transport_seq', 'B_transport' if reuse else 'C_transport'), ('R_transport_reset', 'B_transport' if reuse else 'C_reset')):
            check(entry['deployment_C_state_bytes'][name] == by_stage[stage_name]['persistent_state_bytes'], 'Deployment state bytes mismatch')
        maximum = max(maximum, *entry['deployment_C_state_bytes'].values())
    check(all(record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == maximum, 'Actual parent totals mismatch')
    check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'), 'OOF physical pooling mismatch')
    check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy parent-first mean mismatch')
    return logs, events, dict(oof={name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=record['oneshot_proxy']['parent_mean_metrics']), training


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.)+record['fit_seconds']
    for row in logs:
        group = row['event'].lower()
        for key, value in row.items():
            if key.endswith('_seconds') and key != 'preparation_seconds' and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group+'_'+key+'_sum'
                resources[name] = resources.get(name, 0.)+value
        if row['event'] == 'CANDIDATE_FIT':
            name = 'training_steps_seconds_sum'
            resources[name] = resources.get(name, 0.)+sum(step['step_seconds'] for step in row['steps'])
            for field in ('baseline_binding_distance_evaluation_count', 'baseline_binding_factorization_count'):
                name = 'candidate_fit_'+field+'_sum'; resources[name] = resources.get(name, 0)+row[field]
            for field, values in (('objective_forward_seconds_sum',
                    ([] if row['initial_objective'] is None else [row['initial_objective']])+[t['objective'] for t in row['trials']]),
                    ('objective_backward_seconds_sum', [g['objective'] for g in row['gradients']])):
                resources[field] = resources.get(field, 0.)+sum(value['objective_seconds'] for value in values)
            forward = ([] if row['initial_objective'] is None else [row['initial_objective']])+[t['objective'] for t in row['trials']]
            resources['inner_forward_seconds_sum'] = resources.get('inner_forward_seconds_sum', 0.)+sum(
                fold['forward_seconds'] for value in forward for fold in value['inner_folds'])
            resources['inner_adjoint_seconds_sum'] = resources.get('inner_adjoint_seconds_sum', 0.)+sum(
                fold['adjoint_seconds'] for value in row['gradients'] for fold in value['objective']['inner_folds'])
            resources['final_head_forward_seconds_sum'] = resources.get('final_head_forward_seconds_sum', 0.)+(
                row['final_fit']['forward_seconds'] if row['final_head_fit_count'] else 0.)
        if row['event'] == 'TRANSPORT_PREPARATION':
            prototypes = [fold['prototypes'] for fold in row['inner_folds']]+[row['full_prototypes']]
            resources['prototype_construction_seconds_sum'] = resources.get('prototype_construction_seconds_sum', 0.)+sum(
                value['prototype_seconds'] for value in prototypes)
        for key in ('persistent_state_bytes', 'adapter_state_bytes', 'prototype_state_bytes', 'head_state_bytes', 'lineage_state_bytes', 'prepared_numeric_state_bytes', 'optimizer_state_bytes', 'transient_distance_bytes'):
            if key in row: resources[group+'_maximum_'+key] = max(resources.get(group+'_maximum_'+key, 0), row[key])


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec; validate_spec(spec)
    root = Path(run_root or spec['execution']['remote_run_root']); out = Path(output)
    if out.exists(): raise FileExistsError(out)
    launch, complete, state = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    check(launch['spec'] == spec and complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == 4
        and complete['episodes'] == 160 and complete['commit'] == launch['commit'], 'Full four-row pilot incomplete')
    check(set(state) == {row['row_id'] for row in spec['rows']} and all(v['status'] == STATUS for v in state.values()), 'Incomplete row state')
    check(all(v.get('query_access') is False and v.get('source_sample_access') is False for v in (launch, complete)), 'Forbidden run access')
    lanes = []
    # Verify ALL completion/source bindings before opening any outer-held score trace.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root/row['row_id']/'probe'
        marker = verify_marker(lane/'probe_complete.json', spec, row); startup = read(lane/'startup.json'); check_bind(startup, row, co)
        check(startup['config'] == dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
            and startup['scope'] == marker['scope'] == SCOPE and startup['episodes'] == 40
            and startup['producer_episodes'] == co['expected_split_count'], 'Startup/config mismatch')
        for value in (startup, marker):
            check(all(value.get('channel', {}).get(key) == expected for key, expected in CHANNEL.items())
                and value['scenarios'] == SCENARIOS and value['query_rows_used'] == value['source_rows_used'] == 0
                and value['truth_read'] is False, 'Channel or forbidden access mismatch')
            check(value['payload_audit']['new_source_payload_bytes'] == value['payload_audit']['new_ground_statistics_bytes'] == 0,
                'Unexpected additional ground data payload')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
            and startup['actual_A'] is None and startup['adapted_state_inherited'] is True
            and startup['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION', 'Invented or forbidden adaptation state')
        feature_root = Path(startup['support_features']); check(feature_root == Path(row['support_features']), 'Unexpected support cache reference')
        feature, plan, provenance = [read(feature_root/name) for name in ('features_complete.json', 'support_splits.json', 'checkpoint_provenance.json')]
        for value in (feature, plan):
            check(value['capsule_id'] == co['capsule_id'] and value['checkpoint_sha256'] == row['expected_checkpoint_sha256'], 'Producer binding mismatch')
        check_bind(feature, row, co)
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == co['expected_split_count']
            and provenance == startup['provenance'] and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH'
            and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance'] == [], 'Producer provenance mismatch')
        old = feature['classes']; check(len(old) == 6, 'Fixed six-old-class pilot required')
        chosen = selected_tasks([(s, None, None) for s in plan['splits']], co['selection'], old)
        lanes.append((row, lane, marker, {s['split_id']: s for s, _, _ in chosen}, old, startup))
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; training = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    resource_strata = {name: {} for name in ('by_k_new_count', 'by_receiver_scene', 'by_model_cohort', 'by_row')}
    for row, lane, marker, expected, old, startup in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stages = iter(jsonlines(lane/'fit_stages.jsonl'))
        event_stream = iter(jsonlines(lane/'training_events.jsonl')); small_events = iter(jsonlines(lane/'training_events_compact.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact length mismatch'); sid = record['split_id']
            check(sid in expected and sid not in seen, 'Unexpected/duplicate parent'); seen.add(sid)
            logs, events, measurements, train = verify_record(record, expected[sid], old); accumulate_resources(resources, record, logs)
            resource_keys = dict(by_k_new_count=(record['k'], record['new_count']),
                by_receiver_scene=(row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']),
                by_model_cohort=(row['seeds']['model'], row['cohort'], record['k'], record['new_count']), by_row=(row['row_id'],))
            for group_name, group_key in resource_keys.items():
                cell = resource_strata[group_name].setdefault(group_key, {})
                accumulate_resources(cell, record, logs); cell['parents'] = cell.get('parents', 0)+1
                for key in COUNTERS[4:]: cell[key] = cell.get(key, 0)+record[key]
            check(small == compact_record(record), 'Compact evidence mismatch')
            for value in logs: check(next(stages, None) == dict(compact_event(value), split_id=sid), 'Stage stream mismatch')
            for value in events:
                full = dict(value, split_id=sid)
                check(next(event_stream, None) == full and next(small_events, None) == compact_event(full), 'Full/compact training event mismatch')
            training.extend(dict(row_id=row['row_id'], model_seed=row['seeds']['model'], cohort=row['cohort'],
                receiver=record['receiver'], scenario=record['scenario'], **value) for value in train)
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1)
            counts['oof_episodes'] += int(record['k'] > 1); counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for key in COUNTERS[4:]: counts[key] += record[key]
            for diagnostic, paths in measurements.items():
                if paths is None: paths = {name: dict.fromkeys(METRICS) for name in PATHS}
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                for name, metrics in paths.items():
                    _add(strata['overall'], (diagnostic, name, population), metrics)
                    _add(strata['by_k_new_count'], (diagnostic, name, record['k'], record['new_count']), metrics)
                    _add(strata['by_receiver_scene'], (diagnostic, name, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), metrics)
                    _add(strata['by_model_cohort'], (diagnostic, name, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), metrics)
        check(seen == set(expected) and next(stages, None) is next(event_stream, None) is next(small_events, None) is None, 'Missing parents or extra events')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row actual totals mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
    check(all(coverage[key] == complete[key] for key in COUNTERS), 'Run totals mismatch')
    check(all(coverage[key] == value for key, value in EXACT_COUNTS.items())
        and all(coverage[key] <= value for key, value in MAX_COUNTS.items()), 'Fixed pilot actual budget mismatch')
    check(complete['finished'] >= launch['started'], 'Run wall-clock bound mismatch')
    resources.update(run_wall_seconds=complete['finished']-launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _ in lanes
            if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        additional_ground_data_payload_bytes=0, additional_ground_statistics_bytes=0,
        remote_code_release_archive_bytes=None, remote_code_release_archive_reason='Release transport artifact is separate from method data payload and is not measured by this evaluator',
        separate_inner_kernel_seconds=None,
        unmeasured_reason='CPU only; no deployment package/transfer measured; RSS null if runtime unavailable; objective wall times include their measured forward or cached backward work; standalone kernel timing is not measured',
        fitted_state_scope='Full numeric transport adapter, two prototype tables, LocalRidge head, original/adapted b/a, raw support and labels retained for inheritance; excludes original frozen Phase1 model delivery, external shared feature archive, Python and serialization overhead',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment']) for row, _, _, _, _, startup in lanes],
        interpretation='Stage duration sums measure work, not concurrent elapsed time. Shared preparation is charged once. Final scoring timings are measured on the stated CPU/thread configuration.')
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    resource_dimensions = dict(by_k_new_count=('k', 'new_count'),
        by_receiver_scene=('cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('model_seed', 'cohort', 'k', 'new_count'), by_row=('row_id',))
    resource_tables = {name: [dict(zip(resource_dimensions[name], key), **value) for key, value in sorted(groups.items())]
                       for name, groups in resource_strata.items()}
    summary = dict(status=SUMMARY_STATUS, scope=SCOPE, run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=coverage, resources=resources, resource_statistics=resource_tables,
        algorithm=PROBE_CONFIG, channel=CHANNEL, statistics=tables,
        old_class_count=6, actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        query_rows_used=0, source_rows_used=0, training_stage_count=len(training),
        raw_training_sources=[dict(row_id=row['row_id'], fit_trace=str(lane/'fit_trace.jsonl'),
            full_training_events=str(lane/'training_events.jsonl'), compact_training_events=str(lane/'training_events_compact.jsonl'))
            for row, lane, _, _, _, _ in lanes],
        interpretation=[
            'R0 remains original LocalRidge. R_transport_seq is the declared sequential prototype transport candidate; R_transport_reset changes C initialization and proximal anchor to zero.',
            'Sequential and reset share one trained B and one C preparation; each inner head uses only its physical inner-train subset. C sequential inherits B adapter and anchor; C reset starts at zero.',
            'Only class RMS of pooled physical inner-support half squared-hinge margin losses supervises the transport adapter. Inner-held labels are part of training and are never called independent validation.',
            'Training permits four normalized projected gradient iterations and three Armijo trials per iteration. Every accepted/rejected trial and bounded stop is retained; final parameters use the last accepted cache without best-step selection or extra final objective forward.',
            'True K1 has no fitting or held score. Proxy trainK1 uses exact R0 identity forward for every path; no K1 benefit is claimed.',
            'A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.',
            'OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.',
            'Old-only reuse and repeated old support across new-count rows are not independent observations.',
            'Ten stored transport parameters (nine constrained degrees of freedom) do not imply low total compute or an 80-byte deployed model; full prototype/binding/head state, preparation memory and every trial/adjoint solve are counted.',
            'Every parameter/gradient coordinate and all initial, gradient, trial and accepted-step events are verified against the full raw streams. Compact logs retain separate beta/eta summaries; full vectors remain in the referenced row logs.',
            '10/1/3 percentage-point ideal directions are descriptive and impose no automatic promotion gate.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out/'summary.json', summary)
    for name, rows in tables.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, rows in resource_tables.items():
        fields = sorted({key for row in rows for key in row})
        with (out/('resources_'+name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in rows)
    with (out/'training_objectives.jsonl').open('x', encoding='utf-8') as stream:
        for row in training: stream.write(json.dumps(row, allow_nan=False)+'\n')
    with (out/'training_objectives.csv').open('x', encoding='utf-8', newline='') as stream:
        compact_training = [compact_event(row) for row in training]
        fields = sorted({key for row in compact_training for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in compact_training)
    lines = ['# PrototypeTransport-LocalRidge support pilot', '',
        '完整160 parent、4 row、三路径均已核验；旧类固定6个。A与B−A为N/A。R_transport_seq为预声明顺序主线，R_transport_reset仅作继承对照。', '',
        '| 诊断 | 路径 | K | 新类数 | A旧 | B0旧 | B旧 | C旧类列 | C旧 | C新 | H | B−B0 | B−C旧类列 | 新竞争损失 | 注册下降 | 新旧差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = {}
    for row in tables['by_k_new_count']:
        cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    display = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy', 'C_old_accuracy', 'C_new_accuracy',
        'C_h', 'support_adaptation_B_minus_B0', 'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    for key, values in sorted(cells.items()):
        lines.append('| '+' | '.join([str(value) for value in key]+['N/A' if values[m] is None else f'{100*values[m]:.3f}' for m in display])+' |')
    lines += ['', '准确率为百分数，差值为百分点。全部分层、相对R0变化与六类正确性转换见CSV。内部训练目标、参数、梯度、全部回溯试探和停止原因见training_objectives，不把内部训练准确率称为独立验证。', '',
        f"实际有效训练阶段{coverage['trained_transport_stage_count']}个、投影更新{coverage['optimizer_steps']}次、内层head拟合{coverage['inner_head_fit_count']}次、新增最终head拟合{coverage['final_head_fit_count']}次。",
        f"实测运行墙钟{resources['run_wall_seconds']:.3f} s；完整适配器和分类头状态、线程与分项工作量见summary.json。", '']
    lines += ['- '+line for line in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8'); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
