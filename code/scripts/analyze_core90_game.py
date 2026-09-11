"""Read complete CORE90 game artifacts; never score target data or promote runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import statistics
from pathlib import Path


STREAMS = {'epochs': 'logs.jsonl', 'actions': 'game_actions.jsonl',
           'audits': 'game_audit.jsonl', 'curriculum': 'curriculum_events.jsonl',
           'capability_audits': 'capability_audit.jsonl',
           'response': 'response_tracking.jsonl', 'jacobian': 'jacobian_audit.jsonl'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_artifact(path, *, stream=False):
    """Read every nonblank line, preserving parse failures as incomplete evidence."""
    path = Path(path)
    if not path.exists():
        return ([] if stream else None), {'path': str(path), 'status': 'pending', 'records': None}
    errors, rows = [], []
    if stream:
        with path.open(encoding='utf-8-sig') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected object')
                    rows.append(row)
                except (ValueError, TypeError) as error:
                    errors.append({'line': number, 'error': str(error)})
        value = rows
    else:
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError('expected object')
        except (ValueError, TypeError) as error:
            value = None
            errors.append({'error': str(error)})
    return value, {'path': str(path), 'status': 'incomplete' if errors else 'read_complete',
                   'records': len(rows) if stream else int(value is not None), 'errors': errors}


def counted(rows, key, evidence):
    """Missing field or unreadable/missing stream means unknown, never zero."""
    if evidence['status'] != 'read_complete' or any(not finite(r.get(key)) for r in rows):
        return None
    return sum(r[key] for r in rows)


def first_capability(audits, events, epochs, calibration):
    config = (calibration or {}).get('curriculum') or {}
    threshold_keys = ('identity_enter', 'margin_enter', 'lag_max', 'consistency_enter')
    if not all(finite(config.get(k)) for k in threshold_keys):
        return {'status': 'pending', 'reason': 'complete_frozen_capability_thresholds_unavailable'}
    first = next((r for r in audits if r.get('valid') and r.get('identity_valid', True)
                  and not r.get('collapsed', False)
                  and all(finite(r.get(k)) for k in ('identity', 'margin', 'G_lag', 'consistency'))
                  and r['identity'] >= config['identity_enter'] and r['margin'] >= config['margin_enter']
                  and r['G_lag'] <= config['lag_max'] and r['consistency'] >= config['consistency_enter']), None)
    confirmed = next((r for r in events if r.get('changed') and r.get('level', 0) > r.get('previous_level', 0)), None)
    if first is None:
        return {'status': 'not_observed', 'first_confirmed_curriculum_step': confirmed.get('step') if confirmed else None}
    # audit elapsed_seconds is its duration, NOT elapsed training time.
    upper = next((r.get('elapsed_seconds') for r in epochs
                  if finite(r.get('total_step')) and r['total_step'] > first['step']
                  and finite(r.get('elapsed_seconds'))), None)
    return {'status': 'observed', 'first_threshold_step': first['step'],
            'training_seconds_upper_bound': upper,
            'time_basis': 'containing_epoch_end_conservative_upper_bound' if upper is not None else 'unavailable',
            'first_confirmed_curriculum_step': confirmed.get('step') if confirmed else None,
            'thresholds': {k: config[k] for k in threshold_keys}}


def analyze_run(directory, *, row=None, seed=None):
    directory = Path(directory)
    streams, evidence = {}, {}
    for name, filename in STREAMS.items():
        streams[name], evidence[name] = read_artifact(directory / filename, stream=True)
    objects = {}
    for name, filename in {'config': 'resolved_config.json', 'resources': 'resource_summary.json',
                           'completion': 'completion.json', 'calibration': 'source_calibration.json',
                           'coverage': 'coverage_audit.json',
                           'scores': 'source_final_eval/source_scores.json'}.items():
        objects[name], evidence[name] = read_artifact(directory / filename)
    cfg, resource = objects['config'] or {}, objects['resources'] or {}
    epochs, actions = streams['epochs'], streams['actions']
    duplicate_steps = [s for s, n in Counter(r.get('step') for r in actions).items() if n > 1]
    duplicate_epochs = [s for s, n in Counter(r.get('epoch') for r in epochs).items() if n > 1]
    if duplicate_steps or duplicate_epochs:
        evidence['actions']['status'] = 'incomplete'
    complete_actions = evidence['actions']['status'] == 'read_complete'
    accepted = [a for a in actions if a.get('accepted') is True]
    total_steps = max((e['total_step'] for e in epochs if finite(e.get('total_step'))), default=None)
    closed = complete_actions and all(isinstance(a.get('accepted'), bool) for a in actions) and total_steps is not None and len(actions) == total_steps and set(a.get('step') for a in actions) == set(range(total_steps))
    mechanism = {
        'requested_action_counts': dict(Counter(a['requested_action'] for a in actions)) if complete_actions and all('requested_action' in a for a in actions) else None,
        'decided_action_counts': dict(Counter(a.get('action', 'unknown') for a in actions)) if complete_actions else None,
        'committed_correct_actions': sum(a.get('action') == 'CORRECT' and a.get('field_evaluations', 0) > 1 for a in accepted) if complete_actions else None,
        'accepted_action_counts': dict(Counter(a.get('action', 'unknown') for a in accepted)) if complete_actions else None,
        'rejected_action_counts': dict(Counter(a.get('action', 'unknown') for a in actions if a.get('accepted') is False)) if complete_actions else None,
        'accepted_main_updates': len(accepted) if complete_actions else None,
        'actual_solver_updates': dict(Counter(a.get('algorithm', a.get('main_solver', 'unknown')) for a in accepted)) if complete_actions else None,
        'committed_head_steps': counted(actions, 'committed_head_steps', evidence['actions']),
        'requested_head_steps': counted(actions, 'requested_head_steps', evidence['actions']),
        'failed_main_updates': sum(a.get('accepted') is False for a in actions) if complete_actions else None,
        'failure_stages': dict(Counter(a.get('failure_stage') or 'unspecified' for a in actions if a.get('accepted') is False)) if complete_actions else None,
        'action_reasons': dict(Counter(a.get('reason', 'unspecified') for a in actions)) if complete_actions else None,
    }
    for name, predicate in [('audits', lambda r: r.get('valid') is True),
                            ('curriculum', lambda r: r.get('changed') is True),
                            ('response', lambda r: r.get('accepted') is True),
                            ('jacobian', lambda r: r.get('valid') is True)]:
        mechanism[name] = {'observations': len(streams[name]) if evidence[name]['status'] == 'read_complete' else None,
                           'activated_or_valid': sum(predicate(r) for r in streams[name]) if evidence[name]['status'] == 'read_complete' else None}
    sources = objects['scores']
    scores = sources if sources and sources.get('complete') is True and sources.get('source_only') is True else None
    prediction_paths = [directory/'source_final_eval'/name for name in
                        ('source_predictions.jsonl', 'source_prediction_manifest.json', 'source_truth.jsonl')]
    if scores and all(p.exists() for p in prediction_paths):
        try:
            try:
                from scripts.analyze_core90_conditional_robustness import closed_source_metrics
            except ModuleNotFoundError:
                from analyze_core90_conditional_robustness import closed_source_metrics
            supplements = closed_source_metrics(*prediction_paths)
            for scene, metrics in supplements.items(): scores['scenes'][scene].update(metrics)
            evidence['detailed_source_metrics'] = {'status': 'read_complete'}
        except (ValueError, KeyError, TypeError) as error:
            evidence['detailed_source_metrics'] = {'status': 'incomplete', 'error': str(error)}
    else:
        evidence['detailed_source_metrics'] = {'status': 'pending'}
        if scores:
            for scene in scores.get('scenes', {}).values():
                recalls = [v for rx in scene.get('per_rx', {}).values() for v in rx.get('per_tx_accuracy', {}).values() if finite(v)]
                scene.setdefault('worst_rx_tx_accuracy', min(recalls) if recalls else None)
    totals = {
        'training_wall_seconds': resource.get('total_seconds'),
        'source_evaluation_wall_seconds': (scores or {}).get('resources', {}).get('total_wall_seconds'),
        'epoch_wall_seconds_sum': counted(epochs, 'epoch_seconds', evidence['epochs']),
        'main_field_evaluations': counted(actions, 'field_evaluations', evidence['actions']),
        'logged_training_forward_calls': counted(actions, 'forward_calls', evidence['actions']),
        'main_backward_evaluations': counted(actions, 'main_backward_evaluations', evidence['actions']),
        'telemetry_backward_evaluations': counted(actions, 'telemetry_backward_evaluations', evidence['actions']),
        'extra_head_forward_backward_evaluations': counted(actions, 'extra_head_forward_backward_evaluations', evidence['actions']),
        'audit_model_forwards': counted(streams['audits'], 'model_forwards', evidence['audits']),
        'audit_gradient_backward_evaluations': counted(streams['audits'], 'gradient_backward_evaluations', evidence['audits']),
        'audit_seconds': counted(streams['audits'], 'elapsed_seconds', evidence['audits']),
        'audit_probe_steps': counted(streams['audits'], 'probe_steps', evidence['audits']),
        'capability_probe_steps': counted([a.get('capability', {}) for a in streams['audits']], 'probe_steps', evidence['audits']),
        'response_seconds': counted(streams['response'], 'elapsed_seconds', evidence['response']),
        'response_hvp_iterations': counted(streams['response'], 'hvp_iterations', evidence['response']),
        'response_model_forwards': counted(streams['response'], 'model_forwards', evidence['response']),
        'response_head_forwards': counted(streams['response'], 'head_forwards', evidence['response']),
        'jacobian_seconds': counted(streams['jacobian'], 'elapsed_seconds', evidence['jacobian']),
        'jacobian_field_evaluations': counted(streams['jacobian'], 'field_evaluations', evidence['jacobian']),
        'jacobian_model_forwards': counted(streams['jacobian'], 'model_forwards', evidence['jacobian']),
        'jacobian_head_forwards': counted(streams['jacobian'], 'head_forwards', evidence['jacobian']),
        'sample_presentations': counted(actions, 'sample_count', evidence['actions']),
        'satellite_presentations': counted(actions, 'satellite_count', evidence['actions']),
        'recorded_budget_lifetime': resource.get('budget', {}).get('lifetime'),
        'recorded_resource_summary': resource or None,
        'peak_training_cuda_allocated_bytes': max((e['peak_memory_bytes'] for e in epochs if finite(e.get('peak_memory_bytes'))), default=None),
    }
    totals['training_plus_source_eval_wall_seconds'] = sum(totals[k] for k in ('training_wall_seconds', 'source_evaluation_wall_seconds')) if all(finite(totals[k]) for k in ('training_wall_seconds', 'source_evaluation_wall_seconds')) else None
    full_epochs = bool(resource.get('full_budget_completed'))
    expected = ['epochs', 'actions']
    if cfg.get('game_no_audit') is False:
        expected.append('audits')
    if cfg.get('game_response_tracking'):
        expected.append('response')
    if cfg.get('game_jacobian_interval') and not cfg.get('game_no_audit'):
        expected.append('jacobian')
    # Curriculum calibration may never succeed; absence is pending evidence,
    # while its absence must never be counted as zero successful upgrades.
    if cfg.get('game_curriculum') == 'capability':
        expected.append('curriculum')
    expected_complete = all(evidence[k]['status'] == 'read_complete' for k in expected)
    return {'run': str(directory.resolve()), 'row': row or cfg.get('game_row') or directory.name,
            'seed': seed if seed is not None else cfg.get('seed'), 'configuration': cfg,
            'status': 'artifacts_complete' if closed and full_epochs and scores and objects['completion'] and expected_complete else 'pending_or_partial',
            'evidence': evidence, 'log_closure': {'actions_match_epoch_total': closed, 'duplicate_steps': duplicate_steps,
                                               'duplicate_epochs': duplicate_epochs},
            'mechanisms': mechanism, 'resources': totals,
            'quality_and_exposure': quality_and_exposure(streams, evidence),
            'failure_diagnostics': failure_diagnostics(actions, stream_complete=closed),
            'capability': first_capability(streams['audits'], streams['curriculum'], epochs, objects['calibration']) if evidence['audits']['status'] == 'read_complete' else {'status': 'pending', 'reason': 'source_audit_stream_missing_or_incomplete'},
            'coverage': objects['coverage'],
            'deployment': {'path': str(directory/'deployment.pth'),
                           'status': 'present_not_loaded_or_verified' if (directory/'deployment.pth').exists() else 'pending'},
            'source_performance': scores, 'source_performance_status': 'available' if scores else 'pending',
            'scientific_verdict': 'SOURCE_ONLY_NO_PROMOTION', 'target_evidence': 'not_analyzed'}


def paired_differences(runs, baseline='B0'):
    pairs = []
    by_seed = {}
    for run in runs:
        key = (run['seed'], run['row'])
        if key in by_seed:
            raise ValueError('Duplicate seed/row; pairing would be ambiguous: ' + str(key))
        by_seed[key] = run
    for run in runs:
        if run['row'] == baseline:
            continue
        base = by_seed.get((run['seed'], baseline))
        if base is None:
            pairs.append({'row': run['row'], 'seed': run['seed'], 'status': 'pending_baseline'})
            continue
        deltas = {}
        for scene in ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak'):
            a = (base['source_performance'] or {}).get('scenes', {}).get(scene, {})
            b = (run['source_performance'] or {}).get('scenes', {}).get(scene, {})
            deltas[scene] = {k: b[k]-a[k] if finite(a.get(k)) and finite(b.get(k)) else None
                             for k in ('accuracy', 'macro_recall', 'macro_f1', 'worst_rx_accuracy', 'worst_tx_accuracy', 'worst_rx_tx_accuracy')}
        costs = {k: {'baseline': base['resources'].get(k), 'candidate': run['resources'].get(k)}
                 for k in ('training_wall_seconds', 'main_field_evaluations', 'sample_presentations')}
        for value in costs.values():
            value['difference'] = value['candidate']-value['baseline'] if finite(value['candidate']) and finite(value['baseline']) else None
        pairs.append({'row': run['row'], 'seed': run['seed'], 'baseline': baseline,
                      'status': 'source_metrics_available' if base['source_performance'] and run['source_performance'] else 'pending_scores',
                      'source_fraction_differences': deltas, 'actual_costs': costs,
                      'matched_compute_claim': False, 'reason': 'same_seed_pairing_does_not_establish_same_actual_compute'})
    return pairs


def failure_diagnostics(actions, *, stream_complete):
    """Observable pathology timelines, never a root-cause claim from final accuracy."""
    histograms = [r for r in actions if isinstance(r.get('prediction_histogram'), list)
                  and r['prediction_histogram'] and all(finite(v) and v >= 0 for v in r['prediction_histogram'])
                  and sum(r['prediction_histogram']) > 0]
    collapse = next((r for r in histograms if max(r['prediction_histogram']) / sum(r['prediction_histogram']) >= .99), None)
    first = dict(status='observed' if collapse else 'not_observed_at_logged_samples' if histograms else 'UNKNOWN',
                 criterion='descriptive dominant predicted class share >= 0.99; not a causal or identity-collapse verdict',
                 histogram_observations=len(histograms), first_observed_step=collapse.get('step') if collapse else None,
                 first_observed_epoch=collapse.get('epoch') if collapse else None,
                 exact_first_collapse_established=bool(collapse and stream_complete and len(histograms) == len(actions)))
    cosine_values = [r['history_next_gradient_cosine'] for r in actions if finite(r.get('history_next_gradient_cosine'))]
    history = [r for r in actions if type(r.get('optimistic_history_used')) is bool]
    return dict(first_collapse=first, complete_stream=stream_complete,
        optimistic=dict(history_used_observations=sum(r['optimistic_history_used'] for r in history) if history else None,
            history_observations=len(history),
            history_reset_reasons=dict(Counter(r['history_reset_reason'] for r in actions if r.get('history_reset_reason'))),
            gradient_predictive_value=dict(status='observed' if cosine_values else 'UNKNOWN', values=cosine_values,
                mean=statistics.mean(cosine_values) if cosine_values else None),
            direction_records=[{k:r[k] for k in ('step','epoch','raw_optimistic_cosine','tx_adv_gradient_ratio','prediction_entropy') if k in r}
                               for r in actions if any(k in r for k in ('raw_optimistic_cosine','tx_adv_gradient_ratio','prediction_entropy'))],
            stage_boundary_records=[r for r in actions if r.get('stage_boundary') is True]),
        root_cause='UNKNOWN; observations alone do not authorize B4/B7 repair experiments')


def quality_and_exposure(streams, evidence):
    audits = streams['audits']
    quality = None
    if evidence['audits']['status'] == 'read_complete':
        quality = {}
        for estimand in ('lag', 'cross_tx_readout', 'gradient', 'capability'):
            records = (streams.get('capability_audits', []) if estimand=='capability' else
                       [r.get(estimand, {}) for r in audits if r.get('schema') == 'game_audit_v2'])
            quality[estimand] = dict(status_counts=dict(Counter(r.get('status', 'UNAVAILABLE') for r in records)),
                reason_counts=dict(Counter(code for r in records for code in r.get('reason_codes', []))),
                v2_observations=len(records))
        quality['legacy_v1_observations'] = sum(r.get('schema') != 'game_audit_v2' for r in audits)
    events = streams['curriculum']
    changes = sum(r.get('effective_policy_changed') is True for r in events) if evidence['curriculum']['status'] == 'read_complete' and all('effective_policy_changed' in r for r in events) else None
    # Actual counts must be explicitly logged; configuration probabilities are not exposure.
    exposure = {}
    for name, predicate in [('pre_E80_bn', lambda r: r.get('epoch', 0) < 80), ('E80_plus_direct_ce', lambda r: r.get('epoch', 0) >= 80)]:
        rows = [r for r in streams['actions'] if predicate(r)]
        valid = evidence['actions']['status'] == 'read_complete' and all('epoch' in r and isinstance(r.get('actual_scenario_counts'), dict) for r in rows)
        counts = Counter()
        if valid:
            for r in rows:
                if not all(finite(v) and v >= 0 for v in r['actual_scenario_counts'].values()): valid = False; break
                counts.update(r['actual_scenario_counts'])
        exposure[name] = dict(counts) if valid else None
    return dict(probe_quality=quality, effective_policy_changes=changes, actual_exposure=exposure)


def factorial_differences(runs):
    indexed = {}
    for r in runs:
        key = r['seed'], r['row']
        if key in indexed: raise ValueError('Duplicate seed/row in factorial analysis')
        indexed[key] = r
    per_seed = []
    aggregate = {}
    for seed in sorted({r['seed'] for r in runs if r['row'].startswith('V2_')}, key=str):
        missing = [c for c in 'ABCDEF' if (seed, 'V2_'+c) not in indexed]
        if missing:
            per_seed.append(dict(seed=seed, status='pending_rows', missing_rows=missing)); continue
        scenes = {}
        for scene in ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak'):
            scenes[scene] = {}
            for metric in ('accuracy', 'macro_f1', 'worst_rx_accuracy', 'worst_tx_accuracy', 'worst_rx_tx_accuracy'):
                values = [(indexed[seed, 'V2_'+c].get('source_performance') or {}).get('scenes', {}).get(scene, {}).get(metric) for c in 'ABCDEF']
                if all(finite(v) for v in values):
                    a,b,c,d,e,f = values
                    contrasts = dict(b8_interaction=(d-c)-(b-a), eg_interaction=(f-e)-(b-a),
                                     d_minus_b=d-b, ordinary_adv=b-a, b8_zero_adv=c-a)
                    for name, value in contrasts.items(): aggregate.setdefault((scene, metric, name), []).append(value)
                    scenes[scene][metric] = contrasts
                else: scenes[scene][metric] = None
        per_seed.append(dict(seed=seed, status='paired_rows_present', scenes=scenes))
    summaries = [dict(scene=s, metric=m, contrast=c, n=len(v), mean=statistics.mean(v),
                      sample_std=statistics.stdev(v) if len(v)>1 else None, values=v)
                 for (s,m,c),v in sorted(aggregate.items())]
    return dict(per_seed=per_seed, aggregate=summaries,
                precision_note='paired seed descriptive dispersion only; no target selection or statistical certainty claim')


def report(runs, baseline='B0'):
    return {'schema': 'core90_game_full_artifact_analysis_v1', 'runs': runs,
            'paired_differences': paired_differences(runs, baseline),
            'factorial_differences': factorial_differences(runs),
            'budget_interpretation': {
                'fixed_epochs': 'Compare at recorded epochs/sample presentations; solver/probe compute may differ.',
                'fixed_wall_time': 'Epoch-end stopping can overshoot; compare actual measured time, not requested caps.',
                'time_to_capability': 'Frozen source thresholds only; epoch elapsed supplies conservative upper bound when event time is absent.',
                'resource_scope': 'Training wall time already includes audits/response/Jacobian; do not add component seconds again. Missing counters remain null; logged forwards are not proof of all compute.'},
            'scientific_verdict': 'SOURCE_ONLY_NO_PROMOTION'}


def markdown(result):
    lines = ['# CORE90完整产物分析', '', '结论边界：仅源域证据，不作目标域泛化或晋级声明。缺失产物为pending，不按0计。', '',
             '|行|seed|状态|主更新|头step|训练秒|源域评分|', '|---|---|---|---|---|---|---|']
    for r in result['runs']:
        def cell(v): return 'pending' if v is None else str(v)
        lines.append('|'+ '|'.join(map(cell,[r['row'],r['seed'],r['status'],r['mechanisms']['accepted_main_updates'],r['mechanisms']['committed_head_steps'],r['resources']['training_wall_seconds'],r['source_performance_status']]))+'|')
    lines += ['', '完整逐动作激活、失败、能力首次达标上界、四场景/RX指标和同seed差异见同名JSON。', '',
              '三预算口径分别是固定训练轮次、实际训练时间、达到冻结源能力阈值的时间。相同seed、相同预算上限不等于实际计算成本匹配；训练总秒数已包含内部审计，不能重复累加。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=[], help='Run directory; repeatable')
    parser.add_argument('--matrix', help='Matrix manifest with runs and config.output_dir')
    parser.add_argument('--baseline', default='B0')
    parser.add_argument('--output', required=True, help='New report filename stem')
    args = parser.parse_args(argv)
    specs = [(p, None, None) for p in args.run]
    if args.matrix:
        manifest = json.loads(Path(args.matrix).read_text(encoding='utf-8-sig'))
        specs += [(r['config']['output_dir'], r['row'], r['seed']) for r in manifest['runs']]
    if not specs:
        parser.error('Provide --run or --matrix')
    result = report([analyze_run(p, row=row, seed=seed) for p,row,seed in specs], args.baseline)
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, value in (('.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)), ('.md', markdown(result))):
        with Path(str(stem)+suffix).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(value+'\n')
    return 0


if __name__ == '__main__':
    main()
