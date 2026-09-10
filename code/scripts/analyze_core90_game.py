"""Read complete CORE90 game artifacts; never score target data or promote runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


STREAMS = {'epochs': 'logs.jsonl', 'actions': 'game_actions.jsonl',
           'audits': 'game_audit.jsonl', 'curriculum': 'curriculum_events.jsonl',
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
    config = (calibration or {}).get('curriculum', {})
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
    totals = {
        'training_wall_seconds': resource.get('total_seconds'),
        'source_evaluation_wall_seconds': (scores or {}).get('resources', {}).get('total_wall_seconds'),
        'epoch_wall_seconds_sum': counted(epochs, 'epoch_seconds', evidence['epochs']),
        'main_field_evaluations': counted(actions, 'field_evaluations', evidence['actions']),
        'logged_training_forward_calls': counted(actions, 'forward_calls', evidence['actions']),
        'main_backward_evaluations': counted(actions, 'main_backward_evaluations', evidence['actions']),
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
                             for k in ('accuracy', 'macro_recall', 'macro_f1', 'worst_rx_accuracy', 'worst_tx_accuracy')}
        costs = {k: {'baseline': base['resources'].get(k), 'candidate': run['resources'].get(k)}
                 for k in ('training_wall_seconds', 'main_field_evaluations', 'sample_presentations')}
        for value in costs.values():
            value['difference'] = value['candidate']-value['baseline'] if finite(value['candidate']) and finite(value['baseline']) else None
        pairs.append({'row': run['row'], 'seed': run['seed'], 'baseline': baseline,
                      'status': 'source_metrics_available' if base['source_performance'] and run['source_performance'] else 'pending_scores',
                      'source_fraction_differences': deltas, 'actual_costs': costs,
                      'matched_compute_claim': False, 'reason': 'same_seed_pairing_does_not_establish_same_actual_compute'})
    return pairs


def report(runs, baseline='B0'):
    return {'schema': 'core90_game_full_artifact_analysis_v1', 'runs': runs,
            'paired_differences': paired_differences(runs, baseline),
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
