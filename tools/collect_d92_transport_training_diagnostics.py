"""Reconcile complete frozen PrototypeTransport training logs without scoring.

Standard library only; optional SSH stdin transport is read-only. Parent traces,
query, outer-held scores, registries and historical reports are never opened.
"""
import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

STATUS = 'PROTOTYPE_TRANSPORT_PROBE_COMPLETE'
COORDS = ('row_id', 'split_id', 'scope', 'fold', 'outer_trial', 'state', 'train_k')
STATES = dict(B_transport='B', C_transport='C_seq', C_reset='C_reset')
EXACT = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
    sequence_paths=1760, baseline_head_fit_count=3168, transport_preparation_count=3168,
    transport_stage_count=4576, diagnostic_fit_count=0)
MAXIMUM = dict(trained_transport_stage_count=936, optimizer_steps=3744,
    inner_objective_evaluation_count=12168, inner_head_fit_count=36504, inner_factorization_count=36504,
    final_head_fit_count=936, final_factorization_count=936, head_fit_count=40608, factorization_count=40608,
    baseline_factorization_count=3168, derivative_triangular_solve_count=22464)
ADDITIVE = ('optimizer_steps', 'optimizer_iterations', 'inner_objective_evaluation_count',
    'inner_head_fit_count', 'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count',
    'derivative_triangular_solve_count', 'backward_evaluation_count', 'transport_forward_evaluation_count',
    'accepted_trial_count', 'rejected_trial_count', 'trial_count', 'trial_attempt_count')
PREP_COSTS = ('prototype_construction_count', 'prototype_distance_evaluation_count', 'prepared_distance_evaluation_count')
COUNTERS = tuple(dict.fromkeys(tuple(EXACT)+tuple(MAXIMUM)+ADDITIVE+PREP_COSTS+
    ('final_score_evaluation_count', 'final_score_physical_count')))
H = math.log(2)/2
METRICS = ('loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy',
    'inner_training_correct_count', 'inner_training_physical_count', 'inner_training_margin_mean',
    'inner_training_margin_min', 'gradient_norm', 'step_size', 'update_norm', 'objective_seconds',
    'inner_forward_seconds', 'inner_adjoint_seconds', 'armijo_slack', 'class_loss_min', 'class_loss_max',
    'source_validation', 'prediction_change_count')
PATTERNS = {key: re.compile(pattern) for key, pattern in dict(
    error=r'(?i)\b(?:error|failed|failure|exception)\b', warning=r'(?i)\b(?:warning|warn)\b',
    traceback=r'(?i)\btraceback\b', oom_or_killed=r'(?i)\b(?:out of memory|oom|killed)\b',
    nonfinite_text=r'(?i)(?<![A-Za-z_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z_])',
    recovery_or_resume=r'(?i)\b(?:resume|resumed|recover|recovery|restart|retry)\b',
    solver_stop=r'"stop_reason"\s*:', configuration=r'"(?:config|algorithm)"\s*:',
    environment=r'"(?:python|hardware|blas_environment)"\s*:', launch_command=r'"argv"\s*:').items()}


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            require(line.strip(), f'Blank record {path}:{number}')
            yield json.loads(line)


def csvlines(path):
    with Path(path).open(encoding='utf-8', newline='') as source: yield from csv.DictReader(source)


def csv_value(value):
    if value is None: return 'N/A'
    if isinstance(value, (dict, list)): return json.dumps(value, sort_keys=True, allow_nan=False)
    return str(value)


def check_csv(row, source, path):
    require(None not in row and set(source) <= set(row), 'Malformed/omitted CSV columns: '+str(path))
    require(all(row[key] == csv_value(source.get(key)) for key in row), 'CSV/JSON mismatch: '+str(path))


def scalars(value):
    if not isinstance(value, dict): return value
    return {key: scalars(item) for key, item in value.items()
            if isinstance(item, dict) or item is None or type(item) in (str, bool, int, float)}


def compact_event(event):
    """Independent copy of the frozen 10-coordinate compact event schema."""
    result = scalars(event)
    for key, value in event.items():
        if isinstance(value, (list, tuple)) and len(value) == 10 and all(type(v) in (int, float) for v in value):
            blocks = {name: dict(norm=math.hypot(*block), sum=math.fsum(block), minimum=min(block), maximum=max(block))
                for name, block in (('beta', value[:5]), ('eta', value[5:]))}
            result[key+'_summary'] = dict(count=10, norm=math.hypot(*value), minimum=min(value), maximum=max(value), blocks=blocks)
    if isinstance(event.get('projection_zero_sum_residuals'), (list, tuple)):
        result['projection_zero_sum_residuals_by_block'] = dict(zip(('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'), event['projection_zero_sum_residuals']))
    for prefix, objective in (('', event), ('objective_', event.get('objective')), ('final_', event.get('final_objective'))):
        if prefix and isinstance(objective, dict): result.update({prefix+key: value for key, value in scalars(objective).items()})
        if isinstance(objective, dict) and isinstance(objective.get('inner_folds'), list):
            folds = objective['inner_folds']
            if folds and all('held_training_correct_count' in f and 'held_physical_count' in f for f in folds):
                n = sum(f['held_physical_count'] for f in folds)
                result[prefix+'inner_training_accuracy'] = sum(f['held_training_correct_count'] for f in folds)/n if n else None
                result[prefix+'inner_training_physical_count'] = n
                margins = [f.get('held_training_margin_mean') for f in folds]; minima = [f.get('held_training_margin_min') for f in folds]
                result[prefix+'inner_training_margin_mean'] = sum(m*f['held_physical_count'] for m, f in zip(margins, folds))/n if n and all(v is not None for v in margins) else None
                result[prefix+'inner_training_margin_min'] = min(minima) if minima and all(v is not None for v in minima) else None
    return result


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def close(actual, expected, message):
    require(number(actual) is not None and number(expected) is not None and
        math.isclose(actual, expected, rel_tol=1e-10, abs_tol=2e-12), message)


def stage_key(row_id, row, training=False):
    trial = row.get('outer_trial') if training else row.get('trial')
    scope, fold = row.get('scope'), row.get('fold')
    require(scope in ('support_oof', 'support_oneshot_proxy'), 'Unexpected support training scope')
    require((type(fold) is int and trial is None) if scope == 'support_oof' else
        (fold is None and type(trial) is int), 'Missing/ambiguous physical fold/anchor')
    require(row.get('row_id', row_id) == row_id and row.get('split_id') and type(row.get('train_k')) is int,
        'Missing stage source coordinates')
    return (row_id, row['split_id'], scope, fold, trial, row['state'], row['train_k'])


def identity(key): return dict(zip(COORDS, key))


def valid_u(value):
    tolerance = 128*sys.float_info.epsilon*5*H
    return (isinstance(value, list) and len(value) == 10 and all(number(v) is not None for v in value)
        and max(abs(v) for v in value[:5]) <= math.pi/4+tolerance
        and max(abs(v) for v in value[5:]) <= H+tolerance and abs(math.fsum(value[5:])) <= tolerance)


def verify_projection(proposal, actual):
    require(valid_u(actual), 'Transport box/eta zero-sum constraint mismatch')
    for v, u in zip(proposal[:5], actual[:5]): close(u, min(math.pi/4, max(-math.pi/4, v)), 'Beta projection mismatch')
    free = [v-u for v, u in zip(proposal[5:], actual[5:]) if -H < u < H]
    lower = max((v-u for v, u in zip(proposal[5:], actual[5:]) if u <= -H), default=-math.inf)
    upper = min((v-u for v, u in zip(proposal[5:], actual[5:]) if u >= H), default=math.inf)
    tolerance = 128*sys.float_info.epsilon*max(1., max(abs(v) for v in proposal[5:]))
    if free:
        multiplier = math.fsum(free)/len(free)
        require(max(abs(v-multiplier) for v in free) <= tolerance and lower <= multiplier+tolerance
            and upper >= multiplier-tolerance, 'Eta projection KKT mismatch')
    else: require(lower <= upper+tolerance, 'Eta projection active-set KKT mismatch')


def vector_metrics(value, prefix, anchor=None):
    result = {}
    for name, block in (('', value), ('beta_', None if value is None else value[:5]),
                        ('eta_', None if value is None else value[5:])):
        label = prefix+'_'+name
        measurements = dict(norm=None, rms=None, minimum=None, maximum=None, sum=None, nonzero_coordinates=None)
        if block is not None:
            norm = math.hypot(*block)
            measurements.update(norm=norm, rms=norm/math.sqrt(len(block)), minimum=min(block), maximum=max(block),
                                sum=math.fsum(block), nonzero_coordinates=sum(v != 0 for v in block))
        result.update({label+key: item for key, item in measurements.items()})
    result[prefix+'_anchor_distance'] = math.dist(value, anchor) if value is not None and anchor is not None else None
    return result


def metrics(event=None, u=None, anchor=None):
    event = event or {}; objective = event.get('objective', event); compact = compact_event(objective)
    result = {field: number(compact.get(field, event.get(field))) for field in METRICS}
    folds = objective.get('inner_folds', [])
    result['inner_training_correct_count'] = sum(f['held_training_correct_count'] for f in folds) if folds else None
    for output, field in (('inner_forward_seconds', 'forward_seconds'), ('inner_adjoint_seconds', 'adjoint_seconds')):
        values = [f.get(field) for f in folds]
        result[output] = sum(v for v, f in zip(values, folds) if field != 'forward_seconds' or f['head_fit_count']) if values and all(number(v) is not None for v in values) else None
    values = objective.get('class_loss_values')
    result['class_loss_min'] = min(values) if values else None; result['class_loss_max'] = max(values) if values else None
    if 'armijo_rhs' in event: result['armijo_slack'] = event['armijo_rhs']+event['armijo_tolerance']-event['loss_after']
    result.update(vector_metrics(u, 'u', anchor)); result.update(vector_metrics(event.get('gradient'), 'gradient'))
    return result


def validate_objective(value, u, anchor, prep):
    require(value['loss_scope'] == 'CLASS_RMS_INNER_HELD_MARGIN_PLUS_PROXIMAL'
        and value['classes'] == prep['classes'] and len(value['inner_folds']) == len(prep['inner_folds']), 'Objective scope/fold/class mismatch')
    c = len(prep['classes']); sums = [0.]*c; counts = [0]*c
    for fold, original in zip(value['inner_folds'], prep['inner_folds']):
        require(fold['training_physical_ids'] == original['training_physical_ids']
            and fold['held_physical_ids'] == original['held_physical_ids']
            and fold['held_physical_count'] == original['held_physical_count'], 'Inner objective physical binding mismatch')
        require(len(fold['class_margin_loss_sums']) == len(fold['class_physical_counts']) == c, 'Missing per-class risk evidence')
        for i, (loss, count) in enumerate(zip(fold['class_margin_loss_sums'], fold['class_physical_counts'])):
            require(number(loss) is not None and loss >= 0 and type(count) is int and count >= 0, 'Invalid class loss/count')
            sums[i] += loss; counts[i] += count
    require(sum(counts) == prep['train_physical_count'] and all(counts) and value['class_held_counts'] == counts, 'Pooled physical class count mismatch')
    require(len(value['class_loss_values']) == c, 'Missing class losses')
    losses = [loss/count for loss, count in zip(sums, counts)]
    for actual, expected in zip(value['class_loss_values'], losses): close(actual, expected, 'Pooled class loss mismatch')
    close(value['loss_data'], math.hypot(*losses)/math.sqrt(c), 'Pooled class RMS mismatch')
    close(value['loss_proximal'], math.fsum((x-y)**2 for x, y in zip(u, anchor))/(2*sum(counts)), 'Proximal scale mismatch')
    close(value['loss_total'], value['loss_data']+value['loss_proximal'], 'Loss components mismatch')
    for total, field in (('inner_head_fit_count', 'head_fit_count'), ('inner_factorization_count', 'factorization_count'),
        ('transport_forward_evaluation_count', 'transport_forward_evaluation_count'), ('derivative_triangular_solve_count', 'derivative_triangular_solve_count')):
        require(value[total] == sum(f[field] for f in value['inner_folds']), 'Objective actual cost mismatch: '+total)
    require(value['inner_objective_evaluation_count'] == int(not value['forward_cache_reused']), 'Objective cache/forward count mismatch')
    if value['forward_cache_reused']:
        require(value['inner_head_fit_count'] == value['inner_factorization_count'] == value['transport_forward_evaluation_count'] == 0,
            'Cache reuse refitted a head')


def validate_solver(full):
    prep = full['preparation']; anchor = full['anchor']; u = anchor[:]
    require(valid_u(anchor) and full['u_anchor'] == anchor, 'Invalid anchor')
    steps, gradients, trials = full['steps'], full['gradients'], full['trials']; initial = full['initial_objective']
    require(full['optimizer_steps'] == len(steps) <= 4 and full['optimizer_iterations'] == len(gradients) <= 4
        and full['trial_attempt_count'] == full['trial_count'] == len(trials) <= 12
        and full['accepted_trial_count'] == len(steps) and full['rejected_trial_count'] == len(trials)-len(steps), 'Actual bounded solver counts mismatch')
    noinfo = prep['train_k'] == 1 or len(prep['classes']) == 1
    require(full['no_information'] == prep['no_information'] == noinfo, 'Physical information-stage rule mismatch')
    if noinfo:
        require(not steps and not gradients and not trials and initial is None and full['final_objective'] is None
            and full['stop_reason'] == prep['no_information_reason'], 'No-information stage invented training')
    else:
        require(initial is not None and not initial['forward_cache_reused'] and initial['backward_evaluation_count'] == 0, 'Missing initial forward')
        validate_objective(initial, u, anchor, prep)
    accepted = []; used = 0; forward = [] if noinfo else [initial]; curve = []
    if initial: curve.append(dict(kind='initial', iteration=0, trial=None, accepted=None, metrics=metrics(initial, u, anchor)))
    for iteration, gradient in enumerate(gradients, 1):
        require(gradient['iteration'] == iteration and gradient['u'] == u and len(gradient['gradient']) == 10, 'Accepted-cache gradient coordinates mismatch')
        norm = math.hypot(*gradient['gradient']); close(gradient['gradient_norm'], norm, 'Gradient norm mismatch')
        obj = gradient['objective']; validate_objective(obj, u, anchor, prep)
        require(obj['forward_cache_reused'] and obj['backward_evaluation_count'] == 1
            and obj['gradient'] == gradient['gradient'], 'Gradient did not reuse accepted cache/coordinates')
        curve.append(dict(kind='gradient', iteration=iteration, trial=None, accepted=None, metrics=metrics(gradient, u, anchor)))
        group = [trial for trial in trials if trial['iteration'] == iteration]; require(len(group) <= 3, 'Trial budget exceeded')
        if norm == 0: require(not group, 'Zero gradient must not evaluate trials')
        accepted_trial = None
        for index, trial in enumerate(group, 1):
            require(accepted_trial is None and trial['trial'] == index and trial['u_pre'] == u
                and trial['step_size'] == .125*(.5**(index-1)), 'Armijo trial sequence mismatch')
            proposed = [x-trial['step_size']*g/norm for x, g in zip(u, gradient['gradient'])]
            verify_projection(proposed, trial['u_trial']); close(trial['update_norm'], math.dist(u, trial['u_trial']), 'Trial displacement mismatch')
            close(trial['loss_before'], obj['loss_total'], 'Trial accepted-cache loss mismatch')
            rhs = obj['loss_total']+1e-4*math.fsum(g*(v-x) for g, v, x in zip(gradient['gradient'], trial['u_trial'], u))
            close(trial['armijo_rhs'], rhs, 'Armijo directional bound mismatch')
            tolerance = 128*sys.float_info.epsilon*max(1., abs(trial['loss_before']), abs(trial['loss_after']), abs(rhs))
            close(trial['armijo_tolerance'], tolerance, 'Frozen Armijo tolerance mismatch')
            validate_objective(trial['objective'], trial['u_trial'], anchor, prep)
            require(not trial['objective']['forward_cache_reused'] and trial['objective']['backward_evaluation_count'] == 0, 'Trial cache/gradient mismatch')
            close(trial['loss_after'], trial['objective']['loss_total'], 'Trial loss mismatch')
            require(type(trial['accepted']) is bool and trial['accepted'] == (trial['loss_after'] <= rhs+tolerance), 'Armijo accept/reject mismatch')
            used += 1; forward.append(trial['objective'])
            curve.append(dict(kind='trial', iteration=iteration, trial=index, accepted=trial['accepted'], metrics=metrics(trial, trial['u_trial'], anchor)))
            if trial['accepted']: accepted_trial = trial
        if accepted_trial:
            step = steps[len(accepted)]
            require(step['step'] == len(accepted)+1 and step['iteration'] == iteration and step['trial'] == accepted_trial['trial']
                and step['u_pre'] == u and step['u_post'] == accepted_trial['u_trial'] and step['gradient'] == gradient['gradient']
                and step['objective'] == accepted_trial['objective'], 'Accepted step/gradient/trial mismatch')
            u = step['u_post']; accepted.append(step)
        else: require(iteration == len(gradients), 'Solver continued after bounded stop')
    require(accepted == steps and used == len(trials) and full['u'] == u and valid_u(u), 'Final accepted state/trial coverage mismatch')
    if not noinfo:
        final = full['final_objective']; validate_objective(final, u, anchor, prep)
        require(final['forward_cache_reused'] and final['backward_evaluation_count'] == final['derivative_triangular_solve_count'] == 0, 'Final cache performed extra work')
        close(final['loss_total'], steps[-1]['loss_after'] if steps else initial['loss_total'], 'Final cache is not last accepted objective')
        reason = full['stop_reason']; require(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'ZERO_PROJECTED_STEP', 'ARMIJO_BUDGET_EXHAUSTED'), 'Unknown solver stop reason')
        if reason == 'MAX_ITERATIONS': require(len(gradients) == len(steps) == 4, 'Premature maximum stop')
        if reason == 'ZERO_GRADIENT': require(gradients and gradients[-1]['gradient_norm'] == 0, 'False zero-gradient stop')
        if reason == 'ARMIJO_BUDGET_EXHAUSTED': require(len([t for t in trials if t['iteration'] == len(gradients)]) == 3 and not trials[-1]['accepted'], 'False Armijo exhaustion')
        if reason == 'ZERO_PROJECTED_STEP':
            last = gradients[-1]; index = len([t for t in trials if t['iteration'] == len(gradients)])+1
            require(1 <= index <= 3 and last['gradient_norm'] > 0, 'False projected-zero stop')
            verify_projection([x-.125*(.5**(index-1))*g/last['gradient_norm'] for x, g in zip(u, last['gradient'])], u)
        curve.append(dict(kind='final_cached', iteration=None, trial=None, accepted=None, metrics=metrics(final, u, anchor)))
    else: curve.append(dict(kind='no_information', iteration=None, trial=None, accepted=None, metrics=metrics(None, u, anchor)))
    require(full['inner_objective_evaluation_count'] == len(forward) and full['backward_evaluation_count'] == len(gradients), 'Actual forward/backward count mismatch')
    for field in ('inner_head_fit_count', 'inner_factorization_count'):
        require(full[field] == sum(obj[field] for obj in forward), 'Actual forward resource mismatch: '+field)
    require(full['derivative_triangular_solve_count'] == sum(g['objective']['derivative_triangular_solve_count'] for g in gradients), 'Actual adjoint solve count mismatch')
    require(full['transport_forward_evaluation_count'] == full['inner_head_fit_count']+full['final_head_fit_count'], 'Actual total forward count mismatch')
    require(full['u_changed_from_anchor'] == (u != anchor) and full['nonzero_projected_update_count'] == len(steps), 'Final update flag mismatch')
    close(full['u_update_norm'], math.dist(u, anchor), 'Final anchor distance mismatch')
    return curve


def add_stats(destination, values):
    for key, value in values.items():
        cell = destination.setdefault(key, dict(count=0, missing_count=0, sum=0., minimum=None, maximum=None))
        if value is None: cell['missing_count'] += 1
        else:
            cell['count'] += 1; cell['sum'] += value
            cell['minimum'] = value if cell['minimum'] is None else min(cell['minimum'], value)
            cell['maximum'] = value if cell['maximum'] is None else max(cell['maximum'], value)


def finish_stats(values):
    return {key: dict(count=v['count'], missing_count=v['missing_count'], mean=v['sum']/v['count'] if v['count'] else None,
                     minimum=v['minimum'], maximum=v['maximum']) for key, v in values.items()}


def structured_observations(value):
    counts = Counter()
    if isinstance(value, dict):
        counts['technical_failure_statuses'] += int(str(value.get('status', '')).upper() in ('FAILED', 'TECHNICAL_FAILURE'))
        counts['explicit_error_fields'] += sum(bool(value.get(k)) for k in ('error', 'error_type', 'failure_reason'))
        for child in value.values(): counts.update(structured_observations(child))
    elif isinstance(value, list):
        for child in value: counts.update(structured_observations(child))
    elif isinstance(value, float) and not math.isfinite(value): counts['numeric_nonfinite_values'] += 1
    elif isinstance(value, str) and value in ('NaN', 'Infinity', '-Infinity'): counts['encoded_nonfinite_values'] += 1
    return counts


def scan_text(path, expected_events=None, expected_stages=None):
    """Scan every line; never deserialize SUPPORT_PARENT_COMPLETE payloads."""
    path = Path(path); counts = Counter(); events = Counter(); lines = 0
    expected = iter(expected_events) if expected_events is not None else None
    stages = iter(expected_stages) if expected_stages is not None else None
    with path.open(encoding='utf-8') as source:
        for line in source:
            lines += 1
            for name, pattern in PATTERNS.items(): counts[name] += int(bool(pattern.search(line)))
            if line.startswith('PROTOTYPE_TRANSPORT_TRAINING '):
                row = json.loads(line.split(' ', 1)[1]); events[row['event']] += 1
                if expected is not None: require(row == next(expected, None), 'Text/event stream mismatch: '+str(path))
            elif line.startswith('{'):
                match = re.search(r'"event"\s*:\s*"([^"\\]+)"', line); event = match.group(1) if match else None
                if event: events[event] += 1
                if event in ('BASE_FIT', 'TRANSPORT_PREPARATION', 'CANDIDATE_FIT') and stages is not None:
                    require(json.loads(line) == next(stages, None), 'Text/stage stream mismatch: '+str(path))
    if expected is not None: require(next(expected, None) is None, 'Text log missing training events')
    if stages is not None: require(next(stages, None) is None, 'Text log missing fit stages')
    return dict(path=str(path), bytes=path.stat().st_size, lines=lines, marker_line_counts=dict(counts), events=dict(events))


def collect_lane(lane, row_id, algorithm):
    lane = Path(lane); names = ('fit_stages.jsonl', 'fit_stages.csv', 'training_events.jsonl', 'training_events_compact.jsonl', 'training_events_compact.csv')
    paths = {name: lane/name for name in names}; candidates = {}; preparations = {}; baseline = set(); counts = Counter(); observed = Counter(); resources = {}
    for row, csvrow in itertools.zip_longest(jsonlines(paths['fit_stages.jsonl']), csvlines(paths['fit_stages.csv'])):
        require(row is not None and csvrow is not None, 'Fit-stage stream length mismatch'); check_csv(csvrow, row, paths['fit_stages.csv'])
        observed.update(structured_observations(row)); key = stage_key(row_id, row); counts['fit_stage_records'] += 1
        if row['event'] == 'BASE_FIT':
            require(key not in baseline, 'Duplicate baseline fit'); baseline.add(key)
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += row['factorization_calls']
            counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += row['held_physical_count']
        elif row['event'] == 'TRANSPORT_PREPARATION':
            require(key not in preparations, 'Duplicate preparation'); preparations[key] = row; counts['transport_preparation_count'] += 1
            for field in PREP_COSTS: counts[field] += row[field]
        else:
            require(row['event'] == 'CANDIDATE_FIT' and row['state'] in STATES and key not in candidates, 'Unexpected/duplicate candidate fit')
            candidates[key] = row
        resource_key = (row['scope'], row['state'], row['train_k'])
        add_stats(resources.setdefault(resource_key, {}), {field: number(row.get(field)) for field in
            ('fit_seconds', 'score_seconds', 'fit_and_score_seconds', 'prepare_seconds', 'baseline_binding_seconds',
             'persistent_state_bytes', 'adapter_state_bytes', 'prototype_state_bytes', 'head_state_bytes', 'lineage_state_bytes',
             'prepared_numeric_state_bytes', 'transient_distance_bytes', 'optimizer_state_bytes')})
    pending = {}; completed = {}; inner = {}; events_count = Counter(); b_vectors = {}; bindings = {}; contexts = []
    for full, compact, csvrow in itertools.zip_longest(jsonlines(paths['training_events.jsonl']),
            jsonlines(paths['training_events_compact.jsonl']), csvlines(paths['training_events_compact.csv'])):
        require(full is not None and compact is not None and csvrow is not None, 'Training stream length mismatch')
        require(compact_event(full) == compact, 'Full/compact event mismatch'); check_csv(csvrow, compact, paths['training_events_compact.csv'])
        observed.update(structured_observations(full)); kind = full['event']; events_count[kind] += 1; key = stage_key(row_id, full, True)
        require(full.get('source_validation') is None and full.get('objective_scope') == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION', 'Forbidden training scope/validation')
        if kind == 'TRANSPORT_INNER_PREPARED':
            prepkey = key[:5]+(full['state'].removesuffix('_prepare'), key[6]); require(prepkey in preparations, 'Absent preparation reference')
            folds = inner.setdefault(prepkey, {}); require(full['inner_fold'] not in folds, 'Duplicate inner preparation')
            folds[full['inner_fold']] = full; continue
        require(key in candidates and key not in completed and full['mode'] == STATES[full['state']], 'Absent/finalized candidate event')
        if kind != 'TRANSPORT_FIT':
            require(kind in ('TRANSPORT_INITIAL', 'TRANSPORT_GRADIENT', 'TRANSPORT_TRIAL', 'TRANSPORT_STEP'), 'Unexpected transport event')
            pending.setdefault(key, []).append(full); continue
        require(full['status'] == 'TRANSPORT_STAGE_COMPLETE' and full['config'] == algorithm, 'Incomplete/unfrozen transport fit')
        # Event envelopes additionally contain prepared metadata. Compare every
        # overlapping field plus the required final-state schema, not that envelope.
        for field, value in compact.items():
            if field in candidates[key] and field not in ('event', 'outer_trial'):
                require(candidates[key][field] == value, 'Final event/candidate fit mismatch: '+field)
        for field in ADDITIVE+('mode', 'status', 'config', 'stop_reason', 'u_summary', 'anchor_summary',
                              'initial_objective', 'final_objective', 'preparation', 'persistent_state_bytes'):
            require(field in candidates[key] and candidates[key][field] == compact[field], 'Missing/mismatched final state: '+field)
        actual = pending.pop(key, []); expected = []
        if full['initial_objective'] is not None: expected.append(('TRANSPORT_INITIAL', dict(u=full['anchor'], anchor=full['anchor'], objective=full['initial_objective'])))
        for gradient in full['gradients']:
            expected.append(('TRANSPORT_GRADIENT', gradient))
            for trial in (t for t in full['trials'] if t['iteration'] == gradient['iteration']):
                expected.append(('TRANSPORT_TRIAL', trial))
                if trial['accepted']:
                    step = next(s for s in full['steps'] if s['iteration'] == gradient['iteration']); expected.append(('TRANSPORT_STEP', step))
        require(len(actual) == len(expected), 'Solver event coverage mismatch')
        for event, (kind, record) in zip(actual, expected):
            require(event['event'] == kind and all(event.get(k) == v for k, v in record.items()), 'Solver event order/full audit mismatch')
        prep = full['preparation']; prepkey = key[:5]+(full['state'][0], key[6]); require(prepkey in preparations, 'Absent shared preparation')
        for field, value in scalars(prep).items(): require(preparations[prepkey].get(field) == value, 'Preparation binding mismatch: '+field)
        physical = set(prep['training_physical_ids']); seen = set(); folded_events = inner.get(prepkey, {})
        require(len(folded_events) == len(prep['inner_folds']), 'Inner preparation event coverage mismatch')
        for folded in prep['inner_folds']:
            require(all(folded_events[folded['inner_fold']].get(k) == v for k, v in folded.items()), 'Inner preparation full audit mismatch')
            train, held = set(folded['training_physical_ids']), set(folded['held_physical_ids'])
            require(not train & held and train | held == physical and not seen & held and folded['prototypes']['training_physical_ids'] == folded['training_physical_ids'], 'Inner prototype/physical isolation mismatch')
            seen.update(held)
        require(not prep['inner_folds'] or seen == physical, 'Missing pooled physical ID')
        require(prep['prototype_construction_count'] == len(prep['inner_folds'])+1
            and prep['prototype_distance_evaluation_count'] == 3*(len(prep['inner_folds'])+1)
            and prep['prepared_distance_evaluation_count'] == 2*(len(prep['inner_folds'])+1), 'Prepared actual costs mismatch')
        proto = prep['full_prototypes']; inherited = full['mode'] != 'B'
        require(proto['training_physical_ids'] == prep['training_physical_ids']
            and proto['old_prototypes_bitwise_inherited'] == inherited
            and sorted(proto['inherited_old_prototype_classes']) == (sorted(prep['old_classes']) if inherited else []),
            'Final old-prototype inheritance binding mismatch')
        bkey = key[:5]+('B_transport', key[6])
        if full['mode'] == 'C_seq': require(bkey in b_vectors and full['anchor'] == b_vectors[bkey], 'Sequential B anchor mismatch')
        else: require(full['anchor'] == [0.]*10, 'B/reset anchor must be zero')
        if full['mode'] == 'B':
            b_vectors[bkey] = full['u']
            bindings[bkey] = (row_id, full['scope'], full['parent_k'], full['train_k'], tuple(prep['classes']), tuple(prep['training_physical_ids']))
        else: contexts.append(dict(B_stage=identity(bkey), C_stage=identity(key), mode=full['mode']))
        curve = validate_solver(full); initial = metrics(full['initial_objective'], full['anchor'], full['anchor']); final = metrics(full['final_objective'], full['u'], full['anchor'])
        stages = dict(identity=identity(key), mode=full['mode'], parent_k=full['parent_k'],
            class_count=len(full['classes']), old_class_count=len(full['old_classes']),
            information_stage=not full['no_information'], updated_stage=bool(full['steps']), stop_reason=full['stop_reason'],
            identity_forward=full['identity_forward'], initial=initial, final=final,
            final_minus_initial={field: final[field]-initial[field] if final[field] is not None and initial[field] is not None else None for field in METRICS},
            curve=curve, counts={field: full[field] for field in ADDITIVE},
            costs={field: full.get(field) for field in ('fit_seconds', 'persistent_state_bytes', 'adapter_state_bytes', 'prototype_state_bytes',
                'head_state_bytes', 'lineage_state_bytes', 'optimizer_state_bytes', 'trainable_parameter_count', 'effective_parameter_count',
                'baseline_binding_seconds', 'baseline_binding_distance_evaluation_count', 'baseline_binding_factorization_count')},
            score_seconds=candidates[key].get('score_seconds'), full_vectors_source=str(paths['training_events.jsonl']))
        completed[key] = stages; counts['transport_stage_count'] += 1; counts['information_stage_count'] += int(stages['information_stage'])
        counts['trained_transport_stage_count'] += int(stages['updated_stage'])
        for field in ADDITIVE: counts[field] += full[field]
        counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += candidates[key]['held_physical_count']
    require(not pending and set(completed) == set(candidates), 'Missing final candidate events')
    for key, prep in preparations.items():
        values = [f['prototypes'].get('prototype_seconds') for f in inner.get(key, {}).values()]
        values.append(prep['full_prototypes'].get('prototype_seconds'))
        add_stats(resources.setdefault((prep['scope'], prep['state'], prep['train_k']), {}),
            dict(prototype_construction_seconds=sum(values) if all(number(v) is not None for v in values) else None))
    path_new = {}
    for key, stage in completed.items():
        parentkey = key[:5]+key[6:]; path_new[parentkey] = max(path_new.get(parentkey, 0), stage['class_count']-stage['old_class_count'])
    for key, stage in completed.items(): stage['parent_new_count'] = path_new[key[:5]+key[6:]]
    groups = {}
    for key, binding in bindings.items(): groups.setdefault(binding, []).append(key)
    dedup = []
    for binding, keys in groups.items():
        dedup.append(dict(representative=identity(keys[0]), actual_fit_contexts=[identity(key) for key in keys],
            actual_fit_count=len(keys), repeated_context_count=len(keys)-1,
            final_parameter_variants=len({tuple(b_vectors[key]) for key in keys})))
    counts['head_fit_count'] = counts['baseline_head_fit_count']+counts['inner_head_fit_count']+counts['final_head_fit_count']
    counts['factorization_count'] = counts['baseline_factorization_count']+counts['inner_factorization_count']+counts['final_factorization_count']
    counts['sequence_paths'] = len({key[:5]+key[6:] for key in completed})
    texts = [scan_text(path, jsonlines(paths['training_events_compact.jsonl']), jsonlines(paths['fit_stages.jsonl']))
        for path in (lane/'training.log', lane.parent/'probe.log')]
    return dict(row_id=row_id, stages=list(completed.values()), coverage=dict(counts), event_counts=dict(events_count),
        B_reuse_contexts=contexts, B_physical_training_groups=dedup,
        deduplicated_B_stages=[completed[group_keys[0]] for group_keys in groups.values()],
        structured_observations=dict(observed), text_logs=texts,
        resource_groups=[dict(row_id=row_id, scope=k[0], state=k[1], train_k=k[2], metrics=finish_stats(v)) for k, v in sorted(resources.items())],
        inventory=[dict(path=str(p), bytes=p.stat().st_size) for p in paths.values()], failure_artifact_present=(lane/'probe_failed.json').exists())


def aggregate(stages):
    groups = {}; snapshots = {}; curves = {}; strata = {}; resources = {}
    for stage in stages:
        i = stage['identity']; key = (i['scope'], i['state'], i['train_k']); counts = groups.setdefault(key, Counter())
        counts['actual_stages'] += 1; counts['information_stages'] += int(stage['information_stage']); counts['updated_stages'] += int(stage['updated_stage'])
        counts['zero_update_information_stages'] += int(stage['information_stage'] and not stage['updated_stage']); counts.update(stage['counts'])
        stratum = (i['row_id'],)+key+(stage['parent_k'], stage['parent_new_count']); strata.setdefault(stratum, Counter()).update(dict(actual_stages=1, **stage['counts']))
        for name in ('initial', 'final', 'final_minus_initial'): add_stats(snapshots.setdefault(key+(name,), {}), stage[name])
        add_stats(resources.setdefault(key, {}), dict(stage['costs'], score_seconds=stage['score_seconds']))
        for point in stage['curve']:
            add_stats(curves.setdefault(key+(point['kind'], point['iteration'], point['trial'], point['accepted']), {}), point['metrics'])
    coords = lambda key: dict(scope=key[0], state=key[1], train_k=key[2])
    return dict(group_counts=[dict(coords(k), **v) for k, v in sorted(groups.items())],
        stratified_counts=[dict(row_id=k[0], **coords(k[1:]), parent_k=k[4], parent_new_count=k[5], **v) for k, v in sorted(strata.items())],
        snapshots=[dict(coords(k), timing=k[3], metrics=finish_stats(v)) for k, v in sorted(snapshots.items())],
        training_resource_groups=[dict(coords(k), metrics=finish_stats(v)) for k, v in sorted(resources.items())],
        curves=[dict(coords(k), kind=k[3], iteration=k[4], trial=k[5], accepted=k[6], metrics=finish_stats(v))
                for k, v in sorted(curves.items(), key=lambda item: repr(item[0]))])


def collect(run_root, run_log=None):
    root = Path(run_root); startup, complete, states = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    spec = startup['spec']; rows = spec['rows']; algorithm = spec['probe']['algorithm']
    require(complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == len(rows) == 4
        and complete['commit'] == startup['commit'], 'Complete fixed four-row pilot required')
    require(spec['probe']['exact_counts'] == EXACT and spec['probe']['maximum_counts'] == MAXIMUM, 'Frozen pilot budget mismatch')
    require(algorithm['method'] == 'D92-PrototypeTransportLocalRidge-v1' and algorithm['max_iterations'] == 4
        and algorithm['max_trials'] == 3 and algorithm['initial_step_size'] == .125 and algorithm['backtrack_factor'] == .5
        and algorithm['armijo_coefficient'] == 1e-4, 'Frozen transport optimizer mismatch')
    require(set(states) == {r['row_id'] for r in rows} and startup['query_access'] is complete['query_access'] is False
        and startup['source_sample_access'] is complete['source_sample_access'] is False, 'Forbidden/incomplete run metadata')
    metadata = []
    # Complete all source/config checks before opening any training stream.
    for row in rows:
        lane = root/row['row_id']/'probe'; marker = read(lane/'probe_complete.json'); launch = read(lane/'startup.json'); co = spec['probe']['cohorts'][row['cohort']]
        require(marker['status'] == states[row['row_id']]['status'] == STATUS and marker['algorithm'] == launch['config']['algorithm'] == algorithm, 'Lane status/algorithm mismatch')
        require(marker['selection'] == launch['config']['selection'] == co['selection']
            and marker['producer_matrix'] == launch['config']['producer_matrix'] == co['matrix'], 'Lane matrix/selection mismatch')
        for field, expected in (('capsule_id', co['capsule_id']), ('checkpoint_sha256', row['expected_checkpoint_sha256']), ('model_seed', row['seeds']['model'])):
            require(marker[field] == launch[field] == expected, 'Lane source binding mismatch: '+field)
        require(marker['query_rows_used'] == marker['source_rows_used'] == launch['query_rows_used'] == launch['source_rows_used'] == 0
            and marker['truth_read'] is launch['truth_read'] is launch['query_iq_access'] is False, 'Forbidden lane access')
        for field in COUNTERS:
            require(type(marker[field]) is int and marker[field] == states[row['row_id']][field] and marker[field] >= 0, 'Invalid lane counter: '+field)
            if field in EXACT: require(marker[field] == EXACT[field]//4, 'Fixed lane coverage mismatch')
            if field in MAXIMUM: require(marker[field] <= MAXIMUM[field]//4, 'Lane declared budget exceeded')
        metadata.append(dict(row_id=row['row_id'], marker=marker, config=launch['config'], hardware=launch.get('hardware'),
            python=launch.get('python'), blas_environment=launch.get('blas_environment'), payload_audit=launch.get('payload_audit')))
    lanes = []; stages = []; dedup_stages = []; groups = []; contexts = []; resource_groups = []
    for item in metadata:
        parsed = collect_lane(root/item['row_id']/'probe', item['row_id'], algorithm)
        for field in ('baseline_head_fit_count', 'baseline_factorization_count', 'transport_preparation_count', 'transport_stage_count',
            'trained_transport_stage_count', 'head_fit_count', 'factorization_count', 'sequence_paths')+ADDITIVE+PREP_COSTS+('final_score_evaluation_count', 'final_score_physical_count'):
            require(parsed['coverage'].get(field, 0) == item['marker'][field], 'Measured lane count mismatch: '+field)
        stages.extend(parsed.pop('stages')); dedup_stages.extend(parsed.pop('deduplicated_B_stages'))
        groups.extend(parsed.pop('B_physical_training_groups')); contexts.extend(parsed.pop('B_reuse_contexts')); lanes.append(parsed)
        resource_groups.extend(parsed['resource_groups'])
    for field in COUNTERS:
        require(sum(item['marker'][field] for item in metadata) == complete[field], 'Run/row actual total mismatch: '+field)
    run_scan = scan_text(run_log or root/'run.log'); markers = Counter(run_scan['marker_line_counts'])
    for lane in lanes:
        for scan in lane['text_logs']: markers.update(scan['marker_line_counts'])
    observations = Counter()
    for lane in lanes: observations.update(lane['structured_observations'])
    failure = any(observations[k] for k in ('technical_failure_statuses', 'explicit_error_fields', 'numeric_nonfinite_values', 'encoded_nonfinite_values')) or any(l['failure_artifact_present'] for l in lanes)
    return dict(status='COMPLETE_SCAN_FAILURE_EVIDENCE_FOUND' if failure else 'COMPLETE_TRANSPORT_TRAINING_LOG_SCAN_VERIFIED', run_id=spec['run_id'], release_commit=complete['commit'],
        coverage={field: complete[field] for field in COUNTERS}, algorithm=algorithm, source_metadata=metadata,
        lanes=lanes, stage_summaries=stages, B_reuse_contexts=contexts, B_physical_training_groups=groups, resource_groups=resource_groups,
        deduplicated_B_statistics=aggregate(dedup_stages), release_run_log=run_scan, text_marker_line_counts=dict(markers),
        structured_observations=dict(observations),
        totals=dict(actual_stages=len(stages), information_stages=sum(s['information_stage'] for s in stages),
            updated_stages=sum(s['updated_stage'] for s in stages), zero_update_information_stages=sum(s['information_stage'] and not s['updated_stage'] for s in stages),
            unique_B_physical_training_groups=len(groups), repeated_B_actual_contexts=sum(g['repeated_context_count'] for g in groups),
            B_C_inheritance_reuse_contexts=len(contexts)), **aggregate(stages), limitations=[
            'All listed complete structured records and every text line are scanned. Parent held-result payloads are not deserialized; fit_trace/parent compact/query/registry/history files are never opened.',
            'Information stage means physical K>=2 and multiple classes; updated stage means at least one accepted update. Zero-update information stages still pay initial forward and cached backward costs.',
            'Costs count each actual fit once. One B fit shared by seq/reset is not counted again for either inheritance context. Repeated B physical bindings across new-count parents are reported separately and deduplicated only in labeled B statistics, not actual resource totals.',
            'B dedup binding uses row, scope, parent K, train K, class IDs and exact ordered training physical IDs. Context counts and parameter variants are retained; these fits are not independent evidence.',
            'Initial, every cached gradient, every accepted/rejected Armijo trial and final accepted cache are retained. Inner accuracy/class risks supervise training and are not independent validation.',
            'SOURCE validation, individual prediction transitions, deployment bytes and missing standalone timings remain N/A. Correct-count equality does not establish identical predictions.',
            'Regex markers overlap and may be declarations; only structured errors establish structured failure evidence. Solver bounded stopping is expected, not a performance gate.',
            'Collector never selects parameters, changes optimizer budgets, fits or scores; healthy runs remain untouched.'])


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('x', encoding='utf-8', newline='') as source:
        if not rows: return
        fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(source, fieldnames=fields); writer.writeheader()
        writer.writerows({key: csv_value(row.get(key)) for key in fields} for row in rows)


def write_outputs(output, result):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out/'training_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    stages = []; points = []
    for stage in result['stage_summaries']:
        row = dict(stage['identity'], **{k: v for k, v in stage.items() if k not in ('identity', 'curve', 'initial', 'final', 'final_minus_initial', 'counts', 'costs')})
        row.update(stage['counts']); row.update(stage['costs'])
        for name in ('initial', 'final', 'final_minus_initial'): row.update({name+'_'+k: v for k, v in stage[name].items()})
        stages.append(row); points.extend(dict(stage['identity'], **{k: v for k, v in p.items() if k != 'metrics'}, **p['metrics']) for p in stage['curve'])
    write_csv(out/'stages.csv', stages); write_csv(out/'stage_curves.csv', points)
    for name in ('group_counts', 'stratified_counts', 'B_reuse_contexts', 'B_physical_training_groups'): write_csv(out/(name+'.csv'), result[name])
    for name in ('snapshots', 'curves', 'training_resource_groups', 'resource_groups'):
        write_csv(out/(name+'.csv'), (dict({k: v for k, v in row.items() if k != 'metrics'}, metric=metric, **stats)
            for row in result[name] for metric, stats in row['metrics'].items()))
    (out/'summary.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'stage_summaries'}, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# PrototypeTransport完整训练日志诊断', '', f"状态：{result['status']}；run：{result['run_id']}。", '',
        '完整日志已核对。information stage与有接受更新的stage分别统计；初始前向、零梯度反向及拒绝trial均按实际费用记录。', '',
        'stages.csv与stage_curves.csv保存所有阶段和全部训练位置；snapshots.csv与curves.csv保存分层聚合。B复用与物理训练绑定去重表独立保存，资源总量保留每次实际拟合。', '',
        '| 计数 | 实际值 |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in dict(result['coverage'], **result['totals']).items()]
    lines += ['', *['- '+item for item in result['limitations']], '']; (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-root', required=True)
    parser.add_argument('--run-log'); parser.add_argument('--output'); parser.add_argument('--ssh-host'); parser.add_argument('--ssh-config')
    parser.add_argument('--remote-python', default='python3'); parser.add_argument('--collect-stdout', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); require(args.collect_stdout or args.output, '--output required for local delivery')
    if args.output: require(not Path(args.output).exists(), 'Output already exists')
    if args.ssh_host:
        require(not args.collect_stdout, 'Nested SSH collection forbidden')
        command = [args.remote_python, '-', '--run-root', args.run_root, '--collect-stdout']
        if args.run_log: command += ['--run-log', args.run_log]
        ssh = ['ssh']+(['-F', args.ssh_config] if args.ssh_config else [])
        completed = subprocess.run(ssh+['-T', '-o', 'BatchMode=yes', args.ssh_host, shlex.join(command)],
            input=Path(__file__).read_text(encoding='utf-8'), text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        require(completed.returncode == 0, 'SSH collector failed: '+completed.stderr[-4000:]); result = json.loads(completed.stdout)
    else: result = collect(args.run_root, args.run_log)
    if args.collect_stdout: json.dump(result, sys.stdout, allow_nan=False); sys.stdout.write('\n')
    else: write_outputs(args.output, result); print(json.dumps(dict(status=result['status'], coverage=result['coverage'], output=str(Path(args.output).resolve()))))


if __name__ == '__main__': main()
