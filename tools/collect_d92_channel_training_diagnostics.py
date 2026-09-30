"""Fully reconcile completed joint training logs without fitting or scoring.

Standard library only. SSH stdin mode reads remote files and returns JSON; only
the local writer creates a new output directory. Outer-held results are ignored.
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

STATUS = 'JOINT_CHANNEL_PROBE_COMPLETE'
COORDS = ('split_id', 'scope', 'fold', 'trial', 'state', 'train_k')
STATES = dict(B_channel='B', C_channel='C_seq', C_reset='C_reset')
BRANCHES = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')
BLOCK_WIDTHS = (160, 96, 160, 160, 160)
PARAMETER_BOUND = math.log(2)/2
EXACT = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
    sequence_paths=1760, baseline_head_fit_count=3168, channel_preparation_count=3168,
    channel_stage_count=4576, diagnostic_fit_count=0)
MAXIMUM = dict(trained_channel_stage_count=936, optimizer_steps=7488,
    inner_objective_evaluation_count=8424, inner_head_fit_count=25272, inner_factorization_count=25272,
    final_head_fit_count=936, final_factorization_count=936, head_fit_count=29376, factorization_count=29376,
    baseline_factorization_count=3168, derivative_triangular_solve_count=44928)
ADDITIVE = ('optimizer_steps', 'inner_objective_evaluation_count', 'inner_head_fit_count',
    'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count',
    'derivative_triangular_solve_count')
METRICS = ('loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy',
    'inner_training_correct_count', 'inner_training_physical_count', 'inner_training_margin_mean',
    'inner_training_margin_min', 'gradient_norm', 'clipped_gradient_norm', 'gradient_clip_scale',
    'update_norm', 'unprojected_update_norm', 'learning_rate', 'step_seconds', 'active_box_count',
    'recorded_post_u_rms', 'recorded_post_u_max_abs', 'residual_rms', 'base_logits_rms')
VECTOR_METRICS = ('norm', 'rms', 'minimum', 'maximum', 'sum', 'nonzero_coordinates')
COUNT_FIELDS = ('nonzero_projected_updates', 'zero_projected_updates', 'projection_cancelled_updates',
    'gradient_zero_steps', 'clipped_steps', 'box_active_steps', 'objective_increase_edges',
    'objective_decrease_edges', 'objective_equal_edges')

PATTERNS = {
    'error': r'(?i)\b(?:error|failed|failure|exception)\b',
    'warning': r'(?i)\b(?:warning|warn)\b', 'traceback': r'(?i)\btraceback\b',
    'oom_or_killed': r'(?i)\b(?:out of memory|oom|killed)\b',
    'nonfinite_text': r'(?i)(?<![A-Za-z_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z_])',
    'recovery_or_resume': r'(?i)\b(?:resume|resumed|resuming|recover|recovery|restart|retry)\b',
    'early_stop_mention': r'(?i)\bearly[ _-]stop(?:ping|ped)?\b',
    'early_stop_disabled_declaration': r'(?i)"early_stopping"\s*:\s*false',
    'configuration': r'"(?:config|algorithm|learning_rate)"\s*:|(?i:\bconfiguration\b)',
    'environment': r'"(?:python|hardware|blas_environment)"\s*:|(?i:\bCUDA_VISIBLE_DEVICES\b)',
    'launch_command': r'"argv"\s*:|(?i:\b(?:command|launching)\b)',
}
PATTERNS = {key: re.compile(value) for key, value in PATTERNS.items()}


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scalars(value):
    if not isinstance(value, dict): return value
    return {key: scalars(item) for key, item in value.items()
            if isinstance(item, dict) or item is None or isinstance(item, (str, int, float, bool))}


def compact_event(event):
    """Keep measured vector summaries; full parameters/gradients remain in JSONL."""
    result = scalars(event)
    for key, value in event.items():
        if isinstance(value, (list, tuple)) and len(value) == 736 and all(isinstance(v, (int, float)) for v in value):
            result[key+'_summary'] = dict(count=736, norm=math.hypot(*value), minimum=min(value), maximum=max(value))
            start = 0; blocks = {}
            for name, width in zip(BRANCHES, (160, 96, 160, 160, 160)):
                block = value[start:start+width]; start += width
                blocks[name] = dict(norm=math.hypot(*block), sum=math.fsum(block), minimum=min(block), maximum=max(block))
            result[key+'_summary']['blocks'] = blocks
    if isinstance(event.get('projection_zero_sum_residuals'), (list, tuple)):
        result['projection_zero_sum_residuals_by_block'] = dict(zip(BRANCHES, event['projection_zero_sum_residuals']))
    for prefix, objective in (('', event), ('final_', event.get('final_objective'))):
        if isinstance(objective, dict) and isinstance(objective.get('inner_folds'), list):
            folds = objective['inner_folds']
            if folds and all('held_training_correct_count' in fold and 'held_physical_count' in fold for fold in folds):
                n = sum(fold['held_physical_count'] for fold in folds)
                result[prefix+'inner_training_accuracy'] = sum(fold['held_training_correct_count'] for fold in folds)/n if n else None
                result[prefix+'inner_training_physical_count'] = n
                margins = [fold.get('held_training_margin_mean') for fold in folds]
                minima = [fold.get('held_training_margin_min') for fold in folds]
                result[prefix+'inner_training_margin_mean'] = sum(v*fold['held_physical_count'] for v, fold in zip(margins, folds))/n if n and all(v is not None for v in margins) else None
                result[prefix+'inner_training_margin_min'] = min(minima) if minima and all(v is not None for v in minima) else None
    return result



def parameter_blocks(values):
    offset = 0
    for width in BLOCK_WIDTHS:
        yield values[offset:offset+width]
        offset += width


def valid_u(values):
    if not isinstance(values, list) or len(values) != 736: return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values): return False
    for block in parameter_blocks(values):
        tolerance = 128*sys.float_info.epsilon*len(block)*PARAMETER_BOUND
        if max(abs(v) for v in block) > PARAMETER_BOUND+tolerance or abs(math.fsum(block)) > tolerance: return False
    return True


def verify_box_projection(proposal, actual):
    """Verify the projection KKT conditions independently of core bisection."""
    require(valid_u(actual), 'Channel parameter zero-sum/box constraint violation')
    for proposed, block in zip(parameter_blocks(proposal), parameter_blocks(actual)):
        tolerance = 128*sys.float_info.epsilon*len(block)*max(PARAMETER_BOUND, max(abs(v) for v in proposed))
        free = [v-u for v, u in zip(proposed, block) if -PARAMETER_BOUND < u < PARAMETER_BOUND]
        lower = max((v-u for v, u in zip(proposed, block) if u <= -PARAMETER_BOUND), default=-math.inf)
        upper = min((v-u for v, u in zip(proposed, block) if u >= PARAMETER_BOUND), default=math.inf)
        if free:
            multiplier = math.fsum(free)/len(free)
            require(max(abs(v-multiplier) for v in free) <= tolerance
                and lower <= multiplier+tolerance and upper >= multiplier-tolerance, 'Projection KKT mismatch')
        else: require(lower <= upper+tolerance, 'Projection active-set KKT mismatch')



def csv_value(value):
    if value is None: return 'N/A'
    if isinstance(value, (dict, list)): return json.dumps(value, sort_keys=True, allow_nan=False)
    return str(value)


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            require(line.strip(), f'Blank record: {path}:{number}')
            yield json.loads(line)


def csvlines(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def check_csv(row, source, path):
    require(None not in row and set(source) <= set(row), 'Malformed/omitted CSV columns: '+str(path))
    require(all(row[key] == csv_value(source.get(key)) for key in row), 'CSV/JSON mismatch: '+str(path))


def finite_count(value):
    if isinstance(value, dict): return sum(finite_count(v) for v in value.values())
    if isinstance(value, list): return sum(finite_count(v) for v in value)
    return int(isinstance(value, float) and not math.isfinite(value))


def stage_key(row_id, row):
    scope, fold, trial = row.get('scope'), row.get('fold'), row.get('trial')
    require(scope in ('support_oof', 'support_oneshot_proxy'), 'Unexpected training scope')
    require((type(fold) is int and trial is None) if scope == 'support_oof' else
            (fold is None and type(trial) is int), 'Missing/ambiguous physical fold/anchor')
    require(type(row.get('train_k')) is int and row['train_k'] > 0, 'Missing physical train K')
    require(row.get('row_id', row_id) == row_id and row.get('split_id'), 'Stage row/parent mismatch')
    return (row_id,)+tuple(row.get(key) for key in COORDS)


def identity(key):
    return dict(zip(('row_id',)+COORDS, key))


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def vector_metrics(values, prefix, anchor=None):
    if values is None:
        names = [prefix+'_'+field for field in VECTOR_METRICS]
        names += [prefix+'_'+name+'_'+field for name in BRANCHES for field in VECTOR_METRICS]
        if prefix.startswith('u'):
            names += [prefix+'_anchor_distance']+[prefix+'_'+name+'_gain_condition_number' for name in BRANCHES]
        return dict.fromkeys(names)
    result = {}
    def put(vector, label):
        norm = math.hypot(*vector)
        result.update({label+'_norm': norm, label+'_rms': norm/math.sqrt(len(vector)),
            label+'_minimum': min(vector), label+'_maximum': max(vector), label+'_sum': math.fsum(vector),
            label+'_nonzero_coordinates': sum(v != 0 for v in vector)})
    put(values, prefix)
    for name, block in zip(BRANCHES, parameter_blocks(values)):
        put(block, prefix+'_'+name)
        if prefix.startswith('u'): result[prefix+'_'+name+'_gain_condition_number'] = math.exp(max(block)-min(block))
    if prefix.startswith('u'): result[prefix+'_anchor_distance'] = math.dist(values, anchor) if anchor is not None else None
    return result


def metrics(objective, u=None, anchor=None):
    objective = objective or {}; small = compact_event(objective)
    values = {key: number(small.get(key)) for key in METRICS}
    folds = objective.get('inner_folds')
    if isinstance(folds, list) and folds and all('held_training_correct_count' in f for f in folds):
        values['inner_training_correct_count'] = sum(f['held_training_correct_count'] for f in folds)
    values['recorded_post_u_rms'] = number(objective.get('u_rms'))
    values['recorded_post_u_max_abs'] = number(objective.get('u_max_abs'))
    values.update(vector_metrics(u, 'u', anchor))
    values.update(vector_metrics(objective.get('u_post'), 'u_post', anchor))
    values.update(vector_metrics(objective.get('gradient'), 'gradient'))
    return values


def close(actual, expected, message):
    require(number(actual) is not None and number(expected) is not None
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=2e-12), message)


def structured_observations(value):
    result = Counter()
    if isinstance(value, dict):
        if str(value.get('status', '')).upper() in ('FAILED', 'TECHNICAL_FAILURE', 'CHANNEL_STAGE_FAILED'):
            result['technical_failure_statuses'] += 1
        for key in ('error', 'error_type', 'failure_reason'):
            if value.get(key): result['explicit_error_fields'] += 1
        for child in value.values(): result.update(structured_observations(child))
    elif isinstance(value, list):
        for child in value: result.update(structured_observations(child))
    elif isinstance(value, float) and not math.isfinite(value): result['numeric_nonfinite_values'] += 1
    elif isinstance(value, str) and value in ('NaN', 'Infinity', '-Infinity'): result['encoded_nonfinite_values'] += 1
    return result

def add_stats(destination, values):
    for key, value in values.items():
        cell = destination.setdefault(key, dict(count=0, missing_count=0, sum=0., minimum=None, maximum=None))
        if value is None: cell['missing_count'] += 1
        else:
            cell['count'] += 1; cell['sum'] += value
            cell['minimum'] = value if cell['minimum'] is None else min(cell['minimum'], value)
            cell['maximum'] = value if cell['maximum'] is None else max(cell['maximum'], value)


def finish_stats(destination):
    return {key: dict(count=v['count'], missing_count=v['missing_count'], mean=v['sum']/v['count'] if v['count'] else None,
        minimum=v['minimum'], maximum=v['maximum']) for key, v in destination.items()}


def scan_text(path, expected_events=None, expected_stages=None):
    """Scan every text line, without deserializing parent held-result payloads."""
    path = Path(path); matches = Counter(); events = Counter(); lines = 0; nonfinite = 0
    expected = iter(expected_events) if expected_events is not None else None
    stages = iter(expected_stages) if expected_stages is not None else None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            lines += 1
            for name, pattern in PATTERNS.items(): matches[name] += int(bool(pattern.search(line)))
            if re.search(r'"event"\s*:\s*"EARLY_STOP(?:PED|PING)?"', line, re.I) or (
                not line.lstrip().startswith(('{', 'JOINT_CHANNEL_TRAINING '))
                and re.search(r'\bearly[ -]stopp(?:ed|ing)\b', line, re.I)):
                matches['early_stop_actual_message'] += 1
            if line.startswith('JOINT_CHANNEL_TRAINING '):
                row = json.loads(line.split(' ', 1)[1]); events[row['event']] += 1; nonfinite += finite_count(row)
                if expected is not None: require(row == next(expected, None), 'Text/event stream mismatch: '+str(path))
            elif line.startswith('{'):
                match = re.search(r'"event"\s*:\s*"([^"\\]+)"', line)
                event = match.group(1) if match else None
                if event: events[event] += 1
                if event in ('BASE_FIT', 'CHANNEL_PREPARATION', 'CANDIDATE_FIT'):
                    row = json.loads(line); nonfinite += finite_count(row)
                    if stages is not None: require(row == next(stages, None), 'Text/stage stream mismatch: '+str(path))
    if expected is not None: require(next(expected, None) is None, 'Text log missing training events: '+str(path))
    if stages is not None: require(next(stages, None) is None, 'Text log missing fit stages: '+str(path))
    return dict(path=str(path), bytes=path.stat().st_size, lines=lines,
        marker_line_counts={key: matches[key] for key in list(PATTERNS)+['early_stop_actual_message']}, events=dict(events), parsed_training_nonfinite_count=nonfinite)


def validate_objective(value, u, anchor, prepared):
    folds = value['inner_folds']; original = prepared['inner_folds']; n = prepared['train_physical_count']
    require(value['loss_scope'] == 'PHYSICAL_INNER_HELD_MARGIN_PLUS_PROXIMAL'
        and len(folds) == len(original) and value['inner_head_fit_count'] == len(folds), 'Objective scope/fold count mismatch')
    require(sum(f['held_physical_count'] for f in folds) == n, 'Inner objective physical pooling mismatch')
    for fold, before in zip(folds, original):
        require(fold['training_physical_ids'] == before['training_physical_ids']
            and fold['held_physical_ids'] == before['held_physical_ids']
            and fold['held_physical_count'] == before['held_physical_count']
            and 0 <= fold['held_training_correct_count'] <= fold['held_physical_count'], 'Objective inner physical binding mismatch')
    close(value['loss_data'], math.fsum(f['held_margin_loss_sum'] for f in folds)/n, 'Physical margin loss mismatch')
    close(value['loss_proximal'], math.fsum((x-y)**2 for x, y in zip(u, anchor))/(2*n), 'Physical proximal scaling mismatch')
    close(value['loss_total'], value['loss_data']+value['loss_proximal'], 'Total objective mismatch')
    for field in ('inner_factorization_count', 'derivative_triangular_solve_count'):
        innerfield = 'factorization_count' if field == 'inner_factorization_count' else field
        require(value[field] == sum(f[innerfield] for f in folds), 'Objective actual cost mismatch: '+field)


def validate_steps(full, steps, algorithm):
    n = full['optimizer_steps']; prepared = full['preparation']; anchor = full['anchor']; u = anchor
    require(n == len(steps) == len(full['steps']) == (0 if full['no_information'] else 8), 'Incomplete final-stage step coverage')
    require(algorithm['optimizer_steps'] == 8 and full['inner_objective_evaluation_count'] == (9 if n else 0),
        'Fixed update/objective budget mismatch')
    require(valid_u(anchor) and full['u_anchor'] == anchor and valid_u(full['u']), 'Invalid parameter/anchor')
    require(full['optimizer_state_reset'] is True and full['optimizer_state_bytes'] == 11776, 'Adam state inheritance mismatch')
    first = [0.]*736; second = [0.]*736; counts = Counter(dict.fromkeys(COUNT_FIELDS, 0)); evaluations = []
    beta1, beta2 = algorithm['adam_beta1'], algorithm['adam_beta2']
    for index, (event, embedded) in enumerate(zip(steps, full['steps']), 1):
        require(all(event.get(k) == v for k, v in embedded.items()), 'Embedded/event step mismatch')
        require(event['step'] == index and event['u_pre'] == u and event['anchor'] == anchor
            and event['learning_rate'] == algorithm['learning_rate'], 'Step parameter/anchor/config continuity mismatch')
        validate_objective(event, u, anchor, prepared)
        g = event['gradient']; require(len(g) == 736 and not finite_count(g), 'Invalid gradient vector')
        norm = math.hypot(*g); scale = min(1., algorithm['gradient_clip_norm']/norm) if norm else 1.
        close(event['gradient_norm'], norm, 'Gradient norm mismatch')
        close(event['gradient_clip_scale'], scale, 'Gradient clipping mismatch')
        used = [v*scale for v in g]; close(event['clipped_gradient_norm'], math.hypot(*used), 'Clipped gradient norm mismatch')
        first = [beta1*m+(1-beta1)*v for m, v in zip(first, used)]
        second = [beta2*m+(1-beta2)*v*v for m, v in zip(second, used)]
        proposal = [x-algorithm['learning_rate']*(m/(1-beta1**index))/(math.sqrt(v/(1-beta2**index))+algorithm['adam_epsilon'])
                    for x, m, v in zip(u, first, second)]
        post = event['u_post']; verify_box_projection(proposal, post)
        close(event['unprojected_update_norm'], math.dist(proposal, u), 'Adam unprojected norm mismatch')
        close(event['update_norm'], math.dist(post, u), 'Actual projected norm mismatch')
        require(len(event['projection_zero_sum_residuals']) == 5, 'Missing projection residual')
        for actual, block in zip(event['projection_zero_sum_residuals'], parameter_blocks(post)):
            close(actual, math.fsum(block), 'Projection residual mismatch')
        require(event['active_box_count'] == sum(abs(v) >= PARAMETER_BOUND-8*sys.float_info.epsilon for v in post), 'Active box count mismatch')
        changed = post != u; zero = all(v == 0 for v in g)
        require((event['update_norm'] > 0) == changed and (event['gradient_norm'] == 0) == zero, 'Exact zero/update state mismatch')
        counts['nonzero_projected_updates'] += int(changed); counts['zero_projected_updates'] += int(not changed)
        counts['projection_cancelled_updates'] += int(not changed and event['unprojected_update_norm'] > 0)
        counts['gradient_zero_steps'] += int(zero); counts['clipped_steps'] += int(scale < 1.)
        counts['box_active_steps'] += int(event['active_box_count'] > 0)
        u = post; evaluations.append(event)
    require(full['u'] == u and full['u_changed_from_anchor'] == (u != anchor)
        and full['nonzero_projected_update_count'] == counts['nonzero_projected_updates'], 'Final parameter/change counter mismatch')
    close(full['u_update_norm'], math.dist(u, anchor), 'Final anchor distance mismatch')
    if n:
        final = full['final_objective']; require(final is not None and final['gradient'] is None
            and final['derivative_triangular_solve_count'] == 0, 'Final objective incorrectly missing or differentiated')
        validate_objective(final, u, anchor, prepared); evaluations.append(final)
    else: require(full['final_objective'] is None and u == anchor, 'Skipped stage invented objective/update')
    for field in ('inner_head_fit_count', 'inner_factorization_count', 'derivative_triangular_solve_count'):
        require(full[field] == sum(v[field] for v in evaluations), 'Actual stage cost mismatch: '+field)
    for before, after in zip(evaluations, evaluations[1:]):
        delta = after['loss_total']-before['loss_total']
        counts['objective_increase_edges'] += int(delta > 0)
        counts['objective_decrease_edges'] += int(delta < 0)
        counts['objective_equal_edges'] += int(delta == 0)
    return dict(counts)

def collect_lane(lane, row_id, algorithm):
    lane = Path(lane)
    paths = {name: lane/name for name in ('fit_stages.jsonl', 'fit_stages.csv', 'training_events.jsonl',
        'training_events_compact.jsonl', 'training_events_compact.csv')}
    candidates = {}; preparations = {}; counts = Counter(); resources = {}; observed = Counter(); seen_base = set()
    for row, csvrow in itertools.zip_longest(jsonlines(paths['fit_stages.jsonl']), csvlines(paths['fit_stages.csv'])):
        require(row is not None and csvrow is not None, 'Fit-stage JSON/CSV length mismatch')
        check_csv(csvrow, row, paths['fit_stages.csv']); observed.update(structured_observations(row)); counts['fit_stage_records'] += 1
        key = stage_key(row_id, row); event = row['event']
        if event == 'BASE_FIT':
            require(key not in seen_base, 'Duplicate baseline stage'); seen_base.add(key)
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += row['factorization_calls']
        elif event == 'CHANNEL_PREPARATION':
            require(key not in preparations, 'Duplicate preparation'); preparations[key] = row
            counts['channel_preparation_count'] += 1
        else:
            require(event == 'CANDIDATE_FIT' and row['state'] in STATES, 'Unexpected fit stage')
            require(key not in candidates, 'Duplicate candidate stage'); candidates[key] = row
        group = (row['scope'], row['state'], row['train_k'])
        add_stats(resources.setdefault(group, {}), {field: row.get(field) for field in
            ('fit_seconds', 'score_seconds', 'fit_and_score_seconds', 'prepare_seconds', 'baseline_binding_seconds',
             'baseline_binding_distance_evaluation_count', 'baseline_binding_factorization_count',
             'persistent_state_bytes', 'head_state_bytes', 'adapter_state_bytes', 'lineage_state_bytes',
             'transient_distance_bytes', 'prepared_numeric_state_bytes', 'optimizer_state_bytes')})
    stages = {}; pending = {}; inner_prepared = {}; event_counts = Counter(); b_vectors = {}
    for full, compact, csvrow in itertools.zip_longest(jsonlines(paths['training_events.jsonl']),
            jsonlines(paths['training_events_compact.jsonl']), csvlines(paths['training_events_compact.csv'])):
        require(full is not None and compact is not None and csvrow is not None, 'Training event stream length mismatch')
        require(compact_event(full) == compact, 'Full/compact event mismatch')
        check_csv(csvrow, compact, paths['training_events_compact.csv']); observed.update(structured_observations(full))
        require(full.get('source_validation') is None and full.get('objective_scope') == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION',
            'Training-only objective contract mismatch')
        event = full['event']; event_counts[event] += 1; key = stage_key(row_id, full)
        if event == 'CHANNEL_INNER_PREPARED':
            prepkey = stage_key(row_id, dict(full, state=full['state'].removesuffix('_prepare')))
            require(prepkey in preparations, 'Inner preparation references absent preparation')
            folded = inner_prepared.setdefault(prepkey, {})
            require(full['inner_fold'] not in folded, 'Duplicate inner preparation')
            train, held = set(full['training_physical_ids']), set(full['held_physical_ids'])
            require(not train & held and full['train_physical_count'] == len(train)
                and full['held_physical_count'] == len(held), 'Inner physical support partition mismatch')
            folded[full['inner_fold']] = full; continue
        require(key in candidates and full['mode'] == STATES[full['state']], 'Event references absent/mismatched candidate')
        require(key not in stages, 'Event after final candidate')
        if event == 'JOINT_CHANNEL_STEP':
            steps = pending.setdefault(key, [])
            require(full['step'] == len(steps)+1 <= 8, 'Missing/reordered/repeated update')
            steps.append(full); continue
        require(event == 'JOINT_CHANNEL_FIT' and full['status'] == 'CHANNEL_STAGE_COMPLETE', 'Unexpected/incomplete training event')
        require(full['config'] == algorithm and candidates[key]['config'] == scalars(algorithm), 'Frozen stage config mismatch')
        for field, value in compact.items():
            if field != 'event': require(candidates[key].get(field) == value, 'Final event/fit-stage mismatch: '+field)
        steps = pending.pop(key, []); update_counts = validate_steps(full, steps, algorithm)
        prepkey = stage_key(row_id, dict(full, state=full['state'][0]))
        require(prepkey in preparations, 'Candidate references absent preparation')
        prep = full['preparation']
        for field, value in scalars(prep).items():
            require(preparations[prepkey].get(field) == value, 'Preparation lineage mismatch: '+field)
        require(full['no_information'] == prep['no_information']
            and (not steps) == (prep['train_k'] == 1 or len(prep['classes']) == 1), 'No-information physical rule mismatch')
        prepared_events = inner_prepared.get(prepkey, {}); seen_held = set(); physical = set(prep['training_physical_ids'])
        require(len(prep['inner_folds']) == len(prepared_events), 'Inner preparation event coverage mismatch')
        for folded in prep['inner_folds']:
            other = prepared_events[folded['inner_fold']]
            require(all(other.get(field) == value for field, value in folded.items()), 'Inner preparation event/audit mismatch')
            train, held = set(folded['training_physical_ids']), set(folded['held_physical_ids'])
            require(not train & held and train | held == physical and not seen_held & held, 'Inner training sample pool mismatch')
            seen_held.update(held)
        require(not prep['inner_folds'] or seen_held == physical, 'Missing pooled inner physical ID')
        anchor, u = full['anchor'], full['u']
        bkey = stage_key(row_id, dict(full, state='B_channel'))
        if full['mode'] == 'C_seq':
            require(bkey in b_vectors and anchor == b_vectors[bkey], 'Sequential B/C parameter anchor mismatch')
        else: require(anchor == [0.]*736, 'B/reset anchor must be zero')
        if full['mode'] == 'B': b_vectors[bkey] = u
        first = metrics(steps[0], steps[0]['u_pre'], anchor) if steps else metrics(None)
        last = metrics(steps[-1], steps[-1]['u_pre'], anchor) if steps else metrics(None)
        final = metrics(full['final_objective'], u, anchor)
        delta_fields = ('loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy',
            'inner_training_correct_count', 'inner_training_margin_mean', 'inner_training_margin_min')
        deltas = {field: final[field]-first[field] if final[field] is not None and first[field] is not None else None for field in delta_fields}
        curve = [dict(step=v['step'], timing='pre_update_objective_with_separate_post_u_fields',
            metrics=metrics(v, v['u_pre'], anchor)) for v in steps]
        curve.append(dict(step=None, timing='final_after_updates' if steps else 'final_without_objective', metrics=final))
        trained = bool(steps)
        stages[key] = dict(identity=identity(key), mode=full['mode'], parent_k=full.get('parent_k'),
            class_count=len(full['classes']), old_class_count=len(full['old_classes']), steps=len(steps), no_information=full['no_information'],
            trained=trained, no_update_reason=full.get('no_update_reason'), u_changed_from_anchor=u != anchor,
            identity_forward=full['identity_forward'], all_projected_zero=trained and update_counts['nonzero_projected_updates'] == 0,
            all_gradient_zero=trained and update_counts['gradient_zero_steps'] == len(steps),
            inner_correct_count_unchanged=None if not trained else first['inner_training_correct_count'] == final['inner_training_correct_count'],
            inner_prediction_change_count=None, inner_prediction_change_reason='Only aggregate correct counts/margins logged; individual prediction transitions are not measured.',
            counts=update_counts, first=first, last=last, final=final, final_minus_initial=deltas, curve=curve,
            costs={field: full.get(field) for field in ADDITIVE+('fit_seconds', 'persistent_state_bytes', 'adapter_state_bytes',
                'head_state_bytes', 'lineage_state_bytes', 'trainable_parameter_count', 'effective_parameter_count',
                'optimizer_state_bytes', 'baseline_binding_seconds', 'baseline_binding_distance_evaluation_count', 'baseline_binding_factorization_count')},
            score_seconds=candidates[key].get('score_seconds'),
            full_vectors_source=str(paths['training_events.jsonl']))
        counts['channel_stage_count'] += 1; counts['trained_channel_stage_count'] += int(trained)
        for field in ADDITIVE: counts[field] += full[field]
    require(not pending and set(stages) == set(candidates), 'Missing candidate final events')
    new_by_path = {}
    for key, stage in stages.items():
        pathkey = key[:5]+key[6:]
        new_by_path[pathkey] = max(new_by_path.get(pathkey, 0), stage['class_count']-stage['old_class_count'])
    for key, stage in stages.items(): stage['parent_new_count'] = new_by_path[key[:5]+key[6:]]
    counts['head_fit_count'] = counts['baseline_head_fit_count']+counts['inner_head_fit_count']+counts['final_head_fit_count']
    counts['factorization_count'] = counts['baseline_factorization_count']+counts['inner_factorization_count']+counts['final_factorization_count']
    counts['structured_nonfinite_count'] = observed['numeric_nonfinite_values']+observed['encoded_nonfinite_values']
    texts = [scan_text(path, jsonlines(paths['training_events_compact.jsonl']), jsonlines(paths['fit_stages.jsonl']))
             for path in (lane/'training.log', lane.parent/'probe.log')]
    resource_rows = [dict(row_id=row_id, scope=key[0], state=key[1], train_k=key[2], metrics=finish_stats(value))
                     for key, value in sorted(resources.items())]
    return dict(row_id=row_id, stages=list(stages.values()), resource_groups=resource_rows, coverage=dict(counts),
        structured_observations={key: observed[key] for key in ('technical_failure_statuses', 'explicit_error_fields',
            'numeric_nonfinite_values', 'encoded_nonfinite_values')}, event_counts=dict(event_counts),
        inventory=[dict(path=str(p), bytes=p.stat().st_size) for p in paths.values()],
        text_logs=texts, failure_artifact_present=(lane/'probe_failed.json').exists())

def aggregate(stages):
    groups = {}; curves = {}; snapshots = {}; strata = {}
    for stage in stages:
        i = stage['identity']; group = (i['scope'], i['state'], i['train_k'])
        count = groups.setdefault(group, Counter()); count['stages'] += 1; count['steps'] += stage['steps']
        for key in ('trained', 'no_information', 'u_changed_from_anchor', 'identity_forward',
                    'all_projected_zero', 'all_gradient_zero'): count[key+'_stages'] += int(stage[key])
        count['skipped_channel_stages'] += int(not stage['trained'])
        count.update(stage['counts'])
        strat_key = (i['row_id'], i['scope'], i['state'], stage['parent_k'], stage['parent_new_count'], i['train_k'])
        cell = strata.setdefault(strat_key, Counter())
        cell['stages'] += 1; cell['optimizer_steps'] += stage['steps']; cell['trained_stages'] += int(stage['trained'])
        cell['changed_from_anchor_stages'] += int(stage['u_changed_from_anchor']); cell.update(stage['counts'])
        for timing in ('first', 'last', 'final', 'final_minus_initial'):
            add_stats(snapshots.setdefault(group+(timing,), {}), stage[timing])
        for point in stage['curve']:
            add_stats(curves.setdefault(group+(point['timing'], point['step']), {}), point['metrics'])
    coords = lambda key: dict(scope=key[0], state=key[1], train_k=key[2])
    return dict(group_counts=[dict(coords(k), **v) for k, v in sorted(groups.items())],
        stratified_counts=[dict(row_id=k[0], scope=k[1], state=k[2], parent_k=k[3], parent_new_count=k[4], train_k=k[5], **v)
                           for k, v in sorted(strata.items())],
        snapshots=[dict(coords(k), timing=k[3], metrics=finish_stats(v)) for k, v in sorted(snapshots.items())],
        curves=[dict(coords(k), timing=k[3], step=k[4], metrics=finish_stats(v)) for k, v in sorted(curves.items())])


def collect(run_root, run_log=None):
    root = Path(run_root)
    startup, complete, states = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    spec = startup['spec']; algorithm = spec['probe']['algorithm']; rows = spec['rows']
    require(complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == len(rows) == 4
        and complete['commit'] == startup['commit'] and all(complete[k] == v for k, v in EXACT.items()),
        'Complete fixed four-row pilot required')
    require(spec['probe']['exact_counts'] == EXACT and spec['probe']['maximum_counts'] == MAXIMUM, 'Pilot budget/config mismatch')
    require(set(states) == {row['row_id'] for row in rows}, 'Row state coverage mismatch')
    require(startup['query_access'] is complete['query_access'] is False
        and startup['source_sample_access'] is complete['source_sample_access'] is False, 'Forbidden run access')
    metadata = []
    for row in rows:
        lane = root/row['row_id']/'probe'; cohort = spec['probe']['cohorts'][row['cohort']]
        marker, launch = read(lane/'probe_complete.json'), read(lane/'startup.json')
        require(marker['status'] == states[row['row_id']]['status'] == STATUS, 'Incomplete lane')
        require(marker['algorithm'] == launch['config']['algorithm'] == algorithm
            and marker['selection'] == launch['config']['selection'] == cohort['selection']
            and marker['producer_matrix'] == launch['config']['producer_matrix'] == cohort['matrix'], 'Lane config mismatch')
        for field, expected in (('capsule_id', cohort['capsule_id']), ('checkpoint_sha256', row['expected_checkpoint_sha256']),
                                ('model_seed', row['seeds']['model'])):
            require(marker[field] == launch[field] == expected, 'Lane source binding mismatch: '+field)
        for key, total in dict(EXACT, **MAXIMUM).items():
            value = marker[key]
            require(type(value) is int and value == states[row['row_id']][key] and value >= 0, 'Invalid lane counter: '+key)
            require(value == total//4 if key in EXACT else value <= total//4, 'Lane coverage/budget mismatch: '+key)
        require(marker['query_rows_used'] == marker['source_rows_used'] == launch['query_rows_used'] == launch['source_rows_used'] == 0
            and marker['truth_read'] is launch['truth_read'] is launch['query_iq_access'] is False, 'Forbidden lane access')
        metadata.append(dict(row_id=row['row_id'], marker=marker, hardware=launch.get('hardware'),
            blas_environment=launch.get('blas_environment'), python=launch.get('python'), config=launch['config'],
            payload_audit=launch.get('payload_audit'), objective_scope=launch['objective_scope']))
    lanes = []; stages = []; resource_rows = []
    for item in metadata:
        lane = collect_lane(root/item['row_id']/'probe', item['row_id'], algorithm)
        for key in ('baseline_head_fit_count', 'channel_preparation_count', 'channel_stage_count')+tuple(MAXIMUM):
            require(lane['coverage'].get(key, 0) == item['marker'][key], 'Measured lane count mismatch: '+key)
        stages.extend(lane.pop('stages'))
        resource_rows.extend(lane.pop('resource_groups'))
        lanes.append(lane)
    for key in dict(EXACT, **MAXIMUM):
        require(sum(item['marker'][key] for item in metadata) == complete[key], 'Run/row total mismatch: '+key)
    require(complete['optimizer_steps'] == 8*complete['trained_channel_stage_count']
        and complete['inner_objective_evaluation_count'] == 9*complete['trained_channel_stage_count'], 'Run update/objective count mismatch')
    nonfinite = sum(lane['coverage']['structured_nonfinite_count'] for lane in lanes)
    run_scan = scan_text(Path(run_log) if run_log else root/'run.log')
    observations = Counter()
    for scan in [run_scan]+[text for lane in lanes for text in lane['text_logs']]: observations.update(scan['marker_line_counts'])
    technical = sum(lane['structured_observations']['technical_failure_statuses'] for lane in lanes)
    failure_artifacts = sum(lane['failure_artifact_present'] for lane in lanes)
    return dict(status='COMPLETE_CHANNEL_TRAINING_LOG_SCAN_VERIFIED' if not (nonfinite or technical or failure_artifacts) else 'COMPLETE_SCAN_FAILURE_EVIDENCE_FOUND',
        run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=dict({key: complete[key] for key in dict(EXACT, **MAXIMUM)}, candidate_final_events=len(stages),
            structured_nonfinite_count=nonfinite, structured_technical_failure_statuses=technical, failure_artifacts=failure_artifacts),
        algorithm=algorithm, source_metadata=metadata, lanes=lanes, release_run_log=run_scan,
        text_marker_line_counts=dict(observations), stage_summaries=stages, resource_groups=resource_rows,
        totals=dict(candidate_stages=len(stages), trained_channel_stages=sum(v['trained'] for v in stages),
            skipped_channel_stages=sum(not v['trained'] for v in stages),
            all_projected_zero_trained_stages=sum(v['all_projected_zero'] for v in stages),
            all_gradient_zero_trained_stages=sum(v['all_gradient_zero'] for v in stages),
            changed_from_anchor_stages=sum(v['u_changed_from_anchor'] for v in stages),
            identity_forward_stages=sum(v['identity_forward'] for v in stages),
            trained_inner_correct_count_unchanged_stages=sum(v['inner_correct_count_unchanged'] is True for v in stages),
            **{key: sum(v['counts'].get(key, 0) for v in stages) for key in COUNT_FIELDS}),
        **aggregate(stages), limitations=[
            'All records and text lines scanned; full/compact/CSV/text streams reconciled. No query or outer-held scores read.',
            'First and last are pre-update objectives at steps 1 and 8. Final objective is after all 8 updates.',
            'Inner accuracy pools physical inner-held training counts; it is part of the training objective, not independent validation.',
            'Actual stages receive equal weight within scope/state/train K; updates are not independent experiments.',
            'Physical K1/single-class stages skip; ordinary zero-gradient stages still execute eight Adam updates. Unmeasured metrics remain null.',
            'Parameter/gradient arrays are verified coordinate by coordinate, then reduced to per-block statistics. Original full arrays remain in source logs; no spectral contraction bound applies to this method.',
            'The supervised objective pools held_margin_loss_sum; head_training_loss_data/ridge/total are separate closed-form head diagnostics.',
            'Equal aggregate correct counts do not prove identical individual predictions. Individual prediction transitions are not recorded here.',
            'early_stop_disabled_declaration counts configuration false; early_stop_actual_message counts explicit events/plain stopping messages separately.',
            'Text markers count matching lines, overlap, and include config declarations; they do not by themselves prove a failure or restart.',
            'Training/score wall time and RSS refer to the recorded CPU platform; GPU memory and unmeasured metrics remain null.',
            'Ground data payload, full fitted state RAM, and code release archive are distinct. No deployment bytes inferred from adapter vector size.',
            'Original Phase1 model delivery is outside incremental method payload; no deployment package was measured by this collector.'])


def write_csv(path, rows):
    rows = list(rows)
    with path.open('x', encoding='utf-8', newline='') as stream:
        if not rows: return
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: csv_value(row.get(key)) for key in fields} for row in rows)


def write_outputs(output, result):
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    with (destination/'training_diagnostics.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':')); stream.write('\n')
    rows = []; curves = []
    for stage in result['stage_summaries']:
        row = dict(stage['identity'], **{key: value for key, value in stage.items()
            if key not in ('identity', 'curve', 'first', 'last', 'final', 'final_minus_initial', 'counts', 'costs')})
        row.update(stage['counts']); row.update(stage['costs'])
        for timing in ('first', 'last', 'final', 'final_minus_initial'):
            row.update({timing+'_'+key: value for key, value in stage[timing].items()})
        rows.append(row)
        curves.extend(dict(stage['identity'], **{k: v for k, v in p.items() if k != 'metrics'}, **p['metrics']) for p in stage['curve'])
    write_csv(destination/'stages.csv', rows)
    write_csv(destination/'stage_curves.csv', curves)
    write_csv(destination/'group_counts.csv', result['group_counts'])
    write_csv(destination/'stratified_counts.csv', result['stratified_counts'])
    for name in ('snapshots', 'curves', 'resource_groups'):
        write_csv(destination/(name+'.csv'), (dict({k: v for k, v in row.items() if k != 'metrics'}, metric=metric, **stats)
            for row in result[name] for metric, stats in row['metrics'].items()))
    summary = {key: value for key, value in result.items() if key not in ('stage_summaries',)}
    with (destination/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')
    with (destination/'report.md').open('x', encoding='utf-8') as stream:
        stream.write('# Complete channel training log diagnostics\n\n')
        stream.write('Status: `'+result['status']+'`. Run: `'+result['run_id']+'`.\n\n')
        stream.write('All listed structured records and full text logs were read and reconciled. Inner accuracy is a training metric, not independent validation.\n\n')
        stream.write('## Actual counts\n\n| Counter | Actual |\n|---|---:|\n')
        for key, value in dict(result['coverage'], **result['totals']).items(): stream.write(f'| {key} | {value} |\n')
        stream.write('\n## Text marker observations\n\n| Marker | Matching lines |\n|---|---:|\n')
        for key, value in result['text_marker_line_counts'].items(): stream.write(f'| {key} | {value} |\n')
        stream.write('\nMarker counts overlap and can include configuration declarations; they do not automatically diagnose a failure.\n\n')
        stream.write('## Measured objectives and parameters\n\n')
        stream.write('`snapshots.csv` reports first/last/final and final-minus-initial statistics by scope/state/train K. '
            '`curves.csv` contains all eight update positions and final points; `stage_curves.csv` retains each physical stage. '
            '`stages.csv` retains compact parameter/anchor distances, step counts and projection/gradient statistics. '
            'Raw 736-vectors remain in the referenced original training streams. \n\n')
        stream.write('## Boundaries\n\n')
        for item in result['limitations']: stream.write('- '+item+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True); parser.add_argument('--run-log'); parser.add_argument('--output')
    parser.add_argument('--ssh-host'); parser.add_argument('--ssh-config'); parser.add_argument('--remote-python', default='python3')
    parser.add_argument('--collect-stdout', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); require(args.collect_stdout or args.output, '--output required for local delivery')
    if args.output: require(not Path(args.output).exists(), 'Output already exists')
    if args.ssh_host:
        require(not args.collect_stdout, 'Nested SSH collection forbidden')
        command = [args.remote_python, '-', '--run-root', args.run_root, '--collect-stdout']
        if args.run_log: command += ['--run-log', args.run_log]
        ssh = ['ssh']
        if args.ssh_config: ssh += ['-F', args.ssh_config]
        completed = subprocess.run(ssh+['-T', '-o', 'BatchMode=yes', args.ssh_host, shlex.join(command)],
            input=Path(__file__).read_text(encoding='utf-8'), text=True, encoding='utf-8',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        require(completed.returncode == 0, 'SSH collector failed: '+completed.stderr[-4000:])
        result = json.loads(completed.stdout)
    else: result = collect(args.run_root, args.run_log)
    if args.collect_stdout:
        json.dump(result, sys.stdout, allow_nan=False, separators=(',', ':')); sys.stdout.write('\n')
    else:
        write_outputs(args.output, result)
        print(json.dumps(dict(status=result['status'], coverage=result['coverage'], output=str(Path(args.output).resolve()))))


if __name__ == '__main__': main()
