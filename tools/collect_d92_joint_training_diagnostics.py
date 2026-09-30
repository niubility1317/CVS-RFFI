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

STATUS = 'JOINT_SPECTRAL_PROBE_COMPLETE'
COORDS = ('split_id', 'scope', 'fold', 'trial', 'state', 'train_k')
STATES = dict(B_joint='B', C_joint='C_seq', C_reset='C_reset', B_fixed='fixed', C_fixed='fixed')
EXACT = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
    sequence_paths=1760, baseline_head_fit_count=3168, joint_preparation_count=3168,
    joint_stage_count=4576, fixed_stage_count=3168, diagnostic_fit_count=0)
MAXIMUM = dict(trained_joint_stage_count=936, optimizer_steps=7488, geometry_fit_count=3704,
    geometry_factorization_count=2304, inner_objective_evaluation_count=8424,
    inner_head_fit_count=25272, inner_factorization_count=25272, final_head_fit_count=1584,
    final_factorization_count=1584, head_fit_count=30024, factorization_count=30024,
    baseline_factorization_count=3168)
ADDITIVE = ('optimizer_steps', 'inner_objective_evaluation_count', 'inner_head_fit_count',
    'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count',
    'derivative_triangular_solve_count')
METRICS = ('loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy',
    'inner_training_physical_count', 'gradient_0', 'gradient_1', 'gradient_norm',
    'theta_0', 'theta_1', 'theta_sum', 'theta_anchor_distance', 'theta_norm',
    'theta_pre_0', 'theta_pre_1', 'theta_post_0', 'theta_post_1',
    'update_norm', 'unprojected_update_norm', 'learning_rate', 'step_seconds',
    'residual_rms', 'base_logits_rms', 'operator_contraction_universal_bound',
    'inner_geometry_max_spectral_gain', 'full_final_geometry_max_spectral_gain')
PATTERNS = {
    'error': r'(?i)\b(?:error|failed|failure|exception)\b',
    'warning': r'(?i)\b(?:warning|warn)\b', 'traceback': r'(?i)\btraceback\b',
    'oom_or_killed': r'(?i)\b(?:out of memory|oom|killed)\b',
    'nonfinite_text': r'(?i)(?<![A-Za-z_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z_])',
    'recovery_or_resume': r'(?i)\b(?:resume|resumed|resuming|recover|recovery|restart|retry)\b',
    'early_stop': r'(?i)\bearly[ _-]stop(?:ping|ped)?\b',
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
    """Mirror the producer's projection, including the two numerical coordinates."""
    result = scalars(event)
    for key, value in event.items():
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            result[key+'_0'], result[key+'_1'] = value
    for prefix, objective in (('', event), ('final_', event.get('final_objective'))):
        if isinstance(objective, dict) and isinstance(objective.get('inner_folds'), list):
            folds = objective['inner_folds']
            if folds and all('held_training_correct_count' in fold and 'held_physical_count' in fold for fold in folds):
                n = sum(fold['held_physical_count'] for fold in folds)
                result[prefix+'inner_training_accuracy'] = sum(fold['held_training_correct_count'] for fold in folds)/n if n else None
                result[prefix+'inner_training_physical_count'] = n
    return result


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


def metrics(objective, theta=None, anchor=None, spectra=None):
    small = compact_event(objective or {})
    values = {key: small.get(key) for key in METRICS}
    if theta is not None:
        values.update(theta_0=theta[0], theta_1=theta[1], theta_sum=sum(theta), theta_norm=math.hypot(*theta),
            theta_anchor_distance=math.dist(theta, anchor) if anchor is not None else None,
            operator_contraction_universal_bound=sum(theta)/2)
        if spectra is not None:
            values['inner_geometry_max_spectral_gain'] = max(
                (v/(1+v)*(theta[0]+theta[1]*v) for values in spectra for v in values), default=0.)
    return {key: value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            else None for key, value in values.items()}


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
            if line.startswith('JOINT_SPECTRAL_TRAINING '):
                row = json.loads(line.split(' ', 1)[1]); events[row['event']] += 1; nonfinite += finite_count(row)
                if expected is not None: require(row == next(expected, None), 'Text/event stream mismatch: '+str(path))
            elif line.startswith('{'):
                match = re.search(r'"event"\s*:\s*"([^"\\]+)"', line)
                event = match.group(1) if match else None
                if event: events[event] += 1
                if event in ('BASE_FIT', 'JOINT_PREPARATION', 'CANDIDATE_FIT'):
                    row = json.loads(line); nonfinite += finite_count(row)
                    if stages is not None: require(row == next(stages, None), 'Text/stage stream mismatch: '+str(path))
    if expected is not None: require(next(expected, None) is None, 'Text log missing training events: '+str(path))
    if stages is not None: require(next(stages, None) is None, 'Text log missing fit stages: '+str(path))
    return dict(path=str(path), bytes=path.stat().st_size, lines=lines,
        marker_line_counts={key: matches[key] for key in PATTERNS}, events=dict(events), parsed_training_nonfinite_count=nonfinite)


def validate_steps(full, steps, algorithm):
    n = full['optimizer_steps']; fixed = full['mode'] == 'fixed'
    require(n == len(steps) == len(full['steps']) and n in (0, 8), 'Incomplete final-stage step coverage')
    require(n == (0 if fixed or full['no_information'] else algorithm['optimizer_steps']), 'No-information/update mismatch')
    require(full['inner_objective_evaluation_count'] == (9 if n else 0), 'Objective evaluation count mismatch')
    require((full['final_objective'] is not None) == bool(n), 'Missing/unexpected final objective')
    theta, anchor = full['theta'], full['theta_anchor']
    require(len(theta) == len(anchor) == 2 and full['anchor'] == anchor, 'Invalid theta/anchor')
    require(all(v >= 0 for v in theta) and sum(theta) <= 1+1e-14, 'Infeasible final theta')
    counts = Counter(); previous = anchor
    for number, (event, embedded) in enumerate(zip(steps, full['steps']), 1):
        require(all(event.get(key) == value for key, value in embedded.items()), 'Embedded/event step mismatch')
        require(event['step'] == number and event['theta_pre'] == previous and event['theta_anchor'] == anchor,
            'Step parameter/anchor continuity mismatch')
        require(event['learning_rate'] == algorithm['learning_rate'], 'Learning-rate mismatch')
        post, grad = event['theta_post'], event['gradient']
        require(len(post) == len(grad) == 2 and all(v >= 0 for v in post) and sum(post) <= 1+1e-14, 'Invalid step theta/gradient')
        changed = post != previous; zero = all(v == 0 for v in grad)
        require((event['update_norm'] > 0) == changed and (event['gradient_norm'] == 0) == zero,
            'Recorded update/gradient norm inconsistency')
        require(event['active_nonnegative_constraints'] == [v == 0 for v in post]
            and event['active_sum_constraint'] == (sum(post) == 1), 'Active constraint mismatch')
        counts['nonzero_projected_updates'] += int(changed)
        counts['zero_projected_updates'] += int(not changed)
        counts['projection_cancelled_updates'] += int(not changed and event['unprojected_update_norm'] > 0)
        counts['gradient_zero_steps'] += int(zero)
        counts['active_sum_steps'] += int(event['active_sum_constraint'])
        for i in range(2):
            counts['active_zero_coordinate_'+str(i)+'_steps'] += int(post[i] == 0)
            counts['positive_gradient_'+str(i)+'_steps'] += int(grad[i] > 0)
            counts['negative_gradient_'+str(i)+'_steps'] += int(grad[i] < 0)
        previous = post
    require(not n or theta == previous, 'Final theta differs from last update')
    require(full['nonzero_projected_update_count'] == counts['nonzero_projected_updates'], 'Nonzero-update counter mismatch')
    require(full['theta_changed_from_anchor'] == (theta != anchor), 'Theta change flag mismatch')
    if not n and not fixed: require(theta == anchor, 'Skipped joint stage changed theta')
    if fixed: require(theta == [1., 0.], 'Fixed-control theta mismatch')
    if n:
        for key in ('inner_head_fit_count', 'inner_factorization_count', 'derivative_triangular_solve_count'):
            require(full[key] == sum(s[key] for s in steps)+full['final_objective'][key], 'Objective cost mismatch: '+key)
    return dict(counts)


def collect_lane(lane, row_id, algorithm):
    lane = Path(lane)
    paths = {name: lane/name for name in ('fit_stages.jsonl', 'fit_stages.csv', 'training_events.jsonl',
        'training_events_compact.jsonl', 'training_events_compact.csv')}
    candidates = {}; preparations = {}; counts = Counter(); resources = {}; nonfinite = 0; seen_base = set()
    for row, csvrow in itertools.zip_longest(jsonlines(paths['fit_stages.jsonl']), csvlines(paths['fit_stages.csv'])):
        require(row is not None and csvrow is not None, 'Fit-stage JSON/CSV length mismatch')
        check_csv(csvrow, row, paths['fit_stages.csv']); nonfinite += finite_count(row); counts['fit_stage_records'] += 1
        key = stage_key(row_id, row); event = row['event']
        if event == 'BASE_FIT':
            require(key not in seen_base, 'Duplicate baseline stage'); seen_base.add(key)
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += row['factorization_calls']
        elif event == 'JOINT_PREPARATION':
            require(key not in preparations, 'Duplicate preparation'); preparations[key] = row
            counts['joint_preparation_count'] += 1
            for field in ('geometry_fit_count', 'geometry_factorization_count'): counts[field] += row[field]
        else:
            require(event == 'CANDIDATE_FIT' and row['state'] in STATES, 'Unexpected fit stage')
            require(key not in candidates, 'Duplicate candidate stage'); candidates[key] = row
        group = (row['scope'], row['state'], row['train_k'])
        add_stats(resources.setdefault(group, {}), {field: row.get(field) for field in
            ('fit_seconds', 'score_seconds', 'fit_and_score_seconds', 'prepare_seconds', 'geometry_seconds',
             'persistent_state_bytes', 'head_state_bytes', 'shared_geometry_state_bytes', 'theta_state_bytes',
             'transient_distance_bytes', 'inner_geometry_state_bytes', 'geometry_state_bytes')})
    stages = {}; pending = {}; inner_prepared = {}; event_counts = Counter()
    for full, compact, csvrow in itertools.zip_longest(jsonlines(paths['training_events.jsonl']),
            jsonlines(paths['training_events_compact.jsonl']), csvlines(paths['training_events_compact.csv'])):
        require(full is not None and compact is not None and csvrow is not None, 'Training event stream length mismatch')
        require(compact_event(full) == compact, 'Full/compact event mismatch')
        check_csv(csvrow, compact, paths['training_events_compact.csv']); nonfinite += finite_count(full)
        require(full.get('source_validation') is None and full.get('objective_scope') == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION',
            'Training-only objective contract mismatch')
        event = full['event']; event_counts[event] += 1; key = stage_key(row_id, full)
        if event == 'JOINT_INNER_PREPARED':
            prepkey = stage_key(row_id, dict(full, state=full['state'].removesuffix('_prepare')))
            require(prepkey in preparations, 'Inner preparation references absent preparation')
            folded = inner_prepared.setdefault(prepkey, {})
            require(full['inner_fold'] not in folded, 'Duplicate inner preparation')
            train, held, geom = [set(full[field]) for field in
                ('training_physical_ids', 'held_physical_ids', 'geometry_training_physical_ids')]
            require(not train & held and geom <= train and full['train_physical_count'] == len(train)
                and full['held_physical_count'] == len(held), 'Inner physical support partition mismatch')
            folded[full['inner_fold']] = full
            continue
        require(key in candidates and full['mode'] == STATES[full['state']], 'Event references absent/mismatched candidate')
        require(key not in stages, 'Event after final candidate')
        if event == 'JOINT_SPECTRAL_STEP':
            steps = pending.setdefault(key, [])
            require(full['step'] == len(steps)+1 <= 8, 'Missing/reordered/repeated update')
            steps.append(full); continue
        require(event == 'JOINT_SPECTRAL_FIT' and full['status'] == 'JOINT_STAGE_COMPLETE', 'Unexpected/incomplete training event')
        require(full['config'] == algorithm and candidates[key]['config'] == scalars(algorithm), 'Frozen stage config mismatch')
        for field, value in compact.items():
            if field != 'event': require(candidates[key].get(field) == value, 'Final event/fit-stage mismatch: '+field)
        steps = pending.pop(key, []); update_counts = validate_steps(full, steps, algorithm)
        prepkey = stage_key(row_id, dict(full, state=full['state'][0]))
        require(prepkey in preparations, 'Candidate references absent preparation')
        prep = full['preparation']
        for field, value in scalars(prep).items():
            require(preparations[prepkey].get(field) == value, 'Preparation lineage mismatch: '+field)
        prepared_events = inner_prepared.get(prepkey, {})
        require(len(prep['inner_folds']) == len(prepared_events), 'Inner preparation event coverage mismatch')
        for folded in prep['inner_folds']:
            other = prepared_events[folded['inner_fold']]
            require(all(other.get(field) == value for field, value in folded.items()), 'Inner preparation event/audit mismatch')
        spectra = [f['geometry_audit'].get('spectral_eigenvalues') for f in prep['inner_folds']]
        spectra = spectra if spectra and all(isinstance(v, list) for v in spectra) else None
        if spectra is not None:
            require(all(0 <= v <= 1+1e-12 for spectrum in spectra for v in spectrum), 'Invalid normalized geometry spectrum')
        anchor, theta = full['theta_anchor'], full['theta']
        if full['mode'] == 'C_seq':
            bkey = stage_key(row_id, dict(full, state='B_joint'))
            require(bkey in stages and anchor == stages[bkey]['theta'], 'Sequential B/C theta anchor mismatch')
        elif full['mode'] != 'fixed': require(anchor == [0., 0.], 'B/reset anchor must be zero')
        first = metrics(steps[0], steps[0]['theta_pre'], anchor, spectra) if steps else metrics(None)
        last = metrics(steps[-1], steps[-1]['theta_pre'], anchor, spectra) if steps else metrics(None)
        final = metrics(full['final_objective'], theta, anchor, spectra)
        deltas = {field: final[field]-first[field] if final[field] is not None and first[field] is not None else None
                  for field in ('loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy')}
        curve = [dict(step=s['step'], timing='pre_update_objective_with_post_update_theta_fields',
                      active_nonnegative_constraints=s['active_nonnegative_constraints'], active_sum_constraint=s['active_sum_constraint'],
                      metrics=metrics(s, s['theta_pre'], anchor, spectra)) for s in steps]
        curve.append(dict(step=None, timing='final_after_updates' if steps else 'final_without_objective',
            active_nonnegative_constraints=[v == 0 for v in theta], active_sum_constraint=sum(theta) == 1, metrics=final))
        trained = bool(steps); fixed = full['mode'] == 'fixed'
        stages[key] = dict(identity=identity(key), mode=full['mode'], steps=len(steps), no_information=full['no_information'],
            fixed_control=fixed, trained=trained, no_update_reason=full.get('no_update_reason'),
            theta=theta, theta_anchor=anchor, theta_changed_from_anchor=theta != anchor,
            identity_forward=full['identity_forward'], all_projected_zero=trained and update_counts['nonzero_projected_updates'] == 0,
            all_gradient_zero=trained and update_counts['gradient_zero_steps'] == len(steps),
            counts=update_counts, first=first, last=last, final=final, final_minus_initial=deltas,
            inner_spectral_eigenvalues=spectra, full_final_geometry_spectrum=None,
            curve=curve, costs={field: full.get(field) for field in ADDITIVE+('fit_seconds', 'persistent_state_bytes',
                'shared_geometry_state_bytes', 'head_state_bytes', 'theta_state_bytes', 'trainable_parameter_count')},
            score_seconds=candidates[key].get('score_seconds'))
        counts['fixed_stage_count' if fixed else 'joint_stage_count'] += 1
        counts['trained_joint_stage_count'] += int(trained)
        for field in ADDITIVE: counts[field] += full[field]
    require(not pending and set(stages) == set(candidates), 'Missing candidate final events')
    counts['head_fit_count'] = counts['baseline_head_fit_count']+counts['inner_head_fit_count']+counts['final_head_fit_count']
    counts['factorization_count'] = counts['baseline_factorization_count']+counts['inner_factorization_count']+counts['final_factorization_count']
    counts['structured_nonfinite_count'] = nonfinite
    texts = [scan_text(path, jsonlines(paths['training_events_compact.jsonl']), jsonlines(paths['fit_stages.jsonl']))
             for path in (lane/'training.log', lane.parent/'probe.log')]
    resource_rows = [dict(row_id=row_id, scope=key[0], state=key[1], train_k=key[2], metrics=finish_stats(value))
                     for key, value in sorted(resources.items())]
    return dict(row_id=row_id, stages=list(stages.values()), resource_groups=resource_rows, coverage=dict(counts),
        event_counts=dict(event_counts), inventory=[dict(path=str(p), bytes=p.stat().st_size) for p in paths.values()],
        text_logs=texts, failure_artifact_present=(lane/'probe_failed.json').exists())


def aggregate(stages):
    groups = {}; curves = {}; snapshots = {}
    for stage in stages:
        i = stage['identity']; group = (i['scope'], i['state'], i['train_k'])
        count = groups.setdefault(group, Counter()); count['stages'] += 1; count['steps'] += stage['steps']
        for key in ('trained', 'fixed_control', 'no_information', 'theta_changed_from_anchor', 'identity_forward',
                    'all_projected_zero', 'all_gradient_zero'): count[key+'_stages'] += int(stage[key])
        count['skipped_joint_stages'] += int(not stage['fixed_control'] and not stage['trained'])
        count.update(stage['counts'])
        for timing in ('first', 'last', 'final', 'final_minus_initial'):
            add_stats(snapshots.setdefault(group+(timing,), {}), stage[timing])
        for point in stage['curve']:
            add_stats(curves.setdefault(group+(point['timing'], point['step']), {}), point['metrics'])
    coords = lambda key: dict(scope=key[0], state=key[1], train_k=key[2])
    return dict(group_counts=[dict(coords(k), **v) for k, v in sorted(groups.items())],
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
        for key in ('baseline_head_fit_count', 'joint_preparation_count', 'joint_stage_count', 'fixed_stage_count')+tuple(MAXIMUM):
            require(lane['coverage'].get(key, 0) == item['marker'][key], 'Measured lane count mismatch: '+key)
        stages.extend(lane.pop('stages'))
        resource_rows.extend(lane.pop('resource_groups'))
        lanes.append(lane)
    for key in dict(EXACT, **MAXIMUM):
        require(sum(item['marker'][key] for item in metadata) == complete[key], 'Run/row total mismatch: '+key)
    require(complete['optimizer_steps'] == 8*complete['trained_joint_stage_count']
        and complete['inner_objective_evaluation_count'] == 9*complete['trained_joint_stage_count'], 'Run update/objective count mismatch')
    nonfinite = sum(lane['coverage']['structured_nonfinite_count'] for lane in lanes)
    run_scan = scan_text(Path(run_log) if run_log else root/'run.log')
    observations = Counter()
    for scan in [run_scan]+[text for lane in lanes for text in lane['text_logs']]: observations.update(scan['marker_line_counts'])
    return dict(status='COMPLETE_JOINT_TRAINING_LOG_SCAN_VERIFIED' if not nonfinite else 'COMPLETE_SCAN_NONFINITE_VALUES_FOUND',
        run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=dict({key: complete[key] for key in dict(EXACT, **MAXIMUM)}, candidate_final_events=len(stages),
            structured_nonfinite_count=nonfinite, failure_artifacts=sum(lane['failure_artifact_present'] for lane in lanes)),
        algorithm=algorithm, source_metadata=metadata, lanes=lanes, release_run_log=run_scan,
        text_marker_line_counts=dict(observations), stage_summaries=stages, resource_groups=resource_rows,
        totals=dict(candidate_stages=len(stages), trained_joint_stages=sum(s['trained'] for s in stages),
            skipped_joint_stages=sum(not s['fixed_control'] and not s['trained'] for s in stages),
            fixed_control_stages=sum(s['fixed_control'] for s in stages),
            all_projected_zero_trained_stages=sum(s['all_projected_zero'] for s in stages),
            all_gradient_zero_trained_stages=sum(s['all_gradient_zero'] for s in stages),
            changed_from_anchor_joint_stages=sum(not s['fixed_control'] and s['theta_changed_from_anchor'] for s in stages),
            joint_identity_forward_stages=sum(not s['fixed_control'] and s['identity_forward'] for s in stages),
            **{key: sum(s['counts'].get(key, 0) for s in stages) for key in
               ('nonzero_projected_updates', 'zero_projected_updates', 'projection_cancelled_updates', 'gradient_zero_steps')}),
        **aggregate(stages), limitations=[
            'All records and text lines scanned; full/compact/CSV/text streams reconciled. No query or outer-held scores read.',
            'First and last are pre-update objectives at steps 1 and 8. Final objective is after all 8 updates.',
            'Inner accuracy pools physical inner-held training counts; it is part of the training objective, not independent validation.',
            'Actual stages receive equal weight within scope/state/train K; updates are not independent experiments.',
            'Fixed controls are separate from learned joint stages. Skipped/no-information objectives remain null.',
            'Contraction bound theta_sum/2 uses the frozen normalized covariance contract. Inner spectral gain uses recorded inner geometry only; full final geometry gain is unrecorded/null.',
            'Text markers count matching lines, overlap, and include config declarations; they do not by themselves prove a failure or restart.',
            'Training/score wall time and RSS refer to the recorded CPU platform; GPU memory and unmeasured metrics remain null.',
            'Ground data payload, full fitted state RAM, and code release archive are distinct. No deployment bytes inferred from theta size.',
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
    write_csv(destination/'stage_summaries.csv', rows); write_csv(destination/'stages.csv', rows)
    write_csv(destination/'stage_curves.csv', curves)
    write_csv(destination/'group_counts.csv', result['group_counts'])
    for name in ('snapshots', 'curves', 'resource_groups'):
        write_csv(destination/(name+'.csv'), (dict({k: v for k, v in row.items() if k != 'metrics'}, metric=metric, **stats)
            for row in result[name] for metric, stats in row['metrics'].items()))
    summary = {key: value for key, value in result.items() if key not in ('stage_summaries',)}
    with (destination/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')
    with (destination/'report.md').open('x', encoding='utf-8') as stream:
        stream.write('# Complete joint training log diagnostics\n\n')
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
            '`stages.csv` retains exact theta, anchors, step counts and projection/gradient statistics. '
            'Spectral contraction uses recorded inner spectra; unrecorded final full geometry remains null.\n\n')
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
