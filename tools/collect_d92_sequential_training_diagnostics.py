"""Read complete residual training logs; never fit, score, or open held artifacts.

The module is standard-library only and can run from SSH stdin without copying
or writing a file remotely. Only the local CLI writer creates output files.
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

STATUS = 'SEQUENTIAL_RESIDUAL_PROBE_COMPLETE'
METRICS = ('loss_ce', 'loss_proximal_U', 'loss_proximal_V', 'loss_total', 'training_accuracy',
    'residual_rms', 'base_logits_rms', 'residual_to_base_rms', 'post_U_anchor_distance',
    'post_V_anchor_distance', 'post_U_norm', 'post_V_norm', 'gradient_norm_pre_clip',
    'gradient_norm_post_clip', 'gradient_clip_scale', 'update_norm', 'learning_rate', 'elapsed_seconds')
COORDS = ('split_id', 'scope', 'fold', 'trial', 'state', 'train_k')
FIXED = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
    sequence_paths=1760, head_fit_count=3168, residual_training_stages=4576, optimizer_steps=292864)
LOG_PATTERNS = {
    'error': re.compile(r'(?i)\b(?:error|failed|failure|exception)\b'),
    'warning': re.compile(r'(?i)\b(?:warning|warn)\b'),
    'traceback': re.compile(r'(?i)\btraceback\b'),
    'oom_or_killed': re.compile(r'(?i)\b(?:out of memory|oom|killed)\b'),
    'nonfinite_text': re.compile(r'(?i)(?<![A-Za-z_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z_])'),
    'recovery_or_resume': re.compile(r'(?i)\b(?:resume|resumed|resuming|recover|recovery|restart|retry)\b'),
    'early_stop': re.compile(r'(?i)\bearly[ -]stop(?:ping|ped)?\b'),
    'configuration': re.compile(r'"(?:config|algorithm|learning_rate)"\s*:|(?i:\bconfiguration\b)'),
    'environment': re.compile(r'"(?:python|hardware|blas_environment)"\s*:|(?i:\bCUDA_VISIBLE_DEVICES\b)'),
    'launch_command': re.compile(r'"argv"\s*:|(?i:\b(?:command|launching)\b)'),
}


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scalars(value):
    if not isinstance(value, dict): return value
    return {key: scalars(item) for key, item in value.items()
            if isinstance(item, dict) or item is None or isinstance(item, (str, int, float, bool))}


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
    require(None not in row, 'Malformed CSV columns: '+str(path))
    require(all(row[key] == csv_value(source.get(key)) for key in row), 'CSV/JSON mismatch: '+str(path))
    require(set(source) <= set(row), 'CSV omits recorded JSON fields: '+str(path))


def finite_count(value):
    if isinstance(value, dict): return sum(finite_count(v) for v in value.values())
    if isinstance(value, list): return sum(finite_count(v) for v in value)
    return int(isinstance(value, float) and not math.isfinite(value))


def metrics(row):
    result = {}
    for key in METRICS:
        value = row.get(key)
        result[key] = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    numerator, denominator = result['residual_rms'], result['base_logits_rms']
    ratio = numerator/denominator if numerator is not None and denominator is not None and denominator > 0 else None
    result['residual_to_base_rms'] = ratio if ratio is not None and math.isfinite(ratio) else None
    return result


def stage_key(row_id, row):
    require(row.get('state') in ('B', 'C_reset', 'C_seq'), 'Unexpected residual stage')
    scope = row.get('scope')
    require(scope in ('support_oof', 'support_oneshot_proxy'), 'Unexpected training scope')
    fold, trial = row.get('fold'), row.get('trial')
    require((type(fold) is int and trial is None) if scope == 'support_oof' else
            (fold is None and type(trial) is int), 'Missing/ambiguous physical fold/anchor')
    require(type(row.get('train_k')) is int and row['train_k'] > 0, 'Missing physical train K')
    return (row_id,)+tuple(row[key] for key in COORDS)


def stage_identity(key):
    return dict(zip(('row_id',)+COORDS, key))


def add_stats(destination, values):
    for key, value in values.items():
        cell = destination.setdefault(key, dict(count=0, missing_count=0, sum=0., minimum=None, maximum=None))
        if value is None:
            cell['missing_count'] += 1
        else:
            cell['count'] += 1; cell['sum'] += value
            cell['minimum'] = value if cell['minimum'] is None else min(cell['minimum'], value)
            cell['maximum'] = value if cell['maximum'] is None else max(cell['maximum'], value)


def finish_stats(destination):
    return {key: dict(count=cell['count'], missing_count=cell['missing_count'],
        mean=cell['sum']/cell['count'] if cell['count'] else None,
        minimum=cell['minimum'], maximum=cell['maximum']) for key, cell in destination.items()}


def scan_text(path, expected_steps=None, expected_stages=None):
    """Scan every line; compare training records, discard held completion payloads."""
    path = Path(path); patterns = Counter(); events = Counter(); lines = 0; nonfinite = 0
    steps = iter(expected_steps) if expected_steps is not None else None
    stages = iter(expected_stages) if expected_stages is not None else None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            lines += 1
            for name, pattern in LOG_PATTERNS.items(): patterns[name] += int(bool(pattern.search(line)))
            if line.startswith('SUPPORT_RESIDUAL_STEP '):
                row = json.loads(line.split(' ', 1)[1]); events['SUPPORT_RESIDUAL_STEP'] += 1
                nonfinite += finite_count(row)
                if steps is not None: require(row == next(steps, None), 'Text/step stream mismatch: '+str(path))
            elif line.startswith('{'):
                # Parent-complete lines may contain held summaries. Do not deserialize
                # those payloads into this training-only analysis.
                match = re.search(r'"event"\s*:\s*"([^"\\]+)"', line)
                event = match.group(1) if match else None
                if event: events[event] += 1
                if event in ('BASE_FIT', 'RESIDUAL_FIT'):
                    row = json.loads(line); nonfinite += finite_count(row)
                    if stages is not None: require(row == next(stages, None), 'Text/stage stream mismatch: '+str(path))
    if steps is not None: require(next(steps, None) is None, 'Text log missing training steps: '+str(path))
    if stages is not None: require(next(stages, None) is None, 'Text log missing fit stages: '+str(path))
    return dict(path=str(path), bytes=path.stat().st_size, lines=lines,
        marker_line_counts={key: patterns[key] for key in LOG_PATTERNS}, events=dict(events),
        parsed_training_nonfinite_count=nonfinite,
        interpretation='Marker counts are text observations, not automatic diagnoses; counts can overlap.')


def collect_lane(lane, row_id, algorithm):
    """Parse every structured and text record for one lane, preserving real stages."""
    lane = Path(lane); stages = {}; base_count = 0; stage_count = 0; nonfinite = 0
    paths = {'stages': lane/'fit_stages.jsonl', 'stages_csv': lane/'fit_stages.csv',
        'steps': lane/'training_steps.jsonl', 'compact': lane/'training_steps_compact.jsonl',
        'compact_csv': lane/'training_steps_compact.csv'}
    for row, csvrow in itertools.zip_longest(jsonlines(paths['stages']), csvlines(paths['stages_csv'])):
        require(row is not None and csvrow is not None, 'Fit stages JSON/CSV length mismatch')
        check_csv(csvrow, row, paths['stages_csv']); stage_count += 1; nonfinite += finite_count(row)
        if row['event'] == 'BASE_FIT':
            base_count += 1
            continue
        require(row['event'] == 'RESIDUAL_FIT', 'Unexpected fit stage event')
        key = stage_key(row_id, row)
        require(key not in stages, 'Duplicate physical training stage')
        require(row['config'] == scalars(algorithm) and row['optimizer_steps'] == algorithm['steps'] == 64
            and row['final_state_timing'] == 'after_64_updates' and row['source_validation'] is None
            and row['held_used_for_training'] is False and row['optimizer_state_inherited'] is False,
            'Stage final/config/support-only mismatch')
        stages[key] = dict(identity=stage_identity(key), final=metrics(row), steps=0,
            first=None, last=None, clip_count=0, zero_count=0, clip_missing_count=0, zero_missing_count=0,
            inherited=row['inherited'], optimizer_state_inherited=row['optimizer_state_inherited'], q=row['q'],
            final_fit_seconds=row.get('fit_seconds'), classes=row.get('class_count'),
            train_physical_count=row.get('train_physical_count'), final_state_timing=row['final_state_timing'])
    curves = {}; total_steps = 0
    for full, compact, csvrow in itertools.zip_longest(jsonlines(paths['steps']), jsonlines(paths['compact']), csvlines(paths['compact_csv'])):
        require(full is not None and compact is not None and csvrow is not None, 'Training step stream length mismatch')
        require(scalars(full) == compact, 'Full/compact training step mismatch')
        check_csv(csvrow, compact, paths['compact_csv']); nonfinite += finite_count(full)
        key = stage_key(row_id, full); require(key in stages, 'Step references absent final stage')
        stage = stages[key]; step = full['step']
        require(type(step) is int and step == full['optimizer_steps'] == stage['steps']+1 <= 64, 'Missing/reordered/repeated step')
        require(full['state_timing'] == 'loss_gradient_pre_update_parameters_post_update'
            and full['optimizer_state_inherited'] is False and full['inherited'] == stage['inherited']
            and full['q'] == stage['q'] and full['source_validation'] is None,
            'Step inheritance/timing/training contract mismatch')
        for field in ('learning_rate', 'beta1', 'beta2', 'adam_epsilon', 'global_gradient_norm_clip'):
            require(full[field] == algorithm[field], 'Step optimizer config mismatch: '+field)
        values = metrics(full); stage['steps'] += 1; total_steps += 1
        if step == 1: stage['first'] = values
        stage['last'] = values
        clip = values['gradient_clip_scale']; zero = full.get('gradient_zero')
        stage['clip_count'] += int(clip is not None and clip < 1.)
        stage['clip_missing_count'] += int(clip is None)
        stage['zero_count'] += int(zero is True); stage['zero_missing_count'] += int(type(zero) is not bool)
        group = (full['scope'], full['state'], full['train_k'], step)
        add_stats(curves.setdefault(group, {}), values)
    require(all(stage['steps'] == 64 for stage in stages.values()), 'Incomplete final-stage step coverage')
    texts = []
    for path in (lane/'training.log', lane.parent/'probe.log'):
        texts.append(scan_text(path, jsonlines(paths['compact']), jsonlines(paths['stages'])))
    inventory = [dict(path=str(path), bytes=path.stat().st_size) for path in paths.values()]
    return dict(row_id=row_id, stages=list(stages.values()), curves=curves,
        coverage=dict(base_fit_stages=base_count, residual_training_stages=len(stages),
            fit_stage_records=stage_count, optimizer_steps=total_steps, structured_nonfinite_count=nonfinite),
        inventory=inventory, text_logs=texts)


def merge_stats(target, source):
    for name, incoming in source.items():
        cell = target.setdefault(name, dict(count=0, missing_count=0, sum=0., minimum=None, maximum=None))
        cell['count'] += incoming['count']; cell['missing_count'] += incoming['missing_count']; cell['sum'] += incoming['sum']
        for key, compare in (('minimum', min), ('maximum', max)):
            if incoming[key] is not None: cell[key] = incoming[key] if cell[key] is None else compare(cell[key], incoming[key])


def collect(run_root, run_log=None):
    root = Path(run_root)
    startup, complete, states = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    spec = startup['spec']; algorithm = spec['probe']['algorithm']; rows = spec['rows']
    require(complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == len(rows) == 4
        and complete['commit'] == startup['commit'] and all(complete[key] == value for key, value in FIXED.items()),
        'Complete fixed four-row training pilot required')
    require(set(states) == {row['row_id'] for row in rows}, 'Row state coverage mismatch')
    require(startup['query_access'] is complete['query_access'] is False
        and startup['source_sample_access'] is complete['source_sample_access'] is False, 'Forbidden run access')
    markers = []
    # Reconcile all terminal metadata before opening large logs.
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
        for key, value in FIXED.items():
            require(marker[key] == states[row['row_id']][key] == value//4, 'Lane coverage mismatch: '+key)
        require(marker['query_rows_used'] == marker['source_rows_used'] == launch['query_rows_used'] == launch['source_rows_used'] == 0
            and marker['truth_read'] is launch['truth_read'] is False, 'Forbidden lane access')
        markers.append(dict(row_id=row['row_id'], marker=marker, hardware=launch.get('hardware'),
            blas_environment=launch.get('blas_environment'), config=launch['config']))
    stage_rows = []; curve_groups = {}; snapshots = {}; group_counts = {}; lane_results = []
    for item in markers:
        lane = collect_lane(root/item['row_id']/'probe', item['row_id'], algorithm)
        require(lane['coverage']['base_fit_stages'] == item['marker']['head_fit_count']
            and lane['coverage']['residual_training_stages'] == item['marker']['residual_training_stages']
            and lane['coverage']['optimizer_steps'] == item['marker']['optimizer_steps'], 'Measured lane count mismatch')
        for key, cells in lane.pop('curves').items(): merge_stats(curve_groups.setdefault(key, {}), cells)
        for stage in lane.pop('stages'):
            stage_rows.append(stage); identity = stage['identity']
            group = (identity['scope'], identity['state'], identity['train_k'])
            counts = group_counts.setdefault(group, Counter())
            counts['stages'] += 1; counts['steps'] += stage['steps']
            for key in ('clip_count', 'zero_count', 'clip_missing_count', 'zero_missing_count'): counts[key] += stage[key]
            for timing in ('first', 'last', 'final'):
                add_stats(snapshots.setdefault(group+(timing,), {}), stage[timing])
        lane_results.append(lane)
    require(len(stage_rows) == 4576 and sum(lane['coverage']['optimizer_steps'] for lane in lane_results) == 292864,
            'Full actual training-stage coverage mismatch')
    run_scan = scan_text(Path(run_log) if run_log else root/'run.log')
    nonfinite = sum(lane['coverage']['structured_nonfinite_count'] for lane in lane_results)
    return dict(status='COMPLETE_TRAINING_LOG_SCAN_VERIFIED' if nonfinite == 0 else 'COMPLETE_SCAN_NONFINITE_VALUES_FOUND',
        run_id=spec['run_id'], release_commit=complete['commit'], coverage=dict(FIXED, parsed_nonfinite_count=nonfinite),
        algorithm=algorithm, source_metadata=markers, lanes=lane_results, release_run_log=run_scan,
        stage_summaries=stage_rows,
        group_counts=[dict(scope=key[0], stage=key[1], train_k=key[2], **counts) for key, counts in sorted(group_counts.items())],
        snapshots=[dict(scope=key[0], stage=key[1], train_k=key[2], timing=key[3], metrics=finish_stats(values))
                   for key, values in sorted(snapshots.items())],
        curves=[dict(scope=key[0], stage=key[1], train_k=key[2], step=key[3], metrics=finish_stats(values))
                for key, values in sorted(curve_groups.items())],
        limitations=[
            'First/last denote pre-update loss at updates 1/64; final denotes independently logged after-64 objective.',
            'Each curve point weights actual stages equally within scope/stage/train K. Steps are not independent experiments.',
            'Gradient and post-update parameter fields can be absent in final audits and remain null.',
            'Residual/base RMS ratio is derived only when both recorded RMS values exist and base RMS is positive.',
            'All listed JSONL/CSV and text logs were fully parsed/scanned; held results are not extracted or analyzed.',
            'Text error/warning/config/recovery counters are matching lines, can overlap, and do not establish a failure.',
            'Logs belong to a completed immutable pilot; no active process intervention, training, or remote writes occur.'])


def write_csv(path, rows):
    rows = list(rows)
    with path.open('x', encoding='utf-8', newline='') as stream:
        if not rows: return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)


def write_outputs(output, result):
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    with (destination/'training_diagnostics.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':')); stream.write('\n')
    stage_rows = []
    for stage in result['stage_summaries']:
        row = dict(stage['identity'], **{key: stage[key] for key in ('steps', 'clip_count', 'zero_count',
            'clip_missing_count', 'zero_missing_count', 'inherited', 'q', 'final_fit_seconds', 'classes', 'train_physical_count')})
        for timing in ('first', 'last', 'final'):
            row.update({timing+'_'+key: value for key, value in stage[timing].items()})
        stage_rows.append(row)
    write_csv(destination/'stage_summaries.csv', stage_rows)
    write_csv(destination/'group_counts.csv', result['group_counts'])
    for name in ('snapshots', 'curves'):
        write_csv(destination/(name+'.csv'), (dict({key: value for key, value in row.items() if key != 'metrics'},
            metric=metric, **stats) for row in result[name] for metric, stats in row['metrics'].items()))
    summary = {key: value for key, value in result.items() if key not in ('stage_summaries', 'curves')}
    with (destination/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True); parser.add_argument('--run-log'); parser.add_argument('--output')
    parser.add_argument('--ssh-host'); parser.add_argument('--ssh-config')
    parser.add_argument('--remote-python', default='python3')
    parser.add_argument('--collect-stdout', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.collect_stdout or args.output, '--output required for local delivery')
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
    else:
        result = collect(args.run_root, args.run_log)
    if args.collect_stdout:
        json.dump(result, sys.stdout, allow_nan=False, separators=(',', ':')); sys.stdout.write('\n')
    else:
        write_outputs(args.output, result)
        print(json.dumps(dict(status=result['status'], coverage=result['coverage'], output=str(Path(args.output).resolve()))))


if __name__ == '__main__': main()
