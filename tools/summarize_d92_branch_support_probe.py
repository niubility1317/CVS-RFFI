"""Stream complete support-only probe diagnostics; never read IQ or query scores."""
import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG

SCOPE = 'SUPPORT_OOF_DIAGNOSTIC_NOT_QUERY_EVALUATION'
AXES = ('receivers', 'scenarios', 'ks', 'new_counts', 'support_seeds')
COORDS = ('receiver', 'scenario', 'k', 'new_count', 'support_seed')
ARMS = tuple(FROZEN_CONFIG['arms'])
METRICS = ('accuracy', 'macro_accuracy', 'old_accuracy', 'new_accuracy', 'h', 'macro_nll')
STRATA = dict(by_k=('k',), by_model_seed=('model_seed',), by_support_seed=('support_seed',),
              by_cohort=('cohort',), by_receiver_scene=('receiver', 'scenario'),
              by_k_newcount=('k', 'new_count'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        check(math.isfinite(value), 'Nonfinite JSON diagnostic')
    elif isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, list):
        for child in value: finite_tree(child)


def decode(text):
    value = json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))
    finite_tree(value)
    return value


def read(path):
    return decode(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            check(bool(line.strip()), f'Empty record: {path}:{number}')
            yield decode(line)


def close(a, b, message):
    if a is None or b is None:
        check(a is b, message)
    else:
        check(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11), message)


def zero_fields(record, fields):
    for key in fields:
        check(type(record.get(key)) is int and record[key] == 0, 'Expected integer zero: '+key)


def false_fields(record, fields):
    for key in fields:
        check(record.get(key) is False, 'Forbidden or missing flag: '+key)


@dataclass
class Stat:
    count: int = 0
    null_count: int = 0
    total: float = 0.
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value):
        if value is None:
            self.null_count += 1
            return
        check(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'Invalid numeric statistic')
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self):
        return dict(count=self.count, null_count=self.null_count, sum=self.total if self.count else None,
                    mean=self.total/self.count if self.count else None,
                    min=self.minimum if self.count else None, max=self.maximum if self.count else None)


def add(stats, key, value):
    stats.setdefault(key, Stat()).add(value)


def leaves(value, prefix=''):
    for key, child in value.items():
        name = prefix+'.'+key if prefix else key
        if isinstance(child, dict):
            yield from leaves(child, name)
        elif child is None or (isinstance(child, (int, float)) and not isinstance(child, bool)):
            yield name, child


def scalar_tree(value):
    if not isinstance(value, dict): return value
    return {k: scalar_tree(v) for k, v in value.items()
            if isinstance(v, dict) or v is None or isinstance(v, (str, int, float, bool))}


def expected_cells(matrix):
    check(set(matrix) == set(AXES), 'Matrix axes mismatch')
    for key in AXES:
        values = matrix[key]
        check(isinstance(values, list) and values and len(set(values)) == len(values), 'Invalid matrix axis '+key)
    return set(itertools.product(*(matrix[key] for key in AXES)))


def check_bind(record, row, cohort):
    check(record.get('checkpoint_sha256') == row['expected_checkpoint_sha256'], 'Checkpoint marker mismatch')
    check(record.get('capsule_id') == cohort['capsule_id'], 'Capsule marker mismatch')
    check(record.get('model_seed') == row['seeds']['model'], 'Model seed marker mismatch')


def verify_stage(stage, n, c, train_k):
    arm = stage['arm']
    check(arm in ARMS, 'Unexpected probe arm')
    background = 'zfft' if arm.startswith('zfft') else 'z'
    dim = 256 if background == 'zfft' else 160
    d = dim+(dim if arm.endswith('_duplicate') else 480 if arm.endswith('_aux') else 0)
    outputs = c+480 if arm == background else c
    check(stage['background'] == background and stage['design_dim'] == d
          and stage['output_dim'] == outputs and stage['classification_output_dim'] == c, 'Stage dimensions mismatch')
    check(stage['train_physical_count'] == n and stage['train_k'] == train_k
          and stage['physical_loss_mass'] == n and stage['sample_weight'] == 1.
          and stage['ridge_coefficient'] == 1., 'Stage physical-sum contract mismatch')
    check(stage['all_states_estimated_from_trainfold_only'] is True, 'Trainfold isolation declaration missing')
    zero_fields(stage, ('optimizer_steps',))
    check(stage['learning_rate'] is None and stage['epoch'] is None and stage['status'] == 'CLOSED_FORM_SOLVED', 'Analytical fit status mismatch')
    check(stage['factorization_calls'] == 1 and stage['factorization_dim'] == min(n, d)
          and stage['solver'] == ('primal' if d <= n else 'dual'), 'Actual factorization mismatch')
    check(stage['coefficient_bytes'] == 8*d*outputs and stage['intercept_bytes'] == 8*outputs
          and stage['temporary_state_bytes'] == 8*(d+1)*outputs, 'Temporary head storage mismatch')
    close(stage['target_norm_squared'], n*(1-1/c), 'Centered onehot target mass mismatch')
    close(stage['loss_total'], stage['loss_data']+stage['loss_ridge'], 'Classification objective components mismatch')
    if arm == background:
        close(stage['reconstruction_loss_total'], stage['reconstruction_loss_data']+stage['reconstruction_loss_ridge'], 'Reconstruction objective components mismatch')
    else:
        check(all(v is None for key, v in stage.items() if key.startswith('reconstruction_')), 'Unexpected reconstruction in nonbaseline arm')
    for key, value in stage.items():
        if value is not None and (key.endswith('_seconds') or key.endswith('_bytes') or key.startswith('loss_')
                or key.endswith('gradient_norm') or key.endswith('equation_residual')):
            check(value >= 0, 'Negative stage measurement '+key)


def verify_oof(record):
    c, k = len(record['classes']), record['k']
    classes, old = set(record['classes']), set(record['old_classes'])
    check(len(classes) == c and old <= classes, 'Invalid class registry')
    check(record['new_count'] == len(classes-old), 'Old/new class count mismatch')
    n = c*k
    check(record['support_count'] == n, 'Physical count mismatch')
    folds = 0 if k == 1 else min(k, 3)
    check(record['fold_count'] == folds and len(record['folds']) == folds
          and record['factorization_count'] == folds*6, 'Fold/factorization count mismatch')
    zero_fields(record, ('optimizer_steps', 'persistent_state_bytes', 'query_rows_used', 'source_rows_used'))
    if k == 1:
        check(all(record[key] is None for key in ('oof', 'reconstruction', 'paired'))
              and record['physical_fold_assignment'] == []
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'K1 fabricated held evidence')
        return
    assignment = record['physical_fold_assignment']
    lookup = {r['physical_id']: r for r in assignment}
    check(len(assignment) == len(lookup) == n, 'Duplicate or missing physical assignment')
    for cls in classes:
        selected = sorted((r for r in assignment if r['class_id'] == cls), key=lambda r: r['physical_id'])
        check(len(selected) == k and [r['fold'] for r in selected] == [i % folds for i in range(k)], 'Physical stratified fold mismatch')
    check({r['class_id'] for r in assignment} == classes, 'Unexpected assigned class')
    check({f['fold'] for f in record['folds']} == set(range(folds)), 'Duplicate fold')
    for fold in record['folds']:
        held = {pid for pid, r in lookup.items() if r['fold'] == fold['fold']}
        training = set(lookup)-held
        check(set(fold['training_ids']) == training and len(fold['training_ids']) == len(training)
              and set(fold['held_ids']) == held and len(fold['held_ids']) == len(held), 'Held physical isolation mismatch')
        train_k = len(training)//c
        check(fold['train_k'] == train_k and len(fold['stages']) == 6
              and {s['arm'] for s in fold['stages']} == set(ARMS), 'Incomplete six-arm fold')
        for stage in fold['stages']: verify_stage(stage, len(training), c, train_k)
    check(set(record['oof']) == set(ARMS), 'Missing OOF arm')
    for arm, result in record['oof'].items():
        rows = result['rows']; by_id = {r['physical_id']: r for r in rows}
        check(len(rows) == n and set(by_id) == set(lookup), 'Incomplete physical OOF')
        for pid, row in by_id.items():
            check(row['class_id'] == lookup[pid]['class_id'] and row['fold'] == lookup[pid]['fold']
                  and row['predicted_class'] in classes and type(row['correct']) is bool
                  and row['correct'] == (row['class_id'] == row['predicted_class']) and row['nll'] >= 0, 'OOF record mismatch')
        classwise = result['metrics']['classwise']
        check(len(classwise) == c and {r['class_id'] for r in classwise} == classes, 'OOF class metrics mismatch')
        for r in classwise:
            selected = [p for p in rows if p['class_id'] == r['class_id']]
            check(r['count'] == k, 'OOF class count mismatch')
            close(r['accuracy'], sum(p['correct'] for p in selected)/k, 'Class accuracy mismatch')
            close(r['nll'], sum(p['nll'] for p in selected)/k, 'Class NLL mismatch')
        metrics = result['metrics']
        accuracy = sum(p['correct'] for p in rows)/n
        close(metrics['accuracy'], accuracy, 'OOF accuracy mismatch')
        close(metrics['macro_accuracy'], accuracy, 'Class balance mismatch')
        close(metrics['macro_nll'], sum(p['nll'] for p in rows)/n, 'OOF NLL mismatch')
        o = sum(r['accuracy'] for r in classwise if r['class_id'] in old)/len(old) if old else None
        new = sum(r['accuracy'] for r in classwise if r['class_id'] not in old)/(c-len(old)) if c > len(old) else None
        close(metrics['old_accuracy'], o, 'Old accuracy mismatch')
        close(metrics['new_accuracy'], new, 'New accuracy mismatch')
        h = (2*o*new/(o+new) if o+new else 0.) if o is not None and new is not None else None
        close(metrics['h'], h, 'H mismatch')
    check(set(record['reconstruction']) == set(record['paired']) == {'z', 'zfft'}, 'Missing background diagnostic')
    for background, result in record['reconstruction'].items():
        rows = result['rows']
        check(len(rows) == n and {r['physical_id'] for r in rows} == set(lookup), 'Reconstruction physical coverage mismatch')
        check(all(r['fold'] == lookup[r['physical_id']]['fold'] and r['squared_error'] >= 0
                  and r['mean_baseline_squared_error'] >= 0 for r in rows), 'Reconstruction row mismatch')
        sse = sum(r['squared_error'] for r in rows); sst = sum(r['mean_baseline_squared_error'] for r in rows)
        close(result['squared_error_sum'], sse, 'Reconstruction SSE mismatch')
        close(result['mean_baseline_squared_error_sum'], sst, 'Reconstruction reference mismatch')
        close(result['r2_linear'], 1-sse/sst if sst else None, 'Reconstruction R2 mismatch')
        pairs = record['paired'][background]
        check(set(pairs) == {'aux_minus_base', 'duplicate_minus_base', 'aux_minus_duplicate'}, 'Missing paired comparison')
        for name, left, right in (('aux_minus_base', background+'_aux', background),
                                  ('duplicate_minus_base', background+'_duplicate', background),
                                  ('aux_minus_duplicate', background+'_aux', background+'_duplicate')):
            actual = pairs[name]['rows']
            left_rows = {r['physical_id']: r for r in record['oof'][left]['rows']}
            right_rows = {r['physical_id']: r for r in record['oof'][right]['rows']}
            check(len(actual) == n and {r['physical_id'] for r in actual} == set(lookup), 'Paired physical coverage mismatch')
            check(all(r['correct_delta'] == int(left_rows[r['physical_id']]['correct'])-int(right_rows[r['physical_id']]['correct']) for r in actual), 'Paired correctness mismatch')
            close(pairs[name]['mean_correct_delta'], sum(r['correct_delta'] for r in actual)/n, 'Paired mean mismatch')


def episode_metrics(record):
    for key, value in (('episodes', 1), ('k1_episodes', int(record['k'] == 1)),
                       ('oof_episodes', int(record['k'] > 1)), ('factorizations', record['factorization_count'])):
        yield 'coverage', key, value
    for key, value in leaves(record['numerical']):
        yield 'numerical', key, value
    if record['k'] == 1: return
    for arm, result in record['oof'].items():
        for key in METRICS: yield 'arm', arm+'.'+key, result['metrics'][key]
    for background in ('z', 'zfft'):
        for comparison, left, right in (('aux_minus_base', background+'_aux', background),
                                        ('duplicate_minus_base', background+'_duplicate', background),
                                        ('aux_minus_duplicate', background+'_aux', background+'_duplicate')):
            for metric in METRICS:
                a, b = record['oof'][left]['metrics'][metric], record['oof'][right]['metrics'][metric]
                yield 'paired', background+'.'+comparison+'.'+metric, a-b if a is not None and b is not None else None
        r = record['reconstruction'][background]
        for key in ('squared_error_sum', 'mean_baseline_squared_error_sum', 'r2_linear'):
            yield 'reconstruction', background+'.'+key, r[key]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec
    root = Path(run_root or spec['execution']['remote_run_root'])
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    rows = spec['rows']; cohorts = spec['probe']['cohorts']
    check(len(rows) == spec['probe']['model_rows'] == 8 and len({r['row_id'] for r in rows}) == 8, 'Eight unique lanes required')
    check(set(cohorts) == {'rx1', 'rx3'}, 'Both cohorts required')
    check({(r['cohort'], r['seeds']['model']) for r in rows} ==
          {(co, seed) for co in cohorts for seed in range(2026092701, 2026092705)}, 'Complete four-model/cohort matrix required')
    check(spec['probe']['query_access'] is False and spec['probe']['source_payload_bytes'] == 0
          and spec['probe']['full_support_head'] is False, 'Spec permission mismatch')
    expected_total = sum(len(expected_cells(cohorts[r['cohort']]['matrix'])) for r in rows)
    check(expected_total == spec['probe']['total_episodes'], 'Spec total episode mismatch')
    launch = read(root/'startup.json')
    check(launch['spec'] == spec, 'Root startup spec binding mismatch')
    false_fields(launch, ('query_access', 'source_sample_access'))
    complete = read(root/'complete.json')
    check(complete['status'] == 'SUPPORT_PROBE_COMPLETE' and complete['completed_rows'] == complete['model_rows'] == 8
          and complete['episodes'] == expected_total, 'Run not complete')
    false_fields(complete, ('query_access', 'source_sample_access', 'query_performance_claim'))
    zero_fields(complete, ('optimizer_steps',))
    check(complete['commit'] == launch['commit'] and complete['finished'] >= launch['started'], 'Run commit/time binding mismatch')
    state = read(root/'state.json')
    check(set(state) == {r['row_id'] for r in rows} and all(s['status'] == 'SUPPORT_PROBE_COMPLETE' for s in state.values()), 'Lane state incomplete')
    groups = {name: {} for name in STRATA}
    overall = {}; costs = {}; stage_stats = {}; lanes = []
    counts = dict(episodes=0, k1_episodes=0, oof_episodes=0, factorizations=0, physical_oof_records_per_arm=0)
    models = {}
    for row in rows:
        rid = row['row_id']; check(Path(rid).name == rid and '/' not in rid and '\\' not in rid, 'Unsafe lane ID')
        lane = root/rid; co = cohorts[row['cohort']]
        expected = expected_cells(co['matrix'])
        check(len(expected) == co['expected_split_count'], 'Declared cohort matrix incomplete')
        marker = read(lane/'probe/probe_complete.json')
        feature = read(lane/'feature_cache/features_complete.json')
        startup = read(lane/'probe/startup.json')
        for record in (marker, feature, startup): check_bind(record, row, co)
        check(marker['status'] == 'SUPPORT_PROBE_COMPLETE' and marker['scope'] == SCOPE
              and marker['algorithm'] == FROZEN_CONFIG and marker['matrix'] == co['matrix']
              and marker['episodes'] == len(expected), 'Probe marker/config mismatch')
        zero_fields(marker, ('query_rows_used', 'source_rows_used', 'optimizer_steps', 'persistent_state_bytes'))
        false_fields(marker, ('truth_read',))
        check(startup['config'] == dict(algorithm=FROZEN_CONFIG, matrix=co['matrix']), 'Startup config mismatch')
        zero_fields(startup, ('query_rows_used', 'source_rows_used', 'optimizer_steps'))
        false_fields(startup, ('query_iq_access', 'truth_read', 'adapted_state_inherited', 'cross_row_adapted_state_reuse', 'checkpoint_loaded', 'encoder_updated'))
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE'
              and feature['schema'] == 'd92_branch_support_features_v1'
              and feature['view_count_per_observation'] == 1 and feature['split_count'] == len(expected), 'Feature marker mismatch')
        contract = feature['feature_contract']
        check(contract['branch_keys'] == ['z_id', 't_emb', 'f_emb', 'pa_local']
              and contract['branch_dim'] == 160 and contract['fft_dim'] == 96
              and contract['view_count'] == 1 and contract['identity_feature_key'] == 'feat_joint'
              and contract['view'] == 'original_received_observation', 'Unexpected feature branch/view contract')
        false_fields(feature, ('query_iq_access', 'query_used_for_fitting', 'source_data_access', 'truth_read', 'adapted_state_inherited', 'encoder_updated'))
        zero_fields(feature, ('query_rows_read', 'new_source_payload_bytes', 'new_ground_statistics_bytes', 'identity_check_additional_encoder_forwards'))
        check(all(feature.get(k) is True for k in ('native_eval', 'native_parameters_unchanged', 'native_buffers_unchanged')), 'Unfrozen feature producer')
        count = feature['count']
        check(count > 0 and feature['feature_array_bytes'] == count*2944
              and feature['native_physical_forward_count'] == feature['support_iq_rows_read'] == feature['identity_reference_checks'] == count,
              'Feature count/bytes/forward mismatch')
        check((lane/'feature_cache/support_branch_features.npz').stat().st_size == feature['feature_file_bytes'], 'Actual feature cache file bytes mismatch')
        metadata = feature.get('metadata_file_bytes')
        if metadata is not None:
            check(set(metadata) == {'startup.json', 'support_splits.json', 'checkpoint_provenance.json'}, 'Unexpected feature metadata file registry')
            for name, size in metadata.items():
                check((lane/'feature_cache'/name).stat().st_size == size, 'Feature metadata file byte mismatch')
            check(feature['artifact_file_bytes_excluding_completion_marker'] == feature['feature_file_bytes']+sum(metadata.values()), 'Feature artifact byte total mismatch')
        payload = marker['payload_audit']
        check(payload == startup['payload_audit'], 'Payload audit changed')
        for key, source in (('support_feature_array_bytes', 'feature_array_bytes'), ('support_feature_file_bytes', 'feature_file_bytes'),
                            ('model_file_bytes', 'model_file_bytes'), ('native_physical_forward_count', 'native_physical_forward_count'),
                            ('native_batch_calls', 'native_batch_calls')):
            check(payload[key] == feature[source], 'Payload feature cost mismatch '+key)
        zero_fields(payload, ('new_source_payload_bytes', 'new_ground_statistics_bytes'))
        check(payload['feature_extraction_timing'] == feature['timing'], 'Extraction timing mismatch')
        sha = row['expected_checkpoint_sha256']
        check(sha not in models or models[sha] == feature['model_file_bytes'], 'Same model package size mismatch')
        models[sha] = feature['model_file_bytes']
        for key in ('count', 'feature_array_bytes', 'feature_file_bytes', 'model_file_bytes', 'native_physical_forward_count',
                    'native_batch_calls', 'peak_process_rss_bytes', 'index_bytes', 'registry_array_bytes',
                    'artifact_file_bytes_excluding_completion_marker'):
            add(costs, 'export.'+key, feature.get(key))
        for key, value in (feature.get('metadata_file_bytes') or {}).items(): add(costs, 'export.metadata_file_bytes.'+key, value)
        for key, value in feature['timing'].items(): add(costs, 'export.timing.'+key, value)
        seen = set(); splits = set(); k1 = 0; factorizations = 0
        full = jsonlines(lane/'probe/fit_trace.jsonl'); compact = jsonlines(lane/'probe/compact.jsonl')
        for record, small in itertools.zip_longest(full, compact):
            check(record is not None and small is not None, 'Trace/compact length mismatch')
            cell = tuple(record[k] for k in COORDS)
            check(cell in expected and cell not in seen and record['split_id'] not in splits, 'Missing/duplicate/unexpected matrix cell')
            seen.add(cell); splits.add(record['split_id'])
            check(record['scope'] == record['claim_scope'] == SCOPE and record['config'] == FROZEN_CONFIG, 'Trace scientific contract mismatch')
            check(set(record['classes']) == set(record['registered_classes']) and record['old_classes'] == sorted(feature['classes']), 'Trace class binding mismatch')
            verify_oof(record)
            check(all(small[k] == record[k] for k in COORDS+('split_id', 'support_count', 'fold_count', 'factorization_count', 'optimizer_steps', 'persistent_state_bytes')),
                  'Trace/compact metadata mismatch')
            check(small['classes'] == len(record['classes']) and small['completed'] == len(seen)
                  and small['total'] == len(expected), 'Compact completion mismatch')
            zero_fields(small, ('query_rows_used', 'source_rows_used'))
            for key in ('numerical', 'oof', 'reconstruction', 'paired'):
                check(small[key] == scalar_tree(record[key]), 'Trace/compact diagnostic mismatch '+key)
            for key in ('fit_seconds', 'fit_call_seconds'): close(small[key], record[key], 'Compact fit timing mismatch')
            for key in ('fit_seconds', 'fit_call_seconds', 'log_write_seconds', 'total_seconds', 'peak_process_rss_bytes'):
                value = small[key]
                check(value is None or value >= 0, 'Negative compact resource')
                add(costs, 'probe.'+key, value)
            scope = 'old_only' if record['new_count'] == 0 else 'new_present'
            coords = dict(record, cohort=row['cohort'], model_seed=row['seeds']['model'])
            for kind, metric, value in episode_metrics(record):
                add(overall, (scope, kind, metric), value)
                for name, dimensions in STRATA.items():
                    add(groups[name], (scope, *(coords[d] for d in dimensions), kind, metric), value)
            for fold in record['folds']:
                for stage in fold['stages']:
                    for key, value in leaves(stage): add(stage_stats, (stage['arm'], key), value)
            counts['episodes'] += 1
            counts['k1_episodes'] += record['k'] == 1
            counts['oof_episodes'] += record['k'] > 1
            counts['factorizations'] += record['factorization_count']
            counts['physical_oof_records_per_arm'] += record['support_count'] if record['k'] > 1 else 0
            k1 += record['k'] == 1
            factorizations += record['factorization_count']
        check(seen == expected and len(seen) == marker['episodes'] == state[rid]['episodes'], 'Lane matrix incomplete')
        check(marker['k1_episodes'] == k1 and marker['oof_episodes'] == len(seen)-k1
              and marker['factorization_count'] == factorizations, 'Lane aggregate marker mismatch')
        lanes.append(dict(row_id=rid, cohort=row['cohort'], model_seed=row['seeds']['model'],
            episodes=len(seen), k1_episodes=k1, factorizations=factorizations, checkpoint_sha256=sha,
            capsule_id=co['capsule_id'], feature_count=count, export_peak_rss_bytes=feature.get('peak_process_rss_bytes'),
            probe_peak_rss_bytes=marker.get('peak_process_rss_bytes')))
    check(counts['episodes'] == expected_total, 'Global episode count mismatch')
    if expected_total == 4800:
        check((counts['k1_episodes'], counts['oof_episodes'], counts['factorizations']) == (1200, 3600, 64800), 'Full benchmark coverage mismatch')
    out.mkdir(parents=True, exist_ok=False)
    for name, statistics in groups.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            fields = ['population', *STRATA[name], 'kind', 'metric', 'count', 'null_count', 'sum', 'mean', 'min', 'max']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for key, stat in sorted(statistics.items()):
                writer.writerow(dict(zip(fields[:len(key)], key), **stat.result()))
    stage_records = [dict(arm=key[0], metric=key[1], **stat.result()) for key, stat in sorted(stage_stats.items())]
    with (out/'fit_stage_statistics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'metric', 'count', 'null_count', 'sum', 'mean', 'min', 'max'])
        writer.writeheader(); writer.writerows(stage_records)
    summary = dict(status='COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED', scope=SCOPE, run_id=spec.get('run_id'),
        run_root=str(root), release_commit=complete['commit'], run_wall_seconds=complete['finished']-launch['started'], coverage=counts, lanes=lanes,
        statistics=[dict(population=key[0], kind=key[1], metric=key[2], **stat.result()) for key, stat in sorted(overall.items())],
        resources={key: stat.result() for key, stat in sorted(costs.items())},
        fit_stage_statistics=stage_records, unique_model_package_bytes=sum(models.values()),
        unique_model_count=len(models), new_source_payload_bytes=0, query_rows_used=0, source_rows_used=0,
        persistent_classifier_bytes=0, optimizer_steps=0, selected_arm=None, automatic_promotion=False,
        interpretation=[
            'Means are descriptive equal-episode aggregates; overlapping support draws and model/cohort reuse are correlated, not independent samples.',
            'No confidence intervals or p-values treat support draws as independent; no query performance claim.',
            'Old-only and new-present populations are separate; undefined new/H metrics remain null.',
            'Auxiliary classification increments can reflect changed regularization; duplicate controls and held reconstruction do not prove causal TX information.',
            'Numerical rank is arithmetic structure, not held prediction; K1 has no independent holdout.',
            'Completion/access/immutability checks audit recorded metadata, not a second data validation or independent execution replay.',
            'K1 trace omits physical IDs: physical binding was checked by the evaluator; this summary checks counts/nulls and producer bindings.',
            'Process-time sums are work totals, not wall-clock elapsed; RSS maxima are per-process high-water marks, not concurrent aggregate memory.',
            'Model package bytes are existing full checkpoint storage, not inferred new transmission; deployment status remains unknown.',
            'Raw trace/cache artifacts were preserved; cache/IQ arrays and query/source/score/index files were not read.',
        ])
    write_json(out/'summary.json', summary)
    lines = ['# Frozen branch support-only diagnostic', '',
             f"Verified {counts['episodes']} episodes across {len(lanes)} lanes: {counts['k1_episodes']} K1 numerical-only, "
             f"{counts['oof_episodes']} physical OOF, {counts['factorizations']} Cholesky factorizations.", '',
             'This is support evidence only. No arm is selected and no query generalization claim is made.', '',
             '## Six fixed arms and paired changes', '',
             'Values below are equal-episode descriptive means; new-present and old-only populations are separate. Accuracy deltas are fractions (multiply by 100 for percentage points).', '',
             '| Population | Metric | N | Mean |', '|---|---|---:|---:|']
    for record in summary['statistics']:
        if record['kind'] in ('arm', 'paired'):
            mean = 'N/A' if record['mean'] is None else f"{record['mean']:.8g}"
            lines.append(f"| {record['population']} | {record['metric']} | {record['count']} | {mean} |")
    lines += ['', '## Interpretation and costs', '']+[f'- {line}' for line in summary['interpretation']]
    lines += ['', 'All measured extraction, fit, score, logging, cache/model byte and RSS statistics are in summary.json; fold loss/gradient minima, means and maxima are also in fit_stage_statistics.csv.',
              'Stratified CSVs cover K, model seed, support seed, cohort, receiver×scenario and K×new-count. Missing conditional metrics remain empty; no row was selected by its diagnostic outcome.', '']
    with (out/'report.md').open('x', encoding='utf-8') as stream: stream.write('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--run-root')
    parser.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    summary = summarize(**args)
    print(json.dumps(dict(status=summary['status'], coverage=summary['coverage']), allow_nan=False))


if __name__ == '__main__': main()
