"""Recompute paired support evidence only after the entire fixed pilot completes."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_within_class_metric_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, PROBE_CONFIG, PATHS, METRICS, COUNTERS,
    check, read, scalars, csv_record, split_identity, selected_tasks, compact_record,
    assess_paths, pooled_assess, parent_mean, baseline_metrics,
)
from run_d92_within_class_metric_probe import validate_spec, verify_marker
import evaluate_d92_registration_diagnostic as baseline
from summarize_d92_registration_diagnostic import verify_record as verify_baseline, _statistics, write_json
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree

SUMMARY_STATUS = 'COMPLETE_WITHIN_CLASS_METRIC_PROBE_VERIFIED'


def verify_loco(diagnostic, entry, labels, old):
    """Verify every excluded class uses only this outer training fold."""
    check(diagnostic['status'] == 'COMPLETE', 'LOCO completion status mismatch')
    old_train = set(entry['b_training_ids']); seen = set(); factors = 0; fit_seconds = score_seconds = 0.
    for fold in diagnostic['folds']:
        held_class = fold['held_class']
        check(held_class in old and held_class not in seen, 'Duplicate/unregistered LOCO class'); seen.add(held_class)
        held = {pid for pid in old_train if labels[pid] == held_class}; train = old_train-held
        check(len(fold['held_physical_ids']) == len(held) and set(fold['held_physical_ids']) == held
            and len(fold['training_physical_ids']) == len(train) and set(fold['training_physical_ids']) == train
            and not (train | held).intersection(entry['c_ids']), 'LOCO must exclude assessed class and outer held rows')
        fitted = fold['metric_audit']
        check(fitted['training_physical_ids'] == fold['training_physical_ids']
            and sorted(fitted['classes']) == sorted(set(old)-{held_class})
            and fitted['train_k'] == entry['train_k'] and fitted['train_physical_count'] == len(train)
            and fitted['metric_fit_count'] == 1 and fitted['optimizer_steps'] == 0, 'LOCO fit identity mismatch')
        tolerance = fitted['numerical_tolerance']
        for trace, fraction, reason in (
            (fold['held_class_within_trace'], fold['held_class_contraction_fraction'], fold['held_class_unavailable_reason']),
            (fold['training']['within_trace'], fold['training']['contraction_fraction'], fold['training']['unavailable_reason'])):
            check(trace >= 0, 'Negative within-class diagnostic energy')
            if trace == 0:
                check(fraction is None and reason == 'EXACT_ZERO_WITHIN_CLASS_VARIATION', 'Zero-energy diagnostic must be unavailable')
            else:
                check(fraction is not None and -tolerance <= fraction <= .5+tolerance and reason is None,
                    'Invalid within-class contraction fraction')
        factors += fitted['metric_factorization_count']; fit_seconds += fitted['fit_seconds']; score_seconds += fold['diagnostic_score_seconds']
    check(seen == set(old) and factors == diagnostic['diagnostic_factorization_count']
        and fit_seconds == diagnostic['diagnostic_fit_seconds'] and score_seconds == diagnostic['diagnostic_score_seconds'],
        'LOCO class coverage/actual resource mismatch')


def baseline_view(record):
    """Construct only the original R0 evidence for the unchanged verifier."""
    def entry_view(entry):
        value = entry['paths']['R0']
        result = dict(entry, **value)
        result['metrics'] = baseline_metrics(value['diagnostic'])
        return result
    result = dict(record, scope=baseline.SCOPE, persistent_state_bytes=0,
        head_fit_count=record['baseline_head_fit_count'],
        factorization_count=sum(stage['factorization_calls'] for entry in record['folds']+
            ([] if record['oneshot_proxy'] is None else record['oneshot_proxy']['trials']) for stage in entry['stages']))
    result['folds'] = [entry_view(entry) for entry in record['folds']]
    if record['oof'] is not None:
        diagnostic = record['oof']['paths']['R0']['diagnostic']
        result['oof'] = dict(diagnostic=diagnostic, metrics=baseline_metrics(diagnostic), aggregation='one_record_per_physical_held_id')
        trials = [entry_view(entry) for entry in record['oneshot_proxy']['trials']]
        result['oneshot_proxy'] = dict(record['oneshot_proxy'], trials=trials, parent_mean_metrics=baseline.parent_mean(trials))
    return result


def verify_record(record, split, old):
    finite_tree(record)
    check(record['scope'] == SCOPE and record['optimizer_steps'] == record['query_rows_used'] == record['source_rows_used'] == 0,
          'Forbidden parent state/access')
    verify_baseline(baseline_view(record), split, old)
    if split['k'] == 1:
        check(all(record[key] == 0 for key in COUNTERS[4:]) and record['persistent_state_bytes'] == 0, 'True K1 fabricated fitting')
        return [], dict(oof=None, proxy=None), []
    counts = dict.fromkeys(COUNTERS[4:], 0); logs = []; train_diagnostics = []
    old = sorted(old); classes = sorted(split['registered_classes'])
    labels = {pid: split['registered_classes'][label] for pid, label in zip(split['support_ids'], split['support_labels'])}
    entries = record['folds']+record['oneshot_proxy']['trials']; maximum_state = 0
    for entry in entries:
        check(set(entry['paths']) == set(PATHS), 'Both fixed paths required')
        common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
        evidence = {name: dict(common, b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
        fresh = assess_paths(evidence, old)
        for name in PATHS:
            check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]),
                  'Paired metric path result mismatch')
        metric = entry['metric_fit']; identity = metric['identity']; diagnostic = entry['train_only_diagnostic']
        check(type(identity) is bool and metric['metric_fit_count'] == 1 and metric['optimizer_steps'] == 0
            and metric['training_physical_ids'] == entry['b_training_ids'] and sorted(metric['classes']) == old
            and metric['train_physical_count'] == len(entry['b_training_ids']) and metric['train_k'] == entry['train_k']
            and metric['diagnostic_scope'] == 'OLD_TRAIN_SUPPORT_ONLY', 'Metric training binding mismatch')
        check(all(metric[key] == entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')), 'Metric coordinates mismatch')
        check(metric['fit_seconds'] >= 0 and metric['persistent_state_bytes'] >= 0
            and metric['metric_factorization_count'] >= 0, 'Invalid metric resource evidence')
        check(entry['c_reuses_b_metric'] is (classes == old), 'N0 metric reuse flag mismatch')
        if identity:
            check(entry['metric_stages'] == [] and entry['paths']['R_metric'] == entry['paths']['R0'], 'Identity must reuse exact R0 evidence')
        else:
            check([stage['state'] for stage in entry['metric_stages']] == (['B'] if classes == old else ['B', 'C']), 'Metric head stage coverage mismatch')
        if classes == old:
            for name in PATHS:
                check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores']
                    and entry['paths'][name]['metrics']['total_old_accuracy_drop'] == 0., 'N0 must reuse B exactly')
        if entry['train_k'] == 1:
            check(identity and metric['identity_reason'] == 'TRAIN_K1_NO_WITHIN_CLASS_INFORMATION'
                and diagnostic is None, 'Proxy K1 must be identity without fabricated within-class diagnostics')
        else:
            check(entry['scope'] == 'support_oof', 'Only OOF may carry class-internal diagnostics')
            check(diagnostic is not None and diagnostic['diagnostic_fit_count'] == len(old)
                and len(diagnostic['folds']) == len(old) and sorted(diagnostic['classes']) == old
                and diagnostic['training_physical_ids'] == entry['b_training_ids']
                and diagnostic['optimizer_steps'] == 0 and diagnostic['diagnostic_scope'] == 'OLD_TRAIN_SUPPORT_ONLY_NOT_OUTER_HELD',
                'Complete train-only LOCO coverage required')
            check(diagnostic['diagnostic_fit_seconds'] >= 0 and diagnostic['diagnostic_score_seconds'] >= 0, 'Invalid LOCO measured cost')
            verify_loco(diagnostic, entry, labels, old)
            counts['diagnostic_fit_count'] += diagnostic['diagnostic_fit_count']
            counts['diagnostic_factorization_count'] += diagnostic['diagnostic_factorization_count']
        counts['sequence_paths'] += 1; counts['metric_fit_count'] += 1; counts['metric_nonidentity_count'] += int(not identity)
        counts['metric_factorization_count'] += metric['metric_factorization_count']
        for stage in entry['stages']:
            counts['baseline_head_fit_count'] += 1; counts['head_fit_count'] += 1
            counts['factorization_count'] += stage['factorization_calls']; logs.append(dict(event='BASE_FIT', **stage))
        logs.append(dict(event='METRIC_FIT', **metric))
        if diagnostic is not None: logs.append(dict(event='TRAIN_ONLY_LOCO', **diagnostic))
        for stage in entry['metric_stages']:
            is_b = stage['state'] == 'B'; final = stage['final_fit']
            check(stage['metric_head_fit_count'] == 1 and stage['metric_identity_reuse'] is False
                and stage['optimizer_steps'] == 0 and stage['metric_inherited_from'] == 'B_OLD_TRAIN_SUPPORT', 'Metric inheritance/fit mismatch')
            check(stage['training_physical_ids'] == entry['b_training_ids' if is_b else 'c_training_ids']
                and stage['held_physical_count'] == len(entry['b_ids' if is_b else 'c_ids'])
                and stage['train_physical_count'] == len(stage['training_physical_ids'])
                and stage['class_count'] == len(old if is_b else classes), 'Metric head physical binding mismatch')
            reference = entry['stages'][0 if is_b else 1]
            check(final['interaction_centered_trace'] == reference['interaction_centered_trace']
                and final['ridge_coefficient'] == 1. and final['sample_weight'] == 1.
                and final['physical_loss_mass'] == stage['train_physical_count'], 'Original trace/physical ridge contract mismatch')
            check(0 <= final['normal_equation_residual'] <= final['numerical_tolerance']
                and 0 <= final['trace_relative_error'] <= final['numerical_tolerance'], 'Uncertified metric head solve')
            check(stage['head_factorization_count'] == final['factorization_calls']
                and stage['shared_metric_state_bytes'] == metric['persistent_state_bytes']
                and stage['score_seconds'] >= 0 and stage['head_state_bytes'] >= 0, 'Metric head actual cost mismatch')
            counts['metric_head_fit_count'] += stage['metric_head_fit_count']; counts['head_fit_count'] += stage['metric_head_fit_count']
            counts['factorization_count'] += stage['head_factorization_count']; logs.append(dict(event='METRIC_HEAD_FIT', **stage))
        expected_state = entry['stages'][-1]['persistent_state_bytes'] if identity else entry['metric_stages'][-1]['persistent_state_bytes']
        check(entry['deployment_C_state_bytes'] == expected_state, 'Shared deployable state accounting mismatch')
        maximum_state = max(maximum_state, expected_state)
        train_diagnostics.append(dict(split_id=record['split_id'], scope=entry['scope'], fold=entry['fold'], trial=entry['trial'],
            k=record['k'], train_k=entry['train_k'], new_count=record['new_count'], metric=metric, loco=diagnostic,
            evidence_scope='TRAIN_ONLY_NOT_OUTER_HELD_ACCURACY'))
    check(all(record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == maximum_state,
        'Actual parent fitting/state totals mismatch')
    check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'),
        'OOF must pool physical held rows')
    check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy must average anchors within parent')
    return logs, dict(oof={name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=record['oneshot_proxy']['parent_mean_metrics']), train_diagnostics


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.)+record['fit_seconds']
    for stage in logs:
        group = stage['event'].lower()
        for key, value in stage.items():
            if key.endswith('_seconds') and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group+'_'+key+'_sum'
                resources[name] = resources.get(name, 0.)+value
        if stage['event'] == 'METRIC_HEAD_FIT':
            name = 'metric_head_actual_fit_seconds_sum'
            resources[name] = resources.get(name, 0.)+stage['final_fit']['fit_seconds']
        for key in ('persistent_state_bytes', 'shared_metric_state_bytes', 'head_state_bytes', 'direct_difference_pair_count'):
            if key in stage and stage[key] is not None:
                name = group+'_maximum_'+key
                resources[name] = max(resources.get(name, 0), stage[key])


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
    # Check every row's completion and source binding before opening any score trace.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root/row['row_id']/'probe'
        marker = verify_marker(lane/'probe_complete.json', spec, row); startup = read(lane/'startup.json'); check_bind(startup, row, co)
        check(startup['config'] == dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
            and startup['scope'] == marker['scope'] == SCOPE and startup['episodes'] == 40
            and startup['producer_episodes'] == co['expected_split_count'], 'Startup/config mismatch')
        for value in (startup, marker):
            check(all(value.get('channel', {}).get(key) == expected for key, expected in CHANNEL.items())
                and value['scenarios'] == SCENARIOS and value['query_rows_used'] == value['source_rows_used'] == value['optimizer_steps'] == 0
                and value['truth_read'] is False, 'Channel or forbidden access mismatch')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
            and startup['actual_A'] is None and startup['adapted_state_inherited'] is True, 'Invented or forbidden adaptation state')
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
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; diagnostics = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    for row, lane, marker, expected, old, startup in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stream = iter(jsonlines(lane/'fit_stages.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact length mismatch'); sid = record['split_id']
            check(sid in expected and sid not in seen, 'Unexpected/duplicate parent'); seen.add(sid)
            logs, measurements, train = verify_record(record, expected[sid], old); accumulate_resources(resources, record, logs)
            check(small == compact_record(record), 'Compact evidence mismatch')
            for item in logs: check(next(stream, None) == dict(scalars(item), split_id=sid), 'Stage stream mismatch')
            diagnostics.extend(dict(row_id=row['row_id'], model_seed=row['seeds']['model'], cohort=row['cohort'],
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
        check(seen == set(expected) and next(stream, None) is None, 'Missing parents or extra stages')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row actual totals mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
    check(all(coverage[key] == complete[key] for key in COUNTERS), 'Run totals mismatch')
    fixed = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
        sequence_paths=1760, baseline_head_fit_count=3168, metric_fit_count=1760, diagnostic_fit_count=2160, optimizer_steps=0)
    check(all(coverage[key] == value for key, value in fixed.items()), 'Fixed pilot coverage mismatch')
    resources.update(run_wall_seconds=complete['finished']-launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _ in lanes
            if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        unmeasured_reason='CPU only; no deployment package/transfer measured; RSS null if runtime unavailable',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment']) for row, _, _, _, _, startup in lanes],
        interpretation='Stage duration sums measure work, not concurrent elapsed time. LOCO is diagnostic-only cost, not deployable metric fitting.')
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    summary = dict(status=SUMMARY_STATUS, scope=SCOPE, run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=coverage, resources=resources, algorithm=PROBE_CONFIG, channel=CHANNEL, statistics=tables,
        old_class_count=6, actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        query_rows_used=0, source_rows_used=0, train_only_diagnostics=diagnostics,
        interpretation=[
            'B0 and C0 are unchanged LocalRidge; B estimates one metric on old training support, and C inherits it without refitting the metric.',
            'The metric changes Gaussian distances/bandwidth; trace matching retains the corresponding original interaction trace.',
            'True K1 has no fit/holdout. Every proxy trainK1 is exact identity and shares R0 classifiers/scores; no K1 benefit is claimed.',
            'A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.',
            'OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.',
            'Spectra and leave-one-old-class-out contraction are train-only mechanism statistics, not held accuracy or proven channel factors.',
            'LOCO fits use only the remaining old training classes; neither outer held support nor query enters fitting or selection.',
            'Old-only reuse and repeated old support across new-count rows are not independent observations.',
            'Nearest-centroid ordering of protected old means is not expected to change; point-kernel decisions can change.',
            '10/1/3 percentage-point ideal directions are descriptive and impose no automatic promotion gate.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out/'summary.json', summary)
    for name, rows in tables.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out/'train_only_diagnostics.jsonl').open('x', encoding='utf-8') as stream:
        for row in diagnostics: stream.write(json.dumps(row, allow_nan=False)+'\n')
    with (out/'train_only_diagnostics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0])); writer.writeheader()
        writer.writerows(csv_record(row) for row in diagnostics)
    lines = ['# WithinClassMetric support 机制 pilot', '',
        '完整160 parent、4 row、R0与R_metric均已核验；旧类固定6个。A与B−A为N/A。', '',
        '| 诊断 | 路径 | K | 新类数 | A旧 | B0旧 | B旧 | C旧类列 | C旧 | C新 | H | B−B0 | B−C旧类列 | 新竞争损失 | 注册下降 | 新旧差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = {}
    for row in tables['by_k_new_count']:
        cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    display = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy', 'C_old_accuracy', 'C_new_accuracy',
        'C_h', 'support_adaptation_B_minus_B0', 'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    for key, values in sorted(cells.items()):
        lines.append('| '+' | '.join([str(value) for value in key]+['N/A' if values[m] is None else f'{100*values[m]:.3f}' for m in display])+' |')
    lines += ['', '准确率为百分数，差值为百分点。全部分层、相对R0变化与六类正确性转换见CSV；类内谱和类留一统计见独立train-only诊断文件。', '',
        f"原分类头拟合{coverage['baseline_head_fit_count']}次，新增度量分类头{coverage['metric_head_fit_count']}次，度量拟合尝试{coverage['metric_fit_count']}次；LOCO诊断拟合另计{coverage['diagnostic_fit_count']}次。",
        f"实测运行墙钟{resources['run_wall_seconds']:.3f} s；分项工作量及内存口径见summary.json。", '']
    lines += ['- '+line for line in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8'); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
