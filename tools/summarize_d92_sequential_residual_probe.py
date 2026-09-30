"""Verify complete fixed 160-parent support pilot before opening score traces."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_sequential_residual_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, PROBE_CONFIG, PATHS, METRICS, COUNTERS,
    check, read, scalars, split_identity, selected_tasks, compact_record,
    assess_paths, pooled_assess, parent_mean, baseline_metrics,
)
from run_d92_sequential_residual_probe import validate_spec, verify_marker
from summarize_d92_registration_diagnostic import verify_path as verify_base_path, _statistics, write_json
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree

SUMMARY_STATUS = 'COMPLETE_SEQUENTIAL_RESIDUAL_PROBE_VERIFIED'


def verify_path(entry, *, train, held, labels, classes, old, k, scope, index):
    check(set(entry['paths']) == set(PATHS), 'Three fixed paths required')
    common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
    evidence = {name: dict(common, b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
    expected = assess_paths(evidence, old)
    for name in PATHS:
        check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **expected[name]),
              'Paired path diagnostic mismatch')
    base = dict(entry, **entry['paths']['R0'])
    base['metrics'] = baseline_metrics(base['diagnostic'])
    heads, factors = verify_base_path(base, train=train, held=held, labels=labels, classes=classes,
                                     old=old, k=k, scope=scope, index=index)
    check(entry['paths']['R_reset']['b_scores'] == entry['paths']['R_seq']['b_scores'], 'Residual paths must share the same B scores')
    reuse = classes == old
    check(entry['c_reuses_b_residual'] is reuse, 'Residual N0 reuse mismatch')
    if reuse:
        for name in PATHS:
            check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'N0 must reuse B scores exactly')
            check(entry['paths'][name]['metrics']['total_old_accuracy_drop'] == 0., 'N0 drop must be exactly zero')
    stages = entry['residual_stages']
    check([stage['state'] for stage in stages] == (['B'] if reuse else ['B', 'C_reset', 'C_seq']), 'Residual sharing/stage coverage mismatch')
    for stage in stages:
        is_b = stage['state'] == 'B'
        tids = entry['b_training_ids'] if is_b else entry['c_training_ids']
        registry = old if is_b else classes
        check(stage['training_physical_ids'] == tids and stage['train_physical_count'] == len(tids)
              and stage['held_physical_count'] == len(entry['b_ids'] if is_b else entry['c_ids'])
              and stage['class_count'] == len(registry), 'Residual physical training/held mismatch')
        check(all(stage[key] == entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')),
              'Residual stage coordinates mismatch')
        check(stage['inheritance_source'] == ('B' if stage['state'] == 'C_seq' else None)
              and stage['optimizer_steps'] == 64 and stage['all_states_estimated_from_trainfold_only'] is True,
              'Residual inheritance/update budget mismatch')
        check(stage['inherited'] is (stage['state'] == 'C_seq') and stage['optimizer_state_inherited'] is False
              and stage['held_used_for_training'] is False and stage['final_state_timing'] == 'after_64_updates'
              and stage['config'] == PROBE_CONFIG
              and stage['entry_old_feature_equality_checked'] is (stage['state'] == 'C_seq'),
              'Residual optimizer/final-state contract mismatch')
        check(stage['source_validation'] is None and bool(stage['source_validation_reason']), 'Invented source validation')
        check(stage['fit_seconds'] >= 0 and stage['score_seconds'] >= 0
              and stage['persistent_state_bytes'] >= 0, 'Invalid residual measured cost')
    if not reuse:
        check(stages[1]['q'] == stages[2]['q'], 'C reset/seq must share the same train-only score scale')
    return heads, factors, len(stages), 64*len(stages)


def verify_record(record, split, old):
    finite_tree(record)
    check(all(record[key] == value for key, value in split_identity(split, old).items()), 'Parent identity mismatch')
    classes, old = sorted(split['registered_classes']), sorted(old)
    labels = {pid: split['registered_classes'][label] for pid, label in zip(split['support_ids'], split['support_labels'])}
    k, n = split['k'], len(labels)
    check(record['classes'] == classes and record['old_classes'] == old and record['support_count'] == n
          and record['old_class_count'] == len(old) and record['new_class_count'] == len(classes)-len(old), 'Parent registry/count mismatch')
    check(record['scope'] == SCOPE and record['query_rows_used'] == record['source_rows_used'] == 0, 'Forbidden parent access')
    if k == 1:
        check(record['fold_count'] == 0 and record['folds'] == [] and record['physical_fold_assignment'] == []
              and record['oof'] is None and record['oneshot_proxy'] is None
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT'
              and all(record[key] == 0 for key in COUNTERS[4:]) and record['persistent_state_bytes'] == 0,
              'True K1 fabricated fit/holdout')
        return [], dict(oof=None, proxy=None)
    folds = min(k, 3)
    by_class = {cls: sorted(pid for pid, label in labels.items() if label == cls) for cls in classes}
    check(all(len(ids) == k for ids in by_class.values()), 'Unequal physical K')
    assignments = {pid: i % folds for ids in by_class.values() for i, pid in enumerate(ids)}
    physical = record['physical_fold_assignment']
    check(len(physical) == n and {r['physical_id']: (r['class_id'], r['fold']) for r in physical}
          == {pid: (labels[pid], assignments[pid]) for pid in labels}, 'Physical fold assignment mismatch')
    check(record['fold_count'] == len(record['folds']) == folds
          and [e['fold'] for e in record['folds']] == list(range(folds)), 'OOF path coverage mismatch')
    proxy = record['oneshot_proxy']
    check(proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
          and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent'
          and [e['trial'] for e in proxy['trials']] == list(range(k)), 'Incomplete all-anchor proxy')
    counts = [0, 0, 0, 0]; streams = []
    for scope, entries in (('support_oof', record['folds']), ('support_oneshot_proxy', proxy['trials'])):
        for index, entry in enumerate(entries):
            if scope == 'support_oof':
                held = {pid for pid in labels if assignments[pid] == index}; train = set(labels)-held
            else:
                train = {ids[index] for ids in by_class.values()}; held = set(labels)-train
            counts = [left+right for left, right in zip(counts,
                verify_path(entry, train=train, held=held, labels=labels, classes=classes, old=old, k=k, scope=scope, index=index))]
            streams.extend(dict(event='BASE_FIT', **stage) for stage in entry['stages'])
            streams.extend(dict(event='RESIDUAL_FIT', **stage) for stage in entry['residual_stages'])
    check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old),
          aggregation='one_record_per_physical_held_id'), 'OOF must pool physical records once')
    check(proxy['parent_mean_metrics'] == parent_mean(proxy['trials']), 'Proxy parent-first aggregation mismatch')
    check([record[key] for key in ('head_fit_count', 'factorization_count', 'residual_training_stages', 'optimizer_steps')] == counts
          and record['sequence_paths'] == folds+k and record['heldout_unavailable_reason'] is None, 'Actual parent costs mismatch')
    check(record['persistent_state_bytes'] == max(stage['persistent_state_bytes'] for stage in streams if stage['event'] == 'RESIDUAL_FIT'),
          'Resident residual state accounting mismatch')
    return streams, dict(oof={name: record['oof']['paths'][name]['metrics'] for name in PATHS}, proxy=proxy['parent_mean_metrics'])


def accumulate_resources(resources, record, stages):
    check(record['fit_seconds'] >= 0, 'Negative parent cost')
    resources['parent_fit_seconds_sum'] = resources.get('parent_fit_seconds_sum', 0.)+record['fit_seconds']
    for stage in stages:
        group = 'base' if stage['event'] == 'BASE_FIT' else 'residual'
        resources[group+'_fit_stage_count'] = resources.get(group+'_fit_stage_count', 0)+1
        for key, value in stage.items():
            if key.endswith('_seconds') and key != 'prepare_seconds' and value is not None:
                check(value >= 0, 'Negative measured duration')
                target = group+'_'+key+'_sum'
                resources[target] = resources.get(target, 0.)+value
        if group == 'residual':
            for key in ('persistent_state_bytes', 'trainable_parameter_count'):
                if key in stage:
                    resources['maximum_'+key] = max(resources.get('maximum_'+key, 0), stage[key])


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def verify_step_stream(lane, expected_stages):
    full = iter(jsonlines(lane/'training_steps.jsonl')); compact = iter(jsonlines(lane/'training_steps_compact.jsonl'))
    count = 0
    for sid, stage in expected_stages:
        for update in range(1, 65):
            row, small = next(full, None), next(compact, None)
            check(row is not None and small is not None, 'Missing training step log')
            finite_tree(row)
            check(scalars(row) == small and row['split_id'] == sid
                  and all(row[key] == stage[key] for key in ('state', 'scope', 'fold', 'trial', 'parent_k', 'train_k',
                      'inheritance_source', 'training_physical_ids', 'train_physical_count', 'held_physical_count', 'class_count')),
                  'Training step compact/context mismatch')
            check(row['step'] == row['optimizer_steps'] == update
                  and row['optimizer_state_inherited'] is False
                  and row['state_timing'] == 'loss_gradient_pre_update_parameters_post_update', 'Training update sequence mismatch')
            count += 1
    check(next(full, None) is None and next(compact, None) is None, 'Extra training step log')
    return count


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec
    validate_spec(spec)
    root = Path(run_root or spec['execution']['remote_run_root']); out = Path(output)
    if out.exists(): raise FileExistsError(out)
    launch, complete, state = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    check(launch['spec'] == spec and complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == 4
          and complete['episodes'] == 160 and complete['commit'] == launch['commit'], 'Full four-row pilot incomplete')
    check(set(state) == {r['row_id'] for r in spec['rows']} and all(v['status'] == STATUS for v in state.values()), 'Incomplete row state')
    check(all(v.get('query_access') is False and v.get('source_sample_access') is False for v in (launch, complete)), 'Forbidden run access')
    lanes = []
    # All four terminal markers are verified before opening ANY support score trace.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root/row['row_id']/'probe'
        marker = verify_marker(lane/'probe_complete.json', spec, row); startup = read(lane/'startup.json')
        check_bind(startup, row, co)
        check(startup['config'] == dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
              and startup['scope'] == marker['scope'] == SCOPE and startup['episodes'] == 40
              and startup['producer_episodes'] == co['expected_split_count'], 'Startup/config mismatch')
        for value in (startup, marker):
            check(all(value.get('channel', {}).get(key) == expected for key, expected in CHANNEL.items())
                  and value['scenarios'] == SCENARIOS, 'Practical residual channel mismatch')
            check(value['query_rows_used'] == value['source_rows_used'] == 0 and value['truth_read'] is False, 'Forbidden lane access')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
              and startup['actual_A'] is None and startup['adapter_training'] is True
              and startup['adapted_state_inherited'] is True, 'Adaptation/source state mismatch')
        feature_root = Path(startup['support_features'])
        check(feature_root == Path(row['support_features']), 'Unexpected support cache reference')
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
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    for row, lane, marker, expected, old, startup in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stage_rows = iter(jsonlines(lane/'fit_stages.jsonl')); step_stages = []
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact count mismatch')
            sid = record['split_id']; check(sid in expected and sid not in seen, 'Unexpected/duplicate parent')
            seen.add(sid); stages, measurements = verify_record(record, expected[sid], old)
            accumulate_resources(resources, record, stages)
            check(small == compact_record(record), 'Compact evidence mismatch')
            for stage in stages:
                check(next(stage_rows, None) == dict(scalars(stage), split_id=sid), 'Stage stream mismatch')
                if stage['event'] == 'RESIDUAL_FIT': step_stages.append((sid, stage))
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1)
            counts['oof_episodes'] += int(record['k'] > 1); counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for key in COUNTERS[4:]: counts[key] += record[key]
            for diagnostic, paths in measurements.items():
                if paths is None:
                    # Preserve the complete K axis: true K1 is explicitly unmeasured.
                    paths = {name: dict.fromkeys(METRICS) for name in PATHS}
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                for name, metrics in paths.items():
                    _add(strata['overall'], (diagnostic, name, population), metrics)
                    _add(strata['by_k_new_count'], (diagnostic, name, record['k'], record['new_count']), metrics)
                    _add(strata['by_receiver_scene'], (diagnostic, name, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), metrics)
                    _add(strata['by_model_cohort'], (diagnostic, name, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), metrics)
        check(seen == set(expected) and next(stage_rows, None) is None, 'Missing parents or extra stages')
        check(verify_step_stream(lane, step_stages) == counts['optimizer_steps'], 'Step log accounting mismatch')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row completion totals mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
    check(all(coverage[key] == complete[key] for key in COUNTERS), 'Run completion totals mismatch')
    fixed = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
        sequence_paths=1760, head_fit_count=3168, residual_training_stages=4576, optimizer_steps=292864)
    check(all(coverage[key] == value for key, value in fixed.items()), 'Fixed pilot budget mismatch')
    check(resources['base_fit_stage_count'] == coverage['head_fit_count']
          and resources['residual_fit_stage_count'] == coverage['residual_training_stages'], 'Measured resource coverage mismatch')
    check(complete['finished'] >= launch['started'], 'Invalid wall-clock bounds')
    resources.update(run_wall_seconds=complete['finished']-launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _ in lanes
            if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        unmeasured_reason='CPU only; no deployment package or transfer measured; RSS null if runtime unavailable',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment'])
                  for row, _, _, _, _, startup in lanes],
        interpretation='Durations summed over fits are work totals, not parallel elapsed wall time. Numeric residual state excludes base, cache, optimizer and audit copies.')
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    summary = dict(status=SUMMARY_STATUS, scope=SCOPE, run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=coverage, resources=resources, algorithm=PROBE_CONFIG, channel=CHANNEL, statistics=tables,
        old_class_count=6, actual_A=None, adaptation_gain_B_minus_A=None,
        actual_A_unavailable_reason='No legal ground A scores on the same held physical samples',
        automatic_promotion=False, performance_gate=None, query_rows_used=0, source_rows_used=0,
        interpretation=[
            'R0 is original LocalRidge; R_reset and R_seq share one B residual and immutable base fits.',
            'R_seq versus R_reset compares both inherited initialization and proximal anchor, not only warm-start.',
            'C_old is an offline restriction of fixed C scores; deployed classification always uses all registered classes.',
            'OOF pools each held physical ID exactly once; proxy averages all anchors within parent then weights parents equally.',
            'True K1 has no training or independent holdout; all corresponding accuracy fields are N/A.',
            'B minus B0 is support-classifier adaptation increment and cannot replace B minus ground A.',
            'Six B/C_old/C correctness transitions and all fixed K/new-count/model/receiver/scene strata remain in tables.',
            'No source validation, query fitting, query feedback, early stopping or best-state selection is used.',
            '10/1/3 percentage-point ideal directions are descriptive; reduced old/new gap can result from degrading old accuracy.',
            'This pilot supplies support holdout evidence only, not query generalization or unmeasured scene evidence.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out/'summary.json', summary)
    for name, rows in tables.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 顺序残差分类头 support pilot', '',
        '完整 160 parent、4 row、三路径均已核验。旧类数固定 6；A 与 B−A 为 N/A。B−B0 仅是 support 分类器适应增量。', '',
        '| 诊断 | 路径 | K | 新类数 | A旧 | B0旧 | B旧 | C旧类列 | C旧 | C新 | H | B−B0 | B−C旧类列 | 新竞争损失 | 注册总下降 | 新旧绝对差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = {}
    for row in tables['by_k_new_count']:
        cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    display = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy', 'C_old_accuracy',
        'C_new_accuracy', 'C_h', 'support_adaptation_B_minus_B0', 'old_order_change', 'new_competition_loss',
        'total_old_accuracy_drop', 'C_abs_new_old_gap')
    for key, values in sorted(cells.items()):
        lines.append('| '+' | '.join([str(value) for value in key]+['N/A' if values[m] is None else f'{100*values[m]:.3f}' for m in display])+' |')
    lines += ['', '准确率为百分数，差值为百分点。相对 R0 的两类变化、六种正确性转换及全部分层见 CSV。', '',
        f"实际 base fits={coverage['head_fit_count']}，residual stages={coverage['residual_training_stages']}，Adam updates={coverage['optimizer_steps']}。",
        f"运行墙钟 {resources['run_wall_seconds']:.3f} s；各阶段工作量之和不等于并发墙钟。资源分项、硬件与未测项见 summary.json。", '']
    lines += ['- '+text for text in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
