"""Independently verify all 160 support parents before describing registration."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_registration_diagnostic import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, DIAGNOSTIC_CONFIG, FROZEN_CONFIG, METRICS,
    check, read, scalars, split_identity, selected_tasks, compact_record,
    diagnose_evidence, measured_metrics, parent_mean,
)
from run_d92_registration_diagnostic import validate_spec, verify_marker
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close

SUMMARY_STATUS = 'COMPLETE_REGISTRATION_DIAGNOSTIC_VERIFIED'
COUNTERS = ('episodes', 'k1_episodes', 'oof_episodes', 'proxy_anchor_count',
            'sequence_paths', 'head_fit_count', 'factorization_count')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def verify_path(entry, *, train, held, labels, classes, old, k, scope, index):
    oldset = set(old)
    btrain = {pid for pid in train if labels[pid] in oldset}
    bheld = {pid for pid in held if labels[pid] in oldset}
    expected = dict(b_training_ids=btrain, c_training_ids=train, b_ids=bheld, c_ids=held)
    for key, ids in expected.items():
        check(len(entry[key]) == len(ids) and set(entry[key]) == ids, 'Physical path mismatch: '+key)
    check(not train & held and entry['held_labels'] == {pid: labels[pid] for pid in held}, 'Held label/isolation mismatch')
    check(entry['b_classes'] == old and entry['c_classes'] == classes, 'Score class order mismatch')
    check(entry['scope'] == scope and entry['parent_k'] == k
          and entry['fold'] == (index if scope == 'support_oof' else None)
          and entry['trial'] == (index if scope == 'support_oneshot_proxy' else None)
          and entry['train_k'] == len(train)//len(classes) and entry['held_k'] == len(held)//len(classes), 'Path coordinates/count mismatch')
    reuse = classes == old
    check(entry['c_reuses_b0'] is reuse and [s['state'] for s in entry['stages']] == (['B0'] if reuse else ['B0', 'C0']), 'Actual head fit/reuse mismatch')
    if reuse:
        bm = dict(zip(entry['b_ids'], entry['b_scores'])); cm = dict(zip(entry['c_ids'], entry['c_scores']))
        check(bm == cm, 'No-new C0 must reuse B0 scores exactly')
    factors = 0
    for stage in entry['stages']:
        stage_train = btrain if stage['state'] == 'B0' else train
        stage_classes = old if stage['state'] == 'B0' else classes
        stage_held = bheld if stage['state'] == 'B0' else held
        check(stage['scope'] == scope and stage['fold'] == entry['fold'] and stage['trial'] == entry['trial']
              and stage['parent_k'] == k and stage['train_k'] == entry['train_k'], 'Stage coordinate mismatch')
        check(set(stage['training_physical_ids']) == stage_train and len(stage['training_physical_ids']) == len(stage_train)
              and stage['train_physical_count'] == len(stage_train) and stage['held_physical_count'] == len(stage_held)
              and stage['class_count'] == len(stage_classes)
              and stage['all_states_estimated_from_trainfold_only'] is True, 'Stage training/held boundary mismatch')
        check(stage['optimizer_steps'] == 0 and stage['learning_rate'] is None
              and stage['ridge_coefficient'] == 1. and stage['physical_loss_mass'] == len(stage_train)
              and stage['sample_weight'] == 1., 'Frozen fit contract mismatch')
        check(stage['source_validation'] is None and bool(stage['source_validation_reason']), 'Invented source validation')
        expected_factors = int(len(stage_classes) > 1 and stage['interaction_centered_trace'] > 0)
        check(stage['factorization_calls'] == expected_factors and stage['status'] ==
              ('CLOSED_FORM_SOLVED' if expected_factors else 'EXACT_ZERO_CLASSIFIER'), 'Analytical solve accounting mismatch')
        close(stage['loss_total'], stage['loss_data']+stage['loss_ridge'], 'Loss component mismatch')
        check(0 <= stage['normal_equation_residual'] <= stage['numerical_tolerance']
              and 0 <= stage['trace_relative_error'] <= stage['numerical_tolerance'], 'Uncertified numerical result')
        check(all(stage[key] >= 0 for key in ('fit_seconds', 'score_seconds', 'fit_and_score_seconds')), 'Negative measured cost')
        factors += stage['factorization_calls']
    fresh = diagnose_evidence(entry, old)
    check(entry['diagnostic'] == fresh and entry['metrics'] == measured_metrics(fresh), 'Fixed-score diagnostic mismatch')
    return len(entry['stages']), factors


def verify_record(record, split, old):
    finite_tree(record)
    check(all(record[key] == value for key, value in split_identity(split, old).items()), 'Parent identity mismatch')
    classes, old = sorted(split['registered_classes']), sorted(old)
    labels = {pid: split['registered_classes'][label] for pid, label in zip(split['support_ids'], split['support_labels'])}
    k, n = split['k'], len(labels)
    check(record['classes'] == classes and record['old_classes'] == old and record['support_count'] == n
          and record['old_class_count'] == len(old) and record['new_class_count'] == len(classes)-len(old), 'Parent registry/count mismatch')
    check(record['scope'] == SCOPE and record['query_rows_used'] == record['source_rows_used'] == record['optimizer_steps'] == record['persistent_state_bytes'] == 0, 'Forbidden access or persistent state')
    if k == 1:
        check(record['fold_count'] == 0 and record['folds'] == [] and record['physical_fold_assignment'] == []
              and record['oof'] is None and record['oneshot_proxy'] is None
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT'
              and record['head_fit_count'] == record['factorization_count'] == record['sequence_paths'] == 0,
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
    all_stages, heads, factors, brows, crows = [], 0, 0, {}, {}
    for fold, entry in enumerate(record['folds']):
        held = {pid for pid in labels if assignments[pid] == fold}; train = set(labels)-held
        h, f = verify_path(entry, train=train, held=held, labels=labels, classes=classes, old=old, k=k, scope='support_oof', index=fold)
        heads += h; factors += f; all_stages.extend(entry['stages'])
        brows.update(zip(entry['b_ids'], entry['b_scores'])); crows.update(zip(entry['c_ids'], entry['c_scores']))
    evidence = dict(b_ids=sorted(brows), b_classes=old, b_scores=[brows[v] for v in sorted(brows)],
        c_ids=sorted(crows), c_classes=classes, c_scores=[crows[v] for v in sorted(crows)], held_labels=labels)
    fresh = diagnose_evidence(evidence, old)
    check(record['oof'] == dict(diagnostic=fresh, metrics=measured_metrics(fresh), aggregation='one_record_per_physical_held_id'), 'OOF must pool held records, not equally weight folds')
    proxy = record['oneshot_proxy']
    check(proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
          and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent'
          and [e['trial'] for e in proxy['trials']] == list(range(k)), 'Incomplete all-anchor proxy')
    for trial, entry in enumerate(proxy['trials']):
        train = {ids[trial] for ids in by_class.values()}; held = set(labels)-train
        h, f = verify_path(entry, train=train, held=held, labels=labels, classes=classes, old=old, k=k, scope='support_oneshot_proxy', index=trial)
        heads += h; factors += f; all_stages.extend(entry['stages'])
    check(proxy['parent_mean_metrics'] == parent_mean(proxy['trials']), 'Proxy parent-first mean mismatch')
    check(record['head_fit_count'] == heads and record['factorization_count'] == factors
          and record['sequence_paths'] == folds+k and record['heldout_unavailable_reason'] is None, 'Parent fit/path accounting mismatch')
    return all_stages, dict(oof=record['oof']['metrics'], proxy=proxy['parent_mean_metrics'])


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def accumulate_resources(resources, record, stages):
    """Sum measured work, without treating concurrent sums as elapsed wall time."""
    check(record['fit_seconds'] >= 0, 'Negative parent cost')
    resources['parent_fit_seconds_sum'] = resources.get('parent_fit_seconds_sum', 0.)+record['fit_seconds']
    resources['fit_stage_count'] = resources.get('fit_stage_count', 0)+len(stages)
    for key in ('fit_seconds', 'score_seconds', 'fit_and_score_seconds'):
        check(all(stage[key] >= 0 for stage in stages), 'Negative stage cost')
        name = 'stage_'+key+'_sum'
        resources[name] = resources.get(name, 0.)+sum(stage[key] for stage in stages)


def _statistics(groups, dimensions):
    rows = []
    for key, metrics in sorted(groups.items()):
        for metric, values in metrics.items():
            present = [v for v in values if v is not None]
            rows.append(dict(zip(dimensions, key), metric=metric, parent_count=len(values),
                measured_parent_count=len(present), null_parent_count=len(values)-len(present),
                mean=None if not present else sum(present)/len(present),
                minimum=None if not present else min(present), maximum=None if not present else max(present)))
    return rows


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
    # Check EVERY terminal marker and bound metadata before opening any score trace.
    lanes = []
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root/row['row_id']/'probe'
        marker = verify_marker(lane/'probe_complete.json', spec, row); startup = read(lane/'startup.json')
        check_bind(startup, row, co)
        expected_config = dict(algorithm=DIAGNOSTIC_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
        check(startup['config'] == expected_config and startup['scope'] == marker['scope'] == SCOPE
              and startup['episodes'] == 40 and startup['producer_episodes'] == co['expected_split_count'], 'Startup/selection mismatch')
        for value in (startup, marker):
            check(all(value.get('channel', {}).get(key) == expected for key, expected in CHANNEL.items())
                  and value['scenarios'] == SCENARIOS, 'Practical residual channel binding mismatch')
            check(value['query_rows_used'] == value['source_rows_used'] == value['optimizer_steps'] == 0
                  and value['truth_read'] is False, 'Forbidden lane access')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated', 'adapted_state_inherited', 'adapter_training'))
              and startup['actual_A'] is None and startup['adapted_B'] is None, 'Invented adaptation state')
        feature_root = Path(startup['support_features'])
        check(feature_root == Path(row['support_features']), 'Unexpected producer reference')
        feature, plan, provenance = [read(feature_root/name) for name in ('features_complete.json', 'support_splits.json', 'checkpoint_provenance.json')]
        for value in (feature, plan):
            check(value['capsule_id'] == co['capsule_id'] and value['checkpoint_sha256'] == row['expected_checkpoint_sha256'], 'Producer source binding mismatch')
        check_bind(feature, row, co)
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == co['expected_split_count']
              and provenance == startup['provenance'] and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH'
              and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance'] == [], 'Producer provenance mismatch')
        old = feature['classes']; chosen = selected_tasks([(s, None, None) for s in plan['splits']], co['selection'], old)
        lanes.append((row, lane, marker, {s['split_id']: s for s, _, _ in chosen}, old))
    coverage = dict.fromkeys(COUNTERS, 0)
    resources = {}
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    for row, lane, marker, expected, old in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stage_rows = iter(jsonlines(lane/'fit_stages.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact count mismatch')
            sid = record['split_id']; check(sid in expected and sid not in seen, 'Unexpected/duplicate parent')
            seen.add(sid); stages, measurements = verify_record(record, expected[sid], old)
            accumulate_resources(resources, record, stages)
            check(small == compact_record(record), 'Compact evidence mismatch')
            for stage in stages:
                check(next(stage_rows, None) == dict(scalars(stage), split_id=sid), 'Stage stream mismatch')
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1)
            counts['oof_episodes'] += int(record['k'] > 1); counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for key in ('head_fit_count', 'factorization_count', 'sequence_paths'): counts[key] += record[key]
            for diagnostic, metrics in measurements.items():
                if metrics is None: continue
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                _add(strata['overall'], (diagnostic, population), metrics)
                _add(strata['by_k_new_count'], (diagnostic, record['k'], record['new_count']), metrics)
                _add(strata['by_receiver_scene'], (diagnostic, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), metrics)
                _add(strata['by_model_cohort'], (diagnostic, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), metrics)
        check(seen == set(expected) and next(stage_rows, None) is None, 'Missing parents or extra stage rows')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row completion totals mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
    check(all(coverage[key] == complete[key] for key in COUNTERS), 'Run completion totals mismatch')
    check(tuple(coverage[k] for k in COUNTERS[:-1]) == (160, 40, 120, 1400, 1760, 3168), 'Fixed pilot coverage mismatch')
    check(resources['fit_stage_count'] == coverage['head_fit_count'], 'Measured stage resource coverage mismatch')
    check(complete['finished'] >= launch['started'], 'Run wall-clock bounds mismatch')
    resources.update(run_wall_seconds=complete['finished']-launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _ in lanes
                                       if marker['peak_process_rss_bytes'] is not None), default=None),
        interpretation='Parent/stage/lane sums are measured work totals, not parallel elapsed wall time; stage fit_and_score already includes fit and score.',
        peak_rss_unavailable_reason='Null when the runtime does not expose process high-water RSS')
    dimensions = dict(overall=('diagnostic', 'population'), by_k_new_count=('diagnostic', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('diagnostic', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    summary = dict(status=SUMMARY_STATUS, scope=SCOPE, run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=coverage, resources=resources, algorithm=DIAGNOSTIC_CONFIG, channel=CHANNEL, scenarios=SCENARIOS,
        selected_receiver_scenes={name: co['selection']['receiver_scenes'] for name, co in spec['probe']['cohorts'].items()},
        statistics=tables, actual_A=None, adapted_B=None, adaptation_gain=None,
        unavailable_reason='Original LocalRidge B0/C0 support-only registration diagnostic; no ground A or finetuned B',
        automatic_promotion=False, performance_gate=None, query_rows_used=0, source_rows_used=0,
        optimizer_steps=0, adapter_training=False,
        interpretation=[
            'B0 fits old training support; C0 independently refits all registered training support using the unchanged LocalRidge.',
            'C0 old-column argmax is an offline fixed-score counterfactual, never a deployed role-conditioned predictor.',
            'The exact count decomposition separates old-class decision reordering from added new-class competition; it does not isolate individual kernel or coefficient causes.',
            'OOF pools each physical held ID once before computing accuracy; unequal folds are not equally weighted.',
            'Proxy first averages all anchors per parent, then weights parents equally; repeated support draws are correlated.',
            'True K1 is numerical-only. No held accuracy or head fit is fabricated.',
            'Only the explicitly selected practical residual post_sync/noeq receiver/scene pairs are described; unselected scenes are not measured.',
            'The ideal 10/1/3 percentage-point directions are descriptive and are not advancement or stopping gates.'])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'summary.json', summary)
    for name, rows in tables.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 原 LocalRidge 注册机制诊断', '',
        '完整 160 parent、4 个模型/cohort row 已校验。仅原 LocalRidge B0/C0，无 Adapter。真实 A、微调 B 和适应收益均为 N/A。', '',
        '信道继承 practical residual / post_sync / noeq，fs=25 MHz；未新生成信道。仅解释显式选择的 RX×场景。', '',
        '| 诊断 | K | 新类数 | B0旧 | C0旧类列 | C0全类竞争旧 | C0新 | H | 旧类决策变化 | 新类竞争损失 | 总旧损失 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = {}
    for item in tables['by_k_new_count']:
        cells.setdefault((item['diagnostic'], item['k'], item['new_count']), {})[item['metric']] = item['mean']
    display = ('B0_old_accuracy', 'C0_old_columns_accuracy', 'C0_old_accuracy', 'C0_new_accuracy', 'C0_h',
               'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop')
    for key, values in sorted(cells.items()):
        lines.append('| '+' | '.join([str(v) for v in key]+['N/A' if values[m] is None else f'{100*values[m]:.3f}' for m in display])+' |')
    lines += ['', '准确率为百分数，差值为百分点。恢复比例、绝对新旧差及分场景结果见 CSV；proxy 先均 anchor 后等权 parent。', '',
        f"实测 head fits={coverage['head_fit_count']}，factorizations={coverage['factorization_count']}；运行墙钟 {resources['run_wall_seconds']:.3f} s。",
        f"各 head 的 fit 耗时之和 {resources['stage_fit_seconds_sum']:.3f} s，score 耗时之和 {resources['stage_score_seconds_sum']:.3f} s；并发工作量之和不等于墙钟。", '']
    lines += ['- '+line for line in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
