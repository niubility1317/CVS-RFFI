"""Archive both completed OSC cohorts; dry run unless --write.

Reads scores only after BOTH local completion markers are SCORED. Independent
cell aggregation checks published summaries. Never edits raw evidence or state.
"""
import argparse
from collections import Counter, defaultdict
import datetime
from itertools import product
import json
import math
from pathlib import Path

RUNS = {'rx3': '20260929-phase2-d92-osc-repeat-rx3-m4-r02',
        'rx1': '20260929-phase2-d92-osc-repeat-rx1-m4-r02'}
METHODS = ('D92', 'D92-OSC-v1')
BASELINES = {'rx3': '20260928-phase2-d92-scv-confirmation-manytx-m4-r02',
             'rx1': '20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01'}
KEY = ('model_seed', 'receiver', 'scenario', 'k', 'new_count', 'support_seed')
METRICS = ('accuracy', 'old_accuracy', 'new_accuracy', 'harmonic_mean', 'macro_f1',
           'old_macro_f1', 'new_macro_f1', 'old_floor', 'new_floor', 'forgetting')
PRIMARY = ('old_accuracy', 'new_accuracy', 'harmonic_mean')
MARKER = '<!-- OSC_FINAL_ANALYSIS_20260929 -->'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(rows, metric):
    values = [r[metric] for r in rows if r[metric] is not None]
    check(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Nonfinite metric')
    return math.fsum(values) / len(values) if values else None


def paired_metrics(rows, k, seed=None, new_count=None):
    pool = [r for r in rows if r['k'] == k and (seed is None or r['model_seed'] == seed)
            and (new_count is None or r['new_count'] == new_count)]
    all_cells = {m: [r for r in pool if r['method'] == m] for m in METHODS}
    joint = {m: [r for r in rr if r['new_count'] > 0] for m, rr in all_cells.items()}
    old = {m: mean(rr, 'old_accuracy') for m, rr in all_cells.items()}
    delta = {f: 100 * (mean(joint[METHODS[1]], f) - mean(joint[METHODS[0]], f))
             for f in PRIMARY} if joint[METHODS[0]] else None
    return dict(k=k, model_seed=seed, new_count=new_count, joint_cells=len(joint[METHODS[0]]),
        joint_delta_pp=delta,
        joint_percent={m: {f: 100 * mean(rr, f) for f in PRIMARY} for m, rr in joint.items()} if delta else None,
        old_all_cells=dict(cells=len(all_cells[METHODS[0]]), baseline_percent=100 * old[METHODS[0]],
            candidate_percent=100 * old[METHODS[1]], delta_pp=100 * (old[METHODS[1]] - old[METHODS[0]])))


def compare_tables(rows, summary):
    dimensions = {'per_k': ('k',), 'per_k_all': ('k',), 'per_k_new': ('k', 'new_count'),
                  'per_seed_k': ('model_seed', 'k'), 'per_rx_scene_k': ('receiver', 'scenario', 'k')}
    compared, maximum = 0, 0.0
    for table, dims in dimensions.items():
        groups = defaultdict(list)
        for r in rows:
            if r['method'] in METHODS and (table in ('per_k_all', 'per_k_new') or r['new_count'] > 0):
                groups[tuple(r[d] for d in (*dims, 'method'))].append(r)
        seen = set()
        for published in summary['tables'][table]:
            key = tuple(published[d] for d in (*dims, 'method'))
            check(key in groups and key not in seen, 'Unexpected/duplicate summary group')
            seen.add(key)
            values = groups[key]
            check(published['cells'] == len(values), 'Summary cell count mismatch')
            for metric in METRICS:
                actual = mean(values, metric)
                if actual is None:
                    check(published[metric] is None, 'Summary null mismatch')
                else:
                    check(type(published[metric]) in (int, float), 'Summary numeric mismatch')
                    error = abs(actual - published[metric])
                    check(error < 1e-12, 'Summary arithmetic mismatch: ' + table + '/' + metric)
                    maximum = max(maximum, error)
                compared += 1
        check(seen == set(groups), 'Missing summary group')
    return compared, maximum


def interpret(inputs, summary):
    check(set(inputs) == {'rx3', 'rx1'}, 'Two cohorts required')
    matrices = {c: v['summary']['matrix'] for c, v in inputs.items()}
    first, second = matrices['rx3'], matrices['rx1']
    check(all(first[k] == second[k] for k in KEY if k != 'receiver'), 'Cohort axes differ')
    check(len(first['receiver']) == 3 and len(second['receiver']) == 1
          and not set(first['receiver']) & set(second['receiver']), 'Disjoint 3:1 receivers required')
    check(first['model_seed'] == list(range(2026092701, 2026092705)) and first['k'] == [1, 5, 10, 20]
          and first['new_count'] == [0, 2, 5, 10, 20] and len(first['scenario']) == 3
          and len(first['support_seed']) == 5, 'Full preregistered matrix required')
    rows, coverage, compared, maximum = [], {}, 0, 0.0
    for cohort, v in inputs.items():
        matrix, data = matrices[cohort], v['scores']
        check(all(len(a) == len(set(a)) for a in matrix.values()), 'Duplicate matrix axis')
        check(data['status'] == 'SCORED' and data['selection_feedback_forbidden'] is True, 'Scoring incomplete')
        expected = set(product(*(matrix[k] for k in KEY)))
        rr = data['results']
        for method in METHODS:
            chosen = [r for r in rr if r['method'] == method]
            check(len(chosen) == len(expected) and {tuple(r[k] for k in KEY) for r in chosen} == expected,
                  'Incomplete/duplicate paired cells')
        dg = [r for r in rr if r['method'] == 'frozen_dg']
        expected_dg = set(product(*(matrix[k] for k in KEY[:3])))
        check(len(rr) == 2 * len(expected) + len(expected_dg) and len(dg) == len(expected_dg)
              and {tuple(r[k] for k in KEY[:3]) for r in dg} == expected_dg
              and all(r['new_count'] == 0 and tuple(r[k] for k in KEY) in expected for r in dg), 'DG coverage mismatch')
        pairs = {tuple(r[k] for k in KEY): r for r in rr if r['method'] == METHODS[0]}
        for r in rr:
            if r['method'] == METHODS[1]:
                baseline = pairs[tuple(r[k] for k in KEY)]
                check(all(r[k] == baseline[k] for k in ('row_id', 'split_id', 'classes', 'class_count', 'query_count')),
                      'Pair physical/task metadata mismatch')
        n, err = compare_tables(rr, v['summary']); compared += n; maximum = max(maximum, err)
        coverage[cohort] = dict(records=len(rr), paired_cells=len(expected), receivers=matrix['receiver'])
        rows.extend(rr)
    check(len(rows) == 9648, 'Complete 9648 records required')
    check(summary['methods'] == list(METHODS) and summary['rows'] == len(rows), 'Combined candidate/count mismatch')
    check(summary['matrix'] == dict(first, receiver=first['receiver'] + second['receiver']), 'Combined matrix mismatch')
    acceptance = summary['acceptance']
    check(acceptance['old_max_drop'] == 0.01 and acceptance['require_new_improvement'] is True
          and acceptance['require_h_improvement'] is True
          and acceptance['old_guard_scope'] == 'all_cells_including_old_only', 'Registered acceptance mismatch')
    n, err = compare_tables(rows, summary); compared += n; maximum = max(maximum, err)
    primary, seeds, cells = [], [], []
    cohorts = {c: [] for c in inputs}
    check(len(summary['comparisons']) == 4 and {r['k'] for r in summary['comparisons']} == set(first['k']),
          'Comparison K coverage mismatch')
    for k in first['k']:
        metric = paired_metrics(rows, k)
        seed_rows = [paired_metrics(rows, k, seed=s) for s in first['model_seed']]
        seeds.extend(seed_rows)
        metric['seed_consistency'] = {f: dict(improved=sum(r['joint_delta_pp'][f] > 0 for r in seed_rows), total=4,
            range_pp=[min(r['joint_delta_pp'][f] for r in seed_rows), max(r['joint_delta_pp'][f] for r in seed_rows)]) for f in PRIMARY}
        # Compare in native accuracy units, matching the registered inclusive -0.01 guard.
        metric['old_guard_pass'] = metric['old_all_cells']['delta_pp'] / 100 >= -0.01
        metric['pass_all'] = metric['old_guard_pass'] and all(metric['joint_delta_pp'][f] > 0 for f in ('new_accuracy', 'harmonic_mean'))
        reported = next(r for r in summary['comparisons'] if r['k'] == k)
        check(abs(metric['old_all_cells']['delta_pp'] / 100 - reported['old_guard_delta']) < 1e-12
              and metric['old_guard_pass'] == reported['old_guard'], 'Old guard scope/result mismatch')
        for f in PRIMARY:
            check(abs(metric['joint_delta_pp'][f] / 100 - reported['delta'][f]) < 1e-12, 'Comparison delta mismatch')
        check(reported['h_improved'] == (metric['joint_delta_pp']['harmonic_mean'] > 0)
              and reported['new_improved'] == (metric['joint_delta_pp']['new_accuracy'] > 0), 'Improvement flag mismatch')
        for c, v in inputs.items():
            cohorts[c].append(paired_metrics(v['scores']['results'], k))
        for f in PRIMARY:
            check(abs(metric['joint_delta_pp'][f] - (3 * cohorts['rx3'][-1]['joint_delta_pp'][f]
                  + cohorts['rx1'][-1]['joint_delta_pp'][f]) / 4) < 1e-10, '3:1 equal-cell mismatch')
        check(metric['joint_cells'] == 960 and metric['old_all_cells']['cells'] == 1200, 'Per-K count mismatch')
        primary.append(metric)
        cells.extend(paired_metrics(rows, k, new_count=n) for n in first['new_count'])
    passed = all(v['pass_all'] for v in primary)
    check(summary['preregistered_guard_pass'] == passed, 'Acceptance flag mismatch')
    return dict(status='VERIFIED', method=METHODS[1], audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        records=len(rows), method_counts=dict(Counter(r['method'] for r in rows)), coverage=coverage,
        compared_summary_values=compared, maximum_absolute_error=maximum,
        aggregation='Equal paired cells: 720 rx3 + 240 rx1 joint cells per K (3:1); H is mean per-cell H. Old guard includes all 1200 cells per K, including old-only.',
        per_k=primary, per_model_seed_k=seeds, per_cohort_k=cohorts, complete_k_by_new_count=cells,
        preregistered_guard_pass=passed, candidate_promoted=False, goal_complete=False,
        promotion_note='Archival does not automatically promote a candidate or establish independent generalization.',
        conclusion=conclusion(primary, passed), claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS',
        limitations=['Previously scored targets reused; not fresh independent confirmation or unexposed generalization evidence.',
            'Shared query and overlapping support draws are not independent model replicates; no statistical significance claim.',
            'No algorithm modification, parameter choice, selective rerun, or future design based on this audit.'],
        confusion_arithmetic_scope='Existing complete independent arithmetic audits required VERIFIED; confusion matrices not recomputed here.')


def conclusion(per_k, passed):
    counts = {f: sum(v['joint_delta_pp'][f] > 0 for v in per_k) for f in PRIMARY}
    guards = sum(v['old_guard_pass'] for v in per_k)
    return (f"四个 K 中，联合旧类、新类、H 分别有 {counts['old_accuracy']}、{counts['new_accuracy']}、{counts['harmonic_mean']} 个提升；"
            f"全部任务旧类保护条件有 {guards} 个通过。预登记重复基准条件{'通过' if passed else '未通过'}。"
            '本归档不自动判定晋级，独立泛化与整体目标完成尚未确认。')


def costs(fit):
    def finite(value):
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
        elif type(value) in (int, float):
            check(math.isfinite(value), 'Nonfinite fit audit value')
    finite(fit)
    check(fit['status'] == 'VERIFIED' and fit['method'] == METHODS[1], 'OSC fit audit required')
    models = fit['models']
    check(len(models) == 4 and len({m['row_id'] for m in models}) == 4, 'Four fit rows required')
    check(fit['selection'] == 'none_cv_diagnostic_only' and fit['adapted_state_inherited'] is False,
          'OSC selection/inheritance mismatch')
    check(fit['query_rows_used_for_fit'] == fit['source_rows_used_for_fit'] == fit['new_source_payload_bytes'] == 0
          and fit['ground_summary_used'] is fit['optimizer_convergence_claim'] is False, 'OSC fit boundary mismatch')
    for field in ('total_fits', 'fixed_k1_fits', 'diagnostic_cv_fits', 'analytical_fit_calls', 'optimizer_steps'):
        source = 'fits' if field == 'total_fits' else field
        check(fit[field] == sum(m[source] for m in models), 'Fit count mismatch: ' + field)
    check(fit['fixed_k1_fits'] + fit['diagnostic_cv_fits'] == fit['total_fits'], 'Fit partition mismatch')
    check(fit['optimizer_steps'] == 0, 'OSC has no optimizer steps')
    statuses, fold_statuses = Counter(), Counter()
    for m in models:
        check(all(type(m[k]) is int and m[k] >= 0 for k in ('fits', 'fixed_k1_fits', 'diagnostic_cv_fits',
              'analytical_fit_calls', 'fold_fit_calls', 'final_fit_calls', 'optimizer_steps')), 'Invalid analytical counts')
        check(m['fits'] in (300, 900) and m['fixed_k1_fits'] == m['fits'] // 4
              and m['diagnostic_cv_fits'] == 3 * m['fits'] // 4, 'Full K partition required')
        check(m['optimizer_steps'] == 0 and m['final_fit_calls'] == m['fits']
              and m['fold_fit_calls'] == 3 * m['diagnostic_cv_fits']
              and m['analytical_fit_calls'] == m['fold_fit_calls'] + m['final_fit_calls'], 'Analytical call count mismatch')
        for counts, expected in ((m['final_status_counts'], m['fits']), (m['fold_status_counts'], m['fold_fit_calls'])):
            check(set(counts) <= {'ZERO_PHYSICAL_RESIDUAL', 'SHARED_COVARIANCE_FIT'}
                  and all(type(v) is int and v >= 0 for v in counts.values())
                  and sum(counts.values()) == expected, 'Analytical status coverage mismatch')
        statuses.update(m['final_status_counts']); fold_statuses.update(m['fold_status_counts'])
        resource = m['resources']
        check(resource['model_already_deployed'] is resource['model_incremental_transfer_bytes'] is None
              and resource['new_source_payload_bytes'] == resource['feature_extraction_seconds'] == resource['new_native_forward_count'] == 0
              and resource['ground_summary_used'] is resource['checkpoint_loaded'] is False
              and resource['reused_frozen_cache'] is True, 'Payload/deployment mismatch')
        check(type(resource['existing_frozen_model_file_bytes']) is int and resource['existing_frozen_model_file_bytes'] > 0, 'Unknown checkpoint bytes')
        for key in ('head_bytes', 'persistent_state_bytes'):
            values = m['numeric_state_byte_ranges'][key]
            check(values['min'] == 8200 * 6 and values['max'] == 8200 * 26, 'Compiled W+b state byte range mismatch')
        check(m['numeric_state_byte_ranges']['head_bytes'] == m['numeric_state_byte_ranges']['persistent_state_bytes'],
              'OSC numeric state consists of W+b only')
        for item in m['timing'].values():
            check(type(item['total']) in (int, float) and math.isfinite(item['total']) and item['total'] >= 0, 'Invalid timing')
        for scope in ('all', 'fold', 'final'):
            stages = m['analytical_stage_timing'][scope]
            check(all(type(stages[k]['total']) in (int, float) and math.isfinite(stages[k]['total']) and stages[k]['total'] >= 0
                      for k in ('medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds')), 'Invalid analytical timing')
        for key in ('medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds'):
            stage = m['analytical_stage_timing']
            check(math.isclose(stage['all'][key]['total'], stage['fold'][key]['total'] + stage['final'][key]['total'],
                               rel_tol=1e-10, abs_tol=1e-10), 'Analytical scope timing mismatch')
    timing = {k: math.fsum(m['timing'][k]['total'] for m in models)
              for k in ('fit_seconds', 'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds', 'total_seconds')}
    timing['feature_extraction_seconds'] = math.fsum(m['resources']['feature_extraction_seconds'] for m in models)
    check(all(math.isfinite(v) and v >= 0 for v in timing.values()), 'Invalid timing')
    ranges = {k: [min(m['numeric_state_byte_ranges'][k]['min'] for m in models),
                   max(m['numeric_state_byte_ranges'][k]['max'] for m in models)]
              for k in ('head_bytes', 'persistent_state_bytes')}
    analytical = {scope: {key: math.fsum(m['analytical_stage_timing'][scope][key]['total'] for m in models)
                  for key in ('medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds')}
                  for scope in ('all', 'fold', 'final')}
    return dict(total_fits=fit['total_fits'], fixed_k1_fits=fit['fixed_k1_fits'], diagnostic_cv_fits=fit['diagnostic_cv_fits'],
        analytical_fit_calls=fit['analytical_fit_calls'], fold_fit_calls=sum(m['fold_fit_calls'] for m in models),
        final_fit_calls=sum(m['final_fit_calls'] for m in models), optimizer_steps=0,
        final_status_counts=dict(statuses), fold_status_counts=dict(fold_statuses), timing_totals_seconds=timing,
        analytical_stage_totals_seconds=analytical,
        numeric_state_byte_ranges=ranges,
        per_model_resources=[dict(row_id=m['row_id'], **m['resources'], predictor_peak_process_rss_bytes=m['peak_process_rss_bytes'],
                                 predictor_peak_process_rss_reason=m.get('peak_process_rss_reason')) for m in models],
        model_already_deployed=None, model_incremental_transfer_bytes=None,
        model_transfer_reason='Deployment state unknown; training checkpoint package is not a measured minimum inference package',
        new_source_payload_bytes=0, ground_summary_used=False, optimizer_convergence_claim=False,
        selection='none_cv_diagnostic_only', adapted_state_inherited=False,
        timing_scope='Measured stage sums, not parallel wall duration or satellite latency. Core fit is nested within fit_call/total; do not add nested timings.')


def cost_text(c):
    t, b = c['timing_totals_seconds'], c['numeric_state_byte_ranges']
    sizes = [r['existing_frozen_model_file_bytes'] for r in c['per_model_resources']]
    return (f"全部 {c['total_fits']} 次任务拟合中，K1 固定解析规则 {c['fixed_k1_fits']} 次，support 留出诊断任务 {c['diagnostic_cv_fits']} 次。"
        f"共 {c['analytical_fit_calls']} 次解析拟合，其中 fold 内 {c['fold_fit_calls']} 次、最终拟合 {c['final_fit_calls']} 次。"
        "留出诊断不参与选模或温度校准；没有优化器更新、学习率、梯度或 epoch，也不声称迭代收敛。\n\n"
        f"本次新增特征提取 {t['feature_extraction_seconds']:.3f} 秒、核心拟合 {t['fit_seconds']:.3f} 秒、"
        f"query 打分 {t['query_score_seconds']:.3f} 秒、预测写出 {t['prediction_write_seconds']:.3f} 秒，任务总计时 {t['total_seconds']:.3f} 秒。"
        "阶段累计值不等于并行墙钟时间或星载速度；不叠加已经包含的计时。\n\n"
        f"每条物理观察有 4 个固定相位视图，仍只计一个样本。编译后的 W+b 为每类 8,200 字节，"
        f"持久数值状态合计 {b['persistent_state_bytes'][0]} 至 {b['persistent_state_bytes'][1]} 字节；类 ID、审计容器另计。"
        "复用既有冻结 cache，不加载模型或继承适应状态。历史缓存提取耗时与本次计算分列；接收端 cache、临时数组和进程 RSS 不算地面传输。\n\n"
        f"完整训练 checkpoint 包为 {min(sizes):,} 至 {max(sizes):,} 字节，不是经裁剪核实的最小推理包。"
        "模型部署状态未知，新增模型传输字节为 null；新增源域 payload 为 0 字节，不读取地面摘要，query/source 拟合行数均为 0。Phase1 保持冻结。")


def table(per_k):
    lines = ['| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |', '|---|---:|---:|---:|---:|']
    for r in per_k:
        d = r['joint_delta_pp']
        lines.append(f"| {r['k']} | {d['old_accuracy']:+.3f} | {d['new_accuracy']:+.3f} | {d['harmonic_mean']:+.3f} | {r['old_all_cells']['delta_pp']:+.3f} |")
    return '\n'.join(lines)


def preflight(writes, appends):
    for path, _ in writes:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    for path, _ in appends:
        check(path.is_file() and MARKER not in path.read_text(encoding='utf-8'), 'Missing/already appended report: ' + str(path))


def baseline_binding(cohort, startup, original_startup, run_id=None):
    """Check the frozen source/capsule registration without model or feature I/O."""
    spec, original = startup['spec'], original_startup['spec']
    conf = spec['confirmation']; prior = original['confirmation']
    check(spec['run_id'] == (run_id or RUNS[cohort]) and original['run_id'] == BASELINES[cohort], 'Baseline run identity mismatch')
    check(tuple(conf[k] for k in ('candidate_method', 'candidate_folder', 'candidate_predictor', 'candidate_mode')) ==
          ('D92-OSC-v1', 'osc', 'predict_d92_orbit_shared.py', 'd92_osc_registration'), 'OSC alias mismatch')
    check(conf['capsule'] == prior['capsule'] and conf['reuse_validated_capsule_id'] == spec['data']['capsule_id'],
          'Baseline capsule binding mismatch')
    # Fresh-builder historical specs can legitimately have a null capsule ID.
    # The original capsule path and full immutable D92/DG records still bind it.
    original_id = prior.get('reuse_validated_capsule_id') or original['data'].get('capsule_id')
    if original_id is not None:
        check(original_id == spec['data']['capsule_id'], 'Original capsule identity mismatch')
    rows, old_rows = spec['rows'], original['rows']
    check(len(rows) == len(old_rows) == 4 and len({r['row_id'] for r in rows}) == 4
          and len({r['row_id'] for r in old_rows}) == 4, 'Four baseline model bindings required')
    prior_rows = {r['row_id']: r for r in old_rows}
    for row in rows:
        old = prior_rows.get(row['row_id'])
        check(old is not None and row['seeds']['model'] == old['seeds']['model']
              and row['source_root'] == old['source_root'] and row['reuse_row_root'] == old['output_root'],
              'Baseline model/source row binding mismatch')
        sha = row['expected_checkpoint_sha256']
        check(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Missing frozen checkpoint SHA')
    check(sorted(r['seeds']['model'] for r in rows) == list(range(2026092701, 2026092705)), 'Frozen model seeds mismatch')
    return dict(status='VERIFIED', original_run_id=BASELINES[cohort], capsule_id=spec['data']['capsule_id'],
        model_checkpoints={r['row_id']: r['expected_checkpoint_sha256'] for r in rows})


def compare_baseline(current, original, matrix):
    """Compare only D92/DG cells; previous candidates cannot affect this audit."""
    check(original['status'] == current['status'] == 'SCORED', 'Original/current baseline scoring incomplete')
    count = 0
    for method in ('D92', 'frozen_dg'):
        dims = KEY if method == 'D92' else KEY[:3]
        expected = set(product(*(matrix[k] for k in dims)))
        sides = []
        for data in (current, original):
            selected = [r for r in data['results'] if r['method'] == method]
            mapped = {tuple(r[k] for k in dims): r for r in selected}
            check(len(selected) == len(mapped) and set(mapped) == expected, 'Original D92/DG coverage mismatch')
            sides.append(mapped)
        for key in expected:
            a, b = (s[key] for s in sides)
            # Full physical confusion and every reported metric must be unchanged.
            for field in (*KEY, 'row_id', 'split_id', 'classes', 'class_count', 'query_count', 'confusion', 'class_accuracy', *METRICS):
                check(a[field] == b[field], 'Original D92/DG record differs: ' + field)
            count += 1
    return dict(status='VERIFIED', unchanged_original_records=count, methods=['D92', 'frozen_dg'],
                comparison='Exact physical/task metadata, confusion, class accuracy and all reported metrics')


def prepare(workspace, runs=None):
    runs = dict(RUNS if runs is None else runs)
    check(set(runs) == {'rx3', 'rx1'} and len(set(runs.values())) == 2, 'Two distinct cohort runs required')
    check(all(isinstance(v, str) and v and '/' not in v and '\\' not in v and v not in ('.', '..') for v in runs.values()),
          'Run IDs must be single directory names')
    base = Path(workspace) / 'automation_reports/CV-SincNet'
    folders = {c: base / run for c, run in runs.items()}
    combined = folders['rx3'] / 'results/combined_rx4'
    # Check all outputs and BOTH terminal markers before opening any scores.
    preflight([(combined / 'interpretation_audit.json', None)] + [(p / 'results/artifacts.json', None) for p in folders.values()],
              [(p / 'report.md', None) for p in folders.values()] + [(combined / 'report.md', None)])
    completions = {c: read(p / 'results/complete.json') for c, p in folders.items()}
    check(all(v['status'] == 'SCORED' for v in completions.values()), 'Both cohorts must be SCORED before score access')
    inputs = {}
    for c, folder in folders.items():
        result = folder / 'results'
        data = {name: read(result / filename) for name, filename in dict(startup='startup.json',
            fit='fit_audit.json', arithmetic='arithmetic_audit.json', summary='summary/summary.json').items()}
        check(data['fit']['status'] == data['arithmetic']['status'] == 'VERIFIED', 'Independent audits incomplete')
        check(completions[c]['commit'] == data['startup']['commit'], 'Release commit mismatch')
        spec = data['startup']['spec']; axes = spec['data']
        registered = dict(model_seed=[r['seeds']['model'] for r in spec['rows']], receiver=axes['target_receivers'],
            scenario=axes['scenarios'], k=axes['k'], new_count=axes['new_class_counts'], support_seed=axes['support_seeds'])
        check(data['summary']['matrix'] == registered, 'Summary differs from registered matrix')
        fit = data['fit']
        check(fit['total_fits'] == (3600 if c == 'rx3' else 1200), 'Full fit cohort required')
        check(fit['query_rows_used_for_fit'] == fit['source_rows_used_for_fit'] == fit['new_source_payload_bytes'] == 0
              and fit['ground_summary_used'] is fit['optimizer_convergence_claim'] is False, 'Fit boundary mismatch')
        data['cost'] = costs(fit)
        original_folder = base / BASELINES[c] / 'results'
        data['baseline_binding'] = baseline_binding(c, data['startup'], read(original_folder / 'startup.json'), runs[c])
        check({m['row_id'] for m in fit['models']} == {r['row_id'] for r in data['startup']['spec']['rows']},
              'Fit audit model rows differ from registered baseline')
        data['original_scores_path'] = original_folder / 'scores.json'
        inputs[c] = dict(folder=folder, result=result, complete=completions[c], **data)
    for c, data in inputs.items():
        data['scores'] = read(data['result'] / 'scores.json')
        check(data['complete']['records'] == len(data['scores']['results']) == (7236 if c == 'rx3' else 2412), 'Score record count mismatch')
        data['baseline_audit'] = dict(**data['baseline_binding'], **{
            key: value for key, value in compare_baseline(data['scores'], read(data['original_scores_path']), data['summary']['matrix']).items()
            if key != 'status'}, original_scores=str(data['original_scores_path']))
    audit = interpret(inputs, read(combined / 'summary.json'))
    audit['original_baseline_verification'] = {c: v['baseline_audit'] for c, v in inputs.items()}
    writes, appends = [(combined / 'interpretation_audit.json', audit)], []
    for c, data in inputs.items():
        folder, result, cost = data['folder'], data['result'], data['cost']
        artifact = dict(run_id=runs[c], status='ANALYZED', execution_status='SCORED', evidence_status='VERIFIED',
            candidate=METHODS[1], candidate_promoted=False, goal_complete=False,
            joint_preregistered_guard_pass=audit['preregistered_guard_pass'], records=len(data['scores']['results']),
            release_commit=data['complete']['commit'], cost=cost, per_k=audit['per_cohort_k'][c],
            original_baseline_verification=data['baseline_audit'],
            raw_scores=dict(local=str(result / 'scores.json'), remote='/home/szu2070436088/2510044040/CV-SincNet/runs/' + runs[c] + '/scores.json'),
            raw_fit_traces=[m['trace_file'] for m in data['fit']['models']],
            raw_predictions='/home/szu2070436088/2510044040/CV-SincNet/runs/' + runs[c] + '/<row_id>/osc/predictions.jsonl',
            compact_logs='fit_logs/', analytical_stage_logs='Each row/osc/fit_stages.jsonl and fit_stages.csv',
            arithmetic_audit='arithmetic_audit.json', fit_audit='fit_audit.json', cohort_summary='summary/report.md',
            combined_summary=str(combined / 'report.md'), interpretation_audit=str(combined / 'interpretation_audit.json'),
            old_guard_scope='All paired cells including old-only; joint old/new/H table excludes old-only.',
            claim_scope=audit['claim_scope'], selection_feedback_forbidden=True, limitations=audit['limitations'])
        writes.append((result / 'artifacts.json', artifact))
        text = (f"\n\n{MARKER}\n\n## 最终结果、核验与成本\n\n当前终态 SCORED / ANALYZED，证据 VERIFIED；上文启动说明保留为历史记录。"
            f"实际 release commit：`{data['complete']['commit']}`。完整 {len(data['scores']['results'])} 条评分，算术与训练日志审计通过。\n\n"
            + audit['conclusion'] + '\n\n下表为候选减 D92，单位为百分点。前三列仅含新增类数大于 0 的联合任务；末列含 old-only，用于旧类保护条件。\n\n'
            + table(audit['per_cohort_k'][c]) + '\n\n' + cost_text(cost)
            + '\n\n联合验收按真实单元等权，rx3∶rx1 为 3∶1，不能用单批次替代联合结果。目标数据此前已评分，本次不构成新增独立泛化验证；不作统计显著性声明。'
            '结果不回流调参或选择性重跑，完整原始产物保留。\n\n见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[训练审计](results/fit_audit.json)。\n')
        appends.append((folder / 'report.md', text))
    text = (f"\n\n{MARKER}\n\n## 独立解释核验\n\n完整 9,648 条记录，含 4,800 对任务及 48 条冻结 DG；独立复算 "
        f"{audit['compared_summary_values']} 个汇总数值，最大绝对差 {audit['maximum_absolute_error']:.3g}。"
        '每 K 有 720 对 rx3 加 240 对 rx1 联合任务，旧类保护条件另含 old-only，共 1,200 对。H 是单元 H 的均值。\n\n'
        + table(audit['per_k']) + '\n\n单位为百分点。' + audit['conclusion']
        + '\n\n| K | ΔH 为正的模型 seed | Δ新类为正的模型 seed | ΔH 范围（百分点） |\n|---|---:|---:|---:|\n')
    for row in audit['per_k']:
        h, n = row['seed_consistency']['harmonic_mean'], row['seed_consistency']['new_accuracy']
        text += f"| {row['k']} | {h['improved']}/4 | {n['improved']}/4 | {h['range_pp'][0]:+.3f} 至 {h['range_pp'][1]:+.3f} |\n"
    total_calls = sum(v['cost']['analytical_fit_calls'] for v in inputs.values())
    text += (f'\n4,800 次任务拟合共记录 {total_calls} 次解析拟合；完整 fold 与最终状态见各 run 的 fit_audit.json。'
        '留出诊断不用于选模；没有优化器更新，不作迭代收敛声明。原始 D92 与 frozen_dg 的完整逐单元记录、物理混淆矩阵和指标已核对一致。'
        '时间、字节和部署未知字段见各 run 的 artifacts.json；训练 checkpoint 包不是最小推理包。'
        '\n\n完整四模型 seed、两批次及 K×新增类规模见[解释审计](interpretation_audit.json)。共享 query 与重叠 support 抽样不作为独立模型重复；不作统计显著性声明。'
        '这属于已评分目标的透明重复基准，不是独立泛化验证。结果不回流本候选调参，完整结果保留。\n')
    appends.append((combined / 'report.md', text))
    preflight(writes, appends)
    return writes, appends


def persist(writes, appends):
    preflight(writes, appends)
    # Serialize before any mutation so NaN/schema problems cannot leave partial outputs.
    encoded = [(p, json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n') for p, v in writes]
    for path, text in encoded:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        check(path.read_text(encoding='utf-8') == text, 'Artifact readback mismatch')
    for path, text in appends:
        original = path.read_bytes()
        with path.open('ab') as stream:
            stream.write(text.encode('utf-8'))
        check(path.read_bytes() == original + text.encode('utf-8'), 'Report append readback mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('E:/type10-7'))
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--rx3-run', default=RUNS['rx3'])
    parser.add_argument('--rx1-run', default=RUNS['rx1'])
    args = parser.parse_args()
    writes, appends = prepare(args.workspace.resolve(), {'rx3': args.rx3_run, 'rx1': args.rx1_run})
    if args.write:
        persist(writes, appends)
    print(json.dumps(dict(status='VERIFIED' if args.write else 'PREFLIGHT_VERIFIED',
        outputs=[str(p) for p, _ in writes], reports=[str(p) for p, _ in appends],
        preregistered_guard_pass=writes[0][1]['preregistered_guard_pass']), ensure_ascii=False))


if __name__ == '__main__':
    main()
