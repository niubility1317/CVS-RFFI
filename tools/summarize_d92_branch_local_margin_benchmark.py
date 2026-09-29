"""Read six complete frozen repeat matrices; compare Margin directly to LocalRidge.

No fitting, prediction, truth loading, selection, or mutation of input artifacts.
All terminal/spec bindings precede score-file access; all record identities precede
numeric comparisons. Historical targets are repeated, not independent validation.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from summarize_d92_confirmation import KEY, METRICS, matrix_definition, summarize, write_summary
from summarize_d92_repeated_benchmark import combine
from summarize_d92_branch_local_ridge_benchmark import (
    OLD_SCOPE, AGGREGATION, check, read, keyed, _run_root, bind_pair,
    _cohort_specs, _validate_coverage, compare_records, _triple)

CANDIDATE = 'D92-BranchLocalMargin-v1'
LOCAL = 'D92-BranchLocalRidge-v1'
INTERACTION = 'D92-BranchInteraction-v1'
CLAIM = ('REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalMargin predictions; '
         'LocalRidge is the direct baseline; Interaction and original D92 are descriptive references; '
         'not independent confirmation or evidence of unexposed generalization')


def bind_all(specs, local_specs, interaction_specs):
    for group, method in ((specs, CANDIDATE), (local_specs, LOCAL), (interaction_specs, INTERACTION)):
        _cohort_specs(group, method)
    all_specs = specs+local_specs+interaction_specs
    check(len({s['run_id'] for s in all_specs}) == 6, 'Six distinct runs required')
    check(len({_run_root(s) for s in all_specs}) == 6, 'Six distinct registered run roots required')
    return [dict(
        margin_vs_local=bind_pair(new, local, CANDIDATE, LOCAL, 'branch_local_ridge'),
        margin_vs_interaction=bind_pair(new, inter, CANDIDATE, INTERACTION, 'branch_interaction'),
        local_vs_interaction=bind_pair(local, inter, LOCAL, INTERACTION, 'branch_interaction'))
        for new, local, inter in zip(specs, local_specs, interaction_specs)]


def _identity_coverage(data, spec):
    """Traverse identity fields only, before reading any metric values."""
    check(data['status'] == 'SCORED' and data['selection_feedback_forbidden'] is True,
          'Not a frozen complete evaluation')
    methods, _, _, expected, expected_dg = matrix_definition(spec)
    rows = data['results']
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        check(len(group) == len(expected) and {tuple(r[k] for k in KEY) for r in group} == expected,
              'Incomplete paired matrix: '+method)
    dg = [r for r in rows if r['method'] == 'frozen_dg']
    check(len(rows) == 2*len(expected)+len(expected_dg) and len(dg) == len(expected_dg)
          and {(r['model_seed'], r['receiver'], r['scenario']) for r in dg} == expected_dg
          and all(r['new_count'] == 0 and tuple(r[k] for k in KEY) in expected for r in dg), 'Wrong final coverage')
    models = {r['seeds']['model']: r for r in spec['rows']}
    for row in rows:
        check(row['row_id'] == models[row['model_seed']]['row_id'], 'Score row/model binding mismatch')
        classes = row['classes']
        check(isinstance(classes, list) and all(isinstance(c, str) and c for c in classes)
              and len(set(classes)) == len(classes) == row['class_count'] == 6+row['new_count'],
              'Registered class count mismatch')
        check(isinstance(row['split_id'], str) and bool(row['split_id']), 'Invalid score split identity')
        check(type(row['query_count']) is int and row['query_count'] > 0, 'Invalid physical query count')
        for key, value in (('checkpoint_sha256', models[row['model_seed']]['expected_checkpoint_sha256']),
                           ('capsule_id', spec['data']['capsule_id'])):
            if key in row:
                check(row[key] == value, 'Score '+key+' binding mismatch')


def _paired_identity(new, old, new_method, old_method):
    candidate = keyed(new['results'], (new_method,))
    baseline = keyed(new['results'], ('D92',))
    prior = keyed(old['results'], (old_method,))
    for key, row in candidate.items():
        reference = prior[(old_method, *key[1:])]
        original = baseline[('D92', *key[1:])]
        for field in ('row_id', 'split_id', 'class_count', 'classes', 'query_count'):
            check(row[field] == reference[field] == original[field], 'Paired prediction identity mismatch: '+field)


def _paired_tables(result, data):
    """Average same-cell differences, never add deltas across comparison runs."""
    baseline, candidate = result['methods']
    pairs = keyed(data['results'], (baseline, candidate))
    dimensions = dict(per_k=('k',), per_k_all=('k',), per_k_new=('k', 'new_count'),
        per_seed_k=('model_seed', 'k'), per_rx_scene_k=('receiver', 'scenario', 'k'))
    for name, dims in dimensions.items():
        groups = defaultdict(list)
        for key, row in pairs.items():
            if key[0] != candidate or (name not in ('per_k_all', 'per_k_new') and not row['new_count']):
                continue
            reference = pairs[(baseline, *key[1:])]
            delta = {metric: None if row[metric] is None else row[metric]-reference[metric] for metric in METRICS}
            groups[tuple(row[d] for d in dims)].append(delta)
        table = []
        for key, values in sorted(groups.items()):
            means = {}
            for metric in METRICS:
                defined = [v[metric] for v in values if v[metric] is not None]
                means[metric] = math.fsum(defined)/len(defined) if defined else None
            table.append(dict(zip(dims, key), cells=len(values), baseline_method=baseline,
                              candidate_method=candidate, **means))
        result['tables'][name+'_delta'] = table
    per_k = {r['k']: r for r in result['tables']['per_k_delta']}
    all_k = {r['k']: r for r in result['tables']['per_k_all_delta']}
    for row in result['comparisons']:
        row['delta'] = {m: per_k[row['k']][m] for m in METRICS}
        row['old_guard_delta'] = all_k[row['k']]['old_accuracy']
        row.update(h_improved=row['delta']['harmonic_mean'] > 0,
            new_improved=row['delta']['new_accuracy'] > 0,
            new_guard=row['delta']['new_accuracy'] >= -.01, old_guard=row['old_guard_delta'] >= -.01)
        seed_rows = [r for r in result['tables']['per_seed_k_delta'] if r['k'] == row['k']]
        row['paired_seed_delta_range'] = {m: [min(r[m] for r in seed_rows), max(r[m] for r in seed_rows)] for m in METRICS}
    result['preregistered_guard_pass'] = all(r['h_improved'] and r['new_improved'] and r['old_guard'] for r in result['comparisons'])
    result['new_improved_every_k'] = all(r['new_improved'] for r in result['comparisons'])
    return result


def _annotate(result, data, direct):
    result = _paired_tables(result, data)
    result.update(aggregation=AGGREGATION, claim_scope=CLAIM, automatic_promotion=False,
        selected_candidate=None, comparison_role='direct_baseline' if direct else 'descriptive_control',
        advancement_gate=direct, paired_delta_rule='mean_of_candidate_minus_reference_on_identical_cell_keys')
    result['old_new_h_all_strictly_improve'] = all(all(row['delta'][m] > 0
        for m in ('old_accuracy', 'new_accuracy', 'harmonic_mean')) for row in result['comparisons'])
    return result


def analyze(specs, scored, local_specs, local_scores, interaction_specs, interaction_scores):
    check(all(len(v) == 2 for v in (specs, scored, local_specs, local_scores, interaction_specs, interaction_scores)),
          'Six complete rx3/rx1 matrices required')
    bindings = bind_all(specs, local_specs, interaction_specs)
    all_specs, all_scores = specs+local_specs+interaction_specs, scored+local_scores+interaction_scores
    for spec, data in zip(all_specs, all_scores):
        _identity_coverage(data, spec)
    for new, local, inter in zip(scored, local_scores, interaction_scores):
        _paired_identity(new, local, CANDIDATE, LOCAL)
        _paired_identity(new, inter, CANDIDATE, INTERACTION)
        _paired_identity(local, inter, LOCAL, INTERACTION)
    audits = [dict(margin_vs_local=compare_records(new, local, CANDIDATE, LOCAL),
        margin_vs_interaction=compare_records(new, inter, CANDIDATE, INTERACTION),
        local_vs_interaction=compare_records(local, inter, LOCAL, INTERACTION))
        for new, local, inter in zip(scored, local_scores, interaction_scores)]
    for pair in ('margin_vs_local', 'margin_vs_interaction', 'local_vs_interaction'):
        check(sum(a[pair]['unchanged_d92_records'] for a in audits) == 4800
              and sum(a[pair]['unchanged_dg_records'] for a in audits) == 48
              and sum(a[pair]['reused_reference_records'] for a in audits) == 4800,
              'Full 4800/48 paired reference coverage required')
    for spec, data in zip(all_specs, all_scores):
        _validate_coverage(data, spec)
    comparisons = {}
    for name, baseline, references in (('vs_local_ridge', LOCAL, local_scores),
            ('vs_branch_interaction', INTERACTION, interaction_scores), ('vs_d92', 'D92', None)):
        work_specs, work_scores = deepcopy(specs), deepcopy(scored)
        if references is not None:
            for spec, data, reference in zip(work_specs, work_scores, references):
                spec.update(analysis_only=True, baseline_method=baseline)
                data['results'] = [r for r in data['results'] if r['method'] != 'D92']+[
                    deepcopy(r) for r in reference['results'] if r['method'] == baseline]
        pooled_data = dict(results=[r for data in work_scores for r in data['results']])
        comparisons[name] = dict(combined=_annotate(combine(work_specs, work_scores), pooled_data, baseline == LOCAL),
            cohorts=[_annotate(summarize(data, spec), data, baseline == LOCAL) for spec, data in zip(work_specs, work_scores)])
    direct = comparisons['vs_local_ridge']['combined']
    screen = dict(baseline=LOCAL, k=list(direct['matrix']['k']),
        per_k=deepcopy(direct['comparisons']), passed=direct['preregistered_guard_pass'],
        strict_joint_improvement=direct['old_new_h_all_strictly_improve'],
        rule='Each K: positive-new-cell new/H>0; all-cell old delta>=-0.01; no all-strata-positive gate')
    return dict(status='VERIFIED_TRIPLE_BASELINE_ANALYSIS', method=CANDIDATE,
        baselines=[LOCAL, INTERACTION, 'D92'], direct_baseline=LOCAL, descriptive_baselines=[INTERACTION, 'D92'],
        direct_baseline_every_k_screen=screen, bindings=bindings, baseline_record_audits=audits,
        comparisons=comparisons, automatic_promotion=False, selected_candidate=None, claim_scope=CLAIM,
        limitation='H/new use positive-new cells; old guard includes all cells including old-only. '
                   'All strata and per-cell declines remain visible. No added all-strata gate or independence claim.')


def prepare(results, local_references, interaction_references):
    check(all(len(v) == 2 for v in (results, local_references, interaction_references)), 'Six result directories required')
    folders = [Path(p) for p in list(results)+list(local_references)+list(interaction_references)]
    check(len({p.resolve() for p in folders}) == 6, 'Six distinct results directories required')
    complete = [read(p/'complete.json') for p in folders]
    check(all(v['status'] == 'SCORED' for v in complete), 'All six cohorts must be SCORED before any score access')
    startups = [read(p/'startup.json') for p in folders]
    specs = [v['spec'] for v in startups]
    for startup, terminal, spec in zip(startups, complete, specs):
        check(startup['commit'] == terminal['commit'], 'Runtime commit mismatch')
        _, _, _, expected, expected_dg = matrix_definition(spec)
        check(terminal['records'] == 2*len(expected)+len(expected_dg), 'Completion record count is not full matrix')
        if 'run_id' in terminal:
            check(terminal['run_id'] == spec['run_id'], 'Completion run identity mismatch')
    bind_all(specs[:2], specs[2:4], specs[4:])
    evidence = {str(p/name): hashlib.sha256((p/name).read_bytes()).hexdigest()
        for p in folders for name in ('complete.json', 'startup.json', 'scores.json')}
    scores = [read(p/'scores.json') for p in folders]
    check(all(len(d['results']) == c['records'] for d, c in zip(scores, complete)), 'Completion score count mismatch')
    result = analyze(specs[:2], scores[:2], specs[2:4], scores[2:4], specs[4:], scores[4:])
    result.update(input_sha256=evidence, input_results=[str(p) for p in folders])
    return result


def write(result, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, comparison in result['comparisons'].items():
        write_summary(comparison['combined'], out/name/'combined_rx4')
        for cohort in comparison['cohorts']:
            write_summary(cohort, out/name/('rx3' if len(cohort['matrix']['receiver']) == 3 else 'rx1'))
    (out/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    combined = result['comparisons']['vs_local_ridge']['combined']
    lines = ['# BranchLocalMargin 三基线重复基准', '', CLAIM, '',
        'LocalRidge 是唯一直接比较基线；Interaction 与原 D92 保留描述性对照。'
        '全部六个 cohort 完成并核实绑定后才读取分数；三个运行配对中的原 D92/DG 记录逐字段完全一致。', '',
        '旧类固定 6 类，新增 0、2、5、10、20 类，总 support=(6+Nnew)×K。K=1、5、10、20 全保留。'
        '每 K×新增类数为 240 个 cell；每 K 正新增类联合任务为 960 个 cell，全部任务为 1200 个 cell。'
        'rx3/rx1 按实际 cell 贡献 3:1；H 是 cell H 的均值，差值是相同 cell 内相减后等权平均。', '',
        '## 候选绝对性能', '', '| K | 旧类 / 新类 / H（%） |', '|---:|---|']
    for row in combined['tables']['per_k']:
        if row['method'] == CANDIDATE:
            lines.append(f"| {row['k']} | {_triple(row)} |")
    lines += ['', '## 全部 K×新增类数绝对值与差值', '',
        '绝对值为 %，差值为百分点；每格按旧类 / 新类 / H 排列。新增 0 类为旧类单独任务，新类/H 为 N/A。', '',
        '| K | 新增类数 | LocalMargin | Δ vs LocalRidge | Δ vs Interaction | Δ vs D92 |',
        '|---:|---:|---|---|---|---|']
    absolute = {(r['k'], r['new_count']): r for r in combined['tables']['per_k_new'] if r['method'] == CANDIDATE}
    deltas = [{(r['k'], r['new_count']): r for r in result['comparisons'][name]['combined']['tables']['per_k_new_delta']}
              for name in ('vs_local_ridge', 'vs_branch_interaction', 'vs_d92')]
    for k in combined['matrix']['k']:
        for new in combined['matrix']['new_count']:
            key = k, new
            lines.append(f'| {k} | {new} | '+_triple(absolute[key])+' | '+
                ' | '.join(_triple(table[key], delta=True) for table in deltas)+' |')
    lines += ['', '## 每 K 比较条件', '',
        '| 对照角色 / 方法 | K | Δ旧 / Δ新 / ΔH（联合，pp） | Δ旧（全部，pp） | 条件通过 | 三项严格提高 |',
        '|---|---:|---|---:|---|---|']
    for comparison in result['comparisons'].values():
        value = comparison['combined']
        role = '直接基线' if value['advancement_gate'] else '描述性对照'
        for row in value['comparisons']:
            passed = row['h_improved'] and row['new_improved'] and row['old_guard']
            strict = all(row['delta'][m] > 0 for m in ('old_accuracy', 'new_accuracy', 'harmonic_mean'))
            lines.append(f"| {role} / {value['methods'][0]} | {row['k']} | {_triple(row['delta'], delta=True)} | "
                         f"{100*row['old_guard_delta']:+.3f} | {passed} | {strict} |")
    screen = result['direct_baseline_every_k_screen']
    lines += ['', f"LocalRidge 直接基线每 K 条件通过：{screen['passed']}；联合旧/新/H 每 K 均严格提高：{screen['strict_joint_improvement']}。",
        '每 K 条件要求联合新类与 H 严格提高、全部单元（含 old-only）旧类下降不超过 1 个百分点。'
        '两个较早对照不新增推进门槛；全部分层继续报告，不增加全分层正向条件。',
        '三个比较目录分别保存合并/cohort 的 K、K×新增类数、模型 seed、RX×场景绝对值和同 cell 差值 CSV。'
        '固定模型与 support 抽样共享 target/query，不宣称独立重复、统计显著性或未曝光泛化。'
        '分析不回流拟合、选参或选择性重跑，不自动晋级或宣告目标完成。', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs=2, required=True, help='Frozen LocalMargin rx3/rx1 results')
    parser.add_argument('--local-references', nargs=2, required=True, help='Frozen LocalRidge rx3/rx1 results')
    parser.add_argument('--interaction-references', nargs=2, required=True, help='Frozen Interaction rx3/rx1 results')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = prepare(args.results, args.local_references, args.interaction_references)
    write(result, args.output)
    print(json.dumps(dict(status=result['status'], automatic_promotion=False)))


if __name__ == '__main__':
    main()
