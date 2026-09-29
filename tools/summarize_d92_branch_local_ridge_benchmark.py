"""Analyze six complete, bound repeat matrices against three frozen baselines.

No fitting, prediction, truth loading, candidate selection or data mutation.
All six terminal markers and spec bindings precede any score-file access.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from summarize_d92_confirmation import KEY, METRICS, matrix_definition, summarize, write_summary
from summarize_d92_repeated_benchmark import combine
from summarize_d92_branch_interaction_benchmark import delta_tables

CANDIDATE = 'D92-BranchLocalRidge-v1'
BRANCH = 'D92-BranchRidge-v1'
INTERACTION = 'D92-BranchInteraction-v1'
OLD_SCOPE = 'All paired cells at this K, including old-only registration'
CLAIM = ('REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; '
         'D92, BranchRidge and BranchInteraction references unchanged; '
         'not independent confirmation or evidence of unexposed generalization')
AGGREGATION = ('Equal paired-cell means across all four receivers; H is the mean of cell H, '
               'not H of marginal means. Fixed models and support draws share target queries; '
               'no independence, significance, or independent-draw confidence claim.')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def keyed(rows, methods):
    selected = [r for r in rows if r['method'] in methods]
    result = {(r['method'], *(r[k] for k in KEY)): r for r in selected}
    check(len(result) == len(selected), 'Duplicate score record')
    return result


def _run_root(spec):
    root = spec['execution']['remote_run_root']
    check(isinstance(root, str) and PurePosixPath(root).is_absolute()
          and '..' not in PurePosixPath(root).parts
          and PurePosixPath(root).name == spec['run_id'], 'Invalid registered remote run root')
    return str(PurePosixPath(root))


def bind_pair(new, old, new_method, old_method, prefix):
    nm, matrix, acceptance, _, _ = matrix_definition(new)
    om, prior_matrix, prior_acceptance, _, _ = matrix_definition(old)
    check(nm == ('D92', new_method) and om == ('D92', old_method), 'Wrong candidate/reference method')
    check(matrix == prior_matrix, 'Reference matrix mismatch')
    check(acceptance == prior_acceptance, 'Reference acceptance mismatch')
    check(new['run_id'] != old['run_id'], 'Candidate cannot be its own reference')
    new_root, old_root = _run_root(new), _run_root(old)
    check(new['confirmation'][prefix+'_reference_run_id'] == old['run_id'], 'Reference run not preregistered')
    check(new['confirmation'][prefix+'_reference_root'] == old_root, 'Reference root not preregistered')
    for spec in (new, old):
        check(spec['joint_benchmark']['old_accuracy_scope'] == OLD_SCOPE, 'All-cell old guard required')
        check(spec['data']['capsule_id'] == spec['confirmation']['reuse_validated_capsule_id'], 'Capsule identity mismatch')
        check('REPEATED_BENCHMARK' in spec['permissions']['claim_scope'], 'Repeated exposure declaration required')
    for key in ('capsule', 'reuse_validated_capsule_id'):
        check(new['confirmation'][key] == old['confirmation'][key], 'Different data capsule')
    prior = {r['seeds']['model']: r for r in old['rows']}
    current = {r['seeds']['model']: r for r in new['rows']}
    check(len(prior) == len(old['rows']) == len(current) == len(new['rows']) == 4,
          'Four unique model rows required')
    check(set(prior) == set(current), 'Model seed mismatch')
    shas = {}
    for seed, row in current.items():
        original = prior[seed]
        sha = row['expected_checkpoint_sha256']
        check(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Invalid model SHA')
        check(all(row[k] == original[k] for k in ('row_id', 'source_root', 'expected_checkpoint_sha256', 'reuse_row_root')),
              'Source/checkpoint/frozen D92 binding mismatch')
        check(row['seeds'] == original['seeds'], 'Seed roles changed')
        shas[seed] = sha
    return dict(status='VERIFIED', candidate_method=new_method, reference_method=old_method,
        candidate_run_id=new['run_id'], reference_run_id=old['run_id'],
        candidate_root=new_root, reference_root=old_root,
        capsule_id=new['data']['capsule_id'], model_checkpoints=shas)


def _cohort_specs(specs, method):
    check(len(specs) == 2, 'Two complete rx3/rx1 cohorts required')
    infos = [matrix_definition(s) for s in specs]
    check(all(info[0] == ('D92', method) for info in infos), 'Incorrect cohort method')
    first, second = (info[1] for info in infos)
    check(any(len(info[1]['receiver']) == 3 for info in infos)
          and any(len(info[1]['receiver']) == 1 for info in infos), 'Complete rx3/rx1 matrix required')
    receivers = first['receiver']+second['receiver']
    check(len(set(receivers)) == 4, 'Overlapping receiver cohorts')
    check(all(first[k] == second[k] for k in KEY if k != 'receiver'), 'Non-receiver cohort axis mismatch')
    check(infos[0][2] == infos[1][2], 'Cohort acceptance mismatch')
    check(len(first['model_seed']) == 4 and len(first['scenario']) == 3
          and first['k'] == [1, 5, 10, 20] and first['new_count'] == [0, 2, 5, 10, 20]
          and len(first['support_seed']) == 5, 'Full frozen matrix required')
    check(len({s['confirmation']['reuse_validated_capsule_id'] for s in specs}) == 2,
          'Two distinct cohort capsule identities required')


def bind_all(specs, branch_specs, interaction_specs):
    for group, method in ((specs, CANDIDATE), (branch_specs, BRANCH), (interaction_specs, INTERACTION)):
        _cohort_specs(group, method)
    all_specs = specs+branch_specs+interaction_specs
    check(len({s['run_id'] for s in all_specs}) == 6, 'Six distinct runs required')
    check(len({_run_root(s) for s in all_specs}) == 6, 'Six distinct registered run roots required')
    bindings = []
    for new, branch, inter in zip(specs, branch_specs, interaction_specs):
        bindings.append(dict(
            local_vs_branch=bind_pair(new, branch, CANDIDATE, BRANCH, 'branch_ridge'),
            local_vs_interaction=bind_pair(new, inter, CANDIDATE, INTERACTION, 'branch_interaction'),
            interaction_vs_branch=bind_pair(inter, branch, INTERACTION, BRANCH, 'branch_ridge')))
    return bindings


def _validate_coverage(data, spec):
    """All coverage and identity checks precede metric aggregation."""
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
    model_rows = {r['seeds']['model']: r for r in spec['rows']}
    for row in rows:
        check(row['row_id'] == model_rows[row['model_seed']]['row_id'], 'Score row/model binding mismatch')
        classes = row['classes']
        check(isinstance(classes, list) and len(set(classes)) == len(classes)
              and all(isinstance(cls, str) and cls for cls in classes)
              and row['class_count'] == len(classes) == 6+row['new_count'], 'Registered class count mismatch')
        check(isinstance(row['split_id'], str) and bool(row['split_id']), 'Invalid score split identity')
        check(type(row['query_count']) is int and row['query_count'] > 0, 'Invalid physical query count')
        for key, expected_value in (
            ('checkpoint_sha256', model_rows[row['model_seed']]['expected_checkpoint_sha256']),
            ('capsule_id', spec['data']['capsule_id'])):
            if key in row:
                check(row[key] == expected_value, 'Score '+key+' binding mismatch')
        for metric in METRICS:
            value = row[metric]
            if value is not None:
                check(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or invalid metric')
            elif row['new_count'] > 0 or metric in ('accuracy', 'old_accuracy', 'macro_f1', 'old_macro_f1', 'old_floor'):
                raise ValueError('Missing defined metric: '+metric)
        if row['new_count'] == 0:
            check(all(row[k] is None for k in ('new_accuracy', 'harmonic_mean', 'new_macro_f1', 'new_floor')),
                  'Old-only new-class metrics must be null')


def compare_records(new, old, new_method, old_method):
    a = keyed(new['results'], ('D92', 'frozen_dg'))
    b = keyed(old['results'], ('D92', 'frozen_dg'))
    check(a == b, 'D92/DG records changed, including metadata or confusion')
    old_head = keyed(old['results'], (old_method,))
    new_head = keyed(new['results'], (new_method,))
    for key, row in new_head.items():
        baseline = a[('D92', *key[1:])]
        reference = old_head[(old_method, *key[1:])]
        for field in ('row_id', 'split_id', 'class_count', 'classes', 'query_count'):
            check(row[field] == baseline[field] == reference[field], 'Paired prediction identity mismatch: '+field)
    return dict(status='VERIFIED', candidate_method=new_method, reference_method=old_method,
        unchanged_d92_records=sum(k[0] == 'D92' for k in a),
        unchanged_dg_records=sum(k[0] == 'frozen_dg' for k in a), reused_reference_records=len(old_head),
        exact_all_record_fields=True)


def _annotate(result):
    result['aggregation'] = AGGREGATION
    result['claim_scope'] = CLAIM
    result['automatic_promotion'] = False
    result['selected_candidate'] = None
    result['old_new_h_all_strictly_improve'] = all(
        row['delta']['old_accuracy'] > 0 and row['delta']['new_accuracy'] > 0
        and row['delta']['harmonic_mean'] > 0 for row in result['comparisons'])
    return delta_tables(result)


def analyze(specs, scored, branch_specs, branch_scores, interaction_specs, interaction_scores):
    check(all(len(v) == 2 for v in (specs, scored, branch_specs, branch_scores, interaction_specs, interaction_scores)),
          'Six complete rx3/rx1 matrices required')
    bindings = bind_all(specs, branch_specs, interaction_specs)
    all_specs, all_scores = specs+branch_specs+interaction_specs, scored+branch_scores+interaction_scores
    for spec, data in zip(all_specs, all_scores):
        _validate_coverage(data, spec)
    audits = []
    for new, branch, inter in zip(scored, branch_scores, interaction_scores):
        audits.append(dict(local_vs_branch=compare_records(new, branch, CANDIDATE, BRANCH),
            local_vs_interaction=compare_records(new, inter, CANDIDATE, INTERACTION),
            interaction_vs_branch=compare_records(inter, branch, INTERACTION, BRANCH)))
    for pair in ('local_vs_branch', 'local_vs_interaction', 'interaction_vs_branch'):
        check(sum(a[pair]['unchanged_d92_records'] for a in audits) == 4800
              and sum(a[pair]['unchanged_dg_records'] for a in audits) == 48
              and sum(a[pair]['reused_reference_records'] for a in audits) == 4800,
              'Full 4800/48 paired reference coverage required')
    comparisons = {}
    for name, baseline, references in (('vs_d92', 'D92', None),
            ('vs_branch_ridge', BRANCH, branch_scores),
            ('vs_branch_interaction', INTERACTION, interaction_scores)):
        work_specs, work_scores = deepcopy(specs), deepcopy(scored)
        if references is not None:
            for spec, data, reference in zip(work_specs, work_scores, references):
                spec.update(analysis_only=True, baseline_method=baseline)
                data['results'] = [r for r in data['results'] if r['method'] != 'D92']+[
                    deepcopy(r) for r in reference['results'] if r['method'] == baseline]
        comparisons[name] = dict(combined=_annotate(combine(work_specs, work_scores)),
            cohorts=[_annotate(summarize(data, spec)) for spec, data in zip(work_specs, work_scores)])
    return dict(status='VERIFIED_TRIPLE_BASELINE_ANALYSIS', method=CANDIDATE,
        baselines=['D92', BRANCH, INTERACTION], bindings=bindings, baseline_record_audits=audits,
        comparisons=comparisons, automatic_promotion=False, selected_candidate=None, claim_scope=CLAIM,
        limitation='H/new use positive-new cells; old guard includes all cells including old-only. '
                   'Per-cell declines remain visible. No all-strata-positive gate or independence claim.')


def prepare(results, branch_references, interaction_references):
    check(all(len(v) == 2 for v in (results, branch_references, interaction_references)), 'Six result directories required')
    folders = [Path(p) for p in list(results)+list(branch_references)+list(interaction_references)]
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
    # Hash score bytes only after the complete six-input barrier, never earlier.
    evidence = {str(p/name): hashlib.sha256((p/name).read_bytes()).hexdigest()
                for p in folders for name in ('complete.json', 'startup.json', 'scores.json')}
    scores = [read(p/'scores.json') for p in folders]
    check(all(len(d['results']) == c['records'] for d, c in zip(scores, complete)), 'Completion score count mismatch')
    result = analyze(specs[:2], scores[:2], specs[2:4], scores[2:4], specs[4:], scores[4:])
    result.update(input_sha256=evidence, input_results=[str(p) for p in folders])
    return result


def _triple(row, delta=False):
    return ' / '.join('N/A' if row[m] is None else format(100*row[m], '+.3f' if delta else '.3f')
                      for m in ('old_accuracy', 'new_accuracy', 'harmonic_mean'))


def write(result, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, comparison in result['comparisons'].items():
        write_summary(comparison['combined'], out/name/'combined_rx4')
        for cohort in comparison['cohorts']:
            label = 'rx3' if len(cohort['matrix']['receiver']) == 3 else 'rx1'
            write_summary(cohort, out/name/label)
    with (out/'analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    combined = result['comparisons']['vs_d92']['combined']
    lines = ['# BranchLocalRidge 三基线重复基准', '', CLAIM, '',
        '全部六个 cohort 完成及来源绑定核实后才读取分数。三个运行配对中的原 D92/DG 记录逐字段完全相同；BranchRidge、BranchInteraction 参考记录原样复用。', '',
        '旧类固定为 6 类，新增类数为 0、2、5、10、20，总 support=(6+Nnew)×K。K=1、5、10、20 全部保留。'
        '每个 K×新增类数为 240 个 cell；每个 K 的新旧联合任务为 960 个 cell，全部任务为 1200 个 cell。'
        'rx3/rx1 按实际 cell 贡献 3:1，不能等权平均两个 cohort 均值。H 先在 cell 内计算，再等权平均。', '',
        '## 候选绝对性能', '', '| K | 旧类 / 新类 / H（%） |', '|---:|---|']
    for row in combined['tables']['per_k']:
        if row['method'] == CANDIDATE:
            lines.append(f"| {row['k']} | {_triple(row)} |")
    lines += ['', '## 全部 K×新增类数绝对值及配对差值', '',
        '绝对值单位为 %；差值单位为百分点；每格顺序为旧类 / 新类 / H。新增类数为 0 是旧类单独任务，新类与 H 均为 N/A。', '',
        '| K | 新增类数 | LocalRidge | Δ vs D92 | Δ vs BranchRidge | Δ vs BranchInteraction |',
        '|---:|---:|---|---|---|---|']
    absolute = {(r['k'], r['new_count']): r for r in combined['tables']['per_k_new'] if r['method'] == CANDIDATE}
    deltas = [{(r['k'], r['new_count']): r for r in result['comparisons'][name]['combined']['tables']['per_k_new_delta']}
              for name in ('vs_d92', 'vs_branch_ridge', 'vs_branch_interaction')]
    for k in combined['matrix']['k']:
        for new_count in combined['matrix']['new_count']:
            key = (k, new_count)
            lines.append(f'| {k} | {new_count} | '+_triple(absolute[key])+' | '+
                         ' | '.join(_triple(table[key], delta=True) for table in deltas)+' |')
    lines += ['', '## 预定每 K 条件', '',
        '| 对照 | K | Δ旧 / Δ新 / ΔH（联合，pp） | Δ旧（全部，pp） | 条件通过 | 联合三项均严格提升 |',
        '|---|---:|---|---:|---|---|']
    for comparison in result['comparisons'].values():
        value = comparison['combined']
        for row in value['comparisons']:
            delta = row['delta']
            passed = row['h_improved'] and row['new_improved'] and row['old_guard']
            strict = all(delta[m] > 0 for m in ('old_accuracy', 'new_accuracy', 'harmonic_mean'))
            lines.append(f"| {value['methods'][0]} | {row['k']} | {_triple(delta, delta=True)} | "
                         f"{100*row['old_guard_delta']:+.3f} | {passed} | {strict} |")
    lines += ['', '每 K 条件是新类与 H 严格提高，全部单元（含 old-only）的旧类下降不超过 1 个百分点。'
        '完整分层、绝对值及差值均保留，不增加全部分层必须全正的门槛。通过保护条件与旧新严格共同改善分别报告。',
        '三个对照分别保存合并与 cohort 的 K、K×新增类数、model seed、RX×场景 CSV。不同模型和 support 抽样复用 target/query，'
        '这里不宣称独立重复、统计显著性或未曝光泛化。分析不选择候选、不回流拟合、阈值或重跑，也不自动晋级或宣告目标完成。', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs=2, required=True, help='LocalRidge rx3/rx1 results')
    parser.add_argument('--branch-references', nargs=2, required=True, help='Frozen BranchRidge rx3/rx1 results')
    parser.add_argument('--interaction-references', nargs=2, required=True, help='Frozen BranchInteraction rx3/rx1 results')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = prepare(args.results, args.branch_references, args.interaction_references)
    write(result, args.output)
    print(json.dumps(dict(status=result['status'], automatic_promotion=False)))


if __name__ == '__main__':
    main()
