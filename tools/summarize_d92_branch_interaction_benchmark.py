"""Compare complete frozen interaction scores to D92 and unchanged BranchRidge.

This is analysis only. It never invokes a predictor, truth loader, or fit routine.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from summarize_d92_confirmation import KEY, METRICS, matrix_definition, summarize, write_summary
from summarize_d92_repeated_benchmark import combine

CANDIDATE = 'D92-BranchInteraction-v1'
REFERENCE = 'D92-BranchRidge-v1'
OLD_SCOPE = 'All paired cells at this K, including old-only registration'


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


def bind(new, old):
    nm, matrix, _, _, _ = matrix_definition(new)
    om, prior_matrix, _, _, _ = matrix_definition(old)
    check(nm == ('D92', CANDIDATE) and om == ('D92', REFERENCE), 'Wrong candidate/reference method')
    check(matrix == prior_matrix, 'Reference matrix mismatch')
    check(new['run_id'] != old['run_id'], 'Candidate cannot be its own reference')
    check(new['confirmation']['branch_ridge_reference_run_id'] == old['run_id'], 'Reference run not preregistered')
    for spec in (new, old):
        check(spec['joint_benchmark']['old_accuracy_scope'] == OLD_SCOPE, 'All-cell old guard required')
        check(spec['data']['capsule_id'] == spec['confirmation']['reuse_validated_capsule_id'], 'Capsule identity mismatch')
    for key in ('capsule', 'reuse_validated_capsule_id'):
        check(new['confirmation'][key] == old['confirmation'][key], 'Different data capsule')
    prior = {r['seeds']['model']: r for r in old['rows']}
    check(len(prior) == len(old['rows']) == len(new['rows']) == 4, 'Four model rows required')
    shas = {}
    for row in new['rows']:
        original = prior[row['seeds']['model']]
        sha = row['expected_checkpoint_sha256']
        check(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Invalid model SHA')
        check(all(row[k] == original[k] for k in ('row_id', 'source_root', 'expected_checkpoint_sha256', 'reuse_row_root')),
              'Source/checkpoint/frozen baseline binding mismatch')
        check(row['seeds'] == original['seeds'], 'Seed roles changed')
        shas[row['seeds']['model']] = sha
    return dict(status='VERIFIED', reference_run_id=old['run_id'], candidate_run_id=new['run_id'],
                capsule_id=new['data']['capsule_id'], model_checkpoints=shas)


def compare_records(new, old):
    """Exact entire-dict equality, not a comparison of rounded headline metrics."""
    a = keyed(new['results'], ('D92', 'frozen_dg'))
    b = keyed(old['results'], ('D92', 'frozen_dg'))
    check(a == b, 'D92/DG records changed, including metadata or confusion')
    old_head = keyed(old['results'], (REFERENCE,))
    new_head = keyed(new['results'], (CANDIDATE,))
    for key, row in new_head.items():
        baseline = a[('D92', *key[1:])]
        reference = old_head[(REFERENCE, *key[1:])]
        for field in ('row_id', 'split_id', 'class_count', 'classes', 'query_count'):
            check(row[field] == baseline[field] == reference[field], 'Paired prediction identity mismatch: '+field)
        check(row['class_count'] == len(row['classes']) == 6+row['new_count'], 'Registered class count mismatch')
    return dict(status='VERIFIED', unchanged_d92_records=sum(k[0] == 'D92' for k in a),
                unchanged_dg_records=sum(k[0] == 'frozen_dg' for k in a), reused_reference_records=len(old_head),
                exact_all_record_fields=True)


def delta_tables(result):
    tables = {}
    for name, rows in result['tables'].items():
        dimensions = [k for k in rows[0] if k not in ('method', 'cells', *METRICS)]
        pairs = {}
        for row in rows:
            pairs.setdefault(tuple(row[k] for k in dimensions), {})[row['method']] = row
        output = []
        for key, pair in sorted(pairs.items()):
            base, candidate = [pair[m] for m in result['methods']]
            check(base['cells'] == candidate['cells'], 'Unpaired aggregation')
            values = {m: None if base[m] is None or candidate[m] is None else candidate[m]-base[m] for m in METRICS}
            output.append(dict(zip(dimensions, key), cells=base['cells'], baseline_method=result['methods'][0],
                               candidate_method=result['methods'][1], **values))
        tables[name+'_delta'] = output
    result['tables'].update(tables)
    return result


def analyze(specs, scored, reference_specs, reference_scores):
    check(all(len(v) == 2 for v in (specs, scored, reference_specs, reference_scores)), 'Two complete cohorts required')
    bindings = [bind(s, r) for s, r in zip(specs, reference_specs)]
    # Validate all four complete score matrices before making any merged view.
    for s, d in zip(specs+reference_specs, scored+reference_scores):
        summarize(d, s)
    record_audits = [compare_records(d, r) for d, r in zip(scored, reference_scores)]
    vs_d92 = combine(specs, scored)
    alt_specs, alt_scores = deepcopy(specs), deepcopy(scored)
    for spec, data, reference in zip(alt_specs, alt_scores, reference_scores):
        spec.update(analysis_only=True, baseline_method=REFERENCE)
        data['results'] = [r for r in data['results'] if r['method'] != 'D92'] + [r for r in reference['results'] if r['method'] == REFERENCE]
    vs_reference = combine(alt_specs, alt_scores)
    comparisons = {'vs_d92': dict(combined=delta_tables(vs_d92), cohorts=[delta_tables(summarize(d, s)) for s, d in zip(specs, scored)]),
                   'vs_branch_ridge': dict(combined=delta_tables(vs_reference), cohorts=[delta_tables(summarize(d, s)) for s, d in zip(alt_specs, alt_scores)])}
    check(sum(a['unchanged_d92_records'] for a in record_audits) == 4800
          and sum(a['unchanged_dg_records'] for a in record_audits) == 48
          and sum(a['reused_reference_records'] for a in record_audits) == 4800, 'Full 4800/48 reference coverage required')
    return dict(status='VERIFIED_DUAL_BASELINE_ANALYSIS', method=CANDIDATE, baselines=['D92', REFERENCE],
                bindings=bindings, baseline_record_audits=record_audits, comparisons=comparisons,
                automatic_promotion=False, selected_candidate=None,
                claim_scope='Previously scored targets reused; both comparisons are development benchmarks, not independent confirmation',
                limitation='H/new use positive-new cells; old guard includes all cells. Fixed seed draws share query; no independent-draw confidence claim.')


def prepare(results, references):
    folders = [Path(p) for p in results+references]
    check(len(folders) == 4 and len({p.resolve() for p in folders}) == 4, 'Four distinct results directories required')
    complete = [read(p/'complete.json') for p in folders]
    check(all(v['status'] == 'SCORED' for v in complete), 'All cohorts must be SCORED before any score access')
    startups = [read(p/'startup.json') for p in folders]
    specs = [v['spec'] for v in startups]
    for s, c in zip(startups, complete):
        check(s['commit'] == c['commit'], 'Runtime commit mismatch')
    # Binding failures also occur before score access.
    for new, old in zip(specs[:2], specs[2:]):
        bind(new, old)
    evidence = {str(p/name): hashlib.sha256((p/name).read_bytes()).hexdigest()
                for p in folders for name in ('complete.json', 'startup.json', 'scores.json')}
    scores = [read(p/'scores.json') for p in folders]
    check(all(len(d['results']) == c['records'] for d, c in zip(scores, complete)), 'Completion score count mismatch')
    result = analyze(specs[:2], scores[:2], specs[2:], scores[2:])
    result['input_sha256'] = evidence
    result['input_results'] = [str(p) for p in folders]
    return result


def write(result, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, comparison in result['comparisons'].items():
        write_summary(comparison['combined'], out/name/'combined_rx4')
        for audit, cohort in zip(result['bindings'], comparison['cohorts']):
            label = 'rx3' if len(cohort['matrix']['receiver']) == 3 else 'rx1'
            write_summary(cohort, out/name/label)
    with (out/'analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['# BranchInteraction 双基线重复基准', '', result['claim_scope'], '',
             '原 D92/DG 所有记录与冻结 BranchRidge 运行完全一致；BranchRidge 记录原样复用，未重跑基线。', '',
             '旧类固定为 6 类，新增类数为 0、2、5、10、20，总注册类数为 6、8、11、16、26。K 是每类 support 数，总 support=(6+Nnew)×K。联合表仅含 Nnew>0 的四档，按 cell 等权，四档各占 1/4。每个 K×Nnew 为 240 个 cell，每 K 联合为 960 个 cell；3RX 与 1RX cohort 贡献按实际 cell 为 3:1。H 先在每个 cell 内计算，再取均值。', '',
             '## 候选绝对性能', '',
             '| K | 旧类准确率（%） | 新类准确率（%） | H（%） |',
             '|---:|---:|---:|---:|']
    combined = result['comparisons']['vs_d92']['combined']
    for row in combined['tables']['per_k']:
        if row['method'] == CANDIDATE:
            lines.append(f"| {row['k']} | {100*row['old_accuracy']:.3f} | {100*row['new_accuracy']:.3f} | {100*row['harmonic_mean']:.3f} |")
    lines += ['', '| K | 新增 2 类 H（%） | 新增 5 类 H（%） | 新增 10 类 H（%） | 新增 20 类 H（%） |',
              '|---:|---:|---:|---:|---:|']
    by_cell = {(r['k'],r['new_count']):r for r in combined['tables']['per_k_new'] if r['method'] == CANDIDATE}
    for k in combined['matrix']['k']:
        lines.append('| '+str(k)+' | '+' | '.join(f"{100*by_cell[k,n]['harmonic_mean']:.3f}" for n in (2,5,10,20))+' |')
    lines += ['', '新增类数为 0 时只定义旧类准确率，新类准确率与新旧 H 不定义。以下旧类单独任务不混入上方联合表。', '',
              '| K | 旧类单独任务准确率（%） |', '|---:|---:|']
    for k in combined['matrix']['k']:
        lines.append(f"| {k} | {100*by_cell[k,0]['old_accuracy']:.3f} |")
    lines += ['', '## 相对两基线的差值', '',
             '| 对照 | K | Δ旧（联合，pp） | Δ新（pp） | ΔH（pp） | Δ旧（全部，pp） | 每 K 条件通过 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for comparison in result['comparisons'].values():
        value = comparison['combined']
        for row in value['comparisons']:
            d = row['delta']; passed = row['h_improved'] and row['new_improved'] and row['old_guard']
            lines.append(f"| {value['methods'][0]} | {row['k']} | {100*d['old_accuracy']:+.3f} | {100*d['new_accuracy']:+.3f} | {100*d['harmonic_mean']:+.3f} | {100*row['old_guard_delta']:+.3f} | {passed} |")
    lines += ['', '两个对照分别保留联合及 cohort 的 K、新增类数、模型 seed、RX×场景表和对应差值。H 为逐 cell 调和后平均；新增类数为 0 时新类/H 留空。旧类保护包含 old-only。',
              '每 K 条件为 H、新类严格提高且全单元旧类退化不超过 1 个百分点。该分析不自动晋级或完成目标，也不把已曝光数据当作独立确认。', '']
    with (out/'report.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs=2, required=True, help='New rx3/rx1 results directories')
    parser.add_argument('--references', nargs=2, required=True, help='Matching frozen BranchRidge rx3/rx1 results directories')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = prepare(args.results, args.references)
    write(result, args.output)
    print(json.dumps(dict(status=result['status'], automatic_promotion=False)))


if __name__ == '__main__':
    main()
