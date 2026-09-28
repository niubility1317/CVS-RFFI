"""Audit and append the complete fixed MVKME repeated-benchmark outcome.

Dry-run unless --write. Creates audit/artifacts exclusively, appends reports,
and never changes scores, experiment state, events, registry, or algorithm.
"""
import argparse
from collections import defaultdict, Counter
import datetime
import itertools
import json
import math
from pathlib import Path

RUNS = {'rx3': '20260928-phase2-d92-mvkme-repeat-rx3-m4-r02',
        'rx1': '20260928-phase2-d92-mvkme-repeat-rx1-m4-r01'}
METHODS = ('D92', 'D92-MVKME-v1')
KEY = ('model_seed', 'receiver', 'scenario', 'k', 'new_count', 'support_seed')
METRICS = ('accuracy', 'old_accuracy', 'new_accuracy', 'harmonic_mean', 'macro_f1',
           'old_macro_f1', 'new_macro_f1', 'old_floor', 'new_floor', 'forgetting')
PRIMARY = ('old_accuracy', 'new_accuracy', 'harmonic_mean')
MARKER = '<!-- MVKME_FINAL_ANALYSIS_20260929 -->'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(rows, metric):
    values = [r[metric] for r in rows if r[metric] is not None]
    return math.fsum(values)/len(values) if values else None


def paired_metrics(rows, k, seed=None, new_count=None):
    pool = [r for r in rows if r['k'] == k and (seed is None or r['model_seed'] == seed)]
    if new_count is not None:
        pool = [r for r in pool if r['new_count'] == new_count]
    all_cells = {m: [r for r in pool if r['method'] == m] for m in METHODS}
    joint = {m: [r for r in rr if r['new_count'] > 0] for m, rr in all_cells.items()}
    delta = {f: 100*(mean(joint[METHODS[1]], f)-mean(joint[METHODS[0]], f))
             for f in PRIMARY} if joint[METHODS[0]] else None
    old = {m: mean(rr, 'old_accuracy') for m, rr in all_cells.items()}
    return dict(k=k, model_seed=seed, new_count=new_count,
        joint_cells=len(joint[METHODS[0]]), joint_delta_pp=delta,
        joint_percent={m: {f: 100*mean(rr, f) for f in PRIMARY} for m, rr in joint.items()} if delta else None,
        old_all_cells=dict(cells=len(all_cells[METHODS[0]]), baseline_percent=100*old[METHODS[0]],
            candidate_percent=100*old[METHODS[1]], delta_pp=100*(old[METHODS[1]]-old[METHODS[0]])))


def interpret(inputs, summary):
    rows = [r for v in inputs.values() for r in v['scores']['results']]
    assert len(rows) == 9648
    coverage = {}
    for cohort, v in inputs.items():
        matrix = v['summary']['matrix']
        expected = set(itertools.product(*(matrix[k] for k in KEY)))
        rr = v['scores']['results']
        for method in METHODS:
            chosen = [r for r in rr if r['method'] == method]
            assert len(chosen) == len(expected)
            assert {tuple(r[k] for k in KEY) for r in chosen} == expected
        pairs = {tuple(r[k] for k in KEY): r for r in rr if r['method'] == METHODS[0]}
        for r in rr:
            if r['method'] == METHODS[1]:
                baseline = pairs[tuple(r[k] for k in KEY)]
                assert all(r[k] == baseline[k] for k in ('row_id', 'split_id', 'classes', 'class_count', 'query_count'))
        coverage[cohort] = dict(records=len(rr), paired_cells=len(expected), receivers=matrix['receiver'])
    compared, max_error = 0, 0.0
    dimensions = {'per_k': ('k',), 'per_k_all': ('k',), 'per_k_new': ('k', 'new_count'),
                  'per_seed_k': ('model_seed', 'k'), 'per_rx_scene_k': ('receiver', 'scenario', 'k')}
    for table, dims in dimensions.items():
        groups = defaultdict(list)
        for r in rows:
            if r['method'] in METHODS and (table in ('per_k_all', 'per_k_new') or r['new_count'] > 0):
                groups[tuple(r[d] for d in (*dims, 'method'))].append(r)
        assert len(groups) == len(summary['tables'][table])
        for published in summary['tables'][table]:
            values = groups[tuple(published[d] for d in (*dims, 'method'))]
            assert published['cells'] == len(values)
            for metric in METRICS:
                actual = mean(values, metric)
                if actual is None:
                    assert published[metric] is None
                else:
                    error = abs(actual-published[metric])
                    max_error = max(max_error, error)
                    assert error < 1e-12
                compared += 1
    primary, seeds, cells = [], [], []
    cohorts = {c: [] for c in inputs}
    for k in (1, 5, 10, 20):
        metric = paired_metrics(rows, k)
        seed_rows = [paired_metrics(rows, k, seed=s) for s in range(2026092701, 2026092705)]
        seeds.extend(seed_rows)
        metric['seed_consistency'] = {f: dict(improved=sum(r['joint_delta_pp'][f] > 0 for r in seed_rows), total=4,
            range_pp=[min(r['joint_delta_pp'][f] for r in seed_rows), max(r['joint_delta_pp'][f] for r in seed_rows)]) for f in PRIMARY}
        metric['old_guard_pass'] = metric['old_all_cells']['delta_pp'] >= -1
        metric['pass_all'] = metric['old_guard_pass'] and all(metric['joint_delta_pp'][f] > 0 for f in ('new_accuracy', 'harmonic_mean'))
        reported = next(r for r in summary['comparisons'] if r['k'] == k)
        assert abs(metric['old_all_cells']['delta_pp']/100-reported['old_guard_delta']) < 1e-12
        assert metric['old_guard_pass'] == reported['old_guard']
        for c, v in inputs.items():
            cohorts[c].append(paired_metrics(v['scores']['results'], k))
        for f in PRIMARY:
            assert abs(metric['joint_delta_pp'][f]-(3*cohorts['rx3'][-1]['joint_delta_pp'][f]+cohorts['rx1'][-1]['joint_delta_pp'][f])/4) < 1e-10
        assert metric['joint_cells'] == 960 and metric['old_all_cells']['cells'] == 1200
        primary.append(metric)
        cells.extend(paired_metrics(rows, k, new_count=n) for n in (0, 2, 5, 10, 20))
    assert summary['preregistered_guard_pass'] == all(v['pass_all'] for v in primary) is False
    return dict(status='VERIFIED', method=METHODS[1], audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        records=len(rows), method_counts=dict(Counter(r['method'] for r in rows)), coverage=coverage,
        compared_summary_values=compared, maximum_absolute_error=max_error,
        aggregation='Equal actual paired cells: joint new_count>0 has 720 rx3 + 240 rx1 per K, a 3:1 cohort weight. H is the mean of per-cell H. Old guard uses all 1200 paired cells per K, including old-only.',
        per_k=primary, per_model_seed_k=seeds, per_cohort_k=cohorts, complete_k_by_new_count=cells,
        candidate_promoted=False, goal_complete=False,
        conclusion='All four K have lower mean new accuracy; H increases only at K1 and K5, decreases at K10 and K20. Old all-cell guard passes every K, but comprehensive improvement and the preregistered goal are not achieved.',
        claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS',
        limitations=['Previously scored targets reused; not fresh independent confirmation or unexposed generalization evidence.',
            'Shared query and overlapping support draws are not independent model replicates; no statistical significance claim.',
            'No algorithm modification, parameter choice, selective rerun, or future design based on this audit.'],
        confusion_arithmetic_scope='Existing 9648-record independent arithmetic audits VERIFIED; not recomputed here.')


def table(per_k):
    lines = ['| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |', '|---|---:|---:|---:|---:|']
    for r in per_k:
        d = r['joint_delta_pp']
        lines.append(f"| {r['k']} | {d['old_accuracy']:+.3f} | {d['new_accuracy']:+.3f} | {d['harmonic_mean']:+.3f} | {r['old_all_cells']['delta_pp']:+.3f} |")
    return '\n'.join(lines)


def costs(fit):
    models = fit['models']
    timing = {k: math.fsum(m['timing'][k]['total'] for m in models)
              for k in ('fit_seconds', 'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds', 'total_seconds')}
    timing['feature_extraction_seconds'] = math.fsum(m['resources']['feature_extraction_seconds'] for m in models)
    ranges = {k: [min(m['numeric_state_byte_ranges'][k]['min'] for m in models),
                   max(m['numeric_state_byte_ranges'][k]['max'] for m in models)]
              for k in ('head_bytes', 'fourier_matrix_bytes', 'persistent_state_bytes', 'gram_bytes')}
    return dict(total_fits=fit['total_fits'], fixed_k1_fits=fit['fixed_k1_fits'], support_cv_fits=fit['support_cv_fits'],
        timing_totals_seconds=timing, numeric_state_byte_ranges=ranges,
        per_model_resources=[dict(row_id=m['row_id'], **m['resources'], predictor_peak_process_rss_bytes=m['peak_process_rss_bytes']) for m in models],
        model_incremental_transfer_bytes=None, model_transfer_reason='Deployment state unknown; existing training-checkpoint package is not a measured minimum inference package',
        new_source_payload_bytes=0, ground_summary_used=False,
        timing_scope='Measured stage timings, not parallel wall duration or satellite hardware latency; fit_seconds is nested within fit_call_seconds/total_seconds and must not be added to them.')


def cost_text(c):
    t, b = c['timing_totals_seconds'], c['numeric_state_byte_ranges']
    model_bytes = [r['existing_frozen_model_file_bytes'] for r in c['per_model_resources']]
    return (f"全部 {c['total_fits']} 次拟合中，K1 固定配置 {c['fixed_k1_fits']} 次，support 内部交叉验证 {c['support_cv_fits']} 次。"
        f"累计特征提取 {t['feature_extraction_seconds']:.3f} 秒，核心拟合 {t['fit_seconds']:.3f} 秒，query 打分 {t['query_score_seconds']:.3f} 秒，"
        f"预测序列化/写出 {t['prediction_write_seconds']:.3f} 秒；任务总计时 {t['total_seconds']:.3f} 秒另含函数调用与日志开销。"
        "各阶段计时不等于并行墙钟时间或星载硬件速度，不叠加已经包含的拟合计时。采用闭式岭回归与 support 交叉验证，没有梯度训练或迭代收敛结论。\n\n"
        f"每条物理观察执行 16 个固定视图，仍只计一个样本。分类头为 {b['head_bytes'][0]} 至 {b['head_bytes'][1]} 字节，"
        f"固定 Fourier 矩阵为 {b['fourier_matrix_bytes'][0]} 字节，两者数值状态合计 {b['persistent_state_bytes'][0]} 至 {b['persistent_state_bytes'][1]} 字节。"
        "Gram 临时矩阵、接收端派生 cache 和进程 RSS 单独记录在 artifacts.json，不混称持久状态或地面传输。\n\n"
        f"既有模型文件为 {min(model_bytes):,} 至 {max(model_bytes):,} 字节（约 15.99 MB），这是训练 checkpoint 文件包，不是经裁剪核实的最小推理包。"
        "模型是否已部署未知，新增模型传输字节记为 null；本轮新增源域 payload 为 0 字节，不读取地面摘要。query/source 拟合行数均为 0，Phase1 冻结。")


def prepare(workspace):
    base = workspace/'automation_reports/CV-SincNet'
    inputs = {}
    for cohort, run in RUNS.items():
        folder = base/run; result = folder/'results'
        data = {name: read(result/filename) for name, filename in dict(scores='scores.json', complete='complete.json',
            startup='startup.json', fit='fit_audit.json', arithmetic='arithmetic_audit.json', summary='summary/summary.json').items()}
        assert data['complete']['status'] == data['scores']['status'] == 'SCORED'
        assert data['fit']['status'] == data['arithmetic']['status'] == 'VERIFIED'
        assert data['complete']['commit'] == data['startup']['commit']
        assert data['complete']['records'] == len(data['scores']['results']) == (7236 if cohort == 'rx3' else 2412)
        fit = data['fit']
        assert fit['total_fits'] == (3600 if cohort == 'rx3' else 1200)
        assert fit['query_rows_used_for_fit'] == fit['source_rows_used_for_fit'] == fit['new_source_payload_bytes'] == 0
        assert fit['ground_summary_used'] is fit['optimizer_convergence_claim'] is False
        assert all(m['resources']['model_incremental_transfer_bytes'] is None for m in fit['models'])
        assert all(type(m['resources']['existing_frozen_model_file_bytes']) is int and m['resources']['existing_frozen_model_file_bytes'] > 0 for m in fit['models'])
        inputs[cohort] = dict(folder=folder, result=result, **data)
    combined = inputs['rx3']['result']/'combined_rx4'
    audit = interpret(inputs, read(combined/'summary.json'))
    writes = [(combined/'interpretation_audit.json', audit)]
    appends = []
    for cohort, data in inputs.items():
        folder, result = data['folder'], data['result']; run = RUNS[cohort]
        cost = costs(data['fit'])
        artifact = dict(run_id=run, status='ANALYZED', execution_status='SCORED', evidence_status='VERIFIED',
            candidate=METHODS[1], candidate_promoted=False, goal_complete=False, records=len(data['scores']['results']),
            release_commit=data['complete']['commit'], cost=cost, per_k=audit['per_cohort_k'][cohort],
            raw_scores=dict(local=str(result/'scores.json'), remote='/home/szu2070436088/2510044040/CV-SincNet/runs/'+run+'/scores.json'),
            raw_fit_traces=[m['trace_file'] for m in data['fit']['models']],
            raw_predictions='/home/szu2070436088/2510044040/CV-SincNet/runs/'+run+'/<row_id>/mvkme/predictions.jsonl',
            compact_logs='fit_logs/', arithmetic_audit='arithmetic_audit.json', fit_audit='fit_audit.json',
            cohort_summary='summary/report.md', combined_summary=str(combined/'report.md'), interpretation_audit=str(combined/'interpretation_audit.json'),
            old_guard_scope='All paired cells including old-only; joint old/new/H table excludes old-only.',
            failed_predecessor_preserved='20260928-phase2-d92-mvkme-repeat-rx3-m4-r01' if cohort == 'rx3' else None,
            claim_scope=audit['claim_scope'], selection_feedback_forbidden=True, limitations=audit['limitations'])
        writes.append((result/'artifacts.json', artifact))
        text = (f"\n\n{MARKER}\n\n## 最终结果、核验与成本\n\n当前结果为 SCORED / ANALYZED，证据状态 VERIFIED。上文启动或等待说明保留为历史记录。"
            f"实际发布 commit：`{data['complete']['commit']}`。本批次 {len(data['scores']['results'])} 条评分完整，混淆矩阵算术与拟合日志独立审计均通过。"
            "D92-MVKME-v1 未晋级，整体优化目标尚未完成；不得用单一批次或单元替代联合验收。\n\n"
            "下表为候选减 D92，单位为百分点。前三列只含新类数大于 0 的任务，最后一列含 old-only，按预登记用于旧类保护条件。\n\n"
            + table(audit['per_cohort_k'][cohort])+'\n\n'+cost_text(cost)+'\n\n'
            "四个 RX 按真实单元等权，rx3∶rx1 为 3∶1。复用目标此前已经评分，不是新增独立确认，也不支持未接触目标的泛化声明。"
            "结果不会反馈本候选选参或选择性重跑；所有失败结果与原始产物保留。\n\n"
            "详见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[拟合审计](results/fit_audit.json)。\n")
        appends.append((folder/'report.md', text))
    text = (f"\n\n{MARKER}\n\n## 独立解释核验\n\n完整 9,648 条评分对应 4,800 对任务和 48 条冻结 DG。"
        f"独立核对 {audit['compared_summary_values']} 个汇总值，最大绝对差 {audit['maximum_absolute_error']:.3g}。"
        "每 K 联合任务为 720 对 rx3 加 240 对 rx1；旧类保护条件另含 old-only，共 1,200 对。H 为单元 H 的均值。\n\n"
        + table(audit['per_k'])+'\n\n'
        "单位为百分点。所有 K 的新类平均准确率均下降，H 只在 K1、K5 上升，在 K10、K20 下降。"
        "全部任务旧类保护条件各 K 均通过，但预登记的全面改善目标未达成，候选未晋级。\n\n"
        '| K | ΔH 为正的模型 seed | Δ新类为正的模型 seed | ΔH 范围（百分点） |\n|---|---:|---:|---:|\n')
    for r in audit['per_k']:
        h, n = r['seed_consistency']['harmonic_mean'], r['seed_consistency']['new_accuracy']
        text += f"| {r['k']} | {h['improved']}/4 | {n['improved']}/4 | {h['range_pp'][0]:+.3f} 至 {h['range_pp'][1]:+.3f} |\n"
    text += ('\n不同 support 抽样共享 query，不能当作独立模型重复；这里不作统计显著性声明。'
        '四个模型 seed、两个批次和全部 K×新增类规模均保留在[独立解释审计](interpretation_audit.json)，不挑选有利单元。'
        '这是已评分目标上的透明重复基准，不是独立泛化验证。\n\n'
        '成本与模型传输假设见各 run 的 artifacts.json：训练 checkpoint 大小不是最小推理包；部署状态未知，新增模型传输字节为 null。'
        '4,800 次拟合均为闭式头及 support 交叉验证，不称梯度训练收敛。\n')
    appends.append((combined/'report.md', text))
    for path, _ in writes:
        if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    for path, _ in appends:
        if MARKER in path.read_text(encoding='utf-8'): raise ValueError(f'Already appended: {path}')
    return writes, appends


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('E:/type10-7'))
    p.add_argument('--write', action='store_true')
    args = p.parse_args(); writes, appends = prepare(args.workspace.resolve())
    if args.write:
        for path, data in writes:
            with path.open('x', encoding='utf-8', newline='\n') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
            assert read(path) == data
        for path, text in appends:
            with path.open('a', encoding='utf-8', newline='\n') as f: f.write(text)
            assert path.read_text(encoding='utf-8').count(MARKER) == 1
    print(json.dumps(dict(status='VERIFIED' if args.write else 'PREFLIGHT_VERIFIED',
        outputs=[str(p) for p, _ in writes], reports=[str(p) for p, _ in appends],
        per_k=writes[0][1]['per_k'], costs={d['run_id']: d['cost']['timing_totals_seconds'] for _, d in writes[1:]}), ensure_ascii=False))


if __name__ == '__main__':
    main()
