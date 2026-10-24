"""Pure-stdlib summaries of complete registered contract panels, without truth IO."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics

from experiments.cric_dura_comparison.analyze_main import metrics, moments, write_csv, SEEDS, VIEWS

TASKS = ('cross_date', 'cross_rx', 'cross_both')
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')
VARIANTS = tuple((f, c) for c in 'ABCD' for f in FAMILIES) + (
    ('pamf', 'A'), ('pamf', 'C'), ('pamf', 'native_T1'))


def validate_panel(rows):
    expected = {(f, c, s) for f, c in VARIANTS for s in SEEDS}
    actual = [(r['row']['family'], r['row']['cell'], r['row']['model_seed']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('A complete contract requires all 19 variants and three actual F seeds')
    for rec in rows:
        r = rec['row']
        role = 'LU' if r['cell'] in ('B', 'D', 'native_T1') else 'L'
        if r['G_roles'] != role:
            raise ValueError('G role differs from the registered recipe')


def aggregate(obj):
    panels = {p: [r for r in obj['rows'] if r['row']['protocol_id'] == p]
              for p in obj['complete_protocols']}
    if not panels or sum(map(len, panels.values())) != len(obj['rows']):
        raise ValueError('Snapshot panel membership differs')
    values = {}; counts = {}; nets = []; cm_count = 0
    for protocol, records in panels.items():
        validate_panel(records)
        for rec in records:
            row = rec['row']; score = rec['score']; seed = row['model_seed']
            ident = (protocol, row['family'], row['cell'], seed)
            if (score['row_id'] != row['row_id'] or score['model_seed'] != seed
                    or not score['truth_last'] or not score['target_feedback_forbidden']):
                raise ValueError('Independent score identity differs')
            reader = 'DURA-' + row['G_roles']
            expected = {(t, r, v) for t in TASKS for r in ('raw', reader) for v in VIEWS}
            actual = {}
            for result in score['results']:
                key = (result['test'], result['readout'], result['view'])
                if key in actual or result['dimension'] != 'overall' or result['stratum'] != 'ALL':
                    raise ValueError('Duplicate/non-overall score')
                actual[key] = result
                val = metrics(result, len(rec['classes'])); cm_count += 1
                values[(*ident, *key)] = val
                nkey = (protocol, result['test'])
                if nkey in counts and counts[nkey] != result['count']:
                    raise ValueError('Physical population count differs across variants or seeds')
                counts[nkey] = result['count']
            if set(actual) != expected:
                raise ValueError('Missing registered task, route or view')
            paired = {(p['test'], p['readout'], p['view']): p for p in score['paired_raw']}
            if len(paired) != 21 or set(paired) != {(t, reader, v) for t in TASKS for v in VIEWS}:
                raise ValueError('Incomplete paired-raw coverage')
            for (task, _, view), p in paired.items():
                raw = actual[task, 'raw', view]; restored = actual[task, reader, view]
                correct = lambda a: sum(a['confusion'][i][i] for i in range(len(a['confusion'])))
                net = correct(restored) - correct(raw); n = raw['count']
                if (p['count'] != n or p['net'] != net or p['repairs']-p['damage'] != net
                        or min(p['repairs'], p['damage']) < 0
                        or p['repairs'] > n-correct(raw) or p['damage'] > correct(raw)):
                    raise ValueError('Restoration net differs from the confusion matrices')
                nets.append(dict(protocol=protocol, family=row['family'], cell=row['cell'], seed=seed,
                                 test=task, view=view, readout=reader, count=n,
                                 repairs=p['repairs'], damage=p['damage'], net=net))
            for task in TASKS:
                for route in ('raw', reader):
                    six = [values[(*ident, task, route, v)] for v in VIEWS[1:]]
                    values[(*ident, task, route, 'six_practical_mean')] = {
                        k: statistics.mean(a[k] for a in six) for k in ('accuracy', 'macro_f1')}
    per_seed = []; groups = defaultdict(dict); gains = defaultdict(dict)
    for key, val in sorted(values.items()):
        protocol, family, cell, seed, task, route, view = key
        for metric, value in val.items():
            per_seed.append(dict(protocol=protocol, family=family, cell=cell, seed=seed,
                                 test=task, readout=route, view=view, metric=metric, percent=100*value))
            groups[(protocol, family, cell, task, route, view, metric)][seed] = 100*value
            if route != 'raw':
                raw = values[(protocol, family, cell, seed, task, 'raw', view)][metric]
                gains[(protocol, family, cell, task, route, view, metric)][seed] = 100*(value-raw)
    fields = ('protocol', 'family', 'cell', 'test', 'readout', 'view', 'metric')
    summarize = lambda d: [dict(zip(fields, k), **moments([v[s] for s in SEEDS]))
                           for k, v in sorted(d.items())]
    verification = dict(status='VERIFIED_COMPLETE_CONTRACT_SNAPSHOT',
        complete_protocols=list(panels), logical_models=len(obj['rows']), actual_F_seeds=list(SEEDS),
        serialized_overall_CM_recomputed=cm_count, net_cells_recomputed=len(nets),
        population_counts=[dict(protocol=p, test=t, packets=n) for (p, t), n in counts.items()],
        six_scene_mean_within_seed_first=True, sample_SD=True,
        conditional_on_one_fixed_G_per_contract_and_role=True,
        not_full_matrix_delivery=True, target_truth_reads=0, score_feedback_used=False)
    return dict(per_seed=per_seed, summary=summarize(groups), restoration_gain=summarize(gains),
                net_corrections=nets), verification


def run(path, output):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    tables, verification = aggregate(obj)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        write_csv(out/('practical_'+name+'.csv'), rows)
    (out/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lookup = {(r['protocol'], r['family'], r['cell'], r['test'], r['readout'], r['view'], r['metric']): r
              for r in tables['summary']}
    names = dict(riei='RIEI', drift='DRIFT', ibrff='IB-RFF', sdg='SDG', pamf='PAMF')
    labels = dict(cross_date='跨日（源 RX、D0）', cross_rx='跨接收机（目标 RX、D1–D3）',
                  cross_both='跨日且跨接收机（目标 RX、D0）')
    when = datetime.fromtimestamp(obj['observed_at'], timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    text = ['# practical residual：完整接收机划分结果快照', '', f'核查时间：{when}。', '',
        f"总矩阵 627 条预测均已冻结。本次提取时已评分 {obj['scored_rows']}/627 条；仅汇总完全齐备的 "
        + '、'.join(obj['complete_protocols']) + f"，共 {len(obj['rows'])} 个逻辑模型读出。其余契约仍待评分，不混入均值。", '',
        'A＝无信道增强、无 PL；B＝无信道增强、有普通 PL；C＝有 practical residual 增强、无 PL；D＝有增强、有普通 PL。四个对比方法均保留 Full 目标。', '',
        'PAMF A/C 为对应监督配方，关闭 U/EMA；原生 T1 保留原 EMA/选择性半监督机制，不等同于外部普通 PL。'
        'A/C 使用 G-L，B/D/T1 使用 G-LU；因此恢复后跨 PL 组的差异同时包含 G 的源数据权限差异。', '',
        '训练日期沿用 D1–D3。D0 为较早的未见日期，不声明未来日期外推。每格是三 F 种子的均值 ± 样本标准差（%）；'
        '六景先在每个种子内平均，再汇总三种子。G 在每个契约/角色固定，不能将标准差解释为三个独立完整系统的波动。', '']
    def fmt(p, f, c, t, route, view, metric='accuracy'):
        r = lookup[p, f, c, t, route, view, metric]
        return f"{r['mean']:.2f} ± {r['sd']:.2f}"
    for p in obj['complete_protocols']:
        text += [f'## {p}', '']
        for task in TASKS:
            text += [f'### {labels[task]}', '',
                '|方法|组合|raw clean|raw 六景|恢复后 clean|恢复后六景|恢复后六景 Macro-F1|',
                '|---|---|---:|---:|---:|---:|---:|']
            for f, c in VARIANTS:
                route = 'DURA-LU' if c in ('B', 'D', 'native_T1') else 'DURA-L'
                label = 'T1 原生半监督' if c == 'native_T1' else c
                vals = [fmt(p,f,c,task,'raw','clean'), fmt(p,f,c,task,'raw','six_practical_mean'),
                        fmt(p,f,c,task,route,'clean'), fmt(p,f,c,task,route,'six_practical_mean'),
                        fmt(p,f,c,task,route,'six_practical_mean','macro_f1')]
                text.append('|'+ '|'.join([names[f], label, *vals])+'|')
            text += ['']
    text += ['## 核验与边界', '',
        f"已从 {verification['serialized_overall_CM_recomputed']} 份总体混淆矩阵复算 Accuracy/Macro-F1，"
        f"并核对 {verification['net_cells_recomputed']} 个恢复净纠错单元。完整逐 RX/TX/day 结果与训练/祖先成本属于最终全量交付，"
        '本快照不宣称已经完成这些分层复算或最终论文导出。负收益、零 PL 接受率与缺项均保留；评分不回流模型或队列。', '']
    (out/'practical_results.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps(dict(output=str(out), **verification), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.input, args.output)
