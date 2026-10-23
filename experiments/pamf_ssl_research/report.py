"""Matched, three-seed target-test tables from already fixed truth-last scores."""
import argparse
import csv
import math
import statistics
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .analyze import summarize, declared_methods
from .plan import SEEDS, VIEWS


def cells(value):
    result = {}
    for r in value['results']:
        if r['test'] != 'main' or r['readout'] != 'raw':
            continue
        key = (r['view'], r['dimension'], r['stratum'])
        cm, n = r['confusion'], r['count']
        if (key in result or r['view'] not in VIEWS or type(n) is not int or n <= 0
                or len(cm) != 6 or any(len(row) != 6 for row in cm)
                or any(type(v) is not int or v < 0 for row in cm for v in row)
                or sum(sum(row) for row in cm) != n):
            raise ValueError('Invalid or duplicate target stratum counts')
        correct = sum(cm[i][i] for i in range(6))
        if not math.isfinite(r['accuracy']) or abs(r['accuracy'] - correct / n) > 1e-12:
            raise ValueError('Stratum accuracy differs from integer counts')
        result[key] = (correct, n)
    for dimension, stratum in {(k[1], k[2]) for k in result}:
        keys = [(view, dimension, stratum) for view in VIEWS]
        if any(k not in result for k in keys):
            raise ValueError('Every target stratum requires all seven views')
        selected = [result[k] for k in keys[1:]]
        if len({n for _, n in selected}) != 1:
            raise ValueError('Six-view denominators differ within target stratum')
        result['six_mean', dimension, stratum] = (sum(c for c, _ in selected), sum(n for _, n in selected))
    return result


def report(rt, output):
    if read(Path(rt['run_root']) / 'state.json')['status'] != 'ALL_SCORED':
        raise ValueError('All declared target-test rows must finish before reporting')
    methods = declared_methods(rt)
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserve existing report output')
    # The existing analyzer checks source exclusion, exact target coverage and identities.
    summarize(rt, out / 'analysis.json')
    analysis = read(out / 'analysis.json')
    data = {}
    for seed in SEEDS:
        for arm, folder in rt['baseline_scores'][str(seed)].items():
            data[arm, seed] = cells(read(Path(folder) / 'scores.json'))
    for row in rt['rows']:
        c = read(row['score_config'])
        data[row['method'], row['model_seed']] = cells(read(Path(c['output_root']) / 'scores.json'))
    keys = set(data['F20', SEEDS[0]])
    if any(set(value) != keys for value in data.values()):
        raise ValueError('Matched target stratum coverage differs across methods/seeds')
    rows = []
    for view, dimension, stratum in sorted(keys):
        key = (view, dimension, stratum)
        denominators = {value[key][1] for value in data.values()}
        if len(denominators) != 1:
            raise ValueError('Matched target stratum denominators differ across methods/seeds')
        n = denominators.pop()
        for method in ('F20', 'F21') + methods:
            correct = [data[method, seed][key][0] for seed in SEEDS]
            accuracy = [100. * c / n for c in correct]
            r = dict(method=method, view=view, dimension=dimension, stratum=stratum,
                     count_per_seed=n, correct_total=sum(correct),
                     mean_accuracy_percent=100. * sum(correct) / (3 * n),
                     sample_sd_percent=statistics.stdev(accuracy))
            for seed, c in zip(SEEDS, correct):
                r['correct_s' + str(seed)] = c
            for arm in ('F20', 'F21'):
                net = [data[method, seed][key][0] - data[arm, seed][key][0] for seed in SEEDS]
                delta = [100. * x / n for x in net]
                r['delta_' + arm + '_pp'] = 100. * sum(net) / (3 * n)
                r['delta_' + arm + '_sample_sd_pp'] = statistics.stdev(delta)
                r['net_correct_' + arm] = sum(net)
                r['positive_seeds_' + arm] = sum(x > 0 for x in net)
            rows.append(r)
    write(out / 'strata.json', dict(status='ANALYZED', model_seeds=list(SEEDS),
        population='Fixed unseen-receiver F0 main target test only',
        truth_last=True, target_feedback_for_training=False, rows=rows))
    with (out / 'strata.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {(r['method'], r['view']): r for r in analysis['summary']}
    text = ['# PAMF 目标测试集结果', '',
        '口径：同一固定 F0 目标测试集，每视图 168000 条物理样本；raw F 默认头，三个模型种子。表内为准确率均值 ± 模型种子样本 SD，单位为 %。源 V 不计入验收。', '',
        '| 视图 | F20 监督 | F21 普通 PL | ' + ' | '.join(methods) + ' |',
        '|---|' + '---:|' * (2 + len(methods))]
    for view in (*VIEWS, 'six_mean'):
        values = [summary[method, view] for method in ('F20', 'F21') + methods]
        text.append('| ' + view + ' | ' + ' | '.join(f"{r['mean']:.6f} ± {r['sd']:.6f}" for r in values) + ' |')
    text += ['', '目标要求同一方法相对两个对照：clean 至少 +1 个百分点，六景均值严格超过 +1.5 个百分点。验收按整数正确计数计算。', '']
    for verdict in analysis['verdicts']:
        text.append(verdict['method'] + '：' + ('达到目标。' if verdict['goal_met'] else '未达到目标。'))
        for arm, values in verdict['paired_differences_pp'].items():
            text.append(f"相对 {arm}，clean {values['clean']['mean']:+.6f} pp，六景 {values['six_mean']['mean']:+.6f} pp；"
                f"正收益种子数分别为 {values['clean']['positive_seeds']}/3、{values['six_mean']['positive_seeds']}/3。")
        text.append('')
    text += ['逐接收机、TX、日期及接收机×日期的相同口径结果见 strata.csv/strata.json；所有种子保留，不按目标成绩删选。', '']
    (out / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    done = dict(status='ANALYZED', goal_met=analysis['goal_met'], stratum_rows=len(rows), output=str(out))
    write(out / 'complete.json', done)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(report(read(args.config), args.output))
