"""Summarize the complete frozen SSL/DURA matrix; never reads test truth."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METHODS = ('F20', 'F21', 'PSM', 'RXPSM')
SEEDS = (2026092701, 2026092702, 2026092703)
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
READOUTS = ('raw', 'DURA-LU')
RUN = '20261023-phase1-PAMF-SSL-DURA-eval-m6-r01'


def summarize(data):
    if data['state']['status'] != 'ALL_SCORED':
        raise ValueError('Incomplete six-model supplemental matrix')
    indexed = {(r['method'], r['model_seed']): r['score'] for r in data['scores']}
    if len(data['scores']) != 12 or set(indexed) != {(m, s) for m in METHODS for s in SEEDS}:
        raise ValueError('Exactly four methods and three matched seeds required')
    per_seed, strata, paired = [], [], []
    for (method, seed), score in indexed.items():
        if score['status'] != 'SCORED' or score['model_seed'] != seed or score['truth_last'] is not True:
            raise ValueError('Frozen score binding/truth-last differs')
        expected_run = RUN if method in ('PSM', 'RXPSM') else '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
        if score['run_id'] != expected_run or score['row_id'] != method + '-s' + str(seed):
            raise ValueError('Score belongs to another frozen model')
        values = {(r['readout'], r['view']): r for r in score['results']
                  if r['test'] == 'main' and r['dimension'] == 'overall'}
        if set(values) != {(r, v) for r in READOUTS for v in VIEWS}:
            raise ValueError('Incomplete raw/DURA seven-view scores')
        for readout in READOUTS:
            for view in VIEWS:
                metric = values[readout, view]
                if metric['count'] != 168000:
                    raise ValueError('Different target population')
                correct = sum(metric['confusion'][i][i] for i in range(6))
                accuracy = 100 * correct / metric['count']
                if abs(accuracy - 100 * metric['accuracy']) > 1e-9:
                    raise ValueError('Confusion/accuracy count mismatch')
                per_seed.append(dict(method=method, model_seed=seed, readout=readout,
                    view=view, count=metric['count'], correct=correct,
                    accuracy_percent=accuracy, macro_f1_percent=100 * metric['macro_f1']))
        for row in score['results']:
            strata.append(dict(method=method, model_seed=seed, **row))
        for row in score['paired_raw']:
            paired.append(dict(method=method, model_seed=seed, **row))
            if row['dimension'] == 'overall':
                raw = values['raw', row['view']]['confusion']
                restored = values['DURA-LU', row['view']]['confusion']
                difference = sum(restored[i][i] - raw[i][i] for i in range(6))
                if row['net'] != row['repairs'] - row['damage'] or row['net'] != difference:
                    raise ValueError('Paired corrections differ from correct-count change')
    indexed = {(r['method'], r['model_seed'], r['readout'], r['view']): r for r in per_seed}
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for readout in READOUTS:
            cells = {}
            for view in (*VIEWS, 'six_mean'):
                values = [indexed[method, seed, readout, view]['accuracy_percent'] if view != 'six_mean'
                          else mean(indexed[method, seed, readout, v]['accuracy_percent'] for v in VIEWS[1:])
                          for seed in SEEDS]
                cells[view] = dict(mean=mean(values), sample_SD=stdev(values), per_seed=values)
            summary[method][readout] = cells
    comparisons = []
    for method in ('PSM', 'RXPSM'):
        for baseline in ('F20', 'F21'):
            for readout in READOUTS:
                for view in (*VIEWS, 'six_mean'):
                    a = summary[method][readout][view]['per_seed']
                    b = summary[baseline][readout][view]['per_seed']
                    differences = [x - y for x, y in zip(a, b)]
                    comparisons.append(dict(method=method, baseline=baseline, readout=readout,
                        view=view, paired_mean_pp=mean(differences), sample_SD_pp=stdev(differences),
                        paired_seed_pp=differences, positive_seeds=sum(v > 0 for v in differences)))
    return dict(status='ANALYZED', run_id=RUN, methods=summary, paired_comparisons=comparisons), per_seed, strata, paired


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(summary):
    methods = summary['methods']
    text = ['# PAMF 半监督模型接入 DURA v1.0 的目标测试结果', '',
        '同一 F0 目标测试，每景 168000 条物理记录，三个固定模型种子。全部采用 checkpoint 内原默认头；一个公共冻结 DURA v1.0，SD 仅表示 F 的种子变化。单位为 %，差值为百分点。', '',
        '| 训练方式 | raw clean | raw 六景均值 | DURA 六景均值 | DURA 对六景增益 |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        values = methods[method]
        clean = values['raw']['clean']['mean']
        raw = values['raw']['six_mean']['mean']
        restored = values['DURA-LU']['six_mean']['mean']
        text.append(f'| {method} | {clean:.6f} | {raw:.6f} | {restored:.6f} | {restored-raw:+.6f} |')
    text += ['', 'DURA 六景的比较须使用相同 DURA 的监督／普通 PL 对照；此前 raw 结果和原目标判定保留。本补测不训练或更换任何 F、G、分类头。', '',
        '| DURA 输入视图 | F20 监督 | F21 普通 PL | PSM | RXPSM |',
        '|---|---:|---:|---:|---:|']
    for view in (*VIEWS, 'six_mean'):
        cells = [methods[method]['DURA-LU'][view] for method in METHODS]
        text.append('| ' + view + ' | ' + ' | '.join(f"{c['mean']:.6f} ± {c['sample_SD']:.6f}" for c in cells) + ' |')
    text += ['', '## 相同 DURA 下的半监督配对差', '']
    for method in ('PSM', 'RXPSM'):
        for baseline in ('F20', 'F21'):
            raw = methods[method]['raw']['clean']['mean'] - methods[baseline]['raw']['clean']['mean']
            clean = methods[method]['DURA-LU']['clean']['mean'] - methods[baseline]['DURA-LU']['clean']['mean']
            satellite = methods[method]['DURA-LU']['six_mean']['mean'] - methods[baseline]['DURA-LU']['six_mean']['mean']
            text.append(f'{method} 相对 {baseline}：raw clean {raw:+.6f} pp；DURA clean {clean:+.6f} pp；DURA 六景均值 {satellite:+.6f} pp。')
            text.append('')
    text += ['全部补测行七景预测完成后由独立 CPU truth-last 评分；F20／F21 和 raw SSL 决策复用原已完成产物，没有重复训练或选择性重跑。目标基准此前已暴露，此次是用户指定冻结组合的补充评估。F20 指识别器监督训练；共同 G 使用合法源 L/U IQ，整个组合不声明为仅 L 系统。逐种子、RX／TX／day 和净纠错见同目录 CSV。', '']
    return '\n'.join(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary, per_seed, strata, paired = summarize(json.loads(Path(args.input).read_text(encoding='utf-8')))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'target_results.md').exists():
        raise FileExistsError('Preserve previous result artifacts')
    (output / 'target_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'target_results.md').write_text(render(summary), encoding='utf-8')
    write_csv(output / 'target_per_seed.csv', per_seed)
    write_csv(output / 'target_strata.csv', strata)
    write_csv(output / 'target_net_corrections.csv', paired)
    print(render(summary))
