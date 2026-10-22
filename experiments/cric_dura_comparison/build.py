"""Recompute tables from existing frozen scores. No IQ, model, training or truth access."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from .plan import ANALYSIS_RUN, RUN, SEEDS, VIEWS, make_plan

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/phase1_experiments_ieee_20261020'
SCORES = ROOT / 'automation_reports/CV-SincNet/20261021-phase1-SARD-NoAs-DURA-e50-m6-r02/per_seed_strata.csv'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    out = []
    families = {'riei': 'RIEI', 'drift': 'DRIFT', 'ibrff': 'IB-RFF', 'sdg': 'SDG'}
    for r in read_csv(PAPER / 'source_data/primary_metrics_219x7.csv'):
        if r['group'] not in ('F0', 'F0-pl') or r['family'] not in families:
            continue
        out.append(dict(method=families[r['family']] + (' Full' if r['variant'] == 'full' else ' CE'),
            regime='PL' if r['group'] == 'F0-pl' else 'Supervised',
            seed=int(r['model_seed']), scene=r['view'],
            accuracy_pct=100 * float(r['rx_equal_accuracy']),
            packet_accuracy_pct=100 * float(r['packet_accuracy']),
            macro_f1_pct=100 * float(r['macro_f1']),
            source_run='20261019-phase1-external-dg-concat-pl-m219-r03',
            uncertainty='F model seeds; native family-specific training and sat concat'))
    for r in read_csv(SCORES):
        if r['arm'] != 'NOAS' or r['dimension'] != 'overall':
            continue
        assert r['readout'] in ('raw', 'dura') and int(r['n']) == 168000
        acc = float(r['accuracy_pct'])
        out.append(dict(method='CRIC+DURA' if r['readout'] == 'dura' else 'CRIC raw',
            regime='Native SSL', seed=int(r['seed']), scene=r['scene'],
            accuracy_pct=acc, packet_accuracy_pct=acc, macro_f1_pct=float(r['macro_f1_pct']),
            source_run='20261021-phase1-SARD-NoAs-DURA-e50-m6-r02',
            uncertainty='Three F seeds conditional on one frozen G' if r['readout'] == 'dura' else 'Three F seeds'))
    assert len(out) == 378
    keys = {(r['method'], r['regime'], r['seed'], r['scene']) for r in out}
    assert len(keys) == 378
    pairs = sorted({(r['method'], r['regime']) for r in out})
    assert len(pairs) == 18
    for method, regime in pairs:
        assert {(r['seed'], r['scene']) for r in out if (r['method'], r['regime']) == (method, regime)} == set((s, v) for s in SEEDS for v in VIEWS)
    return out


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['method'], r['regime'], r['scene'])].append(r)
    for method, regime in {(r['method'], r['regime']) for r in rows}:
        for seed in SEEDS:
            six = [r for r in rows if (r['method'], r['regime'], r['seed']) == (method, regime, seed) and r['scene'] != 'clean']
            assert len(six) == 6
            groups[(method, regime, 'satellite_six_mean')].append({k: mean(r[k] for r in six) for k in ('accuracy_pct', 'macro_f1_pct')})
    result = []
    for (method, regime, scene), part in sorted(groups.items()):
        assert len(part) == 3
        result.append(dict(method=method, regime=regime, scene=scene, n_F_seeds=3,
            accuracy_mean_pct=mean(r['accuracy_pct'] for r in part),
            accuracy_sd_pp=stdev(r['accuracy_pct'] for r in part),
            macro_f1_mean_pct=mean(r['macro_f1_pct'] for r in part),
            macro_f1_sd_pp=stdev(r['macro_f1_pct'] for r in part)))
    cric = next(r for r in result if r['method'] == 'CRIC+DURA' and r['scene'] == 'clean')
    assert abs(cric['accuracy_mean_pct'] - 85.14285714285715) < 1e-8
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'local_artifacts/cric_dura_comparison_20261021')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    (args.output / 'prospective_matrix.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save_csv(args.output / 'prospective_F_rows.csv', [{k: r[k] for k in ('row_id', 'protocol_id', 'regime', 'model_seed', 'status', 'optimizer_updates', 'U_use', 'satellite_concat', 'G_source_roles')} for r in plan['rows']])
    rows = collect()
    summary = summarize(rows)
    save_csv(args.output / 'f0_per_seed_scores.csv', rows)
    save_csv(args.output / 'f0_method_summary.csv', summary)
    order = [(f + ' ' + v, g) for g in ('Supervised', 'PL') for f in ('RIEI', 'DRIFT', 'IB-RFF', 'SDG') for v in ('CE', 'Full')]
    order += [('CRIC raw', 'Native SSL'), ('CRIC+DURA', 'Native SSL')]
    scenes = list(VIEWS) + ['satellite_six_mean']
    index = {(r['method'], r['regime'], r['scene']): r for r in summary}
    md = ['# F0：完整 CRIC＋DURA 与四种方法／自身CE的已冻结结果并表', '',
        'VERIFIED / ANALYZED：本次仅复算已有评分；54个模型读出、每个7景，共378格；没有新训练或重新访问truth。', '',
        '各格为Accuracy%均值±三个F模型seed的样本SD（pp）。CRIC+DURA共享一个G。外部监督／PL均含sat拼接；CRIC保持原生33L+17U、EMA/门控、无As。此表比较完整配方，不隔离单个机制。', '',
        '|方法|训练|clean|S-high|S-mid|S-low|U-high|U-mid|U-low|六景均值|',
        '|---|---|' + '---:|' * 8]
    tex = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\caption{F0 accuracy (\%) of frozen complete recipes, mean $\pm$ sample SD over three F seeds. All use the same seven target RX and 168000 packets per view. CRIC+DURA conditions on one frozen G. S/U denote suburban/urban; S6 is the per-seed mean of six satellite views.}',
        r'\label{tab:cric-full-f0}', r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
        r'Method & Training & Clean & S-high & S-mid & S-low & U-high & U-mid & U-low & S6 \\', r'\midrule']
    for method, regime in order:
        part = [index[(method, regime, s)] for s in scenes]
        md.append('|%s|%s|%s|' % (method, regime, '|'.join('%.4f ± %.4f' % (r['accuracy_mean_pct'], r['accuracy_sd_pp']) for r in part)))
        if method == 'CRIC raw':
            tex.append(r'\midrule')
        tex.append('%s & %s & %s \\\\' % (method.replace('+', r'$+$'), regime,
            ' & '.join(r'$%.2f\!\pm\!%.2f$' % (r['accuracy_mean_pct'], r['accuracy_sd_pp']) for r in part)))
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    # Paired deltas use the three seed-level values, never SD subtraction.
    seed_values = {(r['method'], r['regime'], r['seed'], r['scene']): r for r in rows}
    for method, regime in order:
        for seed in SEEDS:
            part = [seed_values[(method, regime, seed, s)] for s in VIEWS[1:]]
            seed_values[(method, regime, seed, 'satellite_six_mean')] = {
                k: mean(r[k] for r in part) for k in ('accuracy_pct', 'macro_f1_pct')}
    comparisons = [(f + ' Full', g, f + ' CE', g, 'Full-minus-own-CE')
        for g in ('Supervised', 'PL') for f in ('RIEI', 'DRIFT', 'IB-RFF', 'SDG')]
    comparisons += [(f + ' ' + v, 'PL', f + ' ' + v, 'Supervised', 'PL-minus-supervised')
        for f in ('RIEI', 'DRIFT', 'IB-RFF', 'SDG') for v in ('CE', 'Full')]
    comparisons += [('CRIC+DURA', 'Native SSL', 'CRIC raw', 'Native SSL', 'G-minus-raw')]
    deltas = []
    md += ['', '## 同 seed 配对差', '',
        '以下为Accuracy差的均值±样本SD（pp），不是两个SD相减。负值完整保留。', '',
        '|比较|训练|clean|六景均值|low urban|', '|---|---|---:|---:|---:|']
    for a, ar, b, br, kind in comparisons:
        cells = []
        for scene in scenes:
            current = []
            for seed in SEEDS:
                av, bv = seed_values[(a, ar, seed, scene)], seed_values[(b, br, seed, scene)]
                acc = av['accuracy_pct'] - bv['accuracy_pct']
                deltas.append(dict(comparison=kind, method_A=a, regime_A=ar, method_B=b, regime_B=br,
                    seed=seed, scene=scene, accuracy_delta_pp=acc,
                    macro_f1_delta_pp=av['macro_f1_pct'] - bv['macro_f1_pct']))
                current.append(acc)
            if scene in ('clean', 'satellite_six_mean', 'practical_low_urban'):
                cells.append((scene, '%+.4f ± %.4f' % (mean(current), stdev(current))))
        part = dict(cells)
        md.append('|%s − %s|%s/%s|%s|%s|%s|' % (a, b, ar, br, part['clean'],
            part['satellite_six_mean'], part['practical_low_urban']))
    assert len(deltas) == 408
    save_csv(args.output / 'f0_paired_deltas.csv', deltas)
    save_csv(PAPER / 'source_data/cric_f0_paired_deltas.csv', deltas)
    folder = ROOT / 'automation_reports/CV-SincNet' / ANALYSIS_RUN
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'final_analysis.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (PAPER / 'tables/cric_full_f0.tex').write_text('\n'.join(tex), encoding='utf-8')
    # Copy small sanitized aggregate sources into the paper export, not Git.
    save_csv(PAPER / 'source_data/cric_f0_per_seed_54x7.csv', rows)
    save_csv(PAPER / 'source_data/cric_f0_summary.csv', summary)
    evidence = dict(status='VERIFIED', operation='existing_score_aggregation_only', cells=378,
        method_recipe_readouts=18, F_model_readouts=54, seeds=list(SEEDS), views=list(VIEWS),
        new_training=0, query_truth_reads=0, prospective_status='PLANNED_NOT_LAUNCHED',
        prospective_F_instances=153, prospective_new_F_training=150,
        prospective_G_pipelines=36, prospective_new_G_pipelines=35)
    (args.output / 'build_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
