"""Rank frozen clean results and measured deployment costs; no inference or selection.

The composite is an explicitly chosen descriptive index, not a universal benchmark.
Ranks use full precision; shared composite scores retain shared competition ranks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from pypdf import PdfReader

from plot_standard_ce_clean import MODEL_COLORS, NAMES

METRICS = [
    ('accuracy_pct', 0, False, 'Clean accuracy\n(%; mean ± SD)\n50% weight'),
    ('p95_ms', 1, True, 'P95 latency\nB1 (ms) ↓\n10% weight'),
    ('end_to_end_throughput_samples_s', 128, False, 'Throughput\nB128 (kIQ/s) ↑\n10% weight'),
    ('gpu_peak_allocated_mib', 1, True, 'GPU peak\nB1 (MiB) ↓\n10% weight'),
    ('cpu_peak_rss_mib', 1, True, 'CPU peak RSS\nB1 (MiB) ↓\n10% weight'),
    ('model_storage_mib', 1, True, 'FP32 storage\n(MiB) ↓\n10% weight'),
]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def competition_ranks(values, lower):
    return {v: 1 + sum(x < value if lower else x > value for x in values.values())
            for v, value in values.items()}


def build_ranking(clean, deployment):
    c = {r['variant']: r for r in clean}
    d = {(r['variant'], int(r['batch_size'])): r for r in deployment}
    variants = set(MODEL_COLORS)
    if set(c) != variants or set(d) != {(v, b) for v in variants for b in (1, 128)}:
        raise ValueError('Expected exactly the 13 residual-CVS/comparison models at B1/B128')
    for v in variants:
        if int(c[v]['actual_seed_count']) != 8:
            raise ValueError(f'Incomplete clean seeds: {v}')
        for b, count in ((1, 2400), (128, 800)):
            if int(d[v, b]['actual_seed_count']) != 8 or int(d[v, b]['request_count']) != count:
                raise ValueError(f'Incomplete deployment requests/seeds: {v}/{b}')
    values, ranks = {}, {}
    for field, batch, lower, _ in METRICS:
        values[field] = {v: float((c[v] if batch == 0 else d[v, batch])[field]) for v in variants}
        if not all(np.isfinite(x) and x > 0 for x in values[field].values()):
            raise ValueError(f'Invalid metric: {field}')
        ranks[field] = competition_ranks(values[field], lower)
    # Integers avoid breaking exact composite ties through binary floating-point noise.
    cost_sum = {v: sum(ranks[f][v] for f, *_ in METRICS[1:]) for v in variants}
    weighted_units = {v: 5 * ranks['accuracy_pct'][v] + cost_sum[v] for v in variants}
    overall = competition_ranks(weighted_units, True)
    order = sorted(variants, key=lambda v: (weighted_units[v], -values['accuracy_pct'][v], v))
    rows = []
    for v in order:
        row = dict(variant=v, network=NAMES[v], overall_rank=overall[v],
                   composite_score=100 * (130 - weighted_units[v]) / 120,
                   accuracy_weight=0.5, each_cost_weight=0.1,
                   actual_seed_count=8, accuracy_sd_pp=float(c[v]['accuracy_sd_pp']),
                   total_parameters=int(c[v]['total_parameters']),
                   accuracy_rank=ranks['accuracy_pct'][v],
                   deployment_mean_rank=cost_sum[v] / 5,
                   weighted_rank_units=weighted_units[v])
        for field, *_ in METRICS:
            row[field] = values[field][v]
            row[field + '_rank'] = ranks[field][v]
        rows.append(row)
    sensitivity = []
    for percent in range(101):
        # Exact integer comparison of w*r_accuracy + (1-w)*mean(r_cost).
        units = {v: percent * 5 * ranks['accuracy_pct'][v] + (100-percent) * cost_sum[v]
                 for v in variants}
        positions = competition_ranks(units, True)
        for v in sorted(variants):
            sensitivity.append(dict(accuracy_weight_pct=percent, variant=v,
                                    overall_rank=positions[v],
                                    composite_score=100*(6500-units[v])/6000))
    # Smallest accuracy weight for CVS to strictly beat every alternative.
    cvs = 'residual_fusion'
    threshold = 0.0
    for v in variants - {cvs}:
        cost_gap = (cost_sum[cvs] - cost_sum[v]) / 5
        if cost_gap > 0:
            performance_gain = ranks['accuracy_pct'][v] - ranks['accuracy_pct'][cvs]
            threshold = max(threshold, cost_gap/(cost_gap+performance_gain))
    return rows, sensitivity, threshold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean', type=Path, required=True)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Preserve previous exports; use a new directory: {out}')
    rows, sensitivity, threshold = build_ranking(
        read_csv(args.clean/'network_summary.csv'),
        read_csv(args.deployment/'deployment_summary.csv'))
    notes = json.loads((args.deployment/'measurement_notes.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    for directory in ('pdf', 'svg', 'png', 'preview', 'source_data'):
        (out/directory).mkdir()
    write_csv(out/'source_data/comprehensive_ranking.csv', rows)
    write_csv(out/'source_data/weight_sensitivity.csv', sensitivity)
    shutil.copyfile(args.clean/'network_summary.csv', out/'source_data/network_summary.csv')
    shutil.copyfile(args.deployment/'deployment_summary.csv', out/'source_data/deployment_summary.csv')
    shutil.copyfile(args.deployment/'measurement_notes.json', out/'source_data/measurement_notes.json')

    # One aligned quantitative grid: row ranks, score bars, actual values and metric ranks.
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial', 'DejaVu Sans'],
                         'font.size':7, 'svg.fonttype':'none', 'pdf.fonttype':42})
    width_mm, height_mm = 183, 145
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width_mm), ylim=(0, height_mm))
    ax.axis('off')
    ax.text(5, 139, 'Clean performance and deployment: comprehensive ranking',
            fontsize=9, weight='bold', va='top')
    ax.text(5, 132, '13 models · 8 seeds each · weighted rank score (higher is better)',
            fontsize=7.2, color='#50565D', va='top')
    ax.text(4.5, 121.5, 'Rank', ha='center', va='center', fontsize=6.6)
    ax.text(12, 121.5, 'Model', va='center', fontsize=7)
    ax.text(67, 121.5, 'Score\n0–100', ha='center', va='center', fontsize=7)
    centers = [88, 106, 124, 142, 159, 176]
    for center, (_, _, _, header) in zip(centers, METRICS):
        ax.text(center, 122, header, ha='center', va='center', fontsize=6.7, linespacing=1.4)
    ax.plot([2, 181], [115, 115], color='#A8ADB3', lw=.65)
    for i, row in enumerate(rows):
        y = 111.2 - 5.75*i
        v = row['variant']
        if v == 'residual_fusion':
            ax.add_patch(Rectangle((2, y-2.85), 179, 5.75, facecolor='#FFF2E8', edgecolor='none'))
        elif i % 2:
            ax.add_patch(Rectangle((2, y-2.85), 179, 5.75, facecolor='#F7F8FA', edgecolor='none'))
        ax.text(4.5, y, str(row['overall_rank']), ha='center', va='center',
                weight='bold' if v == 'residual_fusion' else 'normal', fontsize=7.4)
        ax.scatter([9], [y], c=[MODEL_COLORS[v]], s=17, zorder=3, marker='s')
        ax.text(12, y, NAMES[v], va='center', fontsize=6.9,
                weight='bold' if v == 'residual_fusion' else 'normal')
        ax.add_patch(Rectangle((58, y-1.05), 12.0, 2.1, facecolor='#E2E5E9', edgecolor='none'))
        ax.add_patch(Rectangle((58, y-1.05), 12.0*row['composite_score']/100, 2.1,
                              facecolor=MODEL_COLORS[v], edgecolor='none'))
        ax.text(76, y, f"{row['composite_score']:.1f}", ha='center', va='center', fontsize=7.0)
        for j, (center, (field, *_)) in enumerate(zip(centers, METRICS)):
            rank, value = row[field+'_rank'], row[field]
            shade = .1 + .70*(13-rank)/12
            ax.add_patch(Rectangle((center-8.1, y-2.45), 16.2, 4.9,
                                  facecolor=plt.cm.Blues(shade*.33), edgecolor='none'))
            if j == 0:
                text = f"{value:.2f} ± {row['accuracy_sd_pp']:.2f}"
            elif j == 2:
                text = f'{value/1000:.2f}'
            elif j == 5:
                text = f'{value:.3f}' if value < 1 else f'{value:.2f}'
            else:
                text = f'{value:.2f}' if j == 1 else f'{value:.1f}'
            ax.text(center, y+.65, text, ha='center', va='center', fontsize=6.7,
                    weight='bold' if rank == 1 else 'normal')
            ax.text(center, y-1.25, f'#{rank}', ha='center', va='center', fontsize=6.1,
                    color='#1C5B84' if rank == 1 else '#555E68')
    ax.plot([2, 181], [38.8, 38.8], color='#A8ADB3', lw=.65)
    ax.text(5, 34.8, 'Weight check: CVS rank at accuracy weights 0 / 25 / 50 / 75 / 100%:  #3 / #2 / #1 / #1 / #1',
            fontsize=6.6, va='top')
    ax.text(5, 30.2, f'CVS is uniquely #1 when accuracy weight > {threshold*100:.2f}% (five cost weights remain equal).',
            fontsize=6.6, va='top')
    ax.text(5, 23.8, 'Score = 0.50 × accuracy rank-points + 0.10 × each of five cost rank-points; points = 100 × (13 − rank) / 12.',
            fontsize=6.45, va='top')
    ax.text(5, 19.7, 'Nominal ranks use full-precision means; equal scores share ranks. Rankings are descriptive, without significance tests.',
            fontsize=6.45, va='top')
    ax.text(5, 15.6, 'B1/B128 = batch sizes 1/128. RTX 3090; FP32; CPU IQ → GPU model → CPU labels. P95 pools 2,400 requests/model.',
            fontsize=6.45, va='top')
    ax.text(5, 11.5, 'Memory = maximum of 8 inference peaks: GPU allocated; CPU process RSS (includes runtime). Storage = FP32 state_dict.',
            fontsize=6.45, va='top')
    ax.text(5, 7.4, 'Excludes RF capture, synchronization, disk/network/queue time. Clean split 392005; E200; no claim of field deployment.',
            fontsize=6.45, va='top')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.get_window_extent()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > box.x1+1 or b.y1 > box.y1+1:
            raise ValueError(f'Text outside canvas: {text.get_text()}')
    stem = 'fig09_comprehensive_ranking'
    for extension in ('pdf', 'svg'):
        fig.savefig(out/extension/f'{stem}.{extension}', facecolor='white')
    fig.savefig(out/'png'/f'{stem}.png', dpi=600, facecolor='white')
    fig.savefig(out/'preview'/f'{stem}.png', dpi=190, facecolor='white')
    plt.close(fig)
    pdf = PdfReader(out/'pdf'/f'{stem}.pdf')
    assert len(pdf.pages) == 1 and 'CVS' in pdf.pages[0].extract_text()
    svg = ET.parse(out/'svg'/f'{stem}.svg')
    assert svg.findall('.//{http://www.w3.org/2000/svg}text'), 'SVG must retain editable text'
    with Image.open(out/'png'/f'{stem}.png') as im:
        pixels = im.size
        assert abs(pixels[0]-width_mm/25.4*600) < 2
        assert abs(pixels[1]-height_mm/25.4*600) < 2
    checks = {r['variant']: [s['overall_rank'] for s in sensitivity
                           if s['variant'] == r['variant'] and s['accuracy_weight_pct'] in (0,25,50,75,100)]
              for r in rows}
    assert checks['residual_fusion'] == [3,2,1,1,1]
    caption = '''Comprehensive descriptive ranking of 13 networks trained with pure CE for E200 on the fixed split (392005), each evaluated across eight seeds. The composite assigns 50% to clean accuracy and 10% each to single-sample end-to-end P95 latency, batch-128 end-to-end throughput, single-sample GPU allocated inference peak, single-sample CPU process RSS inference peak and FP32 state_dict storage. Each metric is transformed to ordinal rank-points, 100(13-r)/12; higher points are better. This analyst-defined index is relative to this comparison set, not accuracy expressed as a percentage or a standardized deployment rating. Parameters, MACs/FLOPs and macro-F1 are available separately, but are not added to the index to avoid multiplying correlated size/compute/performance proxies. Equal composite scores retain shared competition ranks; tied rows are displayed in decreasing clean-accuracy order. Accuracy is mean ± sample SD across eight seeds. All individual ranks use full precision, with no significance test or uncertainty inference for ranking. Collapsed Transformer seeds remain included.

Deployment uses eight real frozen checkpoints per model, identical synthetic unit-RMS CPU IQ inputs of shape 2×256, exclusive sequential RTX 3090 inference, FP32 and two CPU threads. Each checkpoint contributes 300 timed B1 requests and 100 timed B128 requests after 30 warmups; network P95 pools 2,400 B1 requests, and throughput is B128 divided by pooled mean request time. A separate 50-call inference window measures memory. GPU peak is PyTorch allocated memory; CPU peak is sampled total process RSS including Python/framework/runtime, not the model's isolated RAM requirement. Memory summary is the maximum across the eight checkpoint peaks. Storage is actual uncompressed FP32 state_dict serialization, excluding optimizer, model code and training history. Timing covers CPU input-buffer construction, H2D, full model forward, finite checks, argmax and CPU-label conversion. It excludes RF acquisition, synchronization, normalization from raw receiver data, disk/network/queue and batch-waiting time. These are software-chain measurements, not receiver or satellite field-deployment measurements. No new inference accuracy experiment, model selection or target feedback was performed.
'''
    (out/'caption_en.md').write_text(caption, encoding='utf-8')
    chinese = f'''# 综合排名口径与使用边界

这是一个明确设定权重的描述性综合指标：准确率占 50%；单样本 P95 时延、B128 吞吐量、B1 GPU allocated 推理峰值、B1 CPU 总进程 RSS 推理峰值、FP32 权重序列化存储各占 10%。单项名次转分为 100 × (13 − 名次) / 12。综合得分为加权平均，分数越高越好；不代表准确率，也不是统一行业标准。

使用名次可以避免不同量纲，但不保留差距幅度。原始数值与单项名次均在图中和 CSV 中保留，使用者应按真实部署约束解释。参数量与存储相关、MAC/FLOPs 与实际部署速度不等价，macro-F1 与准确率相关，因此均不再次计权。相关成本仍可能有联系；这个指数没有消除一切统计相关性。

采用全精度数据计算名次；综合同分共享名次，仅按准确率排序展示同分行。种子间差异与均值差距不做显著性判断；排名不是已证明的显著胜出。准确率为 8 个种子的均值 ± 样本 SD。内存为 8 个 checkpoint 推理峰值的最大值，CPU 含解释器和框架运行时，不等于独立模型 RAM。Transformer 退化种子完整保留。

默认权重下 CVS 综合第 1，得分 82.5；ResNet-18-1D 第 2，75.0；ResNet-50-1D 第 3，68.3。CVS 的准确率、B1 allocated 显存峰值和模型存储单项第 1；B1 P95 时延第 8、B128 吞吐量第 3、B1 CPU 进程 RSS 峰值第 13。这些负项未删除。

权重敏感性：仅改变准确率总权重，剩余权重均分给五项部署指标。CVS 在准确率权重严格高于 {threshold*100:.6f}% 时独占第 1；等于阈值时与 ResNet-18-1D 并列。准确率权重为 0%、25%、50%、75%、100% 时 CVS 分别为第 3、2、1、1、1。完整 0 至 100%（步长 1 个百分点）敏感性在 source_data/weight_sensitivity.csv 中。

本图复用已完成的 clean 评分与部署测量，不新增训练、选模或准确率测试。仅支持固定 split 392005、E200、8 个种子的 clean 比较。软件推理链路不含射频采集、同步、原始输入归一化、磁盘、网络、排队和凑批，不可据此声称已完成真实接收机或卫星部署。

PDF/SVG 保留可编辑文字；PNG 为 600 dpi，尺寸 183 × 145 mm。各网络沿用现有固定区别色。原 8 张论文图及其 Excel 不覆盖。
'''
    (out/'README_zh.md').write_text(chinese, encoding='utf-8')
    manifest = dict(status='VERIFIED', figure=stem, dimensions_mm=[width_mm,height_mm],
                    png_pixels=pixels, models=13, seeds_per_model=8,
                    score_formula='100*(13 - (0.5*r_accuracy + 0.1*sum(r_5costs)))/12',
                    shared_ties=True, significance_test=False,
                    accuracy_weight=0.5, each_cost_weight=0.1,
                    cvs_unique_first_accuracy_weight_threshold=threshold,
                    rank_sensitivity_0_25_50_75_100=checks,
                    sources=['network_summary.csv','deployment_summary.csv','measurement_notes.json'],
                    clean_runs=['20261010-phase1-standard-ce-baselines-manysig-m56-r01',
                                '20261012-phase1-standard-ce-seed-extension-manysig-m52-r01'],
                    deployment_run='20261012-phase1-standard-ce-deployment-m104-r02',
                    deployment_measurement_notes=notes)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED', output=str(out), models=13, seeds=8,
                          ranking=[(r['overall_rank'],r['network'],r['composite_score']) for r in rows]),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
