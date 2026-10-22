"""Manuscript figures from frozen scores; no training, inference, or truth access.

python plot_pamf_paper_figures.py --source SOURCE_DATA --output NEW_DIRECTORY --dpi 4800
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from paper_publication_style import (WIDTH_MM, FONT, COLORS, INK, MUTED,
    CMAP, apply_style, canvas, color, contrast, describe, grid, pale)

SEEDS3 = tuple(range(2026092701, 2026092704))
SEEDS8 = tuple(range(2026092701, 2026092709))
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')
SCENES = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SCENE_SHORT = ('Clean', 'Suburban\nhigh', 'Suburban\nmid', 'Suburban\nlow',
               'Urban\nhigh', 'Urban\nmid', 'Urban\nlow')
SCENE_NAMES = ('Clean', 'Suburban high', 'Suburban mid', 'Suburban low',
               'Urban high', 'Urban mid', 'Urban low')
NAMES = dict(riei='RIEI', drift='DRIFT', ibrff='IB-RFF', sdg='SDG', pamf='PAMF',
    resnet18='ResNet-18-1D', resnet34='ResNet-34-1D', resnet50='ResNet-50-1D',
    densenet121='DenseNet-121-1D', vgg16_bn='VGG-16-BN-1D', convnext_tiny='ConvNeXt-Tiny-1D',
    bilstm3='BiLSTM', transformer6='Transformer', resnet18_2d='ResNet-18-2D',
    resnet50_2d='ResNet-50-2D', cvresnet18_1d='Complex ResNet-18-1D', cvresnet18_2d='Complex ResNet-18-2D')
BASELINES = tuple(k for k in NAMES if k not in (*FAMILIES, 'pamf'))
NETWORKS = (*BASELINES, 'pamf')
PAMF_RUN = '20261021-phase1-T1-DURA-comparison-m153-r01'
G_NAMES = {'default': 'DURA', 'coarse_only': 'Coarse only', 'prior_no_CG': 'w/o CG',
    'no_image_action': 'w/o image action', 'no_innovation_input': 'w/o residual input',
    'copy_no_KD': 'Copy, w/o distillation', 'random_with_KD': 'Random + distillation',
    'random_no_KD': 'Random, w/o distillation'}
ROUTES = ('T0_native_GF', 'T1_tensor_GF', 'T2_reuse_GF', 'T3_graph_GF', 'T4_graph_reuse_GF')
ROUTE_NAMES = dict(zip(ROUTES, ('Native', 'Tensor', 'Reuse', 'Graph', 'Graph + reuse')))
SYSTEM_CASES = tuple(f + '_full' for f in FAMILIES) + ('T4_graph_reuse_GF',)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prepare(data):
    if data.get('schema') in ('pamf_manuscript_figures_v2', 'pamf_manuscript_figures_v3'): return data
    if data.get('schema') != 'pamf_unified_figures_v1': raise ValueError('Unsupported scored snapshot')
    # Omit superseded PAMF models from the new deliverable, retaining original evidence elsewhere.
    baselines = [r for r in data['history'] if r['variant'] in BASELINES]
    return dict(schema='pamf_manuscript_figures_v2', snapshot_hkt=data['snapshot_hkt'],
        classes=data['classes'], baselines=baselines,
        parameters={k: data['parameters'][k] for k in BASELINES},
        baseline_summary=[r for r in data['historical_summary'] if r['variant'] in BASELINES],
        deployment=[r for r in data['deployment'] if r['variant'] in BASELINES],
        external=data['external'], t1=data['t1'], g=data['g_ablation'],
        g_summary=data['g_summary'],
        latency=[r for r in data['latency'] if r['case'] in SYSTEM_CASES
                 or (r['case'] in ROUTES and int(r['batch']) == 1)])


def include_pamf(data, score_folder, profile_path):
    if data['schema'] == 'pamf_manuscript_figures_v3': return data
    if score_folder is None or profile_path is None:
        raise ValueError('Current PAMF scored rows and verified parameter profile are required')
    rows = []
    for seed in SEEDS3:
        frozen = load(score_folder / (str(seed) + '.json'))
        complete, scores = frozen['complete'], frozen['scores']
        if (complete['status'] != 'SCORED_COMPLETE' or scores['status'] != 'SCORED'
                or complete['run_id'] != PAMF_RUN or scores['run_id'] != PAMF_RUN
                or int(scores['model_seed']) != seed
                or scores['row_id'] != 'F0-native_cric-s' + str(seed)):
            raise ValueError('PAMF row is not the frozen current T1-E50 comparison')
        results = [r for r in scores['results'] if r['test'] == 'main'
                   and r['readout'] == 'DURA-LU' and r['view'] == 'clean'
                   and r['dimension'] in ('overall', 'receiver', 'transmitter')]
        rows.append(dict(variant='pamf', seed=seed, run_id=PAMF_RUN,
            readout='DURA-LU', view='clean', results=results))
    profile = load(profile_path)
    if (profile['status'] != 'VERIFIED' or profile['F_registered_parameters'] != 232993
            or profile['G_registered_parameters'] != 266951):
        raise ValueError('Current T1-E50/DURA parameter profile changed')
    data = dict(data, schema='pamf_manuscript_figures_v3', pamf=rows,
        parameters=dict(data['parameters'], pamf=499944),
        pamf_definition=dict(recognizer='T1-E50', schedule='13L+37U', restorer='DURA v1.0',
            F_parameters=232993, G_parameters=266951, total_parameters=499944,
            F_model_seeds=list(SEEDS3), shared_fixed_G=True,
            scored_run=PAMF_RUN, readout='DURA-LU', view='clean'),
        pamf_resource_gaps={
            'p95_ms': 'No matched CPU-input-to-returned-label benchmark.',
            'end_to_end_throughput_samples_s': 'Only a different GPU-wall timing boundary is available.',
            'gpu_peak_allocated_mib': 'Available baseline includes all benchmark models, graph pools and a 13 GiB reservation.',
            'cpu_peak_rss_mib': 'No isolated process RSS measurement.',
            'model_storage_mib': 'No matching serialized FP32 state-dictionary measurement; parameter bytes are not file size.'})
    return data


def matrix(rows, fields, value, keys):
    found = {}
    for row in rows:
        a = tuple(row[k] for k in fields); key = a[0] if len(a) == 1 else a
        index = key, int(row['seed']), row.get('scene', row.get('view'))
        if index in found or key not in keys: raise ValueError('Duplicate/unexpected score')
        v = float(row[value])
        if not math.isfinite(v) or not 0 <= v <= 100 or int(row['count']) != 168000:
            raise ValueError('Invalid score units or denominator')
        found[index] = v
    expected = {(k, s, scene) for k in keys for s in SEEDS3 for scene in SCENES}
    if set(found) != expected: raise ValueError('Incomplete score matrix')
    return {k: np.array([[found[k, s, scene] for scene in SCENES] for s in SEEDS3]) for k in keys}


def validate(data):
    if data['schema'] != 'pamf_manuscript_figures_v3': raise ValueError('Current PAMF must be included')
    keys = [(f, r) for f in FAMILIES for r in ('full', 'full_pl')] + [('pamf', 'pamf')]
    external = matrix(data['external'], ('family', 'regime'), 'accuracy_pct', keys)
    t1 = matrix(data['t1'], ('readout',), 'accuracy', ('raw', 'DURA-LU'))
    if not np.allclose(external['pamf', 'pamf'], t1['DURA-LU'], atol=1e-10, rtol=0):
        raise ValueError('PAMF control mismatch')
    arms = tuple(dict.fromkeys(r['arm'] for r in data['g']))
    g = matrix(data['g'], ('arm',), 'accuracy', arms)
    if set(g) != set(G_NAMES) | {'raw'}: raise ValueError('G panel changed')
    if not (np.allclose(g['raw'], t1['raw'], atol=1e-10, rtol=0)
            and np.allclose(g['default'], t1['DURA-LU'], atol=1e-10, rtol=0)):
        raise ValueError('Restoration controls mismatch')
    index = {}
    if len(data['pamf']) != 3 or data['parameters']['pamf'] != 499944:
        raise ValueError('Expected the current three-seed PAMF and complete F+G parameters')
    for row in data['baselines'] + data['pamf']:
        k, seed = row['variant'], int(row['seed'])
        if k not in NETWORKS or seed not in (SEEDS3 if k == 'pamf' else SEEDS8):
            raise ValueError('Excluded/unknown network')
        for score in row['results']:
            dim, stratum = score['dimension'], score['stratum']
            key = k, seed, dim, stratum
            if key in index: raise ValueError('Duplicated baseline score')
            cm = np.array(score['confusion'], dtype=np.int64)
            expected = {'overall': 168000, 'receiver': 24000, 'transmitter': 28000}[dim]
            if cm.shape != (6, 6) or cm.sum() != expected: raise ValueError('Baseline count mismatch')
            if abs(100 * cm.trace() / cm.sum() - 100 * score['accuracy']) > 1e-8:
                raise ValueError('Confusion/accuracy mismatch')
            denom = cm.sum(axis=0) + cm.sum(axis=1)
            f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(6), where=denom != 0).mean()
            if abs(f1 - score['macro_f1']) > 1e-10:
                raise ValueError('Confusion/Macro-F1 mismatch')
            index[key] = score
    if len(data['baselines']) != 96: raise ValueError('Expected 12 baselines with eight seeds each')
    def values(k, field='accuracy', dim='overall', stratum='ALL'):
        return np.array([100 * index[k, s, dim, stratum][field]
                         for s in (SEEDS3 if k == 'pamf' else SEEDS8)])
    receivers = {r['stratum'] for r in data['baselines'][0]['results'] if r['dimension'] == 'receiver'}
    for seed in SEEDS3:
        pamf_results = data['pamf'][SEEDS3.index(seed)]['results']
        if ({r['stratum'] for r in pamf_results if r['dimension'] == 'receiver'} != receivers
                or len(pamf_results) != 14):
            raise ValueError('PAMF receiver/overall score panel is incomplete')
        cm = sum((np.array(index['pamf', seed, 'receiver', rx]['confusion']) for rx in receivers),
                 start=np.zeros((6, 6), dtype=np.int64))
        if not np.array_equal(cm, index['pamf', seed, 'overall', 'ALL']['confusion']):
            raise ValueError('PAMF receiver scores do not sum to the overall confusion matrix')
    for field in ('accuracy', 'macro_f1'):
        expected = [float(next(r[field] for r in data['t1'] if int(r['seed']) == s
                              and r['view'] == 'clean' and r['readout'] == 'DURA-LU')) for s in SEEDS3]
        if not np.allclose(values('pamf', field), expected, atol=1e-10, rtol=0):
            raise ValueError('PAMF network figure disagrees with the published T1+DURA control')
    for r in data['baseline_summary']:
        if not np.allclose(describe(values(r['variant'])),
            [float(r['accuracy_pct']), float(r['accuracy_sd_pp'])], atol=1e-8, rtol=0):
            raise ValueError('Published baseline summary mismatch')
    timing = {(r['case'], int(r['batch'])): r for r in data['latency']}
    if len(timing) != 29: raise ValueError('Incomplete timing subset')
    for r in timing.values():
        median, batch = float(r['median_ms']), int(r['batch'])
        if (not 0 < median <= float(r['P95_ms']) or int(r['rounds']) != 5
                or abs(float(r['throughput_IQ_s']) - 1000 * batch / median) > 1e-7):
            raise ValueError('Timing ratio mismatch')
    deployed = {(r['variant'], int(r['batch_size'])): r for r in data['deployment']}
    if set(deployed) != {(k, b) for k in BASELINES for b in (1, 128)}:
        raise ValueError('Deployment panel changed')
    return external, t1, g, index, values, timing, deployed


def draw_bars(external, scene=None):
    fig, ax = canvas('', 104, (.105, .22, .85, .66))
    keys = (*FAMILIES, 'pamf'); positions = [0, 1, 2, 3, 4.18]
    for family, x in zip(keys, positions):
        for regime in (('pamf',) if family == 'pamf' else ('full', 'full_pl')):
            a = external[family, regime]
            vals = a[:, 1:].mean(axis=1) if scene is None else a[:, scene]
            m, sd = describe(vals); c = COLORS[family]
            dx = 0 if family == 'pamf' else -.17 if regime == 'full' else .17
            ax.bar(x + dx, m, width=.285, color=pale(c) if regime == 'full' else c,
                edgecolor=c, linewidth=.75, hatch='//' if regime == 'full' else None,
                yerr=sd, error_kw=dict(ecolor=INK, elinewidth=.75, capsize=2), zorder=3)
            ax.text(x + dx, m + sd + 1.8, f'{m:.2f}', ha='center', va='bottom', fontsize=8, color=c)
    ax.set(xlim=(-.58, 4.75), ylim=(0, 100), yticks=range(0, 101, 20),
        xticks=positions, xticklabels=[NAMES[k] for k in keys], ylabel='Accuracy (%)')
    ax.tick_params(axis='x', length=0, pad=8); grid(ax)
    for tick, k in zip(ax.get_xticklabels(), keys): tick.set_color(COLORS[k])
    fig.legend(handles=[Patch(fc=pale(MUTED), ec=MUTED, hatch='//', label='Full'),
        Patch(fc=MUTED, label='Full + PL'), Patch(fc=COLORS['pamf'], label='PAMF')],
        loc='center', bbox_to_anchor=(.53, .075), ncol=3, columnspacing=2.3)
    return fig


def forest(labels, samples, colors, xlabel, limits=(0, 100), highlights=(), height=None):
    n = len(labels); height = height or max(87, 26 + n * 7.5)
    lo = min(min(np.min(a), describe(a)[0] - describe(a)[1]) for a in samples)
    hi = max(max(np.max(a), describe(a)[0] + describe(a)[1]) for a in samples)
    limits = min(limits[0], math.floor(lo)), max(limits[1], math.ceil(hi))
    fig, ax = canvas('', height, (.315, .145, .49, .735))
    ax.set(yticks=range(n), yticklabels=labels, ylim=(n - .45, -.6), xlim=limits, xlabel=xlabel)
    ax.tick_params(axis='y', length=0, pad=7); ax.spines['left'].set_visible(False); grid(ax, 'x')
    if limits[0] < 0 < limits[1]: ax.axvline(0, color=MUTED, lw=.75, ls=(0, (3, 3)))
    for i, (a, c) in enumerate(zip(samples, colors)):
        m, sd = describe(a)
        if i in highlights: ax.axhspan(i - .42, i + .42, color=pale(c, .065))
        ax.scatter(a, i + np.linspace(-.15, .15, len(a)), color=c, s=9, alpha=.32, linewidths=0, clip_on=False)
        ax.errorbar(m, i, xerr=sd, color=c, fmt='D', mfc=c if i in highlights else 'white',
                    ms=4.7, capsize=2, lw=.9, clip_on=False)
        ax.text(1.055, i, f'{m:+.2f} ± {sd:.2f}' if limits[0] < 0 else f'{m:.2f} ± {sd:.2f}',
            transform=ax.get_yaxis_transform(), va='center', fontsize=8.3, color=c)
        ax.get_yticklabels()[i].set_color(c)
    return fig


def heatmap(means, labels, columns, sds=None, row_colors=None, height=120, confusion=False):
    fig, ax = canvas('', height, (.28, .225, .67, .655))
    image = ax.pcolormesh(np.arange(means.shape[1] + 1) - .5,
        np.arange(means.shape[0] + 1) - .5, means, cmap=CMAP, vmin=0, vmax=100,
        shading='flat', edgecolors='white', linewidth=.7, rasterized=False)
    ax.set(xlim=(-.5, means.shape[1] - .5), ylim=(means.shape[0] - .5, -.5),
        xticks=range(len(columns)), xticklabels=columns, yticks=range(len(labels)), yticklabels=labels)
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(columns)):
            text = f'{means[i, j]:.1f}'
            if sds is not None: text += f'\n± {sds[i, j]:.1f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8.0 if sds is not None else 8.5,
                    color=contrast(CMAP(means[i, j] / 100)), linespacing=1.3)
    if row_colors:
        for tick, c in zip(ax.get_yticklabels(), row_colors): tick.set_color(c)
    cax = fig.add_axes((.35, .080, .53, .016), label='colorbar')
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=range(0, 101, 20))
    cb.outline.set_visible(False); cb.solids.set_rasterized(False)
    cb.set_label('Row-normalized frequency (%)' if confusion else 'Accuracy (%)', labelpad=3, fontsize=8)
    cb.ax.tick_params(length=2, pad=2, labelsize=7.7)
    if confusion:
        ax.set_xlabel('Predicted transmitter', labelpad=7)
        ax.set_ylabel('True transmitter', labelpad=8)
    return fig


def scatter_balance(external, regime):
    # Separate Full and PL conditions instead of shifting nearly coincident data points.
    fig, ax = canvas('', 106, (.115, .16, .83, .72))
    offsets = {'riei': (6, -13), 'drift': (6, 8), 'ibrff': (7, -13), 'sdg': (6, 8), 'pamf': (-40, 8)}
    for family in (*FAMILIES, 'pamf'):
        a = external[family, 'pamf' if family == 'pamf' else regime]
        x, sx = describe(a[:, 0]); y, sy = describe(a[:, 1:].mean(axis=1))
        c = COLORS[family]
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt='D' if family == 'pamf' else 'o',
            color=c, mfc=c if family == 'pamf' else 'white', ms=5, lw=.9, capsize=2.2)
        ax.annotate(NAMES[family], (x, y), xytext=offsets[family], textcoords='offset points',
                    color=c, fontsize=9)
    ax.set(xlim=(42, 90), ylim=(8, 85), xticks=[45, 55, 65, 75, 85],
        yticks=[10, 25, 40, 55, 70, 85], xlabel='Clean accuracy (%)', ylabel='Mean channel accuracy (%)')
    grid(ax, 'both')
    return fig


def ranked_bars(items, unit, metric, lower=True, digits=2, panel=''):
    missing = [r for r in items if r['value'] is None]
    items = sorted([r for r in items if r['value'] is not None],
                   key=lambda r: (r['value'] if lower else -r['value'], r['label']))
    if not items or any(not math.isfinite(r['value']) or r['value'] <= 0 for r in items):
        raise ValueError('Resource values must be finite and positive')
    best = items[0]['value']
    normalized = [100 * r['value'] / best for r in items]
    n = len(items) + len(missing); height = max(79, 29 + 7.6 * n)
    fig, ax = canvas('', height, (.315, .15, .63, .73))
    extent = max(normalized)
    ax.set(yticks=range(n), yticklabels=[r['label'] for r in items + missing],
        ylim=(n - .4, -.6), xlim=(0, extent * 1.25),
        xlabel=f'{metric} / {"minimum" if lower else "maximum"} (%)')
    ticks = MaxNLocator(nbins=5).tick_values(0, extent * 1.25)
    ax.set_xticks([v for v in ticks if 0 <= v <= extent * 1.25])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=8); grid(ax, 'x')
    records = []
    for i, (item, ratio) in enumerate(zip(items, normalized)):
        c = item['color']
        ax.barh(i, ratio, height=.53, color=c, edgecolor=c, linewidth=.4, zorder=3)
        ax.text(ratio + extent * .018, i, f'{item["value"]:.{digits}f} {unit}',
            ha='left', va='center', fontsize=8.4, color=INK)
        ax.get_yticklabels()[i].set_color(c)
        records.append(dict(panel=panel, rank=i + 1, model=item['label'], metric=metric, value=item['value'],
            unit=unit, lower_is_better=lower, relative_to_best_pct=ratio, measurement_status='measured'))
    if missing:
        ax.axhline(len(items) - .5, color=MUTED, lw=.55, ls=(0, (3, 3)))
    for i, item in enumerate(missing, len(items)):
        ax.get_yticklabels()[i].set_color(item['color'])
        ax.text(extent * .018, i, 'Not measured', color=item['color'], va='center', fontsize=8.4)
        records.append(dict(panel=panel, rank='', model=item['label'], metric=metric, value='',
            unit=unit, lower_is_better=lower, relative_to_best_pct='', measurement_status='unavailable'))
    return fig, records


def parameter_scatter(data, values):
    order = sorted(NETWORKS, key=lambda k: data['parameters'][k])
    fig, ax = canvas('', 124, (.105, .15, .575, .73))
    offsets = {'pamf': (6, -12), 'resnet18': (-13, -15), 'densenet121': (-12, 14), 'resnet34': (-1, -23),
        'cvresnet18_1d': (7, 11), 'resnet18_2d': (-15, -13), 'bilstm3': (-4, -17),
        'resnet50': (-4, 12), 'transformer6': (6, -6), 'cvresnet18_2d': (-13, -15),
        'resnet50_2d': (4, 12), 'convnext_tiny': (7, -12), 'vgg16_bn': (6, 9)}
    for number, k in enumerate(order, 1):
        x = data['parameters'][k] / 1e6; m, sd = describe(values(k)); c = color(k)
        marker = 'D' if k == 'pamf' else '^' if k.startswith('cvresnet') else 's' if k.endswith('_2d') else 'D' if k in ('bilstm3', 'transformer6') else 'o'
        ax.errorbar(x, m, yerr=sd, fmt=marker, color=c, mfc=c if k == 'pamf' else 'white', ms=5,
                    capsize=2, lw=.75)
        ax.annotate(str(number), (x, m), xytext=offsets[k], textcoords='offset points',
                    color=c, fontsize=8.6, weight='bold')
        # A fixed key replaces long, crossing annotation leaders.
        fig.text(.72, .848 - (number - 1) * .052,
                 f'{number:>2}  {NAMES[k]}', color=c, fontsize=7.9, va='center')
    ax.set(xlim=(0, 41), ylim=(25, 89), xticks=[0, 10, 20, 30, 40],
        yticks=[30, 40, 50, 60, 70, 80], xlabel='Parameters (million)', ylabel='Accuracy (%)')
    grid(ax, 'both')
    return fig


def export(fig, out, number, slug, title, caption, tables, preview_only, dpi):
    stem = f'Fig{number:02d}_{slug}'
    fig.texts[0].set_text(f'Fig. {number}. {title}')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); box = fig.get_window_extent()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        if text.get_fontproperties().get_name() != FONT:
            raise ValueError('Non-Times text: ' + text.get_text())
        if any(s in text.get_text() for s in ('historical', 'Historical', 'PAMF residual', 'PAMF SIG_TIME')):
            raise ValueError('Excluded PAMF result in figure')
        b = text.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > box.x1 + 1 or b.y1 > box.y1 + 1):
            raise ValueError(stem + ': clipped text: ' + text.get_text())
    if sum(ax.get_label() != 'colorbar' for ax in fig.axes) != 1:
        raise ValueError('One data chart per image is required')
    for ext in ('svg', 'pdf'): fig.savefig(out / ext / (stem + '.' + ext))
    fig.savefig(out / 'preview' / (stem + '.png'), dpi=200)
    svg = ET.parse(out / 'svg' / (stem + '.svg')).getroot(); ns = '{http://www.w3.org/2000/svg}'
    if not svg.findall('.//' + ns + 'text') or svg.findall('.//' + ns + 'image'):
        raise ValueError('SVG must retain editable vector content')
    pdf = PdfReader(out / 'pdf' / (stem + '.pdf'))
    if len(pdf.pages) != 1 or len(pdf.pages[0].extract_text()) < 40: raise ValueError('PDF text missing')
    pixels = None
    if not preview_only:
        original = fig.dpi; fig.set_dpi(dpi); fig.canvas.draw()
        size = fig.canvas.get_width_height()
        raster = Image.frombuffer('RGBA', size, fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
        rgb = raster.convert('RGB')
        rgb.save(out / 'png' / (stem + '.png'), dpi=(dpi, dpi), compress_level=4)
        rgb.save(out / 'tiff' / (stem + '.tiff'), dpi=(dpi, dpi), compression='tiff_lzw')
        del rgb, raster
        fig.set_dpi(original); fig.canvas.draw(); gc.collect()
        Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, size[0] * size[1])
        for folder, ext in [('png', 'png'), ('tiff', 'tiff')]:
            with Image.open(out / folder / (stem + '.' + ext)) as im:
                if im.size != size or any(abs(v - dpi) > .03 for v in im.info['dpi']):
                    raise ValueError('Raster size/DPI mismatch')
                if ext == 'tiff' and im.info['compression'] != 'tiff_lzw': raise ValueError('TIFF compression mismatch')
        pixels = list(size)
    result = dict(number=number, stem=stem, title=title, caption=caption,
        source_tables=tables, font=FONT, width_mm=WIDTH_MM,
        height_mm=fig.get_figheight() * 25.4, raster_dpi=dpi, png_pixels=pixels,
        data_charts=1, no_subtitle=True, no_footer=True,
        editable_svg_text=True, pdf_text_extractable=True, canvas_clipping_checked=True)
    plt.close(fig)
    print(('PREVIEW ' if preview_only else 'EXPORTED ') + stem, flush=True)
    return result


def render(data, out, preview_only, dpi):
    ext, t1, g, index, values, timing, deployed = validate(data)
    records, resources = [], []
    def save(fig, slug, title, caption='', tables=None):
        records.append(export(fig, out, len(records) + 1, slug, title, caption,
            tables or ['external_per_seed.csv'], preview_only, dpi))
    conditions = ['clean conditions', 'high-quality suburban channels', 'medium-quality suburban channels',
        'low-quality suburban channels', 'high-quality urban channels', 'medium-quality urban channels', 'low-quality urban channels']
    external_caption = ('Three model seeds and 168000 identical physical packets per scene. '
        'Bars show means and sample SD (ddof=1). Full and Full+plain PL have adjacent bars. '
        'PAMF is T1-E50 (13L+37U)+DURA v1.0; three F seeds share one fixed G. '
        'External recipes and training budgets differ from PAMF. Plain PL uses threshold 0.95 and weight 1. '
        'Seven views are correlated. '
        'No CE-only settings are plotted.')
    for i, condition in enumerate(conditions):
        save(draw_bars(ext, i), SCENES[i], 'Recognition accuracy under ' + condition, external_caption)
    save(draw_bars(ext), 'mean_channel_accuracy', 'Mean recognition accuracy across channel conditions', external_caption)
    keys = [(f, r) for f in FAMILIES for r in ('full', 'full_pl')] + [('pamf', 'pamf')]
    labels = [NAMES[f] + (' Full' if r == 'full' else ' Full + PL') if f != 'pamf' else 'PAMF' for f, r in keys]
    save(heatmap(np.stack([ext[k].mean(axis=0) for k in keys]), labels, SCENE_SHORT,
        np.stack([ext[k].std(axis=0, ddof=1) for k in keys]), [COLORS[k[0]] for k in keys], 123),
        'seven_scene_accuracy', 'Recognition performance across channel conditions', external_caption)
    diffs = [ext[f, 'full_pl'][:, 1:].mean(axis=1) - ext[f, 'full'][:, 1:].mean(axis=1) for f in FAMILIES]
    save(forest([NAMES[f] for f in FAMILIES], diffs, [COLORS[f] for f in FAMILIES],
        'Accuracy change (percentage points)', (-8, 8)), 'pl_gain', 'Effect of pseudo-labeling on recognition accuracy',
        'Full+plain PL minus Full, computed within model seed before aggregation. Mean ± paired sample SD; no significance test.')
    for regime, label in [('full', 'Full'), ('full_pl', 'Full + PL')]:
        save(scatter_balance(ext, regime), 'clean_robustness_' + regime,
            'Clean accuracy and channel robustness (' + label + ')',
            'Clean accuracy versus the within-seed six-scenario mean. Error bars are sample SD in each coordinate. '
            'Full and Full+PL are shown in separate figures with the same limits; point coordinates are not jittered. ' + external_caption)
    fig, ax = canvas('', 105, (.105, .23, .85, .65)); x = np.arange(7)
    for readout, dx, c, label in [('raw', -.17, COLORS['baseline'], 'T1'), ('DURA-LU', .17, COLORS['pamf'], 'T1 + DURA')]:
        m, sd = t1[readout].mean(axis=0), t1[readout].std(axis=0, ddof=1)
        ax.bar(x + dx, m, width=.29, color=pale(c) if readout == 'raw' else c, edgecolor=c,
            hatch='//' if readout == 'raw' else None, linewidth=.7, yerr=sd,
            error_kw=dict(ecolor=INK, elinewidth=.75, capsize=2), label=label)
        for i in range(7): ax.text(i + dx, m[i] + sd[i] + 1.8, f'{m[i]:.1f}', ha='center', fontsize=7.8, color=c)
    ax.set(ylim=(0, 100), yticks=range(0, 101, 20), xticks=x, xticklabels=SCENE_SHORT, ylabel='Accuracy (%)')
    ax.tick_params(axis='x', length=0, pad=7); grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.53, .06), ncol=2)
    save(fig, 'dura_accuracy', 'Recognition accuracy with and without DURA',
        'Same three frozen T1-E50 F seeds and packets; one fixed G. Mean ± sample SD conditional on G.', ['t1_per_seed.csv'])
    save(forest(SCENE_NAMES, list((t1['DURA-LU'] - t1['raw']).T), [COLORS['pamf']] * 7,
        'Accuracy gain (percentage points)', (-1, 18)), 'dura_gain', 'Accuracy gain provided by DURA',
        'Within-F-seed restoration differences. All scenes retained; mean ± paired sample SD.', ['t1_per_seed.csv'])
    arms = list(G_NAMES)
    save(forest([G_NAMES[k] for k in arms], [(g[k] - g['default'])[:, 1:].mean(axis=1) for k in arms],
        [COLORS['pamf']] + [COLORS['baseline']] * 7, 'Accuracy change (percentage points)', (-8, 1), (0,)),
        'dura_ablation', 'Effect of DURA components',
        'Paired changes from default. New F/G seeds are paired; default uses one fixed G. '
        'Zero for default is a reference, not zero G variance. This is not a complete initialization×distillation interaction.', ['g_per_seed.csv'])
    save(heatmap(np.stack([g[k].mean(axis=0) for k in arms]), [G_NAMES[k] for k in arms], SCENE_SHORT,
        np.stack([g[k].std(axis=0, ddof=1) for k in arms]), [COLORS['pamf']] + [COLORS['baseline']] * 7, 121),
        'dura_ablation_scenes', 'DURA performance across channel conditions',
        'Mean ± sample SD of paired F/G repeats; one fixed default G. The seven views are correlated.', ['g_per_seed.csv'])
    system_items = [dict(label=NAMES[f], value=float(timing[c, 1]['median_ms']), color=COLORS[f])
                    for c, f in zip(SYSTEM_CASES, (*FAMILIES, 'pamf'))]
    fig, table = ranked_bars(system_items, 'ms', 'Median latency', True, 3, 'system_latency'); resources.extend(table)
    save(fig, 'system_latency', 'Single-sample inference latency',
        'Fixed seed01 models, B1, exclusive RTX3090, FP32/TF32off. Pooled GPU-wall median over five timing rounds, '
        'excluding transfer. Bars are actual latency/minimum latency×100 and sorted fastest first. '
        'External timings refer to Full models, not PL models.', ['latency.csv', 'resource_rankings.csv'])
    fig, table = ranked_bars([dict(label=ROUTE_NAMES[c], value=float(timing[c, 1]['median_ms']),
        color=COLORS['pamf'] if c == ROUTES[-1] else COLORS['baseline']) for c in ROUTES], 'ms', 'Median latency', True, 3,
        'execution_latency')
    resources.extend(table)
    save(fig, 'execution_latency', 'Inference latency across execution paths',
        'Same fixed weights and precision. GPU-wall median, B1; sorted fastest first. '
        'This does not establish full-data prediction equivalence.', ['latency.csv', 'resource_rankings.csv'])
    fig, table = ranked_bars([dict(label=NAMES[f], value=float(timing[c, 128]['throughput_IQ_s']) / 1000,
        color=COLORS[f]) for c, f in zip(SYSTEM_CASES, (*FAMILIES, 'pamf'))], 'kIQ/s', 'Throughput', False, 2,
        'system_throughput')
    resources.extend(table)
    save(fig, 'system_throughput', 'Batch-128 inference throughput',
        'Throughput=batch/pooled median GPU-wall time. B128; sorted highest first and normalized to the maximum. '
        'These are source-input runtime measurements, not additional scientific evaluations.', ['latency.csv', 'resource_rankings.csv'])
    network_caption = ('PAMF is the current T1-E50 (13L+37U)+fixed DURA v1.0, three F seeds sharing one G. '
        'The twelve baseline networks use E200 pure CE, eight seeds each. All seeds are retained, including '
        'collapsed runs. Each clean evaluation contains 168000 packets. Training recipes and budgets differ; '
        'this is a cross-recipe comparison, not a matched-budget architecture ablation. '
        'Dots show model seeds; diamonds and whiskers show the mean and sample SD (ddof=1).')
    for field, slug, title in [('accuracy', 'network_accuracy', 'Classification accuracy of neural networks'),
        ('macro_f1', 'network_macro_f1', 'Macro-F1 scores of neural networks')]:
        order = sorted(NETWORKS, key=lambda k: -values(k, field).mean())
        save(forest([NAMES[k] for k in order], [values(k, field) for k in order], [color(k) for k in order],
            'Accuracy (%)' if field == 'accuracy' else 'Macro-F1 (%)', highlights=(order.index('pamf'),)),
            slug, title, network_caption, ['network_per_seed.csv', 'network_summary.csv'])
    order = sorted(NETWORKS, key=lambda k: -values(k).mean())
    receivers = [r['stratum'] for r in data['baselines'][0]['results'] if r['dimension'] == 'receiver']
    save(heatmap(np.array([[values(k, dim='receiver', stratum=r).mean() for r in receivers] for k in order]),
        [NAMES[k] for k in order], receivers, row_colors=[color(k) for k in order], height=141),
        'network_receiver_accuracy', 'Recognition accuracy across receivers',
        'Mean accuracy across three PAMF F seeds or eight baseline seeds; 24000 packets per receiver and model. '
        'Shared absolute 0–100% color scale. ' + network_caption, ['network_per_seed.csv'])
    save(parameter_scatter(data, values), 'parameters_accuracy', 'Model size and recognition accuracy',
        'Current PAMF includes 232993 F and 266951 G parameters, total 499944. '
        'Mean ± sample SD across three PAMF F seeds or eight baseline seeds. '
        'Numbered labels map to the right-hand key; no annotation leader lines are used. '
        'The x axis is linear, and the y range retains every full mean±SD interval. ' + network_caption,
        ['network_summary.csv'])
    pairs = [('ResNet-18', 'resnet18', 'resnet18_2d'), ('ResNet-50', 'resnet50', 'resnet50_2d'),
             ('Complex ResNet-18', 'cvresnet18_1d', 'cvresnet18_2d')]
    save(forest([p[0] for p in pairs], [values(a) - values(b) for _, a, b in pairs],
        [COLORS['baseline']] * 3, '1D − 2D accuracy (percentage points)', (-5, 17)),
        'dimensionality_comparison', 'Accuracy difference between 1D and 2D networks',
        'Eight same-seed paired differences. E200; 2D inputs use the fixed 16×16 IQ reshape.', ['baseline_per_seed.csv'])
    pairs = [('ResNet-18-1D', 'cvresnet18_1d', 'resnet18'), ('ResNet-18-2D', 'cvresnet18_2d', 'resnet18_2d')]
    save(forest([p[0] for p in pairs], [values(a) - values(b) for _, a, b in pairs],
        [COLORS['baseline']] * 2, 'Complex − real accuracy (percentage points)', (-10, 7)),
        'complexity_comparison', 'Accuracy difference between complex and real networks',
        'Eight same-seed paired differences; full seed spread is visible. No inferential test.', ['baseline_per_seed.csv'])
    cm = np.array([index['resnet50', s, 'overall', 'ALL']['confusion'] for s in SEEDS8], float)
    save(heatmap((100 * cm / cm.sum(axis=2, keepdims=True)).mean(axis=0), data['classes'], data['classes'],
        height=114, confusion=True), 'resnet50_confusion', 'Confusion matrix of ResNet-50-1D',
        'Eight-seed mean of row-normalized confusion matrices; 28000 packets per true TX and model, not an ensemble.',
        ['baseline_per_seed.csv'])
    specs = [(1, 'p95_ms', 'ms', 'P95 latency', True, 1, 3, 'End-to-end tail latency of neural networks'),
        (128, 'end_to_end_throughput_samples_s', 'kIQ/s', 'Throughput', False, .001, 2, 'End-to-end throughput of neural networks'),
        (1, 'gpu_peak_allocated_mib', 'MiB', 'GPU allocation', True, 1, 2, 'Peak GPU allocation of neural networks'),
        (1, 'cpu_peak_rss_mib', 'MiB', 'Process RSS', True, 1, 2, 'Peak CPU memory of neural networks'),
        (1, 'model_storage_mib', 'MiB', 'FP32 storage', True, 1, 2, 'FP32 model storage of neural networks')]
    for batch, metric, unit, name, lower, factor, digits, title in specs:
        items = [dict(label=NAMES[k], value=float(deployed[k, batch][metric]) * factor,
            color=color(k)) for k in BASELINES]
        items.append(dict(label='PAMF', value=None, color=COLORS['pamf']))
        fig, table = ranked_bars(items, unit, name, lower, digits, 'baseline_' + metric)
        resources.extend(table)
        save(fig, metric, title,
            'Twelve baseline networks, eight checkpoints, RTX3090/FP32. '
            'End-to-end timing includes CPU input, H2D, model and returned labels; it is not comparable '
            'to the GPU-wall timing boundary. B1 has 2400 pooled requests; B128 has 800. '
            'Memory is the original inference-window peak, process RSS includes runtime, and storage '
            'is serialized FP32 state including buffers. The best model is first; bar lengths equal actual value/best×100. '
            'Right-hand labels give actual measurements, not normalized scores. '
            'Current PAMF is shown as an unranked missing-data row, with no zero-length substitute bar. '
            + data['pamf_resource_gaps'][metric]
            + ' Its separate GPU-wall latency and throughput are reported in Figs. 17 and 19.',
            ['deployment.csv', 'resource_rankings.csv', 'pamf_resource_gaps.csv'])
    fig, table = ranked_bars([dict(label=NAMES[k], value=data['parameters'][k] / 1e6,
        color=color(k)) for k in NETWORKS], 'M', 'Parameters', True, 3, 'network_parameters')
    resources.extend(table)
    save(fig, 'network_parameters', 'Parameter counts of neural networks',
        'Registered parameter counts, including each default classifier. Current PAMF includes the full '
        'T1-E50 recognizer (232993) and DURA v1.0 restorer (266951), total 499944; optimizer state is excluded. '
        'All thirteen systems are measured quantities, ranked from smallest to largest with true proportional bars.',
        ['network_summary.csv', 'resource_rankings.csv'])
    cm = np.array([index['pamf', s, 'overall', 'ALL']['confusion'] for s in SEEDS3], float)
    # The scored artifact uses checkpoint class IDs; retain those IDs without inventing a physical TX mapping.
    tx_ids = [str(i) for i in range(6)]
    save(heatmap((100 * cm / cm.sum(axis=2, keepdims=True)).mean(axis=0), tx_ids, tx_ids,
        height=114, confusion=True), 'pamf_confusion', 'Confusion matrix of PAMF',
        'Current T1-E50+DURA, three-seed mean of row-normalized confusion matrices, not an ensemble. '
        '28000 packets per true class and F seed. Axes retain the frozen checkpoint class IDs 0–5. '
        'The default G is shared by the three F models.', ['pamf_clean_scores.json', 'network_per_seed.csv'])
    write_csv(out / 'source_data/resource_rankings.csv', resources)
    return records


def tables(data, root):
    ext, t1, g, index, values, timing, deployed = validate(data)
    for name, rows in [('external_per_seed.csv', data['external']), ('t1_per_seed.csv', data['t1']),
        ('g_per_seed.csv', data['g']), ('latency.csv', data['latency']), ('deployment.csv', data['deployment'])]:
        write_csv(root / name, rows)
    write_csv(root / 'baseline_summary.csv', [dict(model=k, display_name=NAMES[k], n=8,
        parameters=data['parameters'][k], accuracy_pct=describe(values(k))[0],
        accuracy_sample_sd_pp=describe(values(k))[1], macro_f1_pct=describe(values(k, 'macro_f1'))[0],
        macro_f1_sample_sd_pp=describe(values(k, 'macro_f1'))[1]) for k in BASELINES])
    per_seed = [dict(model=k, model_seed=s, dimension=dim, stratum=stratum,
        count=int(np.array(r['confusion']).sum()), accuracy_pct=100 * r['accuracy'],
        macro_f1_pct=100 * r['macro_f1']) for (k, s, dim, stratum), r in index.items()]
    write_csv(root / 'baseline_per_seed.csv', [r for r in per_seed if r['model'] != 'pamf'])
    write_csv(root / 'network_per_seed.csv', per_seed)
    write_csv(root / 'network_summary.csv', [dict(model=k, display_name=NAMES[k],
        n=3 if k == 'pamf' else 8, recipe='T1-E50+DURA v1.0' if k == 'pamf' else 'E200 pure CE',
        parameters=data['parameters'][k], accuracy_pct=describe(values(k))[0],
        accuracy_sample_sd_pp=describe(values(k))[1], macro_f1_pct=describe(values(k, 'macro_f1'))[0],
        macro_f1_sample_sd_pp=describe(values(k, 'macro_f1'))[1]) for k in NETWORKS])
    write_json(root / 'pamf_clean_scores.json', data['pamf'])
    write_csv(root / 'pamf_resource_gaps.csv', [dict(model='PAMF', metric=k,
        status='unavailable', reason=v) for k, v in data['pamf_resource_gaps'].items()])


def finish(out, data, records, dpi):
    writer = PdfWriter()
    for r in records: writer.append(str(out / 'pdf' / (r['stem'] + '.pdf')))
    with (out / 'all_figures.pdf').open('wb') as stream: writer.write(stream)
    captions = ['# Figure captions', '',
        'Captions and statistical definitions are provided here for manuscript use; they are intentionally absent from the images.', '',
        'PAMF denotes T1-E50 (13L+37U) plus fixed DURA v1.0. Three recognizer seeds share one G. '
        'New G-student experiments pair F/G seeds. Their spread is not pure G variance. '
        'Full/Full+PL use three model seeds; other neural-network baselines use E200 pure CE and eight seeds. '
        'No training, inference or new truth access occurs. Current GPU wall and baseline end-to-end timings '
        'must not be combined into one speedup. Source data preserve full precision.', '']
    for r in records:
        captions += [f'## Fig. {r["number"]}. {r["title"]}', '', r['caption'], '',
                     'Source tables: ' + ', '.join(r['source_tables']) + '.', '']
    (out / 'captions_en.md').write_text('\n'.join(captions), encoding='utf-8')
    (out / 'README_zh.md').write_text(f'''# 论文图表修订版

共 {len(records)} 张独立图，统一使用 Times New Roman。图内只保留简洁的 Fig. 编号与标题、坐标、数据和必要图例；副标题与图下说明已移除。完整统计定义放在 captions_en.md。

旧 PAMF 版本及其专属诊断未进入本包。当前 PAMF＝T1-E50（13L＋37U）＋DURA v1.0，已加入总体准确率、Macro-F1、各接收机、参数量与准确率、参数数量图，并增加当前 PAMF 的混淆矩阵。PAMF 使用三个 F 种子及一个固定 G；其他网络使用 E200 纯 CE 的八个种子，训练配方和预算不同，完整统计口径见图注。

模型参数与准确率图使用数字索引及右侧名称表，没有交叉指示线。Full 与 Full＋PL 的散点图分别展示，使用相同坐标范围，原始坐标和误差线不变。

资源图使用从零开始的线性横向柱状图。越小越好的指标升序排列，越大越好的指标降序排列，最优方法总在顶部。柱长＝实际值／最优值×100%，柱右给出原始测量及单位。PAMF 的完整参数量为 499,944，包含 F 的 232,993 和 G 的 266,951 个参数。

旧网络端到端时延、吞吐量、GPU 显存、CPU 内存、序列化模型存储五张图缺少同口径的当前 PAMF 测量，已在图中列出 PAMF 并标明 Not measured，不画虚构柱、不参与排名。PAMF 已有的同口径系统 GPU 时延和吞吐量在图 17、19 展示。已有混合进程显存含其他模型、Graph 池和 13 GiB 预约，不能冒充 PAMF 单模型显存；参数字节数也不能冒充序列化文件大小。缺项及原因见 source_data/pamf_resource_gaps.csv。

png/、tiff/ 为 {dpi} dpi 位图，分别打包；svg/、pdf/ 为可编辑矢量图；all_figures.pdf 为逐页合集。source_data/ 提供本包实际使用的逐种子结果、汇总和资源排序。

重绘：python reproduce.py --source source_data --output NEW_DIRECTORY --dpi {dpi}

依赖：matplotlib、numpy、Pillow、pypdf，以及已安装的 Times New Roman。输出目录必须不存在。
数据快照：{data['snapshot_hkt']}。
''', encoding='utf-8')
    write_csv(out / 'figure_index.csv', [dict(number=r['number'], figure=r['stem'], title=r['title'],
        source_tables='; '.join(r['source_tables'])) for r in records])
    write_json(out / 'manifest.json', dict(status='VERIFIED', schema=data['schema'],
        snapshot_hkt=data['snapshot_hkt'], figure_count=len(records), font=FONT,
        raster_dpi=dpi, figures=records, excluded_superseded_pamf_results=True,
        resource_sort='best_first', resource_bar_length='actual_value/best_value*100',
        current_pamf_in_network_accuracy=True, current_pamf_in_network_macro_f1=True,
        current_pamf_in_receiver_accuracy=True, current_pamf_in_parameter_comparisons=True,
        current_pamf_confusion_matrix=True, unmeasured_pamf_resources=list(data['pamf_resource_gaps']),
        no_subtitles=True, no_figure_footnotes=True, no_leader_lines=True))
    shutil.copyfile(Path(__file__), out / 'reproduce.py')
    shutil.copyfile(Path(__file__).with_name('paper_publication_style.py'), out / 'paper_publication_style.py')
    archives = {}
    for kind in ('vector', 'png', 'tiff'):
        archive = out.with_suffix('.zip') if kind == 'vector' else out.with_name(out.name + '-' + kind.upper() + f'-{dpi}dpi.zip')
        if archive.exists(): raise FileExistsError('Preserve existing archive')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED if kind == 'vector' else zipfile.ZIP_STORED) as z:
            for p in sorted(out.rglob('*')):
                if not p.is_file(): continue
                top = p.relative_to(out).parts[0]
                if (kind == 'vector' and top not in ('png', 'tiff')) or top == kind:
                    z.write(p, p.relative_to(out))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None: raise ValueError('ZIP CRC check failed')
        archives[kind] = archive.stat().st_size
    print(json.dumps(dict(status='VERIFIED', figures=len(records), dpi=dpi,
        font=FONT, archive_bytes=archives)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dpi', type=int, default=4800)
    p.add_argument('--pamf-scores', type=Path, help='Directory of the three frozen current PAMF score JSONs')
    p.add_argument('--pamf-profile', type=Path, help='Verified current F/G parameter profile JSON')
    p.add_argument('--preview-only', action='store_true')
    p.add_argument('--resume-render', action='store_true')
    args = p.parse_args()
    data = include_pamf(prepare(load(args.source / 'scores.json')), args.pamf_scores, args.pamf_profile)
    validate(data)
    out = args.output.resolve(); repair = args.resume_render
    if repair:
        if (args.source.resolve() != out / 'source_data' or not out.exists()
                or (out / 'manifest.json').exists()): raise ValueError('Repair only this incomplete export with its own snapshot')
    elif out.exists(): raise FileExistsError('Preserve previous exports; choose a new directory')
    if args.dpi < 300: raise ValueError('Raster DPI must be at least 300')
    out.mkdir(parents=True, exist_ok=repair)
    for sub in ('pdf', 'svg', 'png', 'tiff', 'preview', 'source_data'): (out / sub).mkdir(exist_ok=repair)
    if not repair: write_json(out / 'source_data/scores.json', data)
    tables(data, out / 'source_data'); apply_style()
    records = render(data, out, args.preview_only, args.dpi)
    if args.preview_only:
        write_json(out / 'draft_index.json', records)
        print('PREVIEW_COMPLETE ' + str(len(records)), flush=True)
    else:
        if (out / 'draft_index.json').exists(): (out / 'draft_index.json').unlink()
        finish(out, data, records, args.dpi)


if __name__ == '__main__': main()
