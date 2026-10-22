"""Unify frozen experimental figures; read scored summaries, never run experiments.

Create: python unify_pamf_figures.py --project-root PROJECT --output NEW_DIRECTORY
Redraw: python reproduce.py --source SOURCE_DATA_DIRECTORY --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import csv
import gc
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter

from publication_style import (COLORS, FAMILY_COLORS, INK, MUTED, CMAP,
    WIDTH_MM, apply_style, canvas, contrast_color, describe, grid, note,
    pale, regime_legend, seed_legend)

SEEDS3 = tuple(range(2026092701, 2026092704))
SEEDS8 = tuple(range(2026092701, 2026092709))
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')
SCENES = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SCENE_NAMES = ('Clean', 'Suburban high', 'Suburban mid', 'Suburban low',
               'Urban high', 'Urban mid', 'Urban low')
SCENE_SHORT = ('Clean', 'Suburban\nhigh', 'Suburban\nmid', 'Suburban\nlow',
               'Urban\nhigh', 'Urban\nmid', 'Urban\nlow')
DISPLAY = {'riei': 'RIEI', 'drift': 'DRIFT', 'ibrff': 'IB-RFF', 'sdg': 'SDG',
    'residual_fusion': 'PAMF residual', 'sig_time': 'PAMF SIG_TIME',
    'resnet18': 'ResNet-18-1D', 'resnet34': 'ResNet-34-1D',
    'resnet50': 'ResNet-50-1D', 'densenet121': 'DenseNet-121-1D',
    'vgg16_bn': 'VGG-16-BN-1D', 'convnext_tiny': 'ConvNeXt-Tiny-1D',
    'bilstm3': 'BiLSTM (3 layers)', 'transformer6': 'Transformer (6 layers)',
    'resnet18_2d': 'ResNet-18-2D', 'resnet50_2d': 'ResNet-50-2D',
    'cvresnet18_1d': 'Complex ResNet-18-1D', 'cvresnet18_2d': 'Complex ResNet-18-2D'}
HISTORICAL = tuple(k for k in DISPLAY if k not in FAMILIES)
ARMS = {'default': 'Default DURA', 'coarse_only': 'Coarse only',
    'prior_no_CG': 'Without CG', 'no_image_action': 'Without image action',
    'no_innovation_input': 'Without residual input', 'copy_no_KD': 'Copy, no IQ distillation',
    'random_with_KD': 'Random + IQ distillation', 'random_no_KD': 'Random, no IQ distillation'}
CURRENT_NOTE = 'Three F seeds share one fixed DURA. Native recipes and budgets differ across methods.'
HIST_NOTE = 'Historical E200, pure CE; eight model seeds. These are separate from current T1-E50 results.'
EXT_RUN = '20261019-phase1-external-dg-concat-pl-m219-r03'
MAIN_RUN = '20261021-phase1-T1-DURA-comparison-m153-r01'
HISTORY_RUNS = ('20261010-phase1-standard-ce-baselines-manysig-m56-r01',
    '20261012-phase1-standard-ce-seed-extension-manysig-m52-r01',
    '20261014-phase1-sigtime-purece-e200-clean-manysig-m8-r01')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty source table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def color(model):
    if model in ('residual_fusion', 'sig_time', 'pamf'):
        return COLORS['pamf']
    family = ('complex' if model.startswith('cvresnet') else
              'real2d' if model.endswith('_2d') else
              'sequence' if model in ('bilstm3', 'transformer6') else 'real1d')
    return COLORS.get(model, FAMILY_COLORS[family])


def collect(project):
    """Copy only publication-safe score fields from existing completed artifacts."""
    reports = project / 'automation_reports/CV-SincNet'
    histdir = project / 'exports/paper_figures/20261012-standard-ce-clean-eight-seed-r01-source'
    hist = read_json(histdir / 'verified_results.json')
    if hist['status'] != 'VERIFIED':
        raise ValueError('Historical scores are not verified')
    history = [dict(variant=r['row']['variant'], seed=r['row']['model_seed'],
                    results=r['scores']['results']) for r in hist['rows']]
    sig = read_json(project / 'code/worktrees/standard_ce_baselines_20261010/'
        'local_artifacts/sigtime_purece_e200_20261014_r01/results_readback.json')
    if sig['status'] != 'VERIFIED' or len(sig['rows']) != 8:
        raise ValueError('SIG_TIME eight-seed results incomplete')
    for row in sig['rows']:
        if not (row['source_complete']['epoch'] == 200
                and row['score_complete']['status'] == 'SCORED_COMPLETE'
                and row['score_complete']['truth_last']
                and row['freeze']['status'] == 'FROZEN'
                and not row['freeze']['target_score_used']):
            raise ValueError('SIG_TIME completion/freeze identity changed')
        history.append(dict(variant='sig_time', seed=row['model_seed'],
                            results=row['scores']['results']))
    parameters = {r['variant']: int(r['total_parameters'])
                  for r in read_csv(histdir / 'network_summary.csv')}
    parameters['sig_time'] = sig['rows'][0]['resolved']['total_parameters']
    source = read_json(project / 'code/worktrees/standard_ce_baselines_20261010/'
        'local_artifacts/pamf_source_class_20261019_r02/results.json')
    if not (source['status'] == 'ANALYZED' and source['role'] == 'source_V'
            and not source['target_access'] and len(source['rows']) == 8):
        raise ValueError('Source validation score context changed')
    source_safe = [dict(seed=r['model_seed'], confusion=r['confusion'],
                        recall_pct=r['recall_pct']) for r in source['rows']]
    deployment = read_csv(project / 'exports/paper_figures/'
        '20261012-standard-ce-comprehensive-ranking-r01/source_data/deployment_summary.csv')
    latency = read_csv(reports / '20261021-phase1-T1-DURA-latency-r02/latency_summary.csv')
    check = read_json(reports / '20261021-phase1-T1-DURA-latency-r02/latency_verification.json')
    if check['status'] != 'VERIFIED' or check['case_rounds'] != 500:
        raise ValueError('Current timing panel is not complete')
    bundle = dict(schema='pamf_unified_figures_v1',
        snapshot_hkt=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        classes=hist['classes'], history=history, parameters=parameters,
        historical_summary=read_csv(histdir / 'network_summary.csv'),
        source_validation=source_safe, deployment=deployment, latency=latency,
        external=read_csv(project / 'exports/phase1_external_dg/'
            '20261022-pamf-seven-scene-grouped-bars-r01/source_data.csv'),
        t1=read_csv(reports / MAIN_RUN / 'final_t1_f0_per_seed.csv'),
        g_ablation=read_csv(reports / '20261021-phase1-T1-DURA-G-ablation-m21-r01/G_ablation_per_seed.csv'),
        g_summary=read_csv(reports / '20261021-phase1-T1-DURA-G-ablation-m21-r01/G_ablation_summary.csv'))
    return bundle


def matrix(rows, fields, value, keys, seeds=SEEDS3, scenes=SCENES):
    expected = {(key, s, v) for key in keys for s in seeds for v in scenes}
    found = {}
    for row in rows:
        raw = tuple(row[f] for f in fields)
        key = raw[0] if len(raw) == 1 else raw
        seed = int(row.get('seed', row.get('model_seed', -1)))
        view = row.get('scene', row.get('view'))
        index = key, seed, view
        if index in found or index not in expected:
            raise ValueError('Unexpected/duplicate score cell: ' + str(index))
        a = float(row[value])
        if not math.isfinite(a) or not 0 <= a <= 100 or int(row['count']) != 168000:
            raise ValueError('Percentage/count mismatch: ' + str(index))
        found[index] = a
    if set(found) != expected:
        raise ValueError('Missing score cells: ' + str(expected - set(found)))
    return {key: np.array([[found[key, seed, v] for v in scenes] for seed in seeds])
            for key in keys}


def validate(data):
    """Check existing score identities and independent published aggregate controls."""
    if data['schema'] != 'pamf_unified_figures_v1':
        raise ValueError('Unrecognized source bundle')
    extkeys = tuple((f, r) for f in FAMILIES for r in ('full', 'full_pl')) + (('pamf', 'pamf'),)
    external = matrix(data['external'], ('family', 'regime'), 'accuracy_pct', extkeys)
    for row in data['external']:
        if row['source_run'] != (MAIN_RUN if row['family'] == 'pamf' else EXT_RUN):
            raise ValueError('Unexpected external score run')
    t1 = matrix(data['t1'], ('readout',), 'accuracy', ('raw', 'DURA-LU'))
    if not np.allclose(external['pamf', 'pamf'], t1['DURA-LU'], rtol=0, atol=1e-10):
        raise ValueError('PAMF external and final T1 panels do not reconcile')
    gkeys = tuple(dict.fromkeys(row['arm'] for row in data['g_ablation']))
    g = matrix(data['g_ablation'], ('arm',), 'accuracy', gkeys)
    # Raw and default rows may be stored under existing arm-specific names.
    if 'default' not in g:
        default_key = next(k for k in g if k not in ('raw',) and
            np.allclose(g[k], t1['DURA-LU'], rtol=0, atol=1e-10))
        g['default'] = g.pop(default_key)
    if not np.allclose(g['raw'], t1['raw'], rtol=0, atol=1e-10):
        raise ValueError('G and T1 raw controls differ')
    metric_index = {}
    for row in data['history']:
        variant, seed = row['variant'], int(row['seed'])
        if variant not in HISTORICAL or seed not in SEEDS8:
            raise ValueError('Historical variant/seed changed')
        for result in row['results']:
            key = variant, seed, result['dimension'], result['stratum']
            if key in metric_index:
                raise ValueError('Duplicate historical score')
            cm = np.asarray(result['confusion'], dtype=np.int64)
            n = cm.sum()
            if cm.shape != (6, 6) or n <= 0:
                raise ValueError('Invalid scored confusion matrix')
            expected_n = {'overall': 168000, 'receiver': 24000, 'transmitter': 28000}[result['dimension']]
            if n != expected_n or abs(100 * cm.trace() / n - 100 * result['accuracy']) > 1e-8:
                raise ValueError('Historical score and confusion matrix differ')
            metric_index[key] = result
    for variant in HISTORICAL:
        if {int(r['seed']) for r in data['history'] if r['variant'] == variant} != set(SEEDS8):
            raise ValueError('Historical eight-seed matrix incomplete')
    def hvalues(variant, field='accuracy', dim='overall', stratum='ALL'):
        return np.array([100 * metric_index[variant, s, dim, stratum][field] for s in SEEDS8])
    for row in data['historical_summary']:
        values = hvalues(row['variant'])
        if not np.allclose(describe(values), [float(row['accuracy_pct']), float(row['accuracy_sd_pp'])], rtol=0, atol=1e-8):
            raise ValueError('Historical published mean/SD differ')
    for row in data['g_summary']:
        # Reconcile all seven scenes, not just headline means.
        if row['arm'] in g:
            vals = (g[row['arm']][:, 1:].mean(axis=1) if row['view'] == 'six_view_mean'
                    else g[row['arm']][:, SCENES.index(row['view'])])
            if not np.allclose(describe(vals), [float(row['accuracy_mean']), float(row['accuracy_sd'])], rtol=0, atol=1e-9):
                raise ValueError('G published mean/SD differ')
    for row in data['source_validation']:
        cm = np.asarray(row['confusion'])
        if cm.shape != (6, 6) or not np.all(cm.sum(axis=1) == 4500):
            raise ValueError('Source V class counts differ')
        recalls = cm.diagonal() / 4500 * 100
        if not np.allclose(recalls, [row['recall_pct'][c] for c in data['classes']], rtol=0, atol=1e-9):
            raise ValueError('Source V published class recalls differ')
    timing = {(r['case'], int(r['batch'])): r for r in data['latency']}
    if len(timing) != 100 or any(int(r['rounds']) != 5 for r in timing.values()):
        raise ValueError('Current timing panel incomplete or duplicated')
    for row in timing.values():
        med, p95, b = float(row['median_ms']), float(row['P95_ms']), int(row['batch'])
        if not 0 < med <= p95 or abs(float(row['throughput_IQ_s']) - 1000 * b / med) > 1e-7:
            raise ValueError('Timing units/throughput mismatch')
    deployment = {(r['variant'], int(r['batch_size'])): r for r in data['deployment']}
    if len(deployment) != 26 or any(int(r['actual_seed_count']) != 8 for r in deployment.values()):
        raise ValueError('Historical deployment panel incomplete')
    return external, t1, g, metric_index, hvalues, timing, deployment


def forest(title, subtitle, labels, values, colors, xlabel,
           limits=(0, 100), highlight=(), first_note='', second_note='', delta=False,
           label_width=.32, height=None):
    n = len(labels)
    observed_min = min(min(float(np.min(v)), describe(v)[0] - describe(v)[1]) for v in values)
    observed_max = max(max(float(np.max(v)), describe(v)[0] + describe(v)[1]) for v in values)
    if observed_min < limits[0] or observed_max > limits[1]:
        if not delta:
            raise ValueError('A plotted mean/SD or seed falls outside the declared axis')
        span = observed_max - observed_min
        limits = (min(limits[0], math.floor(observed_min - .05 * span)),
                  max(limits[1], math.ceil(observed_max + .05 * span)))
    height = height or max(110, 52 + n * 6.7)
    fig, ax = canvas(title, subtitle, height,
                     bounds=(label_width, .245, .81 - label_width, .56))
    ax.set(xlim=limits, ylim=(n - .45, -.65), yticks=range(n),
           yticklabels=labels, xlabel=xlabel)
    ax.tick_params(axis='y', length=0, pad=9)
    ax.spines['left'].set_visible(False)
    grid(ax, 'x')
    if delta:
        ax.axvline(0, color=MUTED, lw=.8, ls=(0, (3, 3)))
    for i, (vals, c) in enumerate(zip(values, colors)):
        vals = np.asarray(vals)
        mean, sd = describe(vals)
        if i in highlight:
            ax.axhspan(i - .42, i + .42, color=pale(c, .065), zorder=0)
        ax.scatter(vals, i + np.linspace(-.15, .15, len(vals)), s=10,
                   c=c, alpha=.36, linewidths=0, zorder=3, clip_on=False)
        ax.errorbar(mean, i, xerr=sd, fmt='D', color=c, mfc=c if i in highlight else 'white',
                    ms=5.1, mew=1, elinewidth=1, capsize=2, zorder=4, clip_on=False)
        ax.text(1.065, i, f'{mean:+.2f} ± {sd:.2f}' if delta else f'{mean:.2f} ± {sd:.2f}',
                transform=ax.get_yaxis_transform(), ha='left', va='center',
                color=c, fontsize=7.3, weight='bold' if i in highlight else 'normal')
        ax.get_yticklabels()[i].set_color(c)
        if i in highlight:
            ax.get_yticklabels()[i].set_weight('bold')
    seed_legend(fig)
    note(fig, first_note, second_note)
    return fig


def external_bars(external, column):
    aggregate = column == 'six'
    scene_index = None if aggregate else SCENES.index(column)
    title = 'Six-scenario recognition accuracy' if aggregate else 'Recognition accuracy: ' + SCENE_NAMES[scene_index]
    fig, ax = canvas(title, 'F0 benchmark; 6 TX, 7 unseen RX, 168,000 packets per scene; n = 3',
                     115, (.105, .275, .845, .55))
    ax.set(xlim=(-.58, 4.72), ylim=(0, 100), yticks=[0, 20, 40, 60, 80, 100],
           ylabel='Test accuracy (%)', xticks=[0, 1, 2, 3, 4.16],
           xticklabels=[DISPLAY[f] for f in FAMILIES] + ['PAMF'])
    ax.tick_params(axis='x', length=0, pad=8)
    grid(ax)
    ax.axvline(3.65, color='#D6DDE6', lw=.6, ls=(0, (3, 4)))
    for family, x in zip((*FAMILIES, 'pamf'), [0, 1, 2, 3, 4.16]):
        for regime in (('pamf',) if family == 'pamf' else ('full', 'full_pl')):
            samples = external[family, regime]
            vals = samples[:, 1:].mean(axis=1) if aggregate else samples[:, scene_index]
            mean, sd = describe(vals)
            dx = 0 if family == 'pamf' else -.17 if regime == 'full' else .17
            c = COLORS[family]
            ax.bar(x + dx, mean, width=.285, yerr=sd,
                   fc=pale(c) if regime == 'full' else c, ec=c, lw=.8,
                   hatch='//' if regime == 'full' else None,
                   error_kw=dict(ecolor=INK, elinewidth=.8, capsize=2, capthick=.75), zorder=3)
            # Pair labels occupy fixed separate horizontal slots.
            ax.text(x + dx, mean + sd + 2, f'{mean:.2f}', ha='center', va='bottom',
                    color=c, fontsize=7.1, weight='bold' if family == 'pamf' else 'normal')
    for tick, family in zip(ax.get_xticklabels(), (*FAMILIES, 'pamf')):
        tick.set_color(COLORS[family]); tick.set_fontsize(8)
        if family == 'pamf': tick.set_weight('bold')
    regime_legend(fig)
    note(fig, 'Bars: mean; whiskers: sample SD across model seeds. Full and Full + PL are paired settings.',
         CURRENT_NOTE)
    return fig


def heatmap(title, subtitle, means, labels, columns, row_colors=None,
            sd=None, highlight=(), foot='', second='', height=130, colorbar_label='Accuracy (%)'):
    fig, ax = canvas(title, subtitle, height, (.285, .255, .665, .56))
    image = ax.pcolormesh(np.arange(means.shape[1] + 1) - .5,
        np.arange(means.shape[0] + 1) - .5, means, cmap=CMAP, vmin=0, vmax=100,
        shading='flat', edgecolors='white', linewidth=.7, rasterized=False)
    ax.set(xlim=(-.5, means.shape[1] - .5), ylim=(means.shape[0] - .5, -.5))
    ax.set(xticks=range(len(columns)), xticklabels=columns,
           yticks=range(len(labels)), yticklabels=labels)
    ax.tick_params(length=0, pad=7)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.1)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for y in range(len(labels)):
        for x in range(len(columns)):
            c = contrast_color(CMAP(means[y, x] / 100))
            text = f'{means[y, x]:.1f}'
            if sd is not None: text += f'\n± {sd[y, x]:.1f}'
            ax.text(x, y, text, ha='center', va='center', color=c,
                    fontsize=7.2 if sd is not None else 7.6,
                    weight='bold' if y in highlight else 'normal', linespacing=1.4)
    if row_colors:
        for tick, c in zip(ax.get_yticklabels(), row_colors): tick.set_color(c)
    for y in highlight:
        ax.get_yticklabels()[y].set_weight('bold')
        ax.add_patch(Rectangle((-.5, y - .5), len(columns), 1, fill=False,
                               ec=COLORS['pamf'], lw=1, clip_on=False))
    cax = fig.add_axes((.34, .151, .54, .015), label='colorbar')
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, 20, 40, 60, 80, 100])
    cb.set_label(colorbar_label, fontsize=7.1, labelpad=4)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.ax.tick_params(length=2, labelsize=6.6, pad=2)
    note(fig, foot, second)
    return fig


def restoration(t1):
    fig, ax = canvas('Restoration benefit across observation conditions',
        'Frozen T1-E50 recognizer; DURA v1.0; F0; three matched F seeds', 121,
        (.105, .28, .845, .54))
    x = np.arange(7)
    for readout, dx, c, label in [('raw', -.17, COLORS['baseline'], 'Raw T1-E50'),
                                ('DURA-LU', .17, COLORS['pamf'], 'T1-E50 + DURA')]:
        vals = t1[readout]
        means, sds = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        ax.bar(x + dx, means, width=.29, yerr=sds, color=pale(c) if readout == 'raw' else c,
               ec=c, lw=.7, hatch='//' if readout == 'raw' else None, label=label,
               error_kw=dict(ecolor=INK, elinewidth=.75, capsize=2))
        for i in range(7):
            ax.text(i + dx, means[i] + sds[i] + 1.6, f'{means[i]:.1f}',
                    ha='center', va='bottom', fontsize=6.8, color=c)
    ax.set(ylim=(0, 100), yticks=[0, 20, 40, 60, 80, 100], xticks=x,
           xticklabels=SCENE_SHORT, ylabel='Test accuracy (%)')
    ax.tick_params(axis='x', length=0, pad=8)
    grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.53, .153), ncol=2)
    note(fig, 'Same 168,000 physical packets per scene. Whiskers: sample SD across three recognizer seeds.',
         'One fixed G is shared across F seeds. The seven correlated views are not independent repeats.')
    return fig


def latency_plot(timing, cases, names, title, subtitle, limits, colors):
    fig, ax = canvas(title, subtitle, 115, (.30, .245, .48, .56))
    ax.set(xlim=limits, ylim=(len(cases) - .45, -.6), yticks=range(len(cases)),
           yticklabels=names, xlabel='GPU wall time per request (ms)')
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=8)
    grid(ax, 'x')
    for i, (case, c) in enumerate(zip(cases, colors)):
        row = timing[case, 1]
        med, p95 = float(row['median_ms']), float(row['P95_ms'])
        ax.plot([med, p95], [i, i], color=c, lw=1.3)
        ax.scatter([med], [i], color=c, marker='o', s=24, zorder=4)
        ax.scatter([p95], [i], color=c, marker='|', s=55, linewidths=1.3, zorder=4)
        ax.text(1.08, i, f'{med:.3f} / {p95:.3f}', transform=ax.get_yaxis_transform(),
                fontsize=7.1, va='center', color=c)
        if 'reuse' in case: ax.get_yticklabels()[i].set_color(c)
    fig.legend(handles=[Line2D([], [], marker='o', color=INK, ls='', ms=4, label='Pooled median'),
                        Line2D([], [], marker='|', color=INK, ls='', ms=7, label='Pooled P95')],
               loc='center', bbox_to_anchor=(.53, .15), ncol=2)
    note(fig, 'Fixed seed01 models; batch = 1; five timing rounds. P95 is a request quantile, not a confidence bound.',
         'Exclusive RTX 3090; FP32, TF32 off. GPU wall excludes transfers; this is not an end-to-end boundary.')
    return fig


def export_figure(fig, out, stem, title, caption, sources, section, preview_only=False, raster_dpi=2400):
    if sum(ax.get_label() != 'colorbar' for ax in fig.axes) != 1:
        raise ValueError('Exactly one data chart is required per image')
    fig.canvas.draw()
    renderer, canvas_box = fig.canvas.get_renderer(), fig.get_window_extent()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        bbox = text.get_window_extent(renderer)
        if bbox.width and bbox.height and (bbox.x0 < -1 or bbox.y0 < -1
                or bbox.x1 > canvas_box.x1 + 1 or bbox.y1 > canvas_box.y1 + 1):
            raise ValueError(stem + ': text outside canvas: ' + repr(text.get_text()))
    for fmt in ('svg', 'pdf'):
        fig.savefig(out / fmt / (stem + '.' + fmt))
    fig.savefig(out / 'preview' / (stem + '.png'), dpi=200)
    # Grayscale QA is rendered by the same selected Python backend.
    with Image.open(out / 'preview' / (stem + '.png')) as image:
        image.convert('L').save(out / 'preview' / (stem + '_grayscale.png'))
    svg = ET.parse(out / 'svg' / (stem + '.svg')).getroot()
    ns = '{http://www.w3.org/2000/svg}'
    if not svg.findall('.//' + ns + 'text') or svg.findall('.//' + ns + 'image'):
        raise ValueError('SVG must retain editable text and vector chart marks')
    pdf = PdfReader(out / 'pdf' / (stem + '.pdf'))
    extracted = pdf.pages[0].extract_text()
    if len(pdf.pages) != 1 or len(extracted) < 50:
        raise ValueError('PDF text extraction failed')
    if abs(float(pdf.pages[0].mediabox.width) / 72 * 25.4 - WIDTH_MM) > .05:
        raise ValueError('Inconsistent final figure width')
    if 'CVS' in extracted:
        raise ValueError('Obsolete ambiguous model label in figure')
    pixels = None
    if not preview_only:
        # Render the selected Python canvas once, then encode two lossless formats.
        # Reset the canvas after each image so large buffers do not accumulate.
        original_dpi = fig.dpi
        fig.set_dpi(raster_dpi)
        fig.canvas.draw()
        size = fig.canvas.get_width_height()
        raster = Image.frombuffer('RGBA', size, fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
        rgb = raster.convert('RGB')
        rgb.save(out / 'png' / (stem + '.png'), dpi=(raster_dpi, raster_dpi), compress_level=4)
        rgb.save(out / 'tiff' / (stem + '.tiff'), dpi=(raster_dpi, raster_dpi), compression='tiff_lzw')
        del rgb, raster
        fig.set_dpi(original_dpi)
        fig.canvas.draw()
        gc.collect()
        # Header inspection is sufficient; these are our own known-size renderings.
        # Raise Pillow's pixel threshold only to the exact expected output size.
        Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, size[0] * size[1])
        with Image.open(out / 'png' / (stem + '.png')) as image:
            pixels = list(image.size)
            if abs(image.info.get('dpi', [0])[0] - raster_dpi) > 1: raise ValueError('PNG DPI changed')
        with Image.open(out / 'tiff' / (stem + '.tiff')) as image:
            if (image.size != tuple(pixels) or image.info.get('compression') != 'tiff_lzw'
                    or abs(image.info.get('dpi', [0])[0] - raster_dpi) > 1):
                raise ValueError('TIFF resolution/compression changed')
    record = dict(stem=stem, title=title, section=section, caption=caption,
        source_tables=sources, width_mm=WIDTH_MM, height_mm=fig.get_figheight() * 25.4,
        png_pixels=pixels, raster_dpi=raster_dpi, data_charts=1,
        editable_svg_text=True, extractable_pdf_text=True, no_canvas_clipping=True)
    plt.close(fig)
    print(('PREVIEW ' if preview_only else 'EXPORTED ') + stem, flush=True)
    return record


def render_all(data, out, preview_only=False, raster_dpi=2400):
    external, t1, g, history, hvalues, timing, deployment = validate(data)
    figures = []
    def save(fig, stem, title, caption, sources, section):
        figures.append(export_figure(fig, out, stem, title, caption, sources, section, preview_only, raster_dpi))
    external_caption = ('F0, three model seeds, 168000 packets per scene. Bars show means; '
        'whiskers show sample SD (ddof=1). Full and Full+plain PL are adjacent, and PAMF '
        'is a single T1-E50+fixed DURA system. The six-scenario mean is computed within each '
        'seed before the between-seed SD. No CE-only variants occur in this comparison. '
        'External training includes clean/satellite concatenation (CE weight 0.68); '
        'PAMF retains the native T1 schedule of 13L+37U. Budgets differ. ')
    for index, scene in enumerate(SCENES, 1):
        stem = f'A{index:02d}_{scene}'
        save(external_bars(external, scene), stem, SCENE_NAMES[index - 1] + ': Full / Full + PL',
             external_caption, ['external_per_seed.csv'], 'A Current T1-E50 + DURA')
    save(external_bars(external, 'six'), 'A08_six_scenario_comparison', 'Six-scenario comparison',
         external_caption, ['external_per_seed.csv'], 'A Current T1-E50 + DURA')
    keys = [(f, r) for f in FAMILIES for r in ('full', 'full_pl')] + [('pamf', 'pamf')]
    labels = [DISPLAY[f] + (' Full' if r == 'full' else ' Full + PL') if f != 'pamf' else 'PAMF (T1 + DURA)' for f, r in keys]
    means = np.stack([external[k].mean(axis=0) for k in keys])
    sds = np.stack([external[k].std(axis=0, ddof=1) for k in keys])
    save(heatmap('Recognition across seven observation conditions', 'F0 benchmark; Full / Full + PL; n = 3 model seeds',
         means, labels, SCENE_SHORT, [COLORS[k[0]] for k in keys], sds, (8,),
         'Cells: mean ± sample SD. All cells share the same absolute 0–100% color scale.', CURRENT_NOTE, 136),
         'A09_seven_scene_heatmap', 'Seven-scene comparison', external_caption,
         ['external_per_seed.csv'], 'A Current T1-E50 + DURA')
    deltas = [external[f, 'full_pl'][:, 1:].mean(axis=1) - external[f, 'full'][:, 1:].mean(axis=1) for f in FAMILIES]
    extent = max(6, math.ceil(max(max(float(np.abs(v).max()),
                 abs(describe(v)[0]) + describe(v)[1]) for v in deltas) / 2) * 2)
    save(forest('Effect of plain pseudo-labeling', 'Full + plain PL minus Full; six-scenario mean; three matched seeds',
         [DISPLAY[f] for f in FAMILIES], deltas, [COLORS[f] for f in FAMILIES],
         'Accuracy change (percentage points)', (-extent, extent), first_note='Differences are paired within model seed before aggregation; whiskers: sample SD.',
         second_note='PAMF has its native semi-supervised recipe and is not a Full + plain PL extension.', delta=True, label_width=.22),
         'A10_plain_pl_paired_gain', 'Plain PL paired gain',
         'Three same-seed differences per family; all four families retained. Sample SD, no inferential test.',
         ['external_per_seed.csv', 'paired_differences.csv'], 'A Current T1-E50 + DURA')
    # Direct method labels avoid a long marker key in the clean/robustness scatter.
    fig, ax = canvas('Clean accuracy and channel robustness', 'F0 benchmark; Full and Full + plain PL; n = 3',
                      125, (.12, .255, .81, .56))
    offsets = {'riei': (-14, -20), 'drift': (5, 6), 'ibrff': (-18, -20), 'sdg': (-11, 9), 'pamf': (-34, 10)}
    for family in (*FAMILIES, 'pamf'):
        regimes = ('pamf',) if family == 'pamf' else ('full', 'full_pl')
        points = []
        for regime in regimes:
            vals = external[family, regime]
            x, sx = describe(vals[:, 0]); y, sy = describe(vals[:, 1:].mean(axis=1))
            ax.errorbar(x, y, xerr=sx, yerr=sy, color=COLORS[family], fmt='D' if family == 'pamf' else 'o' if regime == 'full' else 's',
                        mfc='white' if regime == 'full' else COLORS[family], ms=5.3, capsize=2, lw=.8, zorder=4)
            points.append((x, y))
        if len(points) == 2: ax.plot(*np.asarray(points).T, color=COLORS[family], lw=.8, alpha=.65)
        x, y = np.mean(points, axis=0)
        ax.annotate('PAMF' if family == 'pamf' else DISPLAY[family], (x, y), xytext=offsets[family],
                    textcoords='offset points', color=COLORS[family], fontsize=7.4,
                    weight='bold' if family == 'pamf' else 'normal')
    ax.set(xlim=(40, 90), ylim=(0, 90), xticks=[40, 50, 60, 70, 80, 90],
           yticks=[0, 15, 30, 45, 60, 75, 90], xlabel='Clean accuracy (%)', ylabel='Six-scenario mean accuracy (%)')
    grid(ax, 'both')
    fig.legend(handles=[Line2D([], [], marker='o', color=MUTED, mfc='white', ls='', label='Full'),
        Line2D([], [], marker='s', color=MUTED, ls='', label='Full + plain PL'),
        Line2D([], [], marker='D', color=COLORS['pamf'], ls='', label='PAMF')],
        loc='center', bbox_to_anchor=(.53, .15), ncol=3)
    note(fig, 'Whiskers: sample SD in each coordinate. Lines connect settings within a method family.', CURRENT_NOTE)
    save(fig, 'A11_clean_robustness_balance', 'Clean accuracy versus robustness', external_caption,
         ['external_per_seed.csv'], 'A Current T1-E50 + DURA')
    save(restoration(t1), 'A12_restoration_seven_scenes', 'T1-E50 restoration benefit',
         'Same F seeds and same physical test packets; raw T1-E50 versus T1-E50+one fixed DURA. '
         'Seven views are correlated. Whiskers are across F seeds conditional on the fixed G.',
         ['t1_per_seed.csv'], 'A Current T1-E50 + DURA')
    gain = t1['DURA-LU'] - t1['raw']
    save(forest('Paired restoration gain', 'T1-E50 + DURA minus raw T1-E50; matched F seeds and packets',
         SCENE_NAMES, list(gain.T), [COLORS['pamf']] * 7, 'Accuracy change (percentage points)',
         (-1, 19), first_note='Mean and sample SD of three within-seed differences; no confidence intervals.',
         second_note='One fixed DURA is shared across F seeds. No target feedback enters training or model selection.',
         delta=True, label_width=.27), 'A13_restoration_paired_gain', 'Paired restoration gain',
         'Within-seed accuracy differences on all seven views. No seed or scenario is selected by observed gain.',
         ['t1_per_seed.csv', 'paired_differences.csv'], 'A Current T1-E50 + DURA')
    order = ['default'] + [k for k in g if k not in ('raw', 'default')]
    # Preserve original arm IDs in source data; display names do not imply a new model.
    for key in order:
        if key not in ARMS: ARMS[key] = key.replace('_', ' ')
    save(forest('DURA mechanism changes from the default', 'Six-scenario mean; same F seeds; seven G variants and fixed default reference',
         [ARMS[k] for k in order], [(g[k] - g['default'])[:, 1:].mean(axis=1) for k in order],
         [COLORS['pamf']] + [COLORS['baseline']] * (len(order) - 1),
         'Accuracy change from default DURA (percentage points)', (-8, 1), (0,),
         'Within-F-seed differences; mean ± sample SD (n = 3). Default reference is zero by definition.',
         'Student SD mixes F/G variation. This is not a complete initialization × distillation interaction.',
         delta=True, label_width=.33), 'A14_dura_ablation_accuracy', 'DURA mechanism comparison',
         'New G student seed and F seed are paired. Default G is shared across F seeds. '
         'Values are same-F-seed differences from default DURA, with their paired sample SD. '
         'Differences are descriptive and do not complete the missing replicated copy+KD cell.',
         ['g_ablation_per_seed.csv', 'paired_differences.csv'], 'A Current T1-E50 + DURA')
    save(heatmap('DURA mechanisms across seven conditions', 'Same frozen T1-E50 readouts; n = 3 paired repeats per student variant',
         np.stack([g[k].mean(axis=0) for k in order]), [ARMS[k] for k in order], SCENE_SHORT,
         [COLORS['pamf']] + [COLORS['baseline']] * (len(order) - 1),
         sd=np.stack([g[k].std(axis=0, ddof=1) for k in order]), highlight=(0,),
         foot='Cells: mean ± sample SD; 168,000 packets per scene. Default uses a single fixed G.',
         second='Matched F/G student repeats mix both sources of variability; not a pure G variance estimate.', height=137),
         'A15_dura_ablation_heatmap', 'DURA seven-scene mechanism comparison',
         'Same G conditions as A14, displayed without collapsing the six practical scenarios.',
         ['g_ablation_per_seed.csv'], 'A Current T1-E50 + DURA')
    cases = [f + '_full' for f in FAMILIES] + ['T4_graph_reuse_GF']
    save(latency_plot(timing, cases, [DISPLAY[f] + ' Full' for f in FAMILIES] + ['PAMF (T1 + DURA)'],
         'Complete-system inference latency', 'Fixed seed01; external Full methods; PAMF Graph + reuse; B1',
         (0, 6), [COLORS[f] for f in FAMILIES] + [COLORS['pamf']]),
         'A16_system_latency', 'Complete-system latency',
         'Pooled synchronized GPU-wall median and P95 over five timing rounds, fixed seed01. '
         'External supervised Full models only, no PL timing implied. Timing boundaries and execution paths differ '
         'from historical end-to-end measurements; no cross-panel speedup is inferred.',
         ['latency_summary.csv'], 'A Current T1-E50 + DURA')
    routes = ['T0_native_GF', 'T1_tensor_GF', 'T2_reuse_GF', 'T3_graph_GF', 'T4_graph_reuse_GF']
    save(latency_plot(timing, routes, ['Native', 'Tensor', 'Reuse', 'Graph', 'Graph + reuse'],
         'Execution-route latency with fixed weights', 'Same T1-E50 + DURA weights; FP32; B1; five timing rounds',
         (0, 55), [COLORS['baseline']] * 4 + [COLORS['pamf']]),
         'A17_execution_route_latency', 'Execution-route latency',
         'No model weights or precision change. B1 native median 37.189997 ms and Graph+reuse median '
         '4.541187 ms imply an 8.189487-fold route-specific reduction in wall time. '
         'These measurements alone do not establish full-data route prediction equivalence.',
         ['latency_summary.csv'], 'A Current T1-E50 + DURA')
    fig, ax = canvas('Throughput across inference batch sizes', 'Complete-system GPU-wall throughput; fixed seed01; exclusive RTX 3090',
                     117, (.12, .255, .81, .56))
    batches = [1, 8, 32, 64, 128]
    for case, family in zip(cases, (*FAMILIES, 'pamf')):
        y = [float(timing[case, b]['throughput_IQ_s']) / 1000 for b in batches]
        ax.plot(batches, y, marker='D' if family == 'pamf' else 'o', color=COLORS[family],
                label='PAMF' if family == 'pamf' else DISPLAY[family], ms=3.7, lw=1.2)
    ax.set(xscale='log', xlim=(.9, 145), ylim=(0, None), xticks=batches,
           xlabel='Batch size (log scale)', ylabel='Throughput (thousand IQ/s)')
    ax.set_xticklabels([str(b) for b in batches]); ax.minorticks_off(); grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.53, .15), ncol=5)
    note(fig, 'Throughput = batch size / pooled median GPU wall time. Lines connect measured batch sizes only.',
         'Batch-dependent execution is not assumed monotonic. Timing rounds are not model-seed repetitions.')
    save(fig, 'A18_batch_throughput', 'Batch throughput',
         'Five measured batch sizes; throughput recomputed from batch/median. No smoothing or extrapolation.',
         ['latency_summary.csv'], 'A Current T1-E50 + DURA')

    # Historical E200 comparisons remain their own family and sample-size context.
    order_h = sorted(HISTORICAL, key=lambda k: -hvalues(k).mean())
    hlabels = [DISPLAY[k] for k in order_h]
    hcolors = [color(k) for k in order_h]
    hhero = [i for i, k in enumerate(order_h) if k in ('sig_time', 'residual_fusion')]
    for number, field, title in [(1, 'accuracy', 'Historical clean-test accuracy'),
                                  (2, 'macro_f1', 'Historical clean-test macro-F1')]:
        save(forest(title, 'Fourteen architectures; E200; eight completed seeds per architecture',
             hlabels, [hvalues(k, field) for k in order_h], hcolors,
             'Accuracy (%)' if field == 'accuracy' else 'Macro-F1 (%)', (0, 100), hhero,
             'Every completed seed is shown, including collapsed runs. Large markers: mean ± sample SD.', HIST_NOTE,
             label_width=.325, height=158), f'B{number:02d}_historical_{field}', title,
             'Historical pure-CE E200, 10000 updates, same 168000 clean physical test packets; '
             '8 seeds for all 14 architectures. SIG_TIME has 196001 parameters; residual has 164225. '
             'Ordering by accuracy is only for display; no runs or outliers are dropped.',
             ['historical_per_seed.csv', 'historical_summary.csv'], 'B Historical E200')
    rx = [r['stratum'] for r in data['history'][0]['results'] if r['dimension'] == 'receiver']
    save(heatmap('Historical accuracy across unseen receivers', 'E200; eight model seeds; 24,000 packets per RX and model',
         np.array([[hvalues(k, dim='receiver', stratum=r).mean() for r in rx] for k in order_h]),
         hlabels, rx, hcolors, highlight=hhero,
         foot='Cells: mean accuracy (%). Receiver IDs are physical RX, distinct from transmitter labels.',
         second=HIST_NOTE, height=158), 'B03_historical_receiver_heatmap', 'Historical receiver comparison',
         'Shared absolute 0–100% color scale for 14 networks and 7 RX; all 8 model seeds retained.',
         ['historical_per_seed.csv'], 'B Historical E200')
    # One log-scale chart; labels use explicit leader lines instead of an inset panel.
    fig, ax = canvas('Model size and historical clean accuracy', 'E200; eight model seeds; parameter counts include the default head',
                     139, (.11, .24, .83, .58))
    offsets_h = {'sig_time': (14, 9), 'residual_fusion': (12, -18),
        'resnet18': (-5, -21), 'densenet121': (-58, 24), 'resnet34': (-25, -43),
        'cvresnet18_1d': (-18, 38), 'resnet50': (-14, 52), 'vgg16_bn': (-7, 27),
        'bilstm3': (-20, -30), 'convnext_tiny': (10, -23), 'resnet50_2d': (34, 14),
        'resnet18_2d': (-86, -10), 'cvresnet18_2d': (28, -42), 'transformer6': (18, -34)}
    short_names = {'resnet18': 'R18-1D', 'resnet34': 'R34-1D', 'resnet50': 'R50-1D',
        'densenet121': 'Dense121', 'vgg16_bn': 'VGG16', 'bilstm3': 'BiLSTM',
        'convnext_tiny': 'ConvNeXt', 'resnet18_2d': 'R18-2D', 'resnet50_2d': 'R50-2D',
        'cvresnet18_1d': 'Complex R18-1D', 'cvresnet18_2d': 'Complex R18-2D',
        'transformer6': 'Transformer', 'residual_fusion': 'PAMF residual', 'sig_time': 'PAMF SIG_TIME'}
    for k in order_h:
        x, (m, sd), c = data['parameters'][k] / 1e6, describe(hvalues(k)), color(k)
        ax.errorbar(x, m, yerr=sd, fmt='D' if k in ('sig_time', 'residual_fusion') else 'o',
                    color=c, mfc='white' if k == 'residual_fusion' else c, ms=4.7, capsize=2, lw=.8)
        ax.annotate(short_names[k], (x, m), xytext=offsets_h[k], textcoords='offset points',
                    fontsize=6.9, color=c, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color=c, lw=.55, shrinkA=2, shrinkB=4))
    ax.set(xscale='log', xlim=(.10, 100), ylim=(0, 100), xticks=[.1, 1, 10, 100],
           xlabel='Registered parameters (millions; log scale)', ylabel='Clean accuracy (%)')
    ax.set_xticklabels(['0.1', '1', '10', '100']); ax.minorticks_off(); grid(ax, 'both')
    note(fig, 'Markers: mean accuracy; whiskers: sample SD (n = 8). Parameter counts are not total compute or latency.', HIST_NOTE)
    save(fig, 'B04_parameters_accuracy', 'Parameters versus historical accuracy',
         'A single logarithmic parameter axis with no compressed scale or inset; full 0–100% accuracy '
         'range retains the Transformer sample SD. Annotation leaders do not move the measured points.',
         ['historical_summary.csv'], 'B Historical E200')
    comparisons = [('ResNet-18', 'resnet18', 'resnet18_2d'),
        ('ResNet-50', 'resnet50', 'resnet50_2d'), ('Complex ResNet-18', 'cvresnet18_1d', 'cvresnet18_2d')]
    for index, comp, title in [(5, comparisons, 'Historical 1D versus 2D input'),
        (6, [('1D ResNet-18', 'cvresnet18_1d', 'resnet18'),
             ('2D ResNet-18', 'cvresnet18_2d', 'resnet18_2d')], 'Historical complex versus real models')]:
        vals = [hvalues(a) - hvalues(b) for _, a, b in comp]
        save(forest(title, 'Same-seed differences; E200; all eight seed pairs retained',
             [n for n, _, _ in comp], vals, [COLORS['baseline']] * len(comp),
             ('1D minus 2D' if index == 5 else 'Complex minus real') + ' accuracy (percentage points)',
             (-6, 18) if index == 5 else (-6, 6),
             first_note='Mean and sample SD of paired differences, not a difference of unrelated error bars.',
             second_note='Architecture/input adaptations are part of each comparison; 2D inputs use a 16 × 16 IQ reshape.',
             delta=True, label_width=.30), f'B{index:02d}_architecture_paired_difference', title,
             'Differences use the same seed in each paired network and retain all eight pairs. '
             'No significance test or new model selection is performed.',
             ['historical_per_seed.csv', 'paired_differences.csv'], 'B Historical E200')
    for index, dim, strata, title in [(7, 'receiver', rx, 'Historical PAMF residual: receiver accuracy'),
            (8, 'transmitter', data['classes'], 'Historical PAMF residual: transmitter recall')]:
        save(forest(title, 'Frozen residual-fusion E200 model; eight seeds; clean test', strata,
             [hvalues('residual_fusion', dim=dim, stratum=s) for s in strata],
             [COLORS['pamf']] * len(strata), 'Accuracy (%)' if dim == 'receiver' else 'Class recall (%)',
             (0, 100), first_note=('24,000 packets per physical RX and model.' if dim == 'receiver' else
             '28,000 packets per true TX and model; class accuracy is class recall.'),
             second_note='RX and TX identifiers are separate roles, even when their printed names match.',
             label_width=.20), f'B{index:02d}_residual_{dim}', title,
             'Residual model only; no current T1-E50 data are mixed into historical per-RX/per-TX results.',
             ['historical_per_seed.csv'], 'B Historical E200')
    fig, ax = canvas('Source validation and unseen-RX class recall',
        'Historical PAMF residual; same eight frozen E200 models; default head', 121,
        (.18, .255, .60, .56))
    classes = data['classes']; x = np.arange(len(classes))
    source_by_seed = {int(r['seed']): r for r in data['source_validation']}
    for role, dy, c in [('Source V', -.15, COLORS['baseline']), ('Unseen RX', .15, COLORS['pamf'])]:
        vals = np.array([[source_by_seed[s]['recall_pct'][cl] for cl in classes] for s in SEEDS8]) if role == 'Source V' else np.array([hvalues('residual_fusion', dim='transmitter', stratum=cl) for cl in classes]).T
        m, sd = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        ax.errorbar(m, x + dy, xerr=sd, fmt='o' if role == 'Source V' else 'D', ms=5,
                    color=c, mfc='white' if role == 'Source V' else c, lw=1,
                    capsize=2, label=role, clip_on=False)
        for i, value in enumerate(m):
            ax.text(1.08, i + dy, f'{value:.2f} ± {sd[i]:.2f}',
                    transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7.2, color=c)
    ax.set(ylim=(5.6, -.6), xlim=(0, 100), xticks=[0, 20, 40, 60, 80, 100],
           yticks=x, yticklabels=classes, ylabel='True transmitter (TX)', xlabel='Class recall (%)')
    ax.tick_params(axis='y', length=0, pad=8); ax.spines['left'].set_visible(False); grid(ax, 'x')
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.53, .15), ncol=2)
    note(fig, 'Mean ± sample SD (n = 8). Source V: 4,500 packets per TX; unseen RX test: 28,000 per TX.',
         'Source V and target are different physical samples; comparisons are paired by model seed, not by packet.')
    save(fig, 'B09_source_target_class_recall', 'Source versus target transmitter recall',
         'Source-validation recall is from existing frozen CPU diagnostics, not new inference. '
         'Different physical samples and role-specific denominators are explicit. '
         'The 14-7 and 20-19 source/target gaps are retained without claiming a causal mechanism.',
         ['source_validation.csv', 'historical_per_seed.csv'], 'B Historical E200')
    for index, model in [(10, 'residual_fusion'), (11, 'resnet50'), (12, 'sig_time')]:
        cm = np.stack([np.asarray(history[model, s, 'overall', 'ALL']['confusion'], float) for s in SEEDS8])
        normalized = cm / cm.sum(axis=2, keepdims=True) * 100
        title = DISPLAY[model] + ': historical clean confusion'
        fig = heatmap(title, 'Rows: true TX; columns: predicted TX; mean row-normalized frequency; n = 8',
            normalized.mean(axis=0), classes, classes, foot='Each true TX has 28,000 test packets per model. Values are percentages within each true class.',
            second=HIST_NOTE, height=128, colorbar_label='Row-normalized frequency (%)')
        fig.axes[0].set_xlabel('Predicted transmitter (TX)', labelpad=7)
        fig.axes[0].set_ylabel('True transmitter (TX)', labelpad=8)
        save(fig, f'B{index:02d}_{model}_confusion', title,
             'Each seed is row-normalized before averaging. These are model-mean confusion matrices, '
             'not ensemble predictions. Every model uses the same class order and 0–100% color scale.',
             ['historical_confusion.csv'], 'B Historical E200')
    # Deployment costs refer only to the 13 measured residual/comparison models.
    deployed = [k for k in order_h if k != 'sig_time']
    specs = [('B13_historical_latency', 1, 'p95_ms', 'Historical end-to-end P95 latency',
              'End-to-end P95 (ms)', 1, '2,400 pooled B1 requests per architecture, across eight frozen checkpoints.'),
        ('B14_historical_throughput', 128, 'end_to_end_throughput_samples_s', 'Historical end-to-end throughput',
              'Throughput (thousand IQ/s)', .001, '800 pooled B128 requests per architecture, across eight frozen checkpoints.'),
        ('B15_historical_gpu_memory', 1, 'gpu_peak_allocated_mib', 'Historical peak GPU allocation',
              'Peak PyTorch allocated memory (MiB)', 1, 'Maximum inference-window peak across eight checkpoints; not total device usage.'),
        ('B16_historical_cpu_memory', 1, 'cpu_peak_rss_mib', 'Historical peak process memory',
              'Peak process RSS (MiB)', 1, 'RSS includes the framework/runtime baseline; 1 ms sampling may miss narrower spikes.'),
        ('B17_historical_storage', 1, 'model_storage_mib', 'Historical FP32 model storage',
              'Serialized FP32 state size (MiB)', 1, 'Actual uncompressed state-dict bytes, including buffers and metadata; no optimizer.')]
    for stem, b, field, title, xlabel, scale, first in specs:
        vals = [float(deployment[k, b][field]) * scale for k in deployed]
        fig, ax = canvas(title, 'Thirteen measured E200 architectures; eight checkpoints; RTX 3090, FP32',
                         151, (.325, .24, .49, .57))
        ys = np.arange(len(deployed))
        ax.set(yticks=ys, yticklabels=[DISPLAY[k] for k in deployed],
               ylim=(len(deployed) - .45, -.6), xlabel=xlabel, xlim=(0, max(vals) * 1.12))
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=8); grid(ax, 'x')
        for i, (k, v) in enumerate(zip(deployed, vals)):
            c = color(k)
            if k == 'residual_fusion': ax.axhspan(i - .42, i + .42, color=pale(c, .065))
            ax.hlines(i, 0, v, colors=pale(c, .35), lw=1.5)
            ax.scatter(v, i, marker='D' if k == 'residual_fusion' else 'o', color=c, s=22, zorder=4, clip_on=False)
            ax.text(1.05, i, f'{v:.3f}' if field in ('p95_ms', 'model_storage_mib') else f'{v:.2f}',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=7.2, color=c)
            ax.get_yticklabels()[i].set_color(c)
        note(fig, first, 'Historical boundary includes CPU input, H2D, model and returned labels. No cross-panel speedup is inferred.'
             if field in ('p95_ms', 'end_to_end_throughput_samples_s') else
             'SIG_TIME was not measured in this panel and is omitted rather than assigned another model\'s cost.')
        save(fig, stem, title, first + ' No invented error bars. Original measurement boundary is preserved; '
             'SIG_TIME has no measured deployment result in this source panel.',
             ['historical_deployment.csv'], 'B Historical E200')
    return figures


def source_tables(data, source):
    """Publish consistently named, full-precision score tables alongside the figures."""
    external, t1, g, history, hvalues, timing, deployment = validate(data)
    write_csv(source / 'external_per_seed.csv', data['external'])
    write_csv(source / 't1_per_seed.csv', data['t1'])
    write_csv(source / 'g_ablation_per_seed.csv', data['g_ablation'])
    write_csv(source / 'latency_summary.csv', data['latency'])
    write_csv(source / 'historical_deployment.csv', [dict(r, network=DISPLAY[r['variant']]) for r in data['deployment']])
    rows, confusions = [], []
    for (variant, seed, dim, stratum), score in history.items():
        rows.append(dict(variant=variant, method=DISPLAY[variant], model_seed=seed,
            dimension=dim, stratum=stratum, count=int(np.asarray(score['confusion']).sum()),
            accuracy_pct=100 * score['accuracy'], macro_f1_pct=100 * score['macro_f1']))
        if dim == 'overall':
            cm = np.asarray(score['confusion'])
            for i, true in enumerate(data['classes']):
                for j, predicted in enumerate(data['classes']):
                    confusions.append(dict(variant=variant, model_seed=seed, true_tx=true,
                        predicted_tx=predicted, count=int(cm[i, j]),
                        true_class_count=int(cm[i].sum()), row_percentage=100 * cm[i, j] / cm[i].sum()))
    write_csv(source / 'historical_per_seed.csv', rows)
    write_csv(source / 'historical_confusion.csv', confusions)
    write_csv(source / 'historical_summary.csv', [dict(variant=k, method=DISPLAY[k],
        parameters=data['parameters'][k], n=8, accuracy_mean_pct=describe(hvalues(k))[0],
        accuracy_sample_sd_pp=describe(hvalues(k))[1], macro_f1_mean_pct=describe(hvalues(k, 'macro_f1'))[0],
        macro_f1_sample_sd_pp=describe(hvalues(k, 'macro_f1'))[1]) for k in HISTORICAL])
    write_csv(source / 'source_validation.csv', [dict(model_seed=r['seed'], true_tx=c,
        count=4500, recall_pct=r['recall_pct'][c]) for r in data['source_validation'] for c in data['classes']])
    summary = []
    for (family, regime), vals in external.items():
        for metric, a in [('clean', vals[:, 0]), ('six_practical_mean', vals[:, 1:].mean(axis=1)),
                          ('low_urban', vals[:, -1])]:
            m, sd = describe(a)
            summary.append(dict(method='PAMF' if family == 'pamf' else DISPLAY[family],
                regime=regime, metric=metric, n=3, mean_pct=m, sample_sd_pp=sd))
    write_csv(source / 'external_summary.csv', summary)
    paired = []
    for f in FAMILIES:
        vals = external[f, 'full_pl'][:, 1:].mean(axis=1) - external[f, 'full'][:, 1:].mean(axis=1)
        paired.extend(dict(comparison=DISPLAY[f] + ' Full+PL minus Full', view='six_practical_mean',
                           seed=s, accuracy_difference_pp=float(v)) for s, v in zip(SEEDS3, vals))
    for i, scene in enumerate(SCENES):
        paired.extend(dict(comparison='T1+DURA minus raw T1', view=scene, seed=s,
            accuracy_difference_pp=float(t1['DURA-LU'][j, i] - t1['raw'][j, i])) for j, s in enumerate(SEEDS3))
    for arm in g:
        if arm in ('raw', 'default'): continue
        differences = (g[arm] - g['default'])[:, 1:].mean(axis=1)
        paired.extend(dict(comparison=arm + ' minus default DURA', view='six_practical_mean',
            seed=s, accuracy_difference_pp=float(v)) for s, v in zip(SEEDS3, differences))
    for a, b in [('resnet18', 'resnet18_2d'), ('resnet50', 'resnet50_2d'),
        ('cvresnet18_1d', 'cvresnet18_2d'), ('cvresnet18_1d', 'resnet18'), ('cvresnet18_2d', 'resnet18_2d')]:
        paired.extend(dict(comparison=DISPLAY[a] + ' minus ' + DISPLAY[b], view='historical_clean',
            seed=s, accuracy_difference_pp=float(v)) for s, v in zip(SEEDS8, hvalues(a) - hvalues(b)))
    write_csv(source / 'paired_differences.csv', paired)


def finish(out, data, figures, raster_dpi):
    writer = PdfWriter()
    for f in figures:
        writer.append(str(out / 'pdf' / (f['stem'] + '.pdf')))
    with (out / 'all_figures.pdf').open('wb') as stream: writer.write(stream)
    if len(PdfReader(out / 'all_figures.pdf').pages) != len(figures):
        raise ValueError('Compiled PDF page coverage differs')
    index = [dict(figure=f['stem'], section=f['section'], title=f['title'],
                  source_tables='; '.join(f['source_tables']),
                  svg='svg/' + f['stem'] + '.svg', pdf='pdf/' + f['stem'] + '.pdf') for f in figures]
    write_csv(out / 'figure_index.csv', index)
    captions = ['# Unified experimental figure captions', '',
        'Every image contains exactly one data chart. All figures are 183 mm wide. '
        'Accuracy/recall and confusion frequencies use absolute percentage scales. '
        'Error bars are sample SD across the stated repeats, never confidence intervals.', '',
        '## A. Current T1-E50 + DURA', '',
        'PAMF means the final T1-E50 recognizer (13L+37U) plus DURA v1.0 in this section. '
        'Three F seeds share one fixed G, so error bars are conditional on that G. '
        'External methods retain Full and Full+plain PL only. '
        'No new experiments, inference, prediction, IQ, checkpoints or truth arrays are accessed.', '',
        '## B. Historical E200', '',
        'PAMF residual and PAMF SIG_TIME are explicit historical architectures, trained for E200 '
        'with pure CE and 8 seeds. They are not relabeled as the current T1-E50 system. '
        'Deployment measurements cover 13 models; SIG_TIME has no matching cost measurements. '
        'Historical end-to-end times and current GPU-wall times are separate boundaries.', '']
    for f in figures:
        captions += ['## ' + f['stem'] + ' — ' + f['title'], '', f['caption'], '',
                     'Source tables: ' + ', '.join(f['source_tables']) + '.', '']
    (out / 'captions_en.md').write_text('\n'.join(captions), encoding='utf-8')
    readme = f'''# PAMF 实验图表统一版

本包包含 {len(figures)} 张独立科研图。每张只有一个数据图表，统一为 183 mm 宽、Arial 字体、固定方法配色。
外部 Full 为浅色斜线，Full＋普通 PL 为同色实心，PAMF 固定为朱红色；热力图共享 0% 至 100% 绝对色阶。

- `all_figures.pdf`：按目录顺序浏览全部图，每页一图。
- `svg/`、`pdf/`：可编辑文字与矢量图形。
- `png/`、`tiff/`：{raster_dpi} dpi 输出；TIFF 使用无损 LZW。两个超高清格式分别打包。
- `preview/`：普通预览与灰度检查图。
- `figure_index.csv`：图号、主题、数据表及相对文件位置。
- `source_data/`：全精度逐种子结果、汇总和配对差；`captions_en.md` 提供完整英文图注。

A01 至 A18 为当前 T1-E50＋DURA、四种外部 Full／Full＋PL、恢复收益、G 消融和时延。
B01 至 B17 为历史 E200 八种子网络、RX／TX 分层、源域与目标域、混淆矩阵和部署成本。
历史 PAMF residual／SIG_TIME 与当前 PAMF T1-E50＋DURA 分别标注，不混算训练预算或种子数。

统计先在每个种子内计算六种 practical 场景均值，再跨种子计算均值和样本标准差（ddof=1）。
配对差先作同种子相减。七景共享物理样本，不是七次独立重复。默认 G 只有一个，不将三 F 的波动当作纯 G 方差。
当前时延为 GPU wall，历史时延为含传输及 CPU 标签返回的端到端口径，两者不能直接求加速比。
未完成的 17 契约和 F 消融矩阵不作为完整结果绘制；本包只使用已经评分的本地快照。

Python 重绘：`python reproduce.py --source source_data --output NEW_DIRECTORY --dpi {raster_dpi}`。
依赖：matplotlib、numpy、Pillow、pypdf。新输出目录必须不存在，原文件和已有图包保留。

快照时间：{data['snapshot_hkt']}。本次只读已有评分，未启动训练、预测、评分或服务器操作。
'''
    (out / 'README_zh.md').write_text(readme, encoding='utf-8')
    manifest = dict(status='VERIFIED', schema=data['schema'], snapshot_hkt=data['snapshot_hkt'],
        figure_count=len(figures), figures=figures, source_runs=dict(external=EXT_RUN,
        current=MAIN_RUN, history=list(HISTORY_RUNS)),
        validation=dict(score_aggregates_reconciled=True, paired_before_aggregation=True,
        all_completed_seeds_retained=True, single_data_chart_per_figure=True,
        shared_palette=True, shared_width_mm=WIDTH_MM, svg_text_editable=True,
        pdf_text_extractable=True, canvas_clipping_checked=True, raster_dpi=raster_dpi,
        no_model_evaluation=True, no_remote_mutation=True),
        limitations=['Model seeds are not independent data samples.',
        'Three F seeds share one fixed default G; G-student repeats pair F/G seeds.',
        'Current GPU wall and historical end-to-end timing boundaries differ.',
        'Partial current matrices are not promoted to complete results.'])
    write_json(out / 'manifest.json', manifest)
    shutil.copyfile(Path(__file__), out / 'reproduce.py')
    shutil.copyfile(Path(__file__).with_name('publication_style.py'), out / 'publication_style.py')
    archive = out.with_suffix('.zip')
    if archive.exists(): raise FileExistsError('Preserve existing ZIP: ' + str(archive))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file() and file.relative_to(out).parts[0] not in ('png', 'tiff'):
                z.write(file, file.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('ZIP readback failed')
    raster_archives = {}
    for folder, suffix in [('png', 'PNG'), ('tiff', 'TIFF')]:
        target = out.with_name(out.name + '-' + suffix + '-' + str(raster_dpi) + 'dpi.zip')
        if target.exists(): raise FileExistsError('Preserve existing raster ZIP: ' + str(target))
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as z:
            for file in sorted((out / folder).iterdir()):
                if file.is_file(): z.write(file, file.relative_to(out))
        with zipfile.ZipFile(target) as z:
            if len(z.namelist()) != len(figures) or z.testzip() is not None:
                raise ValueError('Raster ZIP coverage/readback failed')
        raster_archives[suffix] = target.stat().st_size
    print(json.dumps(dict(status='VERIFIED', figures=len(figures), pdf_pages=len(figures),
                          raster_dpi=raster_dpi, archive_bytes=archive.stat().st_size,
                          raster_archive_bytes=raster_archives), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--project-root', type=Path)
    group.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--preview-only', action='store_true', help='Render vectors/previews for visual QA before final raster export')
    p.add_argument('--dpi', type=int, default=2400, help='Lossless PNG/TIFF resolution; default: 2400 dpi')
    p.add_argument('--resume-render', action='store_true',
        help='Repair only this incomplete export, using its unchanged source snapshot')
    args = p.parse_args()
    data = collect(args.project_root.resolve()) if args.project_root else read_json(args.source / 'scores.json')
    validate(data)
    if args.validate_only:
        print(json.dumps(dict(status='VERIFIED', historical_models=14, historical_rows=112,
            external_cells=len(data['external']), timing_cells=len(data['latency']),
            g_cells=len(data['g_ablation'])), ensure_ascii=False))
        return
    out = args.output.resolve()
    repair = args.resume_render
    if repair and (not args.source or args.source.resolve() != (out / 'source_data').resolve()
                   or not out.exists() or (out / 'manifest.json').exists()):
        raise ValueError('Repair requires this incomplete output\'s own source_data; completed exports are immutable')
    if (out.exists() and not repair) or out.with_suffix('.zip').exists():
        raise FileExistsError('Use a new output directory and archive name')
    out.mkdir(parents=True, exist_ok=repair)
    for folder in ('svg', 'pdf', 'png', 'tiff', 'preview', 'source_data'):
        (out / folder).mkdir(exist_ok=repair)
    if not repair:
        write_json(out / 'source_data/scores.json', data)
    source_tables(data, out / 'source_data')
    apply_style()
    if args.dpi < 300:
        raise ValueError('Publication raster exports require at least 300 dpi')
    figures = render_all(data, out, args.preview_only, args.dpi)
    if args.preview_only:
        write_json(out / 'draft_index.json', figures)
        print('PREVIEW_COMPLETE ' + str(len(figures)), flush=True)
    else:
        if (out / 'draft_index.json').exists():
            (out / 'draft_index.json').unlink()
        finish(out, data, figures, args.dpi)


if __name__ == '__main__':
    main()
