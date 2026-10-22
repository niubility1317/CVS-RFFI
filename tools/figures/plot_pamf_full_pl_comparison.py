"""Score-only Full / Full+PL / PAMF publication figures; no new model evaluation.

Use --source-csv from an exported bundle to redraw in a fresh output directory.
The figures use fixed F0 rows, three model seeds, and all seven declared scenes.
"""
from __future__ import annotations

import argparse
import csv
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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from pypdf import PdfReader

SEEDS = (2026092701, 2026092702, 2026092703)
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')
SCENES = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
NAMES = dict(riei='RIEI', drift='DRIFT', ibrff='IB-RFF', sdg='SDG', pamf='PAMF')
COLORS = dict(riei='#3F70AE', drift='#258F8F', ibrff='#8A6BB8',
              sdg='#B7832F', pamf='#CE4E43')
INK, MUTED, GRID = '#243247', '#647184', '#E8EDF1'
KEYS = tuple((family, regime) for family in FAMILIES
             for regime in ('full', 'full_pl')) + (('pamf', 'pamf'),)
EXT_RUN = '20261019-phase1-external-dg-concat-pl-m219-r03'
PAMF_RUN = '20261021-phase1-T1-DURA-comparison-m153-r01'
WIDTH_MM = 183
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 7.5,
    'axes.labelsize': 7.8, 'axes.titlesize': 8.5, 'axes.linewidth': .65,
    'axes.edgecolor': '#9BA7B6', 'axes.spines.right': False,
    'axes.spines.top': False, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': INK,
    'xtick.labelsize': 7, 'ytick.labelsize': 7.4,
    'xtick.major.width': .55, 'ytick.major.width': .55,
    'legend.frameon': False, 'legend.fontsize': 7,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.hashsalt': 'pamf-full-pl-figures'})


def describe(values):
    """Sample SD is across the three fixed model seeds, not across scenes."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def metric(values, name):
    values = np.asarray(values, dtype=float)
    if name == 'six':
        return values[:, 1:].mean(axis=1)
    return values[:, {'clean': 0, 'low': 6}[name]]


def arrays_from_records(records):
    expected = {(family, regime, seed, scene) for family, regime in KEYS
                for seed in SEEDS for scene in SCENES}
    found = {}
    for row in records:
        key = (row['family'], row['regime'], int(row['seed']), row['scene'])
        if key not in expected or key in found:
            raise ValueError('Unexpected or duplicate method/seed/scene: ' + str(key))
        value = float(row['accuracy_pct'])
        if not math.isfinite(value) or not 0 <= value <= 100 or int(row['count']) != 168000:
            raise ValueError('Invalid score, percentage scale, or F0 packet count')
        f1 = float(row['macro_f1_pct'])
        if not math.isfinite(f1) or not 0 <= f1 <= 100:
            raise ValueError('Invalid Macro-F1')
        found[key] = value
    if set(found) != expected:
        raise ValueError('Incomplete Full / Full+PL / PAMF matrix')
    return {(family, regime): np.array([[found[family, regime, seed, scene]
        for scene in SCENES] for seed in SEEDS]) for family, regime in KEYS}


def pl_delta(arrays, family):
    return metric(arrays[family, 'full_pl'], 'six') - metric(arrays[family, 'full'], 'six')


def load_sources(external_root, pamf_csv):
    """Read audited scores only. Do not open IQ, weights, predictions, or truth."""
    records = []
    for family in FAMILIES:
        for regime in ('full', 'full_pl'):
            suffix = '-pl' if regime == 'full_pl' else ''
            for seed in SEEDS:
                row_id = f'F0-{family}-full{suffix}-s{seed}'
                item = json.loads((external_root / (row_id + '.json')).read_text(encoding='utf-8-sig'))
                config = item['config']
                if not (item['status'] == 'VERIFIED' and item['run_id'] == EXT_RUN
                        and config['row_id'] == row_id and config['variant'] == 'full'
                        and config['family'] == family and config['model_seed'] == seed
                        and config['training_regime'] == ('pl' if suffix else 'supervised')
                        and item['all_predictions_preflight_before_truth']
                        and item['independent_confusion_and_metrics_match']
                        and item['default_head_only']):
                    raise ValueError('External audited-row identity or completion differs')
                for scene in SCENES:
                    values = [v for v in item['results'] if v['test'] == 'main'
                        and v['view'] == scene and v['dimension'] == 'overall'
                        and v['stratum'] == 'ALL']
                    if len(values) != 1:
                        raise ValueError('Missing/duplicated overall result')
                    value = values[0]
                    cm = np.asarray(value['confusion'], dtype=np.int64)
                    if cm.shape != (6, 6) or cm.sum() != 168000:
                        raise ValueError('F0 confusion matrix differs')
                    accuracy = 100 * np.trace(cm) / cm.sum()
                    if abs(accuracy - value['accuracy'] * 100) > 1e-9:
                        raise ValueError('Accuracy and audited confusion matrix differ')
                    rx = [v for v in item['results'] if v['test'] == 'main'
                        and v['view'] == scene and v['dimension'] == 'receiver']
                    if len(rx) != 7 or any(v['count'] != 24000 for v in rx):
                        raise ValueError('F0 receiver-balanced coverage differs')
                    if abs(np.mean([v['accuracy'] for v in rx]) * 100 - accuracy) > 1e-9:
                        raise ValueError('RX-balanced and packet accuracy differ')
                    records.append(dict(family=family, regime=regime, seed=seed,
                        scene=scene, accuracy_pct=float(accuracy),
                        macro_f1_pct=float(value['macro_f1'] * 100), count=168000,
                        source_run=EXT_RUN, source_row=row_id))
    with pamf_csv.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['readout'] != 'DURA-LU':
                continue
            count, correct = int(row['count']), int(row['correct'])
            if abs(float(row['accuracy']) - 100 * correct / count) > 1e-9:
                raise ValueError('PAMF count and percentage differ')
            records.append(dict(family='pamf', regime='pamf', seed=int(row['seed']),
                scene=row['view'], accuracy_pct=float(row['accuracy']),
                macro_f1_pct=float(row['macro_f1']), count=count,
                source_run=PAMF_RUN, source_row=f"F0-native_cric-s{row['seed']}"))
    arrays_from_records(records)
    return records


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key in ('seed', 'count'):
            row[key] = int(row[key])
        for key in ('accuracy_pct', 'macro_f1_pct'):
            row[key] = float(row[key])
    arrays_from_records(records)
    return records


def write_csv(path, records):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def label(key):
    family, regime = key
    return 'PAMF (ours)' if family == 'pamf' else NAMES[family] + (' Full + PL' if regime == 'full_pl' else ' Full')


def marker(key):
    return 'D' if key[0] == 'pamf' else ('s' if key[1] == 'full_pl' else 'o')


def legend_handles():
    return [Line2D([], [], marker='o', ls='none', color=MUTED, mfc='white', ms=4.7,
                   label='Full'),
            Line2D([], [], marker='s', ls='none', color=MUTED, mfc=MUTED, ms=4.7,
                   label='Full + plain PL'),
            Line2D([], [], marker='D', ls='none', color=COLORS['pamf'], ms=5,
                   label='PAMF (T1 + DURA)')]


def figure_header(fig, title, subtitle):
    fig.text(.045, .953, title, fontsize=11.8, weight='bold')
    fig.text(.045, .912, subtitle, fontsize=7, color=MUTED)


def shared_note(fig, line1):
    fig.text(.045, .066, line1, fontsize=6.5, color=MUTED)
    fig.text(.045, .030,
        'PAMF: three F seeds share one fixed G. Training recipes and budgets differ across methods.',
        fontsize=6.5, color=MUTED)


def accuracy_figure(arrays):
    """One forest plot, with seed points and a separate numeric label column."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 125 / 25.4), dpi=120)
    figure_header(fig, 'Recognition under channel shift',
        'F0 benchmark  |  Six practical scenarios  |  168,000 packets per scenario')
    ax = fig.add_axes((.235, .255, .545, .57), label='data')
    order = [('pamf', 'pamf')] + [(f, r) for f in ('sdg', 'riei', 'drift', 'ibrff')
                                 for r in ('full', 'full_pl')]
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, 8.6)
    ax.invert_yaxis()
    ax.set_yticks(range(9), [label(k) for k in order])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Six-scenario mean accuracy (%)', labelpad=7)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.grid(axis='x', color=GRID, lw=.55)
    ax.set_axisbelow(True)
    ax.axhspan(-.48, .48, color='#FDF0ED', zorder=0)
    for edge in (2.5, 4.5, 6.5):
        ax.axhline(edge, color='#F0F2F6', lw=.6, zorder=0)
    for y, key in enumerate(order):
        samples = metric(arrays[key], 'six')
        center, sd = describe(samples)
        color = COLORS[key[0]]
        ax.scatter(samples, y + np.array([-.16, 0, .16]), s=12, color=color,
                   alpha=.42, linewidths=0, zorder=3)
        ax.errorbar(center, y, xerr=sd, fmt=marker(key), color=color,
            mfc='white' if key[1] == 'full' else color,
            ms=6.0 if key[0] == 'pamf' else 5.0,
            mew=1.15, capsize=2.4, capthick=.9, elinewidth=1.15, zorder=5)
        ax.text(1.09, y, f'{center:.2f} ± {sd:.2f}', transform=ax.get_yaxis_transform(),
                color=color, fontsize=7.3,
                weight='bold' if key[0] == 'pamf' else 'normal', va='center')
    for tick, key in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[key[0]])
    ax.get_yticklabels()[0].set_weight('bold')
    fig.text(.045, .860, 'Method', color=MUTED, fontsize=7, weight='bold')
    fig.text(.829, .860, 'Mean ± SD (%)', color=MUTED, fontsize=7, weight='bold')
    fig.legend(handles=legend_handles(), loc='center', bbox_to_anchor=(.54, .146),
               ncol=3, handletextpad=.6, columnspacing=2.4)
    shared_note(fig, 'Small dots: individual seeds. Large markers and whiskers: mean ± sample SD (n = 3).')
    return fig


def balance_figure(arrays):
    """One scatter plot; both error-bar directions describe model-seed SD."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 130 / 25.4), dpi=120)
    figure_header(fig, 'Clean accuracy and channel robustness',
        'F0 benchmark  |  Full methods and plain-PL extensions  |  3 model seeds')
    ax = fig.add_axes((.12, .255, .81, .59), label='data')
    ax.set(xlim=(40, 90), ylim=(0, 90), xticks=[40, 50, 60, 70, 80, 90],
           yticks=[0, 15, 30, 45, 60, 75, 90], xlabel='Clean accuracy (%)',
           ylabel='Six-scenario mean accuracy (%)')
    ax.grid(color=GRID, lw=.55)
    ax.set_axisbelow(True)
    for family in FAMILIES:
        x = [metric(arrays[family, regime], 'clean').mean() for regime in ('full', 'full_pl')]
        y = [metric(arrays[family, regime], 'six').mean() for regime in ('full', 'full_pl')]
        ax.plot(x, y, color=COLORS[family], alpha=.35, lw=.85, zorder=2)
    for key in KEYS:
        color = COLORS[key[0]]
        x, xs = describe(metric(arrays[key], 'clean'))
        y, ys = describe(metric(arrays[key], 'six'))
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt=marker(key), color=color,
            mfc='white' if key[1] == 'full' else color,
            ms=6.7 if key[0] == 'pamf' else 5.5,
            mew=1.15, elinewidth=.95, capsize=2.5, zorder=4)
    offsets = dict(sdg=(8, 13), riei=(-31, 16), drift=(-17, 17),
                   ibrff=(12, -15), pamf=(-48, 13))
    for family in (*FAMILIES, 'pamf'):
        key = (family, 'pamf' if family == 'pamf' else 'full_pl')
        x = metric(arrays[key], 'clean').mean()
        y = metric(arrays[key], 'six').mean()
        ax.annotate(NAMES[family], (x, y), xytext=offsets[family], textcoords='offset points',
            color=COLORS[family], fontsize=8.3, weight='bold')
    fig.legend(handles=legend_handles(), loc='center', bbox_to_anchor=(.53, .145),
               ncol=3, handletextpad=.6, columnspacing=2.4)
    shared_note(fig, 'Markers: means; horizontal and vertical whiskers: sample SD. Lines join Full / Full + PL.')
    return fig


def pl_effect_figure(arrays):
    """One paired-difference plot, without an unrelated PAMF point or inset."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 100 / 25.4), dpi=120)
    figure_header(fig, 'Effect of plain pseudo-labeling',
        'Within-seed change in six-scenario mean accuracy  |  Full + PL minus Full')
    ax = fig.add_axes((.17, .30, .58, .49), label='data')
    deltas = [pl_delta(arrays, f) for f in FAMILIES]
    range_max = max(max(float(np.abs(v).max()), abs(describe(v)[0]) + describe(v)[1])
                    for v in deltas)
    extent = max(6, math.ceil(range_max / 2) * 2)
    ax.set(xlim=(-extent, extent), ylim=(3.55, -.55), xlabel='Accuracy change (percentage points)',
           yticks=range(4), yticklabels=[NAMES[f] for f in FAMILIES])
    ax.set_xticks(np.linspace(-extent, extent, 5))
    ax.grid(axis='x', color=GRID, lw=.55)
    ax.set_axisbelow(True)
    ax.axvline(0, color='#8A96A4', lw=.9, ls=(0, (3, 3)))
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    for y, (family, values) in enumerate(zip(FAMILIES, deltas)):
        center, sd = describe(values)
        color = COLORS[family]
        ax.scatter(values, y + np.array([-.14, 0, .14]), s=16, color=color,
                   alpha=.45, linewidths=0, zorder=3)
        ax.errorbar(center, y, xerr=sd, fmt='o', color=color, mfc='white',
                    ms=5.6, mew=1.2, elinewidth=1.2, capsize=2.7, zorder=4)
        ax.text(1.09, y, f'{center:+.2f} ± {sd:.2f}', transform=ax.get_yaxis_transform(),
                color=color, fontsize=7.5, va='center')
    for tick, family in zip(ax.get_yticklabels(), FAMILIES):
        tick.set_color(COLORS[family])
        tick.set_weight('bold')
    fig.text(.045, .838, 'Method', color=MUTED, fontsize=7, weight='bold')
    fig.text(.803, .838, 'Mean ± SD (pp)', color=MUTED, fontsize=7, weight='bold')
    handles = [Line2D([], [], marker='o', ls='none', color=MUTED, mfc='white',
                     ms=5, label='Mean ± sample SD'),
               Line2D([], [], marker='o', ls='none', color=MUTED, mfc=MUTED,
                     alpha=.5, ms=3, label='Individual matched seeds')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.51, .151),
               ncol=2, handletextpad=.6, columnspacing=2.5)
    fig.text(.045, .065,
        'Differences are computed within each matched seed before aggregation (n = 3).',
        fontsize=6.5, color=MUTED)
    fig.text(.045, .029, 'Whiskers are sample SD, not confidence intervals. No significance tests are shown.',
             fontsize=6.5, color=MUTED)
    return fig


def scenario_figure(arrays):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 126 / 25.4), dpi=120)
    figure_header(fig, 'Accuracy across seven observation conditions',
        'F0 benchmark  |  Clean, suburban and urban channels  |  168,000 packets per condition')
    ax = fig.add_axes((.245, .245, .715, .55), label='data')
    order = [(f, r) for f in FAMILIES for r in ('full', 'full_pl')] + [('pamf', 'pamf')]
    means = np.stack([arrays[key].mean(axis=0) for key in order])
    sds = np.stack([arrays[key].std(axis=0, ddof=1) for key in order])
    cmap = LinearSegmentedColormap.from_list('teal_ink',
        ['#F2F7F7', '#D4E8E8', '#9CCACE', '#539BA3', '#246676', '#123D51'])
    norm = Normalize(0, 90)
    image = ax.pcolormesh(np.arange(8) - .5, np.arange(10) - .5, means,
        cmap=cmap, norm=norm, edgecolors='white', linewidth=1.1, shading='flat', rasterized=False)
    ax.set(xlim=(-.5, 6.5), ylim=(8.5, -.5), xticks=range(7), yticks=range(9))
    ax.set_xticklabels(['Clean', 'High', 'Mid', 'Low', 'High', 'Mid', 'Low'])
    ax.set_yticklabels([label(key) for key in order], fontsize=7.1)
    ax.tick_params(axis='both', which='both', length=0, pad=7)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, key in enumerate(order):
        for x in range(7):
            rgba = cmap(norm(means[y, x]))
            rgb = np.array(rgba[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            dark_rgb = np.array(matplotlib.colors.to_rgb(INK))
            dark_linear = np.where(dark_rgb <= .04045, dark_rgb / 12.92,
                                   ((dark_rgb + .055) / 1.055) ** 2.4)
            dark_luminance = float(dark_linear @ np.array([.2126, .7152, .0722]))
            if 1.05 / (luminance + .05) >= 4.5:
                ink = 'white'
            elif (luminance + .05) / (dark_luminance + .05) >= 4.5:
                ink = INK
            else:
                ink = 'black'
            ax.text(x, y - .11, f'{means[y, x]:.1f}', ha='center', va='center',
                    fontsize=8.1, color=ink, weight='bold' if key[0] == 'pamf' else 'normal')
            ax.text(x, y + .21, f'± {sds[y, x]:.1f}', ha='center', va='center', fontsize=6.0, color=ink)
        if y in (1, 3, 5, 7):
            ax.axhline(y + .5, color='white', lw=3)
    ax.add_patch(Rectangle((-.5, 7.5), 7, 1, fill=False, ec=COLORS['pamf'], lw=1.1, clip_on=False))
    for tick, key in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[key[0]])
    ax.get_yticklabels()[-1].set_weight('bold')
    ax.plot([.6, 3.4], [-1.42, -1.42], color='#9DAABE', lw=.6, clip_on=False)
    ax.plot([3.6, 6.4], [-1.42, -1.42], color='#9DAABE', lw=.6, clip_on=False)
    ax.text(2, -1.77, 'Suburban', ha='center', va='center', fontsize=7.3, color=MUTED, clip_on=False)
    ax.text(5, -1.77, 'Urban', ha='center', va='center', fontsize=7.3, color=MUTED, clip_on=False)
    bar_ax = fig.add_axes((.31, .150, .59, .018), label='colorbar')
    colorbar = fig.colorbar(image, cax=bar_ax, orientation='horizontal', ticks=[0, 20, 40, 60, 80, 90])
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.tick_params(labelsize=6.3, length=2, width=.5)
    fig.text(.605, .093, 'Accuracy (%)  |  shared absolute scale', ha='center', fontsize=7, color=MUTED)
    fig.text(.045, .052, 'Cells: mean ± sample SD (n = 3). The seven views contain the same physical packets.',
             fontsize=6.5, color=MUTED)
    fig.text(.045, .022, 'PAMF: one fixed G across three F seeds. Recipes and budgets differ across methods.',
             fontsize=6.5, color=MUTED)
    return fig


def single_chart_count(fig):
    return sum(ax.get_label() != 'colorbar' for ax in fig.axes)


def export_figure(fig, out, stem, required_names=None):
    """Export editable vectors and verify final-size typography and clipping."""
    if single_chart_count(fig) != 1:
        raise ValueError('Each figure must contain exactly one data chart')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.get_window_extent()
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1):
            clipped.append(text.get_text())
    if clipped:
        raise ValueError('Text outside figure canvas: ' + repr(clipped))
    fig.savefig(out / (stem + '.svg'))
    fig.savefig(out / (stem + '.pdf'))
    fig.savefig(out / (stem + '.png'), dpi=600)
    fig.savefig(out / (stem + '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(out / 'preview' / (stem + '.png'), dpi=240)
    svg = ET.parse(out / (stem + '.svg')).getroot()
    svg_ns = '{http://www.w3.org/2000/svg}'
    if not svg.findall('.//' + svg_ns + 'text') or svg.findall('.//' + svg_ns + 'image'):
        raise ValueError('SVG must have editable text and vector data marks')
    pdf = PdfReader(out / (stem + '.pdf'))
    extracted = pdf.pages[0].extract_text()
    expected_names = list(NAMES.values()) if required_names is None else required_names
    if len(pdf.pages) != 1 or any(name not in extracted for name in expected_names):
        raise ValueError('PDF text coverage differs')
    width = float(pdf.pages[0].mediabox.width) / 72 * 25.4
    if abs(width - WIDTH_MM) > .05:
        raise ValueError('PDF final publication width differs')
    with Image.open(out / (stem + '.png')) as image:
        if abs(image.info['dpi'][0] - 600) > 1:
            raise ValueError('PNG is not 600 dpi')
        pixels = list(image.size)
    with Image.open(out / (stem + '.tiff')) as image:
        if (image.size != tuple(pixels) or abs(image.info['dpi'][0] - 600) > 1
                or image.info.get('compression') != 'tiff_lzw'):
            raise ValueError('TIFF size, resolution, or lossless compression differs')
    with Image.open(out / 'preview' / (stem + '.png')) as image:
        image.convert('L').save(out / 'preview' / (stem + '_grayscale.png'))
    result = dict(stem=stem, width_mm=WIDTH_MM, height_mm=fig.get_figheight() * 25.4,
        png_pixels=pixels, data_charts=1, editable_svg_text=True, vector_marks=True,
        extractable_pdf_text=True, no_canvas_clipping=True, raster_dpi=600,
        lossless_tiff_verified=True)
    plt.close(fig)
    return result


CAPTIONS = """# Figure captions

## Figure 1 — Six-scenario recognition accuracy

Cross-receiver recognition on the fixed F0 benchmark. Accuracy is first averaged
over six preregistered practical channel scenarios within each model seed.
Large markers and whiskers show means and sample standard deviations (ddof=1)
across model seeds 2026092701/02/03; small dots show all three seed values.
No inferential tests, confidence intervals, or significance symbols are used.
Each exported figure contains exactly one data chart, without composite panels.

The external comparison contains four families, Full and Full + plain PL; no own-CE,
ResNet CE, or raw-T1 baseline is plotted. Full includes each implemented native
recipe; external Full and Full+PL share clean/satellite concatenation with satellite
CE weight 0.68. Plain PL uses a fixed confidence threshold of 0.95 and weight 1.
PAMF is the fixed T1-E50 (13L+37U) recognizer and DURA v1.0, not an added plain-PL
variant. Three F seeds share one fixed G, so the PAMF SD is conditional on that G.
Recipes and native budgets differ; this is not an equal-compute or single-factor
comparison and does not estimate a pure G variance.

F0 contains six TX, five source RX (1,3,4,6,8), and seven disjoint target RX
(0,2,5,7,9,10,11). Source dates are D1–D3; L/U/V=6300/56700/27000. Split seed is
392005. Each scene covers the same 168000 physical packets; each target RX has
24000 packets, so RX-balanced and packet-weighted accuracy coincide. The seven
correlated views do not add independent model repeats. Prediction completion and
independent truth-last scoring precede this score-only rendering. No IQ,
checkpoint, prediction-array or truth access, model inference, training, tuning,
reranking, or default-model replacement occurs during figure preparation.

## Figure 2 — Clean accuracy and channel robustness

The horizontal coordinate is clean accuracy; the vertical coordinate is the same
six-scenario mean used in Figure 1. Markers show means over three model seeds.
Horizontal and vertical whiskers show sample SD for the respective metric.
Thin lines identify the Full and Full + plain PL means from the same family;
they are not fitted regressions. Hollow circles denote Full, filled squares denote
Full + plain PL, and the vermilion diamond denotes PAMF (T1 + DURA). Scientific
scope and limitations are the same as for Figure 1.

## Figure 3 — Effect of plain pseudo-labeling

The within-seed difference of Full + plain PL minus Full in six-scenario mean
accuracy, in percentage points. Small dots show all three matched-seed differences;
large hollow markers and whiskers show their mean and sample SD (ddof=1).
The zero reference denotes no difference. The four families are retained regardless
of the sign or size of the observed mean change. No confidence interval or
significance test is implied by the whiskers. PAMF is not a plain-PL extension and
therefore has no point in this paired-comparison chart.

## Figure 4 — Seven-scene decomposition

Each cell shows accuracy mean and model-seed sample SD for clean and all six practical
scenarios, using one shared absolute color scale without per-row normalization.
Suburban and urban High/Mid/Low columns follow the fixed preregistered scenarios.
PAMF is outlined for identification; its cells use the same color normalization as
all external methods. The colorbar is a scale legend, not a second data chart.
All sample-size, data-contract, and comparison limitations from Figure 1 apply.

## Visual encoding

RIEI is blue, DRIFT teal, IB-RFF violet, SDG ochre, and PAMF vermilion throughout.
The heatmap uses one sequential teal-to-ink scale shared by all cells; family colors
appear only in row labels and the PAMF outline. Direct labels and distinct marker
shapes supplement color. No artificial 3D effects or panel composites are used.

## 中文说明

四张独立图分别为六景平均准确率、clean 与六景表现关系、普通 PL 配对增益、七场景热力图。
每张图片只包含一个数据图表，不再使用 a/b/c 拼接面板。六景均值先在每个种子内计算，
再跨三个种子统计；PL 增益先计算同种子 Full＋普通 PL 减 Full，再统计均值和样本标准差。
误差线均为样本标准差，不是置信区间。热力图逐格列出七场景均值和标准差，保持统一绝对色阶。
PAMF 是最终 T1＋固定 DURA，不将其标成普通 PL。全部重绘只改变视觉表达，不改变源数据。
外部方法与 PAMF 的训练配方和预算不同，不能将系统差异解释为单个组件的因果贡献。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-rows', type=Path)
    parser.add_argument('--pamf-csv', type=Path)
    parser.add_argument('--source-csv', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing exports; choose a new output directory')
    if args.source_csv:
        records = read_csv(args.source_csv)
    elif args.external_rows and args.pamf_csv:
        records = load_sources(args.external_rows, args.pamf_csv)
    else:
        parser.error('Provide --source-csv or both --external-rows and --pamf-csv')
    arrays = arrays_from_records(records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'preview').mkdir()
    write_csv(args.output / 'source_data.csv', records)
    summary, paired = [], []
    for key in KEYS:
        for name in (*SCENES, 'six_scenario_mean'):
            values = metric(arrays[key], 'six') if name == 'six_scenario_mean' else arrays[key][:, SCENES.index(name)]
            center, sd = describe(values)
            summary.append(dict(method=label(key), family=key[0], regime=key[1], metric=name,
                n_seeds=3, mean_accuracy_pct=center, sample_sd_pct=sd))
    for family in FAMILIES:
        for seed, value in zip(SEEDS, pl_delta(arrays, family)):
            paired.append(dict(family=family, seed=seed, metric='six_scenario_mean',
                comparison='Full+PL minus Full', difference_pp=float(value)))
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'paired_pl_changes.csv', paired)
    exported = [
        export_figure(accuracy_figure(arrays), args.output, 'Fig1_SixScenario_Accuracy'),
        export_figure(balance_figure(arrays), args.output, 'Fig2_Clean_Channel_Balance'),
        export_figure(pl_effect_figure(arrays), args.output, 'Fig3_PlainPL_Increment',
                      required_names=[NAMES[f] for f in FAMILIES]),
        export_figure(scenario_figure(arrays), args.output, 'Fig4_SevenScene_Accuracy')]
    (args.output / 'figure_captions.md').write_text(CAPTIONS, encoding='utf-8')
    shutil.copyfile(__file__, args.output / 'reproduce.py')
    (args.output / 'README.md').write_text(
        '# Publication figure bundle — one chart per image\n\n'
        'Python only: matplotlib, numpy, Pillow, pypdf. No model or truth access.\n\n'
        'Redraw in a new directory:\n\n'
        '```text\npython reproduce.py --source-csv source_data.csv --output NEW_FIGURES\n```\n\n'
        'Fig1_SixScenario_Accuracy: one forest plot of six-scenario mean accuracy.\n'
        'Fig2_Clean_Channel_Balance: one scatter plot of clean and six-scenario accuracy.\n'
        'Fig3_PlainPL_Increment: one forest plot of matched-seed PL changes.\n'
        'Fig4_SevenScene_Accuracy: one heatmap of all seven scenes.\n'
        'Each image contains exactly one data chart; no composite panels or insets.\n'
        'All: 183 mm width, editable SVG/PDF, 600-dpi PNG/TIFF, color and grayscale previews.\n'
        'Palette: RIEI blue, DRIFT teal, IB-RFF violet, SDG ochre, PAMF vermilion.\n'
        'source_data.csv: 9 series × 3 seeds × 7 scenes = 189 score-only rows.\n'
        'summary.csv: 72 scene/aggregate summaries; paired_pl_changes.csv: 12 seed differences.\n'
        'See figure_captions.md for statistics and scientific limitations.\n', encoding='utf-8')
    validation = dict(status='VERIFIED', backend='Python', matplotlib_version=matplotlib.__version__,
        numpy_version=np.__version__, series=9, trained_model_rows=27, score_rows=len(records),
        external_full_rows=12, external_full_pl_rows=12, pamf_F_seeds=3, pamf_fixed_G=1,
        own_CE_series=0, independent_scenes=False, model_or_truth_access=False,
        layout='one data chart per image; colorbar is a scale legend', palette=COLORS,
        statistic='Mean and sample SD across 3 fixed model seeds; no inference tests', figures=exported,
        source_runs=[EXT_RUN, PAMF_RUN], comparison='same F0 test; different recipes and budgets')
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(args.output.rglob('*')):
            if file.is_file():
                z.write(file, str(Path(args.output.name) / file.relative_to(args.output)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Export archive readback failed')
    print(json.dumps(dict(status='VERIFIED', score_rows=len(records), series=9,
        figures=exported, archive_bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
