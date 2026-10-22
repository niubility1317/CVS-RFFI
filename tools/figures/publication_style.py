"""Shared visual language for score-only PAMF scientific figures."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

WIDTH_MM = 183
INK = '#27364B'
MUTED = '#626F80'
GRID = '#E7EBF0'
COLORS = {'riei': '#3F70AE', 'drift': '#258F8F', 'ibrff': '#8A6BB8',
          'sdg': '#B7832F', 'pamf': '#CE4E43', 'baseline': '#72849B'}
FAMILY_COLORS = {'real1d': '#3F70AE', 'real2d': '#258F8F',
                 'complex': '#8A6BB8', 'sequence': '#B7832F',
                 'pamf': COLORS['pamf']}
CMAP = LinearSegmentedColormap.from_list('shared_accuracy',
    ['#F5F8FA', '#D6E5ED', '#AACAD9', '#6EA4BC', '#376F8D', '#193C57'])


def apply_style():
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'axes.linewidth': .65,
        'axes.edgecolor': '#A1AAB7', 'axes.spines.right': False,
        'axes.spines.top': False, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': INK, 'xtick.labelsize': 7.4,
        'ytick.labelsize': 7.4, 'xtick.major.width': .55,
        'ytick.major.width': .55, 'legend.frameon': False,
        'legend.fontsize': 7.2, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'hatch.linewidth': .55,
        'svg.hashsalt': 'pamf-unified-publication-v1'})


def describe(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Mean/SD requires at least two finite repetitions')
    return float(a.mean()), float(a.std(ddof=1))


def pale(color, fraction=.20):
    return tuple((1 - fraction) + fraction * np.asarray(to_rgb(color)))


def contrast_color(rgba):
    def lum(rgb):
        rgb = np.asarray(rgb)
        return float(np.where(rgb <= .04045, rgb / 12.92,
                     ((rgb + .055) / 1.055) ** 2.4) @ [.2126, .7152, .0722])
    light = lum(rgba[:3])
    return 'white' if 1.05 / (light + .05) > (light + .05) / (lum(to_rgb(INK)) + .05) else INK


def canvas(title, subtitle, height=120, bounds=(.12, .25, .83, .55)):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height / 25.4), dpi=150)
    fig.text(.045, .951, title, fontsize=11.3, weight='bold', va='top')
    fig.text(.045, .898, subtitle, fontsize=7.2, color=MUTED, va='top')
    ax = fig.add_axes(bounds, label='data')
    return fig, ax


def note(fig, first, second=''):
    fig.text(.045, .067, first, fontsize=6.7, color=MUTED, va='center')
    if second:
        fig.text(.045, .031, second, fontsize=6.7, color=MUTED, va='center')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=.55)
    ax.set_axisbelow(True)


def regime_legend(fig, location=(.53, .15)):
    handles = [Patch(fc='#EBEEF2', ec=MUTED, hatch='//', label='Full'),
               Patch(fc=MUTED, ec=MUTED, label='Full + plain PL'),
               Patch(fc=COLORS['pamf'], ec=COLORS['pamf'], label='PAMF (T1 + DURA)')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=location,
               ncol=3, handlelength=1.6, columnspacing=2.1)


def seed_legend(fig, location=(.57, .15)):
    fig.legend(handles=[
        Line2D([], [], marker='o', ls='', color=MUTED, alpha=.45,
               ms=3, label='Individual model seed'),
        Line2D([], [], marker='D', ls='-', color=INK, mfc='white',
               ms=4, label='Mean ± sample SD')],
        loc='center', bbox_to_anchor=location, ncol=2, columnspacing=2.2)


apply_style()
