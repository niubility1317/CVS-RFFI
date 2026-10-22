"""Times New Roman styling for standalone manuscript figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

WIDTH_MM = 183
FONT = 'Times New Roman'
INK, MUTED, GRID = '#1D2939', '#5E6B7C', '#E7EBEF'
COLORS = {'riei': '#3F70AE', 'drift': '#258F8F', 'ibrff': '#8A6BB8',
          'sdg': '#B7832F', 'pamf': '#CE4E43', 'baseline': '#71849A'}
CMAP = LinearSegmentedColormap.from_list('paper_accuracy',
    ['#F5F8FA', '#D6E5ED', '#AACAD9', '#6EA4BC', '#376F8D', '#193C57'])


def apply_style():
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.exists(): font_manager.fontManager.addfont(str(path))
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': [FONT],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8.3, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
        'axes.linewidth': .65, 'axes.edgecolor': '#98A3B0',
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': INK,
        'xtick.major.width': .55, 'ytick.major.width': .55,
        'legend.frameon': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'hatch.linewidth': .55,
        'svg.hashsalt': 'pamf-manuscript-times-v1'})


def canvas(title, height=102, bounds=(.105, .18, .85, .70)):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height / 25.4), dpi=150)
    fig.text(.5, .961, title, ha='center', va='top', fontsize=10.5,
             fontfamily=FONT, weight='normal')
    return fig, fig.add_axes(bounds, label='data')


def pale(c, fraction=.20):
    return tuple(1 - fraction + fraction * np.asarray(to_rgb(c)))


def grid(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=.5)


def describe(a):
    a = np.asarray(a, float)
    if len(a) < 2 or not np.isfinite(a).all(): raise ValueError('Invalid repetitions')
    return float(a.mean()), float(a.std(ddof=1))


def color(model):
    if model in COLORS: return COLORS[model]
    if model.startswith('cvresnet'): return COLORS['ibrff']
    if model.endswith('_2d'): return COLORS['drift']
    if model in ('bilstm3', 'transformer6'): return COLORS['sdg']
    return COLORS['riei']


def contrast(rgba):
    def lum(rgb):
        rgb = np.asarray(rgb)
        return float(np.where(rgb <= .04045, rgb / 12.92,
                     ((rgb + .055) / 1.055) ** 2.4) @ [.2126, .7152, .0722])
    l = lum(rgba[:3])
    return 'white' if 1.05 / (l + .05) > (l + .05) / (lum(to_rgb(INK)) + .05) else INK


apply_style()
