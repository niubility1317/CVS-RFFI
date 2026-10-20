"""Publication figures from independently verified, frozen Phase1 score rows.

No IQ, checkpoint, truth, model inference, training, or candidate selection.
Each export uses a fresh directory. Source data and editable vectors accompany it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from pypdf import PdfReader

plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 7, 'axes.titlesize': 8,
    'axes.labelsize': 7, 'axes.linewidth': .65, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.5,
    'legend.fontsize': 7, 'legend.frameon': False, 'savefig.facecolor': 'white'})

RUN = '20261019-phase1-external-dg-concat-pl-m219-r03'
SEEDS = (2026092701, 2026092702, 2026092703)
FAMILIES = ('ibrff', 'sdg', 'riei', 'drift')
NAMES = dict(ibrff='IB-RFF', sdg='SDG', riei='RIEI', drift='DRIFT')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SHORT = ('Clean', 'H', 'M', 'L-S', 'H-U', 'M-U', 'L-U')
DATES = ('2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23')
WIDTH = 183 / 25.4
BLUE, TEAL, GREY = '#0F4D92', '#42949E', '#767676'


def load_rows(root):
    rows = {}
    for path in sorted(root.glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8-sig'))
        assert row['status'] == 'VERIFIED' and row['run_id'] == RUN
        c = row['config']
        assert c['model_seed'] in SEEDS and c['satellite_train_augmentation']
        assert c['default_head_joint_training'] and not c['separate_head_training']
        key = (c['group'], c['family'], c['variant'], c['training_regime'], c['model_seed'], c.get('route'))
        assert key not in rows
        rows[key] = row
    return rows


def get(rows, group, family, variant, seed, regime='supervised', route=None):
    if route is None and family in ('resnet18','resnet50'):
        route = 'erm'
    key = (group, family, variant, regime, seed, route)
    if key not in rows:
        raise ValueError('PENDING_DATA: missing verified row ' + str(key))
    return rows[key]


def score(row, view, dimension='overall', stratum='ALL', test='main'):
    return next(v for v in row['results'] if v['test'] == test and v['view'] == view
                and v['dimension'] == dimension and v['stratum'] == stratum)


def accuracy(row, view):
    receivers = [v for v in row['results'] if v['test'] == 'main' and v['view'] == view
                 and v['dimension'] == 'receiver']
    assert len(receivers) == 7 and all(v['count'] == 24000 for v in receivers)
    result = 100 * np.mean([v['accuracy'] for v in receivers])
    assert abs(result - 100 * score(row, view)['accuracy']) < 1e-10
    return float(result)


def panel(ax, letter):
    ax.text(-.14, 1.04, letter, transform=ax.transAxes,
            weight='bold', fontsize=8, ha='right', va='bottom')


def grid(ax):
    ax.grid(axis='y', color='#E6E8EA', linewidth=.5)
    ax.set_axisbelow(True)


def write_csv(path, data):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def export(fig, out, stem, data, caption, rows):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('svg', 'pdf', 'png', 'preview', 'source_data'):
        (out / name).mkdir()
    write_csv(out / 'source_data' / (stem + '.csv'), data)
    fig.canvas.draw()
    canvas, renderer = fig.get_window_extent(), fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1):
            clipped.append(text.get_text())
    if clipped:
        raise ValueError('Labels outside canvas: ' + repr(clipped))
    fig.savefig(out / 'svg' / (stem + '.svg'))
    fig.savefig(out / 'pdf' / (stem + '.pdf'))
    fig.savefig(out / 'png' / (stem + '.png'), dpi=600)
    fig.savefig(out / 'preview' / (stem + '.png'), dpi=160)
    svg = ET.parse(out / 'svg' / (stem + '.svg')).getroot()
    assert svg.findall('.//{http://www.w3.org/2000/svg}text')
    pdf = PdfReader(out / 'pdf' / (stem + '.pdf'))
    assert len(pdf.pages) == 1 and len(pdf.pages[0].extract_text()) > 60
    assert abs(float(pdf.pages[0].mediabox.width) / 72 * 25.4 - 183) < .1
    with Image.open(out / 'png' / (stem + '.png')) as image:
        assert abs(image.info['dpi'][0] - 600) < 1
        pixels = list(image.size)
    needed = {v['row_id'] for v in data} | {v['control_row_id'] for v in data if 'control_row_id' in v}
    meta = dict(status='VERIFIED', run_id=RUN, figure=stem, backend='Python/matplotlib',
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        source_rows=sorted(needed), source_data_rows=len(data),
        width_mm=183, height_mm=round(fig.get_figheight() * 25.4, 2), png_pixels=pixels,
        statistics='Three fixed model seeds; means and sample SD; no inferential tests.',
        editable_svg_text=True, extractable_pdf_text=True, caption=caption,
        figures_only=True, model_or_truth_access=False)
    (out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'caption.md').write_text(caption + '\n', encoding='utf-8')
    # Portable score-only inputs contain no IQ, checkpoints, truth, or runtime paths.
    target = out / 'source_data' / 'verified_rows'
    target.mkdir()
    fields = ('row_id', 'group', 'family', 'variant', 'training_regime', 'model_seed',
              'satellite_train_augmentation', 'default_head_joint_training',
              'separate_head_training', 'batch_size', 'route')
    for row in rows.values():
        c = row['config']
        if c['row_id'] not in needed:
            continue
        safe = dict(status=row['status'], run_id=row['run_id'],
            config={k:c[k] for k in fields if k in c},
            results=[{k:v[k] for k in ('test','view','dimension','stratum','count','accuracy')}
                     for v in row['results'] if v['dimension'] in ('overall','receiver','receiver_date')],
            cost=dict(stages=[dict(updates=s['updates']) for s in row['cost']['stages']]))
        (target / (c['row_id']+'.json')).write_text(json.dumps(safe, ensure_ascii=False), encoding='utf-8')
    assert len(list(target.glob('*.json'))) == len(needed)
    (out / 'reproduce.py').write_bytes(Path(__file__).read_bytes())
    command = ('python reproduce.py --verified-root source_data/verified_rows '
               '--output NEW_EXPORT --figure ' + dict(fig04_pressure='pressure',
               fig03_receiver_date='receiver_date', fig01_source_count='source_count',
               fig02_label_efficiency='label_efficiency')[stem])
    (out / 'README.md').write_text('# Reproduce this figure\n\n'+command+
        '\n\nPython with matplotlib, numpy, Pillow and pypdf. Inputs are score-only subsets '
        'of independently audited frozen rows; rendering performs no model or truth access. '
        'Use a new output directory. CSV, editable SVG, PDF and 600-dpi PNG are included.\n', encoding='utf-8')
    plt.close(fig)
    return meta


def pressure(rows, out):
    keys = (('ce', 'supervised'), ('full', 'supervised'), ('ce', 'pl'), ('full', 'pl'))
    records, arrays = [], {}
    for family in FAMILIES:
        for variant, regime in keys:
            values = []
            for seed in SEEDS:
                row = get(rows, 'F0-pl' if regime == 'pl' else 'F0', family, variant, seed, regime)
                vector = [accuracy(row, view) for view in VIEWS]
                values.append(vector)
                for view, value in zip(VIEWS, vector):
                    records.append(dict(row_id=row['config']['row_id'], family=family,
                        variant=variant, regime=regime, seed=seed, view=view,
                        accuracy_pct=value, clean_drop_pp=vector[0] - value))
            arrays[family, variant, regime] = np.array(values)
    assert len(records) == 336
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 5.2))
    fig.subplots_adjust(left=.09, right=.985, bottom=.17, top=.86, hspace=.4, wspace=.18)
    drops = [v[:, :1] - v for v in arrays.values()]
    drop_max = max(max(float(v.max()), float((v.mean(0)+v.std(0,ddof=1)).max())) for v in drops)
    drop_min = min(min(float(v.min()), float((v.mean(0)-v.std(0,ddof=1)).min())) for v in drops)
    for column, family in enumerate(FAMILIES):
        for row_number in (0, 1):
            ax = axes[row_number, column]
            for variant, regime in keys:
                values = arrays[family, variant, regime]
                values = values if row_number == 0 else values[:, :1] - values
                color = GREY if variant == 'ce' else BLUE
                style, marker = ('-', 'o') if regime == 'supervised' else ('--', 's')
                ax.errorbar(np.arange(7), values.mean(0), yerr=values.std(0, ddof=1),
                            color=color, ls=style, marker=marker, ms=2.4, lw=.9,
                            elinewidth=.5, capsize=1.5, zorder=3)
                for i, offset in enumerate((-.065, 0, .065)):
                    ax.scatter(np.arange(7) + offset, values[i], s=3.5, color=color, alpha=.35, zorder=4)
            ax.set_xticks(range(7), SHORT, rotation=45, ha='right')
            ax.set_xlim(-.3, 6.25)
            ax.set_ylim((0, 100) if row_number == 0 else
                        (min(-2, math.floor(drop_min / 10) * 10), math.ceil((drop_max + 3) / 10) * 10))
            if column:
                ax.tick_params(labelleft=False)
            panel(ax, chr(ord('a') + row_number * 4 + column))
            grid(ax)
        axes[0, column].set_title(NAMES[family], pad=7)
    axes[0, 0].set_ylabel('RX-balanced accuracy (%)')
    axes[1, 0].set_ylabel('Decrease from clean (pp)')
    handles = [Line2D([], [], color=GREY if v == 'ce' else BLUE,
               ls='-' if r == 'supervised' else '--', marker='o' if r == 'supervised' else 's',
               ms=3, lw=.9, label=('CE' if v == 'ce' else 'Full method') +
               ' + satellite' + (' + PL' if r == 'pl' else '')) for v, r in keys]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .985), ncol=2, columnspacing=1.4)
    fig.text(.5, .052, 'Means ± model-seed SD; n = 3. Points: individual seeds. All conditions retain satellite CE.',
             ha='center', fontsize=6.4)
    fig.text(.5, .022, 'H/M/L-S: high / mid / low-suburban; suffix U: urban. Same physical packets across views.',
             ha='center', fontsize=6.1)
    caption = ('F0 pressure results. Upper panels: target-RX-balanced Accuracy; lower panels: each seed\'s '
        'clean Accuracy minus its scenario Accuracy, in percentage points. Error bars are sample SD across '
        'three fixed model seeds (2026092701/02/03), with individual seed points shown. Each model uses the '
        'same 168000 physical packets across seven correlated views. All four conditions retain default '
        'clean+satellite concatenation, satellite CE weight 0.68; PL uses confidence 0.95 and weight 1. '
        'Full-minus-CE represents each native training recipe, including SDG native views and contrastive '
        'objectives. Native maximum budgets differ by method; no between-method equal-compute claim is made. '
        'No inferential tests or significance marks. Data: six TX, five source RX, seven target RX; '
        'L/U/V=6300/56700/27000. Frozen default heads only; no prediction or truth access during plotting.')
    return export(fig, out, 'fig04_pressure', records, caption, rows)


def receiver_date(rows, out):
    records, grids = [], {}
    rx = [v['stratum'] for v in get(rows, 'F0', 'ibrff', 'ce', SEEDS[0])['results']
          if v['test'] == 'main' and v['view'] == 'clean' and v['dimension'] == 'receiver']
    assert len(rx) == 7
    for group in ('F0', 'chrono'):
        for family in FAMILIES:
            data = np.zeros((3, 7, 4))
            for si, seed in enumerate(SEEDS):
                a, b = [get(rows, group, family, variant, seed) for variant in ('ce', 'full')]
                for ri, receiver in enumerate(rx):
                    for di, date in enumerate(DATES):
                        cells = [score(r, 'clean', 'receiver_date', receiver + '|' + date) for r in (a, b)]
                        assert all(v['count'] == 6000 for v in cells)
                        aa, bb = [100 * v['accuracy'] for v in cells]
                        data[si, ri, di] = bb - aa
                        records.append(dict(row_id=b['config']['row_id'], control_row_id=a['config']['row_id'],
                            group=group, family=family, seed=seed, receiver=receiver, date=date,
                            ce_accuracy_pct=aa, full_accuracy_pct=bb, paired_delta_pp=bb-aa))
            grids[group, family] = data.mean(0)
    limit = max(5, 5 * math.ceil(max(np.abs(v).max() for v in grids.values()) / 5))
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 5.25))
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.18, wspace=.22, hspace=.62)
    for gi, group in enumerate(('F0', 'chrono')):
        for fi, family in enumerate(FAMILIES):
            ax, data = axes[gi, fi], grids[group, family]
            image = ax.imshow(data, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_xticks(range(4), ['D0', 'D1', 'D2', 'D3'])
            ax.set_yticks(range(7), rx)
            if fi:
                ax.tick_params(labelleft=False)
            ax.set_title(NAMES[family], pad=7)
            unseen = 0 if group == 'F0' else 3
            ax.add_patch(Rectangle((unseen-.5, -.5), 1, 7, fill=False, ec='#272727', lw=1))
            for ri in range(7):
                for di in range(4):
                    value = float(data[ri, di])
                    label = '0.0' if abs(value) < .05 else f'{value:+.1f}'
                    ax.text(di, ri, label, ha='center', va='center', fontsize=5.6,
                            color='white' if abs(value) > .62 * limit else '#272727')
            panel(ax, chr(ord('a')+gi*4+fi))
    fig.text(.5, .945, 'F0: source dates D1–D3; D0 is earlier and unseen', ha='center', fontsize=7)
    fig.text(.5, .495, 'Chrono: source dates D0–D2; D3 is future and unseen', ha='center', fontsize=7)
    bar = fig.add_axes((.27, .11, .46, .02))
    fig.colorbar(image, cax=bar, orientation='horizontal', label='Full method − own CE (percentage points)')
    fig.text(.5, .012, 'Clean; paired mean across 3 model seeds. Black outline: unseen date. Cells/RX/dates are not independent repeats.',
             ha='center', fontsize=6)
    caption = ('Clean receiver-by-date paired differences, full method minus its own CE, averaged across '
        'three fixed model seeds. Each cell contains 6000 packets per seed; row labels are physical target '
        'receiver names. Columns D0/D1/D2/D3 are 2021-03-01/03-08/03-15/03-23. The outlined unseen-date '
        'column is earlier D0 for F0 and future D3 for Chrono. Chrono predictions contain all four target '
        'dates; its primary future endpoint is the 42000-packet D3 subset, not all-date overall Accuracy. '
        'A shared symmetric scale includes every cell without clipping. Numbers are paired means, not '
        'significance tests; all individual seed values are supplied as source data. All conditions retain '
        'default satellite concatenation; this figure uses supervised rows and frozen default heads.')
    return export(fig, out, 'fig03_receiver_date', records, caption, rows)


def source_count(rows, out):
    chains = ((1, 3, 4, 6, 8), (4, 6, 8, 1, 3), (8, 1, 3, 4, 6))
    records, vectors = [], {}
    for ci in range(1, 4):
        for budget in ('natural', 'fixed'):
            data = []
            for m in (1, 3, 5):
                actual_chain = 1 if m == 5 else ci
                actual_budget = 'natural' if m == 1 else budget
                group = f'count-c{actual_chain}-m{m}-{actual_budget}'
                vector = []
                for seed in SEEDS:
                    row = get(rows, group, 'resnet18', 'ce', seed)
                    assert row['config']['batch_size'] == 180 and row['cost']['stages'][0]['updates'] == 10000
                    value = accuracy(row, 'clean'); vector.append(value)
                    records.append(dict(row_id=row['config']['row_id'], display_chain=ci,
                        label_budget=budget, source_receivers=m, labels=1260*m if budget=='natural' else 1260,
                        seed=seed, accuracy_pct=value, updates=10000,
                        shared_configuration=(m==1 or m==5)))
                data.append(vector)
            vectors[ci, budget] = np.array(data).T
    assert len({v['row_id'] for v in records}) == 33
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.5), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.25, top=.82, wspace=.18)
    for ci, ax in enumerate(axes, 1):
        for budget, color, style, marker in [('natural', BLUE, '-', 'o'), ('fixed', TEAL, '--', 's')]:
            values = vectors[ci, budget]
            ax.errorbar([1,3,5], values.mean(0), yerr=values.std(0, ddof=1), color=color,
                        ls=style, marker=marker, ms=3, lw=1, capsize=2, elinewidth=.6)
            for si, offset in enumerate((-.06,0,.06)):
                ax.scatter(np.array([1,3,5])+offset, values[si], color=color, s=6, alpha=.45)
        ax.set(xlim=(.7,5.3), ylim=(0,100), xticks=[1,3,5], xlabel='Source receivers')
        ax.set_title('Chain '+chr(64+ci)+': '+','.join(map(str,chains[ci-1])), fontsize=7.2)
        panel(ax, chr(96+ci));grid(ax)
    axes[0].set_ylabel('RX-balanced clean accuracy (%)')
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Natural labels: 1260 × source RX'),
                        Line2D([],[],color=TEAL,ls='--',marker='s',label='Fixed labels: 1260')],
               loc='upper center', ncol=2, bbox_to_anchor=(.55,.98))
    fig.text(.5,.09,'Mean ± model-seed SD (n=3); raw points shown. Evaluable configurations: 11 × 3 seeds = 33.',ha='center',fontsize=6.5)
    fig.text(.5,.04,'Shared: m=1 across label budgets; m=5 across chains. All rows: 10000 updates, batch 180, satellite CE.',ha='center',fontsize=6.2)
    caption = ('Source-receiver count with natural or fixed labeled budgets. Three preregistered nested '
        'RX chains are shown separately; the target remains the same seven RX and 168000 packets. '
        'Every distinct configuration has three model seeds, 10000 updates, batch 180, 1800000 labeled '
        'exposures, and matched satellite concatenation. V contains 5400 times source-RX count labels but '
        'does not select the fixed final weights. Natural L=1260*m; fixed L=1260. m=1 budgets share the '
        'same models, and m=5 endpoints share models across chains; repeated displays do not add replicates. '
        'There are 33 unique trained models, not 54. Error bars are model-seed sample SD, not receiver CIs. '
        'This tests ResNet18 only and does not establish the same source-count effect for other methods.')
    return export(fig, out, 'fig01_source_count', records, caption, rows)


def label_efficiency(rows, out):
    conditions = (('resnet18','ce','ResNet18, E200'), ('ibrff','ce','IB-RFF CE, 2 × max E200'),
                  ('ibrff','full','IB-RFF full, 2 × max E200'))
    groups = ('label01','label05','F0')
    records, vectors, updates = [], {}, {}
    for family, variant, _ in conditions:
        data, steps = [], []
        for group, rate, labels in zip(groups,(1,5,10),(630,3150,6300)):
            vector, count = [], []
            for seed in SEEDS:
                actual = 'strongce' if family=='resnet18' and group=='F0' else group
                row = get(rows, actual, family, variant, seed)
                value = accuracy(row,'clean');step = sum(s['updates'] for s in row['cost']['stages'])
                vector.append(value);count.append(step)
                records.append(dict(row_id=row['config']['row_id'],family=family,variant=variant,
                    seed=seed,training_label_percent=rate,labels=labels,validation_labels=27000,
                    accuracy_pct=value,actual_updates=step))
            data.append(vector);steps.append(count)
        vectors[family,variant]=np.array(data).T;updates[family,variant]=np.array(steps)
    lower = min(float(v.mean(0).min()-v.std(0,ddof=1).max()) for v in vectors.values())
    lower = min(lower, min(float(v.min()) for v in vectors.values()))
    upper = max(float(v.mean(0).max()+v.std(0,ddof=1).max()) for v in vectors.values())
    upper = max(upper,max(float(v.max()) for v in vectors.values()))
    low,high=max(0,5*math.floor((lower-3)/5)),min(100,5*math.ceil((upper+3)/5))
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,3.6),sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.37,top=.84,wspace=.18)
    for ai,(ax,(family,variant,title)) in enumerate(zip(axes,conditions)):
        values=vectors[family,variant];color=BLUE if variant=='full' else GREY
        ax.errorbar([1,5,10],values.mean(0),yerr=values.std(0,ddof=1),color=color,
                    marker='o',ms=3,lw=1,capsize=2,elinewidth=.6)
        for si,offset in enumerate((-.1,0,.1)):
            ax.scatter(np.array([1,5,10])+offset,values[si],s=7,color=color,alpha=.5)
        ax.set(xlim=(.5,10.5),ylim=(low,high),xticks=[1,5,10],xticklabels=['1%\n630 labels','5%\n3150 labels','10%\n6300 labels'])
        for x,counts in zip((1,5,10),updates[family,variant]):
            text=str(int(counts[0])) if len(set(counts))==1 else f'{min(counts)}–{max(counts)}'
            ax.text(x,-.36,text,transform=ax.get_xaxis_transform(),ha='center',fontsize=6)
        ax.set_title(title,fontsize=7.2);panel(ax,chr(97+ai));grid(ax)
    axes[0].set_ylabel('RX-balanced clean accuracy (%)')
    fig.text(.5,.13,'Numbers below ticks: actual updates (range across model seeds). Mean ± sample SD; n=3.',ha='center',fontsize=6.4)
    fig.text(.5,.065,'Training-pool labeled fraction; source V adds 27000 labels. Same maximum epochs do not imply equal compute.',ha='center',fontsize=6.1)
    caption = ('Training-label efficiency under the preregistered epoch budgets. Training-pool labeled '
        'fractions 1/5/10% correspond to L=630/3150/6300; V remains 27000 separately labeled samples. '
        'All are supervised with satellite concatenation, without PL. ResNet18 uses E200; its 10% point '
        'is the matching strong-CE row, not the 5000-update ERM or batch180 source-count model. IB-RFF '
        'retains two stages each capped at E200, source-NLL LR reduction and early stopping. Actual '
        'updates are supplied per seed and annotated as seed ranges; differing data/exposure/budgets '
        'prevent an equal-compute interpretation. Means and sample SD across three model seeds, with '
        'all raw points and non-monotonic results retained. Fixed F0 target: seven RX, six TX, 168000 packets.')
    return export(fig,out,'fig02_label_efficiency',records,caption,rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verified-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',choices=('pressure','receiver_date','source_count','label_efficiency'),required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing exports; use a new output directory.')
    rows=load_rows(args.verified_root)
    result=globals()[args.figure](rows,args.output)
    print(json.dumps(dict(status=result['status'],figure=result['figure'],unique_source_rows=len(result['source_rows']),
                         source_points=result['source_data_rows'],width_mm=result['width_mm']),ensure_ascii=False))


if __name__=='__main__':main()
