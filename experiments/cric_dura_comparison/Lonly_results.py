"""Small public tables and optional figure for completed strict L-only readouts."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from .analyze_Lonly import ARMS, READOUTS
from .analyze_main import VIEWS


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def run(source, output, paper=None):
    source, output = Path(source), Path(output)
    verified = json.loads((source / 'Lonly_verification.json').read_text(encoding='utf-8'))
    if verified['status'] != 'ANALYZED_LONLY6' or verified['serialized_CM_recomputed'] != 4032:
        raise ValueError('Only the completed six-row panel may be reported')
    output.mkdir(parents=True, exist_ok=True)
    public = ('Lonly_results.md', 'Lonly_summary_overall.csv', 'Lonly_paired_overall.csv',
              'Lonly_F_cost.csv', 'Lonly_G_cost.csv', 'Lonly_source_geometry.csv',
              'Lonly_source_sensitivity.csv', 'Lonly_verification.json')
    for name in public:
        shutil.copyfile(source / name, output / name)
    if paper is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paper = Path(paper)
    figures = paper / 'images/Lonly6'
    figures.mkdir(parents=True, exist_ok=True)
    summary = rows(source / 'Lonly_summary_overall.csv')
    paired = rows(source / 'Lonly_paired_overall.csv')
    values = {(r['arm'], r['readout'], r['view'], r['metric']): r for r in summary}
    effects = {(r['contrast'], r['arm'], r['readout'], r['view'], r['metric']): r for r in paired}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), sharey=True, constrained_layout=True)
    names = ('Clean', 'High\nsub.', 'Mid\nsub.', 'Low\nsub.', 'High\nurban', 'Mid\nurban', 'Low\nurban')
    for ax, arm in zip(axes, ARMS):
        for reader, color, marker in (('raw', '#666666', 'o'), ('DURA-L', '#176b93', 's')):
            items = [values[arm, reader, v, 'accuracy'] for v in VIEWS]
            ax.errorbar(range(7), [float(r['mean']) for r in items], [float(r['sd']) for r in items],
                        label=reader, color=color, marker=marker, markersize=3.5, linewidth=1.2, capsize=2)
        ax.set_xticks(range(7), names)
        ax.set_ylim(25, 92)
        ax.set_title('F15: X3 CE + sat' if arm == 'X3_CE' else 'F20: supervised T1')
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='lower left', frameon=False, fontsize=7)
    axes[0].set_ylabel('Accuracy (%)')
    fig.savefig(figures / 'Lonly_seven_views.pdf')
    fig.savefig(figures / 'Lonly_seven_views.png', dpi=220)
    plt.close(fig)
    def f(arm, reader, view):
        r = values[arm, reader, view, 'accuracy']
        return r'$%.4f\pm%.4f$' % (float(r['mean']), float(r['sd']))
    def effect(arm, view):
        r = effects['DURA-L_minus_raw', arm, 'paired', view, 'accuracy']
        return r'$%+.4f\pm%.4f$' % (float(r['mean']), float(r['sd']))
    text = [r'\subsubsection{Completed Strict L-only Readouts}',
        'The six reuse-only readouts are complete: F15 and F20 each provide three corrected F seeds, paired with the same F0 G-L whose eight training ancestors use labeled-source IQ only. Each row freezes raw and restored predictions for all seven views before independent truth-last scoring. The target contract remains seven receivers and 168000 packets per view.',
        r'\begin{table}[t]', r'\centering\scriptsize',
        r'\caption{Strict L-only F0 accuracy (\%, mean$\pm$sample SD over three F seeds conditional on one fixed G-L).}',
        r'\label{tab:Lonly6}', r'\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular}{llrrr}', r'\toprule',
        r'F recipe & Readout & Clean & Six views & Low urban \\', r'\midrule']
    for arm in ARMS:
        for reader in READOUTS:
            text.append('%s & %s & %s & %s & %s \\\\' %
                        (arm.replace('_', r'\_'), reader, f(arm, reader, 'clean'),
                         f(arm, reader, 'six_practical_mean'), f(arm, reader, 'practical_low_urban')))
    text += [r'\bottomrule', r'\end{tabular}', r'\end{table}',
        'F15 uses ordinary-view CE plus default satellite-concatenation CE (weight 0.68), with zero label smoothing and no CRIC, paired KL, U or EMA. F20 retains the supervised T1 CRIC, ordinary/raw KL, As CE with As KL zero, and label smoothing 0.01, while disabling U and EMA. Their 50 logical epochs and 11100 updates match, but actual labeled-packet forward evaluations are 384 versus 768 per update and the augmentations differ. The contrast therefore compares complete supervised recipes, rather than identifying one component effect or replacing the matched CE structural controls.',
        'G-L improves the six-view means by %s and %s percentage points for F15 and F20, respectively. The paired clean changes are %s and %s points; these small negative means are retained. Six-view averaging precedes seed-paired differences and sample SD; the latter is not a significance test.' %
        (effect('X3_CE', 'six_practical_mean'), effect('T1_SUP', 'six_practical_mean'),
         effect('X3_CE', 'clean'), effect('T1_SUP', 'clean')),
        r'\begin{figure*}[t]', r'\centering',
        r'\includegraphics[width=\textwidth]{images/Lonly6/Lonly_seven_views.pdf}',
        r'\caption{Completed raw and G-L readouts on all seven F0 views. Error bars are three-F-seed sample SD conditional on the same labeled-only restorer. Every negative clean change is retained; these curves do not select a replacement recipe.}',
        r'\label{fig:Lonly6}', r'\end{figure*}',
        'All 4032 serialized confusion matrices, 2016 stratum-level correction nets, and exact reuse of same-F raw matrices were checked. The six source-physics panels each reuse the same 720 labeled packets across 90 TX/RX/day strata and 11 interventions, with no U, target access, or fitted probe. They characterize sensitivity and inductive bias, not complete hardware disentanglement or identity preservation under synthetic PA changes.',
        'No additional F or G was trained for these readouts. The reused six F trajectories total %.3f shared-worker hours, and the one reused eight-stage G-L totals %.3f hours; these are historical elapsed costs, not incremental or exclusive GPU time. The separate 24-model architecture panel and its dedicated online timing remain incomplete.' %
        (verified['reused_F_shared_worker_seconds'] / 3600, verified['reused_G_shared_worker_seconds'] / 3600)]
    (paper / 'Lonly_results.tex').write_text('\n\n'.join(text) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--paper')
    a = p.parse_args()
    run(a.input, a.output, a.paper)
