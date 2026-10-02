"""Plot the validated frozen clean results, without accessing IQ or truth."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.cvs_channel_order_clean.prepare import RUN


def figures(root):
    evidence=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    verdict=json.loads((evidence/'analysis_validation.json').read_text(encoding='utf-8'))
    assert verdict['status']=='VERIFIED' and verdict['all320_confusions_recomputed']
    data=json.loads((evidence/'final_readback.json').read_text(encoding='utf-8'))
    candidate=verdict['candidate']
    overall=[r for r in data['summary']['summary'] if r['receiver']=='ALL']
    pairs=[r for r in data['summary']['paired'] if r['baseline']=='neural_residual_shallow' and r['receiver']!='ALL']
    assert len(overall)==10 and len(pairs)==7
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    colors=['#0f766e' if r['method']==candidate else '#64748b' for r in overall]
    axes[0].barh([r['method'] for r in overall],[100*r['accuracy_mean'] for r in overall],xerr=[100*r['accuracy_seed_sd'] for r in overall],color=colors,capsize=3)
    axes[0].invert_yaxis();axes[0].set(xlim=(0,100),xlabel='Clean accuracy (%)',title='Frozen benchmark: mean +/- model-seed SD')
    colors=['#0f766e' if r['accuracy_delta_pp_mean']>0 else '#b45309' for r in pairs]
    axes[1].barh([r['receiver'] for r in pairs],[r['accuracy_delta_pp_mean'] for r in pairs],xerr=[r['accuracy_delta_pp_seed_sd'] for r in pairs],color=colors,capsize=3)
    axes[1].axvline(0,color='#334155',linewidth=1);axes[1].invert_yaxis()
    axes[1].set(xlabel='Paired accuracy change (percentage points)',title='Channel dual minus shallow control, by RX')
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('4 model/loader seeds; fixed split; 168,000 packets per model; clean only',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(evidence/('clean_comparison.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='VERIFIED',overall_groups=10,receiver_pairs=7,source='validated final_readback.json',error_bars='sample SD across four paired model seeds, not confidence interval')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);figures(p.parse_args().root)
