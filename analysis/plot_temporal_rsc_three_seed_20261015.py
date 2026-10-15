"""Render frozen full-three-seed metrics and complete saved source scalar curves."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
COLORS={'OLD_GATE_COS50':'#858585','SIG_TIME':'#287da1','MATCHED_RANDOM':'#d18b2f','TEMPORAL_RSC':'#279578'}
def main(root):
    root=Path(root);r=json.loads((root/'results.json').read_text(encoding='utf8'))
    assert r['model_seed_count']==3 and r['metric_records']==216 and len(r['records'])==216
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.6),dpi=180)
    for i,s in enumerate(r['summary']):
        y=np.array(s['seed_accuracy'])*100
        axs[0].bar(i,s['accuracy_mean']*100,color=COLORS[s['arm']],width=.62,alpha=.8)
        axs[0].errorbar(i,s['accuracy_mean']*100,yerr=s['accuracy_sample_SD']*100,fmt='none',ecolor='#333333',capsize=4,lw=1.2)
        axs[0].scatter(i+np.array([-.15,0,.15]),y,color='#222222',s=17,zorder=4)
        axs[0].text(i,76.4,f"{s['accuracy_mean']*100:.3f}%",ha='center',fontsize=10)
    axs[0].axhline(84,color='#b33c38',ls='--',lw=1,label='84% goal')
    axs[0].set(xticks=range(4),xticklabels=['Old gate + cos','SIG_TIME','Random','Temporal RSC'],ylim=(76,85.5),ylabel='Clean accuracy (%)',title='Three predefined seeds; bars: mean +/- sample SD')
    for i,p in enumerate(r['paired'][:3]):
        y=np.array(p['seed_difference_pp']);axs[1].scatter([i]*3,y,color=list(COLORS.values())[i+1],s=34)
        axs[1].plot([i-.17,i+.17],[p['mean_difference_pp']]*2,color='#222222',lw=2)
    axs[1].axhline(0,color='#888888',lw=1)
    axs[1].set(xticks=range(3),xticklabels=['RSC - random','RSC - SIG_TIME','Random - SIG_TIME'],ylabel='Paired accuracy difference (percentage points)',title='All fixed seed differences retained')
    for ax in axs:ax.grid(axis='y',alpha=.16)
    fig.suptitle('Fixed SIG_TIME | temporal masking did not improve generalization',ha='left',x=.065,y=.99,fontsize=14,fontweight='bold')
    fig.text(.065,.015,'Clean 168,000 decisions/model; default joint head; final E50. Known exposed benchmark; no satellite or Phase2 claim.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.045,1,.91));fig.savefig(root/'three_seed_clean_results.png');plt.close(fig)
    source=json.loads((root/'full_source_curves.json').read_text(encoding='utf8'));assert source['source_only'] and len(source['rows'])==12
    fig,axs=plt.subplots(3,3,figsize=(12,8),dpi=180,sharex=True)
    for c in source['rows']:
        if c['arm']=='OLD_GATE_COS50':continue
        seed=int(c['row_id'].split('-s')[-1]);a=seed-2026092701;records=c['records'];assert len(records)==50
        color=COLORS[c['arm']];x=[e['epoch'] for e in records]
        axs[a,0].plot(x,[e['train_loss_tx_labeled'] for e in records],color=color,label=c['arm'],lw=1.2)
        axs[a,1].plot(x,[e['val_tx_acc'] for e in records],color=color,lw=1.2)
        active=[e for e in records if e['train_pseudo_total']>0]
        axs[a,2].plot([e['epoch'] for e in active],[100*e['train_pseudo_selected']/e['train_pseudo_total'] for e in active],color=color,lw=1.2)
    for a in range(3):
        for b in range(3):
            axs[a,b].grid(alpha=.16);axs[a,b].axvline(33.5,color='#999999',ls='--',lw=.7);axs[a,b].set_xlim(1,50)
        axs[a,0].set_ylabel(f'Seed {2026092701+a}');axs[a,0].set_ylim(bottom=0);axs[a,1].set_ylim(75,100);axs[a,2].set_ylim(bottom=0)
    for ax,title in zip(axs[0],['Native selected L CE','Source V accuracy (%)','Accepted U / processed U (%)']):ax.set_title(title,loc='left')
    for ax in axs[-1]:ax.set_xlabel('Epoch')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025))
    fig.suptitle('Fixed SIG_TIME | all nine new source training curves',ha='left',x=.065,y=.99,fontsize=14,fontweight='bold')
    fig.text(.065,.006,'600 saved epoch records incl. 3 historical rows; new rows shown. Native selected CE includes masking. U acceptance is not precision.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(root/'three_seed_source_curves.png');plt.close(fig)
    print(json.dumps({'status':'VERIFIED','metric_records':216,'source_epochs':600,'plotted_new_models':9,'figures':2}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
