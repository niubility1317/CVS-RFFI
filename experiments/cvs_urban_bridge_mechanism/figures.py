"""Render only complete real source or formal artifacts; no training imports."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('NOMINAL','BRIDGE_CE','BRIDGE_SIG')
COLORS=('#666666','#2878b5','#368657')
GROUPS=('increment','feature_area','time_area')
URBAN=('practical_high_urban','practical_mid_urban','practical_low_urban')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def save(fig,out,name):
    fig.savefig(out/name,dpi=180);plt.close(fig)


def source(root,out):
    result=read(root/'results.json')
    if (result['status']!='ANALYZED' or not result['source_only'] or result['query_read'] or result['U_truth_read']
            or result['V_forward_packets'] or result['optimizer_updates'] or result['EMA_updates'] or result['head_fit_updates']
            or result['forward_packets']!=56882 or set(result['rows'])!=set(ARMS)):
        raise ValueError('Complete actual three-model source-only attribution required')
    rows=result['rows']
    if not all(r['model_state_unchanged'] and r['EMA_state_unchanged'] for r in rows.values()):
        raise ValueError('Frozen student and EMA state evidence required')
    out.mkdir(parents=True,exist_ok=True);labels=('High urban','Mid urban','Low urban');low=URBAN[-1]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    x=np.arange(3)
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        full=[rows[arm]['scenes'][s]['full']['overall'] for s in URBAN]
        HQ=[rows[arm]['scenes'][s]['HQ_independent']['overall'] for s in URBAN]
        axes[0,0].plot(labels,[100*v['accuracy'] for v in full],'o-',label=arm,color=color)
        axes[0,0].plot(labels,[100*v['accuracy'] for v in HQ],'x--',color=color,alpha=.6)
        axes[0,1].plot(labels,[100*v['fixed_NOMINAL_retention'] for v in full],'o-',label=arm,color=color)
        axes[1,0].plot(labels,[v['margin'] for v in full],'o-',label=arm,color=color)
        r=rows[arm]['scenes'][low]['full']['overall']
        axes[1,1].bar(i-.18,r['NOMINAL_errors_repaired'],.35,color=color)
        axes[1,1].bar(i+.18,r['NOMINAL_correct_new_errors'],.35,color=color,hatch='///',alpha=.6)
    axes[0,0].set_title('Default heads: full (solid), independent HQ (dashed)');axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,1].set_title('Full: common NOMINAL clean-correct subset');axes[0,1].set_ylabel('Retention (%)')
    axes[1,0].set_title('Full: true logit minus strongest wrong logit / 30');axes[1,0].set_ylabel('Head margin / 30')
    axes[1,0].axhline(0,color='grey',ls=':',lw=1)
    axes[1,1].set_xticks(x,ARMS);axes[1,1].set_title('Low full: repairs (solid), new errors (hatched)');axes[1,1].set_ylabel('Packets in common fixed subset')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"Held-out source720 only; fixed clean-correct n={result['fixed_NOMINAL_clean_correct_count']}; zero updates")
    save(fig,out,'source_default_head_repairs.png')

    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        r=rows[arm]['scenes'][low]['full']
        axes[0].plot(GROUPS,[r['overall']['common_local_group_error'][k] for k in GROUPS],'o-',label=arm,color=color)
        axes[1].plot(GROUPS,[r['SIG_group_geometry'][k]['gap'] for k in GROUPS],'o-',label=arm,color=color)
        axes[2].bar(i,r['gain_contribution']['S']['mean_contribution_to_joint_norm_ratio'],color=color)
    axes[0].set_title('Low full -> own HQ EMA: bounded group loss');axes[0].set_ylabel('Shared CE-epoch5 RMS scales')
    axes[1].set_title('Raw groups: paired minus nearest wrong-TX cosine');axes[1].axhline(0,color='grey',ls=':',lw=1)
    axes[2].set_title('Actual SIG gain contribution / joint norm');axes[2].set_xticks(x,ARMS)
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelrotation=15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Source same-H independent HQ: descriptor similarity and actual identity evidence are separate')
    save(fig,out,'source_SIG_geometry_usage.png')

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    order=dict(state=('LOS','PARTIAL','BLOCKED'),lock=('locked','degraded','unlocked'),quality=('<3','3-8','8-15','>=15'))
    for rr,scene in enumerate(URBAN[1:]):
        for ax,field in zip(axes[rr],('state','lock','quality')):
            reference=rows['NOMINAL']['scenes'][scene]['full']['strata'];present={r[field] for r in reference}
            categories=[v for v in order[field] if v in present]+sorted(present-set(order[field]))
            counts={v:sum(r['count'] for r in reference if r[field]==v) for v in categories}
            for arm,color in zip(ARMS,COLORS):
                rates=[]
                for value in categories:
                    subset=[r for r in rows[arm]['scenes'][scene]['full']['strata'] if r[field]==value]
                    n=sum(r['count'] for r in subset)
                    if not n or n!=counts[value]:raise ValueError('Fixed common source strata differ')
                    rates.append(100*sum(r['accuracy']*r['count'] for r in subset)/n)
                ax.plot(range(len(categories)),rates,'o-',color=color,label=arm)
            ax.set_xticks(range(len(categories)),[f'{v}\nn={counts[v]}' for v in categories],rotation=15)
            ax.set_title(scene.replace('practical_','')+' / '+field);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Full source observations; added-noise SNR bins are not total SNR; no target evidence here')
    save(fig,out,'source_state_quality_strata.png')


def training(root,out):
    summary=read(root/'summary.json');curves=read(root/'training_curves.json');resources=read(root/'resources.json')
    if summary['seed_count']!=1 or len(curves)!=2 or resources['new_models']!=2:raise ValueError('Actual complete two-row pilot required')
    out.mkdir(parents=True,exist_ok=True)
    keys=('urban/clean_CE','urban/satellite_CE','urban/HQ_CE','urban/auxiliary_weighted_loss','urban/local_SIG_loss','urban/local_SIG_weighted_loss')
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,values in curves.items():
        if len(values)!=50 or [v['epoch'] for v in values]!=list(range(1,51)):raise ValueError('Complete E50 curves required')
        for ax,key in zip(axes.flat,keys):
            values_y=[v[key] for v in values]
            if not np.isfinite(values_y).all():raise ValueError('Finite actual curve required')
            ax.plot(range(1,51),values_y,label=row.split('-')[0]);ax.set_title(key.split('/')[1]);ax.set_xlabel('Epoch')
            ax.axvline(5.5,color='grey',ls='--',lw=.7);ax.axvline(33.5,color='grey',ls=':',lw=.7);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Full 11100-step streams, epoch means; transfer starts E6, native U starts E34')
    save(fig,out,'complete_training_curves.png')
    allarms=('CLEAN','NOMINAL','POSITIVE_J','BRIDGE_CE','BRIDGE_SIG');byarm={r['arm']:r for r in summary['rows']}
    views=('clean','practical_high','practical_mid','practical_low_suburban',*URBAN)
    fig,axes=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    for arm in allarms:
        r=byarm[arm];axes[0].plot(range(7),[100*r['views'][v]['accuracy'] for v in views],'o-',label=arm)
        receivers=sorted(r['urban_by_RX'])
        axes[1].plot(range(len(receivers)),[100*r['urban_by_RX'][rx] for rx in receivers],'o-',label=arm)
    axes[0].set_xticks(range(7),[v.replace('practical_','') for v in views],rotation=30,ha='right')
    axes[1].set_xticks(range(len(receivers)),receivers,rotation=40,ha='right')
    for ax in axes:ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105);ax.grid(alpha=.2)
    axes[0].set_title('Original fixed seven-view test, one model seed');axes[1].set_title('Urban three-scene mean by physical target RX')
    axes[0].legend(fontsize=8);fig.suptitle('Original heads and fixed IQ; archived controls reused, only SIG-vs-CE matches five-view budgets')
    save(fig,out,'formal_views_and_receiver_strata.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--training',action='store_true');a=p.parse_args()
    root=Path(a.root);(training if a.training else source)(root,root/'figures')
