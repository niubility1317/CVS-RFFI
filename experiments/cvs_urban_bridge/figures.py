"""Reproduce scientific source figures from the immutable frozen-run outputs."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VIEWS=('full','bridge_shared','bridge_independent','no_added_noise')
LABELS=('Full','HQ shared noise','HQ independent noise','No added noise')
COLORS=('#4c566a','#d08770','#5e81ac','#a3be8c')
URBAN=('practical_high_urban','practical_mid_urban','practical_low_urban')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def aggregate(items):
    n=len(items);fixed=[r for r in items if r['fixed_parent_clean_correct']]
    return dict(count=n,EMA_accuracy=sum(r['EMA_correct'] for r in items)/n,
        student_accuracy=sum(r['student_correct'] for r in items)/n,
        fixed_parent_retention=sum(r['student_correct'] for r in fixed)/len(fixed) if fixed else None,
        student_margin=sum(r['student_margin'] for r in items)/n,
        EMA_entropy=sum(r['EMA_entropy'] for r in items)/n)


def render(root):
    root=Path(root);r=read(root/'results.json');out=root/'figures';out.mkdir(exist_ok=True)
    if (not r['source_only'] or r['query_read'] or r['optimizer_updates'] or r['forward_packets']!=36000
            or not r['model_state_unchanged'] or not r['EMA_state_unchanged']):
        raise ValueError('Need complete unchanged zero-update source result')
    scenes=tuple(r['summary']);x=np.arange(len(scenes));width=.19
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    metrics=(('student_accuracy',100,'Student default-head accuracy (%)'),
        ('EMA_accuracy',100,'Own EMA default-head accuracy (%)'),
        ('fixed_parent_retention',100,'Student retention on fixed 703 clean-correct (%)'),
        ('EMA_entropy',1,'Own EMA prediction entropy (nats)'))
    for ax,(key,scale,title) in zip(axes.flat,metrics):
        for j,(view,label,color) in enumerate(zip(VIEWS,LABELS,COLORS)):
            values=[r['summary'][s][view]['overall'][key]*scale for s in scenes]
            ax.bar(x+(j-1.5)*width,values,width,label=label,color=color)
        ax.set_xticks(x,[s.replace('practical_','').replace('_','\n') for s in scenes]);ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if scale==100:ax.set_ylim(0,103)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Frozen A0 final50 / source L_dev720 / no updates / fixed full receiver actions')
    fig.savefig(out/'frozen_default_head_HQ.png');plt.close(fig)

    targets=('clean_EMA','HQ_shared_EMA','HQ_independent_EMA');target_labels=('Clean EMA','HQ shared EMA','HQ independent EMA')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,scene in enumerate(URBAN):
        identity=r['identity_target_comparisons'][scene]
        a=axes[0,col];xx=np.arange(3)
        a.bar(xx-.17,[identity[t]['pair_cosine'] for t in targets],.34,label='Same sample')
        a.bar(xx+.17,[identity[t]['closest_wrong_TX_cosine'] for t in targets],.34,label='Nearest wrong TX')
        a.set_xticks(xx,target_labels,rotation=15);a.set_ylim(0,1.03);a.set_title(scene.replace('practical_',''))
        for j,t in enumerate(targets):a.text(j,.03,f"gap {identity[t]['pair_minus_wrong_cosine']:+.3f}",ha='center',fontsize=8)
        a=axes[1,col]
        for j,(key,label) in enumerate(zip(('increment','feature_area','time_area'),('6 increments','15 feature areas','6 time areas'))):
            a.bar(xx+(j-1)*.23,[r['SIG_group_loss'][scene][t][key] for t in targets],.23,label=label)
        a.set_xticks(xx,target_labels,rotation=15);a.set_ylabel('Group mean SmoothL1 / fixed source RMS')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Full student vs frozen own EMA: identity separation and actual SIG112 groups')
    fig.savefig(out/'identity_and_SIG_groups.png');plt.close(fig)

    with gzip.open(root/'sample_bridge.jsonl.gz','rt',encoding='utf-8') as f:items=[json.loads(line) for line in f]
    if len(items)!=17280:raise ValueError('Need all source records, not a log sample')
    by_sample=defaultdict(list)
    for item in items:by_sample[item['scene'],item['physical_id']].append(item)
    if len(by_sample)!=4320:raise ValueError('Source physical panel coverage differs')
    frozen=('RX','state','lock','quality','full_added_noise_SNR_db','full_receiver_quality_proxy_db',
        'residual_CFO_hz','residual_kernel_length','applied_AGC_gain_db','fixed_parent_clean_correct','full_student_correct')
    for values in by_sample.values():
        if {v['view'] for v in values}!=set(VIEWS) or any(v[k]!=values[0][k] for v in values for k in frozen):
            raise ValueError('Source views no longer share frozen metadata/observations')
    strata=[];fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,scene in enumerate(URBAN[1:]):
        for col,(dimension,bins) in enumerate((('state',('LOS','PARTIAL','BLOCKED')),
                ('lock',('locked','degraded','unlocked')),('quality',('<3','3-8','8-15','>=15')))):
            a=axes[row,col];xx=np.arange(len(bins));counts=[]
            for j,(view,label,color) in enumerate(zip(VIEWS,LABELS,COLORS)):
                values=[]
                for value in bins:
                    selected=[v for v in items if v['scene']==scene and v['view']==view and v[dimension]==value]
                    summary=aggregate(selected) if selected else dict(count=0,EMA_accuracy=None)
                    strata.append(dict(scene=scene,view=view,dimension=dimension,stratum=value,**summary))
                    values.append(100*summary['EMA_accuracy'] if selected else np.nan)
                    if j==0:counts.append(len(selected))
                a.bar(xx+(j-1.5)*width,values,width,color=color,label=label)
            a.set_xticks(xx,[f'{v}\nn={n}' for v,n in zip(bins,counts)]);a.set_ylim(0,103)
            a.set_title(scene.replace('practical_','')+' / '+dimension);a.set_ylabel('EMA accuracy (%)')
            a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Source strata use full-view state/lock/added-noise SNR bins, frozen across interventions')
    fig.savefig(out/'source_state_quality_HQ.png');plt.close(fig)
    summary=dict(source_only=True,optimizer_updates=0,record_count=len(items),physical_scene_pairs=len(by_sample),
        source_strata=strata,paired_metadata_unchanged=True,figures=[str(p) for p in sorted(out.glob('*.png'))])
    (root/'derived_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('record_count','physical_scene_pairs','paired_metadata_unchanged','figures')}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);render(p.parse_args().root)
