"""Eight-arm figures from complete evidence; never ranks arms for training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=OUT=None
ARMS=('A0','A-T','A-F','A-P','A-R','A-S','A-J','A-ALL')
KEYS=('T','F','P','R','S','J')
COLORS=('#666666','#2878b5','#e68b26','#368657','#c74f50','#8e62a8','#93705a','#cf76a7')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def training_and_formal():
    audit=read(ROOT/'final/training_audit.json');summary=read(ROOT/'final/summary.json')
    assert audit['status']=='COMPLETE_LOG_AUDIT' and len(audit['rows'])==8 and audit['total_updates_parsed']==88800
    rows={r['row_id'].rsplit('-s',1)[0]:r for r in audit['rows']}
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    keys=('urban/clean_CE','urban/satellite_CE','urban/auxiliary_weighted_loss',
          'train/grad_before_clip','urban/T_variance','urban/S_variance')
    for arm,color in zip(ARMS,COLORS):
        for ax,key in zip(axes.flat,keys):
            values=rows[arm]['curves'][key];assert len(values)==50
            ax.plot(range(1,51),values,label=arm,color=color,lw=1.4)
            ax.set_title(key.split('/')[1].replace('_',' '));ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            ax.axvline(5.5,color='grey',ls='--',lw=.7);ax.axvline(33.5,color='grey',ls=':',lw=.7)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('All 50 epochs; dashed: auxiliary starts, dotted: native old-gate U starts')
    fig.savefig(OUT/'complete_training_curves.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for arm,color in zip(ARMS,COLORS):
        records=rows[arm]['learning']['records'];epochs=[r['epoch'] for r in records]
        for ax,key in zip(axes,('clean_accuracy','low','worst_urban_RX')):
            values=[r['scene_accuracy']['practical_low_urban'] if key=='low' else r[key] for r in records]
            ax.plot(epochs,np.array(values)*100,'o-',ms=2,label=arm,color=color,lw=1.2)
            ax.set_title(key.replace('_',' '));ax.set_xlabel('Epoch');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8)
    fig.suptitle('Actual internal L_dev observations (14 fixed epochs); only final50 controls family expansion')
    fig.savefig(OUT/'source_learning_progress.png',dpi=180);plt.close(fig)

    records={r['arm']:r for r in summary['rows']};baseline=records['A0']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for ax,key in zip(axes,('low_urban','urban_mean','clean')):
        bars=ax.bar(ARMS,[100*records[a][key] for a in ARMS],color=COLORS)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.tick_params(axis='x',rotation=35)
        ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title(key.replace('_',' '))
        ax.axhline(100*baseline[key],color='black',ls=':',lw=1)
        if key=='low_urban':ax.axhline(60,color='firebrick',ls='--',lw=1,label='Goal 60%');ax.legend(fontsize=8)
    fig.suptitle('All eight fixed default heads/formal IQ; model seed 2026092701 only; no best-position claim')
    fig.savefig(OUT/'per_seed_formal_accuracy.png',dpi=180);plt.close(fig)


def heatmap(ax,array,title,columns,vmin=None,vmax=None,cmap='viridis'):
    artist=ax.imshow(array,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(8),ARMS);ax.set_title(title,fontsize=10)
    for row in range(8):
        for col in range(len(columns)):
            value=array[row,col];ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=7,
                color='white' if artist.norm(value)<.45 else 'black')
    return artist


def source_mechanism():
    data=read(ROOT/'source_branch_results.json')
    assert data['status']=='ANALYZED' and data['source_only'] and not data['query_read'] and data['source_role']=='L_dev'
    assert data['forward_packets']==46800 and data['optimizer_updates']==0 and data['V_forward_packets']==0
    assert set(data['rows'])==set(ARMS)
    assert all(v['model_state_unchanged'] and v['EMA_state_unchanged'] for v in data['rows'].values())
    rows=data['rows'];low='practical_low_urban';scenes=('practical_high_urban','practical_mid_urban',low)
    fig,axes=plt.subplots(1,3,figsize=(16,6.5),constrained_layout=True)
    for ax,key,title in zip(axes,('clean_pair_cosine','correct_minus_wrong_cosine','mean_variance'),
                            ('Actual branch clean/satellite cosine','Own clean minus nearest wrong-TX cosine','Low urban actual branch variance')):
        values=np.array([[rows[a]['scenes'][low]['actual_branch_variance'][b][key] if key=='mean_variance' else
            rows[a]['scenes'][low]['overall']['identity'][b][key] for b in KEYS] for a in ARMS])
        obj=heatmap(ax,values,title,KEYS,cmap='coolwarm' if key=='correct_minus_wrong_cosine' else 'viridis')
        fig.colorbar(obj,ax=ax,shrink=.7)
    fig.suptitle('Source held-out L_dev720 only: T/F fuse inputs, P/R/S gain-before160, actual J')
    fig.savefig(OUT/'actual_branch_identity_geometry.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    x=np.arange(3);width=.10
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        values=[rows[arm]['scenes'][scene]['overall'] for scene in scenes]
        for ax,key,scale in zip(axes,('fixed_A0_retention','margin','A0_errors_repaired'),(100,1,1)):
            ax.bar(x+(i-3.5)*width,[scale*v[key] for v in values],width,label=arm,color=color)
    for ax,title in zip(axes,('Fixed A0 clean-correct retention (%)','Default-head normalized margin','A0 errors repaired (also inspect new errors)')):
        ax.set_title(title,fontsize=10);ax.set_xticks(x,('High urban','Mid urban','Low urban'));ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=2,fontsize=7)
    fig.suptitle(f"Fixed A0 final50 clean-correct set: {data['fixed_A0_clean_correct_count']}/720; source diagnosis only")
    fig.savefig(OUT/'source_default_head_retention_margin.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    gain=np.array([[rows[a]['scenes'][low]['gain_contribution'][b]['mean_absolute_gain'] for b in ('P','R','S')] for a in ARMS])
    amplitude=np.array([[rows[a]['scenes'][low]['gain_contribution'][b]['mean_contribution_to_joint_norm_ratio'] for b in ('P','R','S')] for a in ARMS])
    for ax,values,title in zip(axes,(gain,amplitude),('Mean absolute deployed gain','Mean gain-vector norm / joint norm (not fractions)')):
        obj=heatmap(ax,values,title,('P','R','S'));fig.colorbar(obj,ax=ax,shrink=.75)
    fig.suptitle('Source low urban; ratios may exceed 1 through cancellation; T/F are nonlinear fuse inputs')
    fig.savefig(OUT/'gain_and_contribution_amplitude.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    aggregate=[]
    for arm in ARMS:
        records=rows[arm]['gradients']
        full=[g['T_S_full_feature_cosine'] for g in records if g['T_S_full_feature_cosine'] is not None]
        shared=[g['T_S_shared_only']['cosine'] for g in records if g['T_S_shared_only'] and g['T_S_shared_only']['cosine'] is not None]
        aggregate.append(dict(arm=arm,T_S_full_cosines=full,T_S_shared_cosines=shared,
            shared_parameter_names=sorted({name for g in records for name in g['T_S_shared_parameter_names']}),
            shared_scope='Current batch gradients of both losses nonzero >1e-12; independent projections excluded'))
    for ax,key,title in zip(axes,('T_S_full_cosines','T_S_shared_cosines'),('Full actual feature parameter vectors','Shared nonzero parameter intersection only')):
        ax.boxplot([v[key] or [np.nan] for v in aggregate],showfliers=False)
        ax.set_xticks(range(1,len(ARMS)+1),ARMS)
        ax.axhline(0,color='grey',ls=':',lw=1);ax.set_ylim(-1.05,1.05);ax.set_ylabel('T/S loss-gradient cosine');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen eval source mid/low batches; raw losses, not clipped stochastic AdamW updates')
    fig.savefig(OUT/'T_S_shared_gradient_angles.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for ax,field,title in zip(axes,('state','lock','quality'),('Visibility state','Receiver lock','Added-noise SNR (not total SNR)')):
        present={s[field] for arm in ARMS for s in rows[arm]['scenes'][low]['strata']}
        order={'state':('LOS','PARTIAL','BLOCKED'),'lock':('locked','degraded','unlocked'),
            'quality':('<3','3-8','8-15','>=15')}[field]
        values=[value for value in order if value in present]+sorted(present-set(order));xp=np.arange(len(values))
        for arm,color in zip(ARMS,COLORS):
            strata=rows[arm]['scenes'][low]['strata'];rates=[]
            for value in values:
                subset=[s for s in strata if s[field]==value];count=sum(s['count'] for s in subset)
                rates.append(100*sum(s['count']*s['accuracy'] for s in subset)/count)
            ax.plot(xp,rates,'o-',label=arm,color=color,lw=1.2,ms=3)
        counts={v:sum(s['count'] for s in rows['A0']['scenes'][low]['strata'] if s[field]==v) for v in values}
        ax.set_xticks(xp,[f'{v}\nn={counts[v]}' for v in values],rotation=20,fontsize=8)
        ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=2,fontsize=7)
    fig.suptitle('Source held-out low-urban720 common draws; state/quality association, not isolated causal damage removal')
    fig.savefig(OUT/'source_state_quality_strata.png',dpi=180);plt.close(fig)
    (ROOT/'source_gradient_aggregate.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',required=True,type=Path)
    parser.add_argument('--phase',choices=('training','source','all'),default='all')
    args=parser.parse_args();ROOT=args.artifacts.resolve();OUT=ROOT/'final/figures'
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase in ('training','all'):training_and_formal()
    if args.phase in ('source','all'):source_mechanism()
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETE_EVIDENCE',figures=[str(p) for p in OUT.glob('*.png')],seed_count=1)))
