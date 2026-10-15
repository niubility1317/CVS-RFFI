"""Scientific figures from complete real artifacts only; no model imports or ranking."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('D0','D1','D2','D3','D4')
COLORS=('#666666','#2878b5','#e68b26','#368657','#8e62a8')
BRANCHES=('T','F','P','R','S','J')
URBAN=('practical_high_urban','practical_mid_urban','practical_low_urban')
NAMES=('High urban','Mid urban','Low urban')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def save(fig,out,name):
    fig.savefig(out/name,dpi=180);plt.close(fig)


def grouped(ax,values,names,ylabel):
    x=np.arange(len(names));width=.145
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        ax.bar(x+(i-2)*width,values[arm],width,label=arm,color=color)
    ax.set_xticks(x,names);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)


def heatmap(ax,values,columns,title,cmap='viridis'):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Finite actual heatmap measurements required')
    limit=max(float(np.abs(values).max()),1e-8)
    scale=dict(vmin=-limit,vmax=limit) if cmap=='coolwarm' else {}
    obj=ax.imshow(values,aspect='auto',cmap=cmap,**scale)
    ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(len(ARMS)),ARMS);ax.set_title(title,fontsize=10)
    for i in range(len(ARMS)):
        for j in range(len(columns)):
            value=values[i,j]
            ax.text(j,i,f'{value:.3g}',ha='center',va='center',fontsize=8,
                color='white' if obj.norm(value)<.45 else 'black')
    return obj


def source_figures(root,out):
    v=read(root/'results.json')
    if (v['status']!='ANALYZED' or not v['source_only'] or v['query_read'] or v['U_truth_read']
            or v['V_forward_packets'] or v['optimizer_updates'] or v['EMA_updates'] or v['head_fit_updates']
            or v['forward_packets']!=36722 or set(v['rows'])!=set(ARMS)):
        raise ValueError('Complete five-model frozen source results required')
    rows=v['rows']
    if not all(r['model_state_unchanged'] and r['EMA_state_unchanged'] for r in rows.values()):
        raise ValueError('Frozen state evidence missing')
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    grouped(axes[0,0],{a:[100*rows[a]['clean_accuracy']]+[100*rows[a]['scenes'][s]['overall']['accuracy'] for s in URBAN] for a in ARMS},
        ('Clean',*NAMES),'Accuracy (%)');axes[0,0].set_title('Original frozen default heads');axes[0,0].set_ylim(0,105)
    grouped(axes[0,1],{a:[100*rows[a]['scenes'][s]['overall']['fixed_A0_retention'] for s in URBAN] for a in ARMS},
        NAMES,'Retention (%)');axes[0,1].set_title('Common fixed A0 clean-correct subset');axes[0,1].set_ylim(0,105)
    for arm,color in zip(ARMS,COLORS):
        axes[1,0].plot(NAMES,[rows[arm]['scenes'][s]['overall']['margin'] for s in URBAN],'o-',label=arm,color=color)
    axes[1,0].axhline(0,color='grey',ls=':',lw=1);axes[1,0].set_ylabel('Logit margin / 30');axes[1,0].set_title('True class minus strongest wrong class')
    x=np.arange(5);low=URBAN[-1]
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        r=rows[arm]['scenes'][low]['overall']
        axes[1,1].bar(i-.18,r['A0_errors_repaired'],.35,color=color)
        axes[1,1].bar(i+.18,r['A0_correct_new_errors'],.35,color=color,hatch='///',alpha=.65)
    axes[1,1].set_xticks(x,ARMS);axes[1,1].set_title('Low urban: repairs (solid), new errors (hatched)');axes[1,1].set_ylabel('Packets within fixed subset')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(ncol=5,fontsize=8)
    fig.suptitle(f"Source L_dev720 only; fixed A0 correct subset n={v['fixed_A0_clean_correct_count']}; no new training here")
    save(fig,out,'source_default_head_retention_margin.png')

    fig,axes=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
    values=[[rows[a]['scenes'][low]['overall']['identity'][b]['correct_minus_wrong_cosine'] for b in BRANCHES] for a in ARMS]
    obj=heatmap(axes[0],values,BRANCHES,'Same packet minus nearest wrong-TX cosine','coolwarm');fig.colorbar(obj,ax=axes[0],shrink=.8)
    values=[[rows[a]['scenes'][low]['branch_variance'][b]['mean_variance']/max(rows[a]['clean_branch_variance'][b]['mean_variance'],1e-12)
        for b in BRANCHES] for a in ARMS]
    obj=heatmap(axes[1],values,BRANCHES,'Low-urban / clean branch variance');fig.colorbar(obj,ax=axes[1],shrink=.8)
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        r=rows[arm]['scenes'][low]['overall']
        axes[2].bar(i-.17,r['urban_to_clean_reference_gap'],.32,color=color)
        axes[2].bar(i+.17,r['clean_to_urban_reference_gap'],.32,color=color,hatch='///',alpha=.65)
    axes[2].axhline(0,color='grey',ls=':',lw=1);axes[2].set_xticks(np.arange(5),ARMS)
    axes[2].set_title('Joint matched-reference mean cosine gap\nu->c solid, c->u hatched',fontsize=10)
    axes[2].set_ylabel('Same-TX positive minus different-TX negative');axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('Source low urban: shared reference RX/day and cross-RX anchors; geometry is separate from head margins')
    save(fig,out,'source_identity_geometry.png')

    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    order=dict(state=('LOS','PARTIAL','BLOCKED'),lock=('locked','degraded','unlocked'),quality=('<3','3-8','8-15','>=15'))
    for rr,scene in enumerate(URBAN[1:]):
        for ax,field in zip(axes[rr],('state','lock','quality')):
            reference=rows['D1']['scenes'][scene]['strata'];present={r[field] for r in reference}
            labels=[value for value in order[field] if value in present]+sorted(present-set(order[field]))
            counts={value:sum(r['count'] for r in reference if r[field]==value) for value in labels}
            for arm,color in zip(ARMS,COLORS):
                strata=rows[arm]['scenes'][scene]['strata'];rates=[]
                for value in labels:
                    selected=[r for r in strata if r[field]==value];n=sum(r['count'] for r in selected)
                    if n!=counts[value] or not n:raise ValueError('All models must share identical source strata counts')
                    rates.append(100*sum(r['count']*r['accuracy'] for r in selected)/n)
                ax.plot(range(len(labels)),rates,'o-',label=arm,color=color,ms=3,lw=1.3)
            ax.set_xticks(range(len(labels)),[f'{value}\nn={counts[value]}' for value in labels],rotation=15,fontsize=8)
            ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(scene.replace('practical_','')+' / '+field)
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Common held-out source observations; added-noise SNR bins are not total SNR; strata show association')
    save(fig,out,'source_state_quality_strata.png')

    fig,axes=plt.subplots(1,2,figsize=(11,5.5),constrained_layout=True)
    for ax,field,title in zip(axes,('mean_absolute_gain','mean_contribution_to_joint_norm_ratio'),
            ('Mean absolute deployed gain','Gain-vector contribution norm / joint norm')):
        values=[[rows[a]['scenes'][low]['gain_contribution'][b][field] for b in ('P','R','S')] for a in ARMS]
        obj=heatmap(ax,values,('P','R','S'),title);fig.colorbar(obj,ax=ax,shrink=.8)
    fig.suptitle('Actual low-urban fusion; contribution ratios are not fractions and may exceed one through cancellation')
    save(fig,out,'source_gain_contribution.png')

    arms=('D2','D3','D4');fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    aggregate=[]
    for arm in arms:
        records=rows[arm]['gradients']
        if any(any(abs(value)>1e-8 for value in r['auxiliary_boundary_norms'].values()) for r in records):
            raise ValueError('Extra W/gain gradient boundary violated')
        aggregate.append(dict(arm=arm,sat_cosines=[r['global_vs_satellite']['cosine'] for r in records if r['global_vs_satellite']['cosine'] is not None],
            clean_cosines=[r['global_vs_clean']['cosine'] for r in records if r['global_vs_clean']['cosine'] is not None],
            ratios=[r['global_vs_satellite']['ratio'] for r in records]))
    for ax,key,title in zip(axes,('sat_cosines','clean_cosines','ratios'),('Extra vs satellite CE cosine','Extra vs clean CE cosine','Extra / satellite CE norm ratio')):
        ax.boxplot([r[key] or [np.nan] for r in aggregate],showfliers=False);ax.set_xticks(range(1,4),arms)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
        if key!='ratios':ax.axhline(0,color='grey',ls=':',lw=1);ax.set_ylim(-1.05,1.05)
    fig.suptitle('Frozen eval paired mid/low source gradients; not stochastic, clipped or AdamW updates')
    save(fig,out,'source_paired_gradient_angles.png')
    (root/'source_gradient_summary.json').write_text(json.dumps(aggregate,indent=2)+'\n',encoding='utf-8')


def training_figures(root,out):
    audit=read(root/'training_audit.json');summary=read(root/'summary.json')
    if (audit['status']!='COMPLETE_TRAINING_EVIDENCE' or audit['new_full_log_steps_read']!=33300
            or audit['reused_verified_steps']!=22200 or len(audit['rows'])!=5 or summary['seed_count']!=1):
        raise ValueError('Complete new and reused training evidence required')
    rows={r['arm']:r for r in audit['rows']};results={r['arm']:r for r in summary['rows']}
    if set(rows)!=set(ARMS) or set(results)!=set(ARMS):raise ValueError('Five fixed rows required')
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    keys=('urban/clean_CE','urban/satellite_CE','urban/auxiliary_weighted_loss','train/grad_before_clip')
    for arm,color in zip(ARMS,COLORS):
        for ax,key in zip(axes.flat,keys):
            values=rows[arm]['curves'].get(key)
            if values is None or len(values)!=50 or any(v is None for v in values):raise ValueError('Incomplete actual curve: '+arm+'/'+key)
            ax.plot(range(1,51),values,label=arm,color=color,lw=1.4)
            ax.set_title(key.split('/')[1].replace('_',' '));ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            ax.axvline(5.5,color='grey',ls='--',lw=.7);ax.axvline(33.5,color='grey',ls=':',lw=.7)
    axes[0,0].legend(ncol=5,fontsize=8)
    fig.suptitle('Full 11100 steps per model; three new logs + two unchanged audited controls; objectives have different units')
    save(fig,out,'complete_training_curves.png')

    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for arm,color in zip(ARMS,COLORS):
        records=rows[arm]['learning']['records'];epochs=[r['epoch'] for r in records]
        for ax,key in zip(axes,('clean_accuracy','low','worst_urban_RX')):
            values=[r['scene_accuracy']['practical_low_urban'] if key=='low' else r[key] for r in records]
            ax.plot(epochs,np.asarray(values)*100,'o-',ms=3,label=arm,color=color,lw=1.3)
            ax.set_xlabel('Epoch');ax.set_title(key.replace('_',' '));ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle('Actual L_dev720 observations; source qualification uses only the predetermined final50 rule')
    save(fig,out,'source_learning_progress.png')

    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for ax,key,reference in zip(axes,('clean','low_urban','urban_mean'),('D0','D1','D1')):
        bars=ax.bar(ARMS,[100*results[a][key] for a in ARMS],color=COLORS)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(key.replace('_',' '))
        ax.axhline(100*results[reference][key],color='black',ls=':',lw=1,label='Reference '+reference)
        if key=='low_urban':ax.axhline(60,color='firebrick',ls='--',lw=1,label='Goal 60%')
        ax.legend(fontsize=8)
    fig.suptitle('Fixed formal IQ/default heads; one model seed only; clean-vs-D0 and low-vs-D1 are separate objectives')
    save(fig,out,'formal_per_seed_accuracy.png')

    metric_rows={a:read(root/'formal_rows'/rows[a]['row_id']/'scoring/scores.json')['results'] for a in ARMS}
    receiver_ids=sorted(results['D1']['urban_by_RX'],key=int)
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
    for ax,scene in zip(axes,('practical_mid_urban','practical_low_urban','urban_mean')):
        values=[]
        for arm in ARMS:
            if scene=='urban_mean':values.append([100*results[arm]['urban_by_RX'][rx] for rx in receiver_ids])
            else:
                measured={r['stratum']:r['accuracy'] for r in metric_rows[arm] if r['dimension']=='receiver' and r['view']==scene}
                if set(measured)!=set(receiver_ids):raise ValueError('All registered formal receivers required')
                values.append([100*measured[rx] for rx in receiver_ids])
        obj=heatmap(ax,values,[f'RX{rx}' for rx in receiver_ids],scene.replace('practical_','')+' accuracy (%)')
        fig.colorbar(obj,ax=ax,shrink=.8)
    fig.suptitle('Every fixed formal receiver; original truth-last scores reused; no selective reruns or receiver exclusion')
    save(fig,out,'formal_receiver_strata.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path)
    parser.add_argument('--training-root',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if not args.source_root and not args.training_root:parser.error('Provide actual complete source and/or training artifacts')
    if args.source_root:source_figures(args.source_root,args.output)
    if args.training_root:training_figures(args.training_root,args.output)
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETE_REAL_EVIDENCE',figures=[str(p.resolve()) for p in args.output.glob('*.png')],seed_count=1)))
