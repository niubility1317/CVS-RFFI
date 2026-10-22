"""Compact tables/figures from the completed, CM-verified core factorial."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

SEEDS=(2026092701,2026092702,2026092703)
MAIN=('111','011','101','110')
READERS=('raw','DURA-L','DURA-LU')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','six_practical_mean')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def csvout(path,values):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
def moments(values):
    if len(values)!=3:raise ValueError('Exactly three actual F seeds are required')
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),positive_seeds=sum(v>0 for v in values))


def prepare(root):
    root=Path(root);source=root/'remote_analysis'
    verified=read(root/'independent_verification.json')
    if verified['status']!='VERIFIED' or verified['logical_cells']!=24 or verified['model_seeds']!=list(SEEDS):
        raise ValueError('Only the completed24-cell, actual-three-seed factorial may be reported')
    per=rows(source/'core_per_seed.csv')
    overall=[r for r in rows(source/'core_summary.csv') if r['test']=='main' and r['dimension']=='overall' and r['stratum']=='ALL']
    effects=[r for r in rows(source/'core_paired_effects.csv') if r['test']=='main' and r['dimension']=='overall' and r['stratum']=='ALL']
    values={(r['bits'],int(r['seed']),r['readout'],r['view'],r['metric']):float(r['percent']) for r in per
            if r['test']=='main' and r['dimension']=='overall' and r['stratum']=='ALL'}
    expected={(bits,seed,g,v,m) for bits in ('000','001','010','011','100','101','110','111') for seed in SEEDS
        for g in READERS for v in VIEWS for m in ('accuracy','macro_f1')}
    if set(values)!=expected:raise ValueError('Incomplete/duplicate overall result matrix')
    for r in overall:
        actual=moments([values[r['bits'],seed,r['readout'],r['view'],r['metric']] for seed in SEEDS])
        if any(abs(actual[k]-float(r[k]))>1e-10 for k in ('mean','sd')):raise ValueError('Overall three-seed moments differ')
    paired=[]
    for bits in ('000','001','010','011','100','101','110','111'):
        for view in VIEWS:
            for metric in ('accuracy','macro_f1'):
                for lo,hi in (('raw','DURA-L'),('raw','DURA-LU'),('DURA-L','DURA-LU')):
                    q=[values[bits,seed,hi,view,metric]-values[bits,seed,lo,view,metric] for seed in SEEDS]
                    paired.append(dict(bits=bits,contrast=hi+' minus '+lo,view=view,metric=metric,**moments(q)))
    cost=rows(source/'core_F_cost.csv')
    for r in cost:
        if r['shared_worker_seconds']:
            r['shared_worker_seconds']=float(r['shared_worker_seconds'])
        else:
            parents=sorted((root/'source_metadata'/('111-s'+r['seed'])).glob('ancestor*/completion.json'))
            if len(parents)!=3:raise ValueError('Original T1 own20/10/20 cost chain missing')
            data=[read(p) for p in parents]
            if any(x.get('target_access') is not False for x in data):raise ValueError('Source cost provenance differs')
            if sum(x['config']['epochs'] for x in data)!=50:raise ValueError('Original T1 cumulative E50 differs')
            r['shared_worker_seconds']=sum(x['elapsed_seconds'] for x in data)
        r['source_metadata_complete']=True
    physics=[];expected_ids=None;geometry=defaultdict(dict);interventions=defaultdict(dict)
    for p in sorted((root/'source_physics').glob('*.json')):
        bits=p.name[:3];seed=int(p.name.split('-s',1)[1].split('-',1)[0]);d=read(p)
        if d['source_U_reads'] or d['target_reads'] or d['fitted_probe_updates'] or d['samples']!=720:
            raise ValueError('Source-L-only physics diagnostic changed permissions/coverage')
        if expected_ids is None:expected_ids=d['physical_ids']
        elif d['physical_ids']!=expected_ids:raise ValueError('Source diagnostic physical-ID pairing differs')
        if (bits,seed) in geometry:raise ValueError('Repeated source diagnostics must not count as independent F seeds')
        geometry[bits,seed]=d['geometry']
        for q in d['interventions']:interventions[bits,q['intervention']][seed]=q
    if set(geometry)!={(b,s) for b in ('000','001','010','011','100','101','110','111') for s in SEEDS}:
        raise ValueError('All24 frozen source-L diagnostic panels are required')
    for bits in ('000','001','010','011','100','101','110','111'):
        for name in ('same_TX_cross_RX_distance','different_TX_same_RX_distance','separation_margin'):
            physics.append(dict(bits=bits,source_metric=name,**moments([geometry[bits,s][name] for s in SEEDS])))
    sensitivity=[]
    for (bits,name),data in sorted(interventions.items()):
        if set(data)!=set(SEEDS):raise ValueError('Missing actual source diagnostic seed')
        for metric in ('feature_cosine_distance','default_head_flip_fraction','reference_descriptor_relative_L2'):
            sensitivity.append(dict(bits=bits,intervention=name,metric=metric,**moments([data[s][metric] for s in SEEDS])))
    return overall,effects,paired,cost,physics,sensitivity,verified


def plot(overall,effects,physics,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');fig.savefig(folder/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
    lookup={(r['bits'],r['readout'],r['view'],r['metric']):r for r in overall}
    bits=('000','001','010','011','100','101','110','111');colors=('#37474f','#0072b2','#d55e00')
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),constrained_layout=True)
    for ax,view,title in zip(axes,('clean','six_practical_mean'),('Clean','Six practical scenes')):
        for g,color,offset,marker in zip(READERS,colors,(-.12,0,.12),('o','s','^')):
            rs=[lookup[b,g,view,'accuracy'] for b in bits]
            ax.errorbar([i+offset for i in range(8)],[float(r['mean']) for r in rs],yerr=[float(r['sd']) for r in rs],
                color=color,marker=marker,markersize=3,capsize=2,linewidth=.8,label=g)
        ax.set_xticks(range(8),bits);ax.set_xlabel('ABC combination');ax.set_ylabel('Accuracy (%)');ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,ncol=3,loc='lower right');save(fig,'core_factorial_accuracy')
    lookup_e={(r['effect'],r['readout'],r['view'],r['metric']):r for r in effects}
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),constrained_layout=True)
    for ax,view,title in zip(axes,('clean','six_practical_mean'),('Clean: full minus removal','Six scenes: full minus removal')):
        for g,color,offset,marker in zip(READERS,colors,(-.16,0,.16),('o','s','^')):
            rs=[lookup_e[a+'_at_other_groups_on',g,view,'accuracy'] for a in ('A','B','C')]
            ax.errorbar([i+offset for i in range(3)],[float(r['mean']) for r in rs],yerr=[float(r['sd']) for r in rs],
                color=color,marker=marker,markersize=3,capsize=2,linewidth=.8,label=g)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(3),('A: CRIC','B: consistency','C: selective SSL'))
        ax.set_ylabel('Paired difference (pp)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,ncol=3,loc='upper right');save(fig,'core_leave_one_group_out')
    fig,ax=plt.subplots(figsize=(3.45,2.5),constrained_layout=True)
    for view,color,offset,label in (('clean',colors[0],-.06,'Clean'),('six_practical_mean',colors[1],.06,'Six scenes')):
        rs=[lookup_e[e,'raw',view,'accuracy'] for e in ('AB_interaction_average','AC_interaction_average','BC_interaction_average','ABC_interaction')]
        ax.errorbar([i+offset for i in range(4)],[float(r['mean']) for r in rs],yerr=[float(r['sd']) for r in rs],
            color=color,marker='o',markersize=3,capsize=2,linewidth=.8,label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(4),('AB','AC','BC','ABC'));ax.set_ylabel('Raw interaction (pp)')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);save(fig,'core_raw_interactions')
    p={(r['bits'],r['source_metric']):r for r in physics};fig,axes=plt.subplots(1,2,figsize=(7.15,2.55),constrained_layout=True)
    for ax,metric,title in zip(axes,('same_TX_cross_RX_distance','separation_margin'),('L-source same-TX cross-RX distance','L-source separation margin')):
        rs=[p[b,metric] for b in MAIN]
        ax.errorbar(range(4),[r['mean'] for r in rs],yerr=[r['sd'] for r in rs],color=colors[0],marker='o',markersize=3,capsize=2,linewidth=.8)
        ax.set_xticks(range(4),('Full','Without A','Without B','Without C'));ax.set_title(title);ax.set_ylabel('Cosine-distance units')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    save(fig,'core_source_geometry')


def run(root,output,figures=None):
    overall,effects,paired,cost,physics,sensitivity,verified=prepare(root)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    csvout(out/'core_overall_summary.csv',overall);csvout(out/'core_overall_effects.csv',effects)
    csvout(out/'core_restoration_paired.csv',paired);csvout(out/'core_completed_F_cost.csv',cost)
    csvout(out/'core_source_geometry.csv',physics);csvout(out/'core_source_interventions.csv',sensitivity)
    (out/'core_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
    new_seconds=sum(r['shared_worker_seconds'] for r in cost if r['new_F']=='True')
    all_seconds=sum(r['shared_worker_seconds'] for r in cost)
    (out/'core_cost_summary.json').write_text(json.dumps(dict(logical_F_models=24,new_F_models=12,reused_F_models=12,
        logical_F_epochs=1200,logical_F_updates=266400,new_F_epochs=600,new_F_updates=133200,
        new_F_shared_worker_hours=new_seconds/3600,all24_F_shared_worker_hours_including_original_T1=all_seconds/3600,
        source_physics_panels=24,source_L_samples_per_panel=720,source_interventions_per_panel=11,
        G_training_in_this_panel=0,head_fit_updates=0,target_truth_reads_in_analysis=0,
        caution='Elapsed shared-worker cost under differing historical contention, not exclusive GPU-hours or sum of end-to-end completion times. G ancestors and other experiments are separate.'),indent=2)+'\n',encoding='utf-8')
    if figures:plot(overall,effects,physics,figures)
    print(json.dumps(dict(status='CORE_REPORT_READY',logical_cells=24,verified_CMs=verified['serialized_CM_recomputed'],
        numeric_comparisons=verified['numeric_cells_compared'],new_F_shared_worker_hours=new_seconds/3600,figures=4 if figures else 0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--figures')
    a=p.parse_args();run(a.input,a.output,a.figures)
