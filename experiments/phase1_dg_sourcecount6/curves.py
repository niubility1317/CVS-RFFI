"""Analyze frozen score aggregates; source-count curves never feed training."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

COUNTS=(1,2,3,4,6)
FAMILIES=('riei','drift','ibrff','sdg')
SEEDS=(2026092701,2026092702,2026092703)
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
REGIMES=(('natural','supervised'),('natural','pl'),('fixed','supervised'))
LABELS=dict(riei='RIEI',drift='DRIFT',ibrff='IB-RFF',sdg='SDG')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def moments(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),actual_seeds=len(values))


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def check_cm(result):
    cm=result['confusion'];assert len(cm)==6 and all(len(row)==6 for row in cm)
    assert all(isinstance(n,int) and n>=0 for row in cm for n in row)
    n=sum(map(sum,cm));assert n==result['count'] and n>0
    accuracy=sum(cm[i][i] for i in range(6))/n
    f1=[]
    for i in range(6):
        den=sum(cm[i])+sum(row[i] for row in cm)
        f1.append(2*cm[i][i]/den if den else 0.)
    assert math.isclose(accuracy,result['accuracy'],abs_tol=1e-12)
    assert math.isclose(statistics.mean(f1),result['macro_f1'],abs_tol=1e-12)


def analyze(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    payload=read(source);assert payload['status']=='ANALYZED' and len(payload['rows'])==360
    expected={(m,b,r,f,v,s) for m in COUNTS for b,r in REGIMES for f in FAMILIES for v in ('ce','full') for s in SEEDS}
    seen=set();series={};public=[];costs=[];matrices=0;target_names=None
    for row in payload['rows']:
        c=row['config'];regime='pl' if c['U_s_use'] else 'supervised'
        key=(c['source_count'],c['label_budget'],regime,c['family'],c['variant'],c['model_seed'])
        assert key in expected and key not in seen and row['independent_prediction_audit'];seen.add(key)
        results=row['results'];assert {r['view'] for r in results}==set(SCENES)
        for r in results:
            if 'confusion' in r:check_cm(r);matrices+=1
        values={}
        for view in SCENES:
            total=[r for r in results if r['view']==view and r['dimension']=='overall'];assert len(total)==1
            total=total[0];assert total['count']==144000
            receivers=[r for r in results if r['view']==view and r['dimension']=='receiver']
            names={r['stratum'] for r in receivers}
            if target_names is None:target_names=names
            assert len(receivers)==len(names)==6 and names==target_names
            assert all(r['count']==24000 for r in receivers)
            assert [[sum(r['confusion'][i][j] for r in receivers) for j in range(6)] for i in range(6)]==total['confusion']
            values[view]=dict(accuracy=statistics.mean(r['accuracy'] for r in receivers),macro_f1=total['macro_f1'],worst_rx=min(r['accuracy'] for r in receivers))
        values['satellite_mean']={metric:statistics.mean(values[v][metric] for v in SCENES[1:]) for metric in values['clean']}
        series[key]=values
        cost=row['cost'];stages=cost['stages'];selected=cost.get('selected_training_telemetry',{})
        clean_cost=dict(row_id=c['row_id'],source_count=c['source_count'],budget=c['label_budget'],regime=regime,family=c['family'],variant=c['variant'],seed=c['model_seed'],
            training_seconds=cost['seconds'],total_epochs=sum(s['epochs'] for s in stages),stages=len(stages),
            batch_updates=sum(s['updates'] for s in stages),optimizer_updates=sum(s['optimizer_steps'] for s in stages),
            labeled_exposure=sum(s['labeled_exposure'] for s in stages),satellite_examples=cost['satellite_examples'],
            satellite_applied_examples=cost['satellite_applied_examples'],pl_seen=cost['pl_seen'],pl_accepted=cost['pl_accepted'],
            pl_acceptance=cost['pl_accepted']/cost['pl_seen'] if cost['pl_seen'] else None,
            selected_epoch=cost['selected_epoch'],selected_pl_seen=selected.get('pl_seen',0),selected_pl_accepted=selected.get('pl_accepted',0))
        assert clean_cost['labeled_exposure']==clean_cost['satellite_examples']
        assert clean_cost['pl_seen']==(clean_cost['labeled_exposure'] if regime=='pl' else 0)
        assert 0<=clean_cost['pl_accepted']<=clean_cost['pl_seen']
        costs.append(clean_cost)
        allowed=('row_id','source_count','label_budget','U_s_use','family','variant','model_seed')
        public_cost={k:v for k,v in cost.items() if k in ('stages','seconds','satellite_examples','satellite_applied_examples','pl_seen','pl_accepted','selected_epoch','selected_training_telemetry')}
        public.append(dict(config={k:c[k] for k in allowed},results=results,cost=public_cost,independent_prediction_audit=True))
    assert seen==expected
    summary=[];dg=[];pl=[]
    for m in COUNTS:
        for budget,regime in REGIMES:
            for family in FAMILIES:
                for view in (*SCENES,'satellite_mean'):
                    for variant in ('ce','full'):
                        for metric in ('accuracy','macro_f1','worst_rx'):
                            values=[series[m,budget,regime,family,variant,s][view][metric] for s in SEEDS]
                            summary.append(dict(source_count=m,budget=budget,regime=regime,family=family,variant=variant,view=view,metric=metric,**moments(values)))
                    for metric in ('accuracy','macro_f1','worst_rx'):
                        differences=[series[m,budget,regime,family,'full',s][view][metric]-series[m,budget,regime,family,'ce',s][view][metric] for s in SEEDS]
                        dg.append(dict(source_count=m,budget=budget,regime=regime,family=family,view=view,metric=metric,**moments(differences),positive_pairs=sum(v>0 for v in differences)))
            if budget=='natural' and regime=='pl':
                for family in FAMILIES:
                    for variant in ('ce','full'):
                        for view in (*SCENES,'satellite_mean'):
                            for metric in ('accuracy','macro_f1','worst_rx'):
                                values=[series[m,budget,'pl',family,variant,s][view][metric]-series[m,budget,'supervised',family,variant,s][view][metric] for s in SEEDS]
                                pl.append(dict(source_count=m,budget=budget,family=family,variant=variant,view=view,metric=metric,**moments(values),positive_pairs=sum(v>0 for v in values)))
    write_csv(out/'sourcecount_summary.csv',summary);write_csv(out/'sourcecount_dg_paired.csv',dg);write_csv(out/'sourcecount_pl_paired.csv',pl);write_csv(out/'sourcecount_training_cost.csv',costs)
    (out/'curve_inputs.json').write_text(json.dumps(dict(status='ANALYZED',rows=public),separators=(',',':'))+'\n',encoding='utf-8')
    nonwins=[r for r in dg if r['metric']=='accuracy' and r['view'] in ('clean','satellite_mean') and r['mean']<=0]
    zero_pl=[r for r in costs if r['regime']=='pl' and r['pl_accepted']==0]
    verify=dict(status='VERIFIED',rows=360,source_counts=list(COUNTS),excluded_source_counts=[5],new_training=360,reused=0,
        raw_view_cells=2520,source_chain=[1,3,4,6,8,11],target_receivers=[0,2,5,7,9,10],target_receiver_names=sorted(target_names),packets_per_view=144000,seeds=list(SEEDS),
        checked_confusion_matrices=matrices,summary_records=len(summary),DG_pair_records=len(dg),PL_pair_records=len(pl),
        core_DG_nonpositive_means=len(nonwins),core_DG_comparisons=120,zero_accepted_PL_rows=len(zero_pl),PL_rows=120,
        summed_worker_training_elapsed_hours=sum(r['training_seconds'] for r in costs)/3600,
        measured_training_time_note='Sum of per-worker source-training elapsed times under actual sharing, not device-exclusive compute hours or wall-clock makespan.',
        remote_prediction_audit='All360 rows independently audited from complete frozen predictions before this aggregate-only recomputation.')
    (out/'sourcecount_verification.json').write_text(json.dumps(verify,indent=2)+'\n',encoding='utf-8')
    lines=['# 外部方法源接收机数量：完整 360 行结果','',
        '有效 m=1/2/3/4/6；自然监督120、自然PL120、固定L/V监督120；四方法各自 Full/CE、三模型 seed、全部七视图，所有结果均为冻结后独立 truth-last。',
        '源 RX 链为 (1,3,4,6,8,11)，固定目标 RX 为 (0,2,5,7,9,10)，每视图144000个物理样本。m=5按用户要求排除；任何五源七目标历史结果均不插入六目标曲线。',
        '自然 L/U/V=1260m/11340m/5400m；固定 L=1080、V=5400、U=12600m−1080，仅监督。两条自然曲线同时改变源数量与数据量；固定监督曲线控制标签数和最大日程，IB早停仍使实际更新数变化。', '',
        '## 核验与总体观察','',f"360行与2520个视图单元闭合；复算 {matrices} 份混淆矩阵的 Accuracy/Macro-F1，核对每视图6个RX之和及144000样本。三seed均值±样本SD；六景先在同seed内平均。",'',
        f"clean/六景两指标共120个预定方法/条件/源数配对均值中，Full−自身CE非正的有{len(nonwins)}个；保留全部正负结果，不能声称DG机制普遍提升。普通PL共120条，有{len(zero_pl)}条训练期间接受数为零。",'',
        '## m1→m6 描述性端点','',
        '| 设置 | 方法 | 版本 | clean m1→m6（%） | 六景 m1→m6（%） |','|---|---|---|---:|---:|']
    lookup={(r['source_count'],r['budget'],r['regime'],r['family'],r['variant'],r['view'],r['metric']):r for r in summary}
    for budget,regime in REGIMES:
        for family in FAMILIES:
            for variant in ('ce','full'):
                def endpoint(view):
                    a=lookup[1,budget,regime,family,variant,view,'accuracy'];b=lookup[6,budget,regime,family,variant,view,'accuracy']
                    return f"{100*a['mean']:.3f}±{100*a['sample_sd']:.3f} → {100*b['mean']:.3f}±{100*b['sample_sd']:.3f}"
                lines.append(f"| {budget}/{regime} | {LABELS[family]} | {variant} | {endpoint('clean')} | {endpoint('satellite_mean')} |")
    lines+=['','全部m/场景/方法/预算/PL与同seed配对结果见CSV和图。训练成本按行报告实测epochs、batch/optimizer更新、曝光、PL接受率和源冻结epoch；有标签与无标签联合BN路径的变化不能解释为全部来自被接受伪标签。', '',
        '结论限定于一条预定RX链和已暴露工程基准；三个模型seed不构成三个RX选择或新的独立盲测，不把七视图当作七倍独立物理样本。结果不回流PAMF研发或任何健康任务。']
    (out/'sourcecount_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary,dg,pl,verify


def plot(summary,dg,pl,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors=dict(zip(FAMILIES,('#0072B2','#D55E00','#009E73','#CC79A7')))
    titles=dict(clean='Clean',satellite_mean='Six-scene mean',practical_high='High suburban',practical_mid='Mid suburban',practical_low_suburban='Low suburban',practical_high_urban='High urban',practical_mid_urban='Mid urban',practical_low_urban='Low urban')
    def panel(ax,records,budget,regime,view,kind):
        for family in FAMILIES:
            for variant in (('ce','full') if kind!='dg' else ('full',)):
                selected=[r for r in records if r.get('budget',budget)==budget and r.get('regime',regime)==regime and r['family']==family and r.get('variant',variant)==variant and r['view']==view and r['metric']=='accuracy']
                selected.sort(key=lambda r:r['source_count']);assert len(selected)==5
                ax.errorbar([r['source_count'] for r in selected],[100*r['mean'] for r in selected],yerr=[100*r['sample_sd'] for r in selected],color=colors[family],ls='--' if variant=='ce' else '-',marker='o' if variant=='ce' else 's',ms=3,lw=1,capsize=2,elinewidth=.65)
        if kind!='accuracy':ax.axhline(0,color='.3',lw=.6)
        ax.set_xticks(COUNTS);ax.set_xlabel('Source RX count');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.set_title(titles[view]);ax.set_ylabel('RX-equal accuracy (%)' if kind=='accuracy' else ('Full − own CE (pp)' if kind=='dg' else 'PL − supervised (pp)'))
    legend=[Line2D([],[],color=colors[f],ls='--' if v=='ce' else '-',marker='o' if v=='ce' else 's',ms=3,label=LABELS[f]+(' CE' if v=='ce' else ' Full')) for f in FAMILIES for v in ('ce','full')]
    def save(fig,name,handles=legend):
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005))
        fig.tight_layout(rect=(0,.095,1,.97));fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    for kind,records in (('accuracy',summary),('dg',dg)):
        fig,axes=plt.subplots(2,3,figsize=(10,5.6))
        for col,(budget,regime) in enumerate(REGIMES):
            for row,view in enumerate(('clean','satellite_mean')):
                panel(axes[row,col],records,budget,regime,view,kind)
                axes[row,col].set_title(f'{budget.capitalize()} / '+('PL' if regime=='pl' else 'supervised')+'\n'+titles[view])
        handles=legend if kind=='accuracy' else [Line2D([],[],color=colors[f],marker='s',label=LABELS[f]) for f in FAMILIES]
        save(fig,'sourcecount_'+kind+'_overview',handles)
        for budget,regime in REGIMES:
            fig,axes=plt.subplots(2,4,figsize=(12,5.5))
            for ax,view in zip(axes.flat,(*SCENES,'satellite_mean')):panel(ax,records,budget,regime,view,kind)
            fig.suptitle(f'{budget.capitalize()} / {regime}; three model seeds; m=5 excluded',fontsize=10)
            save(fig,'sourcecount_'+kind+'_'+budget+'_'+regime+'_all_scenes',handles)
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for ax,view in zip(axes,('clean','satellite_mean')):panel(ax,pl,'natural','pl',view,'pl')
    save(fig,'sourcecount_pl_gain_overview')
    fig,axes=plt.subplots(2,4,figsize=(12,5.5))
    for ax,view in zip(axes.flat,(*SCENES,'satellite_mean')):panel(ax,pl,'natural','pl',view,'pl')
    save(fig,'sourcecount_pl_gain_all_scenes')


def verify_audit(summary,dg,pl,path):
    remote=read(path);assert remote['status']=='ANALYZED' and remote['rows']==remote['expected_rows']==360 and remote['excluded_source_counts']==[5]
    lookup={(r['source_count'],r['budget'],r['regime'],r['family'],r['variant'],r['view'],r['metric']):r for r in summary}
    matched=0
    for r in remote['summary']:
        a,b,c=r['group'].split('-');m=int(a[1:])
        for metric,field in (('accuracy','equal_rx_accuracy'),('macro_f1','macro_f1'),('worst_rx','worst_rx')):
            cell=lookup[m,b,c,r['family'],r['version'],r['view'],metric]
            assert cell['actual_seeds']==r['actual_seeds']==3
            for stat in ('mean','sample_sd'):
                assert math.isclose(cell[stat],r[field+'_'+stat],abs_tol=1e-12);matched+=1
    lookup={(r['source_count'],r['budget'],r['regime'],r['family'],r['view']):r for r in dg if r['metric']=='accuracy'}
    for r in remote['dg_paired']:
        a,b,c=r['group'].split('-');cell=lookup[int(a[1:]),b,c,r['family'],r['view']]
        assert cell['actual_seeds']==r['actual_pairs']==3
        for stat,field in (('mean','equal_rx_mean'),('sample_sd','equal_rx_sample_sd')):
            assert math.isclose(cell[stat],r[field],abs_tol=1e-12);matched+=1
    lookup={(r['source_count'],r['family'],r['variant'],r['view']):r for r in pl if r['metric']=='accuracy'}
    for r in remote['pl_paired']:
        cell=lookup[r['source_count'],r['family'],r['variant'],r['view']]
        assert cell['actual_seeds']==r['actual_pairs']==3
        for stat in ('mean','sample_sd'):
            assert math.isclose(cell[stat],r[stat],abs_tol=1e-12);matched+=1
    assert matched==6440
    return matched


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--figures');p.add_argument('--audit-summary')
    a=p.parse_args();summary,dg,pl,verify=analyze(a.input,a.output)
    if a.audit_summary:verify['remote_summary_and_pair_statistics_matched']=verify_audit(summary,dg,pl,a.audit_summary)
    if a.figures:
        plot(summary,dg,pl,a.figures);verify['figures']=len(list(Path(a.figures).glob('*.pdf')))
    (Path(a.output)/'sourcecount_verification.json').write_text(json.dumps(verify,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verify))
