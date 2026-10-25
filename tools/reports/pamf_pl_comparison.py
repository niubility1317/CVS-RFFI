"""Read frozen aggregates; publish the current PL comparison without retraining."""
import argparse,csv,gzip,json,math,statistics,shutil,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7')
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics'
EXT=ROOT/'local_artifacts/external_practical_residual_2x2_design_20261023/full_analysis_20261024'
OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r18'
SEEDS=(2026092701,2026092702,2026092703)
PIDS=('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural')
METHODS=('riei','drift','ibrff','sdg','supervised','ordinary_PL')
NAMES=dict(zip(METHODS,('RIEI','DRIFT','IB-RFF','SDG','PAMF（监督训练）','PAMF（普通伪标签）')))
SHORT=dict(zip(PIDS,('标准接收机划分','接收机划分一','接收机划分二','大规模十类数据','1%标注','5%标注','1个源接收机','2个源接收机','3个源接收机','4个源接收机','6个源接收机')))
TASKS=('cross_date','cross_rx','cross_both');TN=dict(zip(TASKS,('跨日','跨接收机','跨日＋跨接收机')))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SN=dict(zip(SCENES,('clean','高郊区','中郊区','低郊区','高城市','中城市','低城市')))
def rows(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def write(path,data):
    data=list(data);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if str(path).endswith('.gz') else open)(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def moment(a):
    assert len(a)==3 and all(math.isfinite(v) for v in a)
    return dict(mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(v>0 for v in a),seed_values=';'.join(map(str,a)))
def extract():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    own=[r for r in rows(FROZEN/'analysis/summary_overall.csv') if r['group']=='main' and r['method'] in ('supervised','ordinary_PL')]
    summary=[];seed=[]
    for r in own:summary.append(dict(protocol=r['protocol'],method=r['method'],display_method=NAMES[r['method']],cell='native',test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],mean=r['mean'],sd=r['sd']))
    for r in rows(FROZEN/'external/summary_overall.csv'):
        summary.append(dict(protocol=r['protocol'],method=r['family'],display_method=NAMES[r['family']],cell=r['cell'],test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],mean=r['mean'],sd=r['sd']))
    for r in rows(FROZEN/'analysis/per_seed.csv.gz'):
        if r['group']=='main' and r['method'] in ('supervised','ordinary_PL') and r['dimension']=='overall':
            seed.append(dict(protocol=r['protocol'],method=r['method'],cell='native',test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],seed=r['seed'],percent=r['percent'],packets_per_view=r['packets_per_view']))
    packet_counts={(r['protocol'],r['test']):r['packets_per_view'] for r in seed}
    for r in rows(FROZEN/'external/per_seed_overall.csv.gz'):
        seed.append(dict(protocol=r['protocol'],method=r['family'],cell=r['cell'],test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],seed=r['seed'],percent=r['percent'],packets_per_view=packet_counts[r['protocol'],r['test']]))
    assert len(summary)==20064 and len(seed)==60192
    write(OUT/'tables/complete_summary.csv',summary);write(OUT/'tables/per_seed_overall.csv.gz',seed)
    ix={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric'],int(r['seed'])):float(r['percent']) for r in seed}
    def value(p,m,t,rd,v,metric,s,cell=None):return ix[p,m,cell or ('native' if m in ('supervised','ordinary_PL') else 'D'),t,rd,v,metric,s]
    pair=[]
    for p in PIDS:
        for t in TASKS:
            for v in ('clean','six_practical_mean'):
                for metric in ('accuracy','macro_f1'):
                    for rd in ('raw','DURA-L','DURA-LU'):
                        a=[value(p,'ordinary_PL',t,rd,v,metric,s)-value(p,'supervised',t,rd,v,metric,s) for s in SEEDS]
                        pair.append(dict(protocol=p,test=t,view=v,metric=metric,readout=rd,effect='PL_minus_Sup_fixed_route',**moment(a)))
                    pair.append(dict(protocol=p,test=t,view=v,metric=metric,readout='PL_GLU_vs_Sup_GL',effect='complete_system_PL_minus_Sup',**moment([value(p,'ordinary_PL',t,'DURA-LU',v,metric,s)-value(p,'supervised',t,'DURA-L',v,metric,s) for s in SEEDS])))
                    for m in METHODS:
                        rd='DURA-L' if m=='supervised' else 'DURA-LU'
                        pair.append(dict(protocol=p,test=t,view=v,metric=metric,readout=rd,effect='restoration_gain_'+m,**moment([value(p,m,t,rd,v,metric,s)-value(p,m,t,'raw',v,metric,s) for s in SEEDS])))
                    for m in METHODS[:4]:
                        for rd in ('raw','DURA-LU'):
                            pair.append(dict(protocol=p,test=t,view=v,metric=metric,readout=rd,effect='PAMF_PL_minus_'+m+'_D',**moment([value(p,'ordinary_PL',t,rd,v,metric,s)-value(p,m,t,rd,v,metric,s) for s in SEEDS])))
    factorial=[]
    for m in METHODS[:4]:
        for p in PIDS:
            for t in TASKS:
                for route in ('raw','matching_G'):
                    for v in ('clean','six_practical_mean'):
                        def c(cell,s):return value(p,m,t,('raw' if route=='raw' else 'DURA-LU' if cell in ('B','D') else 'DURA-L'),v,'accuracy',s,cell)
                        for effect,coeff in [('channel_no_PL',{'C':1,'A':-1}),('channel_with_PL',{'D':1,'B':-1}),('PL_no_channel',{'B':1,'A':-1}),('PL_with_channel',{'D':1,'C':-1}),('interaction',{'D':1,'B':-1,'C':-1,'A':1})]:
                            factorial.append(dict(method=m,display_method=NAMES[m],protocol=p,test=t,route=route,view=v,effect=effect,interpretation='F_and_G_U_permissions_change' if route!='raw' and ('PL_' in effect or effect=='interaction') else 'fixed_G_role_or_raw',**moment([sum(k*c(cell,s) for cell,k in coeff.items()) for s in SEEDS])))
    write(OUT/'tables/paired_effects.csv',pair);write(OUT/'tables/external_factorial.csv',factorial)
    exposure=[r for r in rows(FROZEN/'analysis/training_exposures.csv') if r['method'] in ('supervised','ordinary_PL')]
    assert len(exposure)==66;write(OUT/'tables/PAMF_actual_exposures.csv',exposure)
    ec=[r for r in rows(EXT/'F_cost.csv') if r['family'] in METHODS[:4]];assert len(ec)==528;write(OUT/'tables/external_training_and_exposures.csv',ec)
    stages={};use=defaultdict(set)
    for item in records(FROZEN/'inputs/source_inputs.jsonl.gz'):
        active='-ordinary_PL-' in item['row_id'] or '-supervised-' in item['row_id']
        for s in item['complete']['segments']:
            use[s['folder']].add('current_main' if active else 'historical_other_panel')
            stages[s['folder']]=dict(folder=s['folder'],new_in_completion_run=s['config']['run_id']=='20261024-phase1-PAMF-REPL-completion-m201-r01',epochs=s['epoch'],updates=s['updates'],seconds=s['seconds'])
    selected=[dict(r,scope=';'.join(sorted(use[k]))) for k,r in stages.items() if 'current_main' in use[k]]
    write(OUT/'tables/PAMF_main_unique_stage_cost.csv',selected)
    timing=[r for r in rows(FROZEN/'analysis/timing_recomputed.csv') if r['panel']=='main_controls'];assert len(timing)==20
    write(OUT/'tables/PAMF_current_timing.csv',timing)
    matrix=json.loads((FROZEN/'inputs/matrix.json').read_text());protocol=[]
    for p in PIDS:
        d=matrix['protocol_definitions'][p]
        counts={t:next(r['packets_per_view'] for r in seed if r['protocol']==p and r['method']=='ordinary_PL' and r['test']==t) for t in TASKS}
        protocol.append(dict(protocol=p,display=SHORT[p],classes=10 if p=='ManyRx' else 6,source_Rx=';'.join(map(str,d['source_rxs'])),target_Rx=';'.join(map(str,d['target_rxs'])),source_days='1;2;3',**d['expected_counts'],**counts))
    write(OUT/'tables/data_contracts.csv',protocol)
    wins={t:{v:{rd:sum(next(r['mean'] for r in pair if r['protocol']==p and r['test']==t and r['view']==v and r['metric']=='accuracy' and r['effect']=='PL_minus_Sup_fixed_route' and r['readout']==rd)>0 for p in PIDS) for rd in ('raw','DURA-L','DURA-LU')} for v in ('clean','six_practical_mean')} for t in TASKS}
    verification=dict(status='VERIFIED_FROZEN_AGGREGATE_PROJECTION',logical_main_models=198,external_D_models=132,PAMF_main_models=66,external_all_cells_models=528,total_with_all_external_cells=594,summary_rows=len(summary),per_seed_overall_rows=len(seed),methods=list(NAMES.values()),new_training=0,new_prediction=0,new_truth_scoring=0,DURA_new_training=0,source_scope='previously frozen and independently scored aggregates; no raw data or target truth reopened',model_seeds=list(SEEDS),labelled_methods_retired=['PAMF-REPL','PAMF-T1'],current_main_unique_epochs=sum(r['epochs'] for r in selected),current_main_new_unique_epochs=sum(r['epochs'] for r in selected if r['new_in_completion_run']),current_main_new_unique_updates=sum(r['updates'] for r in selected if r['new_in_completion_run']),current_main_new_shared_worker_hours=sum(r['seconds'] for r in selected if r['new_in_completion_run'])/3600,PL_positive_contracts_out_of_11=wins,external_zero_PL_rows=sum(int(r['pl_seen'])>0 and int(r['pl_accepted'])==0 for r in ec))
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False),flush=True)
def load():
    summary=list(rows(OUT/'tables/complete_summary.csv'));ix={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in summary}
    def get(p,m,t='cross_rx',rd=None,v='six_practical_mean',metric='accuracy',cell=None):return ix[p,m,cell or ('native' if m in ('supervised','ordinary_PL') else 'D'),t,rd or ('DURA-L' if m=='supervised' else 'DURA-LU'),v,metric]
    return summary,get
def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
    _,get=load();labels=['RIEI','DRIFT','IB-RFF','SDG','PAMF supervised','PAMF pseudo-label'];colors=['#52769C','#8B6B99','#C18D53','#639A92','#A6B9CC','#254C78'];manifest=[]
    def save(fig,name,caption,data):
        fig.tight_layout();fig.savefig(OUT/'figures'/(name+'.pdf'));fig.savefig(OUT/'figures'/(name+'.svg'));fig.savefig(OUT/'figures'/(name+'.png'),dpi=240);plt.close(fig);write(OUT/'tables'/(name+'.csv'),data);manifest.append(dict(name=name,caption=caption))
    for t in TASKS:
        for rd in ('raw',None):
            rr=[get('F0',m,t,rd=rd) for m in METHODS];fig,ax=plt.subplots(figsize=(4.5,2.85));ax.barh(range(6),[float(r['mean']) for r in rr],xerr=[float(r['sd']) for r in rr],color=colors,height=.62,error_kw={'lw':.7,'capsize':1.5});ax.set(yticks=range(6),yticklabels=labels,xlim=(0,100),xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
            save(fig,'F0_'+t+('_raw' if rd else '_restored'),TN[t]+'：六种信道场景平均准确率。外部固定为信道增强与普通伪标签均启用的完整训练；误差线为三次识别器训练的样本标准差。',rr)
    for slug,pids,xx,xl in [('sources',['m%d-natural'%n for n in (1,2,3,4,6)],[1,2,3,4,6],'Number of source receivers'),('labels',['label01','label05','F0'],[1,5,10],'Labeled fraction of training pool (%)')]:
        fig,ax=plt.subplots(figsize=(4.5,3.05));data=[]
        for m,l,c in zip(METHODS,labels,colors):
            rr=[get(p,m) for p in pids];data+=rr;ax.errorbar(xx,[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],label=l,color=c,marker='o',ms=3,lw=1,capsize=1)
        ax.set(xticks=xx,xlabel=xl,ylabel='S6 accuracy (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(alpha=.14)
        save(fig,slug,'跨接收机六景结果；自然源数量仅1/2/3/4/6且固定六目标。标注率曲线使用1%、5%、10%，保留非单调结果。',data)
    fig,ax=plt.subplots(figsize=(4.5,3.1));data=[]
    for m,l,c in zip(METHODS,labels,colors):
        rr=[get('F0',m,v=v) for v in SCENES];data+=rr;ax.errorbar(range(7),[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],label=l,color=c,marker='o',ms=3,lw=1,capsize=1)
    ax.set(xticks=range(7),xticklabels=['Clean','High S','Mid S','Low S','High U','Mid U','Low U'],ylabel='Accuracy (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.tick_params(axis='x',rotation=28);ax.grid(alpha=.14)
    save(fig,'seven_scenes','F0跨接收机逐景准确率；clean也经过恢复器，无场景旁路。',data)
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('9 comparison figures written',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['extract','figures']);x=a.parse_args();extract() if x.phase=='extract' else figures()
