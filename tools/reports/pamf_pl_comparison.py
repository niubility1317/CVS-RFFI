"""Read frozen aggregates; publish the current PL comparison without retraining."""
import argparse,csv,gzip,json,math,statistics,shutil,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7')
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics'
EXT=ROOT/'local_artifacts/external_practical_residual_2x2_design_20261023/full_analysis_20261024'
OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r19'
PREVIOUS=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r18'
SEEDS=(2026092701,2026092702,2026092703)
PIDS=('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural')
METHODS=('riei','drift','ibrff','sdg','supervised','ordinary_PL')
NAMES=dict(zip(METHODS,('RIEI','DRIFT','IB-RFF','SDG','PAMF（监督训练）','PAMF（普通伪标签）')))
SHORT=dict(zip(PIDS,('标准接收机划分','接收机划分一','接收机划分二','大规模十类数据','1%标注','5%标注','1个源接收机','2个源接收机','3个源接收机','4个源接收机','6个源接收机')))
TASKS=('cross_date','cross_rx','cross_both');TN=dict(zip(TASKS,('跨日','跨接收机','跨日＋跨接收机')))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SN=dict(zip(SCENES,('clean','高郊区','中郊区','低郊区','高城市','中城市','低城市')))
MAIN_PANELS=(('riei','raw','RIEI'),('drift','raw','DRIFT'),('ibrff','raw','IB-RFF'),('sdg','raw','SDG'),('ordinary_PL','raw','PAMF'),('ordinary_PL','DURA-LU','DURA+PAMF'))

def allowed(r):
    return r['cell'] not in ('A','B') or (r['view']=='clean' and r['test'] in ('cross_rx','cross_both'))

def project():
    """Reuse verified aggregates; exclude no-channel practical rows from this report."""
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/'tables').mkdir(parents=True);(OUT/'figures').mkdir()
    for name in ('complete_summary.csv','per_seed_overall.csv.gz'):
        write(OUT/'tables'/name,(r for r in rows(PREVIOUS/'tables'/name) if allowed(r)))
    for name in ('paired_effects.csv','PAMF_actual_exposures.csv','external_training_and_exposures.csv','PAMF_main_unique_stage_cost.csv','PAMF_current_timing.csv','data_contracts.csv'):
        shutil.copy2(PREVIOUS/'tables'/name,OUT/'tables'/name)
    write(OUT/'tables/external_factorial.csv',(r for r in rows(PREVIOUS/'tables/external_factorial.csv') if r['effect']=='PL_with_channel' or (r['view']=='clean' and r['test'] in ('cross_rx','cross_both'))))
    summary=list(rows(OUT/'tables/complete_summary.csv'))
    seed=list(rows(OUT/'tables/per_seed_overall.csv.gz'))
    ix={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric'],int(r['seed'])):float(r['percent']) for r in seed}
    def val(p,m,t,rd,v,metric,s):return ix[p,m,'native' if m in ('supervised','ordinary_PL') else 'D',t,rd,v,metric,s]
    main=[]
    for r in summary:
        for m,rd,label in MAIN_PANELS:
            if r['method']==m and r['readout']==rd and r['cell']==('native' if m=='ordinary_PL' else 'D'):
                main.append(dict(r,display_method=label,experiment_section='main_comparison',channel_augmentation='on'))
    write(OUT/'tables/main_comparison.csv',main)
    gains=[];system=[]
    for p in PIDS:
        for t in TASKS:
            for v in ('clean','six_practical_mean'):
                for metric in ('accuracy','macro_f1'):
                    for m in METHODS[:4]:
                        gains.append(dict(protocol=p,method=m,display_method=NAMES[m],test=t,view=v,metric=metric,readout_before='raw',readout_after='DURA-LU',channel_augmentation='on',**moment([val(p,m,t,'DURA-LU',v,metric,s)-val(p,m,t,'raw',v,metric,s) for s in SEEDS])))
                        system.append(dict(protocol=p,reference_method=m,display_reference=NAMES[m],test=t,view=v,metric=metric,PAMF_readout='DURA-LU',reference_readout='raw',effect='DURA_PAMF_minus_direct_external',**moment([val(p,'ordinary_PL',t,'DURA-LU',v,metric,s)-val(p,m,t,'raw',v,metric,s) for s in SEEDS])))
    write(OUT/'tables/DURA_generality.csv',gains);write(OUT/'tables/main_system_effects.csv',system)
    groups=[]
    for p in PIDS:
        for m in METHODS:
            cells=('native',) if m in ('supervised','ordinary_PL') else ('A','B','C','D')
            for cell in cells:
                on=cell not in ('A','B')
                section='PAMF_supervision_control' if m=='supervised' else 'main_PAMF' if m=='ordinary_PL' else 'external_channel_on' if on else 'external_no_channel_CLEAN_only'
                for s in SEEDS:
                    groups.append(dict(protocol=p,display_protocol=SHORT[p],method=m,display_method=NAMES[m],cell=cell,seed=s,channel_augmentation='on' if on else 'off',PL='on' if m=='ordinary_PL' or cell in ('B','D') else 'off',report_views='clean;'+(';'.join(SCENES[1:])) if on else 'clean',report_populations='cross_date;cross_rx;cross_both' if on else 'cross_rx;cross_both',section=section,main_comparison=bool(m=='ordinary_PL' or cell=='D'),DURA_readout_section='generality_only' if m in METHODS[:4] else 'PAMF_system_and_supervision_control'))
    write(OUT/'tables/experiment_groups.csv',groups)
    v=json.loads((PREVIOUS/'verification.json').read_text(encoding='utf-8'))
    v.update(summary_rows=len(summary),per_seed_overall_rows=len(seed),logical_main_models=165,main_display_system_rows=198,PAMF_supervised_control_models=33,main_display_names=[p[2] for p in MAIN_PANELS],excluded_historical_no_channel_practical_summary_rows=7392,excluded_historical_no_channel_source_date_CLEAN_rows=352,reporting_policy='External methods direct by default; DURA external readouts only in generality; no-channel target CLEAN only',report_group_classifier_rows=len(groups))
    (OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(v,ensure_ascii=False),flush=True)
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
def load():
    summary=list(rows(OUT/'tables/complete_summary.csv'));ix={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in summary}
    def get(p,m,t='cross_rx',rd=None,v='six_practical_mean',metric='accuracy',cell=None):return ix[p,m,cell or ('native' if m in ('supervised','ordinary_PL') else 'D'),t,rd or ('DURA-L' if m=='supervised' else 'DURA-LU'),v,metric]
    return summary,get

def main_records(get,p,t='cross_rx',v='six_practical_mean',metric='accuracy'):
    return [dict(get(p,m,t,rd=rd,v=v,metric=metric),display_method=label) for m,rd,label in MAIN_PANELS]
def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
    _,get=load();labels=[p[2] for p in MAIN_PANELS];colors=['#52769C','#8B6B99','#C18D53','#639A92','#7893AA','#254C78'];manifest=[]
    def save(fig,name,caption,data):
        fig.tight_layout();fig.savefig(OUT/'figures'/(name+'.pdf'));fig.savefig(OUT/'figures'/(name+'.svg'));fig.savefig(OUT/'figures'/(name+'.png'),dpi=240);plt.close(fig);write(OUT/'tables'/(name+'.csv'),data);manifest.append(dict(name=name,caption=caption))
    for t in TASKS:
        rr=main_records(get,'F0',t);fig,ax=plt.subplots(figsize=(4.5,2.85));ax.barh(range(6),[float(r['mean']) for r in rr],xerr=[float(r['sd']) for r in rr],color=colors,height=.62,error_kw={'lw':.7,'capsize':1.5});ax.set(yticks=range(6),yticklabels=labels,xlim=(0,100),xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
        save(fig,'main_'+t,TN[t]+'主比较：仅DURA+PAMF接入恢复器，其余五项直接识别。全部采用有信道增强训练；误差线为三个模型种子的样本标准差。',rr)
    for slug,pids,xx,xl in [('sources',['m%d-natural'%n for n in (1,2,3,4,6)],[1,2,3,4,6],'Number of source receivers'),('labels',['label01','label05','F0'],[1,5,10],'Labeled fraction of training pool (%)')]:
        fig,ax=plt.subplots(figsize=(4.5,3.05));data=[]
        for (m,rd,l),c in zip(MAIN_PANELS,colors):
            rr=[dict(get(p,m,rd=rd),display_method=l) for p in pids];data+=rr;ax.errorbar(xx,[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],label=l,color=c,marker='o',ms=3,lw=1,capsize=1)
        ax.set(xticks=xx,xlabel=xl,ylabel='S6 accuracy (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(alpha=.14)
        save(fig,slug,'跨接收机六景结果；自然源数量仅1/2/3/4/6且固定六目标。标注率曲线使用1%、5%、10%，保留非单调结果。',data)
    fig,ax=plt.subplots(figsize=(4.5,3.1));data=[]
    for (m,rd,l),c in zip(MAIN_PANELS,colors):
        rr=[dict(get('F0',m,rd=rd,v=v),display_method=l) for v in SCENES];data+=rr;ax.errorbar(range(7),[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],label=l,color=c,marker='o',ms=3,lw=1,capsize=1)
    ax.set(xticks=range(7),xticklabels=['Clean','High S','Mid S','Low S','High U','Mid U','Low U'],ylabel='Accuracy (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.tick_params(axis='x',rotation=28);ax.grid(alpha=.14)
    save(fig,'seven_scenes','标准划分跨接收机主比较：仅DURA+PAMF的clean和六景均经过恢复器，其余方法直接识别。',data)
    for t in TASKS:
        fig,ax=plt.subplots(figsize=(4.5,2.8));data=[]
        for i,m in enumerate(METHODS[:4]):
            rr=[get('F0',m,t,rd=rd) for rd in ('raw','DURA-LU')];data+=rr
            ax.barh([i-.17,i+.17],[float(r['mean']) for r in rr],xerr=[float(r['sd']) for r in rr],color=['#A7B5C4','#254C78'],height=.30,error_kw={'lw':.7,'capsize':1.3},label=None)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#A7B5C4',label='Direct'),Patch(color='#254C78',label='With DURA')],loc='lower right',frameon=False)
        ax.set(yticks=range(4),yticklabels=labels[:4],xlim=(0,100),xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
        save(fig,'DURA_generality_'+t,TN[t]+'：DURA泛用性单独验证，同一外部识别器接入恢复器前后配对。仅含有信道增强、普通伪标签开启的固定训练设置。',data)
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('9 separated comparison figures written',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['project','figures']);x=a.parse_args();project() if x.phase=='project' else figures()
