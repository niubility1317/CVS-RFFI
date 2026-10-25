"""Strict classification-only supervised branch report from frozen records."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7');OUT=ROOT/'exports/reports/20261025-PAMF-branch-ablation-r20'
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics'
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('full','SIG_TIME','NO_TIME','SHUFFLE_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE','COV_X3')
NAMES=dict(zip(ARMS,('完整网络','单视图路径描述','去除显式时间坐标','固定打乱局部顺序','去除路径描述支路','去除功放相关支路','去除参考响应支路','打乱参考模板相位','三视图矩统计替代')))
EN=dict(zip(ARMS,('Full network','One path view','No explicit time','Fixed order shuffle','No path descriptor','No PA features','No reference response','Phase-scrambled reference','Moment descriptors')))
PANELS=('supervised_classification',);PN={'supervised_classification':'仅分类损失的监督训练'}
TASKS=('cross_date','cross_rx','cross_both');TN=dict(zip(TASKS,('仅日期变化','仅接收机变化','日期与接收机同时变化')))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SN=dict(zip(SCENES,('clean','高郊区','中郊区','低郊区','高城市','中城市','低城市')))
def method(panel,arm):return 'CE_'+arm
def rows(path):
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def write(path,data):
    it=iter(data);first=next(it);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if path.suffix=='.gz' else open)(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(it)
def moments(a):return dict(mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(x>0 for x in a),seed_values=';'.join(map(str,a)))
def pct(a,q):
    a=sorted(a);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def extract():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    mapping={method(p,a):(p,a) for p in PANELS for a in ARMS};keep=set(mapping)
    seed=[];counts=0
    def filtered_seed():
        nonlocal counts
        for r in rows(FROZEN/'analysis/per_seed.csv.gz'):
            if r['protocol']=='F0' and r['method'] in keep and r['readout'] in ('raw','DURA-L'):
                counts+=1
                if r['dimension']=='overall':seed.append(r)
                yield r
    write(OUT/'tables/all_strata_per_seed.csv.gz',filtered_seed())
    summary=[r for r in rows(FROZEN/'analysis/summary_overall.csv') if r['protocol']=='F0' and r['method'] in keep and r['readout'] in ('raw','DURA-L')]
    write(OUT/'tables/summary_overall.csv',summary);write(OUT/'tables/per_seed_overall.csv',seed)
    ix={(r['method'],r['test'],r['readout'],r['view'],r['metric'],int(r['seed'])):float(r['percent']) for r in seed};paired=[];interaction=[]
    for p in PANELS:
        for a in ARMS[1:]:
            for t in TASKS:
                for rd in ('raw','DURA-L'):
                    for view in (*SCENES,'six_practical_mean'):
                        for metric in ('accuracy','macro_f1'):
                            dif=[ix[method(p,'full'),t,rd,view,metric,s]-ix[method(p,a),t,rd,view,metric,s] for s in SEEDS]
                            paired.append(dict(panel=p,display_panel=PN[p],arm=a,display_arm=NAMES[a],test=t,readout=rd,view=view,metric=metric,**moments(dif)))
                rd='DURA-L'
                for metric in ('accuracy','macro_f1'):
                    dif=[(ix[method(p,'full'),t,rd,'six_practical_mean',metric,s]-ix[method(p,a),t,rd,'six_practical_mean',metric,s])-(ix[method(p,'full'),t,'raw','six_practical_mean',metric,s]-ix[method(p,a),t,'raw','six_practical_mean',metric,s]) for s in SEEDS]
                    interaction.append(dict(panel=p,arm=a,display_arm=NAMES[a],test=t,metric=metric,**moments(dif)))
    write(OUT/'tables/full_minus_variant.csv',paired);write(OUT/'tables/restoration_by_branch_interaction.csv',interaction)
    timing=[];parameters=[];sample_total=0
    (OUT/'inputs').mkdir(exist_ok=True)
    for item in records(FROZEN/'inputs/timing_inputs.jsonl.gz'):
        if item['panel']!='strict_structure':continue
        panel=PANELS[0];pr=item['resolved']['parameters'];groups=defaultdict(list)
        with gzip.open(OUT/'inputs/timing_samples.jsonl.gz','wt',encoding='utf-8') as f:
            for sample in item['samples']:f.write(json.dumps(sample)+'\n')
        metadata={k:item['resolved'][k] for k in ('device','FP32','TF32','source_IQ_only','exclusive_GPU','model_memory_note')}
        (OUT/'inputs/timing_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        for arm in ARMS:
            n=pr[method(panel,arm)];parameters.append(dict(panel=panel,arm=arm,display_arm=NAMES[arm],loaded_parameters=n,removed_parameters=pr[method(panel,'full')]-n,reduction_percent=100*(pr[method(panel,'full')]-n)/pr[method(panel,'full')]))
        for s in item['samples']:groups[s['case'],int(s['batch'])].append(s);sample_total+=1
        for (case,batch),ss in groups.items():
            assert {s['round'] for s in ss}==set(range(5));route='restored' if case.endswith('_GF') else 'direct';m=case[:-3] if route=='restored' else case[:-2];arm=mapping[m][1]
            wall=[v for s in ss for v in s['wall_ms']];transfer=[v for s in ss for v in s['H2D_D2H_ms']]
            timing.append(dict(panel=panel,arm=arm,display_arm=NAMES[arm],route=route,batch=batch,wall_median_ms=statistics.median(wall),wall_P95_ms=pct(wall,.95),round_median_sd=statistics.stdev(statistics.median(s['wall_ms']) for s in ss),CPU_roundtrip_median_ms=statistics.median(transfer),throughput_IQ_s=batch*1000/statistics.median(wall),resident_baseline_bytes=max(s['resident_baseline_bytes'] for s in ss),incremental_peak_bytes=max(s['incremental_peak_bytes'] for s in ss),rounds=5,device=item['resolved']['device']))
    assert sample_total==450;write(OUT/'tables/timing.csv',timing);write(OUT/'tables/parameters.csv',parameters)
    costs=[];stages=[]
    for item in records(FROZEN/'inputs/source_inputs.jsonl.gz'):
        m=item['complete']['config'].get('repl_row',{}).get('method')
        # Exact reused full reader keeps its original historical config.
        rid=item['row_id'];m=rid.removeprefix('F0-').rsplit('-s',1)[0] if rid.startswith('F0-') else m
        if m not in mapping:continue
        p,a=mapping[m];done=item['complete'];assert done['epoch']==50 and done['updates']==11100
        assert done['target_access'] is False and done['U_truth_read'] is False and done['head_fit_updates']==0
        ss=done['segments'];costs.append(dict(panel=p,arm=a,display_arm=NAMES[a],seed=done['config']['model_seed'],epochs=50,updates=11100,reused=item['reused'],cumulative_shared_worker_hours=sum(x['seconds'] for x in ss)/3600,new_in_completion_worker_hours=sum(x['seconds'] for x in ss if x['config']['run_id']=='20261024-phase1-PAMF-REPL-completion-m201-r01')/3600))
        stages.extend(dict(panel=p,arm=a,seed=done['config']['model_seed'],folder=x['folder'],epochs=x['epoch'],updates=x['updates'],seconds=x['seconds']) for x in ss)
    assert len(costs)==27;write(OUT/'tables/source_cost.csv',costs);write(OUT/'tables/source_stage_cost.csv',stages)
    exposure_fields=('row_id','protocol','method','seed','U_enabled','U_seen','U_accepted','L_packet_forward_evaluations','source_actual_EMA_updates','U_truth_read')
    write(OUT/'tables/training_exposures.csv',({k:r[k] for k in exposure_fields} for r in rows(FROZEN/'analysis/training_exposures.csv') if r['method'] in mapping and r['protocol']=='F0'))
    write(OUT/'tables/source_geometry.csv',(r for r in rows(FROZEN/'analysis/source_geometry.csv') if r['method'] in keep))
    write(OUT/'tables/source_interventions.csv',(r for r in rows(FROZEN/'analysis/source_interventions.csv') if r['method'] in keep))
    write(OUT/'tables/net_corrections.csv.gz',(r for r in rows(FROZEN/'analysis/net_corrections.csv.gz') if r['protocol']=='F0' and r['method'] in keep and r['readout']=='DURA-L'))
    epoch_rows=[]
    for item in records(FROZEN/'analysis/F_epoch_inputs.jsonl.gz'):
        folder=item['folder'];marker='/F/F0-'
        if marker not in folder:continue
        m=folder.split(marker,1)[1].split('/')[0].rsplit('-s',1)[0]
        if m not in keep:continue
        epoch=item['epoch'];offset={'segment20':0,'segment30':20,'segment50':30}[folder.rsplit('/',1)[-1]]
        assert not epoch['use_unlabeled'] and not epoch['use_ema_teacher'] and epoch['label_smoothing']==0
        assert epoch['train_T1_CE_control']==1 and epoch['train_T1_L_packets']==384
        epoch_rows.append(dict(method=m,seed=folder.split(marker)[1].split('/')[0].rsplit('-s',1)[1],logical_epoch=offset+epoch['epoch'],epoch_seconds=epoch['epoch_time_s'],lr=epoch['lr'],U_enabled=epoch['use_unlabeled'],EMA_enabled=epoch['use_ema_teacher'],label_smoothing=epoch['label_smoothing'],classification_only=epoch['train_T1_CE_control'],L_forward_evaluations_per_update=epoch['train_T1_L_packets']))
    assert len(epoch_rows)==1350
    write(OUT/'tables/training_epochs.csv',epoch_rows)
    metadata=dict(training_run='20261024-phase1-PAMF-REPL-completion-m201-r01',training_commit='dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f',score_entry_commit='a79bdf385489cb3822b37434a07e3c43e5ce839a',protocol='F0',source_Rx=[1,3,4,6,8],target_Rx=[0,2,5,7,9,10,11],source_days=[1,2,3],unseen_day=0,classes=6,L_packets=6300,unused_U_packets=56700,source_validation_packets=27000,test_packet_counts={'cross_date':30000,'cross_rx':126000,'cross_both':42000},scenarios=list(SCENES),seeds=list(SEEDS),epoch_segments=[20,10,20],updates=11100,loss='CE(first ordinary logits, y) + 0.68 * CE(auxiliary logits, y)',L_batch=128,L_forward_evaluations_per_update=384,auxiliary_slot_exposures_per_model=1420800,loss_logit_packet_evaluations_per_update=256,U_enabled=False,EMA_enabled=False,paired_KL=False,contrast_loss=False,label_smoothing=0,default_head_joint=True,augmentation={'implementation':'ResidualChannel residual/post_sync/noEQ','scenes':{'low_suburban':0.40,'mid_urban':0.35,'low_urban':0.25},'progress_boundaries':[0.20,0.45],'application_probabilities':[0.30,0.60,0.80],'probabilities_are_not_realized_counts':True},restoration={'name':'DURA','source_IQ':'labeled-source collection only; TX labels unused','parameters':266951,'existing_checkpoint':True,'retrained_in_this_report':False,'shared_by_all_model_seeds':True},source_geometry={'packets':720,'TX_RX_day_strata':90,'no_fitted_probe':True},evidence_policy='Existing frozen scores; this report does not reopen IQ, checkpoints or sample-level target truth.')
    (OUT/'inputs/experiment_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(status='FROZEN_SUPERVISED_BRANCH_REPORT_INPUTS_READY',strict_supervised_models=27,network_structures=9,model_seeds=list(SEEDS),readouts=['raw','DURA-L'],summary_rows=len(summary),overall_seed_rows=len(seed),all_strata_seed_rows=counts,paired_rows=len(paired),timing_case_rounds=sample_total,new_training=0,new_predictions=0,new_truth_scoring=0,all_default_heads_joint=True,full_minus_variant_positive_favors_full=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)

def details():
    groups=defaultdict(dict)
    fields=('method','test','readout','view','dimension','stratum','metric')
    for r in rows(OUT/'tables/all_strata_per_seed.csv.gz'):
        key=tuple(r[k] for k in fields);s=int(r['seed'])
        assert s not in groups[key];groups[key][s]=float(r['percent'])
    summaries=[];pairs=[]
    for key,values in groups.items():
        assert set(values)==set(SEEDS)
        a=[values[s] for s in SEEDS]
        summaries.append(dict(zip(fields,key),mean=statistics.mean(a),sd=statistics.stdev(a)))
        if key[0]!='CE_full':
            base=groups[('CE_full',*key[1:])]
            pairs.append(dict(zip(fields,key),**moments([base[s]-values[s] for s in SEEDS])))
    write(OUT/'tables/per_stratum_summary.csv.gz',summaries)
    write(OUT/'tables/per_stratum_full_minus_variant.csv.gz',pairs)
    g=defaultdict(list)
    for r in rows(OUT/'tables/net_corrections.csv.gz'):
        if r['dimension']=='overall':g[r['method'],r['test'],r['view']].append(r)
    net=[]
    for (m,t,v),rr in g.items():
        assert {int(r['seed']) for r in rr}==set(SEEDS)
        d=dict(method=m,test=t,view=v,count=int(rr[0]['count']))
        for k in ('repairs','damage','net'):
            a=[int(r[k]) for r in rr];d[k+'_mean']=statistics.mean(a);d[k+'_sd']=statistics.stdev(a)
        net.append(d)
    write(OUT/'tables/net_corrections_overall.csv',net)
    print(json.dumps(dict(stratum_summary_rows=len(summaries),stratum_paired_rows=len(pairs),net_rows=len(net))),flush=True)
def load():
    ix={(r['method'],r['test'],r['readout'],r['view'],r['metric']):r for r in rows(OUT/'tables/summary_overall.csv')}
    def get(panel,arm,t='cross_rx',rd=None,view='six_practical_mean',metric='accuracy'):
        return ix[method(panel,arm),t,rd or 'DURA-L',view,metric]
    return ix,get
def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
    _,get=load();pair=list(rows(OUT/'tables/full_minus_variant.csv'));timing=list(rows(OUT/'tables/timing.csv'));manifest=[]
    def save(fig,name,caption,data,data_name=None):
        fig.tight_layout();fig.savefig(OUT/'figures'/(name+'.pdf'));fig.savefig(OUT/'figures'/(name+'.svg'));fig.savefig(OUT/'figures'/(name+'.png'),dpi=240);plt.close(fig);write(OUT/'tables'/((data_name or name)+'.csv'),data);manifest.append(dict(name=name,caption=caption,csv='tables/'+(data_name or name)+'.csv'))
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(6.6,3.1));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    boxes=[(0.1,2.0,1.2,0.75,'Received IQ'),(1.8,2.0,1.65,.75,'Shared encoder'),(4.0,3.8,2.35,.8,'PA-related features'),(4.0,2.0,2.35,.8,'Three path descriptors'),(4.0,.2,2.35,.8,'Public reference response'),(7.05,2.0,1.25,.8,'160-d fusion'),(8.7,2.0,1.2,.8,'Existing head')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',linewidth=.7,edgecolor='#52769C',facecolor='#EFF5F7'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.2,wrap=True)
    for start,end in [((1.3,2.375),(1.72,2.375)),((3.45,2.375),(3.92,4.2)),((3.45,2.375),(3.92,2.4)),((1.,2.),(3.92,.6)),((6.35,4.2),(6.97,2.65)),((6.35,2.4),(6.97,2.4)),((6.35,.6),(6.97,2.15)),((8.3,2.4),(8.62,2.4))]:ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=9,color='#52769C',lw=.7))
    ax.text(5.15,3.15,'3 projections × 4 windows\nShared feature sequence',ha='center',va='center',fontsize=7,color='#425566')
    ax.text(5.,4.88,'Shared encoder features also enter the existing identity representation',ha='center',fontsize=7,color='#425566')
    save(fig,'network_branches','结构关系示意。共享骨干的原身份特征仍保留；功放相关、路径描述及参考响应信息在既有表示中融合。箭头表示依赖关系，不表示各路等权相加。',[dict(component=label) for x,y,w,h,label in boxes])
    for p,name in [(PANELS[0],'supervised_results')]:
        fig,ax=plt.subplots(figsize=(5,3.6));data=[]
        for off,(rd,view,label,color) in zip((-.2,0,.2),(('raw','clean','Direct clean','#8B6B99'),('raw','six_practical_mean','Direct channels','#52769C'),(None,'six_practical_mean','Restored channels','#639A92'))):
            rr=[get(p,a,rd=rd,view=view) for a in ARMS];data+=rr;ax.errorbar([float(r['mean']) for r in rr],np.arange(9)+off,xerr=[float(r['sd']) for r in rr],fmt='o',ms=3,capsize=1.3,color=color,label=label)
        ax.set(yticks=range(9),yticklabels=[EN[a] for a in ARMS],xlabel='Accuracy (%)');ax.invert_yaxis();ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(axis='x',alpha=.15)
        save(fig,name,'各网络变体在直接clean、直接六景及匹配恢复后六景下的准确率；三次识别器训练均值与样本标准差。',data)
    fig,ax=plt.subplots(figsize=(5,3.4));data=[]
    for off,(rd,view,label,color) in zip((-.2,0,.2),(('raw','clean','Direct clean','#8B6B99'),('raw','six_practical_mean','Direct channels','#52769C'),('DURA-L','six_practical_mean','Restored channels','#639A92'))):
        rr=[next(r for r in pair if r['panel']==PANELS[0] and r['arm']==a and r['test']=='cross_rx' and r['readout']==rd and r['view']==view and r['metric']=='accuracy') for a in ARMS[1:]];data+=rr;ax.errorbar([float(r['mean']) for r in rr],np.arange(8)+off,xerr=[float(r['sd']) for r in rr],fmt='o',ms=3,capsize=1.3,color=color,label=label)
    ax.set(yticks=range(8),yticklabels=[EN[a] for a in ARMS[1:]],xlabel='Full minus variant (percentage points)');ax.invert_yaxis();ax.axvline(0,color='.5',lw=.7);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(axis='x',alpha=.15)
    save(fig,'supervised_paired','监督结构层的完整网络减变体，先在同一次训练内配对；负值表示变体更好，不能删去。',data)
    for rd,name in [('raw','scene_effects_direct'),('DURA-L','scene_effects_restored')]:
        rr=[[next(r for r in pair if r['panel']==PANELS[0] and r['arm']==a and r['test']=='cross_rx' and r['readout']==rd and r['view']==v and r['metric']=='accuracy') for v in SCENES] for a in ARMS[1:]];z=np.array([[float(r['mean']) for r in a] for a in rr]);lim=max(abs(z.min()),abs(z.max()));fig,ax=plt.subplots(figsize=(5.4,3.5));im=ax.imshow(z,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set(xticks=range(7),xticklabels=['Clean','High S','Mid S','Low S','High U','Mid U','Low U'],yticks=range(8),yticklabels=[EN[a] for a in ARMS[1:]]);ax.tick_params(axis='x',rotation=30)
        for i in range(8):
            for j in range(7):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=6.2,color='white' if abs(z[i,j])>.6*lim else 'black')
        fig.colorbar(im,ax=ax,label='Full minus variant (pp)');save(fig,name,'监督结构层逐场景条件差；颜色和数字均为完整减变体，不表示独立可加的因果贡献。',[r for a in rr for r in a])
    fig,ax=plt.subplots(figsize=(5,3.5));data=[]
    for off,(route,label,color) in zip((-.1,.1),(('direct','Direct eager','#8B6B99'),('restored','Restored graph','#52769C'))):
        rr=[next(r for r in timing if r['panel']==PANELS[0] and r['arm']==a and r['route']==route and r['batch']=='1') for a in ARMS];data+=rr;ax.errorbar([float(r['wall_median_ms']) for r in rr],np.arange(9)+off,xerr=[float(r['round_median_sd']) for r in rr],fmt='o',ms=3,capsize=1.5,label=label,color=color)
    ax.set(yticks=range(9),yticklabels=[EN[a] for a in ARMS],xlabel='Single-packet latency (ms)');ax.invert_yaxis();ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);ax.grid(axis='x',alpha=.15);save(fig,'supervised_latency','同一路径内比较结构延迟；误差线是五轮中位数的标准差，不是三次训练的标准差。',data)
    geom=list(rows(OUT/'tables/source_geometry.csv'));fig,ax=plt.subplots(figsize=(5,3.3));data=[]
    for off,(field,label,color) in zip((-.13,.13),(('same_TX_cross_RX_distance','Same identity, different receiver','#52769C'),('different_TX_same_RX_distance','Different identity, same receiver','#639A92'))):
        means=[];sds=[]
        for a in ARMS:
            rr=[r for r in geom if r['method']==method(PANELS[0],a)];vals=[float(r[field]) for r in rr];means.append(statistics.mean(vals));sds.append(statistics.stdev(vals));data.append(dict(method=method(PANELS[0],a),series=field,mean=statistics.mean(vals),sd=statistics.stdev(vals),seed_values=';'.join(map(str,vals))))
        ax.errorbar(np.arange(9)+off,means,yerr=sds,fmt='o',ms=3,capsize=1.3,label=label,color=color)
    ax.set(xticks=range(9),xticklabels=[EN[a] for a in ARMS],ylabel='Source feature cosine distance');ax.tick_params(axis='x',rotation=55);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=1,frameon=False);ax.grid(alpha=.15);save(fig,'source_geometry','相同720条有标签源数据、90个发射机/接收机/日期单元；仅为冻结表征的几何描述，不是目标域识别贡献证明。',data,data_name='source_geometry_figure')
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(figures=len(manifest))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','details','figures']);a=p.parse_args();{'extract':extract,'details':details,'figures':figures}[a.phase]()
