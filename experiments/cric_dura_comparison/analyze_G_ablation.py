"""Aggregate frozen G-student scores; no IQ, truth, checkpoints or fitting."""
import argparse,csv,json,math,statistics
from pathlib import Path
from .io import read

SEEDS=(2026092701,2026092702,2026092703)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ARMS=('coarse_only','prior_no_CG','no_image_action','no_innovation_input','copy_no_KD','random_with_KD','random_no_KD')
LABELS=dict(raw='Raw T1',default='Default DURA (one fixed G)',coarse_only='Coarse only',prior_no_CG='No CG',no_image_action='No image action',no_innovation_input='No residual input',copy_no_KD='Copy, no IQ distillation',random_with_KD='Random + IQ distillation',random_no_KD='Random, no IQ distillation')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def checked_scores(scores):
    assert scores['status']=='SCORED' and scores['truth_last'] and scores['target_feedback_forbidden']
    actual={};ncm=0
    for r in scores['results']:
        cm=r['confusion'];assert len(cm)==6 and all(len(v)==6 for v in cm)
        assert all(isinstance(v,int) and v>=0 for row in cm for v in row)
        count=sum(map(sum,cm));assert count==r['count'] and count>0
        f1=[]
        for i in range(6):
            den=sum(cm[i])+sum(row[i] for row in cm);f1.append(2*cm[i][i]/den if den else 0.)
        assert math.isclose(sum(cm[i][i] for i in range(6))/count,r['accuracy'],abs_tol=1e-12)
        assert math.isclose(statistics.mean(f1),r['macro_f1'],abs_tol=1e-12)
        key=(r['readout'],r['view'],r['dimension'],r['stratum']);assert key not in actual
        actual[key]=r;ncm+=1
    for pair in scores['paired_raw']:
        suffix=(pair['view'],pair['dimension'],pair['stratum'])
        a=actual[('raw',*suffix)];b=actual[(pair['readout'],*suffix)]
        assert pair['net']==pair['repairs']-pair['damage'] and pair['count']==a['count']==b['count']
        assert math.isclose(pair['net']/pair['count'],b['accuracy']-a['accuracy'],abs_tol=1e-12)
    return actual,ncm,len(scores['paired_raw'])


def analyze(folder,baseline,output,paper=None):
    folder,baseline,output=map(Path,(folder,baseline,output));output.mkdir(parents=True,exist_ok=True)
    rows=[];cost=[];nets=[];base={};per={};parameters={'raw':0,'default':266951};ncm=nnet=0
    def add(arm,seed,scores,actual,readout):
        for view in VIEWS:
            r=actual[readout,view,'overall','ALL'];assert r['count']==168000 and r['test']=='main'
            rx=[v for (rd,vi,dim,_),v in actual.items() if (rd,vi,dim)==(readout,view,'receiver')]
            assert len(rx)==7 and all(v['count']==24000 for v in rx)
            assert [[sum(v['confusion'][i][j] for v in rx) for j in range(6)] for i in range(6)]==r['confusion']
            values=dict(accuracy=100*r['accuracy'],macro_f1=100*r['macro_f1'],worst_rx=100*min(v['accuracy'] for v in rx))
            rows.append(dict(arm=arm,seed=seed,view=view,count=r['count'],**values));per[arm,seed,view]=values
        if arm!='raw':
            for r in scores['paired_raw']:
                if r['readout']==readout and r['dimension']=='overall':nets.append(dict(arm=arm,seed=seed,view=r['view'],count=r['count'],repairs=r['repairs'],damage=r['damage'],net=r['net'],delta_pp=100*r['net']/r['count']))
    for seed in SEEDS:
        record=read(baseline/(str(seed)+'.json'));s=record['scores'];actual,a,b=checked_scores(s);ncm+=a;nnet+=b
        assert s['model_seed']==seed and record['complete']['status']=='SCORED_COMPLETE'
        assert set(r['readout'] for r in s['results'])=={'raw','DURA-LU'}
        base[seed]=actual;add('raw',seed,s,actual,'raw');add('default',seed,s,actual,'DURA-LU')
    contract=None
    for seed in SEEDS:
        for arm in ARMS:
            record=read(folder/(arm+'-s%d.json'%seed));s=record['scores'];actual,a,b=checked_scores(s);ncm+=a;nnet+=b
            assert record['seed']==s['model_seed']==seed and record['arm']==record['source']['arm']==arm
            assert set(r['readout'] for r in s['results'])=={'raw',arm}
            assert record['complete']['status']=='SCORED_COMPLETE' and record['complete']['views']==list(VIEWS)
            assert record['prediction_complete']['status']=='ALL_PREDICTIONS_COMPLETE' and record['prediction_complete']['views']==list(VIEWS)
            assert record['prediction_complete']['truth_read'] is False and record['prediction_complete']['query_fit'] is False
            assert record['configuration']['source_roles']=='LU' and record['configuration']['F_method']=='pamf_t1_e50'
            if contract is None:contract=record['source_contract']
            assert contract==record['source_contract'] and contract['counts']==dict(L_s=6300,U_s=56700,V=27000)
            source=record['source'];assert source['status']=='SOURCE_FROZEN' and source['epoch']==20 and source['updates']==4940
            assert all(source[k]==0 for k in ('TX_labels_read','F_reads','target_reads'))
            training=record['training'];assert [r['epoch'] for r in training]==list(range(1,21))
            for r in training:
                assert r['updates']==247*r['epoch'] and math.isfinite(r['loss']) and r['loss']>=0
                assert r['L_exposures']==6300*r['epoch'] and r['U_exposures']==56700*r['epoch']
            assert source['L_exposures']==126000 and source['U_exposures']==1134000
            for key,r in actual.items():
                if key[0]=='raw':assert r['confusion']==base[seed][key]['confusion']
            parameters.setdefault(arm,source['parameters']);assert parameters[arm]==source['parameters']
            cost.append(dict(arm=arm,seed=seed,epochs=20,updates=4940,parameters=source['parameters'],seconds=source['seconds'],L_exposures=source['L_exposures'],U_exposures=source['U_exposures'],teacher='shared_fixed_R65',incremental_student_only=True))
            add(arm,seed,s,actual,arm)
    summaries=[];deltas=[]
    for arm in ('raw','default',*ARMS):
        for seed in SEEDS:
            per[arm,seed,'six_view_mean']={m:statistics.mean(per[arm,seed,v][m] for v in VIEWS[1:]) for m in ('accuracy','macro_f1','worst_rx')}
        for view in (*VIEWS,'six_view_mean'):
            result=dict(arm=arm,view=view,paired_seed_runs=3,G_parameters=parameters[arm],G_students=0 if arm=='raw' else 1 if arm=='default' else 3)
            for metric in ('accuracy','macro_f1','worst_rx'):
                values=[per[arm,s,view][metric] for s in SEEDS]
                result[metric+'_mean']=statistics.mean(values);result[metric+'_sd']=statistics.stdev(values)
            summaries.append(result)
            if arm!='raw':
                values=[per[arm,s,view]['accuracy']-per['default',s,view]['accuracy'] for s in SEEDS]
                deltas.append(dict(arm=arm,view=view,mean_pp=statistics.mean(values),sample_sd_pp=statistics.stdev(values),actual_pairs=3))
    contrasts=[]
    for name,a,b in (('copy_effect_without_KD','copy_no_KD','random_no_KD'),('KD_effect_from_random','random_with_KD','random_no_KD')):
        for view in (*VIEWS,'six_view_mean'):
            values=[per[a,s,view]['accuracy']-per[b,s,view]['accuracy'] for s in SEEDS]
            contrasts.append(dict(contrast=name,view=view,mean_pp=statistics.mean(values),sample_sd_pp=statistics.stdev(values),actual_pairs=3))
    for filename,data in [('G_ablation_per_seed.csv',rows),('G_ablation_summary.csv',summaries),('G_ablation_vs_default.csv',deltas),('G_ablation_net_corrections.csv',nets),('G_ablation_training_cost.csv',cost),('G_ablation_init_KD_contrasts.csv',contrasts)]:write_csv(output/filename,data)
    verification=dict(status='G_ABLATION_ACCURACY_ANALYZED',student_models=21,arms=7,paired_seed_runs=3,student_epochs=420,student_updates=21*4940,
        student_raw_G_view_cells=294,all_confusion_matrices_recomputed=ncm,all_pair_stratum_nets_recomputed=nnet,
        raw_confusions_identical_to_same_seed_final_T1=True,default_reference_G_models=1,full_F_G_crossed_replication=False,
        all_source_TX_labels_F_reads_target_reads=0,source_counts=contract['counts'],source_receivers=contract['source_rxs'],source_days=contract['source_days'],target_receivers=7,physical_samples_per_view=168000,
        summed_incremental_student_elapsed_hours=sum(r['seconds'] for r in cost)/3600,
        teacher_cost='Shared original R65 ancestry excluded from incremental student time; not free or retrained21 times.',
        interpretation='Each new arm uses three diagonal F-seed/G-student-seed pairs. Default uses three F seeds with one fixed historical G. This is not pure G-seed variability or a fully replicated2x2 copy/KD factorial.',
        timing_status='Mechanism latency still queued; all G variants evaluated eager, default fixed Graph/reuse. Do not infer latency from parameter counts.',target_feedback=False)
    (output/'G_ablation_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    lookup={(r['arm'],r['view']):r for r in summaries};dl={(r['arm'],r['view']):r for r in deltas}
    lines=['# DURA：7机制×3配对种子的完整准确率消融','',
        '21学生模型均完成E20/4940、420轮/103740次更新；同契约L/U IQ、固定R65教师、无TX标签/F/目标训练访问。每行raw/G七视图闭合后独立truth-last，目标每景168000包、7RX。',
        '三次重复按相同编号F seed与G学生seed配对；默认DURA仅一个既有G配三F。SD不是独立G方差估计。默认copy+KD没有新训三G种子，不能将此面板写成完整复制的2×2因果交互。', '',
        '| G设置 | G参数 | clean（%） | 六景（%） | low urban（%） | 六景较默认（pp） |','|---|---:|---:|---:|---:|---:|']
    for arm in ('raw','default',*ARMS):
        cells=[f"{lookup[arm,v]['accuracy_mean']:.4f}±{lookup[arm,v]['accuracy_sd']:.4f}" for v in ('clean','six_view_mean','practical_low_urban')]
        delta='—' if arm=='raw' else f"{dl[arm,'six_view_mean']['mean_pp']:+.4f}±{dl[arm,'six_view_mean']['sample_sd_pp']:.4f}"
        lines.append('| '+' | '.join([LABELS[arm],str(parameters[arm]),*cells,delta])+' |')
    lines+=['','全部场景、Macro-F1、最差RX、逐seed净纠错和实际学生成本见CSV；raw混淆矩阵与既有同seed最终T1逐项一致，所有CM与各stratum净纠错重算通过。', '',
        '复制效应可在no-KD下比较copy_no_KD−random_no_KD；随机初始化下KD效应比较random_with_KD−random_no_KD，均为三次同seed配对。现有单固定G只作默认参考，不用于补造缺少的三G copy+KD组或其交互统计。', '',
        '参数是G唯一注册标量数，不是完整计算量或实测时延。机制附属时延仍在原队列，当前不据参数推断加速；全谱系祖先成本与本轮学生增量成本分列。目标结果不替换默认DURA或改变健康队列。']
    (output/'G_ablation_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if paper:
        text=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Completed DURA mechanism accuracy (\%, mean$\pm$sample SD of three paired runs). Each new G seed is paired with the same-number F seed; the default uses three F seeds and one fixed G. These are not pure G-seed SDs.}',r'\label{tab:G-ablation-complete}',r'\begin{tabular}{lrrrrr}\toprule',r'G setting & G parameters & Clean & Six scenes & Low urban & Six-scene $\Delta$ vs default (pp)\\\midrule']
        for arm in ('raw','default',*ARMS):
            values=['$%.2f\\pm%.2f$'%(lookup[arm,v]['accuracy_mean'],lookup[arm,v]['accuracy_sd']) for v in ('clean','six_view_mean','practical_low_urban')]
            delta='--' if arm=='raw' else '$%+.2f\\pm%.2f$'%(dl[arm,'six_view_mean']['mean_pp'],dl[arm,'six_view_mean']['sample_sd_pp'])
            text.append(' & '.join([LABELS[arm],str(parameters[arm]),*values,delta])+r'\\')
        text+=[r'\bottomrule\end{tabular}\end{table*}'];(Path(paper)/'tables/G_ablation_complete.tex').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summaries,deltas,contrasts,verification


def plot(summary,deltas,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    lookup={(r['arm'],r['view']):r for r in summary};ds={(r['arm'],r['view']):r for r in deltas}
    views=('clean','six_view_mean','practical_low_urban');titles=('Clean','Six-scene mean','Low urban')
    for gain in (False,True):
        arms=ARMS if gain else ('default',*ARMS)
        fig,axes=plt.subplots(1,3,figsize=(11,4.6),sharey=True)
        for ax,view,title in zip(axes,views,titles):
            for y,arm in enumerate(arms):
                row=ds[arm,view] if gain else lookup[arm,view]
                mean=row['mean_pp'] if gain else row['accuracy_mean'];sd=row['sample_sd_pp'] if gain else row['accuracy_sd']
                color='#222222' if arm=='default' else '#0072B2' if arm in ARMS[:4] else '#009E73'
                ax.errorbar(mean,y,xerr=sd,fmt='o',color=color,capsize=3,ms=4,lw=1)
            ax.axvline(0 if gain else lookup['raw',view]['accuracy_mean'],color='.5',ls='--',lw=.8)
            if not gain:ax.set_xlim(0,100)
            ax.set_title(title);ax.set_xlabel('Difference vs default (pp)' if gain else 'Accuracy (%)')
            ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(arms)),[LABELS[a].replace('Default DURA (one fixed G)','Default DURA (fixed G)').replace('Copy, no IQ distillation','Copy, no KD').replace('Random + IQ distillation','Random + KD').replace('Random, no IQ distillation','Random, no KD') for a in arms]);axes[0].invert_yaxis()
        handles=[Line2D([],[],color='#0072B2',marker='o',ls='',label='Structural ablations'),Line2D([],[],color='#009E73',marker='o',ls='',label='Initialization / IQ distillation')]
        handles.append(Line2D([],[],color='.5',ls='--',label='Zero difference' if gain else 'Raw T1 mean'))
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
        fig.tight_layout(rect=(0,.08,1,1))
        name='G_ablation_paired_vs_default' if gain else 'G_ablation_accuracy'
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight');fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--paper');p.add_argument('--figures')
    a=p.parse_args();summary,deltas,_,v=analyze(a.input,a.baseline,a.output,a.paper)
    if a.figures:plot(summary,deltas,a.figures)
    print(json.dumps(v))
