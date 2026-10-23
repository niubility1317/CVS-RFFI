"""Completed main-panel costs, inspectable figures and IEEE tables."""
import argparse,csv,json,statistics
from pathlib import Path
from .analyze_main import PROTOCOLS,REGIMES,SEEDS,read,write_csv


def costs(inputs,output):
    obj=read(Path(inputs)/'G_lineage_cost_inputs.json');rows=[]
    for record in obj['rows']:
        complete=record['completion'];stage=record['stages']
        if len(stage)!=8 or complete['updates']!=39520:raise ValueError('Full eight-stage G required')
        for name,data in stage.items():
            if data['status']!='SOURCE_FROZEN' or data['updates']!=4940 or any(data[k]!=0 for k in ('TX_labels_read','F_reads','target_reads')):
                raise ValueError('G stage budget/permissions differ')
            roles=record['allowed_roles']
            if roles not in ('L','LU',['L_s'],['L_s','U_s']):raise ValueError('Unknown registered G roles')
            if roles in ('L',['L_s']) and data['U_exposures']!=0:raise ValueError('L-only ancestor used U')
            if abs(data['seconds']-complete['seconds_by_stage'][name])>1e-8:raise ValueError('G stage costs disagree')
            rows.append(dict(row_id=record['id'].split('/')[1],model_seed=record['model_seed'],
                roles=record['allowed_roles'],stage=name,epochs=20,updates=4940,
                shared_worker_seconds=data['seconds'],L_exposures=data['L_exposures'],U_exposures=data['U_exposures'],
                TX_labels_read=0,F_reads=0,target_reads=0,execution_commit=record['execution_commit']))
    if len(rows)!=280 or len(obj['rows'])!=35:raise ValueError('Expected35 lineages/280 stage records')
    total=sum(r['shared_worker_seconds'] for r in rows);summary=[]
    for stage in ('coarse','action','r17','r38','r44','r46','r65','r69'):
        secs=[r['shared_worker_seconds'] for r in rows if r['stage']==stage]
        summary.append(dict(stage=stage,lineages=35,total_seconds=sum(secs),median_seconds=statistics.median(secs),
                            cumulative_percent=100*sum(secs)/total))
    out=Path(output);write_csv(out/'main_G_stage_cost.csv',rows);write_csv(out/'main_G_cost_summary.csv',summary)
    verification=dict(status='VERIFIED',complete_lineages=35,complete_stage_records=280,
        model_epochs=5600,optimizer_updates=1383200,shared_worker_hours=total/3600,
        source_TX_F_target_reads=0,includes_original_fixed_F0_G_historical_cost=False,
        scope='All35 newly trained complete G lineages including2 extra m3-natural G seeds; shared observation/IQ caches and21 G students separate')
    (out/'main_G_cost_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    return verification


def render(inputs,analysis,public,paper):
    inputs,analysis,public,paper=map(Path,(inputs,analysis,public,paper));public.mkdir(parents=True,exist_ok=True)
    def csvread(name):
        with (analysis/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    summary=csvread('main_summary_overall.csv');paired=csvread('main_paired_overall.csv');cross=csvread('main_crossed_overall.csv')
    lookup={(r['protocol'],r['regime'],r['readout'],r['test'],r['view'],r['metric']):r for r in summary}
    effects={(r['protocol'],r['effect'],r['readout'],r['test'],r['view'],r['metric']):r for r in paired}
    manifest=read(inputs/'manifest.json');sizes={}
    for item in manifest['rows']:
        if item['protocol_id'] in sizes:continue
        rec=read(inputs/item['member']);row=next(r for r in rec['score']['results'] if r['readout']=='raw' and r['view']=='clean' and r['dimension']=='overall' and r['test']=='main')
        sizes[item['protocol_id']]=row['count']
    def value(row,digits=3,tex=False):return ('%.*f'%(digits,float(row['mean'])))+('\\pm' if tex else ' ± ')+('%.*f'%(digits,float(row['sd'])))
    def s(p,r,reader,view,metric='accuracy',test='main'):return lookup[p,r,reader,test,view,metric]
    def e(p,effect,reader,view,metric='accuracy',test='main'):return effects[p,effect,reader,test,view,metric]
    gains=[e(p,'native_cric_G_minus_raw','DURA-LU','six_practical_mean') for p in PROTOCOLS]
    clean=[e(p,'native_cric_G_minus_raw','DURA-LU','clean') for p in PROTOCOLS]
    worst=min(clean,key=lambda r:float(r['mean']))
    g_cost=read(analysis/'main_G_cost_verification.json');verify=read(analysis/'main_verification.json')
    report=(analysis/'main_results.md').read_text(encoding='utf-8')
    report+='\n## 完整结果的边界\n\n'
    report+='原生T1接G-LU的六景增益在17/17契约均为正，均值范围 %.4f 至 %.4f 个百分点；clean均值有 %d/17 契约非正，最小为 %s 的 %s 个百分点。所有负值和非单调曲线保留。\n\n'%(min(float(r['mean']) for r in gains),max(float(r['mean']) for r in gains),sum(float(r['mean'])<=0 for r in clean),worst['protocol'],value(worst,4))
    for label in ('native_minus_supervised','native_minus_ordinaryPL'):
        rows=[e(p,label,'DURA-LU','six_practical_mean') for p in PROTOCOLS]
        report+='%s 的G-LU六景均值在 %d/17 契约非正，不能宣称原生配方处处占优。\n\n'%(label,sum(float(r['mean'])<=0 for r in rows))
    report+='## 三源自然预算3×3 F/G交叉复现\n\n仅m3-natural，固定六目标RX/144000包。原3条固定G读出加6条新交叉读出组成9格；同F的raw混淆矩阵完全一致。每列SD描述固定G下的三F变化，对角线的三组联合F/G种子用于完整系统重复；九格不能当九次独立系统重复。\n\n|G或配对方式|clean Acc%|六景 Acc%|\n|---|---:|---:|\n'
    cl={(r['summary_type'],str(r['G_seed']),r['test'],r['view'],r['metric']):r for r in cross}
    for label,g in [('G seed01',str(SEEDS[0])),('G seed02',str(SEEDS[1])),('G seed03',str(SEEDS[2])),('三组对角F/G','diagonal')]:
        kind='three_diagonal_independent_systems' if g=='diagonal' else 'fixed_G_over3F'
        report+='|%s|%s|%s|\n'%(label,value(cl[kind,g,'main','clean','accuracy'],4),value(cl[kind,g,'main','six_practical_mean','accuracy'],4))
    report+='\n## 完整成本\n\n150个新主F（包括6条技术替代F0控制）合计 %.6f 个共享worker小时；153个逻辑F含3个原T1复用来源合计 %.6f 小时。已取回450份新三段元数据和9份原T1历史阶段元数据。旧坏输入训练另保留历史，未混入有效模型成本。35条新G谱系包含280个E20阶段、5600模型轮/1383200更新，合计 %.6f 个共享worker小时；包含m3额外两G种子。原固定F0 G的历史训练、共享观测/恢复IQ缓存、21个G学生及预测成本分列，不以复用冒称零成本。这些不是独占GPU小时，也不能按并发数换算为墙钟提速。\n'%(verify['new_F_shared_worker_seconds']/3600,verify['all_F_shared_worker_seconds']/3600,g_cost['shared_worker_hours'])
    report+='\n122598份主矩阵CM和70056个净纠错统计已重新计算；额外6条交叉读出3612份CM另计。repair/damage是冻结评分中的联合预测计数，净值从CM独立核对，并检查合法计数界限，不从CM伪造联合交集。ManyRx按原10类计算Macro-F1，其余按原6类。\n'
    (public/'main_results.md').write_text(report,encoding='utf-8')
    for name in ('main_summary_overall.csv','main_paired_overall.csv','main_F_cost.csv','main_crossed_overall.csv','main_G_cost_summary.csv','main_verification.json','main_G_cost_verification.json'):
        (public/name).write_bytes((analysis/name).read_bytes())
    # Figures retain every registered source-count point, including decreases.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
    images=paper/'images/main153';images.mkdir(parents=True,exist_ok=True)
    labels={REGIMES[0]:'T1 native',REGIMES[1]:'Matched Sup',REGIMES[2]:'Ordinary PL'}
    colors=['#0072B2','#D55E00','#009E73'];ms=[1,2,3,4,6]
    for reader,name in [('raw','main_sourcecount_raw'),('DURA-LU','main_sourcecount_DURA_LU')]:
        fig,axes=plt.subplots(2,2,figsize=(7.05,4.35),layout='constrained')
        for col,budget in enumerate(('natural','fixed')):
            for row,view in enumerate(('clean','six_practical_mean')):
                ax=axes[row,col]
                for r,color in zip(REGIMES,colors):
                    vals=[s('m%d-%s'%(m,budget),r,reader,view) for m in ms]
                    ax.errorbar(ms,[float(v['mean']) for v in vals],yerr=[float(v['sd']) for v in vals],label=labels[r],color=color,marker='o',markersize=3,capsize=2,linewidth=1)
                ax.set_title(budget.capitalize()+(' / clean' if row==0 else ' / six scenes'))
                ax.set_xticks(ms);ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2)
                if row==1:ax.set_xlabel('Source RX count')
        axes[0,0].legend(loc='best')
        for suffix in ('pdf','png'):fig.savefig(images/(name+'.'+suffix),dpi=190)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.55),layout='constrained')
    for ax,budget in zip(axes,('natural','fixed')):
        for view,color,label in [('clean',colors[0],'Clean'),('six_practical_mean',colors[1],'Six scenes')]:
            vals=[e('m%d-%s'%(m,budget),'native_cric_G_minus_raw','DURA-LU',view) for m in ms]
            ax.errorbar(ms,[float(v['mean']) for v in vals],yerr=[float(v['sd']) for v in vals],label=label,color=color,marker='o',markersize=3,capsize=2)
        ax.axhline(0,color='black',linewidth=.7);ax.set_title(budget.capitalize());ax.set_xticks(ms)
        ax.set_xlabel('Source RX count');ax.set_ylabel('G-LU minus raw (points)');ax.grid(alpha=.2)
    axes[0].legend()
    for suffix in ('pdf','png'):fig.savefig(images/('main_restoration_gain.'+suffix),dpi=190)
    plt.close(fig)
    with (analysis/'main_crossed_per_cell.csv').open(encoding='utf-8',newline='') as f:cell=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(3.45,1.9),layout='constrained')
    for ax,view,label in zip(axes,('clean','six_practical_mean'),('Clean','Six scenes')):
        vals={(int(r['F_seed']),int(r['G_seed'])):float(r['percent']) for r in cell if r['test']=='main' and r['dimension']=='overall' and r['view']==view and r['metric']=='accuracy'}
        mat=np.array([[vals[f,g] for g in SEEDS] for f in SEEDS]);ax.imshow(mat,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,'%.2f'%mat[i,j],ha='center',va='center',fontsize=7,color='white' if mat[i,j]>(mat.min()+mat.max())/2 else 'black')
        ax.set_xticks(range(3),('01','02','03'));ax.set_yticks(range(3),('01','02','03'))
        ax.set_xlabel('G seed');ax.set_ylabel('F seed' if ax is axes[0] else '');ax.set_title(label)
        for i in range(3):ax.add_patch(plt.Rectangle((i-.48,i-.48),.96,.96,fill=False,edgecolor='#D55E00',linewidth=1.5))
    for suffix in ('pdf','png'):fig.savefig(images/('main_m3_crossed.'+suffix),dpi=190)
    plt.close(fig)
    tex=[r'\subsubsection{Completed Main Matrix and Crossed F/G Robustness}',
      r'All 153 logical main F rows (17 contracts, three registered recipes and three F seeds) now have complete seven-view predictions and independent scores, including six input-binding replacements for the F0 controls. Local aggregation recomputed 122598 serialized confusion matrices and 70056 repair-minus-damage nets. ManyRx uses its registered ten classes; all other contracts use six. Source/date/TX contracts are kept separate. Native T1 and matched Sup/ordinary-PL differ in their registered training recipes, including augmentation and forward exposure, so their differences do not isolate a single component.',
      r'\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
      r'\caption{Complete native T1 main matrix: Accuracy in percent, mean$\pm$sample SD over three F seeds conditional on the fixed G-LU per contract. S6 averages the six scenes within seed. Different physical target sets are not pooled.}\label{tab:main153}',
      r'\begin{tabular}{lrrrrrr}\toprule Contract & Packets/view & Classes & Raw clean & Raw S6 & G-LU S6 & Paired G gain (points)\\\midrule']
    for p in PROTOCOLS:
        tex.append(p.replace('_',r'\_')+' & '+str(sizes[p])+' & '+str(10 if p=='ManyRx' else 6)+' & '+' & '.join('$'+value(r,3,True)+'$' for r in [s(p,REGIMES[0],'raw','clean'),s(p,REGIMES[0],'raw','six_practical_mean'),s(p,REGIMES[0],'DURA-LU','six_practical_mean'),e(p,'native_cric_G_minus_raw','DURA-LU','six_practical_mean')])+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table*}',
      r'All 17 native-T1 six-scene G-LU/raw mean gains are positive, ranging from %.4f to %.4f points, but five contracts have nonpositive clean mean differences. The largest clean reduction is the one-source fixed-budget condition ($%s$ points). Native T1 minus the supervised and ordinary-PL controls under G-LU is nonpositive in three and five of the 17 six-scene conditions, respectively. These negative and nonmonotonic outcomes are retained; target results do not change the fixed default or queue.'%(min(float(r['mean']) for r in gains),max(float(r['mean']) for r in gains),value(worst,4,True)),
      r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{images/main153/main_sourcecount_raw.pdf}\caption{Matched PAMF source-count curves on raw inputs, three F seeds. The same six target RX/144000 packets are used at every point; no m=5 point or seven-target F0 result is inserted. Error bars are sample SD.}\label{fig:main-source-raw}\end{figure*}',
      r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{images/main153/main_sourcecount_DURA_LU.pdf}\caption{The same trained F models after exact-contract frozen DURA-LU. These curves use U through G even for the supervised F.}\label{fig:main-source-g}\end{figure*}',
      r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{images/main153/main_restoration_gain.pdf}\caption{Within-F-seed G-LU/raw paired gains, preserving all clean reductions and nonmonotonic source-count effects.}\label{fig:main-gain}\end{figure*}',
      r'The predeclared m3-natural robustness panel is also complete: two extra G seeds add six readouts to the original fixed-G column, yielding the complete three-F by three-G grid (Fig.~\ref{fig:main-crossed}). Same-F raw confusion matrices are identical across G choices. Each fixed-G column describes F variability; the three diagonal joint F/G seeds are three complete-system training-seed repetitions. The nine crossed cells are not nine independent complete-system repetitions.',
      r'\begin{table}[t]\centering\footnotesize\caption{m3-natural crossed robustness (Accuracy,\%).}\label{tab:main-crossed}\begin{tabular}{lrr}\toprule Condition & Clean & S6\\\midrule']
    for label,g in [('Fixed G01',str(SEEDS[0])),('Fixed G02',str(SEEDS[1])),('Fixed G03',str(SEEDS[2])),('Joint diagonal','diagonal')]:
        kind='three_diagonal_independent_systems' if g=='diagonal' else 'fixed_G_over3F'
        tex.append(label+' & '+' & '.join('$'+value(cl[kind,g,'main',v,'accuracy'],3,True)+'$' for v in ('clean','six_practical_mean'))+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table}',
      r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{images/main153/main_m3_crossed.pdf}\caption{Complete m3-natural F/G seed grid. Red boxes identify diagonal joint-seed repetitions; all cells are reported.}\label{fig:main-crossed}\end{figure}',
      r'The 150 newly trained main F models sum to %.3f shared-worker elapsed hours; all 153 logical sources including reused historical T1 segments sum to %.3f hours. The 35 new complete G lineages contain 280 fixed E20 stages (5600 model-epochs/1383200 updates), summing to %.3f shared-worker elapsed hours, including the two additional crossed G seeds. These are cumulative costs under shared contention, not exclusive GPU hours or wall-clock speedup measurements. Original fixed-F0 G ancestry, shared observation/IQ-cache production, the 21 G students and prediction cost remain separate. Per-scene/RX/TX/day Accuracy, Macro-F1, chrono auxiliary tasks, net corrections and every cost record are preserved for reproduction. The PL-F/G-L supplement and external-method DURA readouts remain separate pending panels.'%(verify['new_F_shared_worker_seconds']/3600,verify['all_F_shared_worker_seconds']/3600,g_cost['shared_worker_hours'])]
    (paper/'main_matrix_results.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='WRITTEN',public=str(public),paper=str(paper/'main_matrix_results.tex'),figure_groups=4)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--analysis',required=True);p.add_argument('--public',required=True);p.add_argument('--paper',required=True)
    a=p.parse_args();costs(a.input,a.analysis);render(a.input,a.analysis,a.public,a.paper)
