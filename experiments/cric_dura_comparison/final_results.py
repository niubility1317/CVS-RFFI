"""Present the completed final panels using audited CSVs; plots only need matplotlib."""
import argparse,csv,json,shutil
from pathlib import Path
from .analyze_structure import ARMS

STRUCT='20261022-phase1-PAMF-supervised-physics-m24-r01'
EXT='20261023-phase1-external-DURA-analysis-m516-r01'
FAIR='20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01'
FINE_TIME='20261022-phase1-T1-DURA-F-binding-mechanism-timing-r01'
STRUCT_TIME='20261023-phase1-PAMF-supervised-physics-timing-recovery-r01'
FAMILIES=('riei','drift','ibrff','sdg')
LABELS=dict(riei='RIEI',drift='DRIFT-v2',ibrff='IB-RFF',sdg='SDG')


def write_csv(path,data):
    # Public presentation tables are regenerated in place; frozen analysis directories stay immutable.
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def pick(data,**where):
    r=[r for r in data if all(str(r[k])==str(v) for k,v in where.items())]
    if len(r)!=1:raise ValueError(('Expected one result',where,len(r)))
    return r[0]


def pm(r,digits=2,tex=False):
    sep=r'\pm' if tex else '±'
    return f"{float(r['mean']):.{digits}f}{sep}{float(r['sd']):.{digits}f}"


def render(base,project):
    base=Path(base);project=Path(project);paper=project/'paper/phase1_experiments_ieee_20261020'
    image=paper/'images/final_panels';image.mkdir(parents=True,exist_ok=True)
    structure=rows(base/'structure-py313/structure_summary_overall.csv')
    sp=rows(base/'structure-py313/structure_paired_overall.csv')
    external=rows(base/'external-final-py310/external_final_summary_overall.csv')
    ep=rows(base/'external-final-py310/external_final_paired_overall.csv')
    gf=[r for r in rows(base/'replay-py313/gfactor/gfactor_summary.csv') if r['dimension']=='overall']
    effects=[r for r in rows(base/'replay-py313/gfactor/gfactor_paired_effects.csv') if r['dimension']=='overall']
    timing={name:rows(base/'timing-py313'/f'{name}_summary.csv') for name in ('fine_timing','structure_timing')}
    common=dict(test='main',dimension='overall',stratum='ALL',metric='accuracy')
    def st(arm,reader,view):return pick(structure,architecture=arm,readout=reader,view=view,**common)
    def ex(fam,var,reg,reader,view='six_practical_mean',pid='F0'):
        return pick(external,protocol=pid,family=fam,variant=var,regime=reg,task='main',readout=reader,view=view,metric='accuracy')
    def fair(pid,cell):return pick(gf,protocol_id=pid,cell=cell,view='six_practical_mean',**common)
    def effect(pid,name):return pick(effects,protocol_id=pid,effect=name,view='six_practical_mean',**common)
    def report(run):
        p=project/'automation_reports/CV-SincNet'/run;p.mkdir(parents=True,exist_ok=True);return p
    sr,er,fr=report(STRUCT),report(EXT),report(FAIR)
    for src,dest in ((base/'structure-py313',sr),(base/'external-final-py310',er),(base/'replay-py313/gfactor',fr)):
        for p in src.glob('*.json'):
            if p.name not in ('source_cost_reuse.json','reused_costs.json'):shutil.copy2(p,dest/p.name)
    for p in (base/'structure-py313').glob('*.csv'):
        if 'overall' in p.name or p.name.endswith(('F_cost.csv','source_geometry.csv','source_sensitivity.csv')):shutil.copy2(p,sr/p.name)
    for name in ('external_final_summary_overall.csv','external_final_paired_overall.csv','external_final_source_cost.csv','external_final_cache_cost.csv'):
        shutil.copy2(base/'external-final-py310'/name,er/name)
    for name,data in (('gfactor_summary_overall.csv',gf),('gfactor_paired_overall.csv',effects)):write_csv(fr/name,data)
    for name,run in (('fine_timing',FINE_TIME),('structure_timing',STRUCT_TIME)):
        d=report(run)
        for f in (base/'timing-py313').glob(name+'*.csv'):shutil.copy2(f,d/f.name)
        proof=json.loads((base/'timing-py313/mechanism_timing_verification.json').read_text())[name]
        (d/'timing_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    # Explicitly retain negative conditions, without pooling different data contracts.
    ec=[r for r in ep if r['contrast']=='G_minus_raw' and r['view']=='six_practical_mean' and r['metric']=='accuracy' and r['task']=='main']
    clean=[r for r in ep if r['contrast']=='G_minus_raw' and r['view']=='clean' and r['metric']=='accuracy' and r['task']=='main']
    full=[r for r in ep if r['contrast']=='Full_minus_CE' and r['view']=='six_practical_mean' and r['metric']=='accuracy' and r['task']=='main']
    ext_counts=dict(six_restoration_conditions=len(ec),six_nonpositive=sum(float(r['mean'])<=0 for r in ec),
        clean_restoration_conditions=len(clean),clean_nonpositive=sum(float(r['mean'])<=0 for r in clean),
        Full_minus_CE_conditions=len(full),Full_minus_CE_nonpositive=sum(float(r['mean'])<=0 for r in full))
    (er/'effect_condition_counts.json').write_text(json.dumps(ext_counts,indent=2)+'\n',encoding='utf-8')
    text=['# 严格监督结构与物理先验：完整结果','',
        '24 个新 F（8 个结构×3 seed）及3个复用 X3_CE 对照全部完成。F0 为5源RX/7目标RX、168000包、六类；全程只用 L，CE+sat.68、LS0，无CRIC/KL/U/EMA。均50轮/11100更新，默认头联合训练，384个L-packet前向/更新。',
        'raw 与同一全 L 祖先 G-L 七景冻结后独立评分；三 F 变化条件于一个 G，不能称三个独立完整系统。',
        '', '| 结构 | raw clean（%） | raw 六景（%） | G-L 六景（%） | X3−变体 G-L 六景（pp） |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        delta='参照' if arm=='X3_CE' else pm(pick(sp,architecture=arm,contrast='X3_CE_minus_variant',readout='DURA-L',view='six_practical_mean',**common),4)
        text.append(f"| {arm} | {pm(st(arm,'raw','clean'))} | {pm(st(arm,'raw','six_practical_mean'))} | {pm(st(arm,'DURA-L','six_practical_mean'))} | {delta} |")
    proof=json.loads((sr/'structure_verification.json').read_text())
    text+=['','去 PA、signature 和 reference 分支的同预算配对差均为正；这些删除也改变参数或计算。等参数 no-time 与 wrong-reference 控制提供较小差异。Shuffled-time 在 G-L 后六景优于原结构，covariance 均值也略高：结果不支持“原时间顺序不可替代”或“每个物理先验都稳定增益”的结论。raw 与恢复后必须分列，不能用恢复掩盖 raw 的结构敏感性。',
        '',f"24个新F累计 {proof['new_F_shared_worker_seconds']/3600:.6f} 共享worker小时（72源阶段/1200模型轮/266400更新）；3个CE参照复用 {proof['reused_X3_CE_shared_worker_seconds']/3600:.6f} 历史小时，不是新增或独占GPU时间。WRONG_REFERENCE seed01按设备修复run映射；旧失败保留。",'',
        '27个源诊断都使用同720条L、90个TX/RX/day层和11干预，无U、target或拟合探针。几何间隔、特征位移及默认头翻转仅是敏感性/归纳偏置证据；源几何排序不等于目标识别排序，模拟PA不假定身份保持，不证明完整硬件解耦。完整逐景、RX/TX/day、Accuracy/Macro-F1、净纠错与成本在导出包。']
    (sr/'structure_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    text=['# 四种外部方法接入同契约DURA','',
        '516个既有冻结F（156原生比较+360源数量）×raw/G-L/G-LU=1548读出。没有新增F或G训练。每行七景及chrono附任务先冻结后truth-last；34共享缓存仅复用同一FP32恢复IQ，不包含标签/分类输出，按物理ID不再归一化。',
        'F0/各源数量点保留三Fseed、Full/自身CE、原预算和可用的Sup/普通PL。标签比例仅已有IB-RFF。ManyRx按10类，其余按6类；不混合不同契约总体平均。',
        '', '| F0 Full 方法 | F训练 | raw六景（%） | G-L六景（%） | G-LU六景（%） |','|---|---|---:|---:|---:|']
    for fam in FAMILIES:
        for reg in ('supervised','pl'):
            text.append('| '+LABELS[fam]+' | '+reg+' | '+' | '.join(pm(ex(fam,'full',reg,r)) for r in ('raw','DURA-L','DURA-LU'))+' |')
    text+=['',f"全部主任务条件中，恢复六景均值非正 {ext_counts['six_nonpositive']}/{ext_counts['six_restoration_conditions']}，clean非正 {ext_counts['clean_nonpositive']}/{ext_counts['clean_restoration_conditions']}；Full−CE六景非正 {ext_counts['Full_minus_CE_nonpositive']}/{ext_counts['Full_minus_CE_conditions']}。这些只是逐契约/方法/配方/恢复角色条件计数，不是独立数据集重复或显著性检验。负值、零PL接受率和非单调曲线均保留。",'',
        '仅监督F+G-L为完整L-only系统；监督F+G-LU通过G使用U，普通PL通过F使用U。固定同一G下的三F标准差不能称G方差。六景先seed内等权汇总再三seed统计；Full−CE、PL−Sup和LU−L均先同seed配对。',
        '源数量曲线只用m1/2/3/4/6、六目标RX/144000包；F0五源七目标168000单列，不补m5。chrono全日期main不称未来泛化，未来端点保留D3日期分层，source_unseen_date另列。',
        '516模型的654个原训练阶段、增强/PL实际计数已完整取回。原F历史成本与G祖先、一次性缓存、分类/CPU评分分列；缓存分类耗时不是在线G+F时延。源星地拼接审计已PASS，不重训。']
    (er/'external_DURA_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    text=['# PAMF普通PL与恢复数据权限：完整四格','',
        '17契约×3Fseed×4格=204读出。此次只补51个普通PL F×G-L预测、51独立评分和1CPU汇总；无F/G训练、新缓存或时延。F0使用已修复controls-binding来源。三F条件于固定同契约G-L/G-LU；native最终T1原生SSL不混入普通PL四格。',
        '', '| F0 cell | 六景Accuracy（%） |','|---|---:|']
    for cell in ('Sup+G-L','Sup+G-LU','PL+G-L','PL+G-LU'):text.append(f"| {cell} | {pm(fair('F0',cell),4)} |")
    text+=['','| F0配对效应 | 六景差（pp） | 正seed数 |','|---|---:|---:|']
    names=('PL_minus_Sup_at_G_L','PL_minus_Sup_at_G_LU','G_LU_minus_L_at_Sup','G_LU_minus_L_at_PL','F_PL_x_G_U_interaction')
    for name in names:
        r=effect('F0',name);text.append(f"| {name} | {pm(r,4)} | {r['positive_seeds']}/3 |")
    text+=['','只Sup+G-L为完整L-only；PL+G-L仍经F用U。PL收益和G使用U收益因契约而异，不能从F0推出所有场景正交互。差分交互先seed内相减，再汇总三seed；SD不是显著性检验。全部逐RX/TX/day、Accuracy/Macro-F1及相对同F raw的净纠错保留。源F/G历史成本、共享缓存和本次分类成本分别保留。']
    (fr/'gfactor_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    # Scientific figures use the same small audited tables. Statistics above are stdlib-only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(image/(name+'.pdf'),bbox_inches='tight');fig.savefig(image/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,3.0));x=list(range(9))
    for reader,color,offset in (('raw','#cf823c',-.10),('DURA-L','#2678a8',.10)):
        rr=[st(a,reader,'six_practical_mean') for a in ARMS]
        ax.errorbar([z+offset for z in x],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],fmt='o',capsize=2,label=reader,color=color)
    ax.set_xticks(x,[a.replace('_','\n') for a in ARMS]);ax.set_ylabel('Six-view accuracy (%)');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2);save(fig,'structure_accuracy')
    sensitivity=rows(base/'structure-py313/structure_source_sensitivity.csv');geometry=rows(base/'structure-py313/structure_source_geometry.csv')
    inter=sorted({r['intervention'] for r in sensitivity if r['source_metric']=='feature_cosine_distance'})
    fig,(a,b)=plt.subplots(1,2,figsize=(7.1,3.0),gridspec_kw={'width_ratios':[1,2]})
    vals=[pick(geometry,architecture=arm,source_metric='separation_margin') for arm in ARMS]
    a.errorbar([float(r['mean']) for r in vals],x,xerr=[float(r['sd']) for r in vals],fmt='o',capsize=2,color='#2678a8');a.set_yticks(x,ARMS);a.invert_yaxis();a.set_xlabel('Source geometry margin')
    grid=[[float(pick(sensitivity,architecture=arm,intervention=i,source_metric='feature_cosine_distance')['mean']) for i in inter] for arm in ARMS]
    short={'FIR_delay3_echo02':'FIR echo','flat_gain_0.5':'Gain 0.5','flat_gain_2.0':'Gain 2.0',
        'flat_phase_pi2':'Phase pi/2','identity':'Identity','relative_CFO_-100000':'CFO -100k',
        'relative_CFO_-25000':'CFO -25k','relative_CFO_100000':'CFO +100k','relative_CFO_25000':'CFO +25k',
        'synthetic_PA_alpha_0.05':'PA 0.05','synthetic_PA_alpha_0.1':'PA 0.1'}
    im=b.imshow(grid,aspect='auto',cmap='viridis');b.set_yticks(x,[]);b.set_xticks(range(len(inter)),[short[i] for i in inter],rotation=65,ha='right',fontsize=7);b.set_title('Source-only feature cosine displacement');fig.colorbar(im,ax=b,shrink=.75);save(fig,'structure_physics')
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
    for ax,reg in zip(axes,('supervised','pl')):
        for reader,color,offset in (('raw','#888888',-.18),('DURA-L','#2678a8',0),('DURA-LU','#29a07d',.18)):
            rr=[ex(f,'full',reg,reader) for f in FAMILIES];ax.errorbar([z+offset for z in range(4)],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],fmt='o',capsize=2,label=reader,color=color)
        ax.set_xticks(range(4),[LABELS[f] for f in FAMILIES]);ax.set_title('F0 Full: '+reg);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Six-view accuracy (%)');axes[1].legend(fontsize=7);save(fig,'external_F0')
    for schedule,reg in (('natural','supervised'),('natural','pl'),('fixed','supervised')):
        fig,axes=plt.subplots(2,2,figsize=(7.1,4.0),sharex=True,sharey=True)
        for ax,fam in zip(axes.flat,FAMILIES):
            for reader,color in (('raw','#888888'),('DURA-L','#2678a8'),('DURA-LU','#29a07d')):
                rr=[ex(fam,'full',reg,reader,pid=f'm{n}-{schedule}') for n in (1,2,3,4,6)]
                ax.errorbar([1,2,3,4,6],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],marker='o',ms=3,capsize=2,label=reader,color=color)
            cell='PL+G-LU' if reg=='pl' else 'Sup+G-LU';rr=[fair(f'm{n}-{schedule}',cell) for n in (1,2,3,4,6)]
            ax.errorbar([1,2,3,4,6],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],color='k',ls='--',label='PAMF matched '+cell,lw=1,capsize=2)
            ax.set_title(LABELS[fam]);ax.set_xticks([1,2,3,4,6]);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6,ncol=2);fig.supxlabel('Physical source RX (same six target RX)');fig.supylabel('Six-view accuracy (%)');fig.tight_layout();save(fig,'external_source_'+schedule+'_'+reg)
    pids=['F0','F1','F2','ManyRx','chrono','label01','label05']+[f'm{n}-{s}' for s in ('natural','fixed') for n in (1,2,3,4,6)]
    fig,axes=plt.subplots(3,1,figsize=(7.1,5.0),sharex=True)
    legends=('PL-Sup at G-L','PL-Sup at G-LU','G-LU minus G-L at Sup','G-LU minus G-L at PL','PL by G-U interaction')
    for ax,nameset in zip(axes,(names[:2],names[2:4],names[4:])):
        for name in nameset:
            rr=[effect(pid,name) for pid in pids];ax.errorbar(range(17),[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],marker='o',ms=3,capsize=2,label=legends[names.index(name)])
        ax.axhline(0,color='.5',lw=.7);ax.set_ylabel('Paired pp');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2)
    axes[-1].set_xticks(range(17),pids,rotation=55,ha='right');fig.tight_layout();save(fig,'gfactor_effects')
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    for suffix,color,offset in (('_F','#cf823c',-.12),('_GF','#2678a8',.12)):
        rr=[pick(timing['structure_timing'],case=arm+suffix,batch=1) for arm in ARMS]
        ax.errorbar([i+offset for i in range(9)],[float(r['median_ms']) for r in rr],yerr=[float(r['round_median_SD_ms']) for r in rr],fmt='o',capsize=2,color=color,label='F eager' if suffix=='_F' else 'G-L+F graphs/reuse')
    ax.set_xticks(range(9),[a.replace('_','\n') for a in ARMS]);ax.set_ylabel('B1 GPU wall median (ms)');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'structure_latency')
    # New paper content is inserted in the main text, before existing appendices.
    tex=[r'\subsection{Completed Labeled-Only Physical-Structure Panel}',
        r'All 24 new structure models and three reused X3 CE references complete the registered seven-view protocol. The source/input binding repair and the device-only WRONG\_REFERENCE replacement are applied before aggregation; failed historical outputs are excluded. This panel uses matched CE, satellite weight 0.68, no label smoothing, CRIC, KL, U, or EMA, and 50 epochs/11,100 updates per F. It is separate from the native T1 method and the historical single-view E200 architecture study.',
        r'\begin{table*}[t]\centering\footnotesize\caption{Strict L-only architecture results (mean$\pm$sample SD over three F seeds, conditional on one G-L). Six-view means are formed within each seed. X3$-$variant is paired; positive values favor X3. Deletions can alter capacity and computation.}\label{tab:structure-complete}\begin{tabular}{lrrrrr}\toprule',
        r'Structure & Raw clean (\%) & Raw six (\%) & G-L six (\%) & G-L six Macro-F1 (\%) & X3$-$variant (pp)\\\midrule']
    for arm in ARMS:
        delta='---' if arm=='X3_CE' else '$'+pm(pick(sp,architecture=arm,contrast='X3_CE_minus_variant',readout='DURA-L',view='six_practical_mean',**common),3,True)+'$'
        f1=pick(structure,architecture=arm,readout='DURA-L',view='six_practical_mean',test='main',dimension='overall',stratum='ALL',metric='macro_f1')
        tex.append(arm.replace('_',r'\_')+' & '+' & '.join('$'+pm(r,2,True)+'$' for r in (st(arm,'raw','clean'),st(arm,'raw','six_practical_mean'),st(arm,'DURA-L','six_practical_mean'),f1))+' & '+delta+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table*}',
        r'The PA, sequence-residual and known-excitation-response deletions reduce restored six-view accuracy by $2.792\pm0.856$, $2.172\pm0.106$ and $1.478\pm0.008$ pp, respectively. The equal-parameter no-time and wrong-reference controls yield smaller paired differences ($0.499\pm0.150$ and $0.250\pm0.216$ pp). However, shuffled time exceeds X3 after G-L by $1.133\pm0.769$ pp, and covariance X3 is slightly higher on average. These outcomes do not establish that the original sequence ordering is essential or that every physical prior improves recognition. Raw sensitivity and restored performance are reported separately.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.92\textwidth]{images/final_panels/structure_accuracy.pdf}\caption{Matched strict-CE structure controls. Error bars are three-F-seed SD, not significance intervals.}\label{fig:structure-accuracy}\end{figure*}',
        r'All 27 source diagnostics use the same 720 labeled packets across 90 TX/RX/day strata and 11 fixed interventions, with zero U/target reads or probe updates. The source geometry margin does not rank target accuracy: deleting the sequence-residual branch increases the source margin while reducing restored recognition. Fixed interventions therefore characterize sensitivity and inductive bias; they do not recover hardware parameters or prove complete receiver/channel disentanglement. Synthetic PA perturbations are not assumed identity-preserving.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.94\textwidth]{images/final_panels/structure_physics.pdf}\caption{Source-only geometry and deployed-feature response to fixed interventions; no probe fitting or target optimization.}\label{fig:structure-physics}\end{figure*}',
        f"The 24 new F models require {proof['new_F_shared_worker_seconds']/3600:.3f} shared-worker elapsed hours across 72 source stages (1200 model epochs, 266400 updates). The three CE references reuse {proof['reused_X3_CE_shared_worker_seconds']/3600:.3f} historical hours. These are neither isolated GPU hours nor new end-to-end wall time; shared G-L ancestry is reported once separately.",
        r'\subsection{Restoration Transfer to Four External Classifiers}',
        r'All 516 existing external F models now have raw, same-contract G-L and G-LU readouts (1548 cells before views/tasks), with no additional F/G training. All declared predictions are frozen before independent scoring. We reconstruct 516600 serialized confusion matrices, including RX/TX/day strata, and preserve the chronological auxiliary task. ManyRx uses ten classes; the other contracts use six. The 34 FP32 restored-IQ caches contain no labels or classifier outputs and do not renormalize the locked received-IQ observations.',
        r'\begin{table*}[t]\centering\footnotesize\caption{F0 external classifiers after restoration. Each row retains its native training budget and jointly trained default head. Sup+G-L alone is fully L-only; PL uses U through F, and G-LU uses U through restoration. Means$\pm$SD are over three F seeds conditional on fixed G.}\label{tab:external-DURA-complete}\begin{tabular}{lllrrrr}\toprule',
        r'Method & Objective & F regime & Raw clean (\%) & Raw six (\%) & G-L six (\%) & G-LU six (\%)\\\midrule']
    for fam in FAMILIES:
        for reg in ('supervised','pl'):
            for var in ('ce','full'):
                tex.append(LABELS[fam]+' & '+var.upper()+' & '+('Sup' if reg=='supervised' else 'PL')+' & '+' & '.join('$'+pm(r,2,True)+'$' for r in (ex(fam,var,reg,'raw','clean'),ex(fam,var,reg,'raw'),ex(fam,var,reg,'DURA-L'),ex(fam,var,reg,'DURA-LU')))+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table*}',
        f"Across the declared main-task method/recipe/contract/restorer conditions, {ext_counts['six_nonpositive']}/{ext_counts['six_restoration_conditions']} restored-minus-raw six-view means and {ext_counts['clean_nonpositive']}/{ext_counts['clean_restoration_conditions']} clean means are nonpositive. Full-minus-own-CE is nonpositive in {ext_counts['Full_minus_CE_nonpositive']}/{ext_counts['Full_minus_CE_conditions']} six-view conditions. These condition counts are not independent dataset replications or significance tests. Negative effects, nonmonotonic curves and zero ordinary-PL acceptance remain in the analysis.",
        r'\begin{figure*}[t]\centering\includegraphics[width=.91\textwidth]{images/final_panels/external_F0.pdf}\caption{Frozen external Full classifiers with raw and matched L/LU restoration on F0. Restoration helps some classifiers substantially and does not uniformly improve all methods or clean views.}\label{fig:external-F0}\end{figure*}',
        r'Source-count curves retain $m=1,2,3,4,6$ with exactly six target RX/144000 packets. F0 has seven target RX/168000 packets and is not an $m=5$ curve point. Natural supervised, natural ordinary-PL and fixed-L/V supervised curves are exported separately, including own-CE controls. The dashed PAMF line uses the matching supervised or ordinary-PL recipe and the same G-LU permission; native selective T1 SSL remains a separate method. Chronological all-date aggregates are not labeled future generalization; D3 and source-RX unseen-date diagnostics retain their own strata.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.92\textwidth]{images/final_panels/external_source_natural_supervised.pdf}\caption{Natural-budget supervised source-count curves after frozen restoration. Each panel retains its method-native budget; the PAMF matched Sup+G-LU reference uses the same six-target contract. Error bars: three-F-seed SD.}\label{fig:external-source}\end{figure*}',
        r'The 654 original F-stage records retain selection, epochs, updates, augmentation and PL exposures. Shared G ancestry, one-time restored-IQ generation, cached classification and CPU scoring costs are separated. Cache classification time is not an online G+F latency measurement. The already completed concatenation/0.68-gradient audit is reused; no external classifier is retrained.',
        r'\subsection{Ordinary-PL and G-Data-Permission Factorial}',
        r'The 51 added PL-F/G-L predictions close 204 cells over 17 contracts, three F seeds and the four combinations of supervised/ordinary-PL F with L/LU G. Raw decisions are exactly reused for each F. Native selective T1 SSL is excluded from this ordinary-PL factorial. All within-seed task/view aggregates precede paired contrasts and three-seed summaries.',
        r'\begin{table}[t]\centering\footnotesize\caption{F0 data-permission factorial (six-view accuracy). Only Sup+G-L is a fully L-only system.}\label{tab:gfactor-complete}\begin{tabular}{lr}\toprule Cell & Mean$\pm$SD (\%)\\\midrule']
    for cell in ('Sup+G-L','Sup+G-LU','PL+G-L','PL+G-LU'):tex.append(cell+' & $'+pm(fair('F0',cell),3,True)+r'$\\')
    tex += [r'\bottomrule\end{tabular}\end{table}',
        r'On F0, PL-minus-Sup is $-0.134\pm0.551$ pp with G-L and $-0.165\pm0.134$ pp with G-LU. G-LU-minus-G-L is $0.940\pm0.367$ pp at Sup and $0.909\pm0.295$ pp at PL. The paired difference-in-differences is $-0.031\pm0.420$ pp. These results do not establish positive PL/restoration synergy. Effects vary across contracts, including negative G-U contributions on the chronological aggregate. SD is not a significance test.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.92\textwidth]{images/final_panels/gfactor_effects.pdf}\caption{Permission-factorial paired effects across all registered contracts. The chronology main aggregate covers all dates; its D3 future endpoint remains separately available. Error bars are three-F-seed SD.}\label{fig:gfactor-effects}\end{figure*}',
        r'\subsection{Completed Dedicated Mechanism Timing}',
        r'The two source-only adjunct panels finish on exclusive RTX3090 GPUs 5 and 1, excluding GPUs 2/3. The corrected-F/G panel covers 36 cases by five batch sizes by five rounds (900 case-rounds, 252000 GPU-wall and 18000 CPU-return calls); the strict-structure panel covers 18 cases (450 case-rounds, 126000 GPU-wall and 9000 CPU-return calls). Each uses fixed seed01 weights, FP32/TF32 off, batch sizes 1/8/32/64/128, and two CPU threads. Five timing rounds are not five model seeds. Every round median/P95 is independently reconstructed from its samples.',
        r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}\caption{Dedicated strict-structure B1 timing (ms). F-only is eager; G-L+F uses Graphs and residual reuse. These execution routes differ.}\label{tab:structure-timing}\begin{tabular}{lrrr}\toprule Structure & F eager & G-L+F & G-L+F P95\\\midrule']
    for arm in ARMS:
        f=pick(timing['structure_timing'],case=arm+'_F',batch=1);g=pick(timing['structure_timing'],case=arm+'_GF',batch=1)
        tex.append(arm.replace('_',r'\_')+f" & {float(f['median_ms']):.3f} & {float(g['median_ms']):.3f} & {float(g['P95_ms']):.3f}"+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table}',
        r'G-student mechanism timings use eager G (and eager F for G+F); they must not be interpreted as residual-reuse/Graph speedups. Corrected F06--F10 use eager F-only and the locked optimized G+F route. Their complete batch, P95, CPU-return, five-round dispersion, operator and incremental-allocation tables are exported. Absolute resident memory contains all benchmark models and Graph pools; it is not isolated single-model VRAM. Deployment parameter counts exclude the larger training wrapper. First-call values follow loading/profiling, not fresh-process startup.',
        r'The previously delivered uniform latency-r02 result remains the official final T1 default: B1 median 4.541187ms, P95 4.725135ms, and CPU class return 4.611311ms. It is not remeasured or replaced by these adjunct panels. The sole historical RIEI-import timing failure remains recorded and explicitly substituted by r02; the terminal controller label can therefore retain PARTIAL\_TECHNICAL\_FAILURE while every effective scientific phase is closed.',
        r'All current declared training, frozen prediction, independent scoring and timing phases are complete. Unpublished fine-control training and its additional proposed timing rounds remain explicitly deferred, rather than fabricated as completed evidence. No target result changes defaults, parameters, selection, queues or reruns.']
    (paper/'final_panels_results.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    for name,run in (('fine_timing',FINE_TIME),('structure_timing',STRUCT_TIME)):
        txt=['# 附属机制时延完整结果','',f'run `{run}`。独占RTX3090、固定seed01、FP32/TF32off。五轮为计时重复，不是模型seed。',
            'F-only为eager；F机制的GF为Graph+reuse；G学生变体G/GF为eager。只比较明确同执行路线。绝对共驻留显存不能称单模型VRAM；首调用不是新进程冷启动。正式统一r02不重测。',
            '', '| B1 case | GPU median/P95（ms） | CPU返回median（ms） |','|---|---:|---:|']
        for r in timing[name]:
            if r['batch']=='1':txt.append(f"| {r['case']} | {float(r['median_ms']):.6f}/{float(r['P95_ms']):.6f} | {float(r['CPU_return_median_ms']):.6f} |")
        (report(run)/'timing_results.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    return ext_counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--project',required=True)
    a=p.parse_args();print(json.dumps(render(a.base,a.project)))
