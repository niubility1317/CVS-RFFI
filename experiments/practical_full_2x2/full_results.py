"""Public tables, evidence-based IEEE prose and vector figures for the full matrix."""
import argparse,csv,json,statistics,shutil
from pathlib import Path
from .snapshot_results import TASKS,VARIANTS,VIEWS


def render(analysis,public,paper):
    analysis,public,paper=map(Path,(analysis,public,paper));public.mkdir(parents=True,exist_ok=False)
    def rows(name):
        with (analysis/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    summary=rows('summary_overall.csv');effects=rows('factorial_raw_overall.csv');cost=rows('F_cost.csv')
    verification=json.loads((analysis/'verification.json').read_text())
    lookup={(r['protocol'],r['family'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in summary}
    protocols=verification['protocols'];names=dict(riei='RIEI',drift='DRIFT-v2',ibrff='IB-RFF',sdg='SDG',pamf='PAMF')
    titles=dict(cross_date='跨日：源 RX、未见 D0',cross_rx='跨接收机：目标 RX、D1–D3',cross_both='双跨：目标 RX、D0')
    def get(p,f,c,t,route,view='six_practical_mean',metric='accuracy'):
        return lookup[p,f,c,t,route,view,metric]
    def fmt(r,tex=False):
        return (r'\pmv{%0.2f}{%0.2f}' if tex else '%0.2f ± %0.2f')%(float(r['mean']),float(r['sd']))
    def route(c):return 'DURA-LU' if c in ('B','D','native_T1') else 'DURA-L'
    report=['# practical residual Full 四组合：完整 627 行结果','',
      '状态：全部 600 新 F、4 条完整新 G 谱系、22 共享恢复 IQ 缓存、627 条冻结预测与独立评分均已完成。'
      '11 个数据契约分别统计，三 F 种子为 2026092701/02/03。', '',
      'A＝无信道增强/无 PL；B＝无信道增强/普通 PL；C＝practical residual 增强/无 PL；D＝增强/普通 PL。'
      '四种外部方法均保留 Full 原生目标。PAMF A/C 为匹配监督配方，原生 T1 保留原自身 EMA/选择性半监督机制。', '',
      'A/C/PAMF 监督读出使用全 L 祖先 G-L；B/D/T1 使用 G-LU。恢复后 PL 组间差异包含 G 是否使用 U 的差异。'
      '原生 T1 未被改成普通 PL 或共同增强配方；方法间差异是完整配方比较。', '',
      'D1–D3 训练保持；D0 是较早未见日期。下表为百分数的三种子均值 ± 样本标准差。'
      'S6 先在种子内等权平均六景再跨种子汇总。G 在每个契约/角色固定，SD 不是三个独立完整系统的波动或显著性检验。','']
    for p in protocols:
        report += ['## '+p,'']
        for task in TASKS:
            report += ['### '+titles[task],'','|方法|组合|Raw clean|Raw S6|恢复 clean|恢复 S6|恢复 S6 Macro-F1|',
                       '|---|---|---:|---:|---:|---:|---:|']
            for family,cell in VARIANTS:
                vals=[get(p,family,cell,task,'raw','clean'),get(p,family,cell,task,'raw'),
                      get(p,family,cell,task,route(cell),'clean'),get(p,family,cell,task,route(cell)),
                      get(p,family,cell,task,route(cell),metric='macro_f1')]
                report.append('|'+ '|'.join([names[family],cell,*map(fmt,vals)])+'|')
            report.append('')
    negatives={}
    for effect in ('channel_no_PL','channel_with_PL','PL_no_channel','PL_with_channel'):
        selected=[r for r in effects if r['effect']==effect and r['view']=='six_practical_mean' and r['metric']=='accuracy']
        negatives[effect]=dict(total=len(selected),nonpositive=sum(float(r['mean'])<=0 for r in selected))
    report+=['## 完整性与负结果','',
      f"总计 {verification['total_serialized_CM']} 份序列化混淆矩阵、{verification['total_net_cells']} 个净纠错单元。"
      'F1/F2 的 4788 份总体 CM 和 2394 个总体净纠错复用先前核验，其他及新增分层首次复算。ManyRx 按 10 类，其余按 6 类计算 Macro-F1。','',
      '全部原始四组合差值保留在 factorial_raw_overall.csv。以下只统计预登记 raw S6 条件，不根据结果选择默认或重跑：','']
    for k,v in negatives.items():report.append(f"- {k}：{v['nonpositive']}/{v['total']} 个契约×方法×任务均值非正。")
    report+=['',f"普通 PL 的实际接受计数保留，共 {verification['external_zero_PL_acceptance_rows']} 条外部训练接受数为 0；不视为技术失败。",'',
      '## 成本和元数据边界','',
      '600 新 F 共 %.6f 个共享 worker elapsed 小时；27 复用 F 的历史成本 %.6f 小时；4 新 G 的完整八阶段共 %.6f 小时。'
      %(verification['new_F_shared_worker_hours'],verification['reused_F_historical_shared_worker_hours'],verification['new_G_shared_worker_hours']),
      '另有 17 条已知完整历史 G 谱系共 %.6f 小时，原固定 F0 G-LU 的完整祖先耗时未完整序列化，不填造。'
      %(verification['known_reused_G_shared_worker_hours']),
      '3811 个唯一训练增强缓存生产累计 %.6f 小时（每任务 CPU 生产 elapsed，非 CPU 核时）；22 个恢复 IQ 缓存累计 %.6f 小时；'
      '627 个缓存 F 读出累计 %.6f 共享 worker 小时，不能当在线 G+F 延迟。'
      %(verification['augmentation_production_elapsed_hours'],verification['restored_IQ_cache_elapsed_hours'],verification['cached_F_prediction_elapsed_hours']),
      '这些累计值不是独占 GPU 小时、日历耗时或加速倍数。共享缓存只计一次，各 F/G 来源去重，历史成本与本次新增分列。'
      '两条失败 IB 的 52 阶段轮/2600 更新及已知部分耗时约 250.31 秒单列为技术部分成本；其余启动失败的完整耗时未知，不混入有效模型。','',
      'PAMF 原生门控完整接受率、PAMF 每个 F 实际信道施加计数未在此紧凑元数据完整保留，保持缺失；'
      '监督 PAMF 的实际 U 曝光为 0，完整外部方法的增强/PL 计数和全部阶段元数据均保留。训练增强缓存的唯一物理 ID 场景计数是生产量，不冒称每个 F 的消费量。','']
    (public/'full_results.md').write_text('\n'.join(report).rstrip()+'\n',encoding='utf-8')
    with (public/'summary_clean_S6.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(r for r in summary if r['view'] in ('clean','six_practical_mean'))
    for name in ('verification.json','G_cost.csv','restored_cache_cost.csv'):
        shutil.copyfile(analysis/name,public/name)
    cost_summary=[]
    for family,cell in VARIANTS:
        selected=[r for r in cost if r['family']==family and r['cell']==cell]
        for is_new in ('True','False'):
            vals=[r for r in selected if r['new_F']==is_new]
            if vals:cost_summary.append(dict(family=family,cell=cell,new_F=is_new,models=len(vals),
                shared_worker_hours=sum(float(r['shared_worker_seconds']) for r in vals)/3600))
    from experiments.cric_dura_comparison.analyze_main import write_csv
    write_csv(public/'F_cost_summary.csv',cost_summary)
    (public/'effect_coverage.json').write_text(json.dumps(negatives,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
       'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    images=paper/'images/practical_full';images.mkdir(parents=True,exist_ok=True)
    methods=[('riei','D'),('drift','D'),('ibrff','D'),('sdg','D'),('pamf','C'),('pamf','native_T1')]
    labels=['RIEI D','DRIFT D','IB-RFF D','SDG D','PAMF-Sup C','PAMF-T1']
    colors=['#8DA0CB','#A6A6A6','#E3B38C','#84B8A5','#A48ABF','#24558A']
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.55),layout='constrained');ms=[1,2,3,4,6]
    for i,readout in enumerate(('raw','restored')):
        for j,task in enumerate(('cross_rx','cross_both')):
            ax=axes[i,j]
            for (fam,cell),label,color in zip(methods,labels,colors):
                vals=[get(f'm{m}-natural',fam,cell,task,'raw' if readout=='raw' else route(cell)) for m in ms]
                ax.errorbar(ms,[float(v['mean']) for v in vals],yerr=[float(v['sd']) for v in vals],
                    label=label,color=color,marker='o',markersize=2.8,capsize=1.5,linewidth=1.2)
            ax.set_title(('Cross RX' if j==0 else 'Cross date + RX')+' / '+readout)
            ax.set_xticks(ms);ax.set_xlabel('Number of source RX');ax.set_ylabel('S6 accuracy (%)');ax.grid(alpha=.16)
    axes[0,0].legend(ncol=2,loc='lower right')
    for suffix in ('pdf','svg','png'):fig.savefig(images/('source_count.'+suffix),dpi=180)
    svg=images/'source_count.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7.1,4.15),layout='constrained')
    short=['Clean','High sub.','Mid sub.','Low sub.','High urb.','Mid urb.','Low urb.']
    for ax,raw in zip(axes,(True,False)):
        for (fam,cell),label,color in zip(methods,labels,colors):
            vals=[get('F0',fam,cell,'cross_rx','raw' if raw else route(cell),v) for v in VIEWS]
            ax.errorbar(range(7),[float(v['mean']) for v in vals],yerr=[float(v['sd']) for v in vals],
                label=label,color=color,marker='o',markersize=2.5,capsize=1.5,linewidth=1.2)
        ax.set_xticks(range(7),short);ax.set_ylabel('Accuracy (%)');ax.set_title('Raw F' if raw else 'Matched DURA + F');ax.grid(alpha=.16)
    axes[0].legend(ncol=3,loc='lower left')
    for suffix in ('pdf','svg','png'):fig.savefig(images/('F0_seven_views.'+suffix),dpi=180)
    svg=images/'F0_seven_views.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    tex=[r'\subsection{Complete Methods under Practical Residual Channels}',
      r'All 627 registered readouts are complete: 528 newly trained external Full models, 66 PAMF supervised models, six new native-T1 models and 27 exact-contract native-T1 reuses. Each freezes two routes, three disjoint generalization populations and seven observations before independent scoring. The 600 new F models and four new eight-stage G lineages require no further training. We verify 742938 serialized confusion matrices and 371469 net-correction cells, reusing the previously verified F1/F2 overall cells. ManyRx uses ten classes and the other conditions six.',
      r'Table~\ref{tab:residual-f0} retains the complete channel/ordinary-PL four-cell comparison on the F0 cross-RX population. Channel augmentation is not uniformly beneficial, and ordinary PL does not consistently improve a complete baseline. Raw-F and restored performance are therefore reported together. G-L is used by A/C and supervised PAMF, whereas G-LU is used by B/D and native T1; restored PL-group differences include this additional G-data permission. PAMF-T1 retains its own selective self-training and original internal views, so comparison against ordinary-PL baselines concerns complete recipes, not one isolated SSL loss.',
      r'\begin{table*}[!t]\centering\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}',
      r'\caption{F0 cross-RX test (D1--D3; 126000 packets/view). Accuracy and Macro-F1 in percent, mean$\pm$sample SD over three F seeds. A/B/C/D denote channel off/on crossed with PL off/on. No CE-only baseline is included.}\label{tab:residual-f0}',
      r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}\toprule Method & Cell & Raw clean & Raw S6 & Restored clean & Restored S6 & Restored S6 F1\\\midrule']
    for family,cell in VARIANTS:
        vals=[get('F0',family,cell,'cross_rx','raw','clean'),get('F0',family,cell,'cross_rx','raw'),
              get('F0',family,cell,'cross_rx',route(cell),'clean'),get('F0',family,cell,'cross_rx',route(cell)),
              get('F0',family,cell,'cross_rx',route(cell),metric='macro_f1')]
        tex.append(names[family]+' & '+('Native T1' if cell=='native_T1' else cell)+' & '+' & '.join(fmt(r,True) for r in vals)+r'\\')
        if (family,cell) in (('sdg','A'),('sdg','B'),('sdg','C'),('sdg','D')):tex.append(r'\addlinespace[2pt]')
    tex += [r'\bottomrule\end{tabular*}\end{table*}',
      r'\begin{figure*}[!t]\centering\includegraphics[width=\textwidth]{images/practical_full/F0_seven_views.pdf}\caption{F0 cross-RX accuracy on every frozen observation. External D uses the common residual-channel/ordinary-PL recipe; PAMF-Sup C uses its supervised residual-channel recipe and native T1 retains its fixed training method. Raw and restored routes can have different response patterns; error bars are three-F-seed SD.}\label{fig:residual-scenes}\end{figure*}',
      r'\subsubsection{Receiver and Date Generalization}',
      r'The new F1 and F2 source sets each replace four of the five F0 source receivers, and their pairwise source-set overlap is one. Table~\ref{tab:residual-generalization} separates earlier unseen-date evaluation on source RX, unseen RX on training dates, and their conjunction. Dates and receiver populations are not pooled. Native semi-supervision does not outperform the supervised recipe in every cell; all negative differences remain in the archive.',
      r'\begin{table*}[!t]\centering\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}',
      r'\caption{PAMF generalization after matching G (S6 accuracy, \%). Supervised A/C use G-L; native T1 uses G-LU. Each entry is the three-F-seed mean$\pm$SD.}\label{tab:residual-generalization}',
      r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule Split & PAMF recipe & Cross date & Cross RX & Cross date + RX\\\midrule']
    for p,label in [('F0','F0'),('F1_rx_repartition_r01','F1'),('F2_rx_repartition_r01','F2')]:
        for cell,title in [('A','Supervised, channel off'),('C','Supervised, channel on'),('native_T1','Native T1')]:
            tex.append(label+' & '+title+' & '+' & '.join(fmt(get(p,'pamf',cell,t,route(cell)),True) for t in TASKS)+r'\\')
        tex.append(r'\addlinespace[2pt]')
    tex += [r'\bottomrule\end{tabular*}\end{table*}',
      r'\subsubsection{Label Availability, ManyRx and Source Count}',
      r'Table~\ref{tab:residual-data} reports the one/five-percent training-label conditions, the default ten-percent condition and ManyRx. External rows use fixed D rather than selecting a cell from target scores. PAMF-Sup C and native T1 remain separate complete recipes. The source-count curves in Fig.~\ref{fig:residual-source} use m=1,2,3,4,6 with natural source budgets and the same six target RX: 108000 seen-date packets and 36000 D0 packets. No fixed-L budget or m=5 point is added; the seven-target F0 setting is not inserted into these curves.',
      r'\begin{table*}[!t]\centering\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}',
      r'\caption{Cross-RX S6 accuracy (\%) across dataset/label conditions. Columns give each condition separately; classes are six for ManySig and ten for ManyRx.}\label{tab:residual-data}',
      r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}\toprule & \multicolumn{2}{c}{F0, 10\% L} & \multicolumn{2}{c}{F0, 1\% L} & \multicolumn{2}{c}{F0, 5\% L} & \multicolumn{2}{c}{ManyRx}\\',
      r'Method & Raw & G+F & Raw & G+F & Raw & G+F & Raw & G+F\\\midrule']
    for (fam,cell),label in zip(methods,labels):
        vals=[get(p,fam,cell,'cross_rx',r) for p in ('F0','label01','label05','ManyRx') for r in ('raw',route(cell))]
        tex.append(label+' & '+' & '.join(fmt(r,True) for r in vals)+r'\\')
    tex += [r'\bottomrule\end{tabular*}\end{table*}',
      r'\begin{figure*}[!t]\centering\includegraphics[width=.98\textwidth]{images/practical_full/source_count.pdf}\caption{Natural-budget source-count curves, with fixed six-target populations and the same preregistered recipes as Table~\ref{tab:residual-data}. Scenes average within seed before the three-seed SD. Decreases and method-rank changes are retained.}\label{fig:residual-source}\end{figure*}',
      r'The 600 new F models sum to %.2f shared-worker elapsed hours; 27 reused F sources contribute %.2f historical hours. Four new complete G lineages sum to %.2f hours; the 17 other fully timed G lineages sum to %.2f historical hours. Original fixed-F0 G-LU ancestor timings remain incomplete. The 3811 unique CPU augmentation-cache productions sum to %.2f elapsed hours, the 22 restored-IQ caches to %.2f hours and cached F predictions to %.2f shared-worker hours. These are not exclusive GPU hours, CPU core-hours, online G+F latency or an end-to-end speedup. We retain all external augmentation/PL counts, including 61 zero-acceptance training rows. Complete native-PAMF gate acceptance and per-F realized channel counts are not recoverable from the compact telemetry and are left missing.'%(verification['new_F_shared_worker_hours'],verification['reused_F_historical_shared_worker_hours'],verification['new_G_shared_worker_hours'],verification['known_reused_G_shared_worker_hours'],verification['augmentation_production_elapsed_hours'],verification['restored_IQ_cache_elapsed_hours'],verification['cached_F_prediction_elapsed_hours']),
      r'The following PAMF mechanism studies retain their original F0 all-target-date population (168000 packets/view). They are independently frozen historical mechanism evidence, not relabeled as the 126000-packet cross-RX endpoint above. Existing mechanism and timing measurements are reused without retraining or retesting.']
    (paper/'practical_full_results.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='RENDERED',public=str(public),figure_groups=2,summary_rows=len(summary))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--public',required=True);p.add_argument('--paper',required=True)
    a=p.parse_args();render(a.analysis,a.public,a.paper)
