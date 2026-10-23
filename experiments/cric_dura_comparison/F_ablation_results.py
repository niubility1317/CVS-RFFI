"""Public diagnostic tables, the matched F16 factorial, and appendix figures."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

from .analyze_F_ablation import ARMS

DESCRIPTIONS = {
    'F01': 'Shallow-only cross-RX contrast', 'F02': 'Intermediate-only cross-RX contrast',
    'F03': 'Final160-only contrast', 'F04': 'No ordinary/raw paired KL',
    'F05': 'One forward per pair; CE dose retained', 'F06': 'One SIG_TIME view',
    'F07': 'No signature time coordinates', 'F08': 'No signature branch',
    'F09': 'No PA branch', 'F10': 'No reference-response branch',
    'F11': 'Current student instead of EMA PL source', 'F12': 'No adjacent-batch agreement gate',
    'F13': 'No strong-view agreement gate', 'F14': 'No U entropy term',
    'F15': 'CE + default sat.68; LS0; no SSL/CRIC/KL',
    'F16': 'CRIC weight zero; other T1 terms retained',
    'F17': 'As64 replaced by raw64; two CE/KL0 slots',
    'F18': 'As KL restored to one', 'F19': 'No raw pair; CE dose and compute reduced',
    'F20': 'No U/EMA/entropy; T1 supervision retained',
    'F21': 'Ordinary PL.95 replacing native SSL; U256',
}


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def run(source, output, paper=None):
    source, output = Path(source), Path(output)
    verified = json.loads((source / 'F_ablation_verification.json').read_text(encoding='utf-8'))
    if verified['status'] != 'ANALYZED_CORRECTED_FINE63' or verified['corrected_F_models'] != 63:
        raise ValueError('Only the whole corrected fine F panel may be reported')
    output.mkdir(parents=True, exist_ok=True)
    names = ('F_ablation_summary_overall.csv', 'F_ablation_paired_overall.csv',
             'F_ablation_factorial_summary_overall.csv', 'F_ablation_F_cost.csv',
             'F_ablation_F_stage_cost.csv', 'F_ablation_verification.json')
    for name in names:
        shutil.copyfile(source / name, output / name)
    summary = rows(source / 'F_ablation_summary_overall.csv')
    paired = rows(source / 'F_ablation_paired_overall.csv')
    factorial = rows(source / 'F_ablation_factorial_summary_overall.csv')
    cost = rows(source / 'F_ablation_F_cost.csv')
    values = {(r['arm'], r['readout'], r['view'], r['metric']): r for r in summary}
    effects = {(r['contrast'], r['arm'], r['readout'], r['view'], r['metric']): r for r in paired}
    factors = {(r['effect'], r['view'], r['metric']): r for r in factorial}
    def fmt(r, latex=False):
        if latex:
            return r'$%.4f\pm%.4f$' % (float(r['mean']), float(r['sd']))
        return '%.4f ± %.4f' % (float(r['mean']), float(r['sd']))
    def f(a, g, v, latex=False):
        return fmt(values[a, g, v, 'accuracy'], latex)
    report = ['# 修复后 F 细粒度诊断：21 设置 × 3 F seed 全部完成', '',
        '这是附录诊断，主文仍以既定 A/B/C 三功能组为主。只使用 corrected binding-m63 的 63 条结果，最终 T1 三参照直接复用；不使用旧 INVALID_INPUT_BINDING 分数或历史 BASE/ALL_TX factorial。全部七景预测冻结后独立 truth-last，同一 F0 七目标 RX、每景 168000 包。各设置共用同一原始 G-LU，三 F seed 变化不代表三独立完整系统。', '',
        '|设置|定义|raw clean %|raw 六景 %|G-LU clean %|G-LU 六景 %|',
        '|---|---|---:|---:|---:|---:|']
    report.append('|T1|固定完整配方|%s|%s|%s|%s|' %
                  (f('T1','raw','clean'),f('T1','raw','six_practical_mean'),f('T1','DURA-LU','clean'),f('T1','DURA-LU','six_practical_mean')))
    for a in ARMS:
        report.append('|%s|%s|%s|%s|%s|%s|' % (a, DESCRIPTIONS[a], f(a,'raw','clean'),f(a,'raw','six_practical_mean'),f(a,'DURA-LU','clean'),f(a,'DURA-LU','six_practical_mean')))
    report += ['', '配对 CSV 的 T1_minus_variant 统一取“完整配方减变体”，正值有利于完整配方，负值完整保留。F15 是含增强、损失和前向量差异的整体参考；F05 改变随机前向和计算，F19 同时减少 CE 剂量和计算，不能按一个孤立正则项解释。F20 仅 F 不用 U，接 G-LU 的系统仍通过 G 使用 U；严格全 L 读出已在独立六行面板报告。', '',
        '## 匹配最终 T1 的 CRIC×G 2×2', '',
        '仅比较当前完整 T1 与修复 F16，读出为 raw/G-LU。先每 seed 内六景平均，再 seed 配对；交互为 (T1_G−T1_raw)−(F16_G−F16_raw)。', '',
        '|效应|clean pp|六景 pp|', '|---|---:|---:|']
    for name in ('CRIC_at_raw','CRIC_at_G','G_at_T1','G_at_F16','CRIC_x_G'):
        report.append('|%s|%s|%s|' % (name, fmt(factors[name,'clean','accuracy']),fmt(factors[name,'six_practical_mean','accuracy'])))
    report += ['', 'DURA 对完整 T1 与 F16 的六景提升都为正；CRIC 完整条件和交互不支持稳定正增益。样本 SD 不是显著性检验，不据这些目标结果更换默认、调参、重排或选择性重跑。', '',
        '## 曝光与成本', '',
        '63 条修复 F 合计 3150 模型 epoch、699300 更新、189 段，累计 %.6f 共享 worker 小时。原三 T1 参照另复用 %.6f 小时；不是本轮追加墙钟或独占 GPU 小时。F15/F20 和 core 已复用其中权重，跨报告不能重复相加成本。G 祖先、观察/IQ 缓存与正式机制时延分列。' %
        (verified['corrected_F_shared_worker_seconds']/3600,verified['reused_T1_shared_worker_seconds']/3600),
        '所有新 F 均 50 轮/11100 更新。F05/F15 每更新 384 个 L-packet 前向，F19 为 512，其余为 768。F15/F20 实际 U 曝光为 0；其余 F0 原日程实际 U 曝光各2097900（包含末批，不用名义整批量冒充实际量）。',
        'F21 普通 PL 的 seen/accepted/rate 在成本 CSV 中逐 seed 保留；这些计数器不覆盖原生门控 SSL，不能把其他行的普通 PL 专用零计数误当作原生接受率 0。原生完整接受率未在此次取回的紧凑阶段元数据中保存，明确留空。']
    rates = [float(r['ordinary_PL_acceptance'])*100 for r in cost if r['arm']=='F21']
    report += ['F21 三 seed 普通 PL 接受率均值 %.4f%%，样本 SD %.4f 个百分点（%.4f%% 至 %.4f%%）；不是所有其他方法的 PL 接受率。' %
               (statistics.mean(rates),statistics.stdev(rates),min(rates),max(rates)), '',
        '42336 份修复行 CM、2016 份固定 T1 参照 CM 以及22176个净纠错分层已核验。完整 CSV 保留每景、每 RX/TX/day、Accuracy/Macro-F1、配对差、净纠错和全部负结果。机制实测时延仍由原专用队列完成，不能从此准确率或参数推断速度。']
    (output/'F_ablation_results.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    if paper is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paper=Path(paper);figures=paper/'images/F_ablation63';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,5.5),sharey=True,constrained_layout=True)
    for ax,view,title in zip(axes,('clean','six_practical_mean'),('Clean','Six-scene mean')):
        for reader,color,offset,mark in (('raw','#50545a',-.12,'o'),('DURA-LU','#176b93',.12,'s')):
            r=[effects['T1_minus_variant',a,reader,view,'accuracy'] for a in ARMS]
            ax.errorbar([float(x['mean']) for x in r],[i+offset for i in range(21)],
                xerr=[float(x['sd']) for x in r],fmt=mark,color=color,markersize=3,capsize=2,linewidth=.8,label=reader)
        ax.axvline(0,color='#999999',linewidth=.8);ax.set_title(title)
        ax.set_yticks(range(21),ARMS);ax.set_ylim(20.6,-.6);ax.set_xlabel('T1 minus variant (pp)')
        ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,fontsize=7)
    fig.savefig(figures/'F_ablation_paired.pdf');fig.savefig(figures/'F_ablation_paired.png',dpi=220);plt.close(fig)
    primary=[r'\paragraph{Completed Matched T1/F16 Readouts}',
        r'The corrected F16 and the fixed final T1 now supply all three paired F seeds for the registered CRIC by restoration comparison. Both use the final 20+10+20 trajectory and the same original G-LU, while F16 alone sets CRIC weight to zero. The historical BASE/ALL\_TX panel is excluded.',
        r'\begin{table}[t]',r'\centering\scriptsize',
        r'\caption{Matched final T1/F16 accuracy (\%, three F seeds, one fixed G-LU).}',r'\label{tab:F16-factorial}',
        r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llrr}',r'\toprule',
        r'F & Readout & Clean & Six views \\',r'\midrule']
    for arm in ('F16','T1'):
        for g in ('raw','DURA-LU'):
            primary.append('%s & %s & %s & %s \\\\'%(arm,g,f(arm,g,'clean',True),f(arm,g,'six_practical_mean',True)))
    primary += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
        'The six-view G gains are %s points at T1 and %s at F16. The matched interaction is %s points (clean: %s). CRIC-at-G is %s points for the six-view mean, positive in only one of three paired seeds. These data do not establish stable positive CRIC/G synergy. Sample SD is not a significance test, and the fixed default is unchanged.' %
        tuple(fmt(factors[n,v,'accuracy'],True) for n,v in [('G_at_T1','six_practical_mean'),('G_at_F16','six_practical_mean'),('CRIC_x_G','six_practical_mean'),('CRIC_x_G','clean'),('CRIC_at_G','six_practical_mean')])]
    (paper/'F16_factorial_results.tex').write_text('\n\n'.join(primary)+'\n',encoding='utf-8')
    appendix=[r'\section{Supplementary Identification Diagnostics}',r'\label{app:fine-F}',
        'The earlier fine-grained panel is retained as an appendix; the three functional groups remain the primary mechanism study. All 21 settings have three actual F seeds, corrected physical-ID IQ binding, fixed default classifiers and seven-view truth-last scores. No old input-invalid scores enter this analysis. All arms share one original F0 G-LU. Their SD is conditional F-seed variation, not independent full-system G variation.',
        r'\begin{table*}[t]',r'\centering\scriptsize',
        r'\caption{Completed corrected F diagnostics (accuracy \%, mean$\pm$sample SD). The first row is the reused fixed final T1; every negative contrast remains in the full tables.}',
        r'\label{tab:fine-F-complete}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lp{0.28\textwidth}rrrr}',r'\toprule',
        r'ID & Definition & Raw clean & Raw six views & G-LU clean & G-LU six views \\',r'\midrule']
    for arm in ('T1',)+ARMS:
        desc='Fixed complete final recipe' if arm=='T1' else DESCRIPTIONS[arm]
        appendix.append('%s & %s & %s & %s & %s & %s \\\\'%(arm,desc.replace('_',r'\_'),f(arm,'raw','clean',True),f(arm,'raw','six_practical_mean',True),f(arm,'DURA-LU','clean',True),f(arm,'DURA-LU','six_practical_mean',True)))
    appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table*}',
        'The diagnostics do not support a claim that every retained element improves every endpoint. F15 is a whole CE-plus-satellite recipe reference with different augmentation, smoothing and exposure. F05 changes stochastic forwards and compute while retaining CE dose; F19 changes both CE dose and compute. F20 removes U from F, but its G-LU readout still uses U through restoration. The separate G-L readouts establish the fully labeled-only boundary.',
        'F17 replaces only the As64 slots by raw64 while retaining their two CE/KL0 slots. Its raw six-view mean is %s versus %s for T1, whereas its restored mean is %s. The readout boundary can therefore mask a large raw-view sensitivity; the complete raw and restored tables are retained without target-driven selection.' %
        (f('F17','raw','six_practical_mean',True),f('T1','raw','six_practical_mean',True),f('F17','DURA-LU','six_practical_mean',True)),
        'The 63 corrected F runs total 3150 model epochs, 699300 updates and 189 source stages, or %.6f shared-worker hours. The three original final-T1 references reuse %.6f historical hours. These are neither incremental deployment latency nor exclusive GPU hours. F15/F20 and the primary factorial reuse some of these sources; panel totals must not be added as disjoint costs.' %
        (verified['corrected_F_shared_worker_seconds']/3600,verified['reused_T1_shared_worker_seconds']/3600),
        r'Every F uses 11100 updates, but F05/F15 evaluate 384 labeled packets per update, F19 evaluates 512, and the other variants evaluate 768. F15/F20 consume zero U; the native-schedule F0 variants each expose 2097900 U packets including final partial batches. F21 ordinary-PL acceptance is %.4f$\pm$%.4f\%% across its three F seeds. Compact native SSL telemetry does not retain a comparable complete acceptance counter; ordinary-PL-only zeros are not interpreted as zero native acceptance.' % (statistics.mean(rates),statistics.stdev(rates)),
        r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{images/F_ablation63/F_ablation_paired.pdf}',
        r'\caption{Fixed-order diagnostic contrasts, full T1 minus each variant (mean$\pm$sample SD of seed-paired differences). Positive values favor T1; all negative values and zero are retained. F15 is a whole-recipe comparator; F05 and F19 also change compute. No arm or endpoint selects a replacement default.}',
        r'\label{fig:fine-F-paired}',r'\end{figure*}',
        'We recomputed 42336 corrected-row confusion matrices, checked 2016 reused T1 reference matrices, and verified 22176 paired correction strata against CM nets and legal joint-count bounds. Full CSVs retain each scene, receiver, transmitter and date, Accuracy/Macro-F1, all paired differences and training cost. Dedicated mechanism timings remain a separate measurement panel; parameter counts and these target accuracies are not speed measurements.']
    (paper/'F_ablation_results.tex').write_text('\n\n'.join(appendix)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--paper')
    a=p.parse_args();run(a.input,a.output,a.paper)
