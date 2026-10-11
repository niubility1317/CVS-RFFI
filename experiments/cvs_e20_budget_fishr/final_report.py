"""Full fixed-matrix results; never choose training configurations from scores."""
import argparse
import statistics as st
from pathlib import Path
from . import design as d

def build(result,source):
    if result['status']!='COMPLETE' or result['missing_rows'] or len(result['completed_rows'])!=len(d.rows()):raise ValueError('Incomplete scored matrix')
    if source['status']!='COMPLETE' or len(source['rows'])!=24 or source['total_steps']!=106560 or source['pending']:raise ValueError('Incomplete source logs')
    if any(r['stdout_errors'] for r in source['rows']):raise ValueError('Source stdout has unreviewed errors')
    if not all(result['checks'].values()):raise ValueError('Scoring checks incomplete')
    summaries={r['arm']:r for r in result['summary']}
    records=result['records']+result['parent_records'];overall={(r['arm'],r['model_seed']):r for r in records if r['dimension']=='overall'}
    parents={(r['model_seed'],r['dimension'],r['stratum']):r for r in result['parent_records']}
    for r in result['records']:
        parent=parents[(r['model_seed'],r['dimension'],r['stratum'])]
        if r['parent_fix_count']-r['parent_break_count']!=r['net_correct_gain'] or abs((r['accuracy']-parent['accuracy'])*r['query_count']-r['net_correct_gain'])>1e-6:raise ValueError('Correction accounting differs')
    reached=result['complete_arms_reaching_84']
    lines=['# E20 续训 20 轮：浅中层梯度预算 Fishr 与 U 主路径 MixStyle 完整测试','', '**24/24 行完成，八方案各三种子。** '+('达到 clean 三种子均值 84%：'+', '.join(reached)+'。' if reached else '**本批没有方案达到 clean 三种子均值 84%，目标未完成。**'), '', '每行继承对应种子的原生纯 CE＋旧门控＋cos E20，再追加 20 epoch／4,440 次主更新，累计 40 轮。P0 复用原生 E20 固定预测。CE 是同父权重、同增量主更新预算的续训对照；多层方案另有 200 次辅助头预热，成本单列。研究变体使用合法 U_s，无 EMA、无星地训练增强，固定增量 final20。每行相同 168,000 条 clean query、7 RX、4 day，预测完成后独立 truth-last 评分；三种子为 2026092701/02/03。', '', '|方案|seed2701|seed2702|seed2703|均值|SD（百分点）|Macro-F1 均值|相对 P0（百分点）|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('P0',*d.ARMS):
        summary=summaries[arm]
        if not summary['complete_three_seeds'] or set(summary['seeds'])!=set(d.SEEDS):raise ValueError('Missing seed')
        values=[overall[arm,seed] for seed in d.SEEDS]
        lines.append('|'+arm+'|'+'|'.join(f"{r['accuracy']*100:.4f}%" for r in values)+f"|{summary['accuracy_mean']*100:.4f}%|{summary['accuracy_sd']*100:.4f}|{st.mean(r['macro_f1'] for r in values)*100:.4f}%|{(summary['accuracy_mean']-summaries['P0']['accuracy_mean'])*100:+.4f}|")
    contrasts=[]
    for a,b in d.PAIRS:
        diff=[overall[a,s]['accuracy']-overall[b,s]['accuracy'] for s in d.SEEDS]
        contrasts.append(dict(name=a+'−'+b,per_seed=diff,mean=st.mean(diff),sd=st.stdev(diff),positive_seeds=sum(x>0 for x in diff)))
    for name,arms in [('U MixStyle×Budget Fishr',['JOINT','MIX_U','FISHR_BUDGET','MULTI'])]:
        diff=[overall[arms[0],s]['accuracy']-overall[arms[1],s]['accuracy']-overall[arms[2],s]['accuracy']+overall[arms[3],s]['accuracy'] for s in d.SEEDS]
        contrasts.append(dict(name=name,per_seed=diff,mean=st.mean(diff),sd=st.stdev(diff),positive_seeds=sum(x>0 for x in diff)))
    lines+=['','## 同种子配对增量','', '|比较|seed2701|seed2702|seed2703|均值（百分点）|SD（百分点）|正收益种子|','|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append('|'+r['name']+'|'+'|'.join(f'{x*100:+.4f}' for x in r['per_seed'])+f"|{r['mean']*100:+.4f}|{r['sd']*100:.4f}|{r['positive_seeds']}/3|")
    lines+=['','主对照 JOINT−MULTI 描述 U 主路径风格扰动与梯度预算约束相对于共同辅助 CE 的增量；MULTI−SSL 描述辅助监督，MIX_U−MIX_L 描述作用范围及所配目标的差异，FISHR_BUDGET−FISHR_RAW 描述系数控制的增量。交互量为联合−两个单项＋共同对照。三种子是描述性结果，不自动构成显著性证明。','', '## 相对原生 E20 的纠错与破坏','', '|方案|seed|修复原错误|破坏原正确|净变化|','|---|---:|---:|---:|---:|']
    for r in result['records']:
        if r['dimension']=='overall':lines.append(f"|{r['arm']}|{r['model_seed']}|{r['parent_fix_count']}|{r['parent_break_count']}|{r['net_correct_gain']:+d}|")
    for dimension,label,records in [('receiver','接收机',result['records']),('transmitter','发射机',result['records']),('day','日期',result['day_records'])]:
        lines+=['','## '+label+'分层：三种子均值','', '|'+label+'|'+'|'.join(d.ARMS)+'|','|---|'+'|'.join(['---:']*len(d.ARMS))+'|']
        for stratum in sorted({r['stratum'] for r in records if r['dimension']==dimension}):
            cells=[]
            for arm in d.ARMS:
                subset=[r for r in records if r['dimension']==dimension and r['stratum']==stratum and r['arm']==arm]
                if len(subset)!=3 or {r['model_seed'] for r in subset}!=set(d.SEEDS):raise ValueError('Missing stratum seeds')
                cells.append(f"{st.mean(r['accuracy'] for r in subset)*100:.4f}%")
            lines.append('|'+str(stratum)+'|'+'|'.join(cells)+'|')
    lines+=['', '全部逐 seed 的 RX/TX/day、混淆矩阵、Macro-F1、NLL 及纠错计数保存在 evidence/test_results_analysis.json，不用汇总值代替原始记录。','', '## 实测成本','', '|方案|训练及源诊断（分钟）|clean 推理（秒）|峰值已分配显存（MiB）|身份／辅助可训练参数|额外 L 使用|辅助前向|暖头秒数／更新|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        subset=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
        if len(subset)!=3:raise ValueError('Missing resource seeds')
        res=[r['source']['measured_training_resources'] for r in subset];p=res[0]
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in res)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in subset):.2f}|{max(r['peak_cuda_memory_bytes'] for r in res)/1024**2:.2f}|{p['trainable_parameters']-p['auxiliary_trainable_parameters']}/{p['auxiliary_trainable_parameters']}|{p['auxiliary_L']}|{p['auxiliary_forwards']}|{st.mean(r['head_warmup']['seconds'] for r in res):.2f}/{p['head_warmup']['optimizer_steps']}|")
    lines+=['', '硬件 RTX 3090，共享 GPU 墙钟；主更新都是增量 20 epoch，不等于相同总算力。上表主训练时间不含头预热；父 E20 成本复用，未计入本次增量。多层方案额外提取 6300 个 L 的冻结特征并进行 12000 次头训练样本使用；表中额外 L 59940 仅指后续辅助训练。Fishr 解析计算线性辅助头的逐样本 weight/bias 梯度，无逐样本骨干反向循环。两辅助头在预测中不执行；保存并驻留的 876 参数计入资源，不声称已剔除。训练 FLOPs 未测，N/A。','', '## 证据与边界','', '完整核验 106,560 个 step、480 个 epoch、完整源训练 stdout 段，无遗漏种子。浅中层位置为 time_down/t2.pool，MIX_L 使用合法同 TX、同 day、跨 RX 的 L；MIX_U/JOINT 使用双方置信度达标的同伪类别、同 day、跨 RX、不同物理 PID 的 U 主路径。Fishr 仅 clean L 辅助图，正则中头参数停止梯度，辅助 CE 单独训练头；梯度预算在均值前后和合并后分别记录。无 U 身份真值、目标训练或反馈。具体源损失、梯度、层统计及权重位移见 source_trajectory.md 和 evidence/source_full_analysis.json。', '', '测试基准此前已经暴露，本批属于固定方案探索；不能称为全新盲测。所有结果保留，不从本批测试排名回流调参、改变队列或选择性重跑。84% 按某一完整固定方案三种子均值验收，不拼接最佳单 seed。']
    return '\n'.join(lines)+'\n',contrasts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=d.read(a.output/'evidence/test_results_analysis.json');source=d.read(a.output/'evidence/source_full_analysis.json');text,pairs=build(result,source)
    (a.output/'final_analysis.md').write_text(text,encoding='utf-8');d.write(a.output/'evidence/full_paired_analysis.json',dict(contrasts=pairs,target_feedback=False))
    print('VERIFIED 24 rows / 106560 source steps / full stratified report')
