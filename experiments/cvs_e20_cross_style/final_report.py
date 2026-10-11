"""Full fixed-matrix results; never choose training configurations from scores."""
import argparse
import statistics as st
from pathlib import Path
from . import design as d

def build(result,source):
    if result['status']!='COMPLETE' or result['missing_rows'] or len(result['completed_rows'])!=len(d.rows()):raise ValueError('Incomplete scored matrix')
    expected_ids={r['row_id'] for r in d.rows()}
    if (source['status']!='COMPLETE' or len(source['rows'])!=len(expected_ids)
            or {r['row_id'] for r in source['rows']}!=expected_ids
            or set(result['completed_rows'])!=expected_ids
            or source['total_steps']!=4440*len(expected_ids) or source['pending'] or source['errors']):
        raise ValueError('Incomplete source logs')
    if any(r['stdout_errors'] for r in source['rows']):raise ValueError('Source stdout has unreviewed errors')
    if not all(result['checks'].values()):raise ValueError('Scoring checks incomplete')
    summaries={r['arm']:r for r in result['summary']}
    records=result['records']+result['parent_records'];overall={(r['arm'],r['model_seed']):r for r in records if r['dimension']=='overall'}
    parents={(r['model_seed'],r['dimension'],r['stratum']):r for r in result['parent_records']}
    for r in result['records']:
        parent=parents[(r['model_seed'],r['dimension'],r['stratum'])]
        if r['parent_fix_count']-r['parent_break_count']!=r['net_correct_gain'] or abs((r['accuracy']-parent['accuracy'])*r['query_count']-r['net_correct_gain'])>1e-6:raise ValueError('Correction accounting differs')
    reached=result['complete_arms_reaching_84']
    primary=summaries[d.PRIMARY]
    primary_success=primary['complete_three_seeds'] and primary['accuracy_mean']>=.84
    lines=['# E20 续训 20 轮：独立批 Fishr 与 L 主路径 MixStyle 完整测试','',
        f"**{len(expected_ids)}/{len(expected_ids)} 行完成，{len(d.ARMS)} 方案各三种子。** "+
        (f'主方案 {d.PRIMARY} 达到 clean 三种子均值 84%。' if primary_success else f'**主方案 {d.PRIMARY} 未达到 clean 三种子均值 84%，预登记目标未完成。**'),
        '所有完整固定方案中达到 84% 的方案：'+(', '.join(reached) if reached else '无')+'。', '',
        '每行继承对应种子的原生纯 CE＋旧门控＋cos E20，再追加 20 epoch／4,440 次主更新，累计 40 轮。P0 复用固定原 E20 预测。CE 是同父权重、同增量预算对照；其他方案使用另外两个原 E20 的固定概率均值与 ALL soft KL，无 EMA、星地增强或 U 风格混合。固定增量 final20 后逐行对相同 168,000 条 clean query、7 RX、4 day 预测，再由独立 scorer 连接 truth；三种子为 2026092701/02/03。', '',
        '|方案|seed2701|seed2702|seed2703|均值|SD（百分点）|Macro-F1 均值|相对 P0（百分点）|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('P0',*d.ARMS):
        summary=summaries[arm]
        if not summary['complete_three_seeds'] or set(summary['seeds'])!=set(d.SEEDS):raise ValueError('Missing seed')
        values=[overall[arm,seed] for seed in d.SEEDS]
        lines.append('|'+arm+'|'+'|'.join(f"{r['accuracy']*100:.4f}%" for r in values)+f"|{summary['accuracy_mean']*100:.4f}%|{summary['accuracy_sd']*100:.4f}|{st.mean(r['macro_f1'] for r in values)*100:.4f}%|{(summary['accuracy_mean']-summaries['P0']['accuracy_mean'])*100:+.4f}|")
    contrasts=[]
    for a,b in d.PAIRS:
        diff=[overall[a,s]['accuracy']-overall[b,s]['accuracy'] for s in d.SEEDS]
        contrasts.append(dict(name=a+'−'+b,per_seed=diff,mean=st.mean(diff),sd=st.stdev(diff),positive_seeds=sum(x>0 for x in diff)))
    for name,arms in [('L 风格路径×独立批 Fishr',['JOINT','STYLE_BOTH','F_BOTH','SSL'])]:
        diff=[overall[arms[0],s]['accuracy']-overall[arms[1],s]['accuracy']-overall[arms[2],s]['accuracy']+overall[arms[3],s]['accuracy'] for s in d.SEEDS]
        contrasts.append(dict(name=name,per_seed=diff,mean=st.mean(diff),sd=st.stdev(diff),positive_seeds=sum(x>0 for x in diff)))
    lines+=['','## 同种子配对增量','', '|比较|seed2701|seed2702|seed2703|均值（百分点）|SD（百分点）|正收益种子|','|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append('|'+r['name']+'|'+'|'.join(f'{x*100:+.4f}' for x in r['per_seed'])+f"|{r['mean']*100:+.4f}|{r['sd']*100:.4f}|{r['positive_seeds']}/3|")
    lines+=['','F_BOTH−OLD_PAIR 在相同两批 L60 和固定校准下比较估计量；F_BOTH 相对单层项比较层组合。L_REPLAY 具有相同额外 L 身份 CE 和 RNG 隔离而无风格 hook，各 STYLE−L_REPLAY 用于区分额外监督与层扰动。JOINT 相对 F_BOTH、STYLE_BOTH 比较增量；交互量为 JOINT−STYLE_BOTH−F_BOTH＋SSL，包含完整风格路径，不能单独称为纯层扰动交互。三种子为描述性结果，不自动构成显著性证明。','', '## 相对原生 E20 的纠错与破坏','', '|方案|seed|修复原错误|破坏原正确|净变化|','|---|---:|---:|---:|---:|']
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
    lines+=['', '全部逐 seed 的 RX/TX/day、混淆矩阵、Macro-F1、NLL 及纠错计数保存在 evidence/test_results_analysis.json，不用汇总值代替原始记录。','', '## 实测成本','', '|方案|训练及源诊断（分钟）|clean 推理（秒）|峰值已分配显存（MiB）|身份／辅助可训练参数|辅助 L／前向|额外风格或 replay L|暖头秒／更新|校准秒／L 读出|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        subset=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
        if len(subset)!=3:raise ValueError('Missing resource seeds')
        res=[r['source']['measured_training_resources'] for r in subset];p=res[0]
        peaks=[r['peak_cuda_memory_bytes'] for r in res if r['peak_cuda_memory_bytes'] is not None]
        peak=f'{max(peaks)/1024**2:.2f}' if peaks else 'N/A'
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in res)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in subset):.2f}|{peak}|{p['trainable_parameters']-p['auxiliary_trainable_parameters']}/{p['auxiliary_trainable_parameters']}|{p['auxiliary_L']}/{p['auxiliary_forwards']}|{p['style_L_packets']}|{st.mean(r['head_warmup']['seconds'] for r in res):.2f}/{p['head_warmup']['optimizer_steps']}|{st.mean(r['fishr_calibration']['seconds'] for r in res):.2f}/{p['fishr_calibration']['extra_source_readouts']}|")
    lines+=['', '成本为实际共享 GPU 墙钟，主更新相同不等于总计算相同。主训练时间不含初始暖头和校准。Fishr 行额外提取 L6300 冻结特征，完成 200 次头更新／12,000 次头样本使用；之后校准 12 对 L60，1,440 次模型样本读出、24 次前向、零更新。主训练辅助 999 次，每次两批 L60，共 119,880 次样本使用和 1,998 次前向。风格／replay 路径每行另读出 L511,488 次。CE、SSL、L_REPLAY 和 STYLE-only 不创建辅助头。辅助头在预测中不执行；保存在权重中的状态仍计入成本。FLOPs 未测，N/A。']
    lines+=['','## 源机制核验','', '|方案／种子|层|原始 Fishr 均值|负值次数／计算次数|固定校准系数|','|---|---|---:|---:|---:|']
    for row in source['rows']:
        for level,entries in row['layer_summary'].items():
            raw=entries.get('raw_fishr')
            if raw:lines.append(f"|{row['row_id']}|{level}|{raw['mean']:.8g}|{raw['negative']}/{raw['observations']}|{row['calibration']['fixed_coefficients'][level]:.8g}|")
    lines+=['','独立批内积允许负值，保留其频次与幅度；不把负值直接解释为性能提升。OLD_PAIR 使用两批旧平方项均值。全部 Fishr 层系数只在初始 L 校准确定，训练中乘 ramp，选定层之和固定除以 2，未启用层为零。两批各自 60 个不同 PID，批间允许重叠。','', '|方案／种子|额外 L 前向数|clean／额外路径准确率|预测改判比例|clean／额外路径 margin|','|---|---:|---:|---:|---:|']
    for row in source['rows']:
        s=row['style_identity']
        if s:lines.append(f"|{row['row_id']}|{s['style_ce']['observations']}|{s['L_accuracy']['mean']:.4%}／{s['style_L_accuracy']['mean']:.4%}|{s['style_prediction_change']['mean']:.4%}|{s['clean_identity_margin']['mean']:.5f}／{s['styled_identity_margin']['mean']:.5f}|")
    lines+=['', '上述训练 logits 来自普通 dropout 下的不同前向，差值并非隔离的风格因果效应。L_REPLAY 无 hook；STYLE/JOINT 的实际层覆盖、合法 donor、特征变化及逐 epoch 身份指标在 source_trajectory.md 和 evidence/source_full_analysis.json 中保留。供体只用合法 L 真标签，同 TX/day、跨 RX、不同物理 PID，不读取 U 真值。', '', '## 证据与边界','',
        f"完整核验 {source['total_steps']:,} 个 step、{20*len(expected_ids)} 个 epoch、所有暖头与校准记录及完整源训练 stdout 段，无遗漏种子。浅中层位置为 time_down/t2.pool。辅助 CE 仅更新头，Fishr 中头停止梯度而身份特征可导；校准不更新身份或头。U/V 权限、无 EMA 和无目标反馈保持。", '',
        '测试基准此前已经暴露，本批属于固定方案探索；不能称为全新盲测。所有结果保留，不从测试排名回流调参、改变队列或选择性重跑。84% 按完整固定方案三种子均值验收，不拼接最佳单 seed。']
    cache=result.get('teacher_cache')
    if not cache or cache.get('status')!='COMPLETE' or cache.get('packet_occurrences')!=170100:
        raise ValueError('Missing complete teacher cache resource evidence')
    lines+=['','## 独立教师额外成本','',
        f"三份原 E20 在 U56,700 上各前向一次，合计 {cache['packet_occurrences']} 次样本模型读出，缓存生成 {cache['total_seconds']:.2f} 秒，输出 {cache['output_bytes']} 字节。只共享原生 float32 概率，不包含 U 真值、V 或目标样本。",'',
        '每个非 CE 学生额外依赖另外两个已训练 E20：历史模型训练总量为自身20＋两教师40＋学生续训20＝80模型轮，CE 为40。教师训练已经完成并复用，历史成本不为零。共享三模型缓存的增量成本单列，不能按每行重复计完整生成。预测只执行学生模型，教师不参与目标预测。', '',
        '固定教师的源 V 纠错质量复用本次设计前完成的 E20 peer source audit；训练中 V 仅 eval/no_grad 健康诊断。U 教师与学生分歧、同判/分歧 KL 和梯度见 source_trajectory.md，不把分歧称为真实纠错。']
    return '\n'.join(lines)+'\n',contrasts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=d.read(a.output/'evidence/test_results_analysis.json');source=d.read(a.output/'evidence/source_full_analysis.json');text,pairs=build(result,source)
    (a.output/'final_analysis.md').write_text(text,encoding='utf-8');d.write(a.output/'evidence/full_paired_analysis.json',dict(contrasts=pairs,target_feedback=False))
    print(f'VERIFIED {len(d.rows())} rows / {4440*len(d.rows())} source steps / full stratified report')
