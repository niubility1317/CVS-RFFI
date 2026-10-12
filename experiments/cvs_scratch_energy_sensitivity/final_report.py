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
    lines=['# 学生从零训练 20 轮：主身份损失幅度敏感度与困难风格完整测试','',
        f"**{len(expected_ids)}/{len(expected_ids)} 行完成，{len(d.ARMS)} 方案各三种子。** "+
        (f'主方案 {d.PRIMARY} 达到 clean 三种子均值 84%。' if primary_success else f'**主方案 {d.PRIMARY} 未达到 clean 三种子均值 84%，预登记目标未完成。**'),
        '所有完整固定方案中达到 84% 的方案：'+(', '.join(reached) if reached else '无')+'。', '',
        '每个学生身份网络均从随机权重开始，保留原生旧门控、参考响应与分类头，采用 cosine 学习率日程；从零训练 20 epoch／4,440 次更新，累计 20 轮，parent_checkpoint=None。固定历史主方案 DELAYCAP_JOINT 为归一化参考；本批比较幅度保留的 margin 与 CE 敏感度。全部方案仍使用另外两个预训练原 E20 的固定概率均值与 ALL soft KL，无 EMA、星地增强或 U 风格混合。固定 final20 后逐行对相同 168,000 条 clean query、7 RX、4 day 预测，再由独立 scorer 连接 truth；6 行共 1,008,000 次 clean 决策，三种子为 2026092701/02/03。', '',
        'P0 仅复用固定原 E20 预测作为评分参考，其权重没有加载到学生。学生随机初始化不代表整个方法不含预训练：源 U 教师仍是已训练 E20。固定跨批对照另见 reference_comparisons.md；只复用相同学生20轮和既定教师的前批产物，不重跑或选种子。','',
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
    lines+=['','## 同种子配对增量','', '|比较|seed2701|seed2702|seed2703|均值（百分点）|SD（百分点）|正收益种子|','|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append('|'+r['name']+'|'+'|'.join(f'{x*100:+.4f}' for x in r['per_seed'])+f"|{r['mean']*100:+.4f}|{r['sd']*100:.4f}|{r['positive_seeds']}/3|")
    lines+=['','两方案均含同一浅中层Fishr类统计和MixStyle，比较身份目标与幅度保留。三种子是描述性结果。','', '## 相对固定 P0 评分参考的纠错与破坏','', '下列计数比较两份独立预测；P0 不是学生父权重，不能解释为继承后修复或遗忘。','', '|方案|seed|P0 错而当前对|P0 对而当前错|净变化|','|---|---:|---:|---:|---:|']
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
    lines+=['', '全部逐 seed 的 RX/TX/day、混淆矩阵、Macro-F1、NLL 及纠错计数保存在 evidence/test_results_analysis.json，不用汇总值代替原始记录。','', '## 实测成本','', '|方案|训练及源诊断（分钟）|clean 推理（秒）|峰值已分配显存（MiB）|身份／辅助可训练参数|辅助 L／前向|额外有梯度风格或 replay L|额外搜索 eval 前向／L|暖头秒／更新|校准秒／L 读出|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        subset=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
        if len(subset)!=3:raise ValueError('Missing resource seeds')
        res=[r['source']['measured_training_resources'] for r in subset];p=res[0]
        peaks=[r['peak_cuda_memory_bytes'] for r in res if r['peak_cuda_memory_bytes'] is not None]
        peak=f'{max(peaks)/1024**2:.2f}' if peaks else 'N/A'
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in res)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in subset):.2f}|{peak}|{p['trainable_parameters']-p['auxiliary_trainable_parameters']}/{p['auxiliary_trainable_parameters']}|{p['auxiliary_L']}/{p['auxiliary_forwards']}|{p['style_L_packets']}|{p['style_search_forwards']}/{p['style_search_L']}|{st.mean(r['head_warmup']['seconds'] for r in res):.2f}/{p['head_warmup']['optimizer_steps']}|{st.mean(r['fishr_calibration']['seconds'] for r in res):.2f}/{p['fishr_calibration']['extra_source_readouts']}|")
    lines+=['', '实际共享 GPU 墙钟包含并发影响。每行零辅助参数；在444次更新后，两遍使用相同12对L60，共2880次样本读出、48次前向：第一遍固定每层RMS，第二遍固定损失系数。主训练字段扣除校准耗时，含延后校准的原始墙钟另存；校准单列避免重复计费。所有行999次敏感度辅助，每次两批L60。两方案另有3996次风格训练前向和19980次候选eval前向。两方案每辅助步额外两次联合梯度求导；FLOPs未测。']
    lines+=['','## 源机制核验','', '|方案／种子|层|原始 Fishr 均值|负值次数／计算次数|固定校准系数|','|---|---|---:|---:|---:|']
    for row in source['rows']:
        for level,entries in row['layer_summary'].items():
            raw=entries.get('raw_fishr')
            if raw:lines.append(f"|{row['row_id']}|{level}|{raw['mean']:.8g}|{raw['negative']}/{raw['observations']}|{row['calibration']['fixed_coefficients'][level]:.8g}|")
    lines+=['','独立批内积允许负值，保留其频次与幅度；不把负值直接解释为性能提升。两方案使用 Fishr 式 CE/margin 虚拟坐标敏感度方差，不等同于标准模型参数经验 Fisher。校准分母为指定时点学生自身固定尺度 CE/margin 敏感度交叉项梯度。固定系数乘 ramp，两方案再乘每步detached联合限幅标量，选定层之和固定除以 2，未启用层为零。两批各自 60 个不同 PID，批间允许重叠。','', '|方案／种子|额外 L 前向数|clean／额外路径准确率|预测改判比例|clean／额外路径 margin|','|---|---:|---:|---:|---:|']
    for row in source['rows']:
        s=row['style_identity']
        if s:lines.append(f"|{row['row_id']}|{s['style_ce']['observations']}|{s['L_accuracy']['mean']:.4%}／{s['style_L_accuracy']['mean']:.4%}|{s['style_prediction_change']['mean']:.4%}|{s['clean_identity_margin']['mean']:.5f}／{s['styled_identity_margin']['mean']:.5f}|")
    lines+=['','### 候选选择的独立 eval 读出','',
        '|方案／种子|搜索次数|clean／selected CE|候选 CE 均值／范围宽度|margin 变化|准确率变化（百分点）|',
        '|---|---:|---:|---:|---:|---:|']
    for row in source['rows']:
        s=row['style_search_summary']
        if not s:continue
        mean=lambda key:s[key]['mean']
        lines.append(f"|{row['row_id']}|{s['eval_CE_delta']['observations']}|{mean('eval_clean_CE'):.6f}／{mean('eval_selected_CE'):.6f}|{mean('candidate_CE_mean'):.6f}／{mean('candidate_CE_spread'):.6f}|{mean('eval_margin_delta'):+.6f}|{100*mean('eval_accuracy_delta'):+.4f}|")
    lines+=['','### 固定尺度敏感度健康诊断','',
        '|方案／种子|层／批|原始 RMS 均值|低范数率（≤1e-6）|固定尺度后 RMS 均值|',
        '|---|---|---:|---:|---:|']
    for row in source['rows']:
        for level,stats in row['layer_summary'].items():
            for split in ('A','B'):
                prefix='sensitivity_'+split+'/'
                if prefix+'low_norm_rate' not in stats:continue
                get=lambda key:stats[prefix+key]['mean']
                lines.append(f"|{row['row_id']}|{level}/{split}|{get('raw_sensitivity_rms_mean'):.8g}|{get('low_norm_rate'):.4%}|{get('normalized_score_rms_mean'):.6f}|")
    lines+=['', '上述训练 logits 保留正常 dropout，但额外前向使用 clean 之前的 RNG，使掩码一致。候选选择比较独立的 eval/no_grad 读出，训练模式差值不能冒充其因果效果。风格供体只用合法 L 真标签，同 TX/day、跨 RX、不同物理 PID，不读取 U 真值。源层覆盖、A/B 原始敏感度及低范数率、每次四个候选 CE、eval clean/selected 差和实际选定层统计完整保存在 source_trajectory.md 与 evidence/source_full_analysis.json。', '', '## 证据与边界','',
        f"完整核验 {source['total_steps']:,} 个 step、{20*len(expected_ids)} 个 epoch、零暖头记录、所有校准记录及完整源训练 stdout 段，无遗漏种子。浅中层位置为 time_down/t2.pool。两分支均没有辅助头，保留对身份网络的混合二阶导数。MARGIN_ENERGY使用正确 logit 减其余 logits 的 logsumexp，CE_ENERGY直接对未约减CE求导；风格读出 margin 使用正确 logit 减最大错误 logit，两个数值口径不混用。校准不更新身份。U/V 权限、无 EMA 和无目标反馈保持。", '',
        '测试基准此前已经暴露，本批属于固定方案探索；不能称为全新盲测。所有结果保留，不从测试排名回流调参、改变队列或选择性重跑。84% 按完整固定方案三种子均值验收，不拼接最佳单 seed。']
    cache=result.get('teacher_cache')
    if not cache or cache.get('status')!='COMPLETE' or cache.get('packet_occurrences')!=170100:
        raise ValueError('Missing complete teacher cache resource evidence')
    lines+=['','## 独立教师额外成本','',
        f"三份原 E20 在 U56,700 上各前向一次，合计 {cache['packet_occurrences']} 次样本模型读出，缓存生成 {cache['total_seconds']:.2f} 秒，输出 {cache['output_bytes']} 字节。只共享原生 float32 概率，不包含 U 真值、V 或目标样本。",'',
        '每个学生依赖另外两个已训练 E20：历史模型训练总量为随机学生20＋两教师40＝60模型轮。学生没有父权重训练成本；教师预训练已经完成并复用，历史成本仍不为零。共享三模型缓存的增量成本单列，不能按每行重复计完整生成。预测只执行学生模型，教师与 P0 均不参与目标预测。', '',
        '固定教师的源 V 纠错质量复用本次设计前完成的 E20 peer source audit；训练中 V 仅 eval/no_grad 健康诊断。U 教师与学生分歧、同判/分歧 KL 和梯度见 source_trajectory.md，不把分歧称为真实纠错。']
    return '\n'.join(lines)+'\n',contrasts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=d.read(a.output/'evidence/test_results_analysis.json');source=d.read(a.output/'evidence/source_full_analysis.json');text,pairs=build(result,source)
    (a.output/'final_analysis.md').write_text(text,encoding='utf-8');d.write(a.output/'evidence/full_paired_analysis.json',dict(contrasts=pairs,target_feedback=False))
    print(f'VERIFIED {len(d.rows())} rows / {4440*len(d.rows())} source steps / full stratified report')
