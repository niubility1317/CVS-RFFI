"""Complete fixed engineering matrix with every seed and stratified metrics."""
import argparse,statistics as st
from pathlib import Path
from . import design as d

def build(result,source):
    ids={r['row_id'] for r in d.rows()}
    assert result['status']=='COMPLETE' and set(result['completed_rows'])==ids and not result['missing_rows'] and all(result['checks'].values())
    assert source['status']=='COMPLETE' and {r['row_id'] for r in source['rows']}==ids and source['total_steps']==79920 and not source['pending'] and not source['errors']
    assert all(not r['stdout_errors'] for r in source['rows'])
    records=result['records']+result['parent_records'];overall={(r['arm'],r['model_seed']):r for r in records if r['dimension']=='overall'}
    summaries={r['arm']:r for r in result['summary']};reached=result['complete_arms_reaching_84']
    primary=summaries[d.PRIMARY];success=primary['complete_three_seeds'] and primary['accuracy_mean']>=.84
    lines=['# 工程方案完整 clean 测试','',f"18/18 行完成，6 方案各三种子；主方案 {d.PRIMARY} "+('达到' if success else '未达到')+' clean 三种子均值 84%。','达到 84% 的完整方案：'+(', '.join(reached) or '无')+'。','',
        '学生随机初始化旧门控＋cos，训练 20 轮／4440 更新，固定 final20。保留两个其他原 E20 源教师概率均值的 U soft KL；无 EMA、星地增强、Fishr 或辅助头。教师历史训练成本仍为 40 模型轮，预测仅使用学生。每行同一 168000 条 clean query，7 RX／4 day；18 行共 3024000 次决策。prediction 固定后独立 truth-last 评分，结果不回流训练、选模或选择性重跑。','',
        '|方案|2701|2702|2703|均值|SD（百分点）|Macro-F1 均值|相对 P0（百分点）|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('P0',*d.ARMS):
        s=summaries[arm];assert s['complete_three_seeds'] and set(s['seeds'])==set(d.SEEDS)
        values=[overall[arm,seed] for seed in d.SEEDS]
        lines.append('|'+arm+'|'+'|'.join(f"{r['accuracy']*100:.4f}%" for r in values)+f"|{s['accuracy_mean']*100:.4f}%|{s['accuracy_sd']*100:.4f}|{st.mean(r['macro_f1'] for r in values)*100:.4f}%|{(s['accuracy_mean']-summaries['P0']['accuracy_mean'])*100:+.4f}|")
    pairs=[];lines+=['','## 固定批内配对差','', '|比较|2701|2702|2703|均值（百分点）|SD（百分点）|','|---|---:|---:|---:|---:|---:|']
    for a,b in d.PAIRS:
        diff=[overall[a,s]['accuracy']-overall[b,s]['accuracy'] for s in d.SEEDS];v=dict(treatment=a,control=b,per_seed=diff,mean=st.mean(diff),sd=st.stdev(diff));pairs.append(v)
        lines.append('|'+a+'−'+b+'|'+'|'.join(f'{x*100:+.4f}' for x in diff)+f"|{v['mean']*100:+.4f}|{v['sd']*100:.4f}|")
    lines+=['','跨批固定 SSL／L_REPLAY 对照另见 reference_comparisons.md，全部种子固定且不择优重跑。三种子是描述性结果，不能自动声称统计显著。P0 仅为已公开评分参考，没有加载到学生。']
    for dim,label,rs in [('receiver','RX',result['records']),('transmitter','TX',result['records']),('day','day',result['day_records'])]:
        lines+=['','## '+label+' 三种子均值','', '|'+label+'|'+'|'.join(d.ARMS)+'|','|---|'+'|'.join(['---:']*6)+'|']
        for key in sorted({r['stratum'] for r in rs if r['dimension']==dim}):
            cells=[]
            for arm in d.ARMS:
                group=[r for r in rs if r['dimension']==dim and r['stratum']==key and r['arm']==arm];assert len(group)==3 and {r['model_seed'] for r in group}==set(d.SEEDS)
                cells.append(f"{st.mean(r['accuracy'] for r in group)*100:.4f}%")
            lines.append('|'+str(key)+'|'+'|'.join(cells)+'|')
    lines+=['','逐 seed 的 RX/TX/day、混淆矩阵、Macro-F1、NLL、相对 P0 纠错／破坏及资源保留在 evidence/test_results_analysis.json。','', '## 实测增量成本','', '|方案|训练及源诊断（分钟）|推理（秒）|峰值显存（MiB）|SAM 额外前向|hidden mix 额外前向|','|---|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        rs=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')];assert len(rs)==3
        cost=[r['source']['measured_training_resources'] for r in rs];peaks=[r['peak_cuda_memory_bytes'] for r in cost if r['peak_cuda_memory_bytes'] is not None]
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in cost)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in rs):.2f}|{max(peaks)/1024**2:.2f}|{cost[0]['sam_second_forwards']}|{cost[0]['hidden_mix_steps']}|")
    lines+=['','SAM 的 4440 次主更新仍执行两遍目标及反向，每行额外 12432 次前向，额外 L568320／U511488 读出单列；相同训练轮数不代表相同算力。hidden mix 每行 3996 次额外 L 前向／511488 次 L 读出。所有方案零辅助参数、零预热、零校准；FLOPs 未测。共享 GPU 墙钟含并发影响。',
        '完整源日志核验见 source_trajectory.md：每步 LR、损失、SAM 精确恢复、单层混合和覆盖，以及 20 轮参数位移／V 健康记录均检查。测试基准已曾暴露，本批为固定方案探索，不能称为全新盲测。全部负结果保留。']
    cache=result['teacher_cache'];assert cache['packet_occurrences']==170100
    lines+=['',f"三原 E20 源教师生成 U 概率缓存共 {cache['packet_occurrences']} 次样本模型读出，耗时 {cache['total_seconds']:.2f} 秒，{cache['output_bytes']} 字节；共享缓存只计算一次，教师预训练复用仍不为零。"]
    return '\n'.join(lines)+'\n',pairs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();text,pairs=build(d.read(a.output/'evidence/test_results_analysis.json'),d.read(a.output/'evidence/source_full_analysis.json'))
    (a.output/'final_analysis.md').write_text(text,encoding='utf-8');d.write(a.output/'evidence/full_paired_analysis.json',dict(contrasts=pairs,target_feedback=False));print('VERIFIED 18 rows /79920 source steps /full stratified report')
