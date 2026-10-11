"""Full-matrix report from immutable independently scored artifacts only."""
import argparse,statistics
from pathlib import Path
from . import design as d
from .paired_analysis import analyze as paired

def validate_corrections(result):
 parents={(r['model_seed'],r['dimension'],r['stratum']):r for r in result['parent_records']}
 for r in result['records']:
  p=parents[(r['model_seed'],r['dimension'],r['stratum'])]
  n=r['query_count']
  if p['query_count']!=n:raise ValueError('Parent physical coverage differs')
  if r['parent_fix_count']-r['parent_break_count']!=r['net_correct_gain']:raise ValueError('Fix/break accounting differs')
  if abs((r['accuracy']-p['accuracy'])*n-r['net_correct_gain'])>1e-6:raise ValueError('Parent accuracy and net corrections differ')

def build(result):
 if result['status']!='COMPLETE' or result['missing_rows'] or len(result['completed_rows'])!=18:raise ValueError('Final report requires all18 trained and scored rows')
 if not all(result['checks'].values()):raise ValueError('Full source and independent scoring checks required')
 validate_corrections(result);contrasts=paired(result)
 summaries={r['arm']:r for r in result['summary']};all_records=result['records']+result['parent_records'];overall=[r for r in all_records if r['dimension']=='overall']
 reached=result['complete_arms_reaching_84'];lines=['# 原E200续训40轮：SAM与物理风险机制完整结果','', '**完整矩阵18/18行，三种子均已独立完成clean评分。** '+('达到84%的完整方案：'+', '.join(reached)+'。' if reached else '**没有方案达到clean三种子均值84%，目标未完成。**'),'', '全部方案继承对应原纯CE＋旧门控＋cos E200，固定final40/8880更新。没有EMA或星地增强，源V仅健康检查；测试逐样本面对全部6类，固定168000条query、7个目标RX、4个日期，预测固定后独立truth-last评分。执行提交f73d4aa3046f41d4dd3c4d56005af538e9f703da。分析代码更新不改变已发布训练。','', '|方案|seed2701|seed2702|seed2703|准确率均值|SD（百分点）|Macro-F1均值|相对E200（百分点）|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm in ('P0',*d.ARMS):
  s=summaries[arm]
  if not s['complete_three_seeds'] or set(s['seeds'])!=set(d.SEEDS):raise ValueError('Wrong complete seed set')
  values={r['seed']:r['accuracy'] for r in s['per_seed']};f1=statistics.mean(r['macro_f1'] for r in overall if r['arm']==arm)
  lines.append('|'+arm+'|'+'|'.join(f'{values[seed]*100:.4f}%' for seed in d.SEEDS)+f"|{s['accuracy_mean']*100:.4f}%|{s['accuracy_sd']*100:.4f}|{f1*100:.4f}%|{(s['accuracy_mean']-summaries['P0']['accuracy_mean'])*100:+.4f}|")
 lines+=['', '## 配对增量与互补性', '', '|比较|三seed均值差（百分点）|差值SD（百分点）|正收益种子|', '|---|---:|---:|---:|']
 for r in contrasts['pairs']+contrasts['interactions']:
  if r['dimension']!='overall':continue
  if not r['complete_three_seeds']:raise ValueError('Missing final paired seeds')
  lines.append(f"|{r['name']}|{r['mean_difference']*100:+.4f}|{r['sd_difference']*100:.4f}|{r['positive_seeds']}/3|")
 lines+=['', '主比较SAM_RANK−RANK。SAM_by_physical=(SAM_PHYS−PHYS)−(SAM_SSL−SSL)，SAM_by_risk=(SAM_RANK−RANK)−(SAM_PHYS−PHYS)。所有差值先按seed配对，再求均值；正交互不能替代总体准确率和84%验收，三seed描述结果不自动构成显著性证据。', '', '## 相对基座的纠错与破坏', '', '|方案|seed|修复原错误|破坏原正确|净增加正确数|', '|---|---:|---:|---:|---:|']
 for r in result['records']:
  if r['dimension']=='overall':lines.append(f"|{r['arm']}|{r['model_seed']}|{r['parent_fix_count']}|{r['parent_break_count']}|{r['net_correct_gain']:+d}|")
 lines+=['', '上述是固定最终测试的配对描述；净增加=修复−破坏，并与准确率差×query数独立核对。它衡量身份判别的实际变化，不能反馈到损失权重、样本选择或选择性重跑。']
 for dimension,label,records in [('receiver','接收机',result['records']),('transmitter','发射机',result['records']),('day','日期',result['day_records'])]:
  strata=sorted({r['stratum'] for r in records if r['dimension']==dimension})
  # Dimension labels are verified from existing score records; unknown labels must not silently disappear.
  if not strata:raise ValueError('Missing final '+dimension+' strata')
  lines+=['', '## '+label+'分层：三种子均值', '', '|'+label+'|'+'|'.join(d.ARMS)+'|','|---|'+'|'.join(['---:']*len(d.ARMS))+'|']
  for stratum in strata:
   cells=[]
   for arm in d.ARMS:
    rows=[r for r in records if r['dimension']==dimension and r['stratum']==stratum and r['arm']==arm]
    if len(rows)!=3 or {r['model_seed'] for r in rows}!=set(d.SEEDS):raise ValueError('Incomplete stratified seed coverage')
    cells.append(f"{100*statistics.mean(r['accuracy'] for r in rows):.4f}%")
   lines.append('|'+str(stratum)+'|'+'|'.join(cells)+'|')
 lines+=['', '全部逐seed分层、混淆矩阵、Macro-F1和NLL保留在evidence/test_results_analysis.json；正文表格没有用汇总值替代原始记录。', '', '## 增量成本', '', '|方案|源训练及诊断均值（分钟）|clean推理均值（秒）|最大已分配显存（MiB）|身份可训练参数|辅助可训练参数|身份梯度次数|', '|---|---:|---:|---:|---:|---:|---:|']
 for arm in d.ARMS:
  rows=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
  if len(rows)!=3:raise ValueError('Missing resources')
  resources=[r['source']['measured_training_resources'] for r in rows]
  params={r['trainable_parameters'] for r in resources};aux={r['auxiliary_trainable_parameters'] for r in resources}
  if len(params)!=1 or len(aux)!=1:raise ValueError('Seed parameter count differs')
  train=statistics.mean(r['identity_training_and_source_diagnostics_seconds'] for r in resources)/60
  infer=statistics.mean(r['prediction']['inference_seconds']['clean'] for r in rows)
  mem=max(r['peak_cuda_memory_bytes'] for r in resources)/1024**2
  lines.append(f"|{arm}|{train:.2f}|{infer:.2f}|{mem:.2f}|{next(iter(params))}|{next(iter(aux))}|{17760 if arm.startswith('SAM_') else 8880}|")
 lines+=['', '硬件RTX3090，共享GPU墙钟，包括源诊断；不包含父模型E200和已完成的辅助预热。不同并发负载下不能把墙钟比当作独占硬件加速比。SAM同为8880次优化器更新，但身份梯度计算翻倍；训练FLOPs未测，N/A。辅助RANK两组均为1998步、每步最多32L×8候选，测试只保留身份网络。其他存储与状态字节数见资源JSON。','', '## 证据边界与后续', '', '源域准确率接近饱和，父模型已见全部5个源RX。源V、参数位移、风险排序和低训练损失均不能替代未知接收机测试。已完成训练的完整源日志另见source_trajectory.md；若该文件仍标记partial，须明确其只是中期快照。','', '保持所有负结果，不按本批target结果选择机制、rho、epoch或重跑。84%验收按每个完整固定方案的三seed均值，不能用最佳单seed、方案间拼接或测试选checkpoint替代。新的方法探索只能依据合法源证据及明确设计，当前结果不构成目标已完成，除非表中确有完整方案达到阈值。']
 return '\n'.join(lines)+'\n'

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();result=d.read(a.output/'evidence/test_results_analysis.json');text=build(result);(a.output/'final_analysis.md').write_text(text,encoding='utf-8');print('VERIFIED full18row final report')
