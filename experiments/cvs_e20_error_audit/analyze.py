"""Complete source-only paired counts and gradient summaries, without rankings."""
import argparse,csv,json,statistics as st
from pathlib import Path
from . import design as d


def analyze(input_path,output):
 value=d.read(input_path);assert value['status']=='VERIFIED' and len(value['rows'])==3
 output=Path(output);output.mkdir(parents=True,exist_ok=True);paired=[];gradients=[];weak=[]
 for reference,key in [('E20','rows'),('CE_continuation','versus_CE')]:
  for role in ('L','V'):
   for arm in d.config(d.rows()[0])['endpoint_arms']:
    if reference=='CE_continuation' and arm=='CE':continue
    rows=[next(t['result'] for t in r['paired'][key] if t['arm']==arm and t['role']==role) for r in value['rows']]
    totals={name:sum(r['overall'][name] for r in rows) for name in ('count','initial_correct','final_correct','fixed','broken','persistent_error')}
    groups={g:{k:sum(r['initial_groups'][g][k] for r in rows) for k in ('count','fixed','broken','persistent_error')} for g in rows[0]['initial_groups']}
    row=dict(reference=reference,role=role,arm=arm,**totals,net=totals['fixed']-totals['broken'],
     initial_accuracy=totals['initial_correct']/totals['count'],final_accuracy=totals['final_correct']/totals['count'],
     delta_pp=100*(totals['final_correct']-totals['initial_correct'])/totals['count'],
     seeds=[dict(seed=r['row']['model_seed'],**t['overall']) for r,t in zip(value['rows'],rows)],
     groups=groups,initial_ECE=st.mean(r['calibration']['initial']['ECE'] for r in rows),final_ECE=st.mean(r['calibration']['final']['ECE'] for r in rows))
    assert row['net']==row['final_correct']-row['initial_correct'];paired.append(row)
 for arm in d.config(d.rows()[0])['endpoint_arms']:
  entries=[t for r in value['rows'] for t in r['gradient']['rows'] if t['arm']==arm];assert len(entries)==72
  for task in ('ce','KL','aux_ce','fishr','total'):
   for layer in ('shallow','middle','encoder','classifier'):
    vals=[t['tasks'][task]['groups'][layer] for t in entries];valid=[v for v in vals if v['cosine'] is not None]
    def avg(k):return st.mean(v[k] for v in valid) if valid else None
    def median(k):return st.median(v[k] for v in valid) if valid else None
    gradients.append(dict(arm=arm,task=task,layer=layer,count=72,nonzero=len(valid),
     cosine_mean=avg('cosine'),cosine_median=median('cosine'),negative=sum(v['negative'] for v in valid),
     negative_fraction=sum(v['negative'] for v in valid)/len(valid) if valid else None,
     norm_ratio_mean=avg('norm_ratio'),norm_ratio_median=median('norm_ratio'),dot_mean=avg('dot'),
     per_seed=[dict(seed=r['row']['model_seed'],cosines=[t['tasks'][task]['groups'][layer]['cosine'] for t in r['gradient']['rows'] if t['arm']==arm]) for r in value['rows']]))
 for r in value['rows']:
  entries=[t for t in r['gradient']['rows'] if t['arm']=='CE'];groups={}
  for g in entries[0]['weak_main_L_calibration']['initial_groups']:
   groups[g]={k:sum(t['weak_main_L_calibration']['initial_groups'][g][k] for t in entries) for k in ('count','initial_correct','persistent_error')}
  weak.append(dict(seed=r['row']['model_seed'],L_occurrences=24*60,unique_L_ids=len({p for t in entries for p in t['main_L_ids']}),groups=groups))
 summary=dict(status='ANALYZED',source_only=True,target_access=False,identity_updates=0,source_readout_packets=value['source_readout_packets'],gradient_records=576,paired=paired,gradients=gradients,weak_L=weak,resources=[dict(seed=r['row']['model_seed'],**r['completion']['resources']) for r in value['rows']])
 d.write(output/'summary.json',summary)
 def save_csv(name,rows,exclude):
  fields=[k for k in rows[0] if k not in exclude]
  with (output/name).open('w',encoding='utf-8',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
 save_csv('paired_summary.csv',paired,('groups','seeds'));save_csv('gradient_summary.csv',gradients,('per_seed',))
 lines=['# E20 浅中层机制源域诊断结果','',
  '三种子源域诊断完整完成：899,100 次源样本模型读出、576 个目标梯度记录、600 步仅辅助头预热，身份参数更新为 0。V 全程 eval/no_grad；U TX 标签不可见；本轮没有目标测试。',
  '', '## 固定 E20 错误的变化','',
  '以下表格将同一物理样本在 E20 和各个固定终点间配对，数量为三个 seed 的合计。L 共 18,900 个样本模型对，V 共 81,000 个。',
  '', '|角色|方案|E20准确率|终点准确率|纠正|破坏|净纠正|变化/百分点|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in paired:
  if r['reference']=='E20':lines.append(f"|{r['role']}|{r['arm']}|{100*r['initial_accuracy']:.4f}%|{100*r['final_accuracy']:.4f}%|{r['fixed']}|{r['broken']}|{r['net']}|{r['delta_pp']:+.4f}|")
 lines+=['','## 相对纯 CE 续训的机制作用','',
  '此处改用固定 CE 续训终点为参照，置信组也由该终点定义，与上一节 E20 分组分开。',
  '', '|角色|方案|纠正 CE 错误|破坏 CE 正确|净纠正|变化/百分点|','|---|---|---:|---:|---:|---:|']
 for r in paired:
  if r['reference']=='CE_continuation':lines.append(f"|{r['role']}|{r['arm']}|{r['fixed']}|{r['broken']}|{r['net']}|{r['delta_pp']:+.4f}|")
 lines+=['','## E20 高置信错误','',
  '阈值固定 0.95。高置信错误在有标签 L/V 上定义，不能把这些统计当成 U 伪标签的真实准确率。',
  '', '|角色|方案|初始高置信错误|纠正|持续错误|高置信正确被破坏|','|---|---|---:|---:|---:|---:|']
 for r in paired:
  if r['reference']=='E20':
   g=r['groups'];lines.append(f"|{r['role']}|{r['arm']}|{g['high_error']['count']}|{g['high_error']['fixed']}|{g['high_error']['persistent_error']}|{g['high_correct']['broken']}|")
 lines+=['','## 同权重的梯度方向','',
  '每个单元格来自三 seed×24 轮。参照为独立、不同 day 的有标签 L 的 eval CE 梯度。主目标保持训练 dropout，保留原固定目标权重，无优化器更新。encoder 包含投影、融合及门控，排除分类头与 Hz 参数。',
  '', '|方案|损失|层组|有效数|余弦中位数|负向比例|范数比中位数|','|---|---|---|---:|---:|---:|---:|']
 for g in gradients:
  if g['task'] not in ('total','KL','aux_ce','fishr') or g['nonzero']==0 or g['layer']=='classifier':continue
  lines.append(f"|{g['arm']}|{g['task']}|{g['layer']}|{g['nonzero']}|{g['cosine_median']:.5f}|{100*g['negative_fraction']:.2f}%|{g['norm_ratio_median']:.5f}|")
 lines+=['','## 每种子的完整读出与开销','',
  '|角色|方案|seed 2026092701|seed 2026092702|seed 2026092703|','|---|---|---:|---:|---:|']
 for r in paired:
  if r['reference']=='E20':lines.append('|'+r['role']+'|'+r['arm']+'|'+'|'.join(f"{100*t['final_accuracy']:.4f}%" for t in r['seeds'])+'|')
 lines+=['','|seed|耗时/秒|峰值显存/GB|身份更新|辅助头更新|','|---|---:|---:|---:|---:|']
 for r in summary['resources']:lines.append(f"|{r['seed']}|{r['seconds']:.2f}|{r['peak_cuda_memory_bytes']/1e9:.3f}|0|{r['head_updates']}|")
 lines+=['','## 解释边界','',
  'V 属于父模型见过的五个源 RX，不能替代未知接收机测试。相对 CE 的修复和破坏来自实际训练终点，但这批同权重梯度只提供局部一阶方向证据，未回放 AdamW 状态，也不是对整个训练轨迹的因果证明。全部八个方案保留，不按目标成绩筛选子集。',
  '', '逐 seed、置信组、校准分箱及 RX/day/TX 的完整记录见 evidence/source_results.json；摘要见 summary.json、paired_summary.csv 和 gradient_summary.csv。本轮不产生新测试成绩，84% 目标未完成。','']
 (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
 return summary


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();v=analyze(a.input,a.output)
 print(json.dumps(dict(status=v['status'],paired_rows=len(v['paired']),gradient_groups=len(v['gradients']),source_readout_packets=v['source_readout_packets'],identity_updates=0)))
