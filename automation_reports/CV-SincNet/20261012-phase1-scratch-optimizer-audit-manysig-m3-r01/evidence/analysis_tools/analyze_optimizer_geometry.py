from pathlib import Path
import sys,json,statistics,math
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_scratch_optimizer_audit import design as d
out=R/'automation_reports/CV-SincNet'/d.RUN
v=d.read(out/'evidence/complete_source_evidence.json');assert v['status']=='VERIFIED' and v['models']==18
records=[r for w in v['rows'] for r in w['records']]
stats={};details=[]
def summary(values):
 vals=sorted(x for x in values if x is not None)
 return dict(n=len(vals),mean=statistics.mean(vals),median=statistics.median(vals),minimum=min(vals),maximum=max(vals)) if vals else dict(n=0,mean=None,median=None,minimum=None,maximum=None)
for group in ('shallow','middle','rest','all'):
 for mode in d.MODES:
  rows=[]
  for r in records:
   cases={c['objective']:c for c in r['cases'] if c['mode']==mode};f=cases['FULL']['groups'][group];b=cases['IDENTITY']['groups'][group];u=cases['UNCAPPED']['groups'][group]
   rows.append(dict(model=r['model_key'],seed=r['model_seed'],batch=r['batch'],mode=mode,group=group,
    parameter_norm=f['displacement_vs_parameters']['norm_a'],absolute_full_delta=f['displacement_vs_parameters']['norm_b'],
    relative_full_delta=f['displacement_vs_parameters']['norm_ratio_b_over_a'],full_identity_delta_ratio=f['displacement_change_vs_identity']['norm_ratio_b_over_a'],
    full_identity_gradient_cosine=f['gradient_vs_identity']['cosine'],full_identity_delta_cosine=f['displacement_vs_identity']['cosine'],
    uncapped_identity_delta_ratio=u['displacement_change_vs_identity']['norm_ratio_b_over_a'],
    extra_reference_first_order=f['source_reference_directional_derivative']-b['source_reference_directional_derivative']))
  stats[group+'@'+mode]={key:summary([r[key] for r in rows]) for key in rows[0] if key not in ('model','seed','batch','mode','group')}
  stats[group+'@'+mode]['negative_extra_reference_first_order_fraction']=sum(r['extra_reference_first_order']<0 for r in rows)/len(rows)
  details.extend(rows)
proof=dict(status='VERIFIED',models=18,batches=216,groups=stats,scope='final-weight local source counterfactual; no target, no post-update forward; mixed-unit all-parameter relative norm must not be read as uniform functional movement')
d.write(out/'evidence/geometry_summary.json',proof)
lines=['# 优化器诊断的完整解释','',
'18 个模型、216 个源诊断批次和 1,728 个临时 AdamW 条件全部完成。以下汇总使用全部批次，不选种子或模型；仍不构成测试泛化结果。','',
'## 梯度进入了更新，但两者不能用同一比例解释','',
'|参数组|模式|平均 FULL−IDENTITY 位移 / IDENTITY 位移|平均梯度余弦|平均位移余弦|平均参数组相对位移|',
'|---|---|---:|---:|---:|---:|']
for group in ('shallow','middle','rest','all'):
 for mode in d.MODES:
  s=stats[group+'@'+mode]
  def fmt(k):return 'N/A' if s[k]['mean'] is None else f"{s[k]['mean']:.6g}"
  lines.append('|'+group+'|'+mode+'|'+'|'.join(fmt(k) for k in ('full_identity_delta_ratio','full_identity_gradient_cosine','full_identity_delta_cosine','relative_full_delta'))+'|')
lines+=['','FULL 是当前限幅后的完整目标；IDENTITY 包括 CE、U KL、MixStyle 重放等非 Fishr 项。上述比值测量增加 Fishr 后的**实际单步差异**，不是 Fishr 单独优化得到的步长。AdamW 的非线性预条件使梯度限幅 0.2 不意味着位移差也不超过 0.2。',
'保存状态使用 final20 的低学习率和既有动量，重置状态使用登记的峰值学习率；跨模式同时改变两者，不能将差异都解释为“动量压制”。这些结果也不重放训练早期，不说明全部 20 轮一直存在相同幅度。',
'## 全模型相对位移存在量纲混合','',
'原生身份参数包含以 Hz 表示的频率坐标，而卷积和分类参数是另一种数值尺度。`all`/`rest` 的参数范数包含这些大尺度坐标，全模型相对位移很小不能直接推出“神经网络几乎没有更新”。必须同时看 shallow、middle 的同组相对位移、绝对位移和功能变化；频率参数与卷积权重也不应被一个无量纲结论混同。完整原始范数见 geometry_summary.json 和 complete_source_evidence.json。',
'## 源侧方向只作局部证据','',
'额外参照方向是同一批独立 L 梯度与 FULL−IDENTITY 位移的内积。参照 L 原训练已见过，仅与本次主 L 不重叠，并非未知 RX。没有做更新后前向，这个数值不能被写成实测损失或准确率提升。','',
'|参数组|模式|额外一阶方向为负的批次比例|','|---|---|---:|']
for group in ('shallow','middle','rest','all'):
 for mode in d.MODES:lines.append(f"|{group}|{mode}|{stats[group+'@'+mode]['negative_extra_reference_first_order_fraction']:.2%}|")
lines+=['','## 对下一步的约束','',
'本轮足以否定“辅助项完全没有改变身份更新”的绝对说法，但无法把目标性能瓶颈单独归因于辅助项太弱，也无法据此证明一律增大权重有效。下一步应优先改变约束承载的信息，例如保留真实 CE 敏感度的置信度幅度，而不是只把同一种归一化 margin 项放大。对应恒等关系和边界已在 mechanism_followup.md 与合成检查中记录。',
'所有在训行继续使用已冻结配置；目标结果不用于修改其他行、挑选校准策略、改变预算或选择性重跑。新候选仍须独立固定设计后验证完整三种子结果。']
(out/'optimizer_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False))
