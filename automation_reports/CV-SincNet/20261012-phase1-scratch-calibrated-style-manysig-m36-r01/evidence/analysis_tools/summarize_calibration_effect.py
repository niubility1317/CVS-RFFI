from pathlib import Path
import json,statistics,math
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011')
live=json.loads((R/'local_artifacts/calibrated_style_live.json').read_text(encoding='utf-8'))
folder=R/'automation_reports/CV-SincNet'/live['run'];groups={}
for row in live['rows']:
 if not row['resolved']:continue
 c=row['resolved'];v=c['fishr_calibration']
 if v['status']!='FIXED_CALIBRATION_COMPLETE':continue
 key=(c['model_seed'],v['after_identity_updates']);groups.setdefault(key,[]).append((row['row_id'],v))
rows=[]
for seed in sorted({seed for seed,step in groups}):
 if (seed,0) not in groups or (seed,444) not in groups:continue
 for level in ('time_down','t2.pool'):
  before=[v['fixed_coefficients'][level] for _,v in groups[seed,0]]
  after=[v['fixed_coefficients'][level] for _,v in groups[seed,444]]
  entries={}
  for at,values in ((0,before),(444,after)):
   details=[v['levels'][level] for _,v in groups[seed,at]]
   entries[str(at)]=dict(rows=[rid for rid,_ in groups[seed,at]],count=len(values),coefficient_mean=statistics.mean(values),coefficient_min=min(values),coefficient_max=max(values),mean_CE_norm=statistics.mean(x['mean_CE_gradient_norm'] for x in details),mean_R_norm=statistics.mean(x['mean_term_gradient_norm'] for x in details))
  rows.append(dict(seed=seed,level=level,calibration=entries,coefficient_ratio_after_before=statistics.mean(after)/statistics.mean(before)))
data=dict(status='VERIFIED_SOURCE_COMPARISON',read_at=live['read_at'],run_id=live['run'],source_only=True,target_access=False,
 scope='Registered source-L calibration at update0 versus444, completed calibration records only; unequal in-training step counts are not performance comparisons',rows=rows)
(folder/'evidence/calibration_effect.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 校准时点对系数的实际影响','',
 '本表使用已运行行的校准记录，比较随机初始化与第444次CE更新后同一12对L批的梯度尺度。只使用源域；各记录均已核验状态不变、无U/V/target访问。不同策略的当前训练步数不一致，本表不比较训练效果或泛化能力。','',
 '|seed|层|初始化系数（范围）|延后系数（范围）|延后/初始化|CE梯度范数 0→444|原始项梯度范数 0→444|','|---|---|---:|---:|---:|---:|---:|']
for row in rows:
 a,b=row['calibration']['0'],row['calibration']['444']
 lines.append(f"|{row['seed']}|{row['level']}|{a['coefficient_mean']:.6g}（{a['coefficient_min']:.6g}至{a['coefficient_max']:.6g}）|{b['coefficient_mean']:.6g}（{b['coefficient_min']:.6g}至{b['coefficient_max']:.6g}）|{row['coefficient_ratio_after_before']:.6g}|{a['mean_CE_norm']:.6g}→{b['mean_CE_norm']:.6g}|{a['mean_R_norm']:.6g}→{b['mean_R_norm']:.6g}|")
lines+=['','同一seed/时点跨层位方案列出范围，不用均值掩盖差异；参与行明细见evidence/calibration_effect.json。只覆盖当前已有初始与延后配对的seed，未启动种子不补值。',
 '系数下降表示在较成熟表示上估计的梯度尺度不同，不证明原始敏感度方向变好。延后校准仍是一次固定系数，后续可能漂移；联合上限是否持续触发、是否压制有效学习，需保留完整源日志和最终固定测试。现有36行矩阵、参数与队列不变。']
(folder/'calibration_effect.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(data,ensure_ascii=False))
