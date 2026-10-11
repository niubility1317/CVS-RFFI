"""Collect existing source-only six-arm results, without opening model or target data."""
import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh

PAYLOAD=r'''
import json,time
from pathlib import Path
b=Path(BASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for row in ROWS:
 p=b/row['row_id'];files=['L_observations.pt','V_observations.pt']+[a+s for a in ARMS for s in ('_auxiliary.pt','_training.jsonl','_V_predictions.pt')]
 artifacts={f:dict(path=str(p/f),bytes=(p/f).stat().st_size if (p/f).is_file() else 0) for f in files}
 steps={a:sum(1 for line in (p/(a+'_training.jsonl')).open() if line.strip()) if (p/(a+'_training.jsonl')).is_file() else 0 for a in ARMS}
 rows.append(dict(row_id=row['row_id'],resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),result=read(p/'learning_results.json'),artifacts=artifacts,step_counts=steps))
print(json.dumps(dict(read_at=time.time(),rows=rows,completion=read(b/'completion.json'),failure=read(b/'failure.json'))))
'''

def collect(folder):
 payload=PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows())).replace('ARMS',repr(d.ARMS))
 value=json.loads(ssh(payload));rows=[]
 if value['failure']:raise ValueError(value['failure'])
 for row in value['rows']:
  if not row['completion']:continue
  c=d.config(next(r for r in d.rows() if r['row_id']==row['row_id']));done=row['completion'];result=row['result']
  expected=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',identity_updates=0,auxiliary_updates=12000,target_access=False,L_count=6300,V_count=27000,candidates=8,arms=6,source_model_unchanged=True,bank_unchanged=True)
  if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Completion mismatch')
  if row['resolved']['config']!=c or result['config']!=c or not result['complete'] or result['identity_updates'] or result['target_access']:raise ValueError('Provenance mismatch')
  if set(result['arms'])!=set(d.ARMS) or any(v['bytes']<=0 for v in row['artifacts'].values()):raise ValueError('Missing arm/artifact')
  for arm,x in result['arms'].items():
   if row['step_counts'][arm]!=2000 or x['auxiliary_steps']!=2000:raise ValueError('Incomplete fit')
   for role,n in [('L',6300),('V',27000)]+([('V_shuffled_condition',27000)] if arm.endswith('STATE') else []):
    for policy in ('raw','standardized'):
     for target in ('CE','margin'):
      m=x[role][policy][target]
      if m['count']!=n or m['action_count']!=n*8:raise ValueError('Incomplete observations')
  rows.append(row)
 status='COMPLETE' if len(rows)==3 and value['completion'] else 'PARTIAL'
 summary=dict(status=status,completed=len(rows),run_id=d.RUN,read_at=value['read_at'],target_access=False,rows=rows)
 folder.mkdir(parents=True,exist_ok=True);d.write(folder/'evidence/learning_readback.json',value);d.write(folder/'evidence/learning_summary.json',summary)
 lines=['# 六种辅助学习目标的完整源诊断','',f'状态 {status}，完成 {len(rows)}/3 种子。身份参数全程冻结；每种子六个辅助网络各训练 2000 步，只用 L 拟合。','', '|seed|组合|V margin 排序一致率|原始概率困难间隙覆盖率|标准化概率困难间隙覆盖率|打乱身份状态后排序一致率|','|---|---|---:|---:|---:|---:|']
 def fmt(x):return 'N/A' if x is None else f'{x:.6f}'
 for row in rows:
  for arm,x in row['result']['arms'].items():
   v=x['V'];sh=x.get('V_shuffled_condition',{}).get('raw',{}).get('margin',{}).get('pair_concordance')
   vals=[v['raw']['margin']['pair_concordance'],v['raw']['margin']['fraction_oracle_gap_captured'],v['standardized']['margin']['fraction_oracle_gap_captured'],sh]
   lines.append('|'+row['row_id'].split('-s')[1]+'|'+arm+'|'+'|'.join(fmt(z) for z in vals)+'|')
 lines+=['','排序一致率的无信息基准为 0.5；这不是身份准确率。困难间隙覆盖率描述采样是否偏向真实更难的候选，不证明身份训练后能提高泛化。打乱状态只置换身份特征和 logits，物理观测保持不变。','', '完整 JSON 保留 L/V、CE/margin、两种固定概率规则、MSE（仅同单位）、分类保持率、训练步数及原始产物路径。V 没有用于拟合、标准化或调参；全部辅助训练先于 V 观测完成。','', 'V 的接收机已被原 E200 见过，不能作为未知接收机泛化证据。L/V bank 的标签使用规则不同，二者差异不能单独解释为样本泛化；STATE 与 PHYS 网络参数量不同，差异也不能完全归因于身份状态信息。','', '身份和 bank 前后逐值不变，未打开 U 或 target。本结果仅诊断机制，尚无新版身份训练或测试收益结论。']
 (folder/'learning_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();s=collect(a.output)
 print(json.dumps(dict(status=s['status'],completed=s['completed'],rows=[dict(row_id=r['row_id'],arms={a:dict(margin_pair=x['V']['raw']['margin']['pair_concordance'],standardized_gap=x['V']['standardized']['margin']['fraction_oracle_gap_captured']) for a,x in r['result']['arms'].items()}) for r in s['rows']])))
