import argparse,json,statistics as st,csv
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh
PAYLOAD=r'''
from pathlib import Path
import json,time
b=Path(BASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for row in ROWS:
 p=b/row['row_id'];r=p/'utility_records.jsonl'
 records=[json.loads(x) for x in r.read_text().splitlines() if x.strip()] if (p/'completion.json').is_file() else []
 rows.append(dict(row_id=row['row_id'],resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),records=records))
print(json.dumps(dict(read_at=time.time(),rows=rows,completion=read(b/'completion.json'),failure=read(b/'failure.json'))))
'''
def collect(folder):
 raw=json.loads(ssh(PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows()))));summary=[]
 if raw['failure']:raise ValueError(raw['failure'])
 for row in raw['rows']:
  if not row['completion']:continue
  c=d.config(next(r for r in d.rows() if r['row_id']==row['row_id']));done=row['completion'];records=row['records']
  if row['resolved']['config']!=c or done['config']!=c:raise ValueError('Resolved config mismatch')
  expected=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',identity_updates=0,auxiliary_updates=0,target_access=False,V_gradients=False,U_file_opened=False,batches=30,virtual_steps=720,source_model_unchanged=True,auxiliary_unchanged=True)
  if any(done.get(k)!=v for k,v in expected.items()) or len(records)!=30 or [r['batch'] for r in records]!=list(range(30)):raise ValueError('Incomplete diagnostic')
  for record in records:
   if len(record['update_ids'])!=30 or len(record['L_probe_ids'])!=96 or len(record['V_probe_ids'])!=96:raise ValueError('Sample budget differs')
   if set(record['update_ids'])&set(record['L_probe_ids']) or set(record['V_probe_ids'])&(set(record['update_ids'])|set(record['L_probe_ids'])):raise ValueError('Physical sample overlap')
   if [v['relative_step'] for v in record['scales']]!=c['budget']['relative_steps']:raise ValueError('Scale mismatch')
   for scale in record['scales']:
    if len(scale['candidates'])!=8 or set(scale['policies'])!=set(d.POLICIES):raise ValueError('Policy mismatch')
    selected=min(range(8),key=lambda i:(scale['candidates'][i]['L']['CE'],i))
    if selected!=scale['selected_from_L'] or scale['policies']['L_selected']!=scale['candidates'][selected]:raise ValueError('L-only selection differs')
  for i,step in enumerate(c['budget']['relative_steps']):
   for policy in d.POLICIES:
    for role in ('L','V'):
     vals=[r['scales'][i]['policies'][policy][role] for r in records]
     summary.append(dict(row_id=row['row_id'],seed=c['model_seed'],relative_step=step,policy=policy,role=role,batches=30,CE_change_mean=st.mean(x['CE_change'] for x in vals),CE_change_median=st.median(x['CE_change'] for x in vals),CE_improved_batches=sum(x['CE_change']<0 for x in vals),margin_change_mean=st.mean(x['margin_change'] for x in vals),fix_occurrences=sum(x['fix_count'] for x in vals),break_occurrences=sum(x['break_count'] for x in vals)))
 n=sum(bool(r['completion']) for r in raw['rows']);status='COMPLETE' if n==3 and raw['completion'] else 'PARTIAL';value=dict(status=status,completed=n,rows=summary,read_at=raw['read_at'],target_access=False)
 folder.mkdir(parents=True,exist_ok=True);d.write(folder/'evidence/utility_readback.json',raw);d.write(folder/'evidence/utility_summary.json',value)
 if summary:
  with (folder/'utility_summary.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 lines=['# 固定E200：扰动更新效用诊断','',f'状态{status}，完成{n}/3种子；所有预登记30批和两个固定尺度完整保留。虚拟步无持久更新，target/U未读取，V没有梯度或动作选择。','', '|seed|相对步长|策略|L平均CE变化|V平均CE变化|V改善批次/30|V margin变化|V修复−破坏包次|','|---|---:|---|---:|---:|---:|---:|---:|']
 for r in summary:
  if r['role']!='V':continue
  l=next(x for x in summary if x['seed']==r['seed'] and x['relative_step']==r['relative_step'] and x['policy']==r['policy'] and x['role']=='L')
  lines.append(f"|{r['seed']}|{r['relative_step']}|{r['policy']}|{l['CE_change_mean']:+.7f}|{r['CE_change_mean']:+.7f}|{r['CE_improved_batches']}|{r['margin_change_mean']:+.5f}|{r['fix_occurrences']-r['break_occurrences']}|")
 lines+=['','CE变化为更新后减更新前，负值表示改善；margin变化正值表示改善。修复/破坏是预登记探针出现包次，不是完整V的独立样本数。L_selected使用L探针选择单个批级动作，V仅报告此选择的效果。','', '所有参数每步恢复，eval模式无随机训练前向；两个相对步长并非AdamW的学习率。本结果只讨论已见源RX间有限步迁移，不能证明未知RX性能或长期40轮收益。风险加权为期望梯度，和随机单动作训练也不同。零均值附近必须结合逐seed、批次数和两种尺度，不选择性报告。']
 (folder/'utility_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();s=collect(a.output);print(json.dumps(dict(status=s['status'],completed=s['completed'],V=[r for r in s['rows'] if r['role']=='V'])))
