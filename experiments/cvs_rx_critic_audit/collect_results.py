"""Read existing source-only diagnosis; no model, IQ or target access locally."""
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
 p=b/row['row_id'];rows.append(dict(row_id=row['row_id'],resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),result=read(p/'ranking_results.json'),arrays={k:dict(path=str(p/(k+'_actions.pt')),exists=(p/(k+'_actions.pt')).is_file(),bytes=(p/(k+'_actions.pt')).stat().st_size if (p/(k+'_actions.pt')).is_file() else 0) for k in ('L','V')}))
print(json.dumps(dict(read_at=time.time(),rows=rows,completion=read(b/'completion.json'),failure=read(b/'failure.json'),queue=read(b/'queue_state.json'))))
'''
def collect(folder):
 value=json.loads(ssh(PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows()))));rows=[]
 if value['failure']:raise ValueError('Remote diagnostic failure: '+repr(value['failure']))
 for row in value['rows']:
  if not row['completion']:continue
  c=d.config(next(r for r in d.rows() if r['row_id']==row['row_id']));done=row['completion'];result=row['result']
  if any(done.get(k)!=v for k,v in dict(status='SOURCE_DIAGNOSTIC_COMPLETE',identity_updates=0,auxiliary_updates=0,target_access=False,L_count=6300,V_count=27000,candidates=8,source_model_unchanged=True,auxiliary_unchanged=True).items()):raise ValueError('Wrong diagnostic completion')
  if row['resolved']['config']!=c or result['config']!=c or not result['complete'] or result['target_access']:raise ValueError('Result provenance differs')
  for role,n in [('L',6300),('V',27000)]:
   if result['roles'][role]['CE']['count']!=n or result['roles'][role]['CE']['action_count']!=n*8 or not row['arrays'][role]['exists']:raise ValueError('Missing complete role/candidate artifacts')
  rows.append(row)
 status='COMPLETE' if len(rows)==3 and value['completion'] else 'PARTIAL';summary=dict(status=status,completed=len(rows),run_id=d.RUN,read_at=value['read_at'],target_access=False,rows=rows)
 folder.mkdir(parents=True,exist_ok=True);d.write(folder/'evidence/ranking_readback.json',value);d.write(folder/'evidence/ranking_summary.json',summary)
 t=['# 冻结辅助网络的难度排序诊断','',f'状态 {status}，完成 {len(rows)}/3 行。只读取已有源诊断产物，没有目标评分。','', '|seed|角色|CE候选对排序一致率|CE MSE/零预测MSE|采样总变差|oracle间隙覆盖率|margin候选对一致率|','|---|---|---:|---:|---:|---:|---:|']
 def fmt(x):return 'N/A' if x is None else f'{x:.6f}'
 for row in rows:
  for role,x in row['result']['roles'].items():
   ce=x['CE'];ma=x['margin_ranking'];t.append('|'+row['row_id'].split('-s')[1]+'|'+role+'|'+'|'.join(fmt(v) for v in (ce['pair_concordance'],ce['MSE_relative_to_zero'],ce['total_variation_from_uniform_mean'],ce['fraction_oracle_gap_captured'],ma['pair_concordance']))+'|')
 t+=['','候选对一致率在无信息并列预测下为0.5；它是动作排序指标，不是身份识别准确率。MSE比值低于1表示优于零预测器。总变差为0表示采样完全均匀；oracle间隙覆盖率衡量当前概率实际利用了多少可用难度差，oracle不可用于部署。','', 'L使用训练时的TX/day排除bank，V使用TX-blind U bank，因此L/V差异同时含样本与bank角色变化。V是已见RX的独立样本，不能作为未知RX成绩。每seed、RX/day分层及全部候选数组路径保存在JSON；原始数组保留远端。','', '身份网络及辅助网络均被冻结，前后state_dict逐值相等，两个更新次数均为0。结果只支持后续源机制研究，不用于更改已完成目标测试或选择性重跑。']
 (folder/'ranking_results.md').write_text('\n'.join(t)+'\n',encoding='utf-8')
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();s=collect(a.output);print(json.dumps(dict(status=s['status'],completed=s['completed'],rows=[dict(row_id=r['row_id'],roles={role:{k:x['CE'][k] for k in ('pair_concordance','MSE_relative_to_zero','total_variation_from_uniform_mean','fraction_oracle_gap_captured')} for role,x in r['result']['roles'].items()}) for r in s['rows']])))
