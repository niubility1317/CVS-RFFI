import csv,json,math,time,statistics as st
from pathlib import Path
from . import design as d
def analyze(base):
 base=Path(base);rows=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id'];done=d.read(p/'completion.json');v=d.read(p/'results.json');resolved=d.read(p/'resolved_config.json')
  if done['config']!=v['config'] or v['config']!=c or resolved['config']!=c or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates'] or v['auxiliary_updates'] or v['U_truth'] or v['U_fit'] or v['V_forwards'] or v['target_access'] or not v['parameters_buffers_unchanged'] or not done['E20_actual_cache_check']:raise ValueError('Source diagnostic config/permissions differ')
  if len(v['records'])!=96 or len(v['observations'])!=16 or v['source_packet_occurrences']!=1512000 or done['source_packet_occurrences']!=1512000:raise ValueError('Incomplete source diagnostic')
  if v['model_forward_calls']!=2976 or done['model_forward_calls']!=2976 or done['model_count']!=8 or done['records']!=96 or [x['tag'] for x in v['source_provenance']]!=list(d.TAGS):raise ValueError('Fixed models/forward budget differs')
  if resolved['identity_updates'] or resolved['auxiliary_updates'] or resolved['target_access'] or resolved['U_truth'] or resolved['V_forwards'] or not resolved['source_only']:raise ValueError('Resolved diagnostic permissions differ')
  expected={(tag,group,kind,level) for tag in d.TAGS for group in ('L_fit','L_eval','U_all') for kind,level,m in c['tasks']}
  if {(r['model_tag'],r['stratum'],r['kind'],r['level']) for r in v['records']}!=expected:raise ValueError('Missing/duplicate source record')
  for r in v['records']:
   if r['role']!=('U_s' if r['stratum']=='U_all' else 'L_s'):raise ValueError('Source record role differs')
   if r['count']!={'L_fit':5040,'L_eval':1260,'U_all':56700}[r['stratum']] or not all(math.isfinite(x) for x in r['metrics'].values()):raise ValueError('Source record count/nonfinite')
   if any(j['samples']!=128 or j['probes']!=8 for j in r['jacobian'].values()):raise ValueError('Missing fixed Jacobian')
  lines=[json.loads(line) for line in (p/'model_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
  if lines!=v['observations'] or sum(x['source_packet_occurrences'] for x in lines)!=1512000 or any(x['count']!=(6300 if x['role']=='L_s' else 56700) or x['identity_updates'] or x['target_access'] for x in lines):raise ValueError('Source model-role log differs')
  if {(x['model_tag'],x['role']) for x in lines}!={(tag,role) for tag in d.TAGS for role in ('L_s','U_s')} or any(x['forward_calls']!=(39 if x['role']=='L_s' else 333) or x['source_packet_occurrences']!=3*x['count'] or x['U_truth'] or (x['head'] is not None if x['role']=='U_s' else x['head'] is None) for x in lines):raise ValueError('Source observations/hidden U truth differ')
  with (p/'metrics.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
  flat=[dict(**{k:x for k,x in r.items() if k not in ('metrics','jacobian')},**r['metrics']) for r in v['records']]
  if len(table)!=96 or any(any(str(x)!=t[k] for k,x in r.items()) for r,t in zip(flat,table)):raise ValueError('Source CSV differs')
  with (base/'logs'/f"{row['row_id']}.log").open(encoding='utf-8') as f:
   closed=False
   for line in f:
    if any(x in line for x in ('Traceback','CUDA out of memory','FloatingPointError','RuntimeError')):raise ValueError('Source diagnostic stdout failed')
    if line.startswith('SOURCE_DIAGNOSTIC_COMPLETE '):closed=True;break
   if not closed:raise ValueError('Source diagnostic stdout not closed')
  rows.append(dict(row_id=row['row_id'],model_seed=row['model_seed'],completion=done,result=v))
 return dict(status='COMPLETE',rows=rows,run_id=d.RUN,source_only=True,target_access=False,identity_updates=0,auxiliary_updates=0,source_model_packet_occurrences=4536000,records=288,models=24)
def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v)
 lines=['# 固定响应读出可辨识性：完整源诊断','','3/3种子；24份固定源模型、4536000源样本前向、288个完整分层。身份/辅助更新0、V前向0、target访问0。原CE E20与所有7份E40对照均保留，未按目标成绩选择。','','全部MSE为原fit输出标准化单位；分层、Jacobian、head兼容性和原始逐seed证据见analysis.json。置均值/置乱不是因果干预；Jacobian是局部敏感性，不能直接代表身份证据。原模型已见过源RX，不称未知RX身份验证。','','|模型|层/作用|分层|原目标MSE|零原点MSE|真实响应闭合MSE|h置均值MSE|q置均值MSE|参数线性MSE|零参数输出能量|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for tag in d.TAGS:
  for kind,level,method in d.config(d.rows()[0])['tasks']:
   for group in ('L_fit','L_eval','U_all'):
    rec=[r for row in v['rows'] for r in row['result']['records'] if (r['model_tag'],r['kind'],r['level'],r['stratum'])==(tag,kind,level,group)]
    keys=['reference_target','zero_origin_target','actual_closure','fit_h_mean','fit_q_mean','original_parameter_linear','zero_parameter_output_energy'];lines.append('|'+tag+'|'+kind+':'+level+'|'+group+'|'+'|'.join(f'{st.mean(r["metrics"][k] for r in rec):.6f}' for k in keys)+'|')
 lines+=['','没有新增身份checkpoint或目标成绩，84%目标仍未完成。下一步只根据源端读出/真实响应/原点约束和梯度证据独立设计，不据目标提高权重、挑老师或选择性重跑。'];text='\n'.join(lines)+'\n'
 for name in ('report.md','final_analysis.md'):(d.BASE/name).write_text(text,encoding='utf-8')
 spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
 for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
 d.write(d.BASE/'experiment.json',spec)
 with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='SOURCE_IDENTIFIABILITY_COMPLETE',time=time.time(),models=24,source_packet_occurrences=4536000,target_access=False,identity_updates=0))+'\n')
 return dict(status='VERIFIED',rows=3,records=288)
