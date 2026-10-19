"""Aggregate all fixed seeds and audit complete training streams; no IQ/model access."""
import gzip,json,math,statistics
from pathlib import Path
from . import design as d
LABELS=dict(PAOFF='末十轮停止普通 PA 波形注入',HALF_AS='半份 As（B01）',B0='原 As 独立重跑 B0',URAW='额外 raw-U 控制',HARD_LCE='L 硬标签 CE（H01）',FIR_AS='平滑 FIR＋As',ORD128='ORD128 控制',DUP192='DUP192 控制')

def clean_metrics(root):
 modern=root/'scoring/scores.json'
 if modern.is_file():
  done=d.read(root/'scoring/complete.json')
  if done['status']!='SCORED_COMPLETE' or not done['truth_last'] or 'clean' not in done['views']:raise ValueError('Incomplete independent score')
  data=d.read(modern);records=data['results']
 else:
  data=d.read(root/'scores/metrics.json');records=data['records']
  if data['status']!='SCORED' or not data['truth_last'] or not data['independent_recount']:raise ValueError('Legacy independent score incomplete')
 if not data['truth_last']:raise ValueError('Truth-last score required')
 overall=next(x for x in records if x['view']=='clean' and x['dimension']=='overall')
 receivers=[x for x in records if x['view']=='clean' and x['dimension']=='receiver']
 if overall['query_count']!=168000 or len(receivers)!=7:raise ValueError('Fixed clean target coverage differs')
 return dict(accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=min(x['accuracy'] for x in receivers),score_file=str(modern if modern.is_file() else root/'scores/metrics.json'))

def summary(values):return dict(values=values,mean=statistics.mean(values),sample_SD=statistics.stdev(values),n=len(values))

def scan(path):
 opener=gzip.open if path.suffix=='.gz' else open
 count=0;first=None;last=None
 with opener(path,'rt',encoding='utf8') as f:
  for line in f:
   if not line.strip():continue
   value=json.loads(line);count+=1
   if first is None:first=value
   last=value
   def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite structured record')
    if isinstance(x,dict):
     for v in x.values():finite(v)
    elif isinstance(x,list):
     for v in x:finite(v)
   finite(value)
   if path.name=='step_metrics.jsonl.gz':
    if value['step']!=count:raise ValueError('Missing/duplicated optimizer log step')
    metrics=value['metrics']
    if metrics.get('train/pseudo_truth_available',0) or metrics.get('train/skipped_nonfinite_loss',0) or metrics.get('train/skipped_nonfinite_grad',0):raise ValueError('Illegal U truth or nonfinite training')
 return dict(records=count,first_step=None if first is None else first.get('step'),last_step=None if last is None else last.get('step'))

def audit(r):
 rowroot=d.BASE/r['row_id'];execution=d.read(rowroot/'training_execution.json');stages=[]
 expected=10 if r['family'] in ('pa','fir') else 30
 if execution['new_epochs']!=expected or execution['new_updates']!=expected*222 or execution['target_access'] or execution['actual_factory_seed']!=r['model_seed']:raise ValueError('Actual training budget/seed differs')
 for part in (['source'] if expected==10 else ['stage20','source']):
  root=rowroot/part;done=d.read(root/'completion.json');epochs=10 if part=='source' else 20
  if done['status']!='SOURCE_TRAINED' or done['epoch']!=epochs or done['optimizer_steps']!=epochs*222 or done['target_access'] or done['head_fit_updates']:raise ValueError('Source training incomplete/illegal')
  files={p.name:scan(p) for p in sorted(root.glob('*.jsonl*'))}
  if files['step_metrics.jsonl.gz']['records']!=epochs*222 or files['epoch_metrics.jsonl']['records']!=epochs:raise ValueError('Full logs incomplete')
  for name in ('reference_steps.jsonl.gz','unit_steps.jsonl.gz'):
   if files[name]['records']!=epochs*222:raise ValueError('Full reference/unit streams incomplete')
  resultpath=root/'result.json'
  if not resultpath.is_file():resultpath=root/'training_result.json'
  result=d.read(resultpath)
  if result['U_truth_read'] or result['target_access'] or result['head_fit_updates']:raise ValueError('Illegal training access or separate head')
  stages.append(dict(segment=part,epochs=epochs,updates=epochs*222,logs=files,actual_result=result))
 log=(d.BASE/'logs'/(r['row_id']+'.log')).read_text(encoding='utf8',errors='replace')
 if any(x in log for x in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError')):raise ValueError('Technical training failure in complete stdout')
 return dict(status='PASS',new_epochs=expected,cumulative_epochs=30,new_updates=expected*222,stages=stages)

def collect():
 audits={};groups={}
 for r in d.rows():
  complete=d.read(d.BASE/r['row_id']/'prediction/complete.json')
  if complete['views']!=['clean'] or complete['truth_read'] or complete['head_fit_updates'] or not complete['all_parameters_frozen']:raise ValueError('Frozen clean prediction incomplete')
  audits[r['row_id']]=audit(r)
 for group,family,arm,old_run in d.GROUPS:
  selected=[r for r in d.rows() if r['group']==group]
  points={2026092701:clean_metrics(d.PROJECT/'runs'/old_run/selected[0]['seed01_row'])}
  for r in selected:points[r['model_seed']]=clean_metrics(d.BASE/r['row_id'])
  groups[group]=dict(label=LABELS[group],per_seed=points,accuracy=summary([points[s]['accuracy'] for s in sorted(points)]),macro_f1=summary([points[s]['macro_f1'] for s in sorted(points)]),worst_RX=summary([points[s]['worst_RX'] for s in sorted(points)]))
 result=dict(status='ANALYZED',run_id=d.RUN,groups=groups,model_seeds=[2026092701,*d.SEEDS],test_views=['clean'],seed01_reused=True,new_rows=16,new_model_epochs=400,new_updates=88800,max_GPUs=2,max_total_GPU_workers_per_card=3,satellite_three_seed_claim=False,selection_context='User-requested fixed replication after exposed seed01 results; no target-based tuning/selection/retry')
 d.write(d.BASE/'full_log_verification.json',audits);d.write(d.BASE/'results.json',result)
 lines=['# 用户指定 8 个配置的三种子 clean 结果','','状态：ANALYZED。seed01 复用历史封存评分；02、03 保留各自原配方，累计 E30。','','|配置|seed01 %|seed02 %|seed03 %|均值 %|样本标准差 %|','|---|---:|---:|---:|---:|---:|']
 for k,g in groups.items():
  p=g['per_seed'];a=g['accuracy'];lines.append('|'+g['label']+'|'+ '|'.join(f"{100*p[s]['accuracy']:.6f}" for s in sorted(p))+f"|{100*a['mean']:.6f}|{100*a['sample_SD']:.6f}|")
 lines+=['','新增 16 条最终结果，共 400 个模型 epoch、88800 次更新；匹配父模型按配方复用。','仅占物理 GPU 4、6，每卡最多 3 个训练/预测进程（计入其他 owner）；CPU scorer 最多 1 个。','新增种子只测 clean，原 low urban 数字仍是 seed01，不能视为三种子星地结果。','全部固定种子保留；用户在看到 seed01 结果后指定补跑，不构成新的独立盲测。']
 (d.BASE/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');return result
