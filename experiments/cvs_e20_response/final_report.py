import gzip,json,csv,math,time
from . import design as d
from .source import expected_counts
def analyze(base):
 from pathlib import Path
 base=Path(base);rows=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id']/'source';done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json')
  if done['config']!=c or resolved['config']!=c or done['status']!='SOURCE_TRAINED' or done['target_access'] or done['total_epochs']!=40 or done['logged_steps']!=4440 or done['counts']!=expected_counts(c):raise ValueError('Incomplete source row/config')
  if resolved['execution']['EMA'] or resolved['execution']['satellite_generator'] or resolved['execution']['auxiliary_optimizer'] or not resolved['execution']['E20_weights_loaded']:raise ValueError('Wrong actual runtime')
  records=[];totals={k:0 for k in expected_counts(c)}
  with gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
   for line in stream:
    record=json.loads(line);step=len(records)+1
    if record['step']!=step or record['epoch']!=(step-1)//222+1:raise ValueError('Missing/duplicate source step')
    if not all(value is None or math.isfinite(value) for value in record['metrics'].values()):raise ValueError('Nonfinite source metric')
    info=record['details'];totals['identity_updates']+=1
    for key in totals:
     if key!='identity_updates':totals[key]+=info[key]
    if c['response']['enabled']:
     if set(info['role_task_losses'])!=set(c['response']['roles']) or any(len(values)!=4 for values in info['role_task_losses'].values()):raise ValueError('Missing response roles/readers')
    elif info['role_task_losses']:raise ValueError('Unexpected response forward')
    records.append(record)
  if len(records)!=4440 or totals!=expected_counts(c):raise ValueError('Source budget not closed')
  epochs=[json.loads(line) for line in (p/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
  if [e['epoch'] for e in epochs]!=list(range(1,21)) or epochs[-1]['counts']!=totals:raise ValueError('Source epochs incomplete')
  with (p/'epoch_metrics.csv').open(encoding='utf-8',newline='') as stream:csvrows=list(csv.DictReader(stream))
  if len(csvrows)!=20 or any(any(('' if value is None else str(value))!=csvrow[key] for key,value in e.items() if not isinstance(value,(dict,list))) for e,csvrow in zip(epochs,csvrows)):raise ValueError('Epoch CSV differs')
  closed=False;lines=0
  with (base/'logs'/f"{row['row_id']}.log").open(encoding='utf-8') as stream:
   for line in stream:
    lines+=1
    if any(v in line for v in ('Traceback','CUDA out of memory','FloatingPointError','RuntimeError')):raise ValueError('Source stdout failed')
    if line.startswith('SOURCE_COMPLETE '):closed=True;break
  if not closed:raise ValueError('Source stdout not closed')
  # Full records were consumed above; keep compact epoch/diagnostic artifacts.
  rows.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],source_steps=len(records),source_stdout_lines=lines,completion=done,epochs=epochs,gradient_audits=[r['gradient_audit'] for r in records if r['gradient_audit'] is not None],complete_source_read=True))
 return dict(status='COMPLETE',run_id=d.RUN,rows=rows,total_steps=sum(r['source_steps'] for r in rows),target_feedback=False)
def finish():
 from .dispatch import require_scored
 v=analyze(d.BASE);require_scored(d.read(d.BASE/'scoring_complete.json'),21)
 d.write(d.BASE/'evidence/source_full_analysis.json',v)
 text=(d.BASE/'analysis.md').read_text(encoding='utf-8')+'\n完整21/21源训练：93240身份更新，全步日志及420个epoch的JSONL/CSV闭合。实际成本见resources.json；参数移动、梯度竞争和每作用/角色loss见evidence/source_full_analysis.json。全部7arm保留，target不回流调参或重跑。\n'
 (d.BASE/'final_analysis.md').write_text(text,encoding='utf-8');(d.BASE/'report.md').write_text(text,encoding='utf-8')
 spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
 for row in spec['rows']:row['status']='TRAINED_AND_SCORED'
 d.write(d.BASE/'experiment.json',spec)
 with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='COMPLETE_ANALYZED',time=time.time(),source_steps=93240,predictions=3528000))+'\n')
 return dict(status='VERIFIED',rows=21)
