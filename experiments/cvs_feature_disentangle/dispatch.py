"""One owner: fixed source matrix, source-only selection, then automatic tests."""
import os, subprocess, sys, time
from pathlib import Path
from . import design as d
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock, process

def event(status,**values):
 import json
 with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**values))+'\n')

def test_row(rid):
 from . import evaluate as e
 # snapshot enforces the source-selection boundary, also for direct invocation.
 e.EVAL_ROW=rid;e.snapshot(d)
 e.predict(rid);event('PREDICTIONS_COMPLETE',row_id=rid)
 subprocess.run([sys.executable,'-m','experiments.cvs_feature_disentangle.evaluate','--mode','score','--row',rid],
   cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 scored=d.read(d.BASE/'row_scoring'/rid/'scoring_complete.json')
 if scored['models']!=1 or scored['views']!=7 or scored['metric_records']!=98 or scored['day_metric_records']!=28:
  raise ValueError('Incomplete row scoring')
 d.write(d.BASE/rid/'completion.json',dict(status='ANALYZED',views=7,independent_recount='VERIFIED'))
 event('ANALYZED',row_id=rid)

def worker(kind,rid):
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);stable=0
 while stable<3:
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   cap=capacity()[gpu];ready=len(cap['pids']|{os.getpid()})<=2 and cap['free_mb']>=12000
   stable=stable+1 if ready else 0
  if stable<3:time.sleep(1 if ready else 5)
 try:
  r=next(r for r in d.rows() if r['row_id']==rid)
  if kind=='test':test_row(rid);return
  c=d.resolved_config(r)
  if d.read(d.BASE/'configs'/(rid+'.json'))!=c:raise ValueError('Changed frozen configuration')
  from .source import train
  train(c)
  from . import evaluate as e
  e.source_provenance(d,c)
  stress=d.read(Path(c['output_root'])/'source_stress.json')
  if stress['status']!='SOURCE_STRESS_COMPLETE' or stress['target_access']:raise ValueError('Incomplete source validation')
  d.write(d.BASE/rid/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False,
      selection='FIXED_OWN_E200',frozen_at=time.time(),source_stress_ref=str(Path(c['output_root'])/'source_stress.json')))
  event('SOURCE_FROZEN',row_id=rid)
  if r['arm'] in d.SEARCH_ARMS:
   d.write(d.BASE/rid/'completion.json',dict(status='SOURCE_CANDIDATE_FROZEN',target_access=False));return
  test_row(rid)
 except Exception as exc:
  d.write(d.BASE/rid/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',row_id=rid,error=repr(exc));raise

def phase(kind,items,receipts):
 pending=list(items);active={};done=[];failures=[]
 while pending or active:
  for rid,child in list(active.items()):
   rc=child.poll()
   if rc is None:continue
   r=next(r for r in d.rows() if r['row_id']==rid)
   expected='SOURCE_CANDIDATE_FROZEN' if kind=='train' and r['arm'] in d.SEARCH_ARMS else 'ANALYZED'
   p=d.BASE/rid/'completion.json'
   if rc or not p.is_file() or d.read(p)['status']!=expected:failures.append(dict(row_id=rid,exit_code=rc))
   else:done.append(rid)
   del active[rid]
  while pending and len(active)<12 and not failures:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    choices=[(len(x['pids']),-x['free_mb'],gpu) for gpu,x in capacity().items() if len(x['pids'])<2 and x['free_mb']>=12000]
    if not choices:break
    gpu=min(choices)[2];rid=pending.pop(0);log=d.BASE/'logs'/(rid+'-'+kind+'.log')
    cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_feature_disentangle/train_worker.py'),'--kind',kind,'--row',rid]
    with log.open('x') as f:
     child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    if process(child.pid) is None:raise RuntimeError('Worker absent')
    receipts.append(dict(kind=kind,row_id=rid,pid=child.pid,gpu=gpu,log=str(log),argv=cmd,cwd=str(d.ROOT)))
    d.write(d.BASE/'launch.json',dict(rows=receipts));active[rid]=child
   event('RUNNING',row_id=rid,kind=kind,pid=child.pid,gpu=gpu)
  d.write(d.BASE/'queue_state.json',dict(phase=kind,active=list(active),pending=pending,completed=done,failures=failures,
    total_identity_rows=48,total_test_rows=36,per_gpu_limit=2,updated_at=time.time()))
  if failures and not active:raise RuntimeError('Technical failures retained; no retry '+repr(failures))
  if pending or active:time.sleep(10)

def dispatch():
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
  for r in d.rows():d.write(d.BASE/'configs'/(r['row_id']+'.json'),d.config(r))
  d.write(d.BASE/'experiment.json',d.read(d.ROOT/'experiments/cvs_feature_disentangle/experiment.json'))
  d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),owner=d.OWNER))
  receipts=[]
  try:
   phase('train',[r['row_id'] for r in d.rows() if r['arm'] in d.FIXED_ARMS],receipts)
   from .validation import source_select
   selection=source_select()
   allowed=d.test_rows(selection)
   for r in d.rows():
    if r['arm'] in d.SELECTED_ARMS:d.write(d.BASE/'configs'/(r['row_id']+'.json'),d.resolved_config(r))
   # Resolve conditional rows and record exact values before their launch.
   spec=d.read(d.BASE/'experiment.json')
   for row in spec['rows']:
    r=next(r for r in d.rows() if r['row_id']==row['row_id'])
    if r['arm'] in d.SELECTED_ARMS:row['config']=d.resolved_config(r)
    if r['arm'] in d.SEARCH_ARMS and r['row_id'] not in {r['row_id'] for r in allowed}:
     row['status']='SOURCE_ONLY_NOT_SELECTED'
     d.write(d.BASE/r['row_id']/'completion.json',dict(status='SOURCE_ONLY_NOT_SELECTED',target_access=False))
   spec['source_selection']=selection;d.write(d.BASE/'experiment.json',spec)
   event('SOURCE_SELECTION_FROZEN',selected_ratio=selection['selected_ratio'],target_scores_consumed=False)
   # Already trained selected rows enter testing immediately, before new controls.
   phase('test',[r['row_id'] for r in allowed if r['arm'] in d.SEARCH_ARMS],receipts)
   phase('train',[r['row_id'] for r in d.rows() if r['arm'] in d.SELECTED_ARMS],receipts)
   subprocess.run([sys.executable,'-m','experiments.cvs_feature_disentangle.evaluate','--mode','score'],cwd=d.ROOT,
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
   result=d.read(d.BASE/'scoring_complete.json')
   if result['models']!=36 or result['views']!=7 or result['metric_records']!=3528 or result['day_metric_records']!=1008:
    raise ValueError('Incomplete test matrix')
   d.write(d.BASE/'completion.json',dict(status='ANALYZED',trained_models=48,tested_models=36,source_only_models=12,
     views=7,independent_recount='VERIFIED',all_selected_rows_scored_automatically=True))
   spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED';spec['results_ref']=str(d.BASE/'analysis.md');d.write(d.BASE/'experiment.json',spec)
   event('ANALYZED',trained=48,tested=36,source_only=12)
  except Exception as exc:
   d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise

if __name__=='__main__':dispatch()
