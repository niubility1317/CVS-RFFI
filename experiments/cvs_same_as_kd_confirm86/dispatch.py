"""One owner per row: train FINAL30, all seven predictions, independent scorer."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def execute(mode,row):
 env=dict(os.environ)
 if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
 os.execve(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_same_as_kd_confirm86/runner.py'),'--mode',mode,'--row',row],env)
def worker(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row))
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Total GPU process cap3 includes other owners')
  import torch
  from .observer import train
  import gc
  for segment in (20,30):
   train(d.config(next(r for r in d.rows() if r['row_id']==row),segment));gc.collect();torch.cuda.empty_cache()
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .qualification import run as qualify_source
 try:
  with numerical_context(d.FULL_FP32_POLICY):qualify_source(row)
 except Exception as error:
  d.write(d.BASE/row/'source_qualification_failure.json',dict(status='SOURCE_READOUT_FAILED',message=str(error),traceback=traceback.format_exc(),formal_evaluation_still_required=True,no_automatic_retry=True))
 execute('predict',row)
def dispatch():
 if d.BASE.exists():raise FileExistsError('Reconcile own existing run; never duplicate or overwrite')
 d.BASE.mkdir();(d.BASE/'logs').mkdir()
 d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 from .preflight import smoke
 try:
  smoke()
 except BaseException as error:
  d.write(d.BASE/'failure.json',dict(status='PREFLIGHT_FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
 pending=[r for r in d.rows() if not d.is_reused(r)];active={};receipts=[];completed=[];failed=[]
 references=[r['row_id'] for r in d.rows() if d.is_reused(r)]
 def reference_done():
  return all((d.row_root(r)/'complete.json').is_file() and d.read(d.row_root(r)/'complete.json').get('status')=='TRAINED_AND_SCORED' for r in references)
 while pending or active or not reference_done():
  if reference_done():
   for reference in references:
    if reference not in completed:completed.append(reference)
  for row,child in list(active.items()):
   code=child.poll()
   if code is None:continue
   done=d.row_root(row)/'complete.json'
   if code or not done.is_file() or d.read(done).get('status')!='TRAINED_AND_SCORED':failed.append(dict(row_id=row,exit_code=code))
   else:completed.append(row)
   del active[row]
  if not (d.BASE/'source_matrix_decision.json').exists() and all((d.row_root(r['row_id'])/'source_qualification/complete.json').is_file() for r in d.rows()):
   from .report import source_decision
   source_decision()
  if pending and not failed:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    states=capacity();slots=[]
    for gpu,state in states.items():
     reserved={r['pid'] for r in receipts if r['gpu']==gpu and r['row_id'] in active}
     total=state['pids']|reserved
     if len(total)<3 and state['free_mb']>=12000:slots.append((len(total),-state['free_mb'],gpu))
    if slots:
     gpu=min(slots)[2];row=pending.pop(0);cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_same_as_kd_confirm86/runner.py'),'--mode','worker','--row',row['row_id']]
     log=d.BASE/'logs'/(row['row_id']+'.log')
     with log.open('x',encoding='utf8') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
     d.write(d.BASE/'launch.json',dict(rows=receipts))
  d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failed,updated_at=time.time(),max_total_processes_per_GPU=3))
  if failed and not active:break
  if pending or active or not reference_done():time.sleep(3)
 if failed:
  d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,unlaunched=pending,healthy_rows_preserved=True,no_automatic_retry=True));raise RuntimeError('Technical failure preserved; no scientific retuning')
 from .report import collect
 result=collect()
 d.write(d.BASE/'complete.json',dict(status=result['status'],rows=completed,model_seed_count=3,seven_views_per_row=True,crossseed_confirmed=False,goal_complete=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
 if a.mode is not None and a.row not in {r['row_id'] for r in d.rows()}:raise ValueError('Only registered rows permitted')
 if a.mode=='worker':worker(a.row)
 elif a.mode=='predict':
  from .evaluate import predict
  predict(a.row);execute('score',a.row)
 elif a.mode=='score':
  from .evaluate import score
  score(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
 else:dispatch()
