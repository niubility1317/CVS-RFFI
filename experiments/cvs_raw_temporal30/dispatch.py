"""Single launch owner; one bank, immutable four rows, automatic all7 truth-last."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
RUNNER=d.ROOT/'experiments/cvs_raw_temporal30/runner.py'

def execute(mode,row):
 env=dict(os.environ)
 if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
 os.execve(sys.executable,[sys.executable,'-u',str(RUNNER),'--mode',mode,'--row',row],env)

def worker(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row))
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   if len(capacity()[gpu]['pids']|{os.getpid()})>2:raise ValueError('Process cap2 includes all owners')
   import torch
   torch.cuda.init()
  from .train import train
  train(c)
  import gc
  gc.collect();torch.cuda.empty_cache()
  from experiments.cvs_equivariant_identity.precision import numerical_context
  from .diagnose import run
  with numerical_context(d.FULL_FP32_POLICY):run(row)
 execute('predict',row)

def dispatch():
 if d.BASE.exists():raise FileExistsError('Immutable run already exists; reconcile, never overwrite/relaunch')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 from .smoke import run as smoke
 smoke()
 from experiments.cvs_feature_adapters30.bank import open_bank
 L,indices,observations=open_bank()
 d.write(d.BASE/'observation_bank_reuse.json',dict(status='REUSED_EXISTING_VALIDATED_SOURCE_OBSERVATIONS',source_bank=str(d.BANK_BASE/'observation_bank'),updates=2220,L_slots=284160,new_physical_simulations=0,target_access=False,U_used=False))
 del L,indices,observations
 pending=d.rows();active={};receipts=[];completed=[];scored=[];failed=[]
 while pending or active:
  for row,child in list(active.items()):
   code=child.poll()
   if code is None:continue
   done=d.BASE/row/'prediction/complete.json'
   if code or not done.is_file() or d.read(done).get('status')!='PREDICTIONS_COMPLETE':failed.append(dict(row_id=row,exit_code=code));d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,healthy_rows_preserved=True,no_automatic_retry=True))
   else:completed.append(row)
   del active[row]
  if not (d.BASE/'source_matrix_decision.json').exists() and all((d.BASE/r['row_id']/'source_qualification/complete.json').is_file() for r in d.rows()):
   from .report import source_decision
   source_decision()
  if (d.BASE/'source_matrix_decision.json').is_file():
   for row in completed:
    if row not in scored:
     subprocess.run([sys.executable,'-u',str(RUNNER),'--mode','score','--row',row],cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True);scored.append(row)
  if pending and not failed:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    slots=[]
    for gpu,state in capacity().items():
     reserved={r['pid'] for r in receipts if r['gpu']==gpu and r['row_id'] in active};total=state['pids']|reserved
     if len(total)<2 and state['free_mb']>=12000:slots.append((len(total),-state['free_mb'],gpu))
    if slots:
     gpu=min(slots)[2];r=pending.pop(0);cmd=[sys.executable,'-u',str(RUNNER),'--mode','worker','--row',r['row_id']];log=d.BASE/'logs'/(r['row_id']+'.log')
     with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     active[r['row_id']]=child;receipts.append(dict(row_id=r['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=receipts))
  d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,scored=scored,failures=failed,updated_at=time.time(),max_total_processes_per_GPU=2))
  if failed and not active:break
  if pending or active:time.sleep(3)
 if failed:raise RuntimeError('Technical failure; preserve artifacts/healthy rows, no automatic retry')
 from .report import collect
 result=collect();d.write(d.BASE/'complete.json',dict(status=result['status'],rows=completed,scored=scored,model_seed_count=1,goal_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
 if a.mode and a.row not in {r['row_id'] for r in d.rows()}:raise ValueError('Unregistered row')
 if a.mode=='worker':worker(a.row)
 elif a.mode=='predict':
  from .evaluate import predict
  predict(a.row)
 elif a.mode=='score':
  if not (d.BASE/'source_matrix_decision.json').is_file():raise ValueError('Source decision missing; truth closed')
  from .evaluate import score
  score(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
 else:
  try:dispatch()
  except Exception:
   if d.BASE.is_dir():d.write(d.BASE/'failure.json',dict(status='FAILED',stage='dispatcher',traceback=traceback.format_exc(),healthy_rows_preserved=True,no_automatic_retry=True))
   raise
