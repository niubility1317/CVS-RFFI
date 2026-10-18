"""Each fixed trajectory completes both stages unconditionally; scores never choose continuation."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
RUNNER=d.ROOT/'experiments/cvs_practical_as_weight_replicate50/runner.py'

def execute(mode,row,budget,gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='score' else str(gpu))
 os.execve(sys.executable,[sys.executable,'-u',str(RUNNER),'--mode',mode,'--row',row,'--budget',str(budget),'--gpu',str(gpu)],env)

def wait_gpu(gpu):
 while True:
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   state=capacity()[gpu]
   if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:return
  time.sleep(3)

def worker(row,budget,gpu):
 entry=next(r for r in d.rows() if r['row_id']==row);c=d.config(entry,budget)
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  wait_gpu(gpu)
  from .observer import train
  import gc,torch
  train(c);gc.collect();torch.cuda.empty_cache()
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .qualification import run as qualify
 try:
  with numerical_context(d.FULL_FP32_POLICY):qualify(row,budget)
 except Exception as error:d.write(d.stage_root(entry,budget)/'source_qualification_failure.json',dict(status='SOURCE_READOUT_FAILED',error=repr(error),traceback=traceback.format_exc(),formal_still_required=True,no_automatic_retry=True))
 execute('predict',row,budget,gpu)

def dispatch():
 if d.BASE.exists():raise FileExistsError('Existing immutable run;reconcile,never duplicate')
 d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 from .preflight import smoke
 try:smoke()
 except BaseException as error:d.write(d.BASE/'failure.json',dict(status='PREFLIGHT_FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
 pending=d.rows();active={};receipts=[];completed=[];failed=[]
 while pending or active:
  for row,child in list(active.items()):
   code=child.poll()
   if code is None:continue
   done=d.BASE/row/'complete.json'
   if code or not done.is_file() or d.read(done).get('status')!='BOTH_STAGES_TRAINED_AND_SCORED':failed.append(dict(row_id=row,exit_code=code))
   else:completed.append(row)
   del active[row]
  if pending and not failed:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    states=capacity();slots=[]
    for gpu,state in states.items():
     reserved={r['pid'] for r in receipts if r['gpu']==gpu and r['row_id'] in active};total=state['pids']|reserved
     if len(total)<3 and state['free_mb']>=12000:slots.append((len(total),-state['free_mb'],gpu))
    if slots:
     gpu=min(slots)[2];row=pending.pop(0);cmd=[sys.executable,'-u',str(RUNNER),'--mode','worker','--row',row['row_id'],'--budget','30','--gpu',str(gpu)];log=d.BASE/'logs'/(row['row_id']+'.log')
     with log.open('x',encoding='utf8') as stream:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=receipts))
  d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failed,max_total_processes_per_GPU=3,updated_at=time.time()))
  if failed and not active:break
  if pending or active:time.sleep(3)
 if failed:d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,unlaunched=pending,healthy_rows_preserved=True,no_automatic_retry=True));raise RuntimeError('Technical failure preserved')
 from .report import collect
 summary=collect();d.write(d.BASE/'complete.json',dict(status=summary['status'],model_seed_count=2,new_trajectories=4,new_stage_rows=8,rows=completed,goal_complete=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['worker','predict','score']);parser.add_argument('--row');parser.add_argument('--budget',type=int,choices=[30,50]);parser.add_argument('--gpu',type=int);args=parser.parse_args()
 if args.mode is None:return dispatch()
 entry=next((r for r in d.rows() if r['row_id']==args.row),None)
 if entry is None or args.budget not in (30,50) or args.gpu is None:raise ValueError('Only registered fixed stage execution')
 if args.mode=='worker':worker(args.row,args.budget,args.gpu)
 elif args.mode=='predict':
  from .evaluate import predict
  predict(args.row,args.budget);execute('score',args.row,args.budget,args.gpu)
 else:
  from .evaluate import score
  score(args.row,args.budget);d.write(d.stage_root(entry,args.budget)/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=args.row,target_budget=args.budget))
  if args.budget==30:execute('worker',args.row,50,args.gpu)
  else:d.write(d.BASE/args.row/'complete.json',dict(status='BOTH_STAGES_TRAINED_AND_SCORED',row_id=args.row,budgets=[30,50],scientific_continuation_unconditional=True))
