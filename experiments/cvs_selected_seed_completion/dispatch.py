"""One immutable matrix owner, bounded capacity, per-row clean truth-last closure."""
import os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def command(mode,row):return [sys.executable,'-u',str(d.ROOT/'experiments/cvs_selected_seed_completion/runner.py'),'--mode',mode,'--row',row]
def execute(mode,row):
 env=dict(os.environ)
 if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
 os.execve(sys.executable,command(mode,row),env)
def worker(r):
 with lock(d.BASE/(r['row_id']+'.owner.lock'),nonblocking=True):
  gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
  if gpu not in d.GPU_IDS:raise ValueError('Unregistered physical GPU')
  while True:
   with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
    state=capacity()[gpu];pids=state['pids']|{os.getpid()};ready=len(pids)<=d.MAX_GPU_WORKERS and state['free_mb']>=12000
    d.write(d.BASE/(r['row_id']+'.capacity.json'),dict(status='READY' if ready else 'WAITING_CAPACITY',gpu=gpu,total_pids=sorted(pids),free_mb=state['free_mb']))
   if ready:break
   time.sleep(5)
  from .binding import train
  train(r)
 execute('predict',r['row_id'])
def dispatch():
 if d.BASE.exists():raise FileExistsError('Reconcile existing owner; never overwrite or relaunch an immutable root')
 d.BASE.mkdir();(d.BASE/'logs').mkdir()
 d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 # Each runtime is isolated, so family-global bindings cannot leak across smoke or training rows.
 smoke_rows=[r for r in d.rows() if r['model_seed']==d.SEEDS[0] or r['family']=='pa']
 for r in smoke_rows:
  with (d.BASE/'logs'/('smoke-'+r['row_id']+'.log')).open('x',encoding='utf8') as log:
   p=subprocess.run(command('smoke',r['row_id']),cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  path=d.BASE/'smoke'/(r['row_id']+'.json')
  if p.returncode or not path.is_file() or d.read(path)['status']!='PASS':
   d.write(d.BASE/'failure.json',dict(status='REAL_CHECKPOINT_SMOKE_FAILED',row_id=r['row_id'],no_automatic_retry=True,preserve_outputs=True));raise RuntimeError('Real checkpoint smoke failed before training')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',rows=[r['row_id'] for r in smoke_rows],NN=2*len(smoke_rows),source_IQ=0,target_query=False,optimizer_updates=0,continue_immediately=True))
 pending=d.rows();active={};launched=[];completed=[];failures=[]
 while pending or active:
  for name,child in list(active.items()):
   code=child.poll()
   if code is None:continue
   done=d.BASE/name/'complete.json'
   if code or not done.is_file() or d.read(done).get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=name,exit_code=code))
   else:completed.append(name)
   del active[name]
  if pending and not failures:
   with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
    states=capacity()
    for gpu in d.GPU_IDS:
     reserved={x['pid'] for x in launched if x['row_id'] in active and x['gpu']==gpu}
     eligible=[r for r in pending if all(x in completed for x in d.dependencies(r))]
     if not eligible:break
     if len(states[gpu]['pids']|reserved)>=d.MAX_GPU_WORKERS or states[gpu]['free_mb']<12000:continue
     r=eligible[0];pending.remove(r);name=r['row_id'];log=d.BASE/'logs'/(name+'.log')
     with log.open('x',encoding='utf8') as f:child=subprocess.Popen(command('worker',name),cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     active[name]=child;launched.append(dict(row_id=name,group=r['group'],gpu=gpu,pid=child.pid,cwd=str(d.ROOT),argv=command('worker',name),log=str(log)))
     d.write(d.BASE/'launch.json',dict(rows=launched))
  d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=[r['row_id'] for r in pending],completed=completed,failures=failures,gpu_ids=list(d.GPU_IDS),max_total_GPU_workers_per_card=d.MAX_GPU_WORKERS,updated_at=time.time()))
  if failures and not active:break
  if pending or active:time.sleep(5)
 if failures:
  d.write(d.BASE/'failure.json',dict(status='TECHNICAL_PARTIAL',failures=failures,pending=pending,preserve_outputs=True,healthy_rows_preserved=True,no_automatic_retry=True));raise RuntimeError('Technical failure preserved; no automatic retry')
 from .report import collect
 collect();d.write(d.BASE/'complete.json',dict(status='ANALYZED',rows=completed,groups=8,model_seed_count=3,views=['clean'],seed01_reused=True))
def main(args):
 if args.mode is None:dispatch();return
 r=d.row(args.row)
 if args.mode=='smoke':
  from .preflight import run
  run(r)
 elif args.mode=='worker':worker(r)
 elif args.mode=='predict':
  from .binding import configured
  configured(r)
  from .evaluate import predict
  predict(r['row_id']);execute('score',r['row_id'])
 elif args.mode=='score':
  from .binding import configured
  configured(r)
  with lock(d.PROJECT/'runs/selected_eight_CPU_scorer.lock'):
   from .evaluate import score
   score(r['row_id'])
  d.write(d.BASE/r['row_id']/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=r['row_id']))
