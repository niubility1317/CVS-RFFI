import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def worker(row):
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
  while True:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    state=capacity()[gpu]
    if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
   time.sleep(3)
  from .fit import run as fit
  from .predict import run as predict
  seed=int(row);fit(seed);predict(seed)
 os.execve(sys.executable,[sys.executable,'-u','-m','experiments.cvs_report_projection.dispatch','--score',row],dict(os.environ,CUDA_VISIBLE_DEVICES=''))

def dispatch():
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
  spec=d.read(d.ROOT/'registration/experiment.json')
  if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Preregistration matrix differs')
  spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());d.write(d.BASE/'experiment.json',spec)
  d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
  pending=[dict(row_id=str(seed)) for seed in d.SEEDS];active={};receipts=[];completed=[];failures=[]
  while pending or active:
   for rowid,child in list(active.items()):
    code=child.poll()
    if code is None:continue
    p=d.BASE/('s'+rowid)/'completion.json';v=d.read(p) if p.is_file() else {}
    if code or v.get('status')!='SEED_COMPLETE' or v.get('views')!=['clean'] or not v.get('truth_last'):failures.append(dict(row_id=rowid,exit_code=code,error='Source row completion differs'))
    else:completed.append(rowid)
    del active[rowid]
   if pending and not failures:
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
     states=capacity();reserved={r['pid']:r['gpu'] for r in receipts if r['row_id'] in active};ready=[]
     for gpu,state in states.items():
      pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
      if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
     if ready:
      gpu=min(ready)[2];row=pending.pop(0);cmd=[sys.executable,'-u','-m','experiments.cvs_report_projection.dispatch','--worker',row['row_id']];log=d.BASE/'logs'/f"{row['row_id']}.log"
      with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=receipts))
   d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,max_total_processes_per_GPU=3,updated_at=time.time()))
   if failures and not active:break
   if pending or active:time.sleep(3)
  if failures:
   d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('Source pilot failed; artifacts retained')
  if len(completed)!=3:raise ValueError('Incomplete source RX matrix')
  from .score import finish
  a=finish();d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',seeds=completed,models=24,views=['clean'],prediction_count=4032000,truth_last=True,goal_84=a['goal_84']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker)
 elif a.score:
  from .score import run
  run(int(a.score))
 else:dispatch()
