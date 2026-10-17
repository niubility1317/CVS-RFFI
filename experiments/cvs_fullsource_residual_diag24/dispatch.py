"""One immutable root/owner. No training, no target scoring, no automatic retry."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
 if a.worker:
  with lock(d.BASE/'worker.owner.lock',nonblocking=True):
   gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
   with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
    if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Total GPU process cap3 includes all owners')
   from .probe import run
   try:run('cuda:0')
   except BaseException as e:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
  return
 if d.BASE.exists():raise FileExistsError('Reconcile existing run, never duplicate')
 d.BASE.mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 while True:
  with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
   slots=[(len(s['pids']),-s['free_mb'],g) for g,s in capacity().items() if len(s['pids'])<3 and s['free_mb']>=12000]
   if slots:
    gpu=min(slots)[2];args=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_fullsource_residual_diag24/runner.py'),'--worker']
    with (d.BASE/'worker.log').open('x',encoding='utf8') as log:child=subprocess.Popen(args,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    d.write(d.BASE/'launch.json',dict(pid=child.pid,start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),gpu=gpu,cwd=str(d.ROOT),argv=args,owner=d.OWNER));break
  d.write(d.BASE/'queue.json',dict(status='WAIT_CAPACITY',training_updates=0,target_access=False));time.sleep(5)
 while child.poll() is None:time.sleep(3)
 done=d.BASE/'diagnostic/completion.json'
 if child.returncode or not done.is_file() or d.read(done)['status']!='SOURCE_DIAGNOSTIC_COMPLETE':
  if not (d.BASE/'failure.json').exists():d.write(d.BASE/'failure.json',dict(status='FAILED',exit_code=child.returncode,preserve_outputs=True,no_automatic_retry=True))
  raise RuntimeError('Frozen-source diagnostic failed; preserve, no automatic restart')
 d.write(d.BASE/'complete.json',dict(status='ANALYZED',source_only=True,training_updates=0,target_access=False,goal_complete=False))
