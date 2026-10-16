import argparse,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 from experiments.cvs_ordinary_fold20.runner import bootstrap
 bootstrap()
 from experiments.cvs_f_gradient_interference import design as d
 from experiments.cvs_receiver_residual_capacity4.control import lock
 from experiments.cvs_multi_state_action.dispatch import capacity
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
 if a.worker:
  with lock(d.BASE/'worker.owner.lock',nonblocking=True):
   gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
   from experiments.cvs_f_gradient_interference.audit import run
   try:run('cuda:0')
   except Exception as e:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
  return
 with lock(d.ROOT/'gradient_diagnostic.owner.lock',nonblocking=True):
  if d.BASE.exists():raise FileExistsError('Reconcile immutable diagnostic output; no duplicate launch')
  spec=d.read(d.ROOT/'experiments/cvs_f_gradient_interference/registration.json')
  if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered diagnostic configuration differs')
  from experiments.cvs_f_gradient_interference.audit import references
  references();d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER));child=None
  while child is None:
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
    if choices:
     gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_gradient_interference/runner.py'),'--worker'];log=d.BASE/'logs/worker.log'
     with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     d.write(d.BASE/'launch.json',dict(pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
   if child is None:time.sleep(3)
  code=child.wait()
  if code or not (d.BASE/'completion.json').is_file():
   if not (d.BASE/'failure.json').exists():d.write(d.BASE/'failure.json',dict(status='FAILED',exit_code=code,preserve_outputs=True,no_automatic_retry=True))
   raise RuntimeError('Frozen source diagnostic failed')
if __name__=='__main__':main()
