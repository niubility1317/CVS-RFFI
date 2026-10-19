import importlib.util,json,os,sys,time,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned pure F runtime differs')
 spec=importlib.util.spec_from_file_location('_source_sensitivity_boot',ref/'experiments/cvs_F_nontrue20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 p=types.ModuleType('experiments.cvs_sameiq_sensitivity');p.__path__=[str(here/'experiments/cvs_sameiq_sensitivity')];sys.modules[p.__name__]=p
 return here
def dispatch():
 from experiments.cvs_sameiq_sensitivity import design as d
 from experiments.cvs_multi_state_action.dispatch import capacity
 from experiments.cvs_receiver_residual_capacity4.control import lock
 if os.geteuid()==0:raise ValueError('Ordinary account required')
 if d.BASE.exists():raise FileExistsError('Immutable source probe exists; reconcile,no overwrite')
 d.BASE.mkdir();d.write(d.BASE/'resolved_config.json',d.config())
 while True:
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   states=capacity();slots=[(len(s['pids']),-s['free_mb'],g) for g,s in states.items() if len(s['pids'])<2 and s['free_mb']>=12000]
   if slots:
    gpu=min(slots)[2];cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'];d.write(d.BASE/'allocation.json',dict(pid=os.getpid(),gpu=gpu,cwd=str(d.ROOT),argv=cmd,owner=d.OWNER))
    os.execve(sys.executable,cmd,dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
  d.write(d.BASE/'queue.json',dict(status='WAITING_CAPACITY',pid=os.getpid(),max_processes_per_GPU=2));time.sleep(5)
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_sameiq_sensitivity.checks import run
  print(json.dumps(run()))
 elif sys.argv[1:]==['--worker']:
  from experiments.cvs_sameiq_sensitivity.worker import run
  run()
 elif not sys.argv[1:]:dispatch()
 else:raise ValueError('Only registered source probe/checks/worker')
