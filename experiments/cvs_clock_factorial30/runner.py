"""Reuse the frozen full-source F runtime, with only this IQ-slot implementation new."""
import importlib.util,json,os,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());reference=Path(rt['clock_release'])
 spec=importlib.util.spec_from_file_location('pinned_clock20_runner',reference/'experiments/cvs_clock_factorial20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_clock_factorial30';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_clock_factorial30')];sys.modules[name]=p
 return here

if __name__=='__main__':
 bootstrap()
 from experiments.cvs_clock_factorial30 import design as d
 if len(sys.argv)==1:
  from experiments.cvs_clock_factorial30.dispatch import run
  run()
 elif len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_clock_factorial30.checks import run
  d.write(d.ROOT/'synthetic_checks.json',run())
 elif len(sys.argv)==3 and sys.argv[1] in ('--worker','--score'):
  rid=sys.argv[2];c=d.config(next(r for r in d.train_rows() if r['row_id']==rid))
  if sys.argv[1]=='--worker':
   from experiments.cvs_clock_factorial30.source import train
   from experiments.cvs_clock_factorial30.evaluate import predict
   import gc,torch
   train(c);gc.collect();torch.cuda.empty_cache();predict(rid)
   os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--score',rid])
  else:
   from experiments.cvs_clock_factorial30.evaluate import score
   score(rid)
 else:raise ValueError('Unregistered execution override')
