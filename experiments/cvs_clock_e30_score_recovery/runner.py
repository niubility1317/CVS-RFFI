"""Pinned trained runtime;zero optimizer or training entry point."""
import importlib.util,json,os,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());parent=Path(rt['training_release'])
 spec=importlib.util.spec_from_file_location('pinned_E30_trained_runner',parent/'experiments/cvs_clock_factorial30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_clock_e30_score_recovery';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_clock_e30_score_recovery')];sys.modules[name]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_clock_e30_score_recovery import design as d
 if len(sys.argv)==1:
  from experiments.cvs_clock_e30_score_recovery.dispatch import run
  run()
 elif len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_clock_e30_score_recovery.checks import run
  d.write(d.ROOT/'synthetic_checks.json',run())
 elif len(sys.argv)==3 and sys.argv[1] in ('--worker','--score'):
  rid=sys.argv[2]
  if rid not in [r['row_id'] for r in d.rows()]:raise ValueError('Unregistered score row')
  from experiments.cvs_clock_e30_score_recovery.evaluate import predict,score
  if sys.argv[1]=='--worker':
   from experiments.cvs_clock_e30_score_recovery.proof import recover
   c=d.config(next(r for r in d.rows() if r['row_id']==rid))
   if not (d.BASE/rid/'source_recovery.json').is_file():recover(c)
   predict(rid);os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--score',rid])
  else:score(rid)
 else:raise ValueError('Only fixed checkpoint prediction/scoring;no training')
