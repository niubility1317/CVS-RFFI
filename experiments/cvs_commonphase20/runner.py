"""Immutable F backend code only; own B00/B10 models always start scratch."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];backend=Path(json.loads((here/'runtime.json').read_text(encoding='utf8'))['backend'])
 spec=importlib.util.spec_from_file_location('commonphase_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_commonphase20';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_commonphase20')];sys.modules[name]=package
 return here
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_commonphase20 import design as d
 if len(sys.argv)==3 and sys.argv[1]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
  if row is None:raise ValueError('Unregistered source worker')
  from experiments.cvs_commonphase20.source import train
  train(d.config(row))
 elif len(sys.argv)==1:
  from experiments.cvs_commonphase20.dispatch import run
  run()
 else:raise ValueError('No budget/data/phase overrides permitted')
