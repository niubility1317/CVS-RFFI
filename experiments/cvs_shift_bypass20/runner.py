"""Immutable F backend code only; own B00/B10 models always start scratch."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];backend=Path(json.loads((here/'runtime.json').read_text(encoding='utf8'))['backend'])
 spec=importlib.util.spec_from_file_location('shift_bypass_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_shift_bypass20';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_shift_bypass20')];sys.modules[name]=package
 return here
if __name__=='__main__':
 if not (len(sys.argv)==1 or (len(sys.argv)==3 and sys.argv[1]=='--worker' and sys.argv[2] in ('B00-RX1-s2026092701','B10-RX1-s2026092701'))):raise ValueError('No budget/data/shift overrides permitted')
 bootstrap()
 from experiments.cvs_shift_bypass20 import design as d
 if len(sys.argv)==3 and sys.argv[1]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
  if row is None:raise ValueError('Unregistered source worker')
  from experiments.cvs_shift_bypass20.source import train
  train(d.config(row))
 elif len(sys.argv)==1:
  from experiments.cvs_shift_bypass20.dispatch import run
  run()
 else:raise ValueError('No budget/data/PA overrides permitted')
