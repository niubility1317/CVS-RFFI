"""Original immutable F runtime, own scratch E20 workers; no role/budget overrides."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];runtime=json.loads((here/'runtime.json').read_text(encoding='utf8'))
 if set(runtime)!={'backend'}:raise ValueError('Only fixed F runtime dependency permitted')
 backend=Path(runtime['backend']);spec=importlib.util.spec_from_file_location('scene_cycle_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 for suffix in ('cvs_scene_cycle20','cvs_supcon_views20'):
  name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(here/'experiments'/suffix)];sys.modules[name]=package
 return here
if __name__=='__main__':
 if not (len(sys.argv)==1 or (len(sys.argv)==3 and sys.argv[1]=='--worker' and sys.argv[2] in ('Q_sat-RX1-s2026092701','Q_mix-RX1-s2026092701'))):raise ValueError('No budget/data/operator/seed overrides permitted')
 bootstrap()
 from experiments.cvs_supcon_views20 import design as d
 if len(sys.argv)==3:
  row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
  if row is None:raise ValueError('Unregistered source worker')
  from experiments.cvs_supcon_views20.source import train
  train(d.config(row))
 else:
  from experiments.cvs_supcon_views20.dispatch import run
  run()
