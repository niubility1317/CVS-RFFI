"""Fixed verified RX6 F backend; ownsource-only scratchE20 launcher."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01')
 if (backend/'release_commit.txt').read_text().strip()!='367b2f945ef65488e6d97eddd2357388d2477100':raise ValueError('Wrong immutable RX6 backend')
 spec=importlib.util.spec_from_file_location('propgroup_RX6_backend',backend/'experiments/cvs_f_additive_rx6/runner.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.bootstrap()
 name='experiments.cvs_environment_group20';p=types.ModuleType(name);p.__path__=[str(HERE/'experiments/cvs_environment_group20')];sys.modules[name]=p
if __name__=='__main__':
 args=sys.argv[1:]
 if args and not (len(args)==2 and args[0]=='--worker' and args[1] in ('RAW-RX6-s2026092701','IID-RX6-s2026092701','SCENE_GROUP-RX6-s2026092701','PROP_GROUP-RX6-s2026092701')):raise ValueError('No role/policy/seed/budget overrides')
 bootstrap()
 from experiments.cvs_environment_group20 import design as d
 if args:
  from experiments.cvs_environment_group20.source import train
  train(d.config(next(r for r in d.rows() if r['row_id']==args[1])))
 else:
  from experiments.cvs_environment_group20.dispatch import run
  run()
