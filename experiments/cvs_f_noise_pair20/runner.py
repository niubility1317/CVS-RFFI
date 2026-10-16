"""Immutable known F backend only;all three new models start scratch."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap(backend=None):
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01' if backend is None else backend)
 if (backend/'release_commit.txt').read_text().strip()!='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b':raise ValueError('Wrong immutable sourceF seed01 backend')
 spec=importlib.util.spec_from_file_location('noisepair_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 for name in ('cvs_f_noise_pair20','cvs_f_noise_pair_probe'):
  package=types.ModuleType('experiments.'+name);package.__path__=[str(HERE/'experiments'/name)];sys.modules['experiments.'+name]=package
 from experiments.cvs_f_noise_pair20 import design as d
 if d.parent.SEED!=2026092701 or d.parent.base_model_config()['model_seed']!=2026092701:raise ValueError('ActualF constructor seed1 required')
 return d
if __name__=='__main__':
 args=sys.argv[1:]
 if args and not (len(args)==2 and args[0]=='--worker' and args[1] in ('RAW10-RX1-s2026092701','INDEP-RX1-s2026092701','ANTI-RX1-s2026092701')):raise ValueError('No data/noise/loss/epoch/seed overrides allowed')
 d=bootstrap()
 if args:
  from experiments.cvs_f_noise_pair20.source import train
  train(d.config(next(r for r in d.rows() if r['row_id']==args[1])))
 else:
  from experiments.cvs_f_noise_pair20.dispatch import run
  run()
