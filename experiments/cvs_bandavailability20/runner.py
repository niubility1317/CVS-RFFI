"""Immutable old CVS code,isolated guard profiles,ownscratch training only."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def inject():
 name='experiments.cvs_bandavailability20';p=types.ModuleType(name);p.__path__=[str(HERE/'experiments/cvs_bandavailability20')];sys.modules[name]=p

def bootstrap_train():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01')
 if (backend/'release_commit.txt').read_text().strip()!='f788309f5f9e27689ec1e98ae99a6bd83237e1f9':raise ValueError('Wrong unmodified F backend')
 spec=importlib.util.spec_from_file_location('bandavailability_backend',backend/'experiments/cvs_as_timing20/runner.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.bootstrap();inject()

def bootstrap_guard(seed):
 inject()
 from experiments.cvs_bandavailability20 import guard_design as d
 backend=d.resolve(d.MODELS[seed]['backend'])
 if (backend/'release_commit.txt').read_text().strip()!=d.MODELS[seed]['commit']:raise ValueError('Wrong own frozen guard runtime')
 spec=importlib.util.spec_from_file_location('availability_guard_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.bootstrap();inject()

if __name__=='__main__':
 args=sys.argv[1:]
 if len(args)==2 and args[0]=='--guard-seed' and args[1] in ('2026092701','2026092702'):
  seed=int(args[1]);bootstrap_guard(seed)
  from experiments.cvs_bandavailability20.guard import run
  run(seed)
 elif not args or len(args)==2 and args[0]=='--worker' and args[1] in ('RAW-RX1-s2026092701','GAIN-RX1-s2026092701','ORDINARY-RX1-s2026092701','MASK-RX1-s2026092701'):
  bootstrap_train()
  from experiments.cvs_bandavailability20 import design as d
  if args:
   from experiments.cvs_bandavailability20.source import train
   train(d.config(next(r for r in d.rows() if r['row_id']==args[1])))
  else:
   from experiments.cvs_bandavailability20.dispatch import run
   run()
 else:raise ValueError('No role/seed/budget/operator overrides')
