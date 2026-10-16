"""Known immutable backend per existing seed; no arbitrary execution overrides."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap(seed):
 if seed not in (2026092701,2026092702):raise ValueError('Unregistered source model seed')
 name='experiments.cvs_f_noise_pair_probe';package=types.ModuleType(name);package.__path__=[str(HERE/'experiments/cvs_f_noise_pair_probe')]
 # Bind own constants before importing backend packages.
 sys.path.insert(0,str(HERE));import experiments
 sys.modules[name]=package
 from experiments.cvs_f_noise_pair_probe import design as d
 ref=d.REFERENCES[seed];backend=Path(d.PROJECT)/'releases'/ref['backend']
 if (backend/'release_commit.txt').read_text().strip()!=ref['commit']:raise ValueError('Wrong immutable source baseline backend')
 spec=importlib.util.spec_from_file_location('noise_pair_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 sys.modules[name]=package
 from experiments.cvs_additive_as20 import design as original
 if original.SEED!=seed or original.base_model_config()['model_seed']!=seed:raise ValueError('Actual F constructor seed differs')
 return d
if __name__=='__main__':
 args=sys.argv[1:]
 if args and not (len(args)==2 and args[0]=='--worker' and args[1] in ('frozen-D_raw-s2026092701','frozen-D_raw-s2026092702')):raise ValueError('No source/noise/budget/checkpoint overrides allowed')
 seed=int(args[1][-10:]) if args else 2026092701;d=bootstrap(seed)
 if args:
  from experiments.cvs_f_noise_pair_probe.probe import run
  row=next(r for r in d.rows() if r['row_id']==args[1])
  try:run(row,'cuda:0')
  except Exception as e:d.write(d.BASE/(row['row_id']+'.failure.json'),dict(status='FAILED',error=repr(e),no_automatic_retry=True,preserve_outputs=True));raise
 else:
  from experiments.cvs_f_noise_pair_probe.dispatch import run
  run()
