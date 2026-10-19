"""Use the exact historical family runtime; no copied scientific implementation."""
import argparse,importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap(family=None):
 import json
 p=ROOT/'runtime.json'
 if not p.is_file():p=ROOT/'local_artifacts/selected_seed_runtime.json'
 rt=json.loads(p.read_text(encoding='utf8'));ref=rt['common'] if family is None else rt['references'][family]
 source=Path(rt['project'])/'releases'/ref['release']
 if (source/'release_commit.txt').read_text().strip()!=ref['commit']:raise ValueError('Historical runtime commit differs')
 path=source/'experiments'/ref['package']/'runner.py'
 spec=importlib.util.spec_from_file_location('selected_historical_runner',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
 package=types.ModuleType('experiments.cvs_selected_seed_completion');package.__path__=[str(ROOT/'experiments/cvs_selected_seed_completion')];sys.modules[package.__name__]=package
 # P49 supplies the already verified matching PAOFF parents.
 package=types.ModuleType('experiments.cvs_As_replicate49');package.__path__=[str(ROOT/'experiments/cvs_As_replicate49'),str(Path(rt['project'])/'releases'/rt['common']['release']/'experiments/cvs_noise_shape_factorial43')];sys.modules[package.__name__]=package
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['worker','predict','score','smoke']);parser.add_argument('--row');args=parser.parse_args()
 if args.row:
  # Matrix lookup is pure metadata; initialize no training or query objects.
  package=types.ModuleType('experiments.cvs_selected_seed_completion');package.__path__=[str(ROOT/'experiments/cvs_selected_seed_completion')];sys.modules[package.__name__]=package
  sys.path.insert(0,str(ROOT))
  from experiments.cvs_selected_seed_completion import design
  family=design.row(args.row)['family']
 else:family=None
 bootstrap(family)
 from experiments.cvs_selected_seed_completion.dispatch import main
 main(args)
