import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned original F runtime differs')
 spec=importlib.util.spec_from_file_location('_feature_adapters_runtime',ref/'experiments/cvs_F_nontrue20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 namespace=types.ModuleType('experiments.cvs_feature_adapters30');namespace.__path__=[str(here/'experiments/cvs_feature_adapters30')];sys.modules[namespace.__name__]=namespace
 return here

if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_feature_adapters30.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_feature_adapters30.dispatch import main
  main()
