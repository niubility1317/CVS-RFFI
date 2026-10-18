"""Bootstrap the actual weighted30 release before attaching the new E50 namespace."""
import importlib.util,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PARENT_COMMIT='3befd47c8041d81f9430eda53562d0821624ff04'
def bootstrap():
 rt=json.loads((ROOT/'runtime.json').read_text(encoding='utf-8-sig'));parent=Path(rt['parent_release'])
 if rt['parent_commit']!=PARENT_COMMIT or (parent/'release_commit.txt').read_text().strip()!=PARENT_COMMIT:raise ValueError('Only actual weighted30 parent runtime')
 spec=importlib.util.spec_from_file_location('ASWEIGHT50_PINNED_WEIGHT30',parent/'experiments/cvs_practical_as_weight30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_practical_as_weight50';p=types.ModuleType(name);p.__path__=[str(ROOT/'experiments/cvs_practical_as_weight50')];sys.modules[name]=p
 return ROOT
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_practical_as_weight50.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_practical_as_weight50.dispatch import main
  main()
