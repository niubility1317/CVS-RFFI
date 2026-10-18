"""Bootstrap the actual LRaw30 release before attaching the new E50 namespace."""
import importlib.util,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PARENT_COMMIT='d14450bf1b0d12cc281df1f4d0e79e8760e48282'
def bootstrap():
 rt=json.loads((ROOT/'runtime.json').read_text(encoding='utf-8-sig'));parent=Path(rt['parent_release'])
 if rt['parent_commit']!=PARENT_COMMIT or (parent/'release_commit.txt').read_text().strip()!=PARENT_COMMIT:raise ValueError('Only actual LRaw30 parent runtime')
 spec=importlib.util.spec_from_file_location('LRAW50_PINNED_LRAW30',parent/'experiments/cvs_lraw_tail30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_lraw_tail50';p=types.ModuleType(name);p.__path__=[str(ROOT/'experiments/cvs_lraw_tail50')];sys.modules[name]=p
 return ROOT
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_lraw_tail50.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_lraw_tail50.dispatch import main
  main()
