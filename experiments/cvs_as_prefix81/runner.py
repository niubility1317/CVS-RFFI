"""Pin committed LRaw runtime, which pins the original P43 dependency."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text(encoding='utf8'))
 if rt['reference_commit']!='d14450bf1b0d12cc281df1f4d0e79e8760e48282':raise ValueError('Only committed LRaw runtime is supported')
 reference=Path(rt['reference_release'])
 if (reference/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned LRaw runtime differs')
 spec=importlib.util.spec_from_file_location('PREFIX81_LRAW_bootstrap',reference/'experiments/cvs_lraw_tail30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_as_prefix81';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_as_prefix81')];sys.modules[name]=p
 return here
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_as_prefix81.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_as_prefix81.dispatch import main
  main()
