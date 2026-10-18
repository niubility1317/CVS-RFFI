"""Bootstrap actual seed02/03 parent release, then reuse bundled fixed E50 runtime."""
import importlib.util,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PARENT_COMMIT='63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47'

def bootstrap():
 rt=json.loads((ROOT/'runtime.json').read_text())
 if rt['reference_commit']!=PARENT_COMMIT:raise ValueError('Only actual seed02/03 FINAL30 runtime')
 reference=Path(rt['reference_release'])
 if (reference/'release_commit.txt').read_text().strip()!=PARENT_COMMIT:raise ValueError('Pinned seed02/03 parent runtime differs')
 spec=importlib.util.spec_from_file_location('PRACTICAL_BUDGET50_REPLICATION_parent',reference/'experiments/cvs_As_replicate49/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 for name,paths in [('experiments.cvs_practical_budget50',[ROOT/'experiments/cvs_practical_budget50']),('experiments.cvs_practical_budget50_replication',[ROOT/'experiments/cvs_practical_budget50_replication',ROOT/'experiments/cvs_practical_budget50'])]:
  p=types.ModuleType(name);p.__path__=[str(v) for v in paths];sys.modules[name]=p
 return ROOT
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_practical_budget50_replication.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_practical_budget50_replication.dispatch import main
  main()
