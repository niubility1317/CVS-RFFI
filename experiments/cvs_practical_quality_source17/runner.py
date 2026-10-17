"""Install the exact already-used practical engine,never a checkpoint."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 root=Path(__file__).resolve().parents[2];rt=json.loads((root/'runtime.json').read_text());parent=Path(rt['training_release'])
 if (parent/'release_commit.txt').read_text().strip()!=rt['training_commit']:raise ValueError('Pinned practical runtime differs')
 s=importlib.util.spec_from_file_location('pinned_quality_physics_runtime',parent/'experiments/cvs_clock_factorial30/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_practical_quality_source17';p=types.ModuleType(name);p.__path__=[str(root/name.replace('.','/'))];sys.modules[name]=p
if __name__=='__main__':
 bootstrap()
 if len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_practical_quality_source17.checks import run
  from experiments.cvs_practical_quality_source17 import design as d
  d.write(d.ROOT/'checks.json',run())
 elif len(sys.argv)==1:
  from experiments.cvs_practical_quality_source17.run import run
  run()
 else:raise ValueError('Only declared physics diagnostic/checks permitted')
