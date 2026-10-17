"""Reuse exact full-source F runtime;source diagnostics never import target evaluator."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text())
 reference=Path(rt['reference_release'])
 if (reference/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned full-source runtime differs')
 spec=importlib.util.spec_from_file_location('N0_fullsource_bootstrap',reference/'experiments/cvs_fullsource_concat20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_n0_source17';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_n0_source17')];sys.modules[name]=p
 return here
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='--checks':
  here=bootstrap();sys.path.insert(0,str(here))
  from experiments.cvs_n0_source17.checks import run
  (here/'checks.json').write_text(json.dumps(run(),indent=2)+'\n')
 elif len(sys.argv)==1:
  bootstrap()
  from experiments.cvs_n0_source17.worker import run
  run()
 else:raise ValueError('Fixed source guard/checks only')
