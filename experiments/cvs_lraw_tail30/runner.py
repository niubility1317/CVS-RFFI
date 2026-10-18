"""Reuse exact full-source F runtime;source diagnostics never import target evaluator."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text())
 if rt['reference_commit']!='19aedc3604fed9c193237dd92366164178af0983':raise ValueError('Only fixed P43 parent runtime')
 reference=Path(rt['reference_release'])
 if (reference/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned full-source runtime differs')
 spec=importlib.util.spec_from_file_location('LRAW30_P43_bootstrap',reference/'experiments/cvs_noise_shape_factorial43/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_lraw_tail30';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_lraw_tail30')];sys.modules[name]=p
 return here
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_lraw_tail30.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_lraw_tail30.dispatch import main
  main()
