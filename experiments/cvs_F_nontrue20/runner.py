"""Mount the verified original source runtime; never inherit its RDrop/SupCon model."""
import importlib.util,json,sys,types
from pathlib import Path
def namespace(n,p):
 m=types.ModuleType(n);m.__path__=[str(p)];sys.modules[n]=m
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned runtime commit differs')
 spec=importlib.util.spec_from_file_location('_nontrue_runtime',ref/'experiments/cvs_fullsource_concat20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 namespace('experiments.cvs_F_nontrue20',here/'experiments/cvs_F_nontrue20');namespace('experiments.cvs_sameiq_experts',here/'experiments/cvs_sameiq_experts')
 return here
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_F_nontrue20.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_F_nontrue20.dispatch import main
  main()
