"""Pinned weighted algorithm plus actual02/03 factory runtime; local preparation only."""
import importlib.util,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def boot(path,module,label):
 spec=importlib.util.spec_from_file_location(label,Path(path)/module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
def bootstrap():
 rt=json.loads((ROOT/'runtime.json').read_text(encoding='utf-8-sig'))
 expected={'originalreplication_commit':'63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47','weighted50_commit':'64176e7ff1aa5cd2477a504f8b442bb7966079dc','weighted30_commit':'3befd47c8041d81f9430eda53562d0821624ff04'}
 for key,commit in expected.items():
  release=Path(rt[key.replace('_commit','_release')])
  if rt[key]!=commit or (release/'release_commit.txt').read_text().strip()!=commit:raise ValueError('Only pinned actual weighted algorithm and originalseed factory runtime')
 boot(rt['weighted50_release'],'experiments/cvs_practical_as_weight50/runner.py','WEIGHT_REPLICATION_WEIGHT50')
 boot(rt['originalreplication_release'],'experiments/cvs_As_replicate49/runner.py','WEIGHT_REPLICATION_TRUESEED')
 name='experiments.cvs_practical_as_weight_replicate50';p=types.ModuleType(name);p.__path__=[str(ROOT/'experiments/cvs_practical_as_weight_replicate50')];sys.modules[name]=p
 return ROOT
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_practical_as_weight_replicate50.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_practical_as_weight_replicate50.dispatch import main
  main()
