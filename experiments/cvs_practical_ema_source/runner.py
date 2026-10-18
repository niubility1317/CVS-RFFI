"""Bootstrap the unchanged pinned E50 runtime before importing its model API."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2]
 r=json.loads((here/'runtime.json').read_text(encoding='utf-8-sig'))
 if r.get('run_id')!='20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r03' or r.get('parent_commit')!='0f399c910324e21a8b1a60346d879be5c2964a9c':raise ValueError('Fixed source-only run/parent required')
 parent=Path(r['parent_release'])
 if not Path(r['project']).is_absolute() or parent!=Path(r['project'])/'releases'/'cvs_FcrossRX_As_budget_E50_20261018_r01' or (parent/'release_commit.txt').read_text().strip()!=r['parent_commit']:raise ValueError('Pinned original E50 release differs')
 spec=importlib.util.spec_from_file_location('OWN_EMA_PINNED_E50_BOOTSTRAP',parent/'experiments/cvs_practical_budget50/runner.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_practical_ema_source';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_practical_ema_source')];sys.modules[name]=p
 return here

if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_practical_ema_source.checks import run
  print(json.dumps(run()))
 elif not sys.argv[1:]:
  from experiments.cvs_practical_ema_source.evaluate import run
  run()
 else:raise ValueError('Only fixed source-only run or focused checks')
