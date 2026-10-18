"""Pinned SAT18 bootstrap retains original full-source F/P43 native runtime."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text())
 if rt['reference_commit']!='19aedc3604fed9c193237dd92366164178af0983' or rt['satellite_reference_commit']!='735e171fc064094f227fe6807d75836a0b3b0e3a':raise ValueError('Only pinned P43/SAT18 runtime')
 reference=Path(rt['reference_release']);satellite=Path(rt['satellite_reference_release'])
 if (reference/'release_commit.txt').read_text().strip()!=rt['reference_commit'] or (satellite/'release_commit.txt').read_text().strip()!=rt['satellite_reference_commit']:raise ValueError('Pinned source runtime differs')
 sat_runtime=json.loads((satellite/'runtime.json').read_text())
 if sat_runtime['reference_release']!=str(reference) or sat_runtime['reference_commit']!=rt['reference_commit'] or sat_runtime['project']!=rt['project']:raise ValueError('SAT18 upstream runtime/project differs')
 spec=importlib.util.spec_from_file_location('SATELLITE_BUDGET62_SAT18_bootstrap',satellite/'experiments/cvs_satellite_dropout18/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_satellite_budget62';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_satellite_budget62')];sys.modules[name]=p
 return here
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_satellite_budget62.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_satellite_budget62.dispatch import main
  main()
