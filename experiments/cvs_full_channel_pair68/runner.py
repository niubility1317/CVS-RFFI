"""Reuse verified SAT18 AS_CE runtime; own fixed-budget independent As tail."""
import importlib.util,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT.parent/'cvs_F_satellite_dropout_E30_20261018_r01'
REFERENCE_COMMIT='735e171fc064094f227fe6807d75836a0b3b0e3a'
def bootstrap():
 if (REFERENCE/'release_commit.txt').read_text().strip()!=REFERENCE_COMMIT:raise ValueError('Pinned source-only AS_CE runtime differs')
 spec=importlib.util.spec_from_file_location('FULL_AS_PAIR68_original_runtime',REFERENCE/'experiments/cvs_satellite_dropout18/runner.py')
 runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
 package=types.ModuleType('experiments.cvs_full_channel_pair68');package.__path__=[str(ROOT/'experiments/cvs_full_channel_pair68')];sys.modules[package.__name__]=package
 return ROOT
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_full_channel_pair68.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_full_channel_pair68.dispatch import main
  main()
