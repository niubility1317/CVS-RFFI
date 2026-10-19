"""Bootstrap the fixed P43 runtime and reuse immutable common reporting helpers."""
import importlib.util,json,sys,types
from pathlib import Path
def namespace(name,path):
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text())
 ref=Path(rt['reference_release'])
 if rt['reference_commit']!='19aedc3604fed9c193237dd92366164178af0983' or (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Only original P43 runtime')
 spec=importlib.util.spec_from_file_location('_sameIQ_CSIL_P43',ref/'experiments/cvs_noise_shape_factorial43/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 namespace('experiments.cvs_noise_shape_factorial43',ref/'experiments/cvs_noise_shape_factorial43')
 namespace('experiments.cvs_signal_bridge30',Path(rt['project'])/'releases/cvs_FcrossRX_signalBridge_E30_20261019_r01/experiments/cvs_signal_bridge30')
 for name in ('cvs_sameiq_distill30','cvs_sameiq_experts'):namespace('experiments.'+name,here/'experiments'/name)
 return here
if __name__=='__main__':
 bootstrap()
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_sameiq_distill30.checks import run
  print(json.dumps(run()))
 else:
  from experiments.cvs_sameiq_distill30.dispatch import main
  main()
