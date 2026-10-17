"""Exact frozen F dependency with new source package;no overrides or retries."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];runtime=json.loads((here/'runtime.json').read_text(encoding='utf8'))
 if set(runtime)!=set(('backend',)):raise ValueError('Only fixed model-code dependency allowed')
 backend=Path(runtime['backend']);spec=importlib.util.spec_from_file_location('shared_direction_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
 for suffix in ('cvs_commonphase_source','cvs_shared_iq17'):
  name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(here/'experiments'/suffix)];sys.modules[name]=package
 return here
if __name__=='__main__':
 if len(sys.argv)!=1:raise ValueError('No model/source/operator/budget overrides permitted')
 bootstrap()
 from experiments.cvs_shared_iq17.run import run
 run()
