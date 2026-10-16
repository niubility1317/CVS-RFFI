"""Use the exact immutable model code dependency, never another training run."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap():
    here=Path(__file__).resolve().parents[2]
    backend=Path(json.loads((here/'runtime.json').read_text(encoding='utf8'))['backend'])
    spec=importlib.util.spec_from_file_location('source_phase_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
    name='experiments.cvs_commonphase_source';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_commonphase_source')];sys.modules[name]=package
    return here

if __name__=='__main__':
    if len(sys.argv)!=1:raise ValueError('No phase/source/budget overrides permitted')
    bootstrap()
    from experiments.cvs_commonphase_source.run import run
    run()
