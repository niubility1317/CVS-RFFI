from pathlib import Path
import sys
import types

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from experiments.cvs_mixed_iq20.probe_runner import bootstrap
    bootstrap()
    name='experiments.cvs_source_response20';module=types.ModuleType(name)
    module.__path__=[str(root/'experiments/cvs_source_response20')];sys.modules[name]=module
    from experiments.cvs_source_response20.probe import run
    run()
