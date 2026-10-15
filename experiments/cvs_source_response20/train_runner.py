"""Select audited original modules, then substitute only this stage's adapters."""
from pathlib import Path
import sys
import types


def bootstrap(roots=None):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from experiments.cvs_mixed_iq20.train_runner import bootstrap as original
    original(roots)
    name='experiments.cvs_source_response20';module=types.ModuleType(name)
    module.__path__=[str(root/'experiments/cvs_source_response20')];sys.modules[name]=module
    from experiments.cvs_source_response20 import train_design
    package=sys.modules['experiments.cvs_mixed_iq20']
    sys.modules['experiments.cvs_mixed_iq20.train_design']=train_design;package.train_design=train_design
    from experiments.cvs_source_response20 import train_model,train_observer
    sys.modules['experiments.cvs_mixed_iq20.train_model']=train_model;package.train_model=train_model
    sys.modules['experiments.cvs_mixed_iq20.train_observer']=train_observer;package.train_observer=train_observer


if __name__=='__main__':
    bootstrap()
    from experiments.cvs_source_response20.train_dispatch import main
    main()
