"""Reuse the audited native bootstrap, then select this confirmation package."""
from pathlib import Path
import sys
import types


def bootstrap(roots=None):
    from experiments.cvs_ordinary_fold20.runner import bootstrap as original
    original(roots)
    name='experiments.cvs_ordinary_confirm20';module=types.ModuleType(name)
    module.__path__=[str(Path(__file__).resolve().parent)];sys.modules[name]=module


if __name__=='__main__':
    bootstrap()
    from experiments.cvs_ordinary_confirm20.dispatch import main
    main()
