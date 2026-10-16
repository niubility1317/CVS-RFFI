"""One source-only worker after a shared-capacity admission."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def bootstrap(roots=None):
    from experiments.cvs_ordinary_confirm20.runner import bootstrap as original
    original(roots)

if __name__=='__main__':
    bootstrap()
    if '--worker' in sys.argv:
        from experiments.cvs_ordinary_pa_source.run import run
        run()
    else:
        from experiments.cvs_ordinary_pa_source.dispatch import dispatch
        dispatch()
