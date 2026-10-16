import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def bootstrap(roots=None):
    from experiments.cvs_ordinary_fold20.runner import bootstrap as original
    original(roots)
if __name__=='__main__':
    bootstrap()
    from experiments.cvs_origin_jitter20.dispatch import main
    main()
