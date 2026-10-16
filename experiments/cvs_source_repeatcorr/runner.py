import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if __name__=='__main__':
    from experiments.cvs_source_repeatcorr.run import run
    run()
