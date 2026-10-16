import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cvs_source_ellipticity.run import run
if __name__=='__main__':run()
