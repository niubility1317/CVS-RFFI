"""Named prediction reservation visible to existing N607 dispatchers."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
if __name__ == '__main__':
    runpy.run_module('experiments.cvs_phase1_repair.evaluate', run_name='__main__')
