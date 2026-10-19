"""Local checks with the original documented dependency checkout; no data."""
import runpy
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[2]
dependencies = Path('E:/type10-7/code/worktrees/cvs_iq_timing_20261016')
sys.path[:0] = [str(root), str(root / 'code'), str(dependencies / 'code')]
package = types.ModuleType('experiments')
package.__path__ = [str(root / 'experiments'), str(dependencies / 'experiments')]
sys.modules['experiments'] = package
runpy.run_module('experiments.cvs_ce_half_replicate.checks', run_name='__main__')
