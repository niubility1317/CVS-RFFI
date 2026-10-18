"""Explicit local dependencies; compile actual native helpers for zero-NN toys."""
import argparse
import json
import sys
import types
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--dependencies', required=True, type=Path)
p.add_argument('--native-source', type=Path)
a = p.parse_args()
w = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(w), str(w / 'code')]
package = types.ModuleType('experiments')
package.__path__ = [str(a.dependencies / 'experiments'), str(w / 'experiments')]
sys.modules['experiments'] = package
package = types.ModuleType('experiments.cvs_temporal_bank67')
package.__path__ = [str(w / 'experiments/cvs_temporal_bank67')]
sys.modules[package.__name__] = package
from experiments.cvs_temporal_bank67.checks import run
source = a.native_source or a.dependencies / 'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
print(json.dumps(run(source)))
