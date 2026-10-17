"""Reuse immutable F/default head and original IQ pipeline,never cycle Ag."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
REF=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cycleorder_factorial_E30_20261017_r01')
def bootstrap():
 if (REF/'release_commit.txt').read_text().strip()!='0c47f4bcd49c5df43093579c7f33e6c35bc5245d':raise ValueError('Matched original implementation differs')
 spec=importlib.util.spec_from_file_location('cleanretain37_dependencies',REF/'experiments/cvs_cycleorder_factorial32/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 old=types.ModuleType('experiments.cvs_cycleorder_factorial32');old.__path__=[str(REF/'experiments/cvs_cycleorder_factorial32')];sys.modules[old.__name__]=old
 own=types.ModuleType('experiments.cvs_cleanretain37');own.__path__=[str(HERE/'experiments/cvs_cleanretain37')];sys.modules[own.__name__]=own
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_cleanretain37.dispatch import main
 main()
