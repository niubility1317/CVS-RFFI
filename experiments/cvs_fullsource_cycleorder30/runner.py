"""Reuse immutable F implementation; first actual operation is CP smoke."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
BASE=Path('/home/szu2070436088/2510044040/CV-SincNet')
REF=BASE/'releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01'
def bootstrap():
 if (REF/'release_commit.txt').read_text().strip()!='bafba40e365d0914a2e86f861fb55a49f1cd3997':raise ValueError('Immutable F implementation differs')
 spec=importlib.util.spec_from_file_location('ema29_dependencies',REF/'experiments/cvs_fullsource_concat30_25/runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.bootstrap()
 pkg=types.ModuleType('experiments.cvs_fullsource_concat30_25');pkg.__path__=[str(REF/'experiments/cvs_fullsource_concat30_25')];sys.modules[pkg.__name__]=pkg
 own=types.ModuleType('experiments.cvs_fullsource_cycleorder30');own.__path__=[str(HERE/'experiments/cvs_fullsource_cycleorder30')];sys.modules[own.__name__]=own
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_cycleorder30.probe import main
 main()
