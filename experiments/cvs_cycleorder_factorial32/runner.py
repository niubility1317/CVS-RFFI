"""Use P30 immutable operator and original F; no target path."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
P30=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cycleorder_sourceV_E30_20261017_r01')
def bootstrap():
 if (P30/'release_commit.txt').read_text().strip()!='7e8dc09d47571e3e99a241d1ba39672d5bba70f6':raise ValueError('The fixed IQ operator code changed')
 s=importlib.util.spec_from_file_location('cycle32_dependencies',P30/'experiments/cvs_fullsource_cycleorder30/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 p=types.ModuleType('experiments.cvs_cycleorder_factorial32');p.__path__=[str(HERE/'experiments/cvs_cycleorder_factorial32')];sys.modules[p.__name__]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_cycleorder_factorial32.dispatch import main
 main()
