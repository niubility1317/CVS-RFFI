"""Bootstrap only the immutable P32 deployed model; never invoke its test runner."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
P32=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01')
def bootstrap():
 if (P32/'release_commit.txt').read_text().strip()!='c3700b6636bb0ed32a8c7bdb00f8f94ef1a8f632':raise ValueError('Frozen model code commit differs')
 s=importlib.util.spec_from_file_location('delay34_dependencies',P32/'experiments/cvs_practical_attribution33/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 p=types.ModuleType('experiments.cvs_delay_phase_attribution34');p.__path__=[str(HERE/'experiments/cvs_delay_phase_attribution34')];sys.modules[p.__name__]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_delay_phase_attribution34.run import main
 main()
