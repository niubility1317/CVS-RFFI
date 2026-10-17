import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap():
 ref=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01')
 if (ref/'release_commit.txt').read_text().strip()!='19aedc3604fed9c193237dd92366164178af0983':raise ValueError('Pinned source model/runtime commit differs')
 spec=importlib.util.spec_from_file_location('taper46_backend',ref/'experiments/cvs_noise_shape_factorial43/runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.bootstrap()
 pkg=types.ModuleType('experiments.cvs_taper_source46');pkg.__path__=[str(ROOT/'experiments/cvs_taper_source46')];sys.modules[pkg.__name__]=pkg
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_taper_source46.run import main
 main()
