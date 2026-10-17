import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap():
 ref=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01')
 if (ref/'release_commit.txt').read_text().strip()!='ec4c5d7f46aedb5aaece77fdd72dbe6ed4c157f2':raise ValueError('Pinned reference runtime differs')
 s=importlib.util.spec_from_file_location('noise43_runtime',ref/'experiments/cvs_crossview_supcon41/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap();p=types.ModuleType('experiments.cvs_noise_shape_factorial43');p.__path__=[str(ROOT/'experiments/cvs_noise_shape_factorial43')];sys.modules[p.__name__]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_noise_shape_factorial43.dispatch import main
 main()
