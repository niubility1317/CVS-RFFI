"""Bootstrap only existing model/physics code;never its old test or dispatch path."""
import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def package(name,p):
 m=types.ModuleType(name);m.__path__=[str(p)];sys.modules[name]=m

def bootstrap():
 project=Path('/home/szu2070436088/2510044040/CV-SincNet');back=project/'releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01';physics=project/'releases/cvs_FcrossRX_fullsource_residual_diag_20261017_r01'
 if (back/'release_commit.txt').read_text().strip()!='91bbfcdc026b48abfeaa576e14bc4b7f44d3d64a' or (physics/'release_commit.txt').read_text().strip()!='5dc8882cc1609f3b857749368a75f856b57bcd9d':raise ValueError('Existing immutable model/physics code identities differ')
 s=importlib.util.spec_from_file_location('grad40_backend',back/'experiments/cvs_Uurban39/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap();package('experiments.cvs_Uurban39',back/'experiments/cvs_Uurban39');package('experiments.cvs_fullsource_residual_diag24',physics/'experiments/cvs_fullsource_residual_diag24');package('experiments.cvs_noise_gradient40',ROOT/'experiments/cvs_noise_gradient40')
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_noise_gradient40.probe import main
 main()
