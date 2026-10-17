"""Reuse the verified F factory only; no historical initializer for future training."""
from pathlib import Path
import importlib.util,sys,types
ROOT=Path(__file__).resolve().parents[2]
R=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanretain_E30_20261017_r01')
def bootstrap():
 if (R/'release_commit.txt').read_text().strip()!='c577b3daded412337fb4181be05078ce051a64f1':raise ValueError('Actual F factory reference differs')
 spec=importlib.util.spec_from_file_location('U38_dependencies',R/'experiments/cvs_cleanretain37/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_cleanretain37';p=types.ModuleType(name);p.__path__=[str(R/'experiments/cvs_cleanretain37')];sys.modules[name]=p
 name='experiments.cvs_U_teacher38';p=types.ModuleType(name);p.__path__=[str(ROOT/'experiments/cvs_U_teacher38')];sys.modules[name]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_U_teacher38.qualify import run
 run()
