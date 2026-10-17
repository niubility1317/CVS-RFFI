import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_clock_factorial_e20plus10_20261017_r03')
Q=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_clean_teacher_source_20261018_r01')
def package(name,path):
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m

def bootstrap():
 if (OLD/'release_commit.txt').read_text().strip()!='8a83af076a2f632a4ee514da689b2e8d166d0592' or (Q/'release_commit.txt').read_text().strip()!='c2b56188d1728c6cb189fe65c242b5a627ee3424':raise ValueError('Fixed full-reference or qualification code differs')
 s=importlib.util.spec_from_file_location('U39_old',OLD/'experiments/cvs_clock_factorial30/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 package('experiments.cvs_U_teacher38',Q/'experiments/cvs_U_teacher38');package('experiments.cvs_Uurban39',ROOT/'experiments/cvs_Uurban39')
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_Uurban39.dispatch import main
 main()
