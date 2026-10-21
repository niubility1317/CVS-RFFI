import importlib.util,sys,types
from pathlib import Path

def bootstrap():
 root=Path(__file__).resolve().parents[2]
 project=Path('/home/szu2070436088/2510044040/CV-SincNet')
 # Seed-specific E30 checkers and the unchanged E50 scientific implementation.
 for name,ref in [('parent86',project/'releases/cvs_FcrossRX_sameAsKD_confirm_E30_20261019_r02/experiments/cvs_same_as_kd_confirm86/runner.py'),
                  ('e50',project/'releases/cvs_T1_AsKL0_E50_20261020_all7_r03/experiments/cvs_t1_e50/runner.py')]:
  s=importlib.util.spec_from_file_location('confirm_'+name,ref);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 p=types.ModuleType('experiments.cvs_t1_e50_confirm');p.__path__=[str(root/'experiments/cvs_t1_e50_confirm')];sys.modules[p.__name__]=p

if __name__=='__main__':
 bootstrap()
 from experiments.cvs_t1_e50_confirm.dispatch import main
 main()
