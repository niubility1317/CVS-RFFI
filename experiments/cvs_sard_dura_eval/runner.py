import importlib.util,sys,types
from pathlib import Path

def bootstrap():
 root=Path(__file__).resolve().parents[2]
 project=Path('/home/szu2070436088/2510044040/CV-SincNet')
 backend=project/'releases/cvs_T1_AsKL0_E30plus20_m3_20261021_r01/experiments/cvs_t1_e50_confirm/runner.py'
 spec=importlib.util.spec_from_file_location('sard_dura_backend',backend);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 import experiments
 # Reuse the original committed restoration inference implementation, not a new copy.
 ref=project/'releases/pamf_stage2_inference_20261021_r01'
 experiments.__path__.append(str(ref/'experiments'))
 for name in ('tools','comparison_suite'):
  if name in sys.modules and hasattr(sys.modules[name],'__path__'):sys.modules[name].__path__.append(str(ref/name))
 for path in (ref,ref/'code'):
  if str(path) not in sys.path:sys.path.append(str(path))
 name='experiments.cvs_sard_dura_eval';package=types.ModuleType(name);package.__path__=[str(root/'experiments/cvs_sard_dura_eval')];sys.modules[name]=package

if __name__=='__main__':
 bootstrap()
 from experiments.cvs_sard_dura_eval.execution import main
 main()
