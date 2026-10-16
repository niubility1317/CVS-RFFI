"""Use original immutable model/augmentation dependency,install new probe only."""
import importlib.util,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('shift_probe_bootstrap',here/'experiments/cvs_commonphase_source/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_shift_counterfactual16';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_shift_counterfactual16')];sys.modules[name]=p
 return here
if __name__=='__main__':
 if len(sys.argv)!=1:raise ValueError('No role/seed/operator/budget overrides')
 bootstrap()
 from experiments.cvs_shift_counterfactual16.run import run
 run()
