"""Reuse the verified frozen model loader,then install only this source probe."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2]
 spec=importlib.util.spec_from_file_location('spectral_source_bootstrap',here/'experiments/cvs_commonphase_source/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_spectral_iq16';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_spectral_iq16')];sys.modules[name]=p
 return here
if __name__=='__main__':
 if len(sys.argv)!=1:raise ValueError('No data/transform/model/budget overrides')
 bootstrap()
 from experiments.cvs_spectral_iq16.run import run
 run()
