"""Pinned immutable F seed01 backend;one new scratch E20 row only."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap(backend=None):
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01' if backend is None else backend)
 if (backend/'release_commit.txt').read_text().strip()!='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b':raise ValueError('Wrong immutable sourceF seed01 backend')
 spec=importlib.util.spec_from_file_location('timegrid_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 package=types.ModuleType('experiments.cvs_f_timegrid20');package.__path__=[str(HERE/'experiments/cvs_f_timegrid20')];sys.modules['experiments.cvs_f_timegrid20']=package
 from experiments.cvs_f_timegrid20 import design as d
 d.template();return d
if __name__=='__main__':
 args=sys.argv[1:]
 if args and args!=['--worker','TIMEGRID-RX1-s2026092701']:raise ValueError('No data/budget/timegrid/seed/recipe overrides')
 d=bootstrap()
 if args:
  from experiments.cvs_f_timegrid20.source import train
  train(d.config(d.rows()[0]))
 else:
  from experiments.cvs_f_timegrid20.dispatch import run
  run()
