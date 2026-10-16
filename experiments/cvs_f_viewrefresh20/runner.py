"""Pinned immutable F seed01 backend;three matched new scratch E20 rows."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap(backend=None):
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01' if backend is None else backend)
 if (backend/'release_commit.txt').read_text().strip()!='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b':raise ValueError('Wrong immutable sourceF seed01 backend')
 spec=importlib.util.spec_from_file_location('coupling_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 package=types.ModuleType('experiments.cvs_f_viewrefresh20');package.__path__=[str(HERE/'experiments/cvs_f_viewrefresh20')];sys.modules['experiments.cvs_f_viewrefresh20']=package
 from experiments.cvs_f_viewrefresh20 import design as d
 d.template();return d
if __name__=='__main__':
 args=sys.argv[1:]
 if args and (len(args)!=2 or args[0]!='--worker' or args[1] not in ('RAW-RX1-s2026092701','REPLAY-RX1-s2026092701','REFRESH-RX1-s2026092701')):raise ValueError('No data/budget/view/seed/recipe overrides')
 d=bootstrap()
 if args:
  from experiments.cvs_f_viewrefresh20.source import train
  train(d.config(next(r for r in d.rows() if r['row_id']==args[1])))
 else:
  from experiments.cvs_f_viewrefresh20.dispatch import run
  run()
