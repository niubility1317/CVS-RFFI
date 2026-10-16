import importlib.util,json,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def bootstrap(backend=None):
 here=Path(__file__).resolve().parents[2]
 if backend is None:backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01')
 else:backend=Path(backend)
 if (backend/'release_commit.txt').read_text().strip()!='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b':raise ValueError('Wrong immutable additive backend')
 spec=importlib.util.spec_from_file_location('RX6_additive_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_f_additive_rx6';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_f_additive_rx6')];sys.modules[name]=p
 return here
if __name__=='__main__':
 if not (len(sys.argv)==1 or (len(sys.argv)==3 and sys.argv[1]=='--worker' and sys.argv[2] in ('D00-RX6-s2026092701','D_raw-RX6-s2026092701','D_As-RX6-s2026092701'))):raise ValueError('No data/budget/recipe overrides permitted')
 bootstrap()
 from experiments.cvs_f_additive_rx6 import design as d
 if len(sys.argv)==3:
  from experiments.cvs_f_additive_rx6.source import train
  train(d.config(next(r for r in d.rows() if r['row_id']==sys.argv[2])))
 else:
  from experiments.cvs_f_additive_rx6.dispatch import run
  run()
