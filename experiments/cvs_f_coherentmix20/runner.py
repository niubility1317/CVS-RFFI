import importlib.util,json,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def bootstrap(backend=None):
 here=Path(__file__).resolve().parents[2];runtime=here/'experiments/cvs_f_coherentmix20/runtime.json'
 profile=json.loads(runtime.read_text(encoding='utf8'))['profile'] if runtime.exists() else 'coherent_initial_seed02'
 profiles={'coherent_initial_seed02':('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_confirm_sourceRX1_e20_20261016_r01','05984a1229ebac7025a95081dc5fae4108276b28'),'coherent_confirmation_seed01':('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01','7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b')}
 if profile not in profiles:raise ValueError('Unregistered coherent source profile')
 path,commit=profiles[profile];backend=Path(path if backend is None else backend)
 if (backend/'release_commit.txt').read_text().strip()!=commit:raise ValueError('Wrong immutable additive backend for coherent profile')
 spec=importlib.util.spec_from_file_location('coherentmix_backend',backend/'experiments/cvs_additive_as20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_f_coherentmix20';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_f_coherentmix20')];sys.modules[name]=p
 if profile=='coherent_confirmation_seed01':
  from experiments.cvs_f_coherentmix20.confirmation import activate
  activate()
 return here
if __name__=='__main__':
 if not (len(sys.argv)==1 or (len(sys.argv)==3 and sys.argv[1]=='--worker' and sys.argv[2] in tuple(a+'-RX1-s'+str(s) for a in ('SAME_RX','CROSS_RX') for s in (2026092701,2026092702)))):raise ValueError('No data/budget/recipe overrides permitted')
 bootstrap()
 from experiments.cvs_f_coherentmix20 import design as d
 if len(sys.argv)==3:
  row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
  if row is None:raise ValueError('Worker not in active coherent source profile')
  from experiments.cvs_f_coherentmix20.source import train
  train(d.config(row))
 else:
  from experiments.cvs_f_coherentmix20.dispatch import run
  run()
