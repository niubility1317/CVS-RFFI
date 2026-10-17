"""Pinned original full-source F runtime; no source-fold bootstrap."""
import importlib.util,json,os,sys,types
from pathlib import Path

def bootstrap():
 here=Path(__file__).resolve().parents[2];runtime=json.loads((here/'runtime.json').read_text(encoding='utf8'));af=Path(runtime['AF']);request=json.loads((af/'experiments/cvs_rdrop_sinc_units/configs/matrix.json').read_text(encoding='utf8'));original,fold=(Path(request[k]) for k in ('dependency_release','fold_release'))
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_fullsource_concat20',here),('cvs_rdrop_sinc_units',af),('cvs_sigtime_rdrop_legacy',af),('cvs_rdrop_fold20',af),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return here

if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_concat20 import design as d
 if len(sys.argv)==1:
  from experiments.cvs_fullsource_concat20.dispatch import run
  run()
 elif len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_fullsource_concat20.checks_model import run
  d.write(d.ROOT/'synthetic_checks.json',run())
 elif len(sys.argv)==3 and sys.argv[1] in ('--worker','--score'):
  rid=sys.argv[2];c=d.config(next(r for r in d.rows() if r['row_id']==rid))
  if sys.argv[1]=='--worker':
   from experiments.cvs_fullsource_concat20.source import train
   from experiments.cvs_fullsource_concat20.evaluate import predict
   import gc,torch
   train(c);gc.collect();torch.cuda.empty_cache();predict(rid)
   os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--score',rid])
  else:
   from experiments.cvs_fullsource_concat20.evaluate import score
   score(rid)
 else:raise ValueError('No unregistered seed/budget/role/stage overrides')
