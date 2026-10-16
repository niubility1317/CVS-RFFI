"""Source-only no-update Sinc/RDrop diagnosis using immutable A-F execution code."""
import importlib.util,json,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
AF=None
def bootstrap():
 global AF
 AF=Path(json.loads((HERE/'runtime.json').read_text())['AF'])
 request=json.loads((AF/'experiments/cvs_rdrop_sinc_units/configs/matrix.json').read_text());original,fold=(Path(request[k]) for k in ('dependency_release','fold_release'))
 here=Path(__file__).resolve().parents[2]
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_sinc_bandgain20',here),('cvs_rdrop_sinc_units',AF),('cvs_sigtime_rdrop_legacy',AF),('cvs_rdrop_fold20',AF),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;m=types.ModuleType(name);m.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=m
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return here
if __name__=='__main__':
 here=bootstrap()
 from experiments.cvs_sinc_bandgain20 import design as d
 if len(sys.argv)==3 and sys.argv[1]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
  if row is None:raise ValueError('Unregistered source-gain worker')
  from experiments.cvs_sinc_bandgain20.source import train
  train(d.config(row))
 elif len(sys.argv)==1:
  from experiments.cvs_sinc_bandgain20.dispatch import run
  run()
 else:raise ValueError('Unsupported source-gain command')
