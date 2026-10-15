import importlib.util,json,sys,types
from pathlib import Path
def main():
 here=Path(__file__).resolve().parents[2];path=Path(sys.argv[1]).resolve();request=json.loads(path.read_text(encoding='utf-8'));original,fold=(Path(request[k]) for k in ('dependency_release','fold_release'))
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_dropout_transfer20',here),('cvs_rdrop_fold20',here),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 from experiments.cvs_dropout_transfer20 import design as d
 if request!=d.request():raise ValueError('Fixed source exchange request differs')
 if len(sys.argv)==4 and sys.argv[2]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[3]),None)
  if row is None:raise ValueError('Unregistered source worker')
  from experiments.cvs_dropout_transfer20.source import train
  train(d.config(row))
 elif len(sys.argv)==2:
  from experiments.cvs_dropout_transfer20.dispatch import run
  run(path)
 else:raise ValueError('Unsupported source-only entrypoint')
if __name__=='__main__':main()
