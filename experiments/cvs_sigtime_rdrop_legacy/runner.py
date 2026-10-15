import importlib.util,json,sys,types
from pathlib import Path
def main():
 here=Path(__file__).resolve().parents[2];path=Path(sys.argv[1]).resolve();request=json.loads(path.read_text(encoding='utf-8'));original,fold=(Path(request[k]) for k in ('dependency_release','fold_release'))
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_sigtime_rdrop_legacy',here),('cvs_rdrop_fold20',here),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 from experiments.cvs_sigtime_rdrop_legacy import design as d
 if request!=d.request():raise ValueError('Fixed original SIG_TIME E20/E50 request differs')
 if len(sys.argv)==4 and sys.argv[2]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[3]),None)
  if row is None:raise ValueError('Unregistered original SIG_TIME worker')
  from experiments.cvs_sigtime_rdrop_legacy.source import train
  from experiments.cvs_sigtime_rdrop_legacy.evaluate import predict
  import gc,torch
  train(d.config(row));gc.collect();torch.cuda.empty_cache();predict(row['row_id'])
 elif len(sys.argv)==4 and sys.argv[2]=='--score':
  from experiments.cvs_sigtime_rdrop_legacy.evaluate import score
  score(sys.argv[3])
 elif len(sys.argv)==3 and sys.argv[2]=='--finish':
  from experiments.cvs_sigtime_rdrop_legacy.evaluate import finish
  finish()
 elif len(sys.argv)==2:
  from experiments.cvs_sigtime_rdrop_legacy.dispatch import run
  run(path)
 else:raise ValueError('Unsupported original SIG_TIME entrypoint')
if __name__=='__main__':main()
