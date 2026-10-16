import importlib.util,json,sys,types
from pathlib import Path
def main():
 here=Path(__file__).resolve().parents[2];path=Path(sys.argv[1]).resolve();request=json.loads(path.read_text(encoding='utf-8'))
 original,fold,model,gain,center=(Path(request[k]) for k in ('dependency_release','fold_release','model_release','gain_release','center_release'))
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_sigtime_rdrop_features_main',here),('cvs_sigtime_rdrop_gain20',gain),('cvs_sigtime_rdrop_center20',center),('cvs_sigtime_rdrop_legacy',model),('cvs_rdrop_fold20',model),('cvs_sigtime_rdrop_recovery',gain),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 from experiments.cvs_sigtime_rdrop_features_main import design as d
 if request!=d.request():raise ValueError('Fixed12 direct-test feature main request differs')
 if len(sys.argv)==4 and sys.argv[2]=='--worker':
  row=next((r for r in d.rows() if r['row_id']==sys.argv[3]),None)
  if row is None:raise ValueError('Unregistered main row')
  from experiments.cvs_sigtime_rdrop_features_main.source import train
  from experiments.cvs_sigtime_rdrop_features_main.evaluate import predict
  import gc,torch
  train(d.config(row));gc.collect();torch.cuda.empty_cache();predict(row['row_id'])
 elif len(sys.argv)==4 and sys.argv[2]=='--score':
  from experiments.cvs_sigtime_rdrop_features_main.evaluate import score
  score(sys.argv[3])
 elif len(sys.argv)==3 and sys.argv[2]=='--finish':
  from experiments.cvs_sigtime_rdrop_features_main.evaluate import finish
  finish()
 elif len(sys.argv)==2:
  from experiments.cvs_sigtime_rdrop_features_main.dispatch import run
  run(path)
 else:raise ValueError('Only registered train-predict/CPUscore/finish modes allowed')
if __name__=='__main__':main()
