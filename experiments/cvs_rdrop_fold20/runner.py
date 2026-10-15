import importlib.util,json,sys,types
from pathlib import Path
def main():
 here=Path(__file__).resolve().parents[2];path=Path(sys.argv[1]);request=json.loads(path.read_text(encoding='utf-8'))
 original,fold=Path(request['dependency_release']),Path(request['fold_release'])
 sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
 for suffix,root in (('cvs_rdrop_fold20',here),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
  name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=package
 name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 from experiments.cvs_rdrop_fold20 import design as d
 if request!=d.request():raise ValueError('Fixed matched source training matrix differs')
 if len(sys.argv)==4 and sys.argv[2]=='--row':
  from experiments.cvs_rdrop_fold20.source import train
  c=next((d.config(a) for a in d.ARMS if d.config(a)['row_id']==sys.argv[3]),None)
  if c is None:raise ValueError('Unknown source worker row')
  train(c)
 elif len(sys.argv)==2:
  from experiments.cvs_rdrop_fold20.dispatch import run
  run(path)
 else:raise ValueError('Unsupported source entrypoint')
if __name__=='__main__':main()
