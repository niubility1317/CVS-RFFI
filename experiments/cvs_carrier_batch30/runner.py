"""One source-trained row automatically predicts all sevenviews,then CPU truth-last."""
import importlib.util,json,os,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());reference=Path(rt['parent_runtime_release'])
 if (reference/'release_commit.txt').read_text().strip()!=rt['parent_runtime_commit']:raise ValueError('Existing source trainer dependency differs')
 spec=importlib.util.spec_from_file_location('pinned_carrier_source_runtime',reference/'experiments/cvs_view_moment30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap();name='experiments.cvs_carrier_batch30';p=types.ModuleType(name);p.__path__=[str(here/name.replace('.','/'))];sys.modules[name]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_carrier_batch30 import design as d
 if len(sys.argv)==1:
  from experiments.cvs_carrier_batch30.dispatch import run
  run()
 elif len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_carrier_batch30.checks import run
  from experiments.cvs_carrier_batch30.protocol_checks import run as protocol
  d.write(d.ROOT/'checks.json',dict(carrier=run(),protocol=protocol()))
 elif len(sys.argv)==3 and sys.argv[1] in ('--worker','--score'):
  rid=sys.argv[2];c=d.config(next(r for r in d.rows() if r['row_id']==rid))
  if sys.argv[1]=='--worker':
   from experiments.cvs_carrier_batch30.source import train
   from experiments.cvs_carrier_batch30.qualify import run as qualify
   from experiments.cvs_carrier_batch30.evaluate import predict
   import gc,torch
   train(c);gc.collect();torch.cuda.empty_cache();qualify(c);gc.collect();torch.cuda.empty_cache();predict(rid);os.environ['CUDA_VISIBLE_DEVICES']='';os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--score',rid])
  else:
   from experiments.cvs_carrier_batch30.evaluate import score
   score(rid)
 else:raise ValueError('Only fixed complete E30 row/prediction/independent CPU scoring')
