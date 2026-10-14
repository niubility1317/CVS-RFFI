import argparse,os
from pathlib import Path
from . import design as d
from .provenance import source_proof
from experiments.cvs_feature_sequence50 import dispatch as shared
PACKAGE='experiments.cvs_u_schedule50'

def eligible(row,design=None,proof=None):
 if row['model_seed']==d.SEEDS[0]:return True
 first=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]]
 if any(not (Path(c['output_root'])/'completion.json').is_file() for c in first):return False
 evidence={}
 for c in first:
  p=source_proof(c);v=d.read(p/'completion.json')
  if not 0<=v['final_training_L_CE']<=.5:raise ValueError('Source final CE falsification failed:'+c['arm'])
  evidence[c['arm']]=dict(metrics=v['final_source_metrics'],CE=v['final_training_L_CE'],calendar_execution=d.read(p/'calendar_execution.json'))
 b=evidence['TAIL']['metrics']
 for arm in ('MID','SPREAD'):
  v=evidence[arm]['metrics']
  if v['source_val_accuracy']<b['source_val_accuracy']-.01 or v['source_val_worst_rx']<b['source_val_worst_rx']-.02:raise ValueError('Source health falsification failed:'+arm)
 d.write(d.BASE/'source_seed_expansion_decision.json',dict(source_only=True,target_metrics_read=False,passed=True,evidence=evidence,calendars_fixed=True));return True
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:shared.worker(a.worker,design=d,package=PACKAGE)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';shared.replace('--score',a.predict,PACKAGE)
 elif a.score:
  from .score import score
  score(a.score)
 else:shared.eligible=eligible;shared.dispatch(design=d,package=PACKAGE)
