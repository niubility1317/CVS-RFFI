import argparse
import os
from . import design as d
from .provenance import source_proof
from experiments.cvs_feature_sequence50 import dispatch as shared

PACKAGE='experiments.cvs_crossrx_credit50'
_PARENT_ELIGIBLE=shared.eligible

def eligible(row,design=d,proof=None):
 if row['model_seed']==d.SEEDS[0]:return True
 if not _PARENT_ELIGIBLE(row,design=d,proof=source_proof):return False
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]))
  e=d.read(source_proof(c)/'source_training_mechanism.json')
  if e['eligible_calls']/624<d.SOURCE_GATE['eligible_fraction_min']:
   raise ValueError('Source auxiliary cohort coverage failed: '+arm)
  if arm in ('ORDER','CREDIT') and len(e['active_gradient_probe_steps'])<2:
   raise ValueError('Actual auxiliary gradient availability failed: '+arm)
  if arm=='CREDIT' and e['accepted_groups']/(5*max(1,e['eligible_calls']))<.01:
   raise ValueError('Actual early/mid credit availability failed')
 return True

def dispatch():
 original=shared.eligible;shared.eligible=eligible
 try:return shared.dispatch(design=d,package=PACKAGE)
 finally:shared.eligible=original

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:shared.worker(a.worker,design=d,package=PACKAGE)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';shared.replace('--score',a.predict,PACKAGE)
 elif a.score:
  from .score import score
  score(a.score)
 else:dispatch()
