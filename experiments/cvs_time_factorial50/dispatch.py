import argparse,os
from . import design as d
from .provenance import source_proof
from experiments.cvs_feature_sequence50 import dispatch as shared
PACKAGE='experiments.cvs_time_factorial50'
def eligible(row):return shared.eligible(row,design=d,proof=source_proof)
def worker(row):return shared.worker(row,design=d,package=PACKAGE)
def dispatch():return shared.dispatch(design=d,package=PACKAGE)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';shared.replace('--score',a.predict,PACKAGE)
 elif a.score:
  from .score import score
  score(a.score)
 else:dispatch()
