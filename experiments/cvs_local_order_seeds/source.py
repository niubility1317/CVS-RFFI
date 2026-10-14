import torch
from . import design as d,model as api
from .predict import source_proof
from experiments.cvs_local_order50 import observer as observations
from experiments.cvs_feature_sequence50.source import train as shared
def train(c):
 torch.cuda.init();previous=observations.d;observations.d=d;obs=observations.Observer()
 try:shared(c,design=d,model_api=api,observer=obs);obs.finish()
 finally:observations.d=previous
 from .feature_check import run
 source_proof(c);v=run(c['model_seed'],only_row=c['row_id'])
 if v['status']!='VERIFIED' or v['models']!=1:raise ValueError('Actual same-row feature witness missing')
 d.write(d.BASE/c['row_id']/'source/source_feature_witness.json',v)
