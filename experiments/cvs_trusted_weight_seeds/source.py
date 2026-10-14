import torch
from . import design as d,model as api
from .predict import source_proof
from experiments.cvs_trusted_weight50 import observer as observations
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
 torch.cuda.init();old=observations.d;observations.d=d;obs=observations.Observer()
 try:shared_train(c,design=d,model_api=api,observer=obs);obs.finish()
 finally:observations.d=old
 from .feature_check import run
 source_proof(c);value=run(c['model_seed'],only_row=c['row_id'])
 if value['status']!='VERIFIED' or value['models']!=1:raise ValueError('Actual same-row source feature witness missing')
 d.write(d.BASE/c['row_id']/'source/source_feature_witness.json',value)
