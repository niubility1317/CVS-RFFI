import torch
from . import design as d,model as api
from .predict import source_proof
from .observer import Observer
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
 torch.cuda.init();obs=Observer();shared_train(c,design=d,model_api=api,observer=obs);obs.finish()
 from .feature_check import run
 source_proof(c);value=run(c['model_seed'],only_row=c['row_id'])
 if value['status']!='VERIFIED' or value['models']!=1:raise ValueError('Actual same-row feature witness missing')
 d.write(d.BASE/c['row_id']/'source/source_feature_witness.json',value)
