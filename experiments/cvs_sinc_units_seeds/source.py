import torch
from . import design as d,model as api
from experiments.cvs_sinc_units50.observer import Observer
from .predict import source_proof
from .evidence import recount
from experiments.cvs_feature_sequence50.source import train as shared
def train(c):
 d.require_source();torch.cuda.init();obs=Observer(c,design=d);shared(c,design=d,model_api=api,observer=obs);obs.finish();source_proof(c)
 from .feature_check import run
 v=run(c['model_seed'],only_row=c['row_id'])
 if v['status']!='VERIFIED' or v['models']!=1:raise ValueError('Actual SIG feature witness incomplete')
 d.write(d.BASE/c['row_id']/'source/source_feature_witness.json',v)
 row=next(r for r in d.training_rows() if r['row_id']==c['row_id']);d.write(d.BASE/c['row_id']/'source/sinc_actual_final_witness.json',recount(row))
