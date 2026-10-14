import torch
from . import design as d,model as api
from .predict import source_proof
from experiments.cvs_feature_sequence50.source import train as shared_train

def train(c):
 # Initialize the CUDA context before native seeding/body construction.
 # This avoids the historical lazy-CUDA RNG initialization confound.
 torch.cuda.init()
 shared_train(c,design=d,model_api=api)
 from .feature_check import run
 source_proof(c);value=run(c['model_seed'],only_row=c['row_id'])
 if value['status']!='VERIFIED' or value['models']!=1:raise ValueError('Actual feature witness incomplete')
 d.write(d.BASE/c['row_id']/'source/source_feature_witness.json',value)
