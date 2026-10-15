import torch
from types import SimpleNamespace
from . import design as d,model as api
from .observer import Observer
from experiments.cvs_feature_sequence50 import source as original

def train(c):
 view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT')},UPDATES=11100)
 observer=Observer(c);original.train(c,design=view,model_api=api,observer=observer);observer.finish()
 from .feature_check import run
 run(c)
 return d.read(d.BASE/c['row_id']/'source/completion.json')
