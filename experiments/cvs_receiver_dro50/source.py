import torch
from . import design as d, model as api
from .observer import Observer
from .predict import source_proof
from experiments.cvs_feature_sequence50.source import train as shared


def train(c):
    torch.cuda.init()
    obs = Observer()
    shared(c, design=d, model_api=api, observer=obs)
    obs.finish(); source_proof(c)
    from .feature_check import run
    value = run(c['model_seed'], only_row=c['row_id'])
    if value['status'] != 'VERIFIED' or value['models'] != 1:
        raise ValueError('Actual default-feature witness incomplete')
    d.write(d.BASE / c['row_id'] / 'source/source_feature_witness.json', value)
