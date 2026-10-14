import torch
from . import design as d, model as api
from .observer import Observer
from .predict import source_proof
from experiments.cvs_feature_sequence50.source import train as shared


def train(c, *, design=None, model_api=None, proof_fn=None, feature_fn=None):
    d = design or globals()['d']
    api = model_api or globals()['api']
    torch.cuda.init()
    obs = Observer(design=d)
    shared(c, design=d, model_api=api, observer=obs)
    obs.finish(); (proof_fn or source_proof)(c)
    from .feature_check import run
    value = (feature_fn or run)(c['model_seed'], only_row=c['row_id'])
    if value['status'] != 'VERIFIED' or value['models'] != 1:
        raise ValueError('Actual default-feature witness incomplete')
    d.write(d.BASE / c['row_id'] / 'source/source_feature_witness.json', value)
