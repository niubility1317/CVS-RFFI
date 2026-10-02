"""Frozen source-only internal interventions; no data access or fitting.

The core sees the same aligned input in every mode. Reference descriptors keep
the learned conditioner bias; zero_delta separately removes input and bias.
These are neural internal ablations, not identified TX/RX interventions.
"""
from collections import OrderedDict
import torch
from experiments.cvs_coordinate_identity.usage import assert_state_unchanged

MODES=('trained','frequency_reference','quality_reference','coordinate_reference','zero_delta')


@torch.no_grad()
def counterfactual_logits(model,x):
    if model.training:raise ValueError('Frozen eval model required')
    aligned,_,_,descriptor=model.coordinates(x)
    base=model.core.features(aligned);result=OrderedDict()
    for mode in MODES:
        d=descriptor.clone()
        if mode in ('frequency_reference','coordinate_reference'):d[:,:2]=0
        if mode in ('quality_reference','coordinate_reference'):d[:,2]=0
        features=base if mode=='zero_delta' else model.inject(base,d)[0]
        logits=model.classify_features(features)
        if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid counterfactual scores')
        result[mode]=logits
    return result
