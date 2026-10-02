"""Frozen internal-coordinate counterfactuals; no optimizer or data access.

These internal interventions diagnose the trained model's use of coordinates.
They are not TX/RX hardware interventions and must not change source selection.
"""
from collections import OrderedDict
import torch

MODES=('trained','frequency_reference','quality_reference','coordinate_reference','unit_gain')


@torch.no_grad()
def counterfactual_logits(model,x):
    if model.training:raise ValueError('Frozen eval model required')
    aligned,_,_,descriptor=model.coordinates(x)
    base=model.core.features(aligned)
    result=OrderedDict()
    for mode in MODES:
        d=descriptor.clone()
        if mode in ('frequency_reference','coordinate_reference'):d[:,:2]=0
        if mode in ('quality_reference','coordinate_reference'):d[:,2]=0
        # Zero descriptor retains the learned conditioner bias. Unit gain
        # separately removes both its input response and fixed reweighting.
        gain=torch.ones_like(base) if mode=='unit_gain' else 1+.25*torch.tanh(model.conditioner(d))
        logits=model.classify_features(base*gain)
        if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid counterfactual scores')
        result[mode]=logits
    return result


def assert_state_unchanged(model,before):
    actual=model.state_dict()
    if set(actual)!=set(before) or any(not torch.equal(actual[k],v) for k,v in before.items()):
        raise ValueError('Counterfactual changed frozen model state')
