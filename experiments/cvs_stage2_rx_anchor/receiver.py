"""Remove only the directional target from an immutable S3 receiver bank."""
from copy import deepcopy
import torch
from . import design as d
from experiments.cvs_stage2_geometry.receiver import load_receiver


def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def anchor_payload(original):
    if original.get('method')!=d.stable.METHOD or original.get('target_access') is not False:
        raise ValueError('Require the existing source-only stable RX bank')
    result=deepcopy(original)
    result.update(method=d.METHOD,alpha_max=0.,anchor_only=True,original_method=original['method'],
                  refitted=False,fit_seconds=0.,original_fit_seconds=original['fit_seconds'])
    for block in result['blocks']:block['alpha']=0.
    for group in result['groups']:group['target_feature_mean']=group['parent_feature_mean'].clone()
    # Exact structural comparison verifies that reliability, potentials, order,
    # folds, sample membership and control variates are untouched.
    restored=deepcopy(result)
    for key in ('anchor_only','original_method','refitted','original_fit_seconds'):restored.pop(key)
    for key in ('method','alpha_max','fit_seconds'):restored[key]=deepcopy(original[key])
    for new,old in zip(restored['blocks'],original['blocks']):new['alpha']=old['alpha']
    for new,old in zip(restored['groups'],original['groups']):new['target_feature_mean']=old['target_feature_mean'].clone()
    if not same(restored,original):raise ValueError('Anchor control changed more than its directional target')
    return result
