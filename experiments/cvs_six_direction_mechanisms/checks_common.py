"""Focused source legality and frozen diagnostic boundary checks."""
import copy
import json
import torch
from . import design as d
from .common import state_q, normalized_state
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
from experiments.cvs_source_pseudotarget_health.frozen import snapshot, assert_unchanged

def reject(fn):
    try:fn()
    except (ValueError,TypeError):return True
    raise AssertionError('Invalid source input accepted')

def checks():
    x=torch.randn(12,2,256);q=state_q(x);fit=torch.arange(12)<8
    normalized,mean,scale=normalized_state(q,fit)
    assert q.shape==(12,12) and not q.requires_grad and torch.isfinite(normalized).all()
    changed=q.clone();changed[~fit]+=1e3
    _,other_mean,other_scale=normalized_state(changed,fit)
    assert torch.equal(mean,other_mean) and torch.equal(scale,other_scale)
    reject(lambda:state_q(x[:,:,:255]));bad=x.clone();bad[0,0,0]=float('nan');reject(lambda:state_q(bad))
    reject(lambda:normalized_state(q,torch.zeros(12,dtype=torch.bool)))
    u=dict(role='U_s',x=x,rx=torch.tensor([1,3,4,6,8,1,3,4,6,8,1,3]),day=torch.tensor([1,2,3]*4),ids=[str(i) for i in range(12)])
    u.update(domain=torch.zeros(12,dtype=torch.long),eq=torch.zeros(12,dtype=torch.long),
             sig=torch.zeros(12,dtype=torch.long),base_index=torch.arange(12),preprocessing='fixture received IQ')
    validate_unlabeled(u,u['ids'])
    for key in ('y','labels','truth','tx'):
        polluted=dict(u);polluted[key]=torch.zeros(12,dtype=torch.long)
        reject(lambda:validate_unlabeled(polluted,u['ids']))
    for row in d.rows():
        c=d.config(row);d.validate(c);bad=copy.deepcopy(c);bad['identity_updates']=1;reject(lambda:d.validate(bad))
        bad=copy.deepcopy(c);bad['target_access']=True;reject(lambda:d.validate(bad))
    model=torch.nn.Sequential(torch.nn.Linear(3,2)).eval().requires_grad_(False);before=snapshot(model)
    assert_unchanged(model,before)
    with torch.no_grad():model[0].weight.add_(1)
    reject(lambda:assert_unchanged(model,before))
    return dict(status='PASS',checks=['L-fit normalization excludes held packets','finite IQ256 descriptors',
        'U truth/label pollution rejected','fixed three-row zero-update/no-target configs',
        'persistent identity mutation rejected'],identity_updates=0,target_access=False)

if __name__=='__main__':print(json.dumps(checks()))
