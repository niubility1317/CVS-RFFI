"""Synthetic default-head math, source-domain sampling and input gradients."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.logit_energy_core')


def test_complete_reader_samepass_initial_logits_raw_features_and_frozen_Sinc_buffers():
    from test_pamf_phase3_native_identity_train import Reader
    m=api();torch.manual_seed(3);reader=Reader().eval().requires_grad_(False);x=torch.randn(4,2,256,requires_grad=True)
    with torch.no_grad():raw=reader.features(x);expected=reader.classify_features(raw)
    model=m.NativeEnergy(reader);before=deepcopy(model.state_dict());calls=[]
    handle=model.reader.register_forward_pre_hook(lambda *a:calls.append('reader_forward'))
    counter=[];old=model.reader.features
    model.reader.features=lambda a:(counter.append(len(a)) or old(a))
    model.train();features,logits=model(x);assert not model.training and all(not v.training for v in model.modules())
    assert counter==[4] and calls==[]
    assert torch.equal(raw,features) and torch.equal(expected,logits)
    assert sum(p.numel() for p in model.parameters())==232993 and sum(p.numel() for p in model.parameters() if p.requires_grad)==232945
    assert all(k.startswith('reader.') for k in model.state_dict())
    F.cross_entropy(logits,torch.arange(4)).backward()
    assert model.reader.core.id_backbone.cls_head.weight.grad.norm()>0 and x.grad.norm()>0
    assert all(p.grad is None for p in model.reader.core.id_backbone.sinc.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in reader.parameters());handle.remove()


def test_CE_KD_uniform_and_RXday_pair_ranking_exact_teacher_detached():
    m=api();torch.manual_seed(10)
    L=torch.randn(4,6,requires_grad=True);OE=torch.randn(3,6,requires_grad=True);teacher=torch.randn(4,6,requires_grad=True)
    y=torch.tensor([0,1,2,3]);ld=torch.tensor([0,1,0,2]);od=torch.tensor([0,1,2])
    common=F.cross_entropy(L,y)+4*F.kl_div(F.log_softmax(L/2,1),F.softmax(teacher.detach()/2,1),reduction='batchmean')
    uniform=-F.log_softmax(OE,1).mean()-math.log(6)
    rank=F.softplus(torch.logsumexp(OE,1)[None]-torch.logsumexp(L,1)[:,None])[ld[:,None]==od[None]].mean()
    u=m.compute(L,y,teacher,ld,OE,od,'UNIFORM');r=m.compute(L,y,teacher,ld,OE,od,'RXPAIR')
    torch.testing.assert_close(u['total'],common+uniform);torch.testing.assert_close(r['total'],common+rank)
    assert u['rank']==0 and r['uniform']==0 and r['pair_count']==4
    r['total'].backward(retain_graph=True);assert L.grad.norm()>0 and OE.grad.norm()>0 and teacher.grad is None
    L.grad=None;OE.grad=None;u['total'].backward();assert L.grad.norm()>0 and OE.grad.norm()>0
    zeros=torch.zeros(3,6);z=m.compute(L.detach(),y,teacher.detach(),ld,zeros,od,'UNIFORM');assert abs(float(z['uniform']))<2e-7


def test_sampler_full_six_by_fifteen_L_and_public_OE_matching_two_streams():
    m=api();rows=np.asarray([(c,r,d) for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64)
    L=dict(y=rows[:,0],rx_i=rows[:,1],day_i=rows[:,2],domain=np.tile(np.arange(15),6).astype(np.int64))
    OE=dict(rx_i=np.repeat(np.asarray([1,3,4,6,8]),3).astype(np.int64),day_i=np.tile(np.asarray([1,2,3]),5).astype(np.int64),domain=np.arange(15,dtype=np.int64))
    a,b=m.SourceSampler(L,OE),m.SourceSampler(L,OE)
    for step in range(1,13):
        x,z=a.sample(step),b.sample(step)
        assert all(np.array_equal(x[k],z[k]) for k in ('L','OE'))
        assert len(x['L'])==192 and len(x['OE'])==64
        assert np.array_equal(np.bincount(L['y'][x['L']],minlength=6),np.full(6,32))
        assert all(np.sum((L['y'][x['L']]==c)&(L['domain'][x['L']]==d))>=2 for c in range(6) for d in range(15))
        assert np.all(np.bincount(OE['domain'][x['OE']],minlength=15)>=4)
    with pytest.raises(ValueError):m.SourceSampler(L,dict(OE,y=np.zeros(15,np.int64)))


def test_invalid_logits_labels_source_domain_and_input_fail_clearly(subtests):
    m=api();L=torch.randn(3,6);OE=torch.randn(2,6);teacher=L.clone();y=torch.arange(3);ld=torch.tensor([0,1,0]);od=torch.tensor([0,1])
    for bad in ('nan','OElabel','domain','no_pair','dtype','teacher','arm'):
        with subtests.test(bad=bad),pytest.raises((ValueError,TypeError)):
            if bad=='nan':v=L.clone();v[0,0]=float('nan');m.compute(v,y,teacher,ld,OE,od,'UNIFORM')
            elif bad=='OElabel':m.compute(L,y,teacher,ld,OE,od,'RXPAIR',OE_y=torch.zeros(2))
            elif bad=='domain':m.compute(L,y,teacher,ld+15,OE,od,'RXPAIR')
            elif bad=='no_pair':m.compute(L,y,teacher,ld,OE,od+4,'RXPAIR')
            elif bad=='dtype':m.compute(L.double(),y,teacher,ld,OE,od,'UNIFORM')
            elif bad=='teacher':m.compute(L,y,teacher[:2],ld,OE,od,'UNIFORM')
            else:m.compute(L,y,teacher,ld,OE,od,'TARGET')
