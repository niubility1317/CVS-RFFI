"""Native-coordinate supervised contrast: exact masks, OE role and gradients."""
import importlib
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_contrast_core')


def inputs():
    generator=torch.Generator().manual_seed(19)
    L=torch.randn(6,160,generator=generator);OE=torch.randn(3,160,generator=generator)
    y=torch.tensor([0,0,0,1,1,1]);domains=torch.tensor([0,1,5,0,1,5])
    return L,OE,y,domains


def oracle(L,OE,y,d):
    u=F.normalize(L,dim=1,eps=1e-4).double();v=F.normalize(OE,dim=1,eps=1e-4).double()
    known=[];outside=[];positives=0;negatives=0
    for i in range(len(L)):
        pos=[j for j in range(len(L)) if y[j]==y[i] and int(d[j])//3!=int(d[i])//3 and int(d[j])%3!=int(d[i])%3]
        neg=[j for j in range(len(L)) if y[j]!=y[i]]
        a=[float(u[i]@u[j])/.1 for j in pos+neg];p=[float(u[i]@u[j])/.1 for j in pos]
        b=[float(u[i]@v[j])/.1 for j in range(len(OE))]
        known.append(math.log(sum(math.exp(t) for t in a))-sum(p)/len(p))
        outside.append(math.log(sum(math.exp(t) for t in a+b))-sum(p)/len(p))
        positives+=len(pos);negatives+=len(neg)
    return np.mean(known),np.mean(outside),positives,negatives


def test_hand_equation_excludes_same_class_noneligible_and_matches_all_OE_denominator():
    m=api();L,OE,y,d=inputs();expected=oracle(L,OE,y,d)
    for arm,index in (('KCON',0),('OECON',1)):
        a=m.contrastive(L,y,d,OE,arm)
        assert float(a['contrast'])==pytest.approx(expected[index],abs=8e-7)
        assert float(a['known_only_contrast'])==pytest.approx(expected[0],abs=8e-7)
        assert float(a['with_OE_contrast'])==pytest.approx(expected[1],abs=8e-7)
        assert a['positive_pair_count']==expected[2] and a['known_negative_pair_count']==expected[3]
        assert a['selected_OE_negative_pair_count']==(0 if arm=='KCON' else len(L)*len(OE))
        assert a['same_class_excluded_pair_count']==4 and a['anchor_count']==6
        assert float(a['with_OE_contrast'])>=float(a['known_only_contrast'])-1e-6
    permutation=torch.tensor([5,3,0,2,1,4])
    assert float(m.contrastive(L[permutation],y[permutation],d[permutation],OE.flip(0),'OECON')['contrast'])==pytest.approx(expected[1],abs=8e-7)


def test_selected_OE_gradients_and_total_keep_original_centered_teacher_detached():
    m=api();L,OE,y,d=inputs();L.requires_grad_();OE.requires_grad_()
    for arm in m.ARMS:
        a=m.contrastive(L,y,d,OE,arm);g=torch.autograd.grad(a['contrast'],(L,OE),allow_unused=True)
        assert float(g[0].norm())>0
        assert (g[1] is None) if arm=='KCON' else float(g[1].norm())>0
    logits=torch.randn(6,6,requires_grad=True);outside=torch.randn(3,6,requires_grad=True)
    teacher=torch.randn(6,6,requires_grad=True);od=torch.tensor([0,1,5])
    terms=m.compute(L,logits,y,teacher,d,OE,outside,od,'OECON')
    assert float(terms['total'].detach())==pytest.approx(float((terms['CE']+terms['KD']+terms['OE']+.1*terms['contrast']).detach()),abs=1e-6)
    terms['total'].backward()
    assert teacher.grad is None and float(logits.grad.norm())>0 and float(outside.grad.norm())>0
    assert float(L.grad.norm())>0 and float(OE.grad.norm())>0
    assert terms['contrast_weight']==.1 and terms['contrast_temperature']==.1 and terms['domain_count']==3
    assert terms['OE_true_labels_used'] is False and terms['OE_positive_pair_count']==0


def test_native_coordinate_validation_positive_coverage_and_no_OE_identity_API(subtests):
    m=api();L,OE,y,d=inputs()
    cases=[(L.double(),y,d,OE,'KCON'),(L[:,:159],y,d,OE,'KCON'),
           (L,y.float(),d,OE,'KCON'),(L,torch.full_like(y,-1),d,OE,'KCON'),
           (L,y,d.float(),OE,'KCON'),(L,y,torch.full_like(d,15),OE,'KCON'),
           (L,y,torch.zeros_like(d),OE,'KCON'),(L,y,d,OE[:,:159],'KCON'),
           (L,y,d,OE,'OTHER')]
    bad=L.clone();bad[0,0]=float('nan');cases.append((bad,y,d,OE,'KCON'))
    for values in cases:
        with subtests.test(case=len(values),dtype=str(values[0].dtype),arm=values[-1]),pytest.raises(ValueError):m.contrastive(*values)
    with subtests.test(extra_private_truth=True),pytest.raises(TypeError):m.contrastive(L,y,d,OE,'OECON',OE_y=torch.zeros(3,dtype=torch.long))
    with subtests.test(fixed_temperature=True),pytest.raises(TypeError):m.contrastive(L,y,d,OE,'KCON',temperature=.2)
