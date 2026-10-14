"""Synthetic graph checks only; no models, data, optimizer or training."""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .objective import combine, objective


RX=(1,3,4,6,8)


def _fixture():
    y=torch.arange(6).repeat_interleave(5)
    rx=torch.tensor(RX).repeat(6)
    day=torch.ones(30,dtype=torch.int64)
    ids=[f"packet-{i}" for i in range(30)]
    return y,rx,day,ids


def _raises(callback):
    try:callback()
    except ValueError:return
    raise AssertionError("expected fixed source/layer/gate rejection")


def run_checks():
    py_rng=random.getstate();np_rng=np.random.get_state();torch_rng=torch.get_rng_state().clone()
    y,rx,day,ids=_fixture()
    identity=F.one_hot(y,160).to(torch.float64).requires_grad_()
    satisfied=objective(identity,y,rx,day,ids,"ORDER")
    satisfied["weighted_loss"].backward()
    assert satisfied["summary"]["triplet_count"]==600
    assert satisfied["summary"]["unsatisfied_margin_count"]==0
    assert satisfied["weighted_loss"].item()==0. and torch.count_nonzero(identity.grad)==0
    # Pure RX encoding has active margin ties but zero derivative because its
    # detached positive and negative are identical. Do not hide that degeneracy.
    rx_index=torch.tensor([RX.index(int(value)) for value in rx])
    pure_rx=F.one_hot(rx_index,160).to(torch.float64).requires_grad_()
    tied=objective(pure_rx,y,rx,day,ids,"ORDER")
    assert abs(tied["weighted_loss"].item()-.01)<1e-15
    assert tied["summary"]["unsatisfied_margin_count"]==600
    tied_grad=torch.autograd.grad(tied["weighted_loss"],pure_rx)[0]
    # The mathematical donor difference is zero; grouped FP64 reductions may
    # leave cancellation roundoff. Do not call that a useful active gradient.
    assert float(tied_grad.norm()) < 1e-16
    # A weak identity component within RX-confounded features gives a closed
    # form active relation .1-.04/1.04, without adjusting any loss parameter.
    confounded=F.one_hot(rx_index,160).to(torch.float64)
    confounded[torch.arange(30),10+y]=.2
    confounded.requires_grad_()
    active=objective(confounded,y,rx,day,ids,"ORDER")
    expected=.1*(.1-.04/1.04)
    assert abs(active["weighted_loss"].item()-expected)<2e-15
    all_true={str(receiver):True for receiver in RX};all_false={receiver:False for receiver in RX}
    assert torch.equal(combine(active["loss_by_RX"],all_true),active["weighted_loss"])
    chosen=combine(active["loss_by_RX"],{receiver:receiver==3 for receiver in RX})
    assert abs(chosen.item()-expected/5)<2e-15
    chosen_gradient=torch.autograd.grad(chosen,confounded,retain_graph=True)[0]
    assert bool(torch.isfinite(chosen_gradient).all())
    assert bool((chosen_gradient[rx==3].norm(dim=1)>0).all())
    assert torch.count_nonzero(chosen_gradient[rx!=3])==0
    rejected=combine(active["loss_by_RX"],all_false)
    rejected_gradient=torch.autograd.grad(rejected,confounded,retain_graph=True)[0]
    assert rejected.item()==0 and torch.count_nonzero(rejected_gradient)==0
    assert torch.equal(combine(active["loss_by_RX"],torch.ones(5,dtype=torch.bool)),active["weighted_loss"])
    # Single triplet proves donor endpoint graph is stopped independently of
    # whether a donor is also an anchor in another source context.
    single=torch.zeros(3,160,dtype=torch.float64)
    single[0,0]=1.;single[1,1]=1.;single[2,0]=.8;single[2,2]=.6
    single.requires_grad_()
    sy=torch.tensor([0,0,1]);sr=torch.tensor([1,3,3]);sd=torch.ones(3,dtype=torch.long);si=["a","p","n"]
    single_result=objective(single,sy,sr,sd,si,"ORDER")
    assert single_result["tensors"]["ORDER_triplets"].tolist()==[[0,1,2]]
    assert abs(single_result["weighted_loss"].item()-.09)<2e-15
    donor_grad=torch.autograd.grad(single_result["weighted_loss"],single)[0]
    assert donor_grad[0].norm()>0 and torch.count_nonzero(donor_grad[1:])==0
    assert single_result["loss_by_RX"]["1"].item()==single_result["weighted_loss"].item()
    # Mix active/satisfied relations and remove every duplicate occurrence.
    mixed=identity.detach().clone();mixed[0]=0.;mixed[0,1]=1.;mixed.requires_grad_()
    duplicate_ids=list(ids);duplicate_ids[21]=duplicate_ids[20]
    duplicate=objective(mixed,y,rx,day,duplicate_ids,"ORDER")
    eligible=duplicate["tensors"]["cohort_indices"].tolist()
    counts=Counter(duplicate_ids)
    assert 20 not in eligible and 21 not in eligible and duplicate["summary"]["duplicate_excluded_count"]==2
    expected_triples={(i,p,n) for i in eligible for p in eligible for n in eligible
                      if len({duplicate_ids[i],duplicate_ids[p],duplicate_ids[n]})==3
                      and y[i]==y[p] and day[i]==day[p] and rx[i]!=rx[p]
                      and y[n]!=y[p] and rx[n]==rx[p] and day[n]==day[p]}
    actual=duplicate["tensors"]["ORDER_triplets"]
    assert set(map(tuple,actual.tolist()))==expected_triples
    normalized=F.normalize(mixed,dim=1,eps=1e-4)
    relations=torch.stack([.1+normalized[i]@normalized[n]-normalized[i]@normalized[p] for i,p,n in actual.tolist()])
    assert bool((relations>0).any()) and bool((relations<=0).any())
    assert torch.allclose(duplicate["weighted_loss"],.1*relations.relu().sum()/len(actual),atol=1e-15,rtol=0)
    for receiver in RX:
        mask=rx[actual[:,0]]==receiver
        assert torch.allclose(duplicate["loss_by_RX"][str(receiver)],.1*relations[mask].relu().sum()/len(actual),atol=1e-15,rtol=0)
    assert all(counts[duplicate_ids[i]]==1 for i in eligible)
    zero_features=confounded.detach().clone().requires_grad_()
    zero=objective(zero_features,y,rx,day,ids,"ZERO")
    zero_gradient=torch.autograd.grad(zero["weighted_loss"],zero_features)[0]
    assert zero["weighted_loss"].item()==0 and torch.count_nonzero(zero_gradient)==0
    assert zero["summary"]["triplet_count"]==active["summary"]["triplet_count"]
    # Ordinary SupCon keeps all endpoint graphs and has the original formula.
    sup_features=confounded.detach().clone().requires_grad_()
    sup=objective(sup_features,y,rx,day,ids,"SUPCON")
    sf=F.normalize(sup_features,dim=1,eps=1e-4);rows=[]
    for i in range(30):
        positive=[j for j in range(30) if j!=i and y[j]==y[i]]
        logden=torch.logsumexp(torch.stack([sf[i]@sf[j]/.1 for j in range(30) if j!=i]),dim=0)
        rows.append(torch.stack([logden-sf[i]@sf[j]/.1 for j in positive]).mean())
    assert torch.allclose(sup["weighted_loss"],.1*torch.stack(rows).mean(),atol=2e-15,rtol=0)
    stable_features=single.detach().to(torch.float32).requires_grad_()
    stable=objective(stable_features,sy,sr,sd,si,"SUPCON")
    stable_gradient=torch.autograd.grad(stable["weighted_loss"],stable_features)[0]
    assert bool(torch.isfinite(stable["weighted_loss"])) and bool(torch.isfinite(stable_gradient).all())
    class_w=torch.ones(6,160,dtype=torch.float64,requires_grad=True)
    w_grad=torch.autograd.grad(active["weighted_loss"],class_w,allow_unused=True)[0]
    assert w_grad is None
    for mode in ("ZERO","ORDER","SUPCON"):
        empty=torch.zeros(0,160,dtype=torch.float64,requires_grad=True)
        answer=objective(empty,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),[],mode)
        assert answer["weighted_loss"].item()==0 and answer["summary"]["triplet_count"]==0
        assert torch.count_nonzero(torch.autograd.grad(answer["weighted_loss"],empty)[0])==0
    for role in ("U_s","V","target","query","unknown"):
        _raises(lambda role=role:objective(confounded,y,rx,day,ids,"ORDER",role=role))
        _raises(lambda role=role:combine(active["loss_by_RX"],all_true,role=role))
    for layer in ("joint160","SIG_TIME_after_gain160",None):
        _raises(lambda layer=layer:objective(confounded,y,rx,day,ids,"ORDER",layer_kind=layer))
        _raises(lambda layer=layer:combine(active["loss_by_RX"],all_true,layer_kind=layer))
    for gate in ({"1":True}, {str(r):1 for r in RX}, torch.ones(5),torch.ones(4,dtype=torch.bool),
                 {**all_true,"9":False}):
        _raises(lambda gate=gate:combine(active["loss_by_RX"],gate))
    hidden=y.clone();hidden[0]=-1
    _raises(lambda:objective(confounded,hidden,rx,day,ids,"ORDER"))
    _raises(lambda:objective(confounded,y.float(),rx,day,ids,"ORDER"))
    _raises(lambda:objective(confounded,None,rx,day,ids,"ORDER"))
    _raises(lambda:objective(confounded,y,rx,day,ids,"ORDER",margin=.2))
    _raises(lambda:objective(confounded,y,rx,day,ids,"ORDER",coefficient=.2))
    _raises(lambda:objective(confounded,y,rx,day,ids,"ORDER",temperature=.2))
    _raises(lambda:objective(confounded[:,:159],y,rx,day,ids,"ORDER"))
    permutation=torch.tensor([3,5,1,4,0,2])
    permuted=objective(confounded,permutation[y],rx,day,ids,"ORDER")
    assert torch.equal(permuted["weighted_loss"],active["weighted_loss"])
    assert random.getstate()==py_rng and torch.equal(torch.get_rng_state(),torch_rng)
    current=np.random.get_state()
    assert current[0]==np_rng[0] and np.array_equal(current[1],np_rng[1]) and current[2:]==np_rng[2:]
    for result in (satisfied,tied,active,single_result,duplicate,zero,sup):json.dumps(result["summary"],allow_nan=False)
    return {"status":"VERIFIED","torch":torch.__version__,"device":"cpu","dtype":"float64",
            "checks":{"true_class_onehot_loss_gradient_zero":True,"RX_confounded_closed_form":True,
                      "anchor_only_donor_gradient_stopped":True,"single_gate_only_its_RX_anchor_gradient":True,
                      "all_true_ungated_all_false_zero":True,"duplicates_full_triplet_denominator":True,
                      "ordinary_SUPCON_formula":True,"ZERO_exact_graph_zero":True,
                      "float32_SUPCON_finite":True,
                      "W_dependency_none_RNG_unchanged":True,"role_layer_gate_visible_label_negatives":True},
            "single_triplet_endpoint_gradient_norms":donor_grad.norm(dim=1).tolist(),
            "selected_RX_anchor_gradient_norm":float(chosen_gradient[rx==3].norm()),
            "RX_confounded_loss":float(active["weighted_loss"].detach()),
            "pure_RX_active_margin_gradient_norm":float(tied_grad.norm()),
            "boundary":"synthetic loss only; actual AdamW credit and default W_pre utility supplied by caller, no efficacy claim"}


if __name__=="__main__":print(json.dumps(run_checks(),ensure_ascii=False,allow_nan=False,indent=2))
