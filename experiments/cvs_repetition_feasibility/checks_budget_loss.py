"""Synthetic graph, normalization and permission checks for budget_loss."""

from __future__ import annotations

import json
import random

import numpy as np
import torch

from .budget_loss import budget_loss


def _raises(callback):
    try:callback()
    except ValueError:return
    raise AssertionError("expected budget loss input rejection")


def run_checks():
    py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    target=torch.zeros(3,14,dtype=torch.float64,requires_grad=True)
    signal=torch.tensor([.2,.4,0.],dtype=torch.float64,requires_grad=True)
    budget=torch.tensor([.1,.2,.3],dtype=torch.float64,requires_grad=True)
    teacher=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    # Teacher graph must remain detached through every observation.
    observed_target=target*teacher;observed_signal=signal*teacher;observed_budget=budget*teacher
    fulfilled=torch.zeros_like(target,requires_grad=True)
    zero=budget_loss(fulfilled,observed_target,observed_signal,observed_budget)
    assert zero["weighted_loss"].item()==0.
    zero["weighted_loss"].backward()
    assert torch.count_nonzero(fulfilled.grad)==0
    assert all(value.grad is None for value in (target,signal,budget,teacher))
    pred=torch.ones_like(target,requires_grad=True)
    snapshots=[value.detach().clone() for value in (pred,target,signal,budget,teacher)]
    active=budget_loss(pred,observed_target,observed_signal,observed_budget)
    mse=(pred-observed_target.detach()).square().mean(1)
    denominator=observed_signal.detach()+observed_budget.detach()+1e-8
    weight=observed_signal.detach()/denominator
    excess=torch.relu((mse-observed_budget.detach())/denominator)
    expected=.1*(weight*excess/(1+excess)).sum()/3
    assert torch.allclose(active["weighted_loss"],expected,atol=0,rtol=2e-15)
    assert bool((active["tensors"]["per_packet_loss"] <= .1*weight).all())
    gradients=torch.autograd.grad(active["weighted_loss"],(pred,target,signal,budget,teacher),allow_unused=True)
    assert bool(torch.isfinite(gradients[0]).all()) and bool((gradients[0][:2].abs().sum(1)>0).all())
    assert torch.count_nonzero(gradients[0][2])==0 and all(value is None for value in gradients[1:])
    twice=budget_loss(pred,observed_target,observed_signal,observed_budget,normalization_count=6,role="L_s")
    assert torch.equal(twice["weighted_loss"],active["weighted_loss"]/2)
    assert twice["summary"]["normalization_count"]==6
    # The unbounded excess can be large while its bounded contribution stays finite.
    extreme=budget_loss(torch.full((3,14),1e30,dtype=torch.float32,requires_grad=True),
                        torch.zeros(3,14),torch.ones(3),torch.zeros(3))
    assert torch.isfinite(extreme["weighted_loss"]) and extreme["weighted_loss"] <= .1
    boundary_pred=torch.ones_like(pred,requires_grad=True)
    exact_boundary=budget_loss(boundary_pred,target.detach(),torch.ones(3),torch.ones(3))
    boundary_grad=torch.autograd.grad(exact_boundary["weighted_loss"],boundary_pred)[0]
    assert exact_boundary["weighted_loss"].item()==0 and torch.count_nonzero(boundary_grad)==0
    for role in ("V","target","query","U_t"):
        _raises(lambda role=role:budget_loss(pred,target,signal,budget,role=role))
    for count in (2,3.0,True,0):
        _raises(lambda count=count:budget_loss(pred,target,signal,budget,normalization_count=count))
    for coefficient in (-.1,float("nan"),float("inf")):
        _raises(lambda coefficient=coefficient:budget_loss(pred,target,signal,budget,coefficient=coefficient))
    for eps in (0.,-1.,float("nan")):
        _raises(lambda eps=eps:budget_loss(pred,target,signal,budget,eps=eps))
    _raises(lambda:budget_loss(pred[:,:13],target[:,:13],signal,budget))
    _raises(lambda:budget_loss(pred,target,signal[:2],budget))
    _raises(lambda:budget_loss(pred,target,-signal,budget))
    _raises(lambda:budget_loss(pred,target,signal,-budget))
    nonfinite=pred.detach().clone();nonfinite[0,0]=float("inf")
    _raises(lambda:budget_loss(nonfinite,target,signal,budget))
    for value,before in zip((pred,target,signal,budget,teacher),snapshots):assert torch.equal(value.detach(),before)
    assert all(value.grad is None for value in (pred,target,signal,budget,teacher))
    assert random.getstate()==py_state and torch.equal(torch.get_rng_state(),torch_state)
    current_np=np.random.get_state()
    assert current_np[0]==np_state[0] and np.array_equal(current_np[1],np_state[1]) and current_np[2:]==np_state[2:]
    for item in (zero,active,twice,extreme,exact_boundary):json.dumps(item["summary"],allow_nan=False)
    return {"status":"VERIFIED","torch":torch.__version__,"device":"cpu",
            "checks":{"fulfilled_and_boundary_zero_derivative":True,"active_feature_gradient":True,
                      "observations_teacher_detached":True,"full_count_and_extra_denominator":True,
                      "per_packet_bound_extreme_finite":True,"source_role_only":True,
                      "invalid_inputs_constants_rejected":True,"inputs_and_RNG_unchanged":True},
            "active_loss":float(active["weighted_loss"].detach()),
            "active_prediction_gradient_norm":float(gradients[0].norm()),
            "boundary":"observation tolerance motivated by IID noise; no established noise-floor or TX/DG claim"}


if __name__=="__main__":print(json.dumps(run_checks(),ensure_ascii=False,allow_nan=False,indent=2))
