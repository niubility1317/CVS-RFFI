import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_identity_flow.training import align_g10, mix_epoch, objective, FixedAntitheticFlow
from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_identity_flow.source_train import source_view, EPOCHS, STEPS


class Reader(nn.Module):
    def features(self, x):
        return torch.cat((x.mean(2), x.square().mean(2), x[:, :, :64].mean(2)), 1)


def test_full_thirty_epoch_budget_reuses_declared_source_views():
    assert EPOCHS==30 and STEPS==14790
    assert [source_view(i) for i in range(1,31)]==list(range(1,21))+list(range(1,11))
    with pytest.raises(ValueError):source_view(31)


def test_g10_alignment_rejects_duplicate_or_non_source_ids():
    ids=np.array(['a','b','c','d']);roles=np.array(['L_s','U_s','L_s','U_s'])
    assert align_g10(ids,roles,np.array(['c','a'])).tolist()==[2,0]
    with pytest.raises(ValueError):align_g10(ids,roles,np.array(['c','c']))
    with pytest.raises(ValueError):align_g10(ids,roles,np.array(['c','b']))


def test_fixed_epoch_mixture_preserves_matched_clean_and_received():
    clean=np.arange(8,dtype=np.float32)[:,None,None]*np.ones((8,2,256),np.float32)
    received=clean+100
    g10_clean=clean[[0,2]];g10=g10_clean+200
    x,y,counts=mix_epoch(clean,received,g10_clean,g10,np.arange(8),np.array([1,0]))
    assert counts=={'r07':4,'g10':2,'clean':2}
    assert np.all(y[:4]-x[:4]==100)
    assert np.all(y[4:6]-x[4:6]==200)
    assert np.array_equal(y[6:],x[6:])


def test_teacher_frozen_but_guidance_and_flow_both_reach_field():
    torch.manual_seed(8)
    model=IdentityConditionalFlow(width=16,layers=1,heads=4)
    teacher=Reader().eval().requires_grad_(False)
    clean=torch.randn(3,2,256);received=clean+.2*torch.randn_like(clean)
    t=torch.tensor([.2,.4,.7]);noise1=torch.randn_like(clean);noise2=torch.randn_like(clean)
    loss,parts=objective(model,teacher,clean,received,t,noise1,noise2)
    assert loss.shape==(3,) and torch.isfinite(loss).all()
    grad=torch.autograd.grad(parts['identity'].sum(),model.output.weight,retain_graph=True)[0]
    flow=torch.autograd.grad(parts['flow'].sum(),model.output.weight)[0]
    assert grad.isfinite().all() and grad.abs().sum()>0
    assert flow.isfinite().all() and flow.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())


def test_fixed_antithetic_inference_has_no_batch_or_order_dependency():
    torch.manual_seed(7)
    model=FixedAntitheticFlow(IdentityConditionalFlow(width=16,layers=1,heads=4)).eval()
    with torch.no_grad():
        model.flow.output.weight.normal_(std=.01)
        x=torch.randn(3,2,256)
        actual=model(x)
        separate=torch.cat([model(y[None]) for y in x])
        perm=torch.tensor([2,0,1])
        torch.testing.assert_close(actual,separate,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(model(x[perm]),actual[perm],rtol=1e-5,atol=1e-6)
