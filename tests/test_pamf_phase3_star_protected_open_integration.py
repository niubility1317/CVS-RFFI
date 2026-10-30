"""Synthetic cached-source update through the actual model/loss interfaces."""
from copy import deepcopy

import pytest
import torch

from experiments.pamf_phase3_open_world import star_protected_open_model as model_api
from experiments.pamf_phase3_open_world import star_protected_open_loss as loss_api
from test_pamf_phase3_star_reference_open_model import fixture


@pytest.mark.parametrize('enabled',[False,True],ids=['R0','R1'])
def test_cached_episode_neighbor_and_OE_update_keep_identity_and_teacher_frozen(enabled):
    torch.set_num_threads(1)
    model,_=fixture();model_api.install(model,residual_enabled=enabled)
    native={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('open_world.')}
    reader=model.id_backbone.encoder
    native_calls=(reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)
    grid=[(r,d) for r in (1,3,4,6,8) for d in (1,2,3)]
    support_domains=[cell for c in range(6) for cell in grid+[grid[c]]]
    query_domains=[cell for c in range(6) for cell in grid+[grid[(c+1)%15]]]
    domains=support_domains+query_domains
    rx=torch.tensor([r for r,d in domains]);day=torch.tensor([d for r,d in domains])
    labels=torch.arange(6).repeat_interleave(16);all_labels=torch.cat((labels,labels))
    generator=torch.Generator().manual_seed(2026103017)
    centers=torch.randn(6,160,generator=generator)
    original=(centers[all_labels]+.05*torch.randn(192,160,generator=generator)).requires_grad_(True)
    contributions=torch.stack((original*.4,original*.3,original*.2,original*.1),1).detach().requires_grad_(True)
    oe_domains=grid*4+grid[:4]
    oe_rx=torch.tensor([r for r,d in oe_domains]);oe_day=torch.tensor([d for r,d in oe_domains])
    oe_original=torch.randn(64,160,generator=generator).requires_grad_(True)
    oe_contributions=torch.stack((oe_original*.4,oe_original*.3,oe_original*.2,oe_original*.1),1).detach().requires_grad_(True)
    old_parts=deepcopy(model.open_world.state_dict())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        z=model_api.encode(model,original,contributions)['protected160']
        oe_z=model_api.encode(model,oe_original,oe_contributions)['protected160']
        proto=loss_api.query_prototypes(z[:96],labels,rx[:96],day[:96],rx[96:],day[96:])
        oe_proto=loss_api.query_prototypes(z[:96],labels,rx[:96],day[:96],oe_rx,oe_day)
        members=loss_api.member_loss(model_api.relation(model,z[96:],proto),labels,torch.tensor([0,2,4]))
        outside=loss_api.oe_member_loss(model_api.relation(model,oe_z,oe_proto))
        local_L=loss_api.soft_neighbor_loss(z,original,rx,day,torch.arange(192))
        local_OE=loss_api.soft_neighbor_loss(oe_z,oe_original,oe_rx,oe_day,torch.arange(1000,1064))
        total=members['loss']+outside['loss']+.1*.5*(local_L['loss']+local_OE['loss'])
        assert torch.isfinite(total)
        total.backward()
        assert model.open_world.relation_scorer[0].weight.grad.abs().sum()>0
        if enabled:assert model.open_world.residual_gain.grad.abs().sum()>0
        else:assert model.open_world.residual_gain.grad is None
        assert original.grad is contributions.grad is oe_original.grad is oe_contributions.grad is None
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
        optimizer.step()
    assert (reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)==native_calls
    for k,value in native.items():assert torch.equal(model.state_dict()[k],value)
    assert not torch.equal(model.open_world.state_dict()['relation_scorer.0.weight'],old_parts['relation_scorer.0.weight'])
    if enabled:assert not torch.equal(model.open_world.residual_gain,old_parts['residual_gain'])
    else:
        for k,value in old_parts.items():
            if not k.startswith('relation_scorer.'):assert torch.equal(model.open_world.state_dict()[k],value)
