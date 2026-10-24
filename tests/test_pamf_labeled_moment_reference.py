from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from experiments.cric_dura_comparison import f_optimizer
from experiments.pamf_ssl_research.labeled_moment_reference import LabeledMomentReference
from experiments.pamf_ssl_research.labeled_moment_guard import LabeledMomentGuard
from experiments.pamf_ssl_research.optimizer_step_guard import installed
from test_pamf_optimizer_reference import make,grad


@pytest.mark.parametrize('unused',[False,True])
def test_reference_carries_only_L_history_at_current_live_coordinates(unused):
    torch.manual_seed(58);model=make();named=list(model.named_parameters());oracle=deepcopy(model)
    live=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    expected=f_optimizer.build(oracle,oracle.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    reference=LabeledMomentReference()
    for i in range(3):
        live.param_groups[0]['lr']=2e-4/(i+1);expected.param_groups[0]['lr']=live.param_groups[0]['lr']
        gradients=list(grad(model,.001*(i+1)))
        if unused and i==1:gradients[0]=None
        before=deepcopy(model.state_dict());moments=deepcopy(live.state_dict())
        with torch.no_grad():
            for p,q in zip(oracle.parameters(),model.parameters()):p.copy_(q)
        for p,g in zip(oracle.parameters(),gradients):p.grad=None if g is None else g.clone()
        f_optimizer.clip_physical(expected,oracle.parameters(),5.);expected.step()
        with patch.object(live,'state_dict',side_effect=AssertionError('Joint moments must never be copied')):
            values,meta=reference.step(model,named,live,gradients,clip_impl=torch.nn.utils.clip_grad_norm_)
        assert all(torch.equal(a,b) for a,b in zip(values,oracle.parameters()))
        assert all(torch.equal(model.state_dict()[k],v) for k,v in before.items())
        for index,entry in moments['state'].items():
            for key,value in entry.items():
                if torch.is_tensor(value):assert torch.equal(live.state_dict()['state'][index][key],value)
        assert meta['copied_existing_moments'] is False and meta['independent_labeled_moments'] is True
        assert meta['reference_moment_steps']==i+1 and meta['live_optimizer_updates']==0
        for p,g in zip(model.parameters(),grad(model,.01*(i+1))):p.grad=g.clone()
        f_optimizer.clip_physical(live,model.parameters(),5.);live.step()
    assert reference.optimizer._sinc_unit_steps==3
    assert reference.optimizer._sinc_physical_clip_calls==3


def test_independent_reference_guard_can_remove_history_conflict_when_current_U_zero():
    model=make();named=list(model.named_parameters());full=grad(model,.001)
    live=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    guard=LabeledMomentGuard()
    with installed():
        for multiplier in (10.,1.):
            guard.capture(model,named,full,full,full)
            for p,g in zip(model.parameters(),full):p.grad=multiplier*g.clone()
            f_optimizer.clip_physical(live,model.parameters(),5.);live.step()
        state=guard.state_dict()
        assert state['calls']==state['reference_optimizer_steps']==2
        assert state['reference_moment_steps']==2 and state['copied_existing_moments'] is False
        assert state['independent_labeled_moments'] and state['extra_autograd_calls']==state['extra_model_forwards']==0
        assert any(c is not None and c<-1e-6 for v in guard.last_details.values() for c in v['anchor_cosines_before'].values())
        assert all(v['actual_parameter_check'] and v['norm_nonexpansion'] for v in guard.last_details.values())
        assert all(c is None or c>=-1.000001e-6 for v in guard.last_details.values() for c in v['anchor_cosines_after'].values())
        assert live._sinc_unit_steps==2
    assert guard.hook is None and guard.labeled_reference.optimizer is None


@pytest.mark.parametrize('bad',['order','missing','nonfinite','foreign_optimizer','changed_model'])
def test_reference_rejects_misaligned_parameters_gradients_and_optimizer(bad):
    model=make();named=list(model.named_parameters());full=list(grad(model,.001))
    live=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    reference=LabeledMomentReference()
    if bad=='changed_model':
        reference.step(model,named,live,full,clip_impl=torch.nn.utils.clip_grad_norm_)
        model=make();named=list(model.named_parameters())
        live=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    elif bad=='order':named.reverse()
    elif bad=='missing':full.pop()
    elif bad=='nonfinite':full[0].fill_(float('nan'))
    else:
        other=make();live=f_optimizer.build(other,other.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    with pytest.raises(ValueError):reference.step(model,named,live,full,clip_impl=torch.nn.utils.clip_grad_norm_)
