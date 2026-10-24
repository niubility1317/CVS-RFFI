from copy import deepcopy
import pytest
import torch
from experiments.cric_dura_comparison import f_optimizer
from experiments.pamf_ssl_research.optimizer_step_guard import OptimizerStepGuard,installed
from test_pamf_optimizer_reference import make,grad


@pytest.mark.parametrize('history',[0,2])
def test_guard_uses_current_state_and_real_fp32_parameters(history):
    model=make();opt=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
    for i in range(history):
        for p,g in zip(model.parameters(),grad(model,2+i)):p.grad=g.clone()
        f_optimizer.clip_physical(opt,model.parameters(),5.);opt.step()
    initial_steps=opt._sinc_unit_steps
    named=list(model.named_parameters());full=grad(model,1.)
    guard=OptimizerStepGuard()
    with installed():
        guard.capture(model,named,full,full,full)
        for p,g in zip(model.parameters(),grad(model,-2.)):p.grad=g.clone()
        f_optimizer.clip_physical(opt,model.parameters(),5.)
        assert opt._sinc_unit_steps==initial_steps
        after_clip=deepcopy(opt.state_dict())
        opt.step()
        state=guard.state_dict()
        assert state['calls']==state['reference_optimizer_steps']==1
        assert state['pending_anchors'] is False
        assert state['extra_autograd_calls']==state['extra_model_forwards']==0
        assert all(d['actual_parameter_check'] and d['norm_nonexpansion'] for d in guard.last_details.values())
        for d in guard.last_details.values():
            assert all(c is None or c>=-1.000001e-6 for c in d['anchor_cosines_after'].values())
        # Only the real joint step advances the live AdamW moments.
        for parameter,values in opt.state.items():
            assert values['step'].item()==history+1
    assert opt._sinc_unit_steps==history+1 and opt._sinc_physical_clip_calls==history+1
    assert guard.hook is None


def test_guard_requires_active_scope_and_consumes_exactly_once():
    model=make();named=list(model.named_parameters());full=grad(model,1.);guard=OptimizerStepGuard()
    with pytest.raises(ValueError):guard.capture(model,named,full,full,full)
    with pytest.raises(ValueError):
        with installed():
            guard.capture(model,named,full,full,full)
            with pytest.raises(ValueError):guard.capture(model,named,full,full,full)
            guard.before_clip(None,[p for _,p in named],5.,torch.nn.utils.clip_grad_norm_)


def test_scope_restores_adapter_after_failure():
    before=f_optimizer.clip_physical
    with pytest.raises(RuntimeError):
        with installed():raise RuntimeError('synthetic failure')
    assert f_optimizer.clip_physical is before


def test_native_scope_and_shadow_do_not_construct_a_second_owner_optimizer():
    model=make();owner=type('Owner',(),{})();owner.model=model;owner.optimizer=None
    with installed(),f_optimizer.native_scope(owner,'CENTER_UNITS'):
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
        guard=OptimizerStepGuard();named=list(model.named_parameters());full=grad(model,1.)
        guard.capture(model,named,full,full,full)
        for p,g in zip(model.parameters(),grad(model,-1.)):p.grad=g.clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        assert owner.optimizer is opt and guard.state_dict()['calls']==1
