from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from torch import nn
from experiments.pamf_ssl_research.branch_gradient_projection import branch_projected_gradients
from experiments.pamf_ssl_research.source_uncapped_step import KIND,RULES,validate_plan,installed_counterfactual


def plan():
    return dict(kind=KIND,source_rules=deepcopy(RULES),seeds=[2026092701,2026092702,2026092703],source_epoch=20,
        steps_per_seed=1,L_batch=128,U_batch=256,V_reads=0,target_reads=0,U_truth_reads=0,G_training_reads=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_uncapped_keeps_large_feasible_U_and_same_complete_L(dtype):
    p=nn.Parameter(torch.tensor([.2,.3],dtype=dtype));named=[('encoder.t2.weight',p)]
    gl=(torch.tensor([1.,1.],dtype=dtype),);gu=(torch.tensor([10.,10.],dtype=dtype),)
    clean=(torch.tensor([1.,0.],dtype=dtype),);physical=(torch.tensor([0.,1.],dtype=dtype),)
    old,cap=branch_projected_gradients(named,gl,gu,clean,physical,max_u_ratio=2.)
    new,details=branch_projected_gradients(named,gl,gu,clean,physical,max_u_ratio=None)
    torch.testing.assert_close(new[0],gl[0]+gu[0],rtol=0,atol=0)
    assert details['t2']['applied_U_norm']>cap['t2']['applied_U_norm']*4.9
    assert details['t2']['capped'] is False and cap['t2']['capped'] is True
    assert all(c is None or c>=-1e-6 for c in details['t2']['anchor_cosines_after'].values())


def test_uncapped_still_removes_conflicting_directions_without_changing_L():
    p=nn.Parameter(torch.tensor([.2,.3]));named=[('encoder.t2.weight',p)]
    gl=(torch.tensor([1.,1.]),);gu=(torch.tensor([-10.,20.]),)
    clean=(torch.tensor([1.,0.]),);physical=(torch.tensor([0.,1.]),)
    new,details=branch_projected_gradients(named,gl,gu,clean,physical,max_u_ratio=None)
    torch.testing.assert_close(new[0],torch.tensor([1.,21.]),rtol=0,atol=1e-6)
    assert details['t2']['applied_U_norm']==pytest.approx(20.) and not details['t2']['capped']


def test_fixed_source_plan():validate_plan(plan())


@pytest.mark.parametrize('change',[{'target_reads':1},{'U_truth_reads':1},{'source_epoch':30},{'steps_per_seed':2},
    {'seeds':[2026092701]},{'source_rules':dict(RULES,minimum_actual_increment_norm_retention_per_nonzero_group=.1)}])
def test_rejects_changed_scope_or_posthoc_threshold(change):
    p=plan();p.update(change)
    with pytest.raises(ValueError):validate_plan(p)


def test_nested_source_callbacks_restore_on_failure():
    from experiments.pamf_ssl_research import source_branch_gradients as source
    from experiments.pamf_ssl_research import source_optimizer_shadow as shadow
    from experiments.pamf_ssl_research import source_optimizer_counterfactual as optimizer_source
    original=(source.gradient_facts,shadow.build,optimizer_source.shadow_step,torch.autograd.grad)
    with pytest.raises(RuntimeError):
        with installed_counterfactual():raise RuntimeError('synthetic source failure')
    assert original==(source.gradient_facts,shadow.build,optimizer_source.shadow_step,torch.autograd.grad)
