from copy import deepcopy
import pytest
import torch
from test_pamf_optimizer_reference import make,grad
from experiments.pamf_ssl_research.source_moment_history import KIND,RULES,validate_plan,history_probe,installed_counterfactual


def plan():
    return dict(kind=KIND,source_rules=deepcopy(RULES),seeds=[2026092701,2026092702,2026092703],
        source_epoch=20,steps_per_seed=1,history_steps=2,L_batch=128,U_batch=256,
        V_reads=0,target_reads=0,U_truth_reads=0,G_training_reads=0)


def test_same_current_L_gradient_can_retain_U_history_outside_copied_moment_guard():
    torch.manual_seed(31);model=make();named=list(model.named_parameters());full=grad(model,.001)
    original=deepcopy(model.state_dict());joint=tuple(10*g for g in full)
    facts=history_probe(model,named,full,full,full,joint,base_lr=2e-4,weight_decay=.01)
    assert facts['optimizer_steps']==5 and facts['current_U_gradient_zero']
    assert facts['current_copied_reference_matches_actual_L_step']
    assert all(v['raw_increment_norm']==0 and v['applied_increment_norm']==0 for v in facts['current_guard_groups'].values())
    assert facts['negative_history_drift_anchors']
    assert all(torch.equal(model.state_dict()[k],v) for k,v in original.items())
    assert all(p.grad is None for p in model.parameters())


def test_identical_L_history_has_zero_drift():
    model=make();named=list(model.named_parameters());full=grad(model,.001)
    facts=history_probe(model,named,full,full,full,full,base_lr=2e-4,weight_decay=.01)
    assert not facts['negative_history_drift_anchors']
    assert all(v['dot']==0 and v['right_norm']==0 for x in facts['history_extra_descent_relations'].values() for v in x.values())
    assert all(v==0 for group in facts['moment_difference_norms'].values() for v in group.values())


def test_fixed_source_plan():validate_plan(plan())


@pytest.mark.parametrize('change',[{'target_reads':1},{'U_truth_reads':1},{'V_reads':1},{'source_epoch':30},
    {'steps_per_seed':2},{'history_steps':3},{'seeds':[2026092701]},
    {'source_rules':dict(RULES,material_cosine_threshold=-.1)}])
def test_rejects_changed_scope_or_posthoc_rules(change):
    p=plan();p.update(change)
    with pytest.raises(ValueError):validate_plan(p)


def test_source_hooks_restore_after_failure():
    from experiments.pamf_ssl_research import source_branch_gradients as source
    before=(source.gradient_facts,source.BranchDiagnostic,torch.autograd.grad)
    with pytest.raises(RuntimeError):
        with installed_counterfactual():raise RuntimeError('synthetic source failure')
    assert before==(source.gradient_facts,source.BranchDiagnostic,torch.autograd.grad)
