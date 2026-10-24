from copy import deepcopy
import pytest
import torch
from experiments.pamf_ssl_research.rejected_rdrop import RECIPE,rejected_pair_kl,rejected_rdrop
from experiments.pamf_ssl_research.source_rejected_rdrop import KIND,RULES,validate_plan


def example():
    torch.manual_seed(11)
    logits=[torch.randn(4,6,requires_grad=True) for _ in range(6)]
    accepted=torch.tensor([True,False,False,True]);physical=torch.tensor([False,True,False,True])
    return logits,accepted,physical


def symmetric_kl(a,b):
    p,q=a.log_softmax(1),b.log_softmax(1)
    return .5*((p.exp()-q.exp())*(p-q)).sum(1)


def test_only_rejected_nonphysical_rdrop_is_added_with_original_dose():
    logits,accepted,physical=example();before=accepted.clone()
    value,counts=rejected_rdrop(logits,accepted,physical)
    expected=(.5*(symmetric_kl(logits[0],logits[1])*(~accepted)).mean()
        +.25*(symmetric_kl(logits[2][~physical],logits[3][~physical])*(~accepted[~physical])).mean()
        +.25*(symmetric_kl(logits[4],logits[5])*(~accepted)).mean())/1.25
    torch.testing.assert_close(value,expected)
    assert torch.equal(accepted,before)
    assert counts['rejected_U']==2 and counts['rejected_nonphysical_half']==1
    assert counts['rejected_RDrop_pairs']==5 and counts['reused_prediction_views']==10
    assert counts['extra_model_forwards']==0 and not counts['hard_label_or_CE_on_rejected']
    assert counts['physical_As_KL_weight']==0


def test_rejected_physical_half_has_no_as_kl_but_raw_and_ordinary_receive_gradients():
    logits,accepted,physical=example();value,_=rejected_rdrop(logits,accepted,physical);value.backward()
    for index in (0,1,4,5):
        assert logits[index].grad[~accepted].norm()>0
        assert torch.equal(logits[index].grad[accepted],torch.zeros_like(logits[index].grad[accepted]))
    for index in (2,3):
        assert torch.equal(logits[index].grad[physical],torch.zeros_like(logits[index].grad[physical]))
        assert logits[index].grad[2].norm()>0
        assert logits[index].grad[0].norm()==0


def test_accepted_kl_plus_rejected_kl_equals_full_unmasked_kl():
    logits,accepted,_=example();a,b=logits[:2]
    accepted_loss=(symmetric_kl(a,b)*accepted).mean()
    torch.testing.assert_close(accepted_loss+rejected_pair_kl(a,b,accepted),symmetric_kl(a,b).mean())


def test_all_accepted_adds_exact_zero_and_preserves_connected_zero_gradients():
    logits,accepted,physical=example();accepted[:]=True
    value,counts=rejected_rdrop(logits,accepted,physical);assert value==0 and counts['rejected_RDrop_pairs']==0
    value.backward();assert all(torch.count_nonzero(x.grad)==0 for x in logits)


def test_empty_pair_is_finite_differentiable_zero():
    a=torch.empty(0,6,requires_grad=True);b=a.detach().clone().requires_grad_()
    value=rejected_pair_kl(a,b,torch.empty(0,dtype=torch.bool));value.backward()
    assert value==0 and a.grad is not None and b.grad is not None


@pytest.mark.parametrize('fault',['missing_view','odd_batch','wrong_mask_dtype','unequal_half','nonfinite'])
def test_malformed_or_nonfinite_observations_are_rejected(fault):
    logits,accepted,physical=example()
    if fault=='missing_view':logits=logits[:-1]
    elif fault=='odd_batch':logits=[x[:3] for x in logits];accepted=accepted[:3];physical=physical[:3]
    elif fault=='wrong_mask_dtype':accepted=accepted.long()
    elif fault=='unequal_half':physical[:]=False
    else:logits[0]=torch.full((4,6),float('nan'))
    with pytest.raises(ValueError):rejected_rdrop(logits,accepted,physical)


def valid_plan():
    return dict(kind=KIND,source_rules=deepcopy(RULES),recipe=deepcopy(RECIPE),seeds=[2026092701,2026092702,2026092703],
        source_epoch=20,steps_per_seed=1,L_batch=128,U_batch=256,optimizer_state='fresh_moments_first_new_native10_step',
        V_reads=0,target_reads=0,U_truth_reads=0,G_training_reads=0)


def test_only_fixed_original_source_matrix_is_allowed():
    validate_plan(valid_plan())


@pytest.mark.parametrize('key,value',[('source_epoch',30),('steps_per_seed',2),('L_batch',256),('V_reads',1),
    ('target_reads',1),('U_truth_reads',1),('G_training_reads',1),('seeds',[2026092701]),
    ('optimizer_state','inherited_unknown'),('recipe',{}),('source_rules',{})])
def test_other_data_state_or_scopes_are_blocked(key,value):
    plan=valid_plan();plan[key]=value
    with pytest.raises(ValueError):validate_plan(plan)
