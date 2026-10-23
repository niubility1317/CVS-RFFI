from copy import deepcopy
import pytest
import torch
from experiments.pamf_ssl_research.u_cmr_objective import conditional_moments,UnlabeledReceiverMomentsObjective
from experiments.pamf_ssl_research.u_cmr import configuration,validate,RECIPE
from experiments.pamf_ssl_research.plan import configuration as scaffold,SEEDS
from tests.test_pamf_ssl_research import template
from tests.test_pamf_ssl_u_mixstyle import StyleStudent


def test_conditional_rx_mean_shift_has_analytic_value_and_student_only_gradient():
    h=torch.tensor([[-2.],[-1.],[1.],[2.],[8.],[9.],[11.],[12.]],requires_grad=True)
    p=torch.tensor([[.9,.1]]*8,requires_grad=True);w=torch.ones(8,requires_grad=True)
    rx=torch.tensor([1]*4+[3]*4)
    loss,detail=conditional_moments(h,p,w,rx,[1,3,4,6,8],RECIPE)
    expected=torch.tensor(100.)/h.detach().var(0,unbiased=False)
    assert torch.allclose(loss,expected.squeeze())
    assert detail['receiver_pairs']==1 and detail['valid_cells']==2 and detail['valid_classes']==1
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad.norm()>0
    assert p.grad is None and w.grad is None


def test_matching_moments_do_not_pull_different_classes_together():
    h=torch.tensor([[-2.],[-1.],[1.],[2.]]*2+[[18.],[19.],[21.],[22.]]*2,requires_grad=True)
    p=torch.tensor([[1.,0.]]*8+[[0.,1.]]*8)
    rx=torch.tensor(([1]*4+[3]*4)*2)
    loss,d=conditional_moments(h,p,torch.ones(16),rx,[1,3,4,6,8],RECIPE)
    assert loss==0 and d['valid_classes']==2 and d['receiver_pairs']==2
    loss.backward();assert torch.equal(h.grad,torch.zeros_like(h))


def test_standard_deviation_shift_is_penalized_and_weighted_ess_is_used():
    h=torch.tensor([[-1.],[-1.],[1.],[1.],[-2.],[-2.],[2.],[2.]],requires_grad=True)
    p=torch.ones(8,1);rx=torch.tensor([1]*4+[3]*4)
    loss,d=conditional_moments(h,p,torch.ones(8),rx,[1,3,4,6,8],RECIPE)
    assert loss>0 and d['std_loss']>0 and d['mean_loss']==0
    w=torch.tensor([1.,.5,.5,.5]+[1.]*4)
    zero,d=conditional_moments(h,p,w,rx,[1,3,4,6,8],RECIPE)
    assert zero==0 and d['valid_cells']==1 and d['receiver_pairs']==0


def test_low_confidence_no_partner_and_constant_descriptors_are_safe():
    h=torch.ones(8,3,requires_grad=True);p=torch.ones(8,1);rx=torch.tensor([1]*4+[3]*4)
    loss,d=conditional_moments(h,p,torch.zeros(8),rx,[1,3,4,6,8],RECIPE)
    assert loss==0 and d['eligible']==0;loss.backward();assert torch.isfinite(h.grad).all()
    value,d=conditional_moments(h,p,torch.ones(8),rx,[1,3,4,6,8],RECIPE)
    assert value==0 and d['receiver_pairs']==1
    with pytest.raises(ValueError,match='source RX'):
        conditional_moments(h,p,torch.ones(8),torch.full((8,),9),[1,3,4,6,8],RECIPE)


def test_class_aggregation_does_not_weight_large_classes_more():
    h=torch.tensor([[-1.],[1.],[-1.],[1.],[3.],[5.],[3.],[5.]]+[[9.],[11.]]*4+[[11.],[13.]]*4,requires_grad=True)
    p=torch.tensor([[1.,0.]]*8+[[0.,1.]]*16)
    rx=torch.tensor([1]*4+[3]*4+[1]*8+[3]*8)
    value,d=conditional_moments(h,p,torch.ones(24),rx,[1,3,4,6,8],RECIPE)
    assert torch.allclose(value,10./h.detach().var(0,unbiased=False).squeeze())
    assert d['receiver_pairs']==2 and d['valid_classes']==2


def setup():
    parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old');c=configuration(parent,'/p','new')
    c['u_cmr_smoke']=True;student=StyleStudent();x=torch.randn(20,2,256)
    rx=torch.tensor([1]*4+[3]*4+[4]*4+[6]*4+[8]*4);domain=torch.arange(20)
    meta=dict(rx_i=rx,physical_id=torch.arange(20)*17+5)
    student._cd_U_loader=[(x,torch.full((20,),-1),domain,meta)];student._cd_U_iter=iter(student._cd_U_loader)
    from experiments.pamf_ssl_research.objective import Objective
    state=Objective(student,parent).state_dict();state.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    obj=UnlabeledReceiverMomentsObjective(student,c);obj.load_state_dict(state)
    obj.current_epoch,obj.current_step=31,6661;obj.labeled=(x,torch.zeros(20,dtype=torch.long),rx)
    return student,obj,c,x,domain,meta


def test_raw_u_is_the_extra_forward_and_no_new_model_parameters_or_teacher_grad():
    student,obj,c,x,domain,meta=setup();before=set(student.state_dict());calls=[]
    original=student.forward_identity_only
    def capture(z,domain_labels):calls.append(z.detach().clone());return original(z,domain_labels)
    student.forward_identity_only=capture
    loss=obj.loss(student,'cpu');loss.backward()
    assert len(calls)==2;torch.testing.assert_close(calls[-1],x)
    assert obj.u_cmr_calls==1 and obj.u_cmr_seen==20
    assert obj.seen==963920 and obj.physical_seen==481960
    assert all(not layer._forward_hooks for layer in student.levels)
    assert not obj._raw_context_active and set(student.state_dict())==before
    assert all(p.grad is None for p in obj.teacher.parameters())
    obj.after_update(student);state=obj.state_dict();restored=UnlabeledReceiverMomentsObjective(student,c)
    restored.load_state_dict(state);assert restored.ema_updates==6661 and restored.u_cmr_seen==20


def test_exposed_u_truth_rejected_before_additional_raw_forward():
    student,obj,c,x,domain,meta=setup()
    student._cd_U_iter=iter([(x,torch.zeros(20,dtype=torch.long),domain,meta)])
    with pytest.raises(ValueError,match='U TX truth exposed'):obj.loss(student,'cpu')
    assert obj.u_cmr_calls==0 and not obj._raw_context_active


def test_failed_raw_forward_cleans_temporary_hooks_and_does_not_count_call():
    student,obj,c,x,domain,meta=setup()
    def failed(*args,**kwargs):raise RuntimeError('intentional raw failure')
    student.forward_identity_only=failed
    with pytest.raises(RuntimeError,match='intentional raw failure'):
        obj.unlabeled_source_loss(student,x,domain,torch.ones(20,6)/6,torch.ones(20),meta['rx_i'])
    assert obj.u_cmr_calls==0 and not obj._raw_context_active
    assert all(not layer._forward_hooks for layer in student.levels)


def test_fixed_recipe_forbids_target_inputs_and_other_new_mechanisms():
    _,_,c,_,_,_=setup();bad=deepcopy(c);bad['u_cmr']['weight']=.2
    with pytest.raises(ValueError,match='U-CMR recipe'):validate(bad)
    bad=deepcopy(c);bad['truth']='/target/truth'
    with pytest.raises(ValueError,match='source handles'):validate(bad)
    bad=deepcopy(c);bad['u_gradprotect']={'weight':1.}
    with pytest.raises(ValueError,match='U-CMR recipe'):validate(bad)
