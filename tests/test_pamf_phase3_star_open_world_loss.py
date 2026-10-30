import importlib
import math
import numpy as np
import pytest
import torch


def module():
    try:return importlib.import_module('experiments.pamf_phase3_open_world.star_open_world_loss')
    except ModuleNotFoundError:pytest.fail('open-world membership loss is not implemented')


def test_numpy_slot_metadata_never_uses_numpy_torch_bridge(monkeypatch):
    m=module();original=torch.as_tensor
    def no_array_bridge(value,*args,**kwargs):
        if isinstance(value,np.ndarray):raise RuntimeError('historical Torch cannot infer NumPy2 int64 dtype')
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'as_tensor',no_array_bridge)
    support=torch.eye(6,64)[:,None,:].repeat(1,16,1).requires_grad_()
    p,labels=m.prototypes(support,np.asarray([3,0,5,1,4,2],dtype=np.int64),3,'M')
    assert labels.tolist()==[0,5,1,4,2]
    p.square().sum().backward();assert support.grad is not None
    with pytest.raises(ValueError):m.prototypes(support,np.arange(6,dtype=np.float64),0,'M')


def test_prototype_slots_preserve_graph_and_loss_is_column_permutation_invariant():
    m=module();z=torch.eye(6,64,dtype=torch.float64)[:,None,:].repeat(1,16,1).requires_grad_()
    order=np.asarray([3,0,5,1,4,2]);p,labels=m.prototypes(z,order,3,'M')
    assert labels.tolist()==[0,5,1,4,2] and p.shape==(5,64)
    torch.testing.assert_close(p.norm(dim=1),torch.ones(5,dtype=torch.float64))
    query=torch.eye(6,64,dtype=torch.float64)
    l=query@p.T;y=torch.arange(6)
    a=m.member_loss(l,y,labels)
    perm=torch.tensor([4,2,0,3,1]);b=m.member_loss(l[:,perm],y,labels[perm])
    torch.testing.assert_close(a['member_loss'],b['member_loss'])
    assert a['counts']['set_positive']==5 and a['counts']['set_negative']==1
    a['member_loss'].backward();assert z.grad is not None and torch.isfinite(z.grad).all()
    _,all_labels=m.prototypes(z,order,3,'S0');assert all_labels.tolist()==order.tolist()


def test_balanced_pair_and_set_bce_handle_omitted_class_and_oe():
    m=module();logits=torch.zeros((3,2),dtype=torch.float64,requires_grad=True)
    result=m.member_loss(logits,torch.tensor([0,1,-1]),torch.tensor([1,2]))
    assert result['counts']==dict(pair_positive=1,pair_negative=5,set_positive=1,set_negative=2,query_count=3,prototype_count=2)
    assert float(result['pair_bce'].detach())==pytest.approx(math.log(2))
    assert float(result['set_bce'].detach())==pytest.approx(math.log(2))
    assert float(result['member_loss'].detach())==pytest.approx(math.log(2))
    result['member_loss'].backward()
    assert logits.grad[0].min()>0 and logits.grad[2].min()>0 and logits.grad[1,0]<0


def test_degenerate_member_targets_and_nonfinite_inputs_are_errors():
    m=module()
    for y in (torch.tensor([0]),torch.tensor([-1])):
        with pytest.raises(ValueError):m.member_loss(torch.zeros((1,1)),y,torch.tensor([0]))
    with pytest.raises(ValueError):m.member_loss(torch.tensor([[float('nan')],[0.]]),torch.tensor([0,-1]),torch.tensor([0]))
    with pytest.raises(ValueError):m.prototypes(torch.zeros((6,16,64)),np.arange(6),0,'M')


def test_kd_is_teacher_to_student_t2_and_does_not_update_teacher():
    m=module();student=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
    teacher=torch.tensor([[4.,0.]],dtype=torch.float64,requires_grad=True)
    value=m.kd_loss(student,teacher);prob=1/(1+math.exp(-2))
    expected=4*(prob*math.log(2*prob)+(1-prob)*math.log(2*(1-prob)))
    assert float(value.detach())==pytest.approx(expected)
    value.backward();assert teacher.grad is None and student.grad[0,0]<0 and student.grad[0,1]>0
    assert float(m.kd_loss(teacher,teacher).detach())==pytest.approx(0,abs=1e-14)


def test_projection_conflict_cap_auxiliary_and_unused_parameters():
    m=module();shared=torch.nn.Parameter(torch.zeros(2));aux=torch.nn.Parameter(torch.zeros(1));unused=torch.nn.Parameter(torch.zeros(1))
    anchor=shared[0];member=-2*shared[0]+4*shared[1]+3*aux[0]
    grads,d=m.protected_gradients(anchor,member,[('backbone',shared),('open_world.relation.weight',aux),('domain',unused)])
    torch.testing.assert_close(grads[0],torch.tensor([1.,1.]))
    torch.testing.assert_close(grads[1],torch.tensor([3.]));assert grads[2] is None
    assert d['conflict_projected'] and d['cap_scale']==pytest.approx(.25)
    assert d['shared_dot_after']>=-1e-7 and d['shared_member_norm_after']<=d['shared_anchor_norm']+1e-7
    assert d['unused_parameters']==['domain']


def test_zero_anchor_stops_shared_member_but_not_auxiliary():
    m=module();shared=torch.nn.Parameter(torch.tensor([1.]));aux=torch.nn.Parameter(torch.tensor([1.]))
    grads,d=m.protected_gradients((shared*0).sum(),(shared*5+aux*2).sum(),[('backbone',shared),('open_world.relation.weight',aux)])
    torch.testing.assert_close(grads[0],torch.zeros(1));torch.testing.assert_close(grads[1],torch.tensor([2.]))
    assert d['anchor_zero']


def test_s0_detach_preserves_anchor_encoder_gradient_and_nonfinite_raises():
    m=module();shared=torch.nn.Parameter(torch.tensor([2.]));aux=torch.nn.Parameter(torch.tensor([3.]))
    grads,d=m.protected_gradients((shared**2).sum(),(shared.detach()*aux).sum(),[('encoder',shared),('open_world.relation.weight',aux)])
    torch.testing.assert_close(grads[0],torch.tensor([4.]));torch.testing.assert_close(grads[1],torch.tensor([2.]))
    assert 'encoder' in d['member_unused']
    with pytest.raises(ValueError):m.protected_gradients(shared.sum(),(shared*float('inf')).sum(),[('encoder',shared)])
