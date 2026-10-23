import torch
import pytest

from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients, protected_loss


def named(*names):
    return [(n,torch.nn.Parameter(torch.zeros(2))) for n in names]


def test_conflicting_u_is_projected_then_norm_bounded():
    values,details=projected_gradients(named('backbone.weight'),(torch.tensor([1.,0.]),),(torch.tensor([-1.,2.]),))
    assert torch.allclose(values[0],torch.tensor([1.,1.]))
    assert details['other']['conflict'] and details['other']['capped']
    assert abs(details['other']['dot_after'])<1e-7


def test_aligned_u_keeps_direction_and_cap():
    values,details=projected_gradients(named('net.cls_head.weight'),(torch.tensor([2.,0.]),),(torch.tensor([5.,0.]),))
    assert torch.equal(values[0],torch.tensor([4.,0.]))
    assert not details['cls_head']['conflict']


@pytest.mark.parametrize('left',[torch.zeros(2),torch.tensor([1e-7,0.])])
def test_tiny_l_discards_u(left):
    values,details=projected_gradients(named('net.t2.weight'),(left,),(torch.tensor([5.,2.]),))
    assert torch.equal(values[0],left)
    assert details['t2']['tiny_L']


def test_groups_are_independent_and_unused_parameters_stay_unused():
    names=named('net.t2.weight','net.cls_head.weight','unused')
    values,_=projected_gradients(names,(torch.tensor([1.,0.]),torch.tensor([0.,2.]),None),
                                (torch.tensor([-1.,1.]),torch.tensor([0.,-2.]),None))
    assert torch.equal(values[0],torch.tensor([1.,1.]))
    assert torch.equal(values[1],torch.tensor([0.,2.]))
    assert values[2] is None


def test_native_identity_cancellation_and_upstream_scale_use_prepared_gradients():
    parameter=torch.nn.Parameter(torch.tensor([1.,0.]))
    identity=(parameter.square()).sum()
    supervised=.5*parameter.square().sum()
    unsupervised=-parameter[0]+2*parameter[1]
    guided,_,expected=protected_loss(supervised,unsupervised,[('backbone.weight',parameter)])
    native=identity+(guided-identity)
    assert torch.allclose(native.detach(),(supervised+unsupervised).detach())
    (native*2.5).backward()
    assert torch.allclose(parameter.grad,expected[0]*2.5)
    assert torch.allclose(parameter.grad,torch.tensor([2.5,2.5]))


def test_l_only_parameter_receives_original_gradient():
    p=torch.nn.Parameter(torch.tensor([2.,1.]));u=torch.nn.Parameter(torch.tensor([1.,0.]))
    loss,_,expected=protected_loss(p.square().sum(),u.square().sum(),[('a',p),('net.cls_head.weight',u)])
    loss.backward()
    assert torch.equal(p.grad,torch.tensor([4.,2.]))
    assert torch.equal(u.grad,torch.zeros(2))


def test_nonfinite_gradients_are_rejected():
    with pytest.raises(ValueError,match='finite'):
        projected_gradients(named('a'),(torch.ones(2),),(torch.tensor([float('nan'),0.]),))


def setup_objective():
    from experiments.pamf_ssl_research.plan import configuration as scaffold,SEEDS
    from experiments.pamf_ssl_research.u_gradprotect import configuration
    from experiments.pamf_ssl_research.objective import Objective
    from experiments.pamf_ssl_research.u_gradprotect_objective import GradientProtectedObjective
    from tests.test_pamf_ssl_research import template
    from tests.test_pamf_ssl_paired_u import Student
    parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old');config=configuration(parent,'/p','new')
    student=Student();x=torch.randn(8,2,256);rx=torch.tensor([1,3,4,6,8,1,3,4]);domain=torch.arange(8)
    student._cd_U_loader=[(x,torch.full((8,),-1),domain,dict(rx_i=rx,physical_id=torch.arange(8)*17+5))]
    student._cd_U_iter=iter(student._cd_U_loader)
    inherited=Objective(student,parent).state_dict()
    inherited.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    obj=GradientProtectedObjective(student,config);obj.load_state_dict(inherited)
    obj.current_epoch,obj.current_step=31,6661;obj.labeled=(x,torch.zeros(8,dtype=torch.long),rx)
    return student,obj,config,x,domain


def test_original_hidden_u_state_and_default_network_continue_with_one_ema_update():
    from experiments.pamf_ssl_research.u_gradprotect_objective import GradientProtectedObjective
    student,obj,config,x,domain=setup_objective();before=set(student.state_dict())
    supervised=torch.nn.functional.cross_entropy(student.forward_identity_only(x,domain)['tx_logits'],torch.zeros(8,dtype=torch.long))
    zero=obj.loss(student,'cpu');assert not zero.requires_grad and zero==0
    guided=obj.guide(supervised,student);guided.backward()
    assert torch.isfinite(guided)
    assert all(p.grad is None for p in obj.teacher.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert obj.seen==963908 and obj.physical_seen==481954 and obj.u_gradprotect_calls==1
    assert obj.pending_gradient_loss is None and set(student.state_dict())==before
    obj.after_update(student);saved=obj.state_dict();restored=GradientProtectedObjective(student,config)
    restored.load_state_dict(saved)
    assert restored.ema_updates==6661 and restored.u_gradprotect_calls==1
    assert restored.u_gradprotect_conflicts==obj.u_gradprotect_conflicts


def test_hidden_u_truth_and_missing_current_u_graph_are_rejected():
    student,obj,config,x,domain=setup_objective()
    student._cd_U_iter=iter([(x,torch.zeros(8,dtype=torch.long),domain,dict(rx_i=torch.ones(8,dtype=torch.long)))])
    with pytest.raises(ValueError,match='U TX truth exposed'):obj.loss(student,'cpu')
    assert obj.u_gradprotect_calls==0 and obj.pending_gradient_loss is None
    with pytest.raises(ValueError,match='Current hidden-label'):obj.guide(torch.tensor(1.),student)


def test_recipe_or_target_input_changes_are_rejected():
    from copy import deepcopy
    from experiments.pamf_ssl_research.u_gradprotect import validate
    _,_,config,_,_=setup_objective()
    bad=deepcopy(config);bad['u_gradprotect']['norm_cap_U_to_L']=2.
    with pytest.raises(ValueError,match='gradient protection'):validate(bad)
    bad=deepcopy(config);bad['truth']='/target/truth'
    with pytest.raises(ValueError,match='source handles'):validate(bad)
