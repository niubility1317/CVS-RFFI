import math
import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_ssl_research.u_viewbalance_objective import balanced_gradients,balanced_loss


def run(full,raw,ordinary,physical):
    tensors=[torch.tensor(v,dtype=torch.float32) for v in (full,raw,ordinary,physical)]
    named=[('net.t2.weight',torch.nn.Parameter(torch.zeros_like(tensors[0])))]
    value,detail=balanced_gradients(named,*(tuple([v]) for v in tensors))
    return value[0],detail['t2']


def test_both_views_are_normalized_without_original_norm_domination():
    value,d=run([2,0,0],[0,2,0],[.001,.001,0],[0,20,20])
    expected=torch.tensor([2+1/math.sqrt(2),math.sqrt(2),1/math.sqrt(2)])
    assert torch.allclose(value,expected,atol=1e-6)
    assert d['ordinary']['scale']>1 and d['physical']['scale']<1
    assert abs(d['ordinary']['applied_U_norm']-1)<1e-6 and abs(d['physical']['applied_U_norm']-1)<1e-6


def test_view_conflicts_are_corrected_before_they_cancel_in_a_sum():
    value,d=run([2,0],[0,2],[1,-1],[-1,1])
    assert torch.allclose(value,torch.tensor([3.,1.]))
    assert d['ordinary']['active_set']=='raw' and d['physical']['active_set']=='full'
    assert all(v['full_dot_after']>=-1e-7 and v['raw_dot_after']>=-1e-7 for v in (d['ordinary'],d['physical']))


def test_zero_projection_does_not_donate_budget_to_other_view():
    value,d=run([2,0],[0,2],[-1,-1],[1,1])
    assert torch.allclose(value,torch.tensor([2+1/math.sqrt(2),1/math.sqrt(2)]),atol=1e-6)
    assert d['ordinary']['tiny_projected'] and d['ordinary']['applied_U_norm']==0
    assert abs(d['physical']['applied_U_norm']-1)<1e-6


@pytest.mark.parametrize('raw',[[2,0],[-2,0]])
def test_parallel_or_opposite_anchors_preserve_valid_tangent(raw):
    value,d=run([2,0],raw,[-1,2],[-3,7])
    assert torch.allclose(value,torch.tensor([2.,2.]),atol=1e-6)
    assert all(not v['numerical_fallback'] for v in (d['ordinary'],d['physical']))


@pytest.mark.parametrize('full,raw',[([0,0],[1,0]),([1,0],[0,0]),([1e-7,0],[1,0])])
def test_tiny_anchor_drops_both_views_preserving_l(full,raw):
    value,d=run(full,raw,[3,2],[5,9])
    assert torch.equal(value,torch.tensor(full,dtype=torch.float32)) and d['tiny_anchor']
    assert d['ordinary']['applied_U_norm']==d['physical']['applied_U_norm']==0


def test_per_group_norm_and_halfspace_constraints_on_fixed_random_vectors():
    generator=torch.Generator().manual_seed(20261023)
    for _ in range(8):
        full=torch.randn(9,generator=generator);raw=torch.randn(9,generator=generator)
        ordinary=torch.randn(9,generator=generator)*.001;physical=torch.randn(9,generator=generator)*10
        named=[('backbone.weight',torch.nn.Parameter(torch.zeros(9)))]
        values,d=balanced_gradients(named,(full,),(raw,),(ordinary,),(physical,));details=d['other']
        applied=values[0]-full
        assert applied.norm()<=full.norm()*(1+1e-6)
        assert full.dot(applied)>=-1e-6 and raw.dot(applied)>=-1e-6
        for view in ('ordinary','physical'):
            v=details[view]
            assert v['applied_U_norm']<=.5*details['full_L_norm']*(1+1e-6)+1e-8


def test_nonfinite_view_gradient_rejected():
    with pytest.raises(ValueError,match='finite'):run([1,0],[0,1],[math.nan,0],[1,1])


def test_native_identity_cancellation_and_upstream_scaling():
    p=torch.nn.Parameter(torch.ones(2));identity=p.square().sum();full=2*p[0];raw=2*p[1]
    ordinary=p[0]-p[1];physical=-p[0]+p[1]
    guided,d,expected=balanced_loss(full,raw,ordinary,physical,[('a',p)])
    native=identity+(guided-identity)
    assert torch.allclose(native.detach(),(full+ordinary+physical).detach())
    (2.5*native).backward()
    assert torch.allclose(p.grad,torch.tensor([7.5,2.5])) and torch.allclose(p.grad,expected[0]*2.5)


def setup_objective():
    from experiments.pamf_ssl_research.plan import configuration as scaffold,SEEDS
    from experiments.pamf_ssl_research.u_viewbalance import configuration
    from experiments.pamf_ssl_research.objective import Objective
    from experiments.pamf_ssl_research.u_viewbalance_objective import ViewBalancedObjective
    from tests.test_pamf_ssl_research import template
    from tests.test_pamf_ssl_paired_u import Student
    parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old');config=configuration(parent,'/p','new')
    student=Student();x=torch.randn(8,2,256);rx=torch.tensor([1,3,4,6,8,1,3,4]);domain=torch.arange(8)
    student._cd_U_loader=[(x,torch.full((8,),-1),domain,dict(rx_i=rx,physical_id=torch.arange(8)*17+5))]
    student._cd_U_iter=iter(student._cd_U_loader)
    state=Objective(student,parent).state_dict();state.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    objective=ViewBalancedObjective(student,config);objective.load_state_dict(state)
    objective.current_epoch,objective.current_step=31,6661;objective.labeled=(x,torch.zeros(8,dtype=torch.long),rx)
    return student,objective,config,x,domain


def test_actual_original_u_contributions_sum_exactly_without_extra_forward():
    student,obj,config,x,domain=setup_objective();before=set(student.state_dict());calls=[]
    original=student.forward_identity_only
    def forward(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    student.forward_identity_only=forward
    raw=F.cross_entropy(student.forward_identity_only(x,domain)['tx_logits'],torch.zeros(8,dtype=torch.long))
    obj.observe_raw(raw);zero=obj.loss(student,'cpu')
    assert len(calls)==2 and not zero.requires_grad and zero==0
    ordinary,physical=obj.pending_views
    assert torch.allclose(ordinary+physical,obj.pending_total_U)
    assert obj.pending_view_counts=={'ordinary':4,'physical':4}
    guided=obj.guide(raw,student);guided.backward()
    assert torch.isfinite(guided) and obj.u_viewbalance_calls==1 and obj.u_viewbalance_raw_calls==1
    assert obj.pending_views is None and obj.pending_raw_anchor is None and obj.pending_total_U is None
    assert obj.seen==963908 and obj.physical_seen==481954 and set(student.state_dict())==before
    assert all(p.grad is None for p in obj.teacher.parameters())
    obj.after_update(student);state=obj.state_dict()
    from experiments.pamf_ssl_research.u_viewbalance_objective import ViewBalancedObjective
    restored=ViewBalancedObjective(student,config);restored.load_state_dict(state)
    assert restored.ema_updates==6661 and restored.u_viewbalance_calls==1
    assert restored.u_viewbalance_view_seen=={'ordinary':4,'physical':4}


def test_hidden_u_truth_and_missing_raw_anchor_rejected():
    student,obj,config,x,domain=setup_objective()
    with pytest.raises(ValueError,match='raw'):obj.guide(torch.tensor(1.),student)
    student._cd_U_iter=iter([(x,torch.zeros(8,dtype=torch.long),domain,dict(rx_i=torch.ones(8,dtype=torch.long)))])
    with pytest.raises(ValueError,match='U TX truth exposed'):obj.loss(student,'cpu')
    assert obj.u_viewbalance_calls==0


def test_full_u_denominator_and_unequal_partition_are_preserved():
    _,obj,_,_,_=setup_objective()
    per=torch.tensor([1.,2.,3.,4.,5.],requires_grad=True);mask=torch.tensor([True,False,True,False,True])
    obj.unlabeled_view_contributions(per,mask,.5)
    ordinary,physical=obj.pending_views
    assert torch.allclose(ordinary,torch.tensor(.6)) and torch.allclose(physical,torch.tensor(.9))
    assert obj.pending_view_counts=={'ordinary':2,'physical':3}
    assert torch.allclose(torch.autograd.grad(ordinary+physical,per)[0],torch.full((5,),.1))


def test_recipe_and_foreign_mechanisms_rejected():
    from copy import deepcopy
    from experiments.pamf_ssl_research.u_viewbalance import validate
    _,_,config,_,_=setup_objective();bad=deepcopy(config);bad['u_viewbalance']['nonzero_view_norm_to_full_L']=.7
    with pytest.raises(ValueError,match='view balance'):validate(bad)
    bad=deepcopy(config);bad['u_dualanchor']={'start_epoch':31}
    with pytest.raises(ValueError,match='view balance'):validate(bad)
