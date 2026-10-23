import math
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_ssl_research.u_dualanchor_objective import dual_projected_gradients, dual_protected_loss


def case(full,raw,unlabeled,name='backbone.weight'):
    values=[torch.tensor(v,dtype=torch.float32) for v in (full,raw,unlabeled)]
    named=[(name,torch.nn.Parameter(torch.zeros_like(values[0])))]
    combined,details=dual_projected_gradients(named,(values[0],),(values[1],),(values[2],))
    return combined[0],details['other' if name=='backbone.weight' else 't2']


def test_orthogonal_two_boundaries_preserve_unconstrained_direction():
    value,d=case([1,0,0],[0,1,0],[-1,-2,.5])
    assert torch.allclose(value,torch.tensor([1.,0.,.5]))
    assert d['active_set']=='both' and d['full_dot_after']>=-1e-7 and d['raw_dot_after']>=-1e-7


def test_two_halfspaces_fix_sequential_projection_violation():
    value,d=case([1,0],[-1,1],[1,-2])
    assert torch.allclose(value,torch.tensor([1.,0.]),atol=1e-7)
    assert d['active_set']=='both'


@pytest.mark.parametrize('raw',[[1,0],[-1,0]])
def test_parallel_or_antiparallel_anchors_keep_feasible_tangent(raw):
    value,d=case([1,0],raw,[-1,2])
    assert torch.allclose(value,torch.tensor([1.,1.]),atol=1e-6)
    assert d['capped'] and not d['numerical_fallback']


def test_single_active_full_boundary_and_norm_cap():
    value,d=case([1,0],[1,1],[-1,2])
    assert torch.allclose(value,torch.tensor([1.,1.]))
    assert d['active_set']=='full' and d['applied_U_norm']<=d['full_L_norm']*(1+1e-6)


def test_aligned_u_is_retained_without_projection():
    value,d=case([2,0],[1,1],[.2,.1])
    assert torch.allclose(value,torch.tensor([2.2,.1]))
    assert d['active_set']=='none' and not d['capped']


@pytest.mark.parametrize('full,raw',[([0,0],[1,0]),([1,0],[0,0]),([1,0],[1e-7,0])])
def test_tiny_either_anchor_drops_only_u(full,raw):
    value,d=case(full,raw,[3,2])
    assert torch.equal(value,torch.tensor(full,dtype=torch.float32)) and d['tiny_anchor']


def test_nonorthogonal_geometry_matches_known_intersection():
    value,d=case([1,0,0],[-.5,1,0],[-1,-2,.5])
    assert torch.allclose(value,torch.tensor([1.,0.,.5]),atol=1e-6)
    assert d['active_set']=='both'


def test_missing_gradient_groups_remain_separate():
    p=torch.nn.Parameter(torch.zeros(2));names=[('net.t2.weight',p),('net.cls_head.weight',p),('unused',p)]
    full=(torch.tensor([1.,0.]),torch.tensor([0.,2.]),None)
    raw=(torch.tensor([0.,1.]),torch.tensor([1.,0.]),None)
    u=(torch.tensor([-1.,.5]),None,None)
    combined,d=dual_projected_gradients(names,full,raw,u)
    assert torch.equal(combined[0],torch.tensor([1.,.5]))
    assert torch.equal(combined[1],full[1]) and combined[2] is None


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError,match='finite'):
        case([1,0],[0,1],[math.nan,1])


def test_native_identity_cancellation_and_upstream_scale_use_dual_gradients():
    p=torch.nn.Parameter(torch.tensor([1.,1.,1.]))
    full=p[0];raw=p[1];u=-p[0]-2*p[1]+.5*p[2];identity=p.square().sum()
    guided,d,expected=dual_protected_loss(full,raw,u,[('a',p)])
    native=identity+(guided-identity)
    assert torch.allclose(native.detach(),(full+u).detach())
    (2.5*native).backward()
    assert torch.allclose(p.grad,torch.tensor([2.5,0.,1.25]))
    assert torch.allclose(p.grad,2.5*expected[0])


def test_actual_t1_observer_receives_existing_raw_pair_ce_not_as_or_rdrop():
    from experiments.cric_dura_comparison.t1_model import hook
    x=torch.randn(128,2,256);y=torch.arange(128)%6;domain=torch.zeros(128,dtype=torch.long)
    logits=[torch.nn.Parameter(torch.randn(128,6)+i*.1) for i in range(6)];remaining=iter(logits[1:]);observed=[]
    model=SimpleNamespace(training=True,_feature_l_steps=1,_cd_config=dict(ablation='F16',step_offset=6660,
        num_classes=6,satellite_concat=False,regime='native_cric',source_rxs=[1,3,4,6,8]),
        _cd_pending=dict(x=x,first={},second={}),_cd_handles=[],_cd_extra_L=0,_cd_previous_extra=0,
        _cd_As_seen=0,_cd_sat_seen=0,_cd_PL_seen=0,_cd_U_seen=0,_cd_U_accepted=0,_cd_calls=0,
        _cd_practical=lambda raw,meta,epoch,step:raw,_pamf_ssl_raw_L_observer=observed.append,
        forward_identity_only=lambda *args,**kwargs:dict(tx_logits=next(remaining)))
    identity=F.cross_entropy(logits[0],y,label_smoothing=.01)
    correction,metrics=hook(model,model=model,labeled_output=dict(tx_logits=logits[0]),labels=y,
        extra=(None,dict(rx_i=torch.ones(128,dtype=torch.long))),domains=domain,optimizer_step=1,
        identity_loss=identity,labeled_count=128,labeled_inputs=x)
    expected=.5*(F.cross_entropy(logits[4],y,label_smoothing=.01)+F.cross_entropy(logits[5],y,label_smoothing=.01))
    assert len(observed)==1 and torch.equal(observed[0],expected) and metrics['T1_L_packets']==768
    gradients=torch.autograd.grad(observed[0],logits,allow_unused=True)
    assert all(g is None for g in gradients[:4]) and all(g is not None for g in gradients[4:])


def setup_objective():
    from experiments.pamf_ssl_research.plan import configuration as scaffold,SEEDS
    from experiments.pamf_ssl_research.u_dualanchor import configuration
    from experiments.pamf_ssl_research.objective import Objective
    from experiments.pamf_ssl_research.u_dualanchor_objective import DualAnchorObjective
    from tests.test_pamf_ssl_research import template
    from tests.test_pamf_ssl_paired_u import Student
    parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old');config=configuration(parent,'/p','new')
    student=Student();x=torch.randn(8,2,256);rx=torch.tensor([1,3,4,6,8,1,3,4]);domain=torch.arange(8)
    student._cd_U_loader=[(x,torch.full((8,),-1),domain,dict(rx_i=rx,physical_id=torch.arange(8)*17+5))]
    student._cd_U_iter=iter(student._cd_U_loader);inherited=Objective(student,parent).state_dict()
    inherited.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    obj=DualAnchorObjective(student,config);obj.load_state_dict(inherited)
    obj.current_epoch,obj.current_step=31,6661;obj.labeled=(x,torch.zeros(8,dtype=torch.long),rx)
    return student,obj,config,x,domain


def test_hidden_u_original_state_no_extra_parameters_and_one_ema_update():
    from experiments.pamf_ssl_research.u_dualanchor_objective import DualAnchorObjective
    student,obj,config,x,domain=setup_objective();before=set(student.state_dict())
    raw=F.cross_entropy(student.forward_identity_only(x,domain)['tx_logits'],torch.zeros(8,dtype=torch.long))
    obj.observe_raw(raw);zero=obj.loss(student,'cpu');assert not zero.requires_grad and zero==0
    guided=obj.guide(raw,student);guided.backward()
    assert torch.isfinite(guided) and all(p.grad is None for p in obj.teacher.parameters())
    assert obj.seen==963908 and obj.physical_seen==481954 and obj.u_dualanchor_calls==1
    assert obj.pending_gradient_loss is None and obj.pending_raw_anchor is None and set(student.state_dict())==before
    obj.after_update(student);saved=obj.state_dict();restored=DualAnchorObjective(student,config);restored.load_state_dict(saved)
    assert restored.ema_updates==6661 and restored.u_dualanchor_calls==1
    assert restored.u_dualanchor_active_sets==obj.u_dualanchor_active_sets


def test_raw_anchor_and_hidden_u_permission_guards():
    student,obj,config,x,domain=setup_objective()
    with pytest.raises(ValueError,match='raw'):obj.guide(torch.tensor(1.),student)
    with pytest.raises(ValueError,match='raw'):obj.observe_raw(torch.tensor(1.))
    raw=student.forward_identity_only(x,domain)['tx_logits'].sum();obj.observe_raw(raw)
    with pytest.raises(ValueError,match='raw'):obj.observe_raw(raw)
    student._cd_U_iter=iter([(x,torch.zeros(8,dtype=torch.long),domain,dict(rx_i=torch.ones(8,dtype=torch.long)))])
    with pytest.raises(ValueError,match='U TX truth exposed'):obj.loss(student,'cpu')
    assert obj.u_dualanchor_calls==0


def test_recipe_target_and_foreign_mechanisms_rejected():
    from copy import deepcopy
    from experiments.pamf_ssl_research.u_dualanchor import validate
    _,_,config,_,_=setup_objective()
    bad=deepcopy(config);bad['u_dualanchor']['norm_cap_U_to_original_full_L']=2
    with pytest.raises(ValueError,match='dual'):validate(bad)
    bad=deepcopy(config);bad['truth']='/target/truth'
    with pytest.raises(ValueError,match='source handles'):validate(bad)
    bad=deepcopy(config);bad['u_cmr']={'weight':.1}
    with pytest.raises(ValueError,match='dual'):validate(bad)
