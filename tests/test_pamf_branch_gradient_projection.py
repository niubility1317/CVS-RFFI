import pytest
import torch
from experiments.pamf_ssl_research.branch_gradient_projection import branch_projected_gradients,cone_coefficients
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients


def call(l,u,c,a,*,cap=2.,dtype=torch.float64):
    parameter=torch.nn.Parameter(torch.zeros(len(l),dtype=dtype))
    vectors=[(torch.tensor(v,dtype=dtype),) for v in (l,u,c,a)]
    result,detail=branch_projected_gradients([('encoder.t2.weight',parameter)],*vectors,max_u_ratio=cap)
    return result[0],detail['t2']


def test_whole_L_can_allow_a_conflicting_branch():
    result,detail=call([1.,1.],[-1.,2.],[1.,0.],[0.,1.])
    torch.testing.assert_close(result,torch.tensor([1.,3.],dtype=torch.float64))
    assert detail['anchor_cosines_before']['L_full']>0
    assert detail['anchor_cosines_before']['L_clean']<0
    assert detail['constraint_dots_after']['L_clean']==pytest.approx(0.)
    assert detail['constraint_dots_after']['L_As']>=0.


@pytest.mark.parametrize('cap',[1.,2.])
def test_redundant_anchors_reproduce_native_single_halfspace(cap):
    named=[('encoder.t2.weight',torch.nn.Parameter(torch.zeros(3,dtype=torch.float64)))]
    labeled=(torch.tensor([1.,2.,-1.],dtype=torch.float64),)
    unlabeled=(torch.tensor([-3.,2.,4.],dtype=torch.float64),)
    expected,_=projected_gradients(named,labeled,unlabeled,max_u_ratio=cap)
    result,_=branch_projected_gradients(named,labeled,unlabeled,labeled,labeled,max_u_ratio=cap)
    torch.testing.assert_close(result[0],expected[0],rtol=1e-9,atol=1e-10)


def test_two_nonorthogonal_constraints_need_a_joint_solution():
    result,detail=call([0.,1.],[-1.,-1.],[1.,0.],[-1.,1.])
    torch.testing.assert_close(result,torch.tensor([0.,1.],dtype=torch.float64),atol=1e-10,rtol=0.)
    assert max(abs(v) for v in detail['constraint_dots_after'].values())<1e-10


def test_opposite_and_redundant_anchors_keep_only_feasible_subspace():
    result,detail=call([0.,1.],[2.,1.],[1.,0.],[-1.,0.])
    torch.testing.assert_close(result,torch.tensor([0.,2.],dtype=torch.float64))
    assert detail['constraint_dots_after']['L_clean']==pytest.approx(0.)
    assert detail['constraint_dots_after']['L_As']==pytest.approx(0.)


def test_tiny_full_L_still_disables_U():
    result,detail=call([0.,0.],[1.,4.],[1.,0.],[-1.,0.])
    assert torch.count_nonzero(result)==0 and detail['tiny_L']


def test_zero_component_is_omitted_without_replacing_full_L():
    result,detail=call([1.,0.],[-2.,1.],[0.,0.],[1.,0.])
    torch.testing.assert_close(result,torch.tensor([1.,1.],dtype=torch.float64))
    assert detail['near_zero_component_anchors_omitted']==['L_clean']


def test_cap_two_applies_after_cone_projection():
    result,detail=call([1.,1.],[-100.,100.],[1.,0.],[0.,1.],dtype=torch.float32)
    applied=result-torch.tensor([1.,1.])
    assert applied.norm()<=2*torch.tensor([1.,1.]).norm()*(1+1e-6)
    assert detail['capped'] and min(detail['constraint_dots_after'].values())>=0.


def test_unused_parameter_connectivity_stays_none_and_no_parameter_changes():
    named=[('x',torch.nn.Parameter(torch.tensor([3.]))),('unused',torch.nn.Parameter(torch.tensor([4.])))]
    before=[p.detach().clone() for _,p in named]
    result,_=branch_projected_gradients(named,(torch.tensor([1.]),None),(torch.tensor([-1.]),None),
        (torch.tensor([1.]),None),(torch.tensor([0.]),None))
    assert result[1] is None
    assert all(p.grad is None and torch.equal(p,b) for (_,p),b in zip(named,before))


@pytest.mark.parametrize('cap',[0.,-1.,float('nan'),float('inf')])
def test_invalid_caps_are_rejected(cap):
    with pytest.raises(ValueError):call([1.],[1.],[1.],[1.],cap=cap)


def test_nonfinite_gradient_is_rejected():
    with pytest.raises(ValueError):call([1.],[float('nan')],[1.],[1.])


def test_indefinite_gram_is_rejected():
    with pytest.raises(ValueError):cone_coefficients([[1.,2.],[2.,1.]],[-1.,-1.])


def test_active_set_obeys_dual_and_primal_conditions():
    gram=torch.tensor([[1.,.2,-.1],[.2,1.,.3],[-.1,.3,1.]],dtype=torch.float64)
    dots=torch.tensor([-.8,.4,-.2],dtype=torch.float64)
    value=cone_coefficients(gram,dots)
    multipliers=torch.tensor(value['multipliers'],dtype=torch.float64)
    residual=gram@multipliers+dots
    assert min(multipliers)>=-value['KKT_tolerance'] and min(residual)>=-value['KKT_tolerance']
    assert float((multipliers*residual).abs().max())<1e-8


def test_counterfactual_reuses_six_derivatives_and_restores_wrappers(monkeypatch):
    from experiments.pamf_ssl_research import source_branch_gradients as source
    from experiments.pamf_ssl_research.source_branch_anchor_counterfactual import installed_counterfactual
    # The frozen NativeAs release accepts no max_u_ratio keyword.
    def legacy_cap1_only(named,labeled,unlabeled):return projected_gradients(named,labeled,unlabeled)
    monkeypatch.setattr(source,'projected_gradients',legacy_cap1_only)
    p=torch.nn.Parameter(torch.tensor([.3,.8],dtype=torch.float64));named=[('encoder.t2.weight',p)]
    losses=dict(L_clean=p[0],L_As=p[1],L_aux=p.sum()*0.,U_clean=-p[0]+2*p[1],U_As=p.sum()*0.,U_aux=p.sum()*0.)
    gl=torch.autograd.grad(sum(losses[k] for k in ('L_clean','L_As','L_aux')),p,retain_graph=True)
    gu=torch.autograd.grad(sum(losses[k] for k in ('U_clean','U_As','U_aux')),p,retain_graph=True)
    original=source.gradient_facts;grad=torch.autograd.grad;count=0
    def counted(*args,**kwargs):
        nonlocal count
        count+=1;return grad(*args,**kwargs)
    torch.autograd.grad=counted
    try:
        with installed_counterfactual():facts,_=source.gradient_facts(named,losses,reference=(gl,gu))
        assert torch.autograd.grad is counted
    finally:torch.autograd.grad=grad
    assert count==6 and source.gradient_facts is original
    observed=facts['branch_anchor_counterfactual']
    assert observed['extra_autograd_calls']==0 and observed['extra_model_forward_calls']==0
    assert observed['whole_U_norm_retention_vs_native_cap2']>0.
    assert observed['removed_material_component_conflicts'] and p.grad is None


def test_counterfactual_restores_wrapper_after_an_observation_error():
    from experiments.pamf_ssl_research import source_branch_gradients as source
    from experiments.pamf_ssl_research.source_branch_anchor_counterfactual import installed_counterfactual
    original=source.gradient_facts;grad=torch.autograd.grad
    with pytest.raises(ValueError):
        with installed_counterfactual():source.gradient_facts([],{},reference=None)
    assert source.gradient_facts is original and torch.autograd.grad is grad


def test_isolated_cap2_reference_matches_the_actual_native_cap2_math():
    from experiments.pamf_ssl_research.source_cap2_projection_reference import projected_gradients as isolated
    named=[('encoder.t2.weight',torch.nn.Parameter(torch.zeros(3))),('unused',torch.nn.Parameter(torch.zeros(2)))]
    labeled=(torch.tensor([1.,2.,-1.]),None);unlabeled=(torch.tensor([-30.,20.,40.]),None)
    expected,detail=projected_gradients(named,labeled,unlabeled,max_u_ratio=2.)
    actual,observed=isolated(named,labeled,unlabeled,max_u_ratio=2.)
    assert torch.equal(actual[0],expected[0]) and actual[1] is None and observed==detail
