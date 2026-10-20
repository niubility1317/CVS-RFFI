import pytest
import torch
from experiments.pamf_r46_gradient_coordination.gradients import combine_gradients,assign_gradients,parameter_groups


def fixture(a,b):
    p=[torch.nn.Parameter(torch.zeros_like(a))]
    return p,[a],[b],dict(coarse=[0])


def test_symmetric_projection_hand_calculation_and_common_descent():
    a=torch.tensor([1.,0.],dtype=torch.float64);b=torch.tensor([-1.,1.],dtype=torch.float64)
    p,x,y,groups=fixture(a,b)
    got,stats=combine_gradients(x,y,p,groups,'symmetric_module_projection')
    torch.testing.assert_close(got[0],torch.tensor([.5,1.5],dtype=torch.float64))
    assert torch.dot(a,got[0])==pytest.approx(.5) and torch.dot(b,got[0])==pytest.approx(1.)
    assert stats['coarse']['conflict']==stats['coarse']['projected']==1
    raw,_=combine_gradients(x,y,p,groups,'sum');torch.testing.assert_close(raw[0],a+b)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_random_conflicts_have_nonnegative_applied_dot_for_both_objectives(dtype):
    generator=torch.Generator().manual_seed(92)
    for _ in range(32):
        a=torch.randn(23,generator=generator,dtype=dtype)
        b=-.7*a+.3*torch.randn(23,generator=generator,dtype=dtype)
        p,x,y,groups=fixture(a,b);got,stats=combine_gradients(x,y,p,groups,'symmetric_module_projection')
        assert torch.isfinite(got[0]).all()
        assert float(torch.dot(a.double(),got[0].double()))>=-1e-6
        assert float(torch.dot(b.double(),got[0].double()))>=-1e-6


@pytest.mark.parametrize('kind',['aligned','opposed','zero','tiny'])
def test_aligned_opposed_zero_and_below_floor_gradients(kind):
    a=torch.tensor([1.,2.]);b=a.clone()
    if kind=='opposed':b=-a
    elif kind=='zero':b=b*0
    elif kind=='tiny':b=-b*1e-15
    p,x,y,groups=fixture(a,b);got,stats=combine_gradients(x,y,p,groups,'symmetric_module_projection')
    assert torch.isfinite(got[0]).all()
    if kind=='opposed':torch.testing.assert_close(got[0],torch.zeros_like(a))
    else:torch.testing.assert_close(got[0],a+b)
    assert stats['coarse']['projected']==int(kind=='opposed')


def test_projection_uses_original_gradients_and_separate_module_groups():
    p=[torch.nn.Parameter(torch.zeros(2)),torch.nn.Parameter(torch.zeros(2))]
    a=[torch.tensor([1.,0.]),torch.tensor([1.,0.])]
    b=[torch.tensor([-1.,1.]),torch.tensor([2.,0.])]
    got,stats=combine_gradients(a,b,p,dict(coarse=[0],prior=[1]),'symmetric_module_projection')
    torch.testing.assert_close(got[0],torch.tensor([.5,1.5]));torch.testing.assert_close(got[1],torch.tensor([3.,0.]))
    assert stats['coarse']['projected']==1 and stats['prior']['projected']==0


@pytest.mark.parametrize('groups',[dict(coarse=[0],action=[0]),dict(coarse=[]),dict(coarse=[1])])
def test_duplicate_missing_or_wrong_group_indices_rejected(groups):
    p,x,y,_=fixture(torch.ones(2),torch.ones(2))
    with pytest.raises(ValueError):combine_gradients(x,y,p,groups,'sum')


def test_unused_parameter_stays_none_and_adamw_skips_it():
    used=torch.nn.Parameter(torch.tensor([1.,2.]));unused=torch.nn.Parameter(torch.tensor([3.,4.]))
    before=unused.detach().clone();params=[used,unused]
    combined,_=combine_gradients([torch.ones(2),None],[torch.ones(2),None],params,dict(coarse=[0,1]),'sum')
    assert combined[1] is None
    for p,g in zip(params,combined):p.grad=g
    torch.optim.AdamW(params,lr=.1,weight_decay=.1).step()
    assert torch.equal(unused,before)


def test_raw_two_gradients_match_joint_backward_including_unused_parameters():
    p=[torch.nn.Parameter(torch.tensor([.2,.3])),torch.nn.Parameter(torch.tensor([.4]))]
    base=(p[0]**2).sum();moment=(p[0]**3).sum()
    a=torch.autograd.grad(base,p,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(moment,p,retain_graph=True,allow_unused=True)
    expected=torch.autograd.grad(base+moment,p,allow_unused=True)
    got,_=combine_gradients(a,b,p,dict(coarse=[0,1]),'sum')
    torch.testing.assert_close(got[0],expected[0]);assert got[1] is expected[1] is None


def test_nonfinite_gradient_and_unknown_policy_rejected():
    p,a,b,groups=fixture(torch.tensor([float('nan'),1.]),torch.ones(2))
    with pytest.raises(FloatingPointError):combine_gradients(a,b,p,groups,'sum')
    with pytest.raises(ValueError):combine_gradients([torch.ones(2)],b,p,groups,'unknown')
