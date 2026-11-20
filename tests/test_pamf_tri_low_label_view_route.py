import torch
from experiments.pamf_repl_source_adapt import view_route as route


def test_scarce_counts_prefer_raw_only_where_mixed_confidence_drops():
    count=torch.tensor([7,35,7,7])
    raw=torch.tensor([.9,.9,.1,.9],requires_grad=True)
    mixed=torch.tensor([.2,.2,.8,.2])
    accepted=torch.tensor([True,True,True,False])
    w=route.coefficients(count,raw,mixed,accepted,'adaptive',1.)
    assert torch.allclose(w,torch.tensor([1/3,1/7,0.,0.]))
    assert not w.requires_grad and bool((w<=.5).all())


def test_fixed_ce_transfer_preserves_mass_and_excludes_rejected_rows():
    m=torch.randn(4,6,requires_grad=True)
    r=torch.randn(4,6,requires_grad=True)
    labels=torch.tensor([0,1,2,3])
    mask=torch.tensor([True,False,True,False])
    delta,w=route.transfer_ce(m,m,r,r,labels,mask,None,'fixed',.25)
    gm,gr=torch.autograd.grad(delta,(m,r))
    assert torch.equal(w,mask*.25)
    assert torch.count_nonzero(gm[~mask]) == torch.count_nonzero(gr[~mask]) == 0
    assert torch.count_nonzero(gm[mask]) and torch.count_nonzero(gr[mask])
