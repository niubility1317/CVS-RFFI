import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi.cross_response.gradient_audit import audit_decision_parameter_gradients


def test_parameter_budgets_common_reachability_zero_none_and_no_mutation():
    p=torch.nn.Parameter(torch.tensor([[.1,.2],[.3,.4]]))
    q=torch.nn.Parameter(torch.tensor([[1.,-.2],[-.5,.4]]))
    unused=torch.nn.Parameter(torch.ones(2))
    zero=torch.nn.Parameter(torch.ones(2))
    x=torch.tensor([[1.,2.],[3.,1.]])
    logits=x@p@q+zero.sum()*0
    labels=torch.tensor([0,1])
    refs=torch.full_like(logits,10)
    valid=torch.tensor([[False,True],[True,False]])
    identity_loss=p.square().sum()+zero.sum()*0
    response_loss=p.sum()+zero.sum()*0
    for param in (p,q,unused,zero):
        param.grad=torch.full_like(param,7)
    result=audit_decision_parameter_gradients(logits,labels,refs,valid,
        [('p',p),('unused',unused),('zero',zero)],named_classifier_parameters=[('q',q)],
        compared_losses={'identity':identity_loss,'response':response_loss},decision_weight=.3)
    assert result['common_identity_parameters']==['p','zero']
    assert result['total']['identity_all']['none_mask']['unused']
    assert result['total']['identity_common']['zero_mask']['zero']
    margin=logits[:,0]-logits[:,1]
    expected=.3*((10-margin[0]).square()+(10+margin[1]).square())/2
    direct=torch.autograd.grad(expected,p,retain_graph=True)[0].norm().item()
    assert result['total']['identity_common']['l2']==pytest.approx(direct)
    assert sum(r['weighted_loss'] for r in result['by_pair'].values())==pytest.approx(float(expected.detach()))
    for param in (p,q,unused,zero):
        assert torch.all(param.grad==7)


def test_overlapping_classifier_identity_rejected():
    p=torch.nn.Parameter(torch.ones(2,2))
    with pytest.raises(ValueError,match='disjoint'):
        audit_decision_parameter_gradients(p,[0,1],p,torch.ones(2,2,dtype=torch.bool),
            [('identity',p)],named_classifier_parameters=[('classifier',p)])
