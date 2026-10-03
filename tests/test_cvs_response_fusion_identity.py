import pytest
import torch
from torch.nn import functional as F
from experiments.cvs_response_fusion_identity.model import build,VARIANTS,contrast_response,anchored_features,EPSILON
from experiments.cvs_channel_response_identity.model import build as original_build

torch.set_num_threads(2)

def weight_and_null():
    torch.manual_seed(12);w=torch.randn(6,160,dtype=torch.float64)
    w=F.normalize(w,dim=1);contrast=w-w.mean(0)
    _,s,vh=torch.linalg.svd(contrast,full_matrices=True)
    assert (s>1e-8).sum()==5
    return w,vh[5:]


def test_span_removes_classifier_null_response_and_retains_its_range():
    w,null=weight_and_null();r=torch.randn(4,160,dtype=torch.float64)
    used=contrast_response(r,w)
    assert torch.max(torch.abs(used@null.T))<1e-12
    assert torch.max(torch.abs(contrast_response(null[:4],w)))<1e-12


def test_span_is_class_permutation_equivariant_and_differentiable():
    torch.manual_seed(4);w=torch.randn(6,160,requires_grad=True);r=torch.randn(4,160,requires_grad=True)
    got=contrast_response(r,w)
    assert torch.allclose(got,contrast_response(r,w[[2,0,5,1,4,3]]),atol=2e-5,rtol=2e-5)
    got.square().sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0
    assert torch.isfinite(r.grad).all() and r.grad.norm()>0


def test_anchor_null_response_cannot_change_centered_logits():
    w,null=weight_and_null();base=torch.randn(4,160,dtype=torch.float64);r=10*null[:4]
    a=30*(anchored_features(base,r)@w.T);b=30*(F.normalize(base,dim=1,eps=EPSILON)@w.T)
    assert torch.allclose(a-a.mean(1,keepdim=True),b-b.mean(1,keepdim=True),atol=1e-12,rtol=1e-12)
    assert torch.all((anchored_features(base,r)-F.normalize(base,dim=1)).norm(dim=1)<=1+1e-12)


def test_anchor_zero_features_have_finite_gradients():
    b=torch.zeros(3,160,requires_grad=True);r=torch.zeros_like(b,requires_grad=True)
    anchored_features(b,r).sum().backward()
    assert torch.isfinite(b.grad).all() and torch.isfinite(r.grad).all()


@pytest.mark.parametrize('variant',VARIANTS)
def test_exact_initial_classifier_equal_parameter_count_and_identity_null(variant):
    torch.manual_seed(123);original=original_build('response_mean').eval()
    torch.manual_seed(123);model=build(variant).eval()
    x=torch.randn(6,2,256)
    with torch.no_grad():
        assert torch.equal(original(x),model(x))
        assert torch.count_nonzero(model.fusion_components(x)['used_response'])==0
    assert sum(p.numel() for p in model.parameters())==237147
    assert model.contract()['fusion_new_parameters']==0


@pytest.mark.parametrize('variant',VARIANTS)
def test_single_ce_reaches_new_branch_and_head(variant):
    torch.manual_seed(71);model=build(variant);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    x=torch.randn(6,2,256);labels=torch.arange(6);seen=set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),labels);loss.backward()
        assert torch.isfinite(loss)
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            if p.grad.norm()>0:seen.add(name)
        optimizer.step()
    assert set(dict(model.named_parameters()))==seen
    assert not model.contract()['fusion_labels_used_in_forward']
