import torch
from experiments.pamf_dura_direct_train.iq_enhancement import (
    ShapeEnhancer,shape_correction,amplified_reference)


def test_shape_bound_and_gain_phase_orthogonality():
    torch.manual_seed(17)
    base=torch.randn(9,2,256)
    raw=100*torch.randn_like(base)
    delta=shape_correction(raw,base)
    z=torch.complex(base[:,0],base[:,1])
    d=torch.complex(delta[:,0],delta[:,1])
    assert torch.all(delta.square().sum((1,2))<=.0025*base.square().sum((1,2))+1e-5)
    torch.testing.assert_close((z.conj()*d).sum(-1),torch.zeros(9,dtype=torch.complex64),atol=1e-6,rtol=0)
    assert torch.equal(shape_correction(raw,torch.zeros_like(base)),torch.zeros_like(base))


def test_enhancement_gradients_do_not_change_physical_base():
    head=ShapeEnhancer()
    base=torch.randn(2,2,256,requires_grad=True)
    received=torch.randn_like(base,requires_grad=True)
    output=head(received,base)
    assert torch.equal(output,base)
    (output*torch.randn_like(output)).mean().backward()
    assert base.grad is None and received.grad is None
    assert head.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_amplification_preserves_original_head_choices():
    torch.manual_seed(19)
    weight=torch.nn.functional.normalize(torch.randn(6,160),dim=1)
    _,_,vh=torch.linalg.svd(weight-weight.mean(0),full_matrices=False)
    basis=vh[:5].T
    original=torch.nn.functional.normalize(torch.randn(512,160),dim=1)
    target=amplified_reference(original,basis)
    assert torch.equal((original@weight.T).argmax(1),(target@weight.T).argmax(1))
    old=original@weight.T;new=target@weight.T
    assert ((new-new.mean(1,keepdim=True)).norm(dim=1)>
            (old-old.mean(1,keepdim=True)).norm(dim=1)).all()
