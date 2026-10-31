import pytest
import torch

from experiments.pamf_identity_flow.f0_probe import audit_gradients


def test_flow_gradient_cannot_mask_missing_identity_gradient():
    w=torch.nn.Parameter(torch.ones(1))
    a=w*torch.ones(2,2,4)
    b=w*torch.ones(2,2,4)*2
    zero_a=a.mean((1,2))*0
    zero_b=b.mean((1,2))*0
    flow=w.square().expand(2)
    with pytest.raises(ValueError,match='F0 guidance'):
        audit_gradients(zero_a,zero_b,a,b,flow,w)


def test_both_identity_and_flow_gradients_are_reported():
    w=torch.nn.Parameter(torch.ones(1))
    a=w*torch.ones(2,2,4)
    b=w*torch.ones(2,2,4)*2
    anchor=a.square().mean((1,2))
    agreement=b.square().mean((1,2))
    gradients,identity,flow=audit_gradients(anchor,agreement,a,b,w.square().expand(2),w)
    assert len(gradients)==2 and identity.abs().sum()>0 and flow.abs().sum()>0
