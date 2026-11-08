import pytest
import torch
from torch.nn import functional as F
from experiments.phase1_dura_orthogonal.polyphase_conv import polyphase_conv1d


@pytest.mark.parametrize('dilation',[1,2,4,8])
def test_polyphase_matches_dilated_value_and_gradients(dilation):
    torch.manual_seed(41)
    x=torch.randn(2,3,32,dtype=torch.float64,requires_grad=True)
    w=torch.randn(4,3,5,dtype=torch.float64,requires_grad=True)
    b=torch.randn(4,dtype=torch.float64,requires_grad=True)
    grad=torch.randn(2,4,32,dtype=torch.float64)
    baseline=F.conv1d(x,w,b,padding=2*dilation,dilation=dilation)
    candidate=polyphase_conv1d(x,w,b,dilation)
    ga=torch.autograd.grad(baseline,(x,w,b),grad,retain_graph=True)
    gb=torch.autograd.grad(candidate,(x,w,b),grad)
    torch.testing.assert_close(candidate,baseline,rtol=1e-12,atol=1e-12)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_rejects_unsupported_length():
    with pytest.raises(ValueError):polyphase_conv1d(torch.zeros(2,3,31),torch.zeros(4,3,5),None,4)
