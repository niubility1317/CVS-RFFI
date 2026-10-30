import pytest
import torch
from torch.nn import functional as F
from experiments.pamf_dura_efficiency.linear2x import linear2x


@pytest.mark.parametrize('length',[1,3,32,64,128])
def test_half_pixel_interpolation_preserves_edges_values_and_adjoint(length):
    generator=torch.Generator().manual_seed(123+length)
    value=torch.randn(2,3,length,generator=generator,dtype=torch.float64,requires_grad=True)
    probe=torch.randn(2,3,2*length,generator=generator,dtype=torch.float64)
    expected=F.interpolate(value,size=2*length,mode='linear',align_corners=False)
    actual=linear2x(value)
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    assert torch.equal(actual[...,0],value[...,0]) and torch.equal(actual[...,-1],value[...,-1])
    left=torch.autograd.grad((actual*probe).sum(),value)[0]
    right=torch.autograd.grad((expected*probe).sum(),value)[0]
    torch.testing.assert_close(left,right,rtol=1e-14,atol=1e-14)
