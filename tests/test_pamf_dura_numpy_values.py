import numpy as np
import pytest
import torch
from experiments.pamf_dura_diagnostics.runner import numpy_values


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_host_result_is_exact_owned_writable_FP32_for_strided_grad_tensor(dtype):
    value=torch.arange(3*2*256,dtype=dtype).reshape(3,2,256)[...,::2].requires_grad_(True)
    expected=np.asarray(value.detach().cpu().tolist(),dtype=np.float32)
    actual=numpy_values(value)
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==np.float32 and actual.flags.owndata and actual.flags.writeable
    with torch.no_grad():value.fill_(0)
    np.testing.assert_array_equal(actual,expected)
    actual.fill(3)
    assert torch.count_nonzero(value)==0 and value.grad is None


def test_inference_tensor_needs_no_numpy_C_bridge(monkeypatch):
    def unavailable(*args,**kwargs):raise RuntimeError('Torch/NumPy C bridge unavailable')
    monkeypatch.setattr(torch.Tensor,'numpy',unavailable)
    with torch.inference_mode():value=torch.tensor([0.,-0.,1.,-1.])
    actual=numpy_values(value)
    np.testing.assert_array_equal(actual,np.array([0.,-0.,1.,-1.],dtype=np.float32))
    assert np.signbit(actual).tolist()==[False,True,False,True]


def test_finite_FP32_boundaries_keep_their_bits():
    bits=np.array([0,0x80000000,1,0x80000001,0x00800000,0x7f7fffff,0xff7fffff],dtype=np.uint32)
    value=torch.frombuffer(memoryview(bits),dtype=torch.float32)
    np.testing.assert_array_equal(numpy_values(value).view(np.uint32),bits)
