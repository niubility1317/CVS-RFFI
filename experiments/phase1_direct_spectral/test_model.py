import torch
from model import DirectSpectralRestorer

def test_identity_initialization_and_batch_independence():
    torch.manual_seed(1);model=DirectSpectralRestorer().eval();x=torch.randn(4,2,256)
    with torch.no_grad():
        actual=model(x);single=model(x[:1])
    torch.testing.assert_close(actual,x,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(actual[:1],single,rtol=1e-5,atol=1e-5)

def test_output_and_gradients_finite_on_zero_and_nonzero():
    torch.manual_seed(2);model=DirectSpectralRestorer();x=torch.randn(3,2,256);x[0]=0
    y=model(x);assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.spectrum.weight.grad is not None and torch.isfinite(model.spectrum.weight.grad).all()
    assert model.output.weight.grad is not None and torch.isfinite(model.output.weight.grad).all()
    assert model.spectrum.weight.grad.abs().sum()>0 and model.output.weight.grad.abs().sum()>0

def test_wrong_shape_rejected():
    import pytest
    with pytest.raises(ValueError):DirectSpectralRestorer()(torch.zeros(2,2,128))
