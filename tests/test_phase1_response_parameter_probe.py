import torch
from experiments.phase1_dura_orthogonal.source_response_parameter_probe import response_loss, gradient_vector


def test_response_loss_and_parameter_gradient():
    torch.manual_seed(83)
    original=torch.randn(4,2,256)
    delta=.01*torch.randn_like(original)
    assert response_loss(original,original+delta,delta)<1e-9
    torch.testing.assert_close(response_loss(original,original,delta),torch.tensor(1.))
    weight=torch.nn.Parameter(torch.tensor(.3))
    unused=torch.nn.Parameter(torch.tensor(.2))
    loss=response_loss(original,original+weight*delta,delta)
    grad=gradient_vector(loss,(weight,unused),False)
    torch.testing.assert_close(grad,torch.tensor([-1.4,0.]),rtol=1e-4,atol=1e-4)
