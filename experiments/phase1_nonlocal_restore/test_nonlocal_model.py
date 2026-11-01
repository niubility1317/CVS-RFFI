import torch
from nonlocal_model import NoiseNonlocalRestorer

def test_identity_initialization_and_bounded_correction():
    torch.manual_seed(5);net=NoiseNonlocalRestorer();x=torch.randn(2,2,256);z=torch.randn_like(x);b=torch.randn_like(x);noise=torch.tensor([-5.,-2.]);torch.testing.assert_close(net(x,z,b,noise),z)
    with torch.no_grad():net.output.weight.normal_(0,2);net.output.bias.normal_(0,1)
    y=net(x,z,b,noise);assert torch.all((y-z).flatten(1).norm(dim=1)<=.200001*z.flatten(1).norm(dim=1))

def test_source_gradient_and_nonlocal_dependency():
    torch.manual_seed(6);net=NoiseNonlocalRestorer();x=torch.randn(2,2,256);z=torch.randn_like(x);b=torch.randn_like(x);noise=torch.tensor([-5.,-2.]);(net(x,z,b,noise)-z-.01).square().mean().backward();assert torch.isfinite(net.output.weight.grad).all() and net.output.weight.grad.abs().sum()>0
    with torch.no_grad():
        net.output.weight.normal_(0,.01);a=net(x,z,b,noise);x2=x.clone();x2[:,:,200]+=.1;c=net(x2,z,b,noise)
    assert (a[:,:,20]-c[:,:,20]).abs().max()>1e-8
