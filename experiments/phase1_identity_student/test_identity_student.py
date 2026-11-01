import torch
from identity_student_model import IdentityStudent

def test_student_identity_initialization_and_bound():
    torch.manual_seed(3);net=IdentityStudent(.02);x=torch.randn(5,2,256);z=torch.randn_like(x);b=torch.randn_like(x);torch.testing.assert_close(net(x,z,b),z)
    with torch.no_grad():net.output.weight.normal_(0,2);net.output.bias.normal_(0,1)
    y=net(x,z,b);delta=(y-z).flatten(1).norm(dim=1);assert torch.all(delta<=.020001*z.flatten(1).norm(dim=1))

def test_identity_student_has_finite_direction_gradient():
    torch.manual_seed(4);net=IdentityStudent();x=torch.randn(3,2,256);z=torch.randn_like(x);b=torch.randn_like(x);target=z+.01*torch.randn_like(z);((net(x,z,b)-target)**2).mean().backward();g=net.output.weight.grad;assert torch.isfinite(g).all() and g.abs().sum()>0
