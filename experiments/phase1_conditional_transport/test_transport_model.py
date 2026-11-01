import torch
from transport_model import ConditionalTransport,backproject,integrate

def test_transport_starts_at_identity_and_has_finite_training_gradient():
 torch.manual_seed(17);net=ConditionalTransport(width=16);z=torch.randn(2,2,256);x=torch.randn_like(z);a=torch.eye(512).repeat(2,1,1);lv=torch.zeros(2);t=torch.tensor([.2,.8]);b=backproject(x,z,a);assert torch.allclose(b,x-z,atol=1e-6);v=net(z,z,b,t,lv);assert torch.count_nonzero(v)==0;loss=(v-(x-z)).square().mean();loss.backward();assert torch.isfinite(net.output.weight.grad).all() and net.output.weight.grad.abs().sum()>0
 for steps in (1,4,8):assert torch.equal(integrate(net,x,z,a,lv,steps),z)

def test_integrator_follows_known_constant_velocity():
 class Constant(torch.nn.Module):
  def forward(self,state,parent,residual,time,log_variance):return torch.ones_like(state)*.25
 z=torch.randn(2,2,256);a=torch.eye(512).repeat(2,1,1)
 for steps in (1,4,8):assert torch.allclose(integrate(Constant(),z,z,a,torch.zeros(2),steps),z+.25,atol=1e-6)


def test_four_step_rollout_keeps_gradient_to_velocity_network():
 torch.manual_seed(3);net=ConditionalTransport(width=16);z=torch.randn(2,2,256);x=torch.randn_like(z);a=torch.eye(512).repeat(2,1,1);out=integrate(net,x,z,a,torch.zeros(2),4);loss=(out-x).square().mean();loss.backward();assert torch.isfinite(net.output.weight.grad).all() and net.output.weight.grad.abs().sum()>0
