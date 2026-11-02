import torch
from free_mixture import FreeMixtureController

def test_zero_blend_is_physical_parent_and_gradient_flows():
 torch.manual_seed(9);f=torch.randn(3,6,347);lp=torch.log_softmax(torch.randn(3,6),1);states=torch.randn(3,6,512);net=FreeMixtureController();out,w,z=net(f,lp,states,scale=0.)
 assert torch.allclose(out.reshape(3,512),(states*lp.exp().unsqueeze(-1)).sum(1),atol=1e-6)
 assert torch.allclose(w,torch.full_like(w,1/6))
 out,w,z=net(f,lp,states,scale=1.);(out.square().mean()+z.square().mean()).backward();assert net.network[-1].weight.grad.abs().sum()>0

def test_invalid_blend_rejected():
 net=FreeMixtureController()
 try:net(torch.zeros(2,6,347),torch.zeros(2,6),torch.zeros(2,6,512),scale=1.1)
 except ValueError:return
 raise AssertionError('invalid blend accepted')
