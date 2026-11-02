import torch
from mixture_controller import MixtureController

def test_zero_controller_reproduces_posterior():
    torch.manual_seed(7)
    f=torch.randn(3,6,347); lw=torch.log_softmax(torch.randn(3,6),1); states=torch.randn(3,6,512)
    net=MixtureController(); out,w,c=net(f,lw,states)
    assert torch.allclose(out.reshape(3,512),(states*lw.exp().unsqueeze(-1)).sum(1),atol=1e-6)
    assert torch.allclose(w.sum(1),torch.ones(3)) and torch.count_nonzero(c)==0
    (out.square().mean()).backward()
    assert net.network[-1].weight.grad.abs().sum()>0

def test_invalid_shapes_rejected():
    net=MixtureController()
    try: net(torch.zeros(2,5,347),torch.zeros(2,6),torch.zeros(2,6,512))
    except ValueError: return
    raise AssertionError('wrong shape was accepted')
