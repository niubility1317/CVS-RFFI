import torch
from torch import nn
from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_phase1_flow.semantic import semantic_objective


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(6,6,bias=False)
    def features(self,x):return torch.cat((x.mean(2),x.square().mean(2),x[:,:,:64].mean(2)),1)
    def classify_features(self,x):return self.head(x)


def test_semantic_loss_drives_flow_through_frozen_teacher():
    torch.manual_seed(41);model=IdentityConditionalFlow(width=16,layers=1,heads=4)
    teacher=Reader().eval().requires_grad_(False)
    clean=torch.randn(3,2,256);received=clean+.4*torch.randn_like(clean);noise=torch.randn_like(clean)
    loss,parts=semantic_objective(model,teacher,clean,received,noise)
    grad=torch.autograd.grad(parts['semantic'].sum(),model.output.weight,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss.mean().backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert torch.isfinite(loss).all() and loss.shape==(3,)
