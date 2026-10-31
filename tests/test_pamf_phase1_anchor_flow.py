import torch
from torch import nn
from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_phase1_flow.anchor_flow import FixedAnchoredFlow,source_objective


class Reader(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(6,6,bias=False)
    def features(self,x):return torch.cat((x.mean(2),x.square().mean(2),x[:,:,:64].mean(2)),1)
    def classify_features(self,x):return self.head(x)


def test_zero_field_preserves_anchor_without_clean_or_target_metadata():
    torch.manual_seed(9);field=IdentityConditionalFlow(width=16,layers=1,heads=4)
    received=torch.randn(2,2,256);anchor=torch.randn_like(received)
    with torch.no_grad():
        result=FixedAnchoredFlow(field)(received,anchor)
    torch.testing.assert_close(result,anchor,atol=3e-7,rtol=1e-6)


def test_anchored_semantic_gradient_reaches_generator_not_frozen_inputs():
    torch.manual_seed(12);field=IdentityConditionalFlow(width=16,layers=1,heads=4)
    teacher=Reader().eval().requires_grad_(False)
    clean=torch.randn(3,2,256);received=clean+.5*torch.randn_like(clean)
    anchor=(clean+.2*torch.randn_like(clean)).requires_grad_();noise=torch.randn_like(clean)
    loss,terms=source_objective(field,teacher,received,anchor,clean,noise)
    grad=torch.autograd.grad(terms['semantic'].sum(),field.output.weight,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss.mean().backward()
    assert anchor.grad is None and all(p.grad is None for p in teacher.parameters())
