import torch

from experiments.pamf_phase1_flow.filter_residual_experiment import make_source_target, step
from experiments.pamf_phase1_flow.filter_student import FilterStudent


def test_residual_oracle_preserves_exact_identity_and_radius():
    torch.manual_seed(2)
    anchor = torch.randn(3, 2, 256)
    assert torch.equal(make_source_target(anchor, anchor), anchor)
    clean = torch.randn_like(anchor)
    target = make_source_target(anchor, clean)
    assert torch.all((target-anchor).flatten(1).norm(dim=1) <= .200001*anchor.flatten(1).norm(dim=1))


def test_residual_source_step_updates_predicted_filter():
    class Teacher(torch.nn.Module):
        def features(self, x):
            return x.mean((1,2))[:, None]

        def classify_features(self, x):
            return x.expand(-1,6)

    class Dura(torch.nn.Module):
        def forward(self, x):
            return x

    torch.manual_seed(3)
    x=torch.randn(4,2,256)
    model=FilterStudent(feature_dim=1,width=16,layers=1,heads=4)
    teacher=Teacher().eval().requires_grad_(False);dura=Dura().eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    stats=step(model,teacher,dura,x,.95*x,optimizer)
    assert stats['head_grad'] > 0 and stats['grad_norm'] > 0
    assert model.output.weight.detach().abs().sum() > 0
