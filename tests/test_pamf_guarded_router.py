import torch

from experiments.pamf_phase1_flow.guarded_router_experiment import route


class Teacher:
    def features(self, x):
        return x.mean(dim=(1, 2))

    def classify_features(self, features):
        return torch.stack((features, torch.full_like(features, 0.75),
                            *[torch.full_like(features, -10.0) for _ in range(4)]), dim=1)


class RawGate(torch.nn.Module):
    def forward(self, features):
        return torch.tensor([[4.0, 0.0, 0.0, 0.0]], device=features.device).expand(len(features), -1)


def test_guard_selects_raw_only_when_anchor_energy_is_low():
    teacher=Teacher()
    dura=lambda x: x*0.5
    boundary=lambda x, anchor: anchor
    feature=lambda x, anchor, feat, baseline: anchor
    models=(teacher, None, dura, boundary, feature, RawGate(), torch.zeros(26), torch.ones(26))
    x=torch.stack((torch.ones((2,256)),torch.full((2,256),1.3)))
    raw,selected,choice=route(models,x)
    assert raw.tolist()==[0,0]
    assert choice.tolist()==[0,1]
    assert selected.tolist()==[0,1]
