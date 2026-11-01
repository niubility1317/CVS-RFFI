import torch

from experiments.pamf_phase1_flow.multi_utility_experiment import route


class Teacher:
    def features(self, x):
        return x.mean(dim=(1,2))[:,None].expand(-1,160)

    def classify_features(self, x):
        value=x[:,0]
        return torch.stack((value,torch.full_like(value,0.75),
                            *[torch.full_like(value,-10.0) for _ in range(4)]),dim=1)


class Utility(torch.nn.Module):
    def forward(self, x):
        assert x.shape[1]==698
        return torch.tensor([[3.0,-3.0,-3.0,3.0],[-3.0,3.0,3.0,-3.0],[0.0,0.0,0.0,0.0]],device=x.device)


def test_net_utility_selects_raw_feature_or_falls_back():
    teacher=Teacher();dura=lambda x:x*0.5
    models=(teacher,None,dura,lambda x,anchor:anchor,lambda x,anchor,feat,baseline:anchor*3,
            Utility(),torch.zeros(698),torch.ones(698))
    x=torch.ones((3,2,256));raw,selected,choice=route(models,x)
    assert raw.tolist()==[0,0,0]
    assert choice.tolist()==[0,3,1]
    assert selected.tolist()==[0,0,1]
