import torch

from experiments.pamf_phase1_flow.physical_rich_router_experiment import route


class Teacher:
    def features(self, x):
        return x.mean(dim=(1,2))[:,None].expand(-1,160)

    def classify_features(self, x):
        value=x[:,0]
        return torch.stack((value,torch.full_like(value,0.75),
                            *[torch.full_like(value,-10.0) for _ in range(4)]),dim=1)


class Gate(torch.nn.Module):
    def forward(self, x):
        assert x.shape[1]==378
        return torch.tensor([[5.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]],device=x.device)


def test_route_uses_probability_margin_and_accepts_high_energy_inputs():
    teacher=Teacher();dura=lambda x:x*0.5
    models=(teacher,None,dura,lambda x,anchor:anchor,lambda x,anchor,feat,baseline:anchor,
            Gate(),torch.zeros(378),torch.ones(378))
    x=torch.stack((torch.full((2,256),1.3),torch.ones((2,256))))
    raw,selected,choice=route(models,x)
    assert raw.tolist()==[0,0]
    assert choice.tolist()==[0,1]
    assert selected.tolist()==[0,1]
