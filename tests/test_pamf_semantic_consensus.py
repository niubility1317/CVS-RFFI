import torch

from experiments.pamf_phase1_flow.semantic_consensus_experiment import route


class Teacher:
    def features(self, x):
        return x.mean(dim=(1,2))[:,None].expand(-1,160)

    def classify_features(self, x):
        value=x[:,0]
        return torch.stack((value,torch.full_like(value,0.75),
                            *[torch.full_like(value,-10.0) for _ in range(4)]),dim=1)


class Consensus(torch.nn.Module):
    def forward(self, x):
        assert x.shape[1]==698
        return torch.tensor([[5.,0.,0.,0.,0.,0.],[0.,0.,5.,0.,0.,0.],
                             [0.,5.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.]],device=x.device)


def test_consensus_requires_confidence_and_matching_frozen_expert():
    teacher=Teacher();dura=lambda x:x*0.5
    models=(teacher,None,dura,lambda x,anchor:anchor,lambda x,anchor,feat,baseline:anchor*3,
            Consensus(),torch.zeros(698),torch.ones(698))
    raw,selected,choice=route(models,torch.ones((4,2,256)))
    assert raw.tolist()==[0,0,0,0]
    assert choice.tolist()==[3,1,1,1]
    assert selected.tolist()==[0,1,1,1]
