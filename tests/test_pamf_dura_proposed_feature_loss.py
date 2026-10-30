import pytest
import torch
from torch import nn
from experiments.pamf_dura_efficiency.proposed_feature_loss import frozen_feature_loss


class ToyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(512,160,bias=False)

    def features(self,value):return self.project(value.flatten(1))


def test_prototype_propagates_to_IQ_without_updating_F_or_reference():
    reader=ToyReader().eval().requires_grad_(False)
    value=torch.randn(3,2,256,requires_grad=True)
    target=torch.randn(3,160,requires_grad=True)
    before={key:v.clone() for key,v in reader.state_dict().items()}
    loss=frozen_feature_loss(reader,value,target).mean()
    loss.backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert target.grad is None and all(v.grad is None for v in reader.parameters())
    assert all(torch.equal(v,before[key]) for key,v in reader.state_dict().items())


def test_prototype_zero_for_identical_features():
    reader=ToyReader().eval().requires_grad_(False)
    value=torch.randn(2,2,256)
    with torch.no_grad():target=reader.features(value)
    torch.testing.assert_close(frozen_feature_loss(reader,value,target),torch.zeros(2),atol=2e-7,rtol=0)


@pytest.mark.parametrize('state',['train','unfrozen'])
def test_prototype_rejects_F_that_can_change(state):
    reader=ToyReader().eval().requires_grad_(False)
    if state=='train':reader.train()
    else:reader.requires_grad_(True)
    with pytest.raises(ValueError,match='parameters frozen'):
        frozen_feature_loss(reader,torch.randn(2,2,256),torch.randn(2,160))
